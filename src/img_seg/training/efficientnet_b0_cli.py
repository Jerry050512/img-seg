"""Train and evaluate the EfficientNet-B0 2D segmentation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from img_seg.config import deep_get, load_yaml, project_path_from_config, resolve_project_path
from img_seg.data.cases import SegmentationCase, discover_cases
from img_seg.data.slices import NiftiSliceDataset, build_slice_samples
from img_seg.data.splits import CaseSplit, make_case_split
from img_seg.evaluation.metrics import binary_metrics
from img_seg.io.nifti import binarize_mask, load_nifti_array
from img_seg.models.efficientnet_b0 import (
    EfficientNetB0Segmenter,
    load_efficientnet_config,
)


def split_cases_from_config(config: dict[str, Any]) -> tuple[list[SegmentationCase], CaseSplit]:
    dataset_dir = project_path_from_config(config, "data.raw_dataset_dir")
    cases = discover_cases(
        dataset_dir,
        mask_overrides=deep_get(config, "data.mask_overrides", {}),
        require_masks=True,
    )
    split_config = load_yaml(project_path_from_config(config, "data.split_file"))
    ratios = split_config.get("ratios", {})
    split = make_case_split(
        [case.case_id for case in cases],
        seed=int(split_config.get("seed", 42)),
        train_ratio=float(ratios.get("train", 0.7)),
        val_ratio=float(ratios.get("val", 0.15)),
        test_ratio=float(ratios.get("test", 0.15)),
    )
    return cases, split


def dataset_for_split(
    config: dict[str, Any],
    cases: list[SegmentationCase],
    case_ids: list[str],
) -> NiftiSliceDataset:
    samples = build_slice_samples(
        cases,
        case_ids,
        slice_axis=deep_get(config, "data.slice_axis", "z"),
        foreground_slice_keep_ratio=float(
            deep_get(config, "data.foreground_slice_keep_ratio", 1.0)
        ),
        empty_slice_keep_ratio=float(deep_get(config, "data.empty_slice_keep_ratio", 0.25)),
        seed=int(deep_get(config, "training.seed", 42)),
    )
    return NiftiSliceDataset(samples, image_size=deep_get(config, "data.image_size", [512, 512]))


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return bce + (1.0 - dice.mean())


def evaluate_slice_loader(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, object]],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            predictions = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy()
            targets = (masks >= 0.5).detach().cpu().numpy()
            for prediction, target in zip(predictions, targets, strict=True):
                rows.append(binary_metrics(prediction[0], target[0]).as_dict())
    if not rows:
        raise ValueError("No validation rows were produced")
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("dice", "iou", "precision", "recall")
    }


def checkpoint_output_path(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return resolve_project_path(output_dir) / "best.pt"
    return project_path_from_config(config, "checkpoints.best")


def train_model(
    config_path: str | Path,
    *,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_efficientnet_config(config_path)
    torch.manual_seed(int(deep_get(config, "training.seed", 42)))
    cases, split = split_cases_from_config(config)
    train_dataset = dataset_for_split(config, cases, split.train)
    val_dataset = dataset_for_split(config, cases, split.val)

    batch_size = int(deep_get(config, "training.batch_size", 8))
    num_workers = int(deep_get(config, "runtime.num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    segmenter = EfficientNetB0Segmenter(config)
    device = segmenter.device
    model = segmenter.model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(config, "training.learning_rate", 0.0003)),
    )
    epoch_count = int(epochs or deep_get(config, "training.epochs", 100))
    use_amp = bool(deep_get(config, "training.amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_dice = -1.0
    best_path = checkpoint_output_path(config, output_dir)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, epoch_count + 1):
        model.train()
        losses = []
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(autocast_device, enabled=use_amp):
                logits = model(images)
                loss = dice_bce_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_slice_loader(
            model,
            val_loader,
            device=device,
            threshold=float(deep_get(config, "inference.threshold", 0.5)),
        )
        train_loss = float(np.mean(losses)) if losses else 0.0
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_dice": best_dice,
                },
                best_path,
            )

    return {
        "checkpoint": str(best_path),
        "best_val_dice": best_dice,
        "epochs": epoch_count,
        "history": history,
    }


def _case_ids_for_split(split: CaseSplit, split_name: str) -> list[str]:
    if split_name == "train":
        return split.train
    if split_name == "val":
        return split.val
    if split_name == "test":
        return split.test
    raise ValueError(f"Unknown split: {split_name}")


def evaluate_model(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path,
    split_name: str = "test",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_efficientnet_config(config_path)
    cases, split = split_cases_from_config(config)
    case_ids = _case_ids_for_split(split, split_name)
    cases_by_id = {case.case_id: case for case in cases}
    output_root = resolve_project_path(output_dir or deep_get(config, "inference.output_dir"))
    segmenter = EfficientNetB0Segmenter(config)
    segmenter.load(checkpoint_path)

    per_case: dict[str, dict[str, Any]] = {}
    rows = []
    for case_id in case_ids:
        case = cases_by_id[case_id]
        prediction_path = output_root / f"{case_id}_Seg.nii.gz"
        prediction = segmenter.predict_volume(case.image_path, prediction_path)
        if case.mask_path is None:
            raise ValueError(f"Case {case_id} has no reference mask")
        prediction_array, _ = load_nifti_array(prediction_path)
        target_array, _ = load_nifti_array(case.mask_path)
        metrics = binary_metrics(binarize_mask(prediction_array), binarize_mask(target_array))
        row = {**metrics.as_dict(), **prediction}
        per_case[case_id] = row
        rows.append(row)

    if not rows:
        raise ValueError(f"No cases found for split: {split_name}")
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("dice", "iou", "precision", "recall", "elapsed_sec")
    }
    return {"summary": summary, "cases": per_case}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0/base.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train EfficientNet-B0 baseline.")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--output-dir")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate checkpoint on a case split.")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--output-dir")
    eval_parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        result = train_model(args.config, epochs=args.epochs, output_dir=args.output_dir)
    elif args.command == "evaluate":
        result = evaluate_model(
            args.config,
            checkpoint_path=args.checkpoint,
            split_name=args.split,
            output_dir=args.output_dir,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    text = json.dumps(result, indent=2, ensure_ascii=False)
    output_json = getattr(args, "output_json", None)
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
