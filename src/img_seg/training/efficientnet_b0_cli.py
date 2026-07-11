"""Train and evaluate the EfficientNet-B0 2D segmentation baseline."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from img_seg.config import deep_get, load_yaml, project_path_from_config, resolve_project_path
from img_seg.data.cases import SegmentationCase, load_cases_from_manifest
from img_seg.data.slices import NiftiSliceDataset, build_slice_samples
from img_seg.data.splits import CaseSplit, make_case_split
from img_seg.evaluation.metrics import binary_confusion, binary_metrics, safe_divide
from img_seg.io.nifti import binarize_mask, load_nifti_array
from img_seg.models.efficientnet_b0 import (
    EfficientNetB0Segmenter,
    load_efficientnet_config,
)

DEFAULT_CHECKPOINT_NAME = "best.pth"
DEFAULT_TRAIN_LOG_NAME = "train.log"
RUN_NAME_FORMAT = "%Y%m%dT%H%M%S%fZ"


@dataclass(frozen=True)
class TrainingArtifacts:
    run_name: str
    checkpoint: Path
    log_file: Path


@dataclass(frozen=True)
class TrainingLog:
    path: Path
    run_name: str

    def write(self, event: str, **values: object) -> None:
        """Append one self-contained JSON record and flush it to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_name": self.run_name,
            "event": event,
            **values,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_cases_from_config(config: dict[str, Any]) -> tuple[list[SegmentationCase], CaseSplit]:
    cases = load_cases_from_manifest(project_path_from_config(config, "data.dataset_config"))
    if any(case.mask_path is None for case in cases):
        raise ValueError("Every EfficientNet-B0 training case must have a mask")
    split_config = load_yaml(project_path_from_config(config, "data.split_file"))
    strategy = str(split_config.get("strategy", "case_level"))
    if strategy != "case_level":
        raise ValueError(f"Unsupported split strategy: {strategy}")
    split_keys = ("train", "val", "test")
    if all(isinstance(split_config.get(key), list) for key in split_keys):
        split = CaseSplit(**{key: list(split_config[key]) for key in split_keys})
    else:
        ratios = split_config.get("ratios", {})
        split = make_case_split(
            [case.case_id for case in cases],
            seed=int(split_config.get("seed", 42)),
            train_ratio=float(ratios.get("train", 0.7)),
            val_ratio=float(ratios.get("val", 0.15)),
            test_ratio=float(ratios.get("test", 0.15)),
        )

    available_ids = {case.case_id for case in cases}
    assigned_ids = [*split.train, *split.val, *split.test]
    unknown_ids = sorted(set(assigned_ids) - available_ids)
    if unknown_ids:
        raise ValueError(f"Split contains unknown case ids: {unknown_ids}")
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("A case id appears in more than one split")
    missing_ids = sorted(available_ids - set(assigned_ids))
    if missing_ids:
        raise ValueError(f"Split does not assign every dataset case: {missing_ids}")
    return cases, split


def dataset_for_split(
    config: dict[str, Any],
    cases: list[SegmentationCase],
    case_ids: list[str],
    *,
    training: bool,
) -> NiftiSliceDataset:
    foreground_keep_ratio = (
        float(deep_get(config, "data.foreground_slice_keep_ratio", 1.0)) if training else 1.0
    )
    empty_keep_ratio = (
        float(deep_get(config, "data.empty_slice_keep_ratio", 0.25)) if training else 1.0
    )
    samples = build_slice_samples(
        cases,
        case_ids,
        slice_axis=deep_get(config, "data.slice_axis", "z"),
        foreground_slice_keep_ratio=foreground_keep_ratio,
        empty_slice_keep_ratio=empty_keep_ratio,
        seed=int(deep_get(config, "training.seed", 42)),
    )
    return NiftiSliceDataset(
        samples,
        image_size=deep_get(config, "data.image_size", [512, 512]),
        nifti_cache_size=int(deep_get(config, "runtime.nifti_cache_size", 8)),
    )


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
    use_amp: bool = False,
) -> dict[str, float]:
    counts_by_case: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            non_blocking = device.type == "cuda"
            images = batch["image"].to(device, non_blocking=non_blocking)
            masks = batch["mask"].to(device, non_blocking=non_blocking)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            predictions = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy()
            targets = (masks >= 0.5).detach().cpu().numpy()
            case_ids = batch["case_id"]
            for prediction, target, case_id in zip(
                predictions,
                targets,
                case_ids,
                strict=True,
            ):
                counts = binary_confusion(prediction[0], target[0])
                totals = counts_by_case[str(case_id)]
                for index, value in enumerate(counts):
                    totals[index] += value
    if not counts_by_case:
        raise ValueError("No validation cases were produced")

    rows = []
    for tp, fp, fn, _tn in counts_by_case.values():
        rows.append(
            {
                "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
                "iou": safe_divide(tp, tp + fp + fn),
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
            }
        )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("dice", "iou", "precision", "recall")
    }


def timestamped_run_name(timestamp: datetime | None = None) -> str:
    """Return a sortable, filesystem-safe UTC name for one training run."""

    value = timestamp or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime(RUN_NAME_FORMAT)


def resolve_training_artifacts(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    run_name: str | None = None,
) -> TrainingArtifacts:
    if output_dir is not None:
        artifact_dir = resolve_project_path(output_dir)
        checkpoint_template = artifact_dir / DEFAULT_CHECKPOINT_NAME
        log_template = artifact_dir / DEFAULT_TRAIN_LOG_NAME
    else:
        checkpoint_template = project_path_from_config(config, "checkpoints.best")
        log_template = resolve_project_path(
            deep_get(config, "training.log_file", "outputs/efficientnet_b0/train.log")
        )
    if log_file is not None:
        log_template = resolve_project_path(log_file)

    if checkpoint_template.suffix.lower() != ".pth":
        raise ValueError(
            f"EfficientNet-B0 checkpoint must use the .pth suffix: {checkpoint_template}"
        )

    name = run_name or timestamped_run_name()
    return TrainingArtifacts(
        run_name=name,
        checkpoint=checkpoint_template.parent / name / checkpoint_template.name,
        log_file=log_template.parent / name / log_template.name,
    )


def _main_hyperparameters(
    config: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    optimizer: str,
    loss: str,
    device: torch.device,
    num_workers: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    return {
        "model": {
            "architecture": deep_get(config, "model.architecture", "Unet"),
            "encoder_name": deep_get(config, "model.encoder_name", "efficientnet-b0"),
            "encoder_weights": deep_get(config, "model.encoder_weights"),
            "in_channels": int(deep_get(config, "model.in_channels", 1)),
            "classes": int(deep_get(config, "model.classes", 1)),
        },
        "data": {
            "image_size": deep_get(config, "data.image_size", [512, 512]),
            "slice_axis": deep_get(config, "data.slice_axis", "z"),
            "foreground_slice_keep_ratio": float(
                deep_get(config, "data.foreground_slice_keep_ratio", 1.0)
            ),
            "empty_slice_keep_ratio": float(
                deep_get(config, "data.empty_slice_keep_ratio", 0.25)
            ),
        },
        "optimization": {
            "seed": int(deep_get(config, "training.seed", 42)),
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "learning_rate": float(deep_get(config, "training.learning_rate", 0.0003)),
            "weight_decay": float(deep_get(config, "training.weight_decay", 0.01)),
            "loss": loss,
            "amp_requested": bool(deep_get(config, "training.amp", True)),
            "amp_enabled": amp_enabled,
        },
        "runtime": {
            "device": str(device),
            "num_workers": num_workers,
        },
        "validation": {
            "threshold": float(deep_get(config, "inference.threshold", 0.5)),
            "aggregation": "case_macro_all_slices",
        },
    }


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    losses = []
    non_blocking = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=non_blocking)
        masks = batch["mask"].to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = dice_bce_loss(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    if not losses:
        raise ValueError("No training batches were produced")
    return float(np.mean(losses))


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace the best checkpoint to avoid leaving a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_model(
    config_path: str | Path,
    *,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
) -> dict[str, Any]:
    config = load_efficientnet_config(config_path)
    seed = int(deep_get(config, "training.seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    optimizer_name = str(deep_get(config, "training.optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported training.optimizer: {optimizer_name}")
    loss_name = str(deep_get(config, "training.loss", "dice_bce")).lower()
    if loss_name != "dice_bce":
        raise ValueError(f"Unsupported training.loss: {loss_name}")

    cases, split = split_cases_from_config(config)
    train_dataset = dataset_for_split(config, cases, split.train, training=True)
    val_dataset = dataset_for_split(config, cases, split.val, training=False)

    batch_size = int(deep_get(config, "training.batch_size", 8))
    if batch_size <= 0:
        raise ValueError(f"training.batch_size must be positive: {batch_size}")
    num_workers = int(deep_get(config, "runtime.num_workers", 0))
    if num_workers < 0:
        raise ValueError(f"runtime.num_workers cannot be negative: {num_workers}")
    segmenter = EfficientNetB0Segmenter(config, use_pretrained_encoder=True)
    device = segmenter.device
    model = segmenter.model
    pin_memory = device.type == "cuda"
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(config, "training.learning_rate", 0.0003)),
        weight_decay=float(deep_get(config, "training.weight_decay", 0.01)),
    )
    epoch_count = int(deep_get(config, "training.epochs", 100) if epochs is None else epochs)
    if epoch_count <= 0:
        raise ValueError(f"training.epochs must be positive: {epoch_count}")
    use_amp = bool(deep_get(config, "training.amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_dice = -1.0
    artifacts = resolve_training_artifacts(config, output_dir=output_dir, log_file=log_file)
    training_log = TrainingLog(artifacts.log_file, artifacts.run_name)
    training_log.write(
        "run_started",
        config=str(resolve_project_path(config_path)),
        train_cases=len(split.train),
        val_cases=len(split.val),
        train_slices=len(train_dataset),
        val_slices=len(val_dataset),
        checkpoint=str(artifacts.checkpoint),
        encoder_initialization=segmenter.encoder_initialization,
        hyperparameters=_main_hyperparameters(
            config,
            epochs=epoch_count,
            batch_size=batch_size,
            optimizer=optimizer_name,
            loss=loss_name,
            device=device,
            num_workers=num_workers,
            amp_enabled=use_amp,
        ),
    )

    history = []
    for epoch in range(1, epoch_count + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics = evaluate_slice_loader(
            model,
            val_loader,
            device=device,
            threshold=float(deep_get(config, "inference.threshold", 0.5)),
            use_amp=use_amp,
        )
        is_best = val_metrics["dice"] > best_dice
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics, "is_best": is_best}
        history.append(row)
        training_log.write("epoch_completed", **row)
        print(json.dumps(row, ensure_ascii=False))
        if is_best:
            best_dice = val_metrics["dice"]
            _save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_dice": best_dice,
                    "validation": "case_macro_all_slices",
                    "encoder_initialization": segmenter.encoder_initialization,
                    "run_name": artifacts.run_name,
                },
                artifacts.checkpoint,
            )
            training_log.write(
                "checkpoint_saved",
                epoch=epoch,
                best_val_dice=best_dice,
                checkpoint=str(artifacts.checkpoint),
            )

    result = {
        "run_name": artifacts.run_name,
        "checkpoint": str(artifacts.checkpoint),
        "log_file": str(artifacts.log_file),
        "best_val_dice": best_dice,
        "epochs": epoch_count,
        "encoder_initialization": segmenter.encoder_initialization,
        "history": history,
    }
    training_log.write(
        "run_completed",
        best_val_dice=best_dice,
        checkpoint=str(artifacts.checkpoint),
    )
    return result


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
    segmenter = EfficientNetB0Segmenter(config, use_pretrained_encoder=False)
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
    train_parser.add_argument("--config", dest="command_config")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--log-file")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate checkpoint on a case split.")
    eval_parser.add_argument("--config", dest="command_config")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--output-dir")
    eval_parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.command_config or args.config
    if args.command == "train":
        result = train_model(
            config_path,
            epochs=args.epochs,
            output_dir=args.output_dir,
            log_file=args.log_file,
        )
    elif args.command == "evaluate":
        result = evaluate_model(
            config_path,
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
