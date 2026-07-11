"""Training, validation, prediction, and evaluation for MONAI SegResNet."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from img_seg.config import deep_get, load_yaml, resolve_project_path
from img_seg.data.cases import SegmentationCase, load_cases_from_manifest
from img_seg.data.splits import CaseSplit, make_case_split
from img_seg.evaluation.metrics import binary_metrics
from img_seg.io.nifti import binarize_mask, load_nifti_array, require_nibabel
from img_seg.models.monai_segresnet import (
    MonaiSegResNetSegmenter,
    build_model,
    checkpoint_payload,
    load_monai_config,
    resolve_device,
)


@dataclass(frozen=True)
class SegResNetData:
    cases: list[SegmentationCase]
    split: CaseSplit

    def cases_for(self, split_name: str) -> list[SegmentationCase]:
        case_ids = set(getattr(self.split, split_name))
        return [case for case in self.cases if case.case_id in case_ids]


def _require_training_runtime() -> dict[str, Any]:
    try:
        import torch
        from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
        from monai.inferers import sliding_window_inference
        from monai.losses import DiceCELoss, TverskyLoss
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            Lambdad,
            LoadImaged,
            NormalizeIntensityd,
            RandCropByPosNegLabeld,
            RandFlipd,
            RandRotate90d,
            RandScaleIntensityd,
            RandShiftIntensityd,
        )
        from monai.utils import set_determinism
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and MONAI are required for SegResNet training. Run `uv sync`."
        ) from exc
    return {
        "torch": torch,
        "CacheDataset": CacheDataset,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "list_data_collate": list_data_collate,
        "sliding_window_inference": sliding_window_inference,
        "DiceCELoss": DiceCELoss,
        "TverskyLoss": TverskyLoss,
        "Compose": Compose,
        "EnsureChannelFirstd": EnsureChannelFirstd,
        "EnsureTyped": EnsureTyped,
        "Lambdad": Lambdad,
        "LoadImaged": LoadImaged,
        "NormalizeIntensityd": NormalizeIntensityd,
        "RandCropByPosNegLabeld": RandCropByPosNegLabeld,
        "RandFlipd": RandFlipd,
        "RandRotate90d": RandRotate90d,
        "RandScaleIntensityd": RandScaleIntensityd,
        "RandShiftIntensityd": RandShiftIntensityd,
        "set_determinism": set_determinism,
    }


def load_data(config: dict[str, Any]) -> SegResNetData:
    dataset_config = resolve_project_path(deep_get(config, "data.dataset_config"))
    split_path = resolve_project_path(deep_get(config, "data.split_file"))
    cases = load_cases_from_manifest(dataset_config)
    if any(case.mask_path is None for case in cases):
        raise ValueError("Every SegResNet training case must have a mask")
    _validate_case_geometries(cases, deep_get(config, "data.spacing"))

    split_config = load_yaml(split_path)
    explicit_keys = ("train", "val", "test")
    if all(isinstance(split_config.get(key), list) for key in explicit_keys):
        split = CaseSplit(**{key: list(split_config[key]) for key in explicit_keys})
    else:
        ratios = split_config.get("ratios", {})
        split = make_case_split(
            [case.case_id for case in cases],
            seed=int(split_config.get("seed", 42)),
            train_ratio=float(ratios.get("train", 0.7)),
            val_ratio=float(ratios.get("val", 0.15)),
            test_ratio=float(ratios.get("test", 0.15)),
        )
    all_ids = {case.case_id for case in cases}
    assigned = [*split.train, *split.val, *split.test]
    unknown = sorted(set(assigned) - all_ids)
    if unknown:
        raise ValueError(f"Split contains unknown case ids: {unknown}")
    if len(assigned) != len(set(assigned)):
        raise ValueError("A case id appears in more than one split")
    if set(assigned) != all_ids:
        missing = sorted(all_ids - set(assigned))
        raise ValueError(f"Split does not assign every dataset case: {missing}")
    data = SegResNetData(cases=cases, split=split)
    _validate_training_roi(data.cases_for("train"), deep_get(config, "data.roi_size"))
    return data


def _validate_case_geometries(
    cases: list[SegmentationCase], expected_spacing: list[float] | None
) -> None:
    nib = require_nibabel()
    for case in cases:
        image = nib.load(str(case.image_path))
        mask = nib.load(str(case.mask_path))
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {case.case_id}: {image.shape} vs {mask.shape}"
            )
        if not np.allclose(image.affine, mask.affine):
            raise ValueError(f"Image/mask affine mismatch for {case.case_id}")
        if expected_spacing is not None:
            spacing = image.header.get_zooms()[:3]
            if not np.allclose(spacing, expected_spacing, rtol=1e-4, atol=1e-4):
                raise ValueError(
                    f"Unexpected spacing for {case.case_id}: {spacing}; "
                    f"expected {tuple(expected_spacing)}"
                )


def _validate_training_roi(
    cases: list[SegmentationCase], roi_size: list[int]
) -> None:
    nib = require_nibabel()
    roi = tuple(int(value) for value in roi_size)
    for case in cases:
        shape = tuple(int(value) for value in nib.load(str(case.image_path)).shape[:3])
        if any(size < required for size, required in zip(shape, roi, strict=True)):
            raise ValueError(
                f"Training volume {case.case_id} has shape {shape}, smaller than "
                f"data.roi_size={roi}; reduce roi_size or pad the input volume"
            )


def _binarize_label(value: Any) -> Any:
    return (value > 0).to(dtype=value.dtype)


def _records(cases: list[SegmentationCase]) -> list[dict[str, str]]:
    return [
        {"image": str(case.image_path), "label": str(case.mask_path), "case_id": case.case_id}
        for case in cases
    ]


def build_transforms(config: dict[str, Any], *, training: bool) -> Any:
    runtime = _require_training_runtime()
    compose = runtime["Compose"]
    transforms: list[Any] = [
        runtime["LoadImaged"](keys=["image", "label"], image_only=True),
        runtime["EnsureChannelFirstd"](keys=["image", "label"]),
        runtime["EnsureTyped"](
            keys=["image", "label"], dtype=runtime["torch"].float32, track_meta=False
        ),
        runtime["Lambdad"](keys="label", func=_binarize_label),
        runtime["NormalizeIntensityd"](
            keys="image", nonzero=bool(deep_get(config, "data.normalize_nonzero", True))
        ),
    ]
    if training:
        roi_size = tuple(int(v) for v in deep_get(config, "data.roi_size"))
        transforms.extend(
            [
                runtime["RandCropByPosNegLabeld"](
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=roi_size,
                    pos=float(deep_get(config, "augmentation.positive_ratio", 1.0)),
                    neg=float(deep_get(config, "augmentation.negative_ratio", 1.0)),
                    num_samples=int(deep_get(config, "data.samples_per_volume", 2)),
                    image_key="image",
                    image_threshold=float(deep_get(config, "augmentation.image_threshold", 0.0)),
                    allow_smaller=False,
                ),
                runtime["RandFlipd"](
                    keys=["image", "label"], spatial_axis=0, prob=0.5
                ),
                runtime["RandFlipd"](
                    keys=["image", "label"], spatial_axis=1, prob=0.5
                ),
                runtime["RandFlipd"](
                    keys=["image", "label"], spatial_axis=2, prob=0.5
                ),
                runtime["RandRotate90d"](
                    keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)
                ),
                runtime["RandScaleIntensityd"](
                    keys="image", factors=0.1, prob=0.2
                ),
                runtime["RandShiftIntensityd"](
                    keys="image", offsets=0.1, prob=0.2
                ),
            ]
        )
    return compose(transforms)


def build_loaders(config: dict[str, Any], data: SegResNetData) -> tuple[Any, Any]:
    runtime = _require_training_runtime()
    cache_rate = float(deep_get(config, "data.cache_rate", 0.0))
    dataset_class = runtime["CacheDataset"] if cache_rate > 0 else runtime["Dataset"]
    train_kwargs: dict[str, Any] = {}
    val_kwargs: dict[str, Any] = {}
    if cache_rate > 0:
        train_kwargs["cache_rate"] = cache_rate
        val_kwargs["cache_rate"] = cache_rate
    train_dataset = dataset_class(
        data=_records(data.cases_for("train")),
        transform=build_transforms(config, training=True),
        **train_kwargs,
    )
    val_dataset = dataset_class(
        data=_records(data.cases_for("val")),
        transform=build_transforms(config, training=False),
        **val_kwargs,
    )
    workers = int(deep_get(config, "training.workers", 0))
    common = {
        "num_workers": workers,
        "pin_memory": bool(deep_get(config, "training.pin_memory", True)),
        "persistent_workers": workers > 0,
    }
    train_loader = runtime["DataLoader"](
        train_dataset,
        batch_size=int(deep_get(config, "training.batch_size", 1)),
        shuffle=True,
        collate_fn=runtime["list_data_collate"],
        **common,
    )
    val_loader = runtime["DataLoader"](
        val_dataset,
        batch_size=1,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


def build_loss(config: dict[str, Any]) -> Any:
    runtime = _require_training_runtime()
    loss_name = str(deep_get(config, "training.loss", "dice_bce")).lower()
    if loss_name == "dice_bce":
        return runtime["DiceCELoss"](sigmoid=True, squared_pred=True)
    if loss_name == "tversky":
        return runtime["TverskyLoss"](
            sigmoid=True,
            alpha=float(deep_get(config, "training.tversky_alpha", 0.3)),
            beta=float(deep_get(config, "training.tversky_beta", 0.7)),
        )
    raise ValueError(f"Unsupported training.loss: {loss_name}")


def validate(model: Any, loader: Any, config: dict[str, Any], device: Any) -> dict[str, Any]:
    runtime = _require_training_runtime()
    torch = runtime["torch"]
    model.eval()
    roi_size = tuple(int(v) for v in deep_get(config, "data.roi_size"))
    threshold = float(deep_get(config, "inference.threshold", 0.5))
    use_amp = bool(deep_get(config, "training.amp", True)) and device.type == "cuda"
    case_rows: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            case_value = batch["case_id"]
            case_id = str(case_value[0] if isinstance(case_value, (list, tuple)) else case_value)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = runtime["sliding_window_inference"](
                    batch["image"],
                    roi_size=roi_size,
                    sw_batch_size=int(
                        deep_get(config, "inference.sliding_window_batch_size", 1)
                    ),
                    predictor=model,
                    overlap=float(deep_get(config, "inference.overlap", 0.5)),
                    mode="gaussian",
                    sw_device=device,
                    device=torch.device("cpu"),
                )
            prediction = torch.sigmoid(logits) >= threshold
            metrics = binary_metrics(
                prediction.detach().cpu().numpy(),
                (batch["label"] > 0).detach().cpu().numpy(),
            )
            case_rows[case_id] = metrics.as_dict()
    elapsed = time.perf_counter() - started
    if not case_rows:
        raise ValueError("Validation split is empty")
    summary = {
        key: float(np.mean([row[key] for row in case_rows.values()]))
        for key in ("dice", "iou", "precision", "recall")
    }
    summary["infer_sec_per_case"] = elapsed / len(case_rows)
    return {"summary": summary, "cases": case_rows}


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_dice: float,
    config: dict[str, Any],
) -> None:
    runtime = _require_training_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": "monai_segresnet",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "model_config": config.get("model", {}),
        "roi_size": deep_get(config, "data.roi_size"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    runtime["torch"].save(payload, temporary)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def train(
    config_path: str | Path,
    *,
    resume: str | Path | None = None,
    epochs: int | None = None,
    device_name: str | None = None,
    cache_rate: float | None = None,
    sw_batch_size: int | None = None,
) -> dict[str, Any]:
    config = load_monai_config(config_path)
    if cache_rate is not None:
        if not 0.0 <= cache_rate <= 1.0:
            raise ValueError("cache_rate must be between 0 and 1")
        config["data"]["cache_rate"] = cache_rate
    if sw_batch_size is not None:
        if sw_batch_size < 1:
            raise ValueError("sw_batch_size must be positive")
        config["inference"]["sliding_window_batch_size"] = sw_batch_size
    runtime = _require_training_runtime()
    torch = runtime["torch"]
    seed = int(deep_get(config, "training.seed", 42))
    runtime["set_determinism"](seed=seed)
    device = resolve_device(device_name or str(deep_get(config, "runtime.device", "auto")))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(deep_get(config, "runtime.cudnn_benchmark", True))

    data = load_data(config)
    train_loader, val_loader = build_loaders(config, data)
    model = build_model(config).to(device)
    optimizer_name = str(deep_get(config, "training.optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported training.optimizer: {optimizer_name}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(config, "training.learning_rate", 2e-4)),
        weight_decay=float(deep_get(config, "training.weight_decay", 1e-5)),
    )
    total_epochs = int(epochs or deep_get(config, "training.epochs", 200))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs, 1),
        eta_min=float(deep_get(config, "training.min_lr", 1e-6)),
    )
    use_amp = bool(deep_get(config, "training.amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_function = build_loss(config)

    start_epoch = 1
    best_dice = -math.inf
    if resume:
        payload = checkpoint_payload(resume, device)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        if "scaler_state" in payload:
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_dice = float(payload.get("best_dice", -math.inf))

    checkpoint_dir = resolve_project_path(
        deep_get(config, "training.checkpoint_dir", "checkpoints/monai_segresnet")
    )
    history_path = resolve_project_path(
        deep_get(config, "training.history_file", "outputs/monai_segresnet/history.json")
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if resume and history_path.is_file():
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(loaded_history, list):
            history = [row for row in loaded_history if int(row.get("epoch", 0)) < start_epoch]
    val_every = int(deep_get(config, "training.val_every", 5))
    if val_every < 1:
        raise ValueError("training.val_every must be positive")

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        loss_total = 0.0
        batches = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_function(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach().item())
            batches += 1
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_total / max(batches, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_sec": time.perf_counter() - epoch_started,
        }
        should_validate = epoch % val_every == 0 or epoch == total_epochs
        if should_validate:
            validation = validate(model, val_loader, config, device)
            row["validation"] = validation
            dice = float(validation["summary"]["dice"])
            if dice > best_dice:
                best_dice = dice
                _save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_dice=best_dice,
                    config=config,
                )
        history.append(row)
        _write_json_atomic(history_path, history)
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_dice=best_dice,
            config=config,
        )
        validation_text = ""
        if "validation" in row:
            validation_text = f", val_dice={row['validation']['summary']['dice']:.4f}"
        print(
            f"epoch {epoch}/{total_epochs}: loss={row['train_loss']:.4f}"
            f"{validation_text}, sec={row['epoch_sec']:.1f}",
            flush=True,
        )

    return {
        "device": str(device),
        "epochs": total_epochs,
        "best_dice": best_dice,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "history_file": str(history_path),
        "split": {
            "train": data.split.train,
            "val": data.split.val,
            "test": data.split.test,
        },
    }


def predict_split(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split_name: str = "test",
    output_dir: str | Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split_name}")
    config = load_monai_config(config_path)
    data = load_data(config)
    cases = data.cases_for(split_name)
    output_path = resolve_project_path(
        output_dir or deep_get(config, "inference.output_dir", "outputs/monai_segresnet")
    )
    output_path.mkdir(parents=True, exist_ok=True)
    segmenter = MonaiSegResNetSegmenter(config_path, device=device_name)
    segmenter.load(checkpoint_path)
    results = []
    for case in cases:
        result = segmenter.predict_volume(case.image_path, output_path / f"{case.case_id}.nii.gz")
        result["case_id"] = case.case_id
        results.append(result)
        print(f"predicted {case.case_id} in {result['elapsed_sec']:.2f}s", flush=True)
    manifest = {"split": split_name, "checkpoint": str(checkpoint_path), "results": results}
    (output_path / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def evaluate_split(
    config_path: str | Path,
    prediction_dir: str | Path,
    *,
    split_name: str = "test",
) -> dict[str, Any]:
    config = load_monai_config(config_path)
    data = load_data(config)
    prediction_dir = resolve_project_path(prediction_dir)
    per_case: dict[str, dict[str, Any]] = {}
    for case in data.cases_for(split_name):
        prediction_path = prediction_dir / f"{case.case_id}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction for {case.case_id}: {prediction_path}")
        prediction, _ = load_nifti_array(prediction_path)
        reference, _ = load_nifti_array(case.mask_path)
        metrics = binary_metrics(binarize_mask(prediction), binarize_mask(reference))
        per_case[case.case_id] = metrics.as_dict()
    if not per_case:
        raise ValueError(f"Split is empty: {split_name}")
    summary = {
        key: float(np.mean([row[key] for row in per_case.values()]))
        for key in ("dice", "iou", "precision", "recall")
    }
    manifest_path = prediction_dir / "prediction_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        times = [float(row["elapsed_sec"]) for row in manifest.get("results", [])]
        if times:
            summary["infer_sec_per_case"] = float(np.mean(times))
    return {"split": split_name, "summary": summary, "cases": per_case}
