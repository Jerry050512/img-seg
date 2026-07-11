"""Compatibility helpers for the course-mandated root entry scripts."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from img_seg.config import load_yaml, resolve_project_path
from img_seg.evaluation.metrics import binary_confusion, binary_metrics
from img_seg.inference import collect_inputs, make_preview, run_batch_inference
from img_seg.inference.batch import InferenceResult, is_nifti_path
from img_seg.io.nifti import binarize_mask, load_nifti_array

MODEL_CONFIGS = {
    "nnunet_v2": Path("configs/nnunet_v2/base.yaml"),
    "monai_segresnet": Path("configs/monai_segresnet/base.yaml"),
    "efficientnet_b0": Path("configs/efficientnet_b0/base.yaml"),
}


def normalize_legacy_argv(argv: list[str]) -> list[str]:
    """Accept the course handout's ``image PATH weight PATH`` spelling."""

    aliases = {
        "image": "--image",
        "weight": "--weight",
        "output": "--output",
        "data_path": "--data-path",
        "weight_path": "--weight-path",
    }
    return [aliases.get(value, value) for value in argv]


def config_path_for(model: str, config: str | Path | None) -> Path:
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model}")
    return resolve_project_path(config or MODEL_CONFIGS[model])


@contextmanager
def config_with_dataset(
    model: str,
    config_path: str | Path,
    data_path: str | Path | None,
    output_dir: str | Path | None = None,
) -> Iterator[Path]:
    """Yield a config whose dataset and artifact roots come from the command line."""

    source = resolve_project_path(config_path)
    if data_path is None and (output_dir is None or model == "efficientnet_b0"):
        yield source
        return
    config = load_yaml(source)
    with tempfile.TemporaryDirectory(prefix="imgseg_course_config_") as directory:
        temp = Path(directory)
        if data_path is not None:
            dataset_root = resolve_project_path(data_path)
            if not dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
            manifest_path = temp / "dataset.yaml"
            split_path = temp / "split.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "root": str(dataset_root),
                        "layout": "case_dir_pair",
                        "image_name": "image.nii.gz",
                        "mask_name": "mask.nii.gz",
                        "mask_policy": "nonzero_to_one",
                        "case_id_policy": "directory_name",
                        "exclude": ["processed", "_standardizing_tmp"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                yaml.safe_dump(
                    {
                        "seed": 42,
                        "strategy": "case_level",
                        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            if model == "nnunet_v2":
                config.setdefault("paths", {})["dataset_file"] = str(manifest_path)
                config["paths"]["split_file"] = str(split_path)
                config["paths"]["raw_dataset_dir"] = str(dataset_root)
            else:
                config.setdefault("data", {})["dataset_config"] = str(manifest_path)
                config["data"]["split_file"] = str(split_path)
        if output_dir is not None:
            artifact_root = resolve_project_path(output_dir)
            if model == "monai_segresnet":
                config.setdefault("training", {})["checkpoint_dir"] = str(artifact_root)
                config["training"]["history_file"] = str(artifact_root / "history.json")
            elif model == "nnunet_v2":
                config.setdefault("paths", {})["nnunet_results"] = str(artifact_root)
        generated = temp / "config.yaml"
        generated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        yield generated


def run_training(
    *,
    model: str,
    config_path: str | Path,
    data_path: str | Path | None,
    epochs: int | None,
    device: str | None,
    resume: str | Path | None,
    output_dir: str | Path | None,
    skip_prepare: bool,
    skip_plan: bool,
    dry_run: bool,
) -> dict[str, Any]:
    with config_with_dataset(model, config_path, data_path, output_dir) as effective_config:
        if model == "efficientnet_b0":
            from img_seg.training.efficientnet_b0_cli import train_model

            return train_model(effective_config, epochs=epochs, output_dir=output_dir)
        if model == "monai_segresnet":
            from img_seg.training.monai_segresnet import train

            return train(
                effective_config,
                resume=resume,
                epochs=epochs,
                device_name=device,
            )
        from img_seg.models.nnunet_v2 import plan_and_preprocess, prepare_nnunet_dataset, train

        if not skip_prepare:
            prepare_nnunet_dataset(effective_config)
        if not skip_plan:
            plan_and_preprocess(effective_config, dry_run=dry_run)
        train(
            effective_config,
            configuration="2d",
            fold="0",
            continue_training=resume is not None,
            dry_run=dry_run,
        )
        return {"model": model, "configuration": "2d", "fold": "0", "dry_run": dry_run}


def run_predictions(
    *,
    model: str,
    checkpoint: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    device: str | None,
) -> list[InferenceResult]:
    inputs = collect_inputs(input_path)
    return run_batch_inference(
        model_key=model,
        inputs=inputs,
        output_dir=output_dir,
        checkpoint_path_or_name=checkpoint,
        config_path=config_path,
        device=device,
    )


def _reference_for_input(result: InferenceResult, reference_root: Path | None) -> Path:
    suffixes = (".nii.gz", ".nii") if result.input_kind == "nifti" else (".png", ".tif", ".tiff")
    roots = [reference_root] if reference_root is not None else [result.input_path.parent]
    names = [
        "mask.nii.gz",
        "mask.nii",
        f"{result.case_id}_mask.nii.gz",
        f"{result.case_id}_Seg.nii.gz",
        f"{result.case_id}.nii.gz",
        f"{result.case_id}_mask.png",
        f"{result.case_id}.png",
    ]
    for root in roots:
        if root is None:
            continue
        for name in names:
            candidate = root / name
            if candidate.is_file() and candidate.resolve() != result.input_path.resolve():
                return candidate
        for candidate in root.rglob("*"):
            if not candidate.is_file() or not candidate.name.lower().endswith(suffixes):
                continue
            lowered = candidate.name.lower()
            is_reference = any(x in lowered for x in ("mask", "seg", "label"))
            if result.case_id.lower() in lowered and is_reference:
                return candidate
    raise FileNotFoundError(f"No reference mask found for case {result.case_id}")


def _load_mask(path: Path) -> np.ndarray:
    if is_nifti_path(path):
        array, _ = load_nifti_array(path)
        return binarize_mask(array)
    import cv2

    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise ValueError(f"Could not read mask: {path}")
    return (array > 0).astype(np.uint8)


def checkpoint_parameter_count(checkpoint: str | Path) -> int:
    """Count model tensors without depending on backend-specific key names."""

    import torch

    payload = torch.load(resolve_project_path(checkpoint), map_location="cpu", weights_only=False)
    candidates: list[Any] = [payload]
    if isinstance(payload, dict):
        candidates = [
            payload.get("model_state_dict"),
            payload.get("model_state"),
            payload.get("network_weights"),
            payload.get("state_dict"),
            payload,
        ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            tensors = [
                value
                for name, value in candidate.items()
                if torch.is_tensor(value)
                and not str(name).endswith(("running_mean", "running_var", "num_batches_tracked"))
            ]
            if tensors:
                return sum(int(tensor.numel()) for tensor in tensors)
    return 0


def evaluate_predictions(
    results: list[InferenceResult],
    *,
    checkpoint: str | Path,
    reference_path: str | Path | None,
) -> dict[str, float]:
    reference_root = resolve_project_path(reference_path) if reference_path else None
    rows = []
    mean_ious = []
    total_tp = total_fp = total_fn = total_tn = 0
    elapsed = 0.0
    for result in results:
        if result.status != "done":
            raise RuntimeError(f"Prediction failed for {result.case_id}: {result.message}")
        reference = _reference_for_input(result, reference_root)
        prediction_array = _load_mask(result.output_path)
        reference_array = _load_mask(reference)
        metrics = binary_metrics(prediction_array, reference_array)
        rows.append(metrics)
        tp, fp, fn, tn = binary_confusion(prediction_array, reference_array)
        background_iou = tn / (tn + fp + fn) if (tn + fp + fn) else 1.0
        mean_ious.append((metrics.iou + background_iou) / 2)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        elapsed += result.elapsed_sec
    if not rows:
        raise ValueError("No predictions were evaluated")

    def mean(key: str) -> float:
        return float(np.mean([getattr(row, key) for row in rows]))

    total_pixels = total_tp + total_fp + total_fn + total_tn
    params = checkpoint_parameter_count(checkpoint)
    return {
        "mIoU": float(np.mean(mean_ious)),
        "Dice": mean("dice"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "PixelAcc": (total_tp + total_tn) / total_pixels if total_pixels else 0.0,
        "FPS": len(rows) / elapsed if elapsed > 0 else 0.0,
        "Params(M)": params / 1_000_000,
    }


def write_prediction_artifacts(
    results: list[InferenceResult],
    *,
    requested_output: str | Path | None,
    output_dir: str | Path,
) -> list[dict[str, str]]:
    """Keep binary masks and create a readable overlay/preview for every result."""

    root = resolve_project_path(output_dir)
    preview_root = root / "previews"
    artifacts: list[dict[str, str]] = []
    single_output = resolve_project_path(requested_output) if requested_output else None
    if single_output is not None and len(results) != 1:
        raise ValueError("--output can only be used with one input; use --output-dir for folders")
    for result in results:
        mask_path = result.output_path
        requested_preview: Path | None = None
        if single_output is not None and single_output.name.lower().endswith((".nii", ".nii.gz")):
            single_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mask_path), str(single_output))
            mask_path = single_output
            requested_preview = single_output.with_name(
                single_output.name.removesuffix(".gz").removesuffix(".nii") + "_preview.png"
            )
            preview_dir = requested_preview.parent
        elif single_output is not None:
            requested_preview = single_output
            preview_dir = single_output.parent
        else:
            preview_dir = preview_root
        preview = make_preview(result.input_path, mask_path, preview_dir)
        if requested_preview is not None and preview.resolve() != requested_preview.resolve():
            requested_preview.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(preview), str(requested_preview))
            preview = requested_preview
        artifacts.append(
            {"case_id": result.case_id, "mask": str(mask_path), "preview": str(preview)}
        )
    return artifacts


def rounded_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in payload.items()
        },
        ensure_ascii=False,
    )
