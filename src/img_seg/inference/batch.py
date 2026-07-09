"""Batch inference helpers shared by the WebUI and future CLIs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from img_seg.config import deep_get, resolve_project_path
from img_seg.evaluation.metrics import binary_metrics
from img_seg.evaluation.visualize import choose_slice, normalize_image, take_slice
from img_seg.io.nifti import binarize_mask, load_nifti_array, require_nibabel, save_like
from img_seg.models.nnunet_v2 import (
    dataset_folder_name,
    load_nnunet_config,
    nnunet_paths_from_config,
)

DEFAULT_NNUNET_CONFIG = Path("configs/nnunet_v2/base.yaml")
DEFAULT_OUTPUT_DIR = Path("Test_Seg")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ModelKey = Literal["nnunet_v2"]
InputKind = Literal["nifti", "image"]


@dataclass(frozen=True)
class ModelInfo:
    key: ModelKey
    label: str
    runnable: bool = True


@dataclass(frozen=True)
class CheckpointInfo:
    label: str
    path: Path
    checkpoint_name: str


@dataclass(frozen=True)
class InferenceInput:
    source_path: Path
    case_id: str
    kind: InputKind


@dataclass(frozen=True)
class InferenceResult:
    case_id: str
    input_kind: InputKind
    input_path: Path
    output_path: Path
    elapsed_sec: float
    shape: list[int]
    status: str
    message: str = ""


class InferenceCancelledError(RuntimeError):
    """Raised when the user requests cancellation of a running inference."""


@dataclass(frozen=True)
class ResultMetrics:
    case_id: str
    foreground: int
    total: int
    foreground_ratio: float
    dice: float | None = None
    iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    reference_path: Path | None = None
    message: str = ""


def list_available_models() -> list[ModelInfo]:
    return [ModelInfo(key="nnunet_v2", label="nnU-Net v2 2D")]


def strip_known_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".jpeg", ".jpg", ".png"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def sanitize_case_id(value: str) -> str:
    value = value.removesuffix("_0000")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "case"


def normalize_reference_case_id(path: Path) -> str:
    case_id = sanitize_case_id(strip_known_suffix(path))
    case_id = re.sub(r"([._-]?(seg|mask|label))$", "", case_id, flags=re.IGNORECASE)
    return sanitize_case_id(case_id)


def is_nifti_path(path: Path) -> bool:
    return path.name.lower().endswith((".nii", ".nii.gz"))


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _coerce_upload_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    name = getattr(value, "name", None)
    if name:
        return Path(name)
    path = getattr(value, "path", None)
    if path:
        return Path(path)
    return None


def _iter_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if is_nifti_path(path) or is_image_path(path) else []
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files = [
        file
        for file in path.rglob("*")
        if file.is_file() and (is_nifti_path(file) or is_image_path(file))
    ]
    return sorted(files, key=lambda item: str(item).lower())


def collect_inputs(
    input_path: str | Path | None = None,
    uploaded_files: list[object] | None = None,
) -> list[InferenceInput]:
    """Collect NIfTI and image inputs from a local path and/or Gradio uploads."""

    paths: list[Path] = []
    if input_path:
        paths.extend(_iter_supported_files(resolve_project_path(input_path)))
    for uploaded in uploaded_files or []:
        path = _coerce_upload_path(uploaded)
        if path is not None:
            paths.extend(_iter_supported_files(path))

    unique: list[Path] = []
    seen_paths: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique.append(path)

    used_ids: dict[str, int] = {}
    inputs: list[InferenceInput] = []
    for path in unique:
        base_case_id = sanitize_case_id(strip_known_suffix(path))
        count = used_ids.get(base_case_id, 0)
        used_ids[base_case_id] = count + 1
        case_id = base_case_id if count == 0 else f"{base_case_id}_{count + 1}"
        kind: InputKind = "nifti" if is_nifti_path(path) else "image"
        inputs.append(InferenceInput(source_path=path, case_id=case_id, kind=kind))

    if not inputs:
        raise ValueError(
            "No supported input files found. Expected .nii, .nii.gz, .jpg, .jpeg, or .png."
        )
    return inputs


def collect_reference_masks(
    reference_path: str | Path | None = None,
    uploaded_files: list[object] | None = None,
) -> dict[str, Path]:
    paths: list[Path] = []
    if reference_path:
        paths.extend(_iter_supported_files(resolve_project_path(reference_path)))
    for uploaded in uploaded_files or []:
        path = _coerce_upload_path(uploaded)
        if path is not None:
            paths.extend(_iter_supported_files(path))

    references: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        case_id = normalize_reference_case_id(path)
        if case_id in references:
            raise ValueError(
                f"Duplicate reference mask for case {case_id}: {references[case_id]} and {path}"
            )
        references[case_id] = path
    return references


def list_checkpoints(
    model_key: ModelKey = "nnunet_v2",
    *,
    config_path: str | Path = DEFAULT_NNUNET_CONFIG,
    configuration: str = "2d",
    fold: str = "0",
) -> list[CheckpointInfo]:
    if model_key != "nnunet_v2":
        return []
    config = load_nnunet_config(config_path)
    paths = nnunet_paths_from_config(config)
    dataset_name = dataset_folder_name(
        int(deep_get(config, "dataset.id")),
        str(deep_get(config, "dataset.name")),
    )
    trainer = str(deep_get(config, "training.trainer", "nnUNetTrainer"))
    fold_dir = (
        paths.results
        / dataset_name
        / f"{trainer}__nnUNetPlans__{configuration}"
        / f"fold_{fold}"
    )
    checkpoints = sorted(fold_dir.glob("checkpoint*.pth"), key=lambda item: item.name)

    def priority(path: Path) -> tuple[int, str]:
        if path.name == "checkpoint_final.pth":
            return (0, path.name)
        if path.name == "checkpoint_best.pth":
            return (1, path.name)
        return (2, path.name)

    return [
        CheckpointInfo(
            label=f"{checkpoint.name} - {checkpoint.parent.name}",
            path=checkpoint,
            checkpoint_name=checkpoint.name,
        )
        for checkpoint in sorted(checkpoints, key=priority)
    ]


def _nnunet_fold_dir(config: dict, configuration: str, fold: str) -> Path:
    paths = nnunet_paths_from_config(config)
    dataset_name = dataset_folder_name(
        int(deep_get(config, "dataset.id")),
        str(deep_get(config, "dataset.name")),
    )
    trainer = str(deep_get(config, "training.trainer", "nnUNetTrainer"))
    return (
        paths.results
        / dataset_name
        / f"{trainer}__nnUNetPlans__{configuration}"
        / f"fold_{fold}"
    )


def _copy_nifti_to_nnunet_input(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.name.lower().endswith(".nii.gz"):
        shutil.copyfile(source_path, output_path)
        return
    data, reference = load_nifti_array(source_path)
    save_like(reference, np.asanyarray(data), output_path)


def _image_to_nifti(source_path: Path, output_path: Path) -> list[int]:
    cv2 = _require_cv2()
    nib = require_nibabel()
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")
    volume = image.astype(np.float32)[:, :, np.newaxis]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(output_path))
    return list(image.shape)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for .jpg/.png WebUI inference.") from exc
    return cv2


def _load_binary_mask(path: Path) -> np.ndarray:
    if is_nifti_path(path):
        data, _ = load_nifti_array(path)
        return binarize_mask(data)
    if is_image_path(path):
        cv2 = _require_cv2()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read mask image: {path}")
        return binarize_mask(image)
    raise ValueError(f"Unsupported mask format: {path}")


def _is_cancelled(cancel_event: threading.Event | None) -> bool:
    return bool(cancel_event and cancel_event.is_set())


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if _is_cancelled(cancel_event):
        raise InferenceCancelledError("推理已由用户终止。")


def _prepare_nnunet_input(
    inputs: list[InferenceInput],
    nnunet_input_dir: Path,
    cancel_event: threading.Event | None = None,
) -> dict[str, list[int]]:
    source_shapes: dict[str, list[int]] = {}
    for item in inputs:
        _raise_if_cancelled(cancel_event)
        nnunet_path = nnunet_input_dir / f"{item.case_id}_0000.nii.gz"
        if item.kind == "nifti":
            _copy_nifti_to_nnunet_input(item.source_path, nnunet_path)
            data, _ = load_nifti_array(item.source_path)
            source_shapes[item.case_id] = list(np.asarray(data).shape)
        else:
            source_shapes[item.case_id] = _image_to_nifti(item.source_path, nnunet_path)
    return source_shapes


def _binarize_nifti_prediction(path: Path) -> list[int]:
    prediction, reference = load_nifti_array(path)
    binary = binarize_mask(prediction)
    save_like(reference, binary, path)
    return list(binary.shape)


def _export_png_prediction(nifti_prediction_path: Path, output_path: Path) -> list[int]:
    cv2 = _require_cv2()
    prediction, _ = load_nifti_array(nifti_prediction_path)
    mask = np.squeeze(binarize_mask(prediction))
    if mask.ndim != 2:
        mask = take_slice(mask, axis=2, index=choose_slice(mask, axis=2))
    png = (mask > 0).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), png)
    return list(png.shape)


def _checkpoint_name(checkpoint_path_or_name: str | Path | None) -> str | None:
    if not checkpoint_path_or_name:
        return None
    return Path(checkpoint_path_or_name).name


def _prepare_checkpoint_name(
    checkpoint_path_or_name: str | Path | None,
    *,
    config_path: str | Path,
    configuration: str,
    folds: str,
) -> str | None:
    if not checkpoint_path_or_name:
        return None
    checkpoint_path = Path(checkpoint_path_or_name)
    if not checkpoint_path.exists():
        return _checkpoint_name(checkpoint_path_or_name)

    first_fold = str(folds).split(",")[0].strip() or "0"
    config = load_nnunet_config(config_path)
    fold_dir = _nnunet_fold_dir(config, configuration, first_fold)
    if checkpoint_path.parent.resolve() == fold_dir.resolve():
        return checkpoint_path.name

    fold_dir.mkdir(parents=True, exist_ok=True)
    target = fold_dir / checkpoint_path.name
    if not target.exists():
        shutil.copyfile(checkpoint_path, target)
    return target.name


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    process.terminate()


def _run_nnunet_predict(
    config_path: str | Path,
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    configuration: str,
    folds: str,
    checkpoint_name: str | None,
    cancel_event: threading.Event | None = None,
) -> None:
    config = load_nnunet_config(config_path)
    paths = nnunet_paths_from_config(config)
    dataset_id = str(deep_get(config, "dataset.id"))
    trainer = str(deep_get(config, "training.trainer", "nnUNetTrainer"))
    command = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        dataset_id,
        "-tr",
        trainer,
        "-c",
        configuration,
        "-f",
        folds,
    ]
    if checkpoint_name:
        command.extend(["-chk", checkpoint_name])

    print("+ " + " ".join(command), flush=True)
    _raise_if_cancelled(cancel_event)
    process = subprocess.Popen(
        command,
        env=paths.env(deep_get(config, "runtime", {})),
        text=True,
    )
    while process.poll() is None:
        if _is_cancelled(cancel_event):
            _terminate_process_tree(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise InferenceCancelledError("推理已由用户终止。")
        time.sleep(0.5)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def run_batch_inference(
    *,
    model_key: ModelKey,
    inputs: list[InferenceInput],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint_path_or_name: str | Path | None = None,
    config_path: str | Path = DEFAULT_NNUNET_CONFIG,
    configuration: str = "2d",
    folds: str = "0",
    cancel_event: threading.Event | None = None,
) -> list[InferenceResult]:
    if model_key != "nnunet_v2":
        raise ValueError(f"Unsupported model: {model_key}")
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="imgseg_webui_") as temp_dir:
        nnunet_input_dir = Path(temp_dir) / "nnunet_input"
        source_shapes = _prepare_nnunet_input(inputs, nnunet_input_dir, cancel_event)
        started = time.perf_counter()
        _run_nnunet_predict(
            config_path,
            input_dir=nnunet_input_dir,
            output_dir=output_path,
            configuration=configuration,
            folds=folds,
            checkpoint_name=_prepare_checkpoint_name(
                checkpoint_path_or_name,
                config_path=config_path,
                configuration=configuration,
                folds=folds,
            ),
            cancel_event=cancel_event,
        )
        elapsed = time.perf_counter() - started

    per_case_elapsed = elapsed / max(len(inputs), 1)
    results: list[InferenceResult] = []
    for item in inputs:
        _raise_if_cancelled(cancel_event)
        nnunet_prediction = output_path / f"{item.case_id}.nii.gz"
        if not nnunet_prediction.exists():
            results.append(
                InferenceResult(
                    case_id=item.case_id,
                    input_kind=item.kind,
                    input_path=item.source_path,
                    output_path=nnunet_prediction,
                    elapsed_sec=per_case_elapsed,
                    shape=source_shapes.get(item.case_id, []),
                    status="failed",
                    message=f"Prediction file was not created: {nnunet_prediction}",
                )
            )
            continue

        if item.kind == "image":
            final_output = output_path / f"{item.case_id}.png"
            shape = _export_png_prediction(nnunet_prediction, final_output)
            nnunet_prediction.unlink(missing_ok=True)
        else:
            final_output = nnunet_prediction
            shape = _binarize_nifti_prediction(final_output)

        results.append(
            InferenceResult(
                case_id=item.case_id,
                input_kind=item.kind,
                input_path=item.source_path,
                output_path=final_output,
                elapsed_sec=per_case_elapsed,
                shape=shape,
                status="done",
            )
        )
    return results


def evaluate_result_metrics(
    results: list[InferenceResult],
    references: dict[str, Path] | None = None,
) -> list[ResultMetrics]:
    references = references or {}
    rows: list[ResultMetrics] = []
    for result in results:
        if result.status != "done":
            rows.append(
                ResultMetrics(
                    case_id=result.case_id,
                    foreground=0,
                    total=0,
                    foreground_ratio=0.0,
                    message=result.message or "Prediction failed.",
                )
            )
            continue

        prediction = _load_binary_mask(result.output_path)
        foreground = int(prediction.sum())
        total = int(prediction.size)
        foreground_ratio = foreground / total if total else 0.0
        reference_path = references.get(result.case_id)
        if reference_path is None:
            rows.append(
                ResultMetrics(
                    case_id=result.case_id,
                    foreground=foreground,
                    total=total,
                    foreground_ratio=foreground_ratio,
                    message="No reference mask; showing prediction statistics only.",
                )
            )
            continue

        reference = _load_binary_mask(reference_path)
        if prediction.shape != reference.shape:
            rows.append(
                ResultMetrics(
                    case_id=result.case_id,
                    foreground=foreground,
                    total=total,
                    foreground_ratio=foreground_ratio,
                    reference_path=reference_path,
                    message=(
                        f"Reference shape mismatch: prediction {prediction.shape}, "
                        f"reference {reference.shape}."
                    ),
                )
            )
            continue

        metrics = binary_metrics(prediction, reference)
        rows.append(
            ResultMetrics(
                case_id=result.case_id,
                foreground=foreground,
                total=total,
                foreground_ratio=foreground_ratio,
                dice=metrics.dice,
                iou=metrics.iou,
                precision=metrics.precision,
                recall=metrics.recall,
                reference_path=reference_path,
            )
        )
    return rows


def make_preview(
    input_path: str | Path,
    prediction_path: str | Path,
    preview_dir: str | Path,
    *,
    axis: int | None = None,
    slice_index: int | None = None,
) -> Path:
    input_path = Path(input_path)
    prediction_path = Path(prediction_path)
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = preview_dir / f"{strip_known_suffix(prediction_path)}_preview.png"
    if is_image_path(input_path):
        return _make_image_preview(input_path, prediction_path, output_path)
    return _make_nifti_preview(
        input_path,
        prediction_path,
        output_path,
        axis=axis,
        slice_index=slice_index,
    )


def _make_nifti_preview(
    image_path: Path,
    prediction_path: Path,
    output_path: Path,
    *,
    axis: int | None = None,
    slice_index: int | None = None,
) -> Path:
    image, _ = load_nifti_array(image_path)
    prediction, _ = load_nifti_array(prediction_path)
    pred_mask = binarize_mask(prediction)
    if pred_mask.ndim == 2:
        index = 0
        image_slice = normalize_image(np.asarray(image))
        pred_slice = pred_mask
    else:
        axis = 2 if axis is None else axis
        if axis < 0 or axis >= pred_mask.ndim:
            raise ValueError(f"Preview axis {axis} is out of range for shape {pred_mask.shape}")
        index = choose_slice(pred_mask, axis) if slice_index is None else slice_index
        if index < 0 or index >= pred_mask.shape[axis]:
            raise ValueError(
                f"Preview slice {index} is out of range for axis {axis} "
                f"with size {pred_mask.shape[axis]}"
            )
        image_slice = normalize_image(take_slice(np.asarray(image), axis, index))
        pred_slice = take_slice(pred_mask, axis, index)
    mask_display = np.rot90(pred_slice).astype(float)

    plt.figure(figsize=(7, 7), dpi=150)
    plt.imshow(np.rot90(image_slice), cmap="gray")
    if np.any(pred_slice):
        plt.imshow(mask_display, cmap="Reds", alpha=mask_display * 0.35, vmin=0, vmax=1)
        plt.contour(mask_display, levels=[0.5], colors=["#dc2626"], linewidths=0.8)
    plt.title(f"slice={index}, red=prediction region")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def _make_image_preview(image_path: Path, prediction_path: Path, output_path: Path) -> Path:
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Could not read preview inputs: {image_path}, {prediction_path}")
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    overlay[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(image, 0.68, overlay, 0.32, 0)
    cv2.imwrite(str(output_path), blended)
    return output_path
