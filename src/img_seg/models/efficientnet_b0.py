"""EfficientNet-B0 encoder adapter for 2D lightweight segmentation."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import URLError

import cv2
import numpy as np
import torch

from img_seg.config import deep_get, load_yaml
from img_seg.data.slices import axis_to_index, normalize_slice, resize_slice, take_slice
from img_seg.io.nifti import load_nifti_array, save_like
from img_seg.models.base import PredictionResult

MODEL_KEY = "efficientnet_b0"
DEFAULT_CONFIG_PATH = Path("configs/efficientnet_b0/base.yaml")


def load_efficientnet_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("model_key") != MODEL_KEY:
        raise ValueError(f"Expected model_key={MODEL_KEY!r} in {config_path}")
    return config


def device_from_config(config: dict[str, Any], requested: str | None = None) -> torch.device:
    device_name = requested or str(deep_get(config, "runtime.device", "auto"))
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _build_smp_model(
    config: dict[str, Any],
    *,
    use_pretrained_encoder: bool,
) -> tuple[torch.nn.Module, str]:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation-models-pytorch is required. Install dependencies with `uv sync`."
        ) from exc

    architecture = str(deep_get(config, "model.architecture", "Unet"))
    encoder_name = str(deep_get(config, "model.encoder_name", "efficientnet-b0"))
    encoder_weights = (
        deep_get(config, "model.encoder_weights", None) if use_pretrained_encoder else None
    )
    kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": int(deep_get(config, "model.in_channels", 1)),
        "classes": int(deep_get(config, "model.classes", 1)),
    }
    try:
        constructor = getattr(smp, architecture)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown segmentation_models_pytorch architecture: {architecture}"
        ) from exc
    try:
        model = constructor(**kwargs)
        initialization = str(encoder_weights) if encoder_weights is not None else "random"
        return model, initialization
    except (OSError, RuntimeError, URLError) as pretrained_error:
        if encoder_weights is None:
            raise
        kwargs["encoder_weights"] = None
        model = constructor(**kwargs)
        warnings.warn(
            "Could not initialize pretrained encoder weights; falling back to random "
            f"initialization ({type(pretrained_error).__name__}: {pretrained_error}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return model, "random_fallback"


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dict or a mapping containing model weights")
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _validate_checkpoint_config(
    checkpoint: object,
    current_config: Mapping[str, Any],
) -> None:
    if not isinstance(checkpoint, dict):
        return
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        return

    mismatches = []
    for key in (
        "model_key",
        "model.architecture",
        "model.encoder_name",
        "model.in_channels",
        "model.classes",
        "data.image_size",
        "data.slice_axis",
    ):
        saved_value = deep_get(saved_config, key)
        current_value = deep_get(current_config, key)
        if key == "data.image_size" and saved_value is not None and current_value is not None:
            saved_value = tuple(int(value) for value in saved_value)
            current_value = tuple(int(value) for value in current_value)
        elif key == "data.slice_axis" and saved_value is not None and current_value is not None:
            saved_value = axis_to_index(saved_value)
            current_value = axis_to_index(current_value)
        if saved_value is not None and current_value is not None and saved_value != current_value:
            mismatches.append(f"{key}: checkpoint={saved_value!r}, config={current_value!r}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Checkpoint configuration is incompatible with the current config: " + details
        )


def _case_id_from_nifti(path: Path) -> str:
    for suffix in (".nii.gz", ".nii"):
        if path.name.endswith(suffix):
            return path.name.removesuffix(suffix).removesuffix("_0000")
    return path.stem


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


class EfficientNetB0Segmenter:
    name = "2D U-Net with EfficientNet-B0 encoder"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str | torch.device | None = None,
        use_pretrained_encoder: bool = False,
    ) -> None:
        self.config = config or load_efficientnet_config(config_path)
        self.device = (
            torch.device(device) if device is not None else device_from_config(self.config)
        )
        self.image_size = tuple(
            int(x) for x in deep_get(self.config, "data.image_size", [512, 512])
        )
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError(
                f"data.image_size must contain two positive integers: {self.image_size}"
            )
        self.slice_axis = axis_to_index(deep_get(self.config, "data.slice_axis", "z"))
        self.threshold = float(deep_get(self.config, "inference.threshold", 0.5))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"inference.threshold must be between 0 and 1: {self.threshold}")
        self.batch_size = int(deep_get(self.config, "inference.batch_size", 8))
        if self.batch_size <= 0:
            raise ValueError(f"inference.batch_size must be positive: {self.batch_size}")
        self.use_amp = (
            bool(deep_get(self.config, "inference.amp", True)) and self.device.type == "cuda"
        )
        model, self.encoder_initialization = _build_smp_model(
            self.config,
            use_pretrained_encoder=use_pretrained_encoder,
        )
        self.model = model.to(self.device)
        self.model.eval()

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        _validate_checkpoint_config(checkpoint, self.config)
        self.model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
        self.model.to(self.device)
        self.model.eval()
        self.encoder_initialization = "checkpoint"

    def _predict_resized_masks(
        self,
        image_slices: Sequence[np.ndarray],
        output_shapes: Sequence[tuple[int, int]],
    ) -> list[np.ndarray]:
        if not image_slices:
            return []
        if len(image_slices) != len(output_shapes):
            raise ValueError("image_slices and output_shapes must have the same length")

        images = [
            resize_slice(
                normalize_slice(image_slice),
                self.image_size,
                interpolation=cv2.INTER_LINEAR,
            )
            for image_slice in image_slices
        ]
        array = np.stack(images, axis=0)[:, None, ...].astype(np.float32, copy=False)
        tensor = torch.from_numpy(array).to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=self.use_amp),
        ):
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)
        masks = (probabilities[:, 0] >= self.threshold).to(torch.uint8).cpu().numpy()
        return [
            _as_uint8_mask(
                resize_slice(mask, output_shape, interpolation=cv2.INTER_NEAREST)
            )
            for mask, output_shape in zip(masks, output_shapes, strict=True)
        ]

    def _predict_resized_mask(
        self,
        image_slice: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        return self._predict_resized_masks([image_slice], [output_shape])[0]

    def predict_volume(
        self,
        image_path: str | Path,
        output_path: str | Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> PredictionResult:
        started = time.perf_counter()
        image_path = Path(image_path)
        image, reference = load_nifti_array(image_path)
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume, got shape {image.shape}: {image_path}")
        output = np.zeros(tuple(int(dim) for dim in image.shape), dtype=np.uint8)
        slice_count = output.shape[self.slice_axis]
        for start in range(0, slice_count, self.batch_size):
            if check_cancelled is not None:
                check_cancelled()
            indices = list(range(start, min(start + self.batch_size, slice_count)))
            image_slices = [take_slice(image, self.slice_axis, index) for index in indices]
            output_shapes = [tuple(int(dim) for dim in item.shape) for item in image_slices]
            masks = self._predict_resized_masks(image_slices, output_shapes)
            for slice_index, mask in zip(indices, masks, strict=True):
                slicer: list[object] = [slice(None), slice(None), slice(None)]
                slicer[self.slice_axis] = slice_index
                output[tuple(slicer)] = mask
        saved_path = save_like(reference, output, output_path)
        return {
            "case_id": _case_id_from_nifti(image_path),
            "output_path": str(saved_path),
            "elapsed_sec": time.perf_counter() - started,
            "shape": [int(dim) for dim in output.shape],
        }

    def predict_image(self, image_path: str | Path, output_path: str | Path) -> PredictionResult:
        started = time.perf_counter()
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        mask = self._predict_resized_mask(image, tuple(int(dim) for dim in image.shape))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), (mask * 255).astype(np.uint8)):
            raise OSError(f"Could not write prediction image: {output_path}")
        return {
            "case_id": image_path.stem,
            "output_path": str(output_path),
            "elapsed_sec": time.perf_counter() - started,
            "shape": [int(dim) for dim in image.shape],
        }
