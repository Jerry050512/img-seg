"""EfficientNet-B0 encoder adapter for 2D lightweight segmentation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

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


def _build_smp_model(config: dict[str, Any]) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation-models-pytorch is required. Install dependencies with `uv sync`."
        ) from exc

    architecture = str(deep_get(config, "model.architecture", "Unet"))
    encoder_name = str(deep_get(config, "model.encoder_name", "efficientnet-b0"))
    encoder_weights = deep_get(config, "model.encoder_weights", None)
    kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": int(deep_get(config, "model.in_channels", 1)),
        "classes": int(deep_get(config, "model.classes", 1)),
    }
    constructor = getattr(smp, architecture)
    try:
        return constructor(**kwargs)
    except Exception:
        if encoder_weights is None:
            raise
        kwargs["encoder_weights"] = None
        print(
            "Could not initialize pretrained encoder weights; "
            "falling back to random initialization."
        )
        return constructor(**kwargs)


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dict or a mapping containing model weights")
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _case_id_from_nifti(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name.removesuffix(".nii.gz")
    if name.endswith(".nii"):
        return name.removesuffix(".nii")
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
    ) -> None:
        self.config = config or load_efficientnet_config(config_path)
        self.device = (
            torch.device(device) if device is not None else device_from_config(self.config)
        )
        self.image_size = tuple(
            int(x) for x in deep_get(self.config, "data.image_size", [512, 512])
        )
        self.slice_axis = axis_to_index(deep_get(self.config, "data.slice_axis", "z"))
        self.threshold = float(deep_get(self.config, "inference.threshold", 0.5))
        self.model = _build_smp_model(self.config).to(self.device)
        self.model.eval()

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
        self.model.to(self.device)
        self.model.eval()

    def _predict_resized_mask(
        self,
        image_slice: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        image = normalize_slice(image_slice)
        image = resize_slice(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image[None, None, ...].astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)
        mask = (probabilities[0, 0].detach().cpu().numpy() >= self.threshold).astype(np.uint8)
        mask = resize_slice(mask, output_shape, interpolation=cv2.INTER_NEAREST)
        return _as_uint8_mask(mask)

    def predict_volume(self, image_path: str | Path, output_path: str | Path) -> PredictionResult:
        started = time.perf_counter()
        image_path = Path(image_path)
        image, reference = load_nifti_array(image_path)
        output = np.zeros(tuple(int(dim) for dim in image.shape), dtype=np.uint8)
        for slice_index in range(output.shape[self.slice_axis]):
            image_slice = take_slice(image, self.slice_axis, slice_index)
            mask = self._predict_resized_mask(
                image_slice,
                tuple(int(dim) for dim in image_slice.shape),
            )
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
        cv2.imwrite(str(output_path), (mask * 255).astype(np.uint8))
        return {
            "case_id": image_path.stem,
            "output_path": str(output_path),
            "elapsed_sec": time.perf_counter() - started,
            "shape": [int(dim) for dim in image.shape],
        }
