"""MONAI SegResNet model factory and unified NIfTI inference adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from img_seg.config import deep_get, load_yaml, resolve_project_path
from img_seg.io.nifti import load_nifti_array, save_like
from img_seg.models.base import PredictionResult

MODEL_KEY = "monai_segresnet"
DEFAULT_CONFIG_PATH = Path("configs/monai_segresnet/base.yaml")


def _require_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from monai.inferers import sliding_window_inference
        from monai.networks.nets import SegResNet
        from monai.transforms import NormalizeIntensity
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and MONAI are required for SegResNet. "
            "Install project dependencies with `uv sync`."
        ) from exc
    return torch, sliding_window_inference, SegResNet, NormalizeIntensity


def load_monai_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(resolve_project_path(config_path))
    if config.get("model_key") != MODEL_KEY:
        raise ValueError(f"Expected model_key={MODEL_KEY!r} in {config_path}")
    roi_size = deep_get(config, "data.roi_size")
    if not isinstance(roi_size, list) or len(roi_size) != 3 or any(int(v) <= 0 for v in roi_size):
        raise ValueError("data.roi_size must contain three positive integers")
    if int(deep_get(config, "model.spatial_dims", 3)) != 3:
        raise ValueError("MONAI SegResNet workflow requires model.spatial_dims=3")
    if int(deep_get(config, "model.in_channels", 1)) != 1:
        raise ValueError("Current NIfTI workflow requires model.in_channels=1")
    if int(deep_get(config, "model.out_channels", 1)) != 1:
        raise ValueError("Binary segmentation requires model.out_channels=1")
    init_filters = int(deep_get(config, "model.init_filters", 16))
    if init_filters <= 0:
        raise ValueError("model.init_filters must be positive")
    blocks_down = deep_get(config, "model.blocks_down", [1, 2, 2, 4])
    blocks_up = deep_get(config, "model.blocks_up", [1, 1, 1])
    if not isinstance(blocks_down, (list, tuple)) or not blocks_down:
        raise ValueError("model.blocks_down must contain positive integers")
    if not isinstance(blocks_up, (list, tuple)):
        raise ValueError("model.blocks_up must contain positive integers")
    if any(int(value) <= 0 for value in (*blocks_down, *blocks_up)):
        raise ValueError("model.blocks_down and model.blocks_up must be positive")
    if len(blocks_up) != len(blocks_down) - 1:
        raise ValueError(
            "model.blocks_up must have exactly one fewer level than model.blocks_down"
        )
    return config


def resolve_device(requested: str = "auto") -> Any:
    torch, _, _, _ = _require_runtime()
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    return device


def build_model(config: dict[str, Any]) -> Any:
    _, _, segresnet_class, _ = _require_runtime()
    model_config = config.get("model", {})
    return segresnet_class(
        spatial_dims=int(model_config.get("spatial_dims", 3)),
        in_channels=int(model_config.get("in_channels", 1)),
        out_channels=int(model_config.get("out_channels", 1)),
        init_filters=int(model_config.get("init_filters", 16)),
        blocks_down=tuple(int(v) for v in model_config.get("blocks_down", [1, 2, 2, 4])),
        blocks_up=tuple(int(v) for v in model_config.get("blocks_up", [1, 1, 1])),
        dropout_prob=float(model_config.get("dropout_prob", 0.0)),
        norm=str(model_config.get("norm", "INSTANCE")),
    )


def normalize_image(image: np.ndarray, *, nonzero: bool) -> np.ndarray:
    """Apply the same MONAI intensity transform used by the training pipeline."""

    _, _, _, normalize_intensity = _require_runtime()
    transform = normalize_intensity(nonzero=nonzero)
    return np.asarray(transform(np.asarray(image, dtype=np.float32)), dtype=np.float32)


def checkpoint_payload(checkpoint_path: str | Path, device: Any) -> dict[str, Any]:
    torch, _, _, _ = _require_runtime()
    path = resolve_project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"SegResNet checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid SegResNet checkpoint: {path}")
    return payload


class MonaiSegResNetSegmenter:
    """Adapter exposing the shared ``load`` and ``predict_volume`` interface."""

    name = "MONAI SegResNet 3D"

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        device: str | None = None,
    ) -> None:
        self.config_path = resolve_project_path(config_path)
        self.config = load_monai_config(self.config_path)
        requested_device = device or str(deep_get(self.config, "runtime.device", "auto"))
        self.device = resolve_device(requested_device)
        self.model = build_model(self.config).to(self.device)
        self.model.eval()
        self.checkpoint_path: Path | None = None

    def load(self, checkpoint_path: str | Path) -> None:
        path = resolve_project_path(checkpoint_path)
        payload = checkpoint_payload(path, self.device)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.eval()
        self.checkpoint_path = path

    def _predict_array(self, image: np.ndarray) -> np.ndarray:
        if self.checkpoint_path is None:
            raise RuntimeError("Load a SegResNet checkpoint before prediction")
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume, got shape {image.shape}")

        torch, sliding_window_inference, _, _ = _require_runtime()
        normalized = normalize_image(
            image,
            nonzero=bool(deep_get(self.config, "data.normalize_nonzero", True)),
        )
        inputs = torch.from_numpy(normalized[None, None])
        roi_size = tuple(int(v) for v in deep_get(self.config, "data.roi_size"))
        sw_batch_size = int(deep_get(self.config, "inference.sliding_window_batch_size", 1))
        overlap = float(deep_get(self.config, "inference.overlap", 0.5))
        threshold = float(deep_get(self.config, "inference.threshold", 0.5))
        use_amp = bool(deep_get(self.config, "training.amp", True)) and self.device.type == "cuda"

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                logits = sliding_window_inference(
                    inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=self.model,
                    overlap=overlap,
                    mode="gaussian",
                    sw_device=self.device,
                    device=torch.device("cpu"),
                )
            prediction = (torch.sigmoid(logits) >= threshold).to(torch.uint8)
        return prediction[0, 0].numpy()

    def predict_volume(
        self,
        image_path: str | Path,
        output_path: str | Path,
    ) -> PredictionResult:
        image_path = resolve_project_path(image_path)
        output_path = resolve_project_path(output_path)
        image, reference = load_nifti_array(image_path)
        torch, _, _, _ = _require_runtime()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prediction = self._predict_array(np.asarray(image))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        save_like(reference, prediction, output_path)
        case_id = image_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        if case_id.lower() in {"image", "volume"}:
            case_id = image_path.parent.name
        return {
            "case_id": case_id,
            "output_path": str(output_path),
            "elapsed_sec": elapsed,
            "shape": list(prediction.shape),
        }
