"""Shared model adapter interface for all segmentation backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict


class PredictionResult(TypedDict):
    case_id: str
    output_path: str
    elapsed_sec: float
    shape: list[int]


class Segmenter(Protocol):
    name: str

    def load(self, checkpoint_path: str | Path) -> None:
        """Load model weights or backend-specific checkpoint metadata."""

    def predict_volume(self, image_path: str | Path, output_path: str | Path) -> PredictionResult:
        """Predict a binary segmentation mask for one volume."""


class ImageSegmenter(Protocol):
    def predict_image(self, image_path: str | Path, output_path: str | Path) -> PredictionResult:
        """Predict a binary segmentation mask for one 2D image."""
