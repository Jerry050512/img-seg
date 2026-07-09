"""Generate NIfTI segmentation overlay images."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from img_seg.io.nifti import binarize_mask, load_nifti_array


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def choose_slice(mask: np.ndarray, axis: int) -> int:
    foreground = np.asarray(mask).astype(bool)
    if not foreground.any():
        return foreground.shape[axis] // 2
    projection_axes = tuple(idx for idx in range(foreground.ndim) if idx != axis)
    counts = foreground.sum(axis=projection_axes)
    return int(np.argmax(counts))


def take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, indices=index, axis=axis)


def save_overlay(
    image_path: str | Path,
    reference_mask_path: str | Path,
    output_path: str | Path,
    *,
    prediction_mask_path: str | Path | None = None,
    axis: int = 2,
    slice_index: int | None = None,
) -> Path:
    image, _ = load_nifti_array(image_path)
    reference, _ = load_nifti_array(reference_mask_path)
    prediction = None
    if prediction_mask_path:
        prediction, _ = load_nifti_array(prediction_mask_path)

    ref_mask = binarize_mask(reference)
    pred_mask = binarize_mask(prediction) if prediction is not None else None
    mask_for_slice = ref_mask if pred_mask is None else np.logical_or(ref_mask, pred_mask)
    index = choose_slice(mask_for_slice, axis)
    if slice_index is not None:
        index = slice_index

    image_slice = normalize_image(take_slice(image, axis, index))
    ref_slice = take_slice(ref_mask, axis, index)
    pred_slice = take_slice(pred_mask, axis, index) if pred_mask is not None else None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7), dpi=150)
    plt.imshow(np.rot90(image_slice), cmap="gray")
    plt.contour(np.rot90(ref_slice), levels=[0.5], colors=["lime"], linewidths=1.0)
    if pred_slice is not None:
        plt.contour(np.rot90(pred_slice), levels=[0.5], colors=["red"], linewidths=1.0)
    plt.title(f"slice={index}, green=reference, red=prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--prediction-mask")
    parser.add_argument("--output", required=True)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--slice-index", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = save_overlay(
        args.image,
        args.reference_mask,
        args.output,
        prediction_mask_path=args.prediction_mask,
        axis=args.axis,
        slice_index=args.slice_index,
    )
    print(output)


if __name__ == "__main__":
    main()
