from __future__ import annotations

import numpy as np

from img_seg.evaluation.visualize import build_parser, choose_slice, normalize_image, take_slice


def test_normalize_image_scales_percentile_range() -> None:
    image = np.array([0, 1, 2, 3, 100], dtype=np.float32)

    normalized = normalize_image(image)

    assert normalized.dtype == np.float32
    assert normalized.min() == 0
    assert normalized.max() == 1


def test_normalize_image_handles_constant_input() -> None:
    normalized = normalize_image(np.ones((2, 3), dtype=np.float32))

    assert np.array_equal(normalized, np.zeros((2, 3), dtype=np.float32))


def test_choose_slice_uses_largest_foreground_area() -> None:
    mask = np.zeros((4, 5, 6), dtype=np.uint8)
    mask[1:3, 1:4, 2] = 1
    mask[2, 2, 4] = 1

    assert choose_slice(mask, axis=2) == 2


def test_choose_slice_uses_midpoint_for_empty_mask() -> None:
    mask = np.zeros((4, 5, 6), dtype=np.uint8)

    assert choose_slice(mask, axis=1) == 2


def test_take_slice_selects_requested_axis() -> None:
    array = np.arange(24).reshape(2, 3, 4)

    assert np.array_equal(take_slice(array, axis=1, index=2), array[:, 2, :])


def test_build_parser_accepts_overlay_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--image",
            "image.nii.gz",
            "--reference-mask",
            "reference.nii.gz",
            "--prediction-mask",
            "prediction.nii.gz",
            "--output",
            "overlay.png",
            "--axis",
            "1",
            "--slice-index",
            "3",
        ]
    )

    assert args.image == "image.nii.gz"
    assert args.reference_mask == "reference.nii.gz"
    assert args.prediction_mask == "prediction.nii.gz"
    assert args.output == "overlay.png"
    assert args.axis == 1
    assert args.slice_index == 3
