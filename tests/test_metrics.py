from __future__ import annotations

import numpy as np
import pytest

from img_seg.evaluation.metrics import binary_confusion, binary_metrics


def test_binary_metrics() -> None:
    prediction = np.array([[1, 1, 0], [0, 0, 0]])
    target = np.array([[1, 0, 1], [0, 0, 0]])

    metrics = binary_metrics(prediction, target)

    assert binary_confusion(prediction, target) == (1, 1, 1, 3)
    assert metrics.dice == pytest.approx(0.5)
    assert metrics.iou == pytest.approx(1 / 3)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)


def test_binary_metrics_empty_masks_are_perfect() -> None:
    prediction = np.zeros((2, 2), dtype=np.uint8)
    target = np.zeros((2, 2), dtype=np.uint8)

    metrics = binary_metrics(prediction, target)

    assert metrics.dice == 1.0
    assert metrics.iou == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
