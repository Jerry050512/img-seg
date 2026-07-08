"""Segmentation metrics shared by all model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def safe_divide(numerator: float, denominator: float, empty_value: float = 1.0) -> float:
    if denominator == 0:
        return empty_value
    return numerator / denominator


def binary_confusion(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    pred = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape}, target {truth.shape}")

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())
    return tp, fp, fn, tn


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> BinaryMetrics:
    tp, fp, fn, tn = binary_confusion(prediction, target)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return BinaryMetrics(
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )
