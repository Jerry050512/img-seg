"""2D slice dataset helpers for lightweight segmentation models."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from img_seg.data.cases import SegmentationCase
from img_seg.io.nifti import binarize_mask, load_nifti_array, require_nibabel

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SliceSample:
    case_id: str
    image_path: Path
    mask_path: Path
    slice_axis: int
    slice_index: int
    original_shape: tuple[int, int, int]
    has_foreground: bool


def axis_to_index(axis: str | int) -> int:
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError(f"slice axis must be 0, 1, or 2: {axis}")
        return axis
    key = axis.lower()
    if key not in AXIS_TO_INDEX:
        raise ValueError(f"slice axis must be one of x/y/z: {axis}")
    return AXIS_TO_INDEX[key]


def take_slice(array: object, axis: int, index: int) -> np.ndarray:
    slicer: list[object] = [slice(None), slice(None), slice(None)]
    slicer[axis] = index
    return np.asanyarray(array[tuple(slicer)])


def normalize_slice(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def resize_slice(
    array: np.ndarray,
    image_size: Sequence[int],
    *,
    interpolation: int,
) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    return cv2.resize(np.asarray(array), (width, height), interpolation=interpolation)


def _keep_sample(rng: random.Random, keep_ratio: float) -> bool:
    if keep_ratio >= 1.0:
        return True
    if keep_ratio <= 0.0:
        return False
    return rng.random() < keep_ratio


def build_slice_samples(
    cases: Sequence[SegmentationCase],
    case_ids: Iterable[str],
    *,
    slice_axis: str | int = "z",
    foreground_slice_keep_ratio: float = 1.0,
    empty_slice_keep_ratio: float = 0.25,
    seed: int = 42,
) -> list[SliceSample]:
    """Build deterministic slice samples while preserving case-level splits."""

    for name, keep_ratio in (
        ("foreground_slice_keep_ratio", foreground_slice_keep_ratio),
        ("empty_slice_keep_ratio", empty_slice_keep_ratio),
    ):
        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1: {keep_ratio}")
    axis = axis_to_index(slice_axis)
    cases_by_id = {case.case_id: case for case in cases}
    samples: list[SliceSample] = []
    projection_axes = tuple(idx for idx in range(3) if idx != axis)
    for case_id in case_ids:
        case = cases_by_id[case_id]
        if case.mask_path is None:
            raise ValueError(f"Case {case_id} has no mask for slice training")

        mask, _ = load_nifti_array(case.mask_path)
        binary_mask = binarize_mask(mask)
        if binary_mask.ndim != 3:
            raise ValueError(
                f"Expected a 3D mask for case {case_id}, got shape {binary_mask.shape}"
            )
        original_shape = tuple(int(dim) for dim in binary_mask.shape)
        foreground_counts = binary_mask.sum(axis=projection_axes)
        rng = random.Random(f"{seed}:{case_id}")

        for slice_index, foreground_count in enumerate(foreground_counts.tolist()):
            has_foreground = int(foreground_count) > 0
            keep_ratio = foreground_slice_keep_ratio if has_foreground else empty_slice_keep_ratio
            if not _keep_sample(rng, keep_ratio):
                continue
            samples.append(
                SliceSample(
                    case_id=case.case_id,
                    image_path=case.image_path,
                    mask_path=case.mask_path,
                    slice_axis=axis,
                    slice_index=int(slice_index),
                    original_shape=original_shape,
                    has_foreground=has_foreground,
                )
            )
    if not samples:
        raise ValueError("No 2D slice samples were selected")
    return samples


class NiftiSliceDataset(Dataset[dict[str, object]]):
    """Torch dataset that reads one 2D NIfTI slice per sample."""

    def __init__(
        self,
        samples: Sequence[SliceSample],
        *,
        image_size: Sequence[int],
        nifti_cache_size: int = 8,
    ) -> None:
        self.samples = list(samples)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if any(size <= 0 for size in self.image_size):
            raise ValueError(f"image_size must contain positive values: {self.image_size}")
        self.nifti_cache_size = int(nifti_cache_size)
        if self.nifti_cache_size < 0:
            raise ValueError(f"nifti_cache_size cannot be negative: {self.nifti_cache_size}")
        self._nifti_cache: OrderedDict[tuple[Path, Path], tuple[object, object]] = OrderedDict()
        require_nibabel()

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict[str, object]:
        """Do not pickle open NIfTI proxies when DataLoader starts worker processes."""

        state = self.__dict__.copy()
        state["_nifti_cache"] = OrderedDict()
        return state

    def _load_nifti_pair(self, sample: SliceSample) -> tuple[object, object]:
        key = (sample.image_path, sample.mask_path)
        cached = self._nifti_cache.get(key)
        if cached is not None:
            self._nifti_cache.move_to_end(key)
            return cached

        nib = require_nibabel()
        image_obj = nib.load(str(sample.image_path), keep_file_open=True)
        mask_obj = nib.load(str(sample.mask_path), keep_file_open=True)
        if tuple(image_obj.shape) != tuple(mask_obj.shape):
            raise ValueError(
                f"Image/mask shape mismatch for case {sample.case_id}: "
                f"{image_obj.shape} vs {mask_obj.shape}"
            )
        if not np.allclose(image_obj.affine, mask_obj.affine):
            raise ValueError(f"Image/mask affine mismatch for case {sample.case_id}")
        pair = (image_obj, mask_obj)
        if self.nifti_cache_size > 0:
            self._nifti_cache[key] = pair
            self._nifti_cache.move_to_end(key)
            while len(self._nifti_cache) > self.nifti_cache_size:
                self._nifti_cache.popitem(last=False)
        return pair

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_obj, mask_obj = self._load_nifti_pair(sample)
        image_slice = take_slice(image_obj.dataobj, sample.slice_axis, sample.slice_index)
        mask_slice = take_slice(mask_obj.dataobj, sample.slice_axis, sample.slice_index)

        image = normalize_slice(image_slice)
        mask = binarize_mask(mask_slice)
        image = resize_slice(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = resize_slice(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = binarize_mask(mask)

        return {
            "image": torch.from_numpy(image[None, ...].astype(np.float32)),
            "mask": torch.from_numpy(mask[None, ...].astype(np.float32)),
            "case_id": sample.case_id,
            "slice_index": sample.slice_index,
            "has_foreground": sample.has_foreground,
        }
