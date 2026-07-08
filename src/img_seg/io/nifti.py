"""NIfTI IO helpers used by data conversion and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required for NIfTI conversion/evaluation. "
            "Install project dependencies with `uv sync`."
        ) from exc
    return nib


def load_nifti_array(path: str | Path) -> tuple[np.ndarray, object]:
    nib = require_nibabel()
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj), image


def save_like(reference_image: object, data: np.ndarray, output_path: str | Path) -> Path:
    nib = require_nibabel()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_image.header.copy()
    if data.dtype == np.uint8:
        header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(data, reference_image.affine, header=header)
    nib.save(image, str(output_path))
    return output_path


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)
