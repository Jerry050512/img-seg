from __future__ import annotations

from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch

from img_seg.data.cases import discover_cases
from img_seg.data.slices import NiftiSliceDataset, build_slice_samples
from img_seg.data.splits import make_case_split
from img_seg.models.efficientnet_b0 import EfficientNetB0Segmenter


def write_nifti(path: Path, data: np.ndarray, *, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine=np.eye(4) if affine is None else affine)
    nib.save(image, str(path))


def tiny_config() -> dict:
    return {
        "model_key": "efficientnet_b0",
        "data": {"slice_axis": "z", "image_size": [64, 64]},
        "model": {
            "architecture": "Unet",
            "encoder_name": "efficientnet-b0",
            "encoder_weights": None,
            "in_channels": 1,
            "classes": 1,
        },
        "runtime": {"device": "cpu"},
        "inference": {"threshold": 0.5},
    }


def test_slice_samples_preserve_case_split_and_binarize_masks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for case_id in ("case_a", "case_b", "case_dup"):
        write_nifti(dataset_dir / case_id / "image.nii", np.zeros((4, 4, 3), dtype=np.uint8))
        mask = np.zeros((4, 4, 3), dtype=np.uint16)
        mask[1:3, 1:3, 1] = 65535
        write_nifti(dataset_dir / case_id / "mask_Seg.nii", mask)
    write_nifti(dataset_dir / "case_dup" / "mask_Seg(1).nii", np.ones((4, 4, 3), dtype=np.uint16))

    cases = discover_cases(dataset_dir, mask_overrides={"case_dup": "mask_Seg.nii"})
    split = make_case_split(
        [case.case_id for case in cases],
        seed=1,
        train_ratio=0.34,
        val_ratio=0.33,
        test_ratio=0.33,
    )
    samples = build_slice_samples(
        cases,
        split.train,
        slice_axis="z",
        foreground_slice_keep_ratio=1.0,
        empty_slice_keep_ratio=0.0,
    )
    dataset = NiftiSliceDataset(samples, image_size=[16, 16])
    row = dataset[0]

    assert {sample.case_id for sample in samples}.issubset(set(split.train))
    assert {int(value) for value in torch.unique(row["mask"]).tolist()} <= {0, 1}
    assert row["image"].shape == (1, 16, 16)
    assert row["mask"].shape == (1, 16, 16)


def test_efficientnet_segmenter_predicts_nifti_and_image(tmp_path: Path) -> None:
    config = tiny_config()
    segmenter = EfficientNetB0Segmenter(config, device="cpu")
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": segmenter.model.state_dict()}, checkpoint_path)

    loaded = EfficientNetB0Segmenter(config, device="cpu")
    loaded.load(checkpoint_path)

    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    image_path = tmp_path / "case_a.nii.gz"
    image_data = np.random.default_rng(1).integers(
        0,
        255,
        (8, 8, 2),
        dtype=np.uint8,
    )
    write_nifti(image_path, image_data, affine=affine)

    output_path = tmp_path / "case_a_Seg.nii.gz"
    result = loaded.predict_volume(image_path, output_path)
    saved = nib.load(str(output_path))
    data = np.asanyarray(saved.dataobj)

    assert result["case_id"] == "case_a"
    assert result["shape"] == [8, 8, 2]
    assert data.shape == (8, 8, 2)
    assert np.allclose(saved.affine, affine)
    assert set(np.unique(data).astype(int).tolist()) <= {0, 1}

    png_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(png_path), np.full((12, 10), 120, dtype=np.uint8))
    png_output = tmp_path / "sample.png"
    image_result = loaded.predict_image(png_path, png_output)
    output_png = cv2.imread(str(png_output), cv2.IMREAD_GRAYSCALE)

    assert image_result["case_id"] == "sample"
    assert output_png is not None
    assert output_png.shape == (12, 10)
    assert set(np.unique(output_png).astype(int).tolist()) <= {0, 255}
