from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from img_seg.models.nnunet_v2 import (
    dataset_folder_name,
    evaluate_predictions,
    prepare_nnunet_dataset,
)


def write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, str(path))


def write_config(tmp_path: Path, dataset_dir: Path, split_file: Path) -> Path:
    config = {
        "model_key": "nnunet_v2",
        "dataset": {
            "id": 777,
            "name": "Tiny",
            "channel_names": {"0": "scan"},
            "labels": {"background": 0, "target": 1},
            "mask_overrides": {"case_dup": "mask_Seg.nii"},
        },
        "paths": {
            "raw_dataset_dir": str(dataset_dir),
            "nnunet_raw": str(tmp_path / "nnunet_raw"),
            "nnunet_preprocessed": str(tmp_path / "nnunet_preprocessed"),
            "nnunet_results": str(tmp_path / "nnunet_results"),
            "split_file": str(split_file),
        },
        "training": {"trainer": "nnUNetTrainer"},
        "inference": {"output_dir": str(tmp_path / "predictions")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_prepare_nnunet_dataset_and_evaluate(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for case_id in ("case_a", "case_b", "case_dup"):
        write_nifti(dataset_dir / case_id / "image.nii", np.zeros((4, 4, 3), dtype=np.uint8))
        mask = np.zeros((4, 4, 3), dtype=np.uint16)
        mask[1:3, 1:3, 1] = 65535
        write_nifti(dataset_dir / case_id / "mask_Seg.nii", mask)
    write_nifti(dataset_dir / "case_dup" / "mask_Seg(1).nii", np.ones((4, 4, 3), dtype=np.uint16))

    split_file = tmp_path / "split.yaml"
    split_file.write_text(
        yaml.safe_dump(
            {"seed": 1, "ratios": {"train": 0.34, "val": 0.33, "test": 0.33}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = write_config(tmp_path, dataset_dir, split_file)

    prepared = prepare_nnunet_dataset(config_path)

    dataset_dir_out = tmp_path / "nnunet_raw" / dataset_folder_name(777, "Tiny")
    assert prepared.dataset_dir == dataset_dir_out
    dataset_json = json.loads((dataset_dir_out / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["numTraining"] == 2
    assert len(list((dataset_dir_out / "imagesTr").glob("*_0000.nii.gz"))) == 2
    assert len(list((dataset_dir_out / "labelsTr").glob("*.nii.gz"))) == 2
    assert len(list((dataset_dir_out / "imagesTs").glob("*_0000.nii.gz"))) == 1
    assert len(list((dataset_dir_out / "labelsTs").glob("*.nii.gz"))) == 1
    splits_final = tmp_path / "nnunet_preprocessed" / dataset_folder_name(777, "Tiny")
    splits = json.loads((splits_final / "splits_final.json").read_text(encoding="utf-8"))
    assert splits == [{"train": prepared.split.train, "val": prepared.split.val}]

    label_path = next((dataset_dir_out / "labelsTr").glob("*.nii.gz"))
    labels = np.asanyarray(nib.load(str(label_path)).dataobj)
    assert set(np.unique(labels).astype(int).tolist()) == {0, 1}

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    reference_path = next((dataset_dir_out / "labelsTs").glob("*.nii.gz"))
    prediction_path = prediction_dir / reference_path.name
    prediction_path.write_bytes(reference_path.read_bytes())

    result = evaluate_predictions(prediction_dir, dataset_dir_out / "labelsTs")
    assert result["summary"]["dice"] == 1.0
    assert result["summary"]["iou"] == 1.0
