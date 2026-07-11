from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from img_seg.models.monai_segresnet import (
    MonaiSegResNetSegmenter,
    build_model,
    load_monai_config,
    normalize_image,
)
from img_seg.training.monai_segresnet import (
    _write_json_atomic,
    evaluate_split,
    load_data,
    predict_split,
)


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine if affine is not None else np.eye(4)), str(path))


def write_tiny_config(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    case_ids = ["case_train", "case_val", "case_test"]
    for index, case_id in enumerate(case_ids):
        image = np.zeros((8, 8, 8), dtype=np.float32)
        image[2:6, 2:6, 2:6] = index + 1
        mask = (image > 0).astype(np.uint8)
        write_nifti(dataset_dir / case_id / "image.nii.gz", image)
        write_nifti(dataset_dir / case_id / "mask.nii.gz", mask)

    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "root": str(dataset_dir),
                "layout": "case_dir_pair",
                "image_name": "image.nii.gz",
                "mask_name": "mask.nii.gz",
                "case_id_policy": "directory_name",
            }
        ),
        encoding="utf-8",
    )
    split_config = tmp_path / "split.yaml"
    split_config.write_text(
        yaml.safe_dump(
            {"train": ["case_train"], "val": ["case_val"], "test": ["case_test"]}
        ),
        encoding="utf-8",
    )
    config = {
        "model_key": "monai_segresnet",
        "model_name": "Tiny SegResNet",
        "data": {
            "dataset_config": str(dataset_config),
            "split_file": str(split_config),
            "roi_size": [8, 8, 8],
            "samples_per_volume": 1,
            "cache_rate": 0.0,
            "normalize_nonzero": True,
        },
        "model": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "init_filters": 2,
            "blocks_down": [1, 1],
            "blocks_up": [1],
            "dropout_prob": 0.0,
            "norm": "INSTANCE",
        },
        "training": {"amp": False},
        "inference": {
            "sliding_window_batch_size": 1,
            "overlap": 0.25,
            "threshold": 0.5,
            "output_dir": str(tmp_path / "predictions"),
        },
        "runtime": {"device": "cpu"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_load_monai_config_and_explicit_case_split(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path)

    config = load_monai_config(config_path)
    data = load_data(config)

    assert data.split.train == ["case_train"]
    assert data.split.val == ["case_val"]
    assert data.split.test == ["case_test"]


def test_load_monai_config_rejects_mismatched_architecture(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["blocks_up"] = [1, 1]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one fewer"):
        load_monai_config(config_path)


def test_load_data_rejects_training_volume_smaller_than_roi(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["roi_size"] = [9, 8, 8]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="smaller than data.roi_size"):
        load_data(load_monai_config(config_path))


def test_normalize_image_uses_monai_transform() -> None:
    pytest.importorskip("monai")
    image = np.zeros((4, 4, 4), dtype=np.float32)
    image[1:3, 1:3, 1:3] = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 1

    normalized = normalize_image(image, nonzero=True)

    assert normalized.dtype == np.float32
    assert np.all(normalized[image == 0] == 0)
    assert np.isclose(normalized[image != 0].mean(), 0.0, atol=1e-6)
    assert np.isclose(normalized[image != 0].std(), 1.0, atol=1e-6)


def test_write_json_atomic_replaces_target(tmp_path: Path) -> None:
    target = tmp_path / "history.json"
    target.write_text("old", encoding="utf-8")

    _write_json_atomic(target, [{"epoch": 1}])

    assert json.loads(target.read_text(encoding="utf-8")) == [{"epoch": 1}]
    assert not target.with_suffix(".json.tmp").exists()


def test_monai_segmenter_loads_checkpoint_and_preserves_nifti_geometry(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    config_path = write_tiny_config(tmp_path)
    config = load_monai_config(config_path)
    model = build_model(config)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "epoch": 1, "best_dice": 0.5}, checkpoint)
    affine = np.diag([0.5, 0.75, 2.0, 1.0])
    image = np.zeros((8, 8, 8), dtype=np.float32)
    image[2:6, 2:6, 2:6] = 1
    input_path = tmp_path / "case_direct" / "image.nii.gz"
    output_path = tmp_path / "output.nii.gz"
    write_nifti(input_path, image, affine)

    segmenter = MonaiSegResNetSegmenter(config_path, device="cpu")
    segmenter.load(checkpoint)
    result = segmenter.predict_volume(input_path, output_path)

    output = nib.load(str(output_path))
    values = np.asanyarray(output.dataobj)
    assert result["case_id"] == "case_direct"
    assert result["shape"] == [8, 8, 8]
    assert output.shape == (8, 8, 8)
    assert np.allclose(output.affine, affine)
    assert set(np.unique(values).tolist()).issubset({0, 1})


def test_evaluate_split_reports_perfect_binary_prediction(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path)
    prediction_dir = tmp_path / "predictions"
    reference = nib.load(str(tmp_path / "dataset" / "case_test" / "mask.nii.gz"))
    write_nifti(
        prediction_dir / "case_test.nii.gz",
        np.asanyarray(reference.dataobj),
        reference.affine,
    )

    result = evaluate_split(config_path, prediction_dir, split_name="test")

    assert result["summary"]["dice"] == 1.0
    assert result["summary"]["iou"] == 1.0
    assert result["summary"]["precision"] == 1.0
    assert result["summary"]["recall"] == 1.0


def test_predict_split_uses_dataset_case_id(monkeypatch, tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path)

    class FakeSegmenter:
        def __init__(self, config_path, device=None):
            del config_path, device

        def load(self, checkpoint_path):
            del checkpoint_path

        def predict_volume(self, image_path, output_path):
            del image_path
            return {
                "case_id": "image",
                "output_path": str(output_path),
                "elapsed_sec": 0.1,
                "shape": [8, 8, 8],
            }

    monkeypatch.setattr(
        "img_seg.training.monai_segresnet.MonaiSegResNetSegmenter", FakeSegmenter
    )

    manifest = predict_split(config_path, "unused.pt", split_name="test", device_name="cpu")

    assert manifest["results"][0]["case_id"] == "case_test"
