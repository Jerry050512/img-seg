from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pytest
import torch

from img_seg.data.cases import SegmentationCase, discover_cases
from img_seg.data.slices import NiftiSliceDataset, build_slice_samples
from img_seg.data.splits import CaseSplit, make_case_split
from img_seg.models.efficientnet_b0 import EfficientNetB0Segmenter
from img_seg.training import efficientnet_b0_cli
from img_seg.training.efficientnet_b0_cli import (
    build_parser,
    evaluate_slice_loader,
    resolve_training_artifacts,
    split_cases_from_config,
    timestamped_run_name,
)


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


def test_split_cases_from_config_prefers_explicit_case_lists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_ids = ["case_train", "case_val", "case_test"]
    cases = [
        SegmentationCase(
            case_id=case_id,
            image_path=tmp_path / case_id / "image.nii.gz",
            mask_path=tmp_path / case_id / "mask.nii.gz",
        )
        for case_id in case_ids
    ]
    monkeypatch.setattr(efficientnet_b0_cli, "load_cases_from_manifest", lambda _path: cases)
    monkeypatch.setattr(
        efficientnet_b0_cli,
        "load_yaml",
        lambda _path: {
            "strategy": "case_level",
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "train": ["case_train"],
            "val": ["case_val"],
            "test": ["case_test"],
        },
    )
    config = {
        "data": {
            "dataset_config": str(tmp_path / "dataset.yaml"),
            "split_file": str(tmp_path / "split.yaml"),
        }
    }

    loaded_cases, split = split_cases_from_config(config)

    assert loaded_cases == cases
    assert split == CaseSplit(train=["case_train"], val=["case_val"], test=["case_test"])


def test_efficientnet_segmenter_predicts_nifti_and_image(tmp_path: Path) -> None:
    config = tiny_config()
    segmenter = EfficientNetB0Segmenter(config, device="cpu")
    checkpoint_path = tmp_path / "best.pth"
    torch.save(
        {
            "model_state_dict": segmenter.model.state_dict(),
            "config": config,
            "epoch": 3,
            "best_val_dice": 0.75,
            "run_name": "20260710T083015123456Z",
        },
        checkpoint_path,
    )

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


def test_efficientnet_volume_forward_uses_inference_batch_size(tmp_path: Path) -> None:
    config = tiny_config()
    config["inference"]["batch_size"] = 2
    segmenter = EfficientNetB0Segmenter(config, device="cpu")

    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(int(images.shape[0]))
            return torch.ones(
                (images.shape[0], 1, images.shape[2], images.shape[3]),
                dtype=images.dtype,
                device=images.device,
            )

    counting_model = CountingModel()
    segmenter.model = counting_model
    image_path = tmp_path / "five_slices.nii.gz"
    write_nifti(image_path, np.ones((7, 9, 5), dtype=np.float32))

    output_path = tmp_path / "five_slices_Seg.nii.gz"
    segmenter.predict_volume(image_path, output_path)

    assert counting_model.batch_sizes == [2, 2, 1]
    assert np.asanyarray(nib.load(str(output_path)).dataobj).shape == (7, 9, 5)


def test_efficientnet_rejects_checkpoint_with_incompatible_preprocessing(
    tmp_path: Path,
) -> None:
    config = tiny_config()
    segmenter = EfficientNetB0Segmenter(config, device="cpu")
    saved_config = tiny_config()
    saved_config["data"]["image_size"] = [32, 32]
    checkpoint_path = tmp_path / "incompatible.pth"
    torch.save(
        {
            "model_state_dict": segmenter.model.state_dict(),
            "config": saved_config,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="data.image_size"):
        segmenter.load(checkpoint_path)


def test_efficientnet_volume_checks_cancellation_between_batches(tmp_path: Path) -> None:
    config = tiny_config()
    config["inference"]["batch_size"] = 2
    segmenter = EfficientNetB0Segmenter(config, device="cpu")
    image_path = tmp_path / "cancel.nii.gz"
    write_nifti(image_path, np.ones((4, 4, 5), dtype=np.float32))
    checks = 0

    def check_cancelled() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        segmenter.predict_volume(
            image_path,
            tmp_path / "cancel_Seg.nii.gz",
            check_cancelled=check_cancelled,
        )

    assert checks == 2


def test_nifti_slice_dataset_loads_each_case_file_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "case_a" / "image.nii.gz"
    mask_path = tmp_path / "case_a" / "mask_Seg.nii.gz"
    write_nifti(image_path, np.ones((4, 5, 3), dtype=np.float32))
    write_nifti(mask_path, np.ones((4, 5, 3), dtype=np.uint8))
    cases = discover_cases(tmp_path, require_masks=True)
    samples = build_slice_samples(
        cases,
        ["case_a"],
        foreground_slice_keep_ratio=1.0,
        empty_slice_keep_ratio=1.0,
    )

    real_load = nib.load
    load_calls: list[Path] = []

    def counting_load(filename: str, *args: object, **kwargs: object):
        load_calls.append(Path(filename))
        return real_load(filename, *args, **kwargs)

    monkeypatch.setattr(nib, "load", counting_load)
    dataset = NiftiSliceDataset(samples, image_size=[8, 8])

    for index in range(len(dataset)):
        dataset[index]

    assert load_calls.count(image_path) == 1
    assert load_calls.count(mask_path) == 1
    assert len(load_calls) == 2


def test_evaluate_slice_loader_aggregates_confusion_by_case() -> None:
    loader = [
        {
            "image": torch.tensor([[[[10.0]]], [[[10.0]]], [[[-10.0]]]]),
            "mask": torch.tensor([[[[1.0]]], [[[0.0]]], [[[0.0]]]]),
            "case_id": ["case_a", "case_a", "case_b"],
        }
    ]

    metrics = evaluate_slice_loader(
        torch.nn.Identity(),
        loader,
        device=torch.device("cpu"),
        threshold=0.5,
    )

    assert metrics == pytest.approx(
        {
            "dice": 5 / 6,
            "iou": 3 / 4,
            "precision": 3 / 4,
            "recall": 1.0,
        }
    )


def test_training_writes_pth_checkpoint_and_jsonl_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tiny_config()
    config["training"] = {
        "seed": 42,
        "batch_size": 1,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "loss": "dice_bce",
        "amp": False,
    }
    config["runtime"].update({"num_workers": 0, "nifti_cache_size": 0})
    config["checkpoints"] = {"best": str(tmp_path / "configured" / "best.pth")}

    sample = {
        "image": torch.zeros((1, 4, 4), dtype=torch.float32),
        "mask": torch.ones((1, 4, 4), dtype=torch.float32),
        "case_id": "case_a",
    }

    class FakeSegmenter:
        def __init__(self, _config: dict, *, use_pretrained_encoder: bool) -> None:
            assert use_pretrained_encoder is True
            self.device = torch.device("cpu")
            self.model = torch.nn.Conv2d(1, 1, kernel_size=1)
            self.encoder_initialization = "test"

    monkeypatch.setattr(efficientnet_b0_cli, "load_efficientnet_config", lambda _path: config)
    monkeypatch.setattr(
        efficientnet_b0_cli,
        "split_cases_from_config",
        lambda _config: ([], CaseSplit(train=["case_a"], val=["case_b"], test=[])),
    )
    monkeypatch.setattr(
        efficientnet_b0_cli,
        "dataset_for_split",
        lambda *_args, **_kwargs: [sample],
    )
    monkeypatch.setattr(efficientnet_b0_cli, "EfficientNetB0Segmenter", FakeSegmenter)
    run_name = "20260710T083015123456Z"
    monkeypatch.setattr(efficientnet_b0_cli, "timestamped_run_name", lambda: run_name)
    monkeypatch.setattr(
        efficientnet_b0_cli,
        "evaluate_slice_loader",
        lambda *_args, **_kwargs: {
            "dice": 0.75,
            "iou": 0.6,
            "precision": 0.8,
            "recall": 0.7,
        },
    )

    result = efficientnet_b0_cli.train_model(
        tmp_path / "config.yaml",
        epochs=1,
        output_dir=tmp_path / "run",
    )

    checkpoint_path = Path(result["checkpoint"])
    log_path = Path(result["log_file"])
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert checkpoint_path.name == "best.pth"
    assert checkpoint_path.exists()
    assert checkpoint_path.parent.name == run_name
    assert log_path.name == "train.log"
    assert log_path.parent == checkpoint_path.parent
    assert result["run_name"] == run_name
    assert [record["event"] for record in records] == [
        "run_started",
        "epoch_completed",
        "checkpoint_saved",
        "run_completed",
    ]
    assert {record["run_name"] for record in records} == {run_name}
    hyperparameters = records[0]["hyperparameters"]
    assert hyperparameters["model"]["encoder_name"] == "efficientnet-b0"
    assert hyperparameters["data"]["image_size"] == [64, 64]
    assert hyperparameters["optimization"] == {
        "seed": 42,
        "epochs": 1,
        "batch_size": 1,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "loss": "dice_bce",
        "amp_requested": False,
        "amp_enabled": False,
    }
    assert hyperparameters["runtime"] == {"device": "cpu", "num_workers": 0}
    assert records[1]["dice"] == pytest.approx(0.75)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["run_name"] == run_name


def test_timestamped_run_name_is_sortable_utc_timestamp() -> None:
    timestamp = datetime(2026, 7, 10, 8, 30, 15, 123456, tzinfo=UTC)

    assert timestamped_run_name(timestamp) == "20260710T083015123456Z"


def test_training_rejects_non_pth_checkpoint_path(tmp_path: Path) -> None:
    config = {"checkpoints": {"best": str(tmp_path / "best.pt")}}

    with pytest.raises(ValueError, match=r"\.pth suffix"):
        resolve_training_artifacts(config, run_name="20260710T083015123456Z")


def test_efficientnet_parser_accepts_config_before_or_after_subcommand() -> None:
    parser = build_parser()

    before = parser.parse_args(["--config", "before.yaml", "train"])
    after = parser.parse_args(
        ["train", "--config", "after.yaml", "--log-file", "custom.log"]
    )

    assert (before.command_config or before.config) == "before.yaml"
    assert (after.command_config or after.config) == "after.yaml"
    assert after.log_file == "custom.log"
