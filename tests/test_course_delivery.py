from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from img_seg.course_delivery import (
    checkpoint_parameter_count,
    config_with_dataset,
    evaluate_predictions,
    normalize_legacy_argv,
)
from img_seg.inference.batch import InferenceResult
from img_seg.io.nifti import require_nibabel


def write_nifti(path: Path, data: np.ndarray) -> None:
    nib = require_nibabel()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_normalize_legacy_argv_accepts_handout_spelling() -> None:
    assert normalize_legacy_argv(["image", "scan.nii.gz", "weight", "best.pth"]) == [
        "--image",
        "scan.nii.gz",
        "--weight",
        "best.pth",
    ]


def test_segresnet_course_config_overrides_dataset_and_artifact_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump({"model_key": "monai_segresnet", "data": {}, "training": {}}),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "artifacts"

    with config_with_dataset("monai_segresnet", source, dataset, output) as generated:
        config = yaml.safe_load(generated.read_text(encoding="utf-8"))
        manifest = yaml.safe_load(Path(config["data"]["dataset_config"]).read_text())

    assert manifest["root"] == str(dataset)
    assert config["training"]["checkpoint_dir"] == str(output)
    assert config["training"]["history_file"] == str(output / "history.json")


def test_course_evaluation_reports_required_metrics(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_1"
    image = case_dir / "image.nii.gz"
    reference = case_dir / "mask.nii.gz"
    prediction = tmp_path / "predictions" / "case_1_Seg.nii.gz"
    write_nifti(image, np.zeros((2, 2), dtype=np.float32))
    write_nifti(reference, np.array([[1, 0], [0, 0]], dtype=np.uint8))
    write_nifti(prediction, np.array([[1, 1], [0, 0]], dtype=np.uint8))
    checkpoint = tmp_path / "best.pth"
    torch.save(
        {
            "model_state_dict": {
                "weight": torch.ones(10),
                "norm.running_mean": torch.ones(4),
            }
        },
        checkpoint,
    )
    result = InferenceResult(
        case_id="case_1",
        input_kind="nifti",
        input_path=image,
        output_path=prediction,
        elapsed_sec=2.0,
        shape=[2, 2],
        status="done",
    )

    metrics = evaluate_predictions([result], checkpoint=checkpoint, reference_path=None)

    assert set(metrics) == {
        "mIoU",
        "Dice",
        "precision",
        "recall",
        "PixelAcc",
        "FPS",
        "Params(M)",
    }
    assert metrics["mIoU"] == (0.5 + 2 / 3) / 2
    assert metrics["FPS"] == 0.5
    assert metrics["Params(M)"] == 0.00001


def test_checkpoint_parameter_count_deduplicates_aliased_state_dict_keys(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "nnunet.pth"
    weight = torch.ones(3, 4)
    torch.save(
        {"network_weights": {"layer.weight": weight, "layer.alias": weight}},
        checkpoint,
    )

    assert checkpoint_parameter_count(checkpoint) == 12
