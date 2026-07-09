from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from img_seg.inference import batch
from img_seg.inference.batch import (
    InferenceResult,
    collect_inputs,
    collect_reference_masks,
    evaluate_result_metrics,
    list_checkpoints,
    run_batch_inference,
)
from img_seg.io.nifti import require_nibabel


def write_nifti(path: Path, data: np.ndarray) -> None:
    nib = require_nibabel()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def write_png(path: Path, data: np.ndarray) -> None:
    cv2 = batch._require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), data)


def test_collect_inputs_supports_nifti_images_and_unique_case_ids(tmp_path: Path) -> None:
    write_nifti(tmp_path / "Case A_0000.nii.gz", np.zeros((2, 3, 4), dtype=np.float32))
    write_png(tmp_path / "Case A.png", np.zeros((8, 9), dtype=np.uint8))

    inputs = collect_inputs(tmp_path)

    assert [item.kind for item in inputs] == ["image", "nifti"]
    assert sorted(item.case_id for item in inputs) == ["Case_A", "Case_A_2"]


def test_list_checkpoints_prefers_final_then_best(tmp_path: Path) -> None:
    results_dir = tmp_path / "checkpoints"
    fold_dir = (
        results_dir
        / "Dataset777_Tiny"
        / "nnUNetTrainer_200epochs__nnUNetPlans__2d"
        / "fold_0"
    )
    fold_dir.mkdir(parents=True)
    (fold_dir / "checkpoint_best.pth").write_bytes(b"best")
    (fold_dir / "checkpoint_final.pth").write_bytes(b"final")
    config = {
        "model_key": "nnunet_v2",
        "dataset": {"id": 777, "name": "Tiny"},
        "paths": {
            "nnunet_raw": str(tmp_path / "raw"),
            "nnunet_preprocessed": str(tmp_path / "preprocessed"),
            "nnunet_results": str(results_dir),
        },
        "training": {"trainer": "nnUNetTrainer_200epochs"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    checkpoints = list_checkpoints(config_path=config_path)

    assert [checkpoint.checkpoint_name for checkpoint in checkpoints] == [
        "checkpoint_final.pth",
        "checkpoint_best.pth",
    ]


def test_run_batch_inference_exports_nifti_and_png(monkeypatch, tmp_path: Path) -> None:
    nifti_path = tmp_path / "scan.nii"
    png_path = tmp_path / "slice.png"
    write_nifti(nifti_path, np.ones((4, 5, 2), dtype=np.float32))
    write_png(png_path, np.ones((6, 7), dtype=np.uint8) * 128)
    inputs = collect_inputs(uploaded_files=[nifti_path, png_path])

    def fake_nnunet_predict(
        config_path,
        *,
        input_dir,
        output_dir,
        configuration,
        folds,
        checkpoint_name=None,
        cancel_event=None,
    ):
        del config_path, configuration, folds, checkpoint_name, cancel_event
        for image_path in Path(input_dir).glob("*_0000.nii.gz"):
            case_id = image_path.name.removesuffix("_0000.nii.gz")
            data, _ = batch.load_nifti_array(image_path)
            prediction = (np.asarray(data) > 0).astype(np.uint8)
            write_nifti(Path(output_dir) / f"{case_id}.nii.gz", prediction)

    monkeypatch.setattr(batch, "_run_nnunet_predict", fake_nnunet_predict)

    results = run_batch_inference(
        model_key="nnunet_v2",
        inputs=inputs,
        output_dir=tmp_path / "out",
        checkpoint_path_or_name="checkpoint_final.pth",
    )

    outputs = {result.case_id: result.output_path for result in results}
    assert outputs["scan"].name == "scan.nii.gz"
    assert outputs["slice"].name == "slice.png"
    assert outputs["scan"].exists()
    assert outputs["slice"].exists()


def test_evaluate_result_metrics_uses_reference_when_available(tmp_path: Path) -> None:
    prediction_path = tmp_path / "case.nii.gz"
    reference_path = tmp_path / "case_Seg.nii.gz"
    write_nifti(prediction_path, np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8))
    write_nifti(reference_path, np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8))
    result = InferenceResult(
        case_id="case",
        input_kind="nifti",
        input_path=tmp_path / "case_input.nii.gz",
        output_path=prediction_path,
        elapsed_sec=1.0,
        shape=[2, 3],
        status="done",
    )

    rows = evaluate_result_metrics([result], {"case": reference_path})

    assert rows[0].foreground == 2
    assert rows[0].foreground_ratio == 2 / 6
    assert rows[0].dice == 0.5
    assert rows[0].iou == 1 / 3


def test_evaluate_result_metrics_reports_statistics_without_reference(tmp_path: Path) -> None:
    prediction_path = tmp_path / "case.png"
    write_png(prediction_path, np.array([[255, 0], [0, 0]], dtype=np.uint8))
    result = InferenceResult(
        case_id="case",
        input_kind="image",
        input_path=tmp_path / "case.png",
        output_path=prediction_path,
        elapsed_sec=1.0,
        shape=[2, 2],
        status="done",
    )

    rows = evaluate_result_metrics([result])

    assert rows[0].foreground == 1
    assert rows[0].total == 4
    assert rows[0].dice is None
    assert "No reference" in rows[0].message


def test_collect_reference_masks_normalizes_labels_and_deduplicates(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "Case A_mask_0000.nii.gz"
    write_nifti(reference_path, np.zeros((2, 2), dtype=np.uint8))

    references = collect_reference_masks(tmp_path, [reference_path])

    assert references == {"Case_A": reference_path}
