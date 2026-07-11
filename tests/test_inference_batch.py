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
    natural_sort_key,
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


def test_collect_inputs_keeps_scan_with_mask_like_substring(tmp_path: Path) -> None:
    image_path = tmp_path / "segment_scan.nii.gz"
    mask_path = tmp_path / "segment_scan_Seg.nii.gz"
    write_nifti(image_path, np.zeros((2, 3, 4), dtype=np.float32))
    write_nifti(mask_path, np.zeros((2, 3, 4), dtype=np.float32))

    inputs = collect_inputs(tmp_path)

    assert [item.source_path for item in inputs] == [image_path.resolve()]


def test_natural_sort_orders_numbered_cases_human_readably() -> None:
    paths = [Path("Case10.nii.gz"), Path("Case2.nii.gz"), Path("Case1.nii.gz")]

    assert [path.name for path in sorted(paths, key=natural_sort_key)] == [
        "Case1.nii.gz",
        "Case2.nii.gz",
        "Case10.nii.gz",
    ]


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


def test_list_efficientnet_checkpoints_uses_pth_and_prefers_configured_best(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "efficientnet"
    checkpoint_dir.mkdir()
    older_run = checkpoint_dir / "20260710T010000000000Z"
    newer_run = checkpoint_dir / "20260710T020000000000Z"
    older_run.mkdir()
    newer_run.mkdir()
    (older_run / "best.pth").write_bytes(b"older best")
    (older_run / "epoch_10.pth").write_bytes(b"older epoch")
    (newer_run / "best.pth").write_bytes(b"newer best")
    (checkpoint_dir / "legacy_best.pth").write_bytes(b"legacy")
    (checkpoint_dir / "legacy.pt").write_bytes(b"legacy")
    config_path = tmp_path / "efficientnet.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_key": "efficientnet_b0",
                "checkpoints": {
                    "output_dir": str(checkpoint_dir),
                    "best": str(checkpoint_dir / "best.pth"),
                },
            }
        ),
        encoding="utf-8",
    )

    checkpoints = list_checkpoints(model_key="efficientnet_b0", config_path=config_path)

    assert [checkpoint.checkpoint_name for checkpoint in checkpoints] == [
        "best.pth",
        "best.pth",
        "epoch_10.pth",
        "legacy_best.pth",
    ]
    assert [checkpoint.path.parent.name for checkpoint in checkpoints] == [
        newer_run.name,
        older_run.name,
        older_run.name,
        checkpoint_dir.name,
    ]


def test_list_checkpoints_discovers_timestamped_run_dirs(tmp_path: Path) -> None:
    results_dir = tmp_path / "checkpoints" / "nnunet_v2"
    run_fold_dir = (
        tmp_path
        / "checkpoints"
        / "nnunet_v2_runs"
        / "nnunet_v2_2d_fold0_200epochs_20260710_114102"
        / "Dataset777_Tiny"
        / "nnUNetTrainer_200epochs__nnUNetPlans__2d"
        / "fold_0"
    )
    run_fold_dir.mkdir(parents=True)
    (run_fold_dir / "checkpoint_best.pth").write_bytes(b"best")
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

    assert len(checkpoints) == 1
    assert checkpoints[0].path == run_fold_dir / "checkpoint_best.pth"
    assert "20260710_114102" in checkpoints[0].label


def test_prepare_checkpoint_name_rejects_non_checkpoint_file(tmp_path: Path) -> None:
    bad_checkpoint = tmp_path / "checkpoint.txt"
    bad_checkpoint.write_text("not a checkpoint", encoding="utf-8")

    try:
        batch._prepare_checkpoint_name(
            bad_checkpoint,
            config_path=tmp_path / "missing.yaml",
            configuration="2d",
            folds="0",
        )
    except ValueError as exc:
        assert ".pth" in str(exc)
    else:
        raise AssertionError("Expected non-.pth checkpoint to be rejected")


def test_prepare_checkpoint_name_avoids_external_name_collisions(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    configured_fold_dir = (
        results_dir
        / "Dataset777_Tiny"
        / "nnUNetTrainer_200epochs__nnUNetPlans__2d"
        / "fold_0"
    )
    configured_fold_dir.mkdir(parents=True)
    (configured_fold_dir / "checkpoint_best.pth").write_bytes(b"old")
    external_model_dir = (
        tmp_path
        / "runs"
        / "run_1"
        / "Dataset777_Tiny"
        / "nnUNetTrainer_200epochs__nnUNetPlans__2d"
    )
    external_fold_dir = external_model_dir / "fold_0"
    external_fold_dir.mkdir(parents=True)
    external_checkpoint = external_fold_dir / "checkpoint_best.pth"
    external_checkpoint.write_bytes(b"new")
    (external_model_dir / "dataset.json").write_text("{}", encoding="utf-8")
    (external_model_dir / "plans.json").write_text("{}", encoding="utf-8")
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

    checkpoint_name = batch._prepare_checkpoint_name(
        external_checkpoint,
        config_path=config_path,
        configuration="2d",
        folds="0",
    )

    assert checkpoint_name != "checkpoint_best.pth"
    assert checkpoint_name.startswith("checkpoint_best_")
    assert (configured_fold_dir / checkpoint_name).read_bytes() == b"new"
    assert (configured_fold_dir / "checkpoint_best.pth").read_bytes() == b"old"
    assert (configured_fold_dir.parent / "dataset.json").exists()
    assert (configured_fold_dir.parent / "plans.json").exists()


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


def test_run_batch_inference_routes_efficientnet_nifti_and_png(
    monkeypatch,
    tmp_path: Path,
) -> None:
    nifti_path = tmp_path / "scan.nii.gz"
    png_path = tmp_path / "slice.png"
    write_nifti(nifti_path, np.ones((4, 5, 2), dtype=np.float32))
    write_png(png_path, np.ones((6, 7), dtype=np.uint8) * 128)
    inputs = collect_inputs(uploaded_files=[nifti_path, png_path])
    checkpoint_path = tmp_path / "best.pth"
    checkpoint_path.write_bytes(b"fake checkpoint")
    config_path = tmp_path / "efficientnet.yaml"
    calls: list[tuple[str, Path]] = []

    class FakeEfficientNetSegmenter:
        def __init__(
            self,
            *,
            config_path: str | Path,
            device: str | None,
            use_pretrained_encoder: bool,
        ) -> None:
            assert Path(config_path) == config_path_expected
            assert device == "cpu"
            assert use_pretrained_encoder is False

        def load(self, path: str | Path) -> None:
            calls.append(("load", Path(path)))

        def predict_volume(
            self,
            image_path: str | Path,
            output_path: str | Path,
            *,
            check_cancelled=None,
        ) -> dict[str, object]:
            assert check_cancelled is not None
            check_cancelled()
            input_path = Path(image_path)
            final_output = Path(output_path)
            calls.append(("nifti", input_path))
            data, _ = batch.load_nifti_array(input_path)
            write_nifti(final_output, np.ones_like(data, dtype=np.uint8))
            return {
                "output_path": str(final_output),
                "elapsed_sec": 0.1,
                "shape": list(data.shape),
            }

        def predict_image(
            self,
            image_path: str | Path,
            output_path: str | Path,
        ) -> dict[str, object]:
            input_path = Path(image_path)
            final_output = Path(output_path)
            calls.append(("image", input_path))
            image = batch._require_cv2().imread(str(input_path), 0)
            assert image is not None
            write_png(final_output, np.ones(image.shape, dtype=np.uint8) * 255)
            return {
                "output_path": str(final_output),
                "elapsed_sec": 0.2,
                "shape": list(image.shape),
            }

    config_path_expected = config_path
    monkeypatch.setattr(batch, "EfficientNetB0Segmenter", FakeEfficientNetSegmenter)

    results = run_batch_inference(
        model_key="efficientnet_b0",
        inputs=inputs,
        output_dir=tmp_path / "out",
        checkpoint_path_or_name=checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    outputs = {result.case_id: result.output_path for result in results}
    assert calls == [
        ("load", checkpoint_path),
        ("nifti", nifti_path),
        ("image", png_path),
    ]
    assert outputs["scan"].name == "scan_Seg.nii.gz"
    assert outputs["slice"].name == "slice.png"
    assert all(result.status == "done" for result in results)
    assert all(result.output_path.exists() for result in results)


def test_cli_batch_wrapper_raises_when_backend_reports_failed_case(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "case.nii.gz"
    write_nifti(image_path, np.zeros((2, 2, 2), dtype=np.uint8))
    failed = InferenceResult(
        case_id="case",
        input_kind="nifti",
        input_path=image_path,
        output_path=tmp_path / "missing.nii.gz",
        elapsed_sec=0.0,
        shape=[2, 2, 2],
        status="failed",
        message="prediction missing",
    )
    monkeypatch.setattr(batch, "run_batch_inference", lambda **_kwargs: [failed])

    try:
        batch.run_batch_prediction(
            model_key="nnunet_v2",
            checkpoint_path="checkpoint.pth",
            input_dir=image_path,
        )
    except RuntimeError as exc:
        assert "case: prediction missing" in str(exc)
    else:
        raise AssertionError("Expected a failed backend result to fail the CLI wrapper")


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
