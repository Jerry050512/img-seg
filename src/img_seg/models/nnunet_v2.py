"""nnU-Net v2 dataset preparation and command wrappers."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from img_seg.config import deep_get, load_yaml, project_path_from_config, resolve_project_path
from img_seg.data.cases import SegmentationCase, discover_cases
from img_seg.data.splits import CaseSplit, make_case_split
from img_seg.evaluation.metrics import binary_metrics
from img_seg.io.nifti import binarize_mask, load_nifti_array, require_nibabel, save_like
from img_seg.utils.subprocess import run_command

MODEL_KEY = "nnunet_v2"


@dataclass(frozen=True)
class NnUNetPaths:
    raw: Path
    preprocessed: Path
    results: Path

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["nnUNet_raw"] = str(self.raw)
        env["nnUNet_preprocessed"] = str(self.preprocessed)
        env["nnUNet_results"] = str(self.results)
        return env


@dataclass(frozen=True)
class PreparedDataset:
    dataset_dir: Path
    dataset_name: str
    dataset_id: int
    split: CaseSplit
    cases: list[SegmentationCase]


def dataset_folder_name(dataset_id: int, name: str) -> str:
    return f"Dataset{dataset_id:03d}_{name}"


def load_nnunet_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("model_key") != MODEL_KEY:
        raise ValueError(f"Expected model_key={MODEL_KEY!r} in {config_path}")
    return config


def nnunet_paths_from_config(config: dict[str, Any]) -> NnUNetPaths:
    return NnUNetPaths(
        raw=project_path_from_config(config, "paths.nnunet_raw"),
        preprocessed=project_path_from_config(config, "paths.nnunet_preprocessed"),
        results=project_path_from_config(config, "paths.nnunet_results"),
    )


def _case_map(cases: list[SegmentationCase]) -> dict[str, SegmentationCase]:
    return {case.case_id: case for case in cases}


def _copy_or_convert_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.name.endswith(".nii.gz"):
        shutil.copyfile(src, dst)
        return
    data, reference = load_nifti_array(src)
    save_like(reference, np.asanyarray(data), dst)


def _convert_label(src: Path, dst: Path) -> None:
    data, reference = load_nifti_array(src)
    save_like(reference, binarize_mask(data), dst)


def _write_dataset_json(dataset_dir: Path, *, config: dict[str, Any], num_training: int) -> None:
    dataset_json = {
        "channel_names": deep_get(config, "dataset.channel_names", {"0": "scan"}),
        "labels": deep_get(config, "dataset.labels", {"background": 0, "target": 1}),
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_split_json(dataset_dir: Path, split: CaseSplit) -> None:
    payload = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
        "nnunet_train_cases": split.train_val,
    }
    with (dataset_dir / "img_seg_split.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def prepare_nnunet_dataset(config_path: str | Path) -> PreparedDataset:
    """Convert local cases into the nnU-Net v2 raw dataset layout."""

    require_nibabel()
    config = load_nnunet_config(config_path)
    paths = nnunet_paths_from_config(config)
    dataset_id = int(deep_get(config, "dataset.id"))
    dataset_name = str(deep_get(config, "dataset.name"))
    dataset_dir = paths.raw / dataset_folder_name(dataset_id, dataset_name)

    raw_dataset_dir = project_path_from_config(config, "paths.raw_dataset_dir")
    cases = discover_cases(
        raw_dataset_dir,
        mask_overrides=deep_get(config, "dataset.mask_overrides", {}),
        require_masks=True,
    )
    split_config = load_yaml(project_path_from_config(config, "paths.split_file"))
    ratios = split_config.get("ratios", {})
    split = make_case_split(
        [case.case_id for case in cases],
        seed=int(split_config.get("seed", 42)),
        train_ratio=float(ratios.get("train", 0.7)),
        val_ratio=float(ratios.get("val", 0.15)),
        test_ratio=float(ratios.get("test", 0.15)),
    )
    cases_by_id = _case_map(cases)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for directory in (images_tr, labels_tr, images_ts, labels_ts):
        directory.mkdir(parents=True, exist_ok=True)

    for case_id in split.train_val:
        case = cases_by_id[case_id]
        _copy_or_convert_image(case.image_path, images_tr / f"{case_id}_0000.nii.gz")
        if case.mask_path is None:
            raise ValueError(f"Training case has no mask: {case_id}")
        _convert_label(case.mask_path, labels_tr / f"{case_id}.nii.gz")

    for case_id in split.test:
        case = cases_by_id[case_id]
        _copy_or_convert_image(case.image_path, images_ts / f"{case_id}_0000.nii.gz")
        if case.mask_path is not None:
            _convert_label(case.mask_path, labels_ts / f"{case_id}.nii.gz")

    _write_dataset_json(dataset_dir, config=config, num_training=len(split.train_val))
    _write_split_json(dataset_dir, split)
    return PreparedDataset(
        dataset_dir=dataset_dir,
        dataset_name=dataset_folder_name(dataset_id, dataset_name),
        dataset_id=dataset_id,
        split=split,
        cases=cases,
    )


def plan_and_preprocess(
    config_path: str | Path, *, verify: bool = True, dry_run: bool = False
) -> None:
    config = load_nnunet_config(config_path)
    dataset_id = str(deep_get(config, "dataset.id"))
    paths = nnunet_paths_from_config(config)
    command = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id]
    if verify:
        command.append("--verify_dataset_integrity")
    run_command(command, env=paths.env(), dry_run=dry_run)


def train(
    config_path: str | Path,
    *,
    configuration: str,
    fold: str | int,
    dry_run: bool = False,
) -> None:
    config = load_nnunet_config(config_path)
    dataset_id = str(deep_get(config, "dataset.id"))
    trainer = str(deep_get(config, "training.trainer", "nnUNetTrainer"))
    paths = nnunet_paths_from_config(config)
    command = ["nnUNetv2_train", dataset_id, configuration, str(fold), "-tr", trainer]
    run_command(command, env=paths.env(), dry_run=dry_run)


def predict(
    config_path: str | Path,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    configuration: str,
    folds: str = "0",
    dry_run: bool = False,
) -> None:
    config = load_nnunet_config(config_path)
    paths = nnunet_paths_from_config(config)
    dataset_id = int(deep_get(config, "dataset.id"))
    dataset_name = str(deep_get(config, "dataset.name"))
    dataset_dir = paths.raw / dataset_folder_name(dataset_id, dataset_name)
    input_path = Path(input_dir) if input_dir else dataset_dir / "imagesTs"
    output_path = resolve_project_path(output_dir or deep_get(config, "inference.output_dir"))
    command = [
        "nnUNetv2_predict",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-d",
        str(dataset_id),
        "-c",
        configuration,
        "-f",
        folds,
    ]
    run_command(command, env=paths.env(), dry_run=dry_run)


def evaluate_predictions(prediction_dir: str | Path, reference_dir: str | Path) -> dict[str, Any]:
    prediction_dir = Path(prediction_dir)
    reference_dir = Path(reference_dir)
    per_case: dict[str, dict[str, Any]] = {}
    metric_rows = []
    for reference_path in sorted(reference_dir.glob("*.nii.gz")):
        case_id = reference_path.name.removesuffix(".nii.gz")
        prediction_path = prediction_dir / reference_path.name
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction for {case_id}: {prediction_path}")
        prediction, _ = load_nifti_array(prediction_path)
        reference, _ = load_nifti_array(reference_path)
        metrics = binary_metrics(binarize_mask(prediction), binarize_mask(reference))
        row = metrics.as_dict()
        per_case[case_id] = row
        metric_rows.append(row)
    if not metric_rows:
        raise ValueError(f"No reference masks found in {reference_dir}")
    summary = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in ("dice", "iou", "precision", "recall")
    }
    return {"summary": summary, "cases": per_case}
