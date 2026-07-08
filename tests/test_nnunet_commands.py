from __future__ import annotations

from pathlib import Path

import yaml

from img_seg.models.nnunet_v2 import plan_and_preprocess, predict, train


def write_command_config(tmp_path: Path) -> Path:
    config = {
        "model_key": "nnunet_v2",
        "dataset": {"id": 777, "name": "Tiny"},
        "paths": {
            "raw_dataset_dir": str(tmp_path / "dataset"),
            "nnunet_raw": str(tmp_path / "nnunet_raw"),
            "nnunet_preprocessed": str(tmp_path / "nnunet_preprocessed"),
            "nnunet_results": str(tmp_path / "nnunet_results"),
            "split_file": str(tmp_path / "split.yaml"),
        },
        "training": {"trainer": "nnUNetTrainer"},
        "inference": {"output_dir": str(tmp_path / "predictions")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_nnunet_command_wrappers_support_dry_run(tmp_path: Path) -> None:
    config_path = write_command_config(tmp_path)

    plan_and_preprocess(config_path, dry_run=True)
    train(config_path, configuration="2d", fold=0, dry_run=True)
    train(config_path, configuration="2d", fold=0, continue_training=True, dry_run=True)
    predict(config_path, configuration="2d", folds="0", dry_run=True)
