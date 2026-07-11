from __future__ import annotations

from pathlib import Path

import yaml

from img_seg.models.nnunet_v2 import NnUNetPaths, plan_and_preprocess, predict, train


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


def test_nnunet_runtime_env_supports_worker_and_external_trainer_paths(tmp_path: Path) -> None:
    paths = NnUNetPaths(
        raw=tmp_path / "raw",
        preprocessed=tmp_path / "preprocessed",
        results=tmp_path / "results",
    )

    env = paths.env(
        {
            "nnunet_n_proc_da": 1,
            "nnunet_def_n_proc": 2,
            "nnunet_npp": 1,
            "nnunet_nps": 1,
            "nnunet_ext_trainer": str(tmp_path / "trainers"),
        }
    )

    assert env["nnUNet_n_proc_DA"] == "1"
    assert env["nnUNet_def_n_proc"] == "2"
    assert env["nnUNet_npp"] == "1"
    assert env["nnUNet_nps"] == "1"
    assert env["nnUNet_extTrainer"] == str(tmp_path / "trainers")


def test_nnunet_predict_supports_low_memory_options(tmp_path: Path, capsys) -> None:
    config_path = write_command_config(tmp_path)

    predict(
        config_path,
        configuration="2d",
        folds="0",
        checkpoint_name="checkpoint_best.pth",
        device="cpu",
        disable_tta=True,
        not_on_device=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        num_parts=2,
        part_id=0,
        dry_run=True,
    )

    command = capsys.readouterr().out
    assert "-device cpu" in command
    assert "--disable_tta" in command
    assert "--not_on_device" in command
    assert "-npp 1" in command
    assert "-nps 1" in command
    assert "-num_parts 2 -part_id 0" in command
