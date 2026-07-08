from __future__ import annotations

import json
from pathlib import Path

from img_seg.evaluation.nnunet_report import generate_report, parse_training_log


def test_parse_training_log_and_generate_report(tmp_path: Path) -> None:
    training_dir = tmp_path / "fold_0"
    training_dir.mkdir()
    log_path = training_dir / "training_log_2026_07_08.txt"
    log_path.write_text(
        "\n".join(
            [
                "Epoch 0",
                "train_loss -0.1000",
                "val_loss -0.0500",
                "Pseudo dice [np.float32(0.2500)]",
                "Epoch 1",
                "train_loss -0.2000",
                "val_loss -0.1500",
                "Pseudo dice [0.5000]",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_training_log(log_path)
    assert len(rows) == 2
    assert rows[-1].epoch == 1
    assert rows[-1].pseudo_dice == 0.5

    output_dir = tmp_path / "report"
    summary = generate_report(training_dir, output_dir)

    assert summary["best_pseudo_dice"] == 0.5
    assert summary["last_complete_epoch"] == 1
    assert (output_dir / "epoch_metrics.csv").exists()
    assert (output_dir / "loss_curve.png").exists()
    assert (output_dir / "pseudo_dice_curve.png").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["summary"]["best_epoch"] == 1
