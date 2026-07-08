"""Parse nnU-Net training logs and generate report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
TRAIN_LOSS_RE = re.compile(r"train_loss\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)
VAL_LOSS_RE = re.compile(r"val_loss\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)
PSEUDO_DICE_RE = re.compile(r"Pseudo dice\s+\[([^\]]+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float | None = None
    val_loss: float | None = None
    pseudo_dice: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "pseudo_dice": self.pseudo_dice,
        }


def _parse_float_list(text: str) -> list[float]:
    normalized = re.sub(r"np\.float\d+\(([^)]+)\)", r"\1", text)
    return [float(match) for match in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized)]


def parse_training_log(log_path: str | Path) -> list[EpochMetrics]:
    """Parse epoch-level metrics from nnU-Net's plain-text training log."""

    rows: list[EpochMetrics] = []
    current: dict[str, Any] | None = None
    for line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            if current is not None:
                rows.append(EpochMetrics(**current))
            current = {"epoch": int(epoch_match.group(1))}
            continue
        if current is None:
            continue

        if match := TRAIN_LOSS_RE.search(line):
            current["train_loss"] = float(match.group(1))
        elif match := VAL_LOSS_RE.search(line):
            current["val_loss"] = float(match.group(1))
        elif match := PSEUDO_DICE_RE.search(line):
            values = _parse_float_list(match.group(1))
            current["pseudo_dice"] = values[0] if values else None

    if current is not None:
        rows.append(EpochMetrics(**current))
    return rows


def find_latest_training_log(training_dir: str | Path) -> Path:
    logs = sorted(Path(training_dir).glob("training_log*.txt"))
    if not logs:
        raise FileNotFoundError(f"No training_log*.txt found in {training_dir}")
    return logs[-1]


def summarize(rows: list[EpochMetrics]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No epoch metrics parsed")
    complete_rows = [
        row for row in rows if row.train_loss is not None or row.pseudo_dice is not None
    ]
    final = complete_rows[-1] if complete_rows else rows[-1]
    dice_rows = [row for row in rows if row.pseudo_dice is not None]
    best = max(dice_rows, key=lambda row: row.pseudo_dice) if dice_rows else None
    return {
        "epochs_observed": len(rows),
        "epochs_complete": len(complete_rows),
        "last_epoch": rows[-1].epoch,
        "last_complete_epoch": final.epoch if complete_rows else None,
        "best_pseudo_dice": best.pseudo_dice if best else None,
        "best_epoch": best.epoch if best else None,
        "final_train_loss": final.train_loss,
        "final_val_loss": final.val_loss,
        "final_pseudo_dice": final.pseudo_dice,
    }


def write_csv(rows: list[EpochMetrics], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "pseudo_dice"])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    return output_path


def _plot_series(
    rows: list[EpochMetrics], output_path: Path, fields: list[str], ylabel: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row.epoch for row in rows]
    plt.figure(figsize=(8, 5), dpi=150)
    for field in fields:
        values = [getattr(row, field) for row in rows]
        if any(value is not None for value in values):
            plt.plot(epochs, values, marker="o", linewidth=1.5, markersize=3, label=field)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(training_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    log_path = find_latest_training_log(training_dir)
    rows = parse_training_log(log_path)
    summary = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "epoch_metrics.csv")
    (output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "epochs": [row.as_dict() for row in rows]}, indent=2),
        encoding="utf-8",
    )
    _plot_series(rows, output_dir / "loss_curve.png", ["train_loss", "val_loss"], "loss")
    _plot_series(rows, output_dir / "pseudo_dice_curve.png", ["pseudo_dice"], "pseudo Dice")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = generate_report(args.training_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
