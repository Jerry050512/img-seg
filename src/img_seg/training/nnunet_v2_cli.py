"""Command line interface for the nnU-Net v2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from img_seg.models.nnunet_v2 import (
    evaluate_predictions,
    plan_and_preprocess,
    predict,
    prepare_nnunet_dataset,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/nnunet_v2/base.yaml",
        help="Path to the nnU-Net v2 config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Convert local dataset to nnU-Net raw layout.")

    plan_parser = subparsers.add_parser("plan", help="Run nnU-Net planning/preprocessing.")
    plan_parser.add_argument("--no-verify", action="store_true")
    plan_parser.add_argument("--dry-run", action="store_true")

    train_parser = subparsers.add_parser("train", help="Run nnU-Net training.")
    train_parser.add_argument("--configuration", default="2d")
    train_parser.add_argument("--fold", default="0")
    train_parser.add_argument("--continue-training", action="store_true")
    train_parser.add_argument("--dry-run", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Run nnU-Net prediction.")
    predict_parser.add_argument("--input-dir")
    predict_parser.add_argument("--output-dir")
    predict_parser.add_argument("--configuration", default="2d")
    predict_parser.add_argument("--folds", default="0")
    predict_parser.add_argument("--dry-run", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against labels.")
    eval_parser.add_argument("--prediction-dir", required=True)
    eval_parser.add_argument("--reference-dir", required=True)
    eval_parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)

    if args.command == "prepare":
        prepared = prepare_nnunet_dataset(config_path)
        print(f"Prepared {prepared.dataset_name} at {prepared.dataset_dir}")
        print(
            "Split: "
            f"train={len(prepared.split.train)}, "
            f"val={len(prepared.split.val)}, "
            f"test={len(prepared.split.test)}"
        )
    elif args.command == "plan":
        plan_and_preprocess(config_path, verify=not args.no_verify, dry_run=args.dry_run)
    elif args.command == "train":
        train(
            config_path,
            configuration=args.configuration,
            fold=args.fold,
            continue_training=args.continue_training,
            dry_run=args.dry_run,
        )
    elif args.command == "predict":
        predict(
            config_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            configuration=args.configuration,
            folds=args.folds,
            dry_run=args.dry_run,
        )
    elif args.command == "evaluate":
        result = evaluate_predictions(args.prediction_dir, args.reference_dir)
        text = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
        print(text)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
