"""Command line workflow for the 3D MONAI SegResNet model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from img_seg.models.monai_segresnet import build_model, load_monai_config
from img_seg.training.monai_segresnet import evaluate_split, load_data, predict_split, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/monai_segresnet/base.yaml",
        help="Path to the MONAI SegResNet config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect", help="Show dataset split and model parameter count.")

    train_parser = subparsers.add_parser("train", help="Train or resume SegResNet.")
    train_parser.add_argument("--resume", help="Checkpoint used to resume training.")
    train_parser.add_argument("--epochs", type=int, help="Override total epoch count.")
    train_parser.add_argument("--device", help="Override runtime device, e.g. cuda or cpu.")
    train_parser.add_argument(
        "--cache-rate",
        type=float,
        help="Override MONAI cache rate (0 to 1); requires enough system memory.",
    )
    train_parser.add_argument(
        "--sw-batch-size",
        type=int,
        help="Override validation sliding-window batch size.",
    )
    train_parser.add_argument("--output-json", help="Write the training summary as JSON.")

    predict_parser = subparsers.add_parser("predict", help="Predict one case-level split.")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    predict_parser.add_argument("--output-dir")
    predict_parser.add_argument("--device")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate saved predictions.")
    evaluate_parser.add_argument("--prediction-dir", required=True)
    evaluate_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate_parser.add_argument("--output-json")
    return parser


def _print_json(payload: dict, output_json: str | None = None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if args.command == "inspect":
        config = load_monai_config(config_path)
        data = load_data(config)
        model = build_model(config)
        _print_json(
            {
                "model": config["model_name"],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "roi_size": config["data"]["roi_size"],
                "split": {
                    "train": data.split.train,
                    "val": data.split.val,
                    "test": data.split.test,
                },
            }
        )
    elif args.command == "train":
        _print_json(
            train(
                config_path,
                resume=args.resume,
                epochs=args.epochs,
                device_name=args.device,
                cache_rate=args.cache_rate,
                sw_batch_size=args.sw_batch_size,
            ),
            args.output_json,
        )
    elif args.command == "predict":
        _print_json(
            predict_split(
                config_path,
                args.checkpoint,
                split_name=args.split,
                output_dir=args.output_dir,
                device_name=args.device,
            )
        )
    elif args.command == "evaluate":
        _print_json(
            evaluate_split(config_path, args.prediction_dir, split_name=args.split),
            args.output_json,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
