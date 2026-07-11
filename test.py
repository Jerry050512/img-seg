"""Course delivery batch evaluation entry point; final stdout line is JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from img_seg.course_delivery import (
    config_path_for,
    evaluate_predictions,
    normalize_legacy_argv,
    rounded_json,
    run_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["nnunet_v2", "monai_segresnet", "efficientnet_b0"],
        default="efficientnet_b0",
    )
    parser.add_argument("--config")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--reference-path", help="Optional separate mask directory.")
    parser.add_argument("--weight-path", "--weight", dest="weight_path", required=True)
    parser.add_argument("--output-dir", default="outputs/course_test")
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args(normalize_legacy_argv(sys.argv[1:]))
    results = run_predictions(
        model=args.model,
        checkpoint=args.weight_path,
        input_path=args.data_path,
        output_dir=args.output_dir,
        config_path=config_path_for(args.model, args.config),
        device=args.device,
    )
    metrics = evaluate_predictions(
        results,
        checkpoint=args.weight_path,
        reference_path=args.reference_path,
    )
    print(rounded_json(metrics))


if __name__ == "__main__":
    main()
