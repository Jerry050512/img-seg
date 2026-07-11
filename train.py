"""Course delivery training entry point for all three models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from img_seg.course_delivery import config_path_for, normalize_legacy_argv, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["nnunet_v2", "monai_segresnet", "efficientnet_b0"],
        default="efficientnet_b0",
    )
    parser.add_argument("--config")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--skip-prepare", action="store_true", help="nnU-Net: reuse converted data."
    )
    parser.add_argument("--skip-plan", action="store_true", help="nnU-Net: reuse preprocessing.")
    parser.add_argument(
        "--dry-run", action="store_true", help="nnU-Net: print external commands only."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args(normalize_legacy_argv(sys.argv[1:]))
    result = run_training(
        model=args.model,
        config_path=config_path_for(args.model, args.config),
        data_path=args.data_path,
        epochs=args.epochs,
        device=args.device,
        resume=args.resume,
        output_dir=args.output_dir,
        skip_prepare=args.skip_prepare,
        skip_plan=args.skip_plan,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
