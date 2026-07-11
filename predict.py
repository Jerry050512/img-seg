"""Course delivery single-file or folder prediction entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from img_seg.course_delivery import (
    config_path_for,
    normalize_legacy_argv,
    run_predictions,
    write_prediction_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["nnunet_v2", "monai_segresnet", "efficientnet_b0"],
        default="efficientnet_b0",
    )
    parser.add_argument("--config")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image")
    sources.add_argument("--input-dir")
    parser.add_argument("--weight", "--weight-path", dest="weight", required=True)
    parser.add_argument(
        "--output", help="Exact preview path for a single input, or exact .nii.gz mask path."
    )
    parser.add_argument("--output-dir", default="Test_Seg")
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args(normalize_legacy_argv(sys.argv[1:]))
    input_path = args.image or args.input_dir
    results = run_predictions(
        model=args.model,
        checkpoint=args.weight,
        input_path=input_path,
        output_dir=args.output_dir,
        config_path=config_path_for(args.model, args.config),
        device=args.device,
    )
    artifacts = write_prediction_artifacts(
        results,
        requested_output=args.output,
        output_dir=args.output_dir,
    )
    print(json.dumps({"results": artifacts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
