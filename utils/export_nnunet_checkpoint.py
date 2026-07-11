"""Export an nnU-Net training checkpoint as a smaller inference-only checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from img_seg.utils.nnunet_checkpoint import export_inference_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Full nnU-Net training checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to <source_stem>_inference.pth next to the source",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.source.with_name(f"{args.source.stem}_inference.pth")
    result = export_inference_checkpoint(args.source, output, overwrite=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
