"""Unified batch prediction CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from img_seg.inference.batch import run_batch_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="efficientnet_b0", help="Model key to run.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--input-dir", required=True, help="Input directory or file.")
    parser.add_argument("--output-dir", default="Test_Seg", help="Output directory.")
    parser.add_argument("--config", help="Optional model config path.")
    parser.add_argument("--device", help="Optional torch device override, e.g. cpu or cuda.")
    parser.add_argument("--output-json", help="Optional JSON summary path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_batch_prediction(
        model_key=args.model,
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
    )
    payload = {"results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
