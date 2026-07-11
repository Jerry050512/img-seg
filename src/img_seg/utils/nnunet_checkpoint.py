"""Utilities for exporting compact nnU-Net inference checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

INFERENCE_KEYS = (
    "network_weights",
    "trainer_name",
    "init_args",
    "inference_allowed_mirroring_axes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_payload(payload: Any, source: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a checkpoint dictionary in {source}")
    missing = [key for key in INFERENCE_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Checkpoint {source} is missing required keys: {missing}")
    if not isinstance(payload["network_weights"], dict):
        raise TypeError("network_weights must be a state dictionary")
    if not isinstance(payload["init_args"], dict) or "configuration" not in payload["init_args"]:
        raise KeyError("init_args.configuration is required by nnU-Net inference")
    return payload


def export_inference_checkpoint(
    source: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Remove training-only state while preserving exact FP32 inference weights."""

    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Output must differ from the source checkpoint")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --force to replace it")

    payload = _validate_payload(
        torch.load(source, map_location="cpu", weights_only=False),
        source,
    )
    inference_payload = {key: payload[key] for key in INFERENCE_KEYS}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        torch.save(inference_payload, temporary)
        restored = _validate_payload(
            torch.load(temporary, map_location="cpu", weights_only=False),
            temporary,
        )
        for name, expected in inference_payload["network_weights"].items():
            actual = restored["network_weights"].get(name)
            if actual is None or not torch.equal(expected, actual):
                raise RuntimeError(f"Export verification failed for tensor {name!r}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    original_bytes = source.stat().st_size
    output_bytes = output.stat().st_size
    return {
        "source": str(source),
        "output": str(output),
        "original_bytes": original_bytes,
        "output_bytes": output_bytes,
        "saved_bytes": original_bytes - output_bytes,
        "reduction_percent": round((1 - output_bytes / original_bytes) * 100, 2),
        "sha256": _sha256(output),
        "dtype_preserved": True,
        "retained_keys": list(INFERENCE_KEYS),
    }
