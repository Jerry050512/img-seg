from pathlib import Path

import pytest
import torch

from img_seg.utils.nnunet_checkpoint import (
    INFERENCE_KEYS,
    export_inference_checkpoint,
)


def _write_training_checkpoint(path: Path) -> dict:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    payload = {
        "network_weights": {"layer.weight": weight, "layer.alias": weight},
        "optimizer_state": {"state": {0: {"momentum_buffer": torch.ones(12)}}},
        "grad_scaler_state": {"scale": 65536.0},
        "logging": {"train_losses": [1.0]},
        "trainer_name": "nnUNetTrainer",
        "init_args": {"configuration": "2d", "fold": 0},
        "inference_allowed_mirroring_axes": (0, 1),
    }
    torch.save(payload, path)
    return payload


def test_export_keeps_exact_inference_state_and_drops_training_state(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint_best.pth"
    output = tmp_path / "checkpoint_best_inference.pth"
    original = _write_training_checkpoint(source)

    result = export_inference_checkpoint(source, output)
    exported = torch.load(output, map_location="cpu", weights_only=False)

    assert tuple(exported) == INFERENCE_KEYS
    assert "optimizer_state" not in exported
    assert torch.equal(
        exported["network_weights"]["layer.weight"],
        original["network_weights"]["layer.weight"],
    )
    assert exported["network_weights"]["layer.weight"].dtype == torch.float32
    assert result["dtype_preserved"] is True
    assert len(result["sha256"]) == 64


def test_export_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint_best.pth"
    output = tmp_path / "checkpoint_best_inference.pth"
    _write_training_checkpoint(source)
    output.touch()

    with pytest.raises(FileExistsError):
        export_inference_checkpoint(source, output)
