"""Project-local nnU-Net trainer variants.

These classes are discovered through nnU-Net's ``nnUNet_extTrainer`` directory
scan. Keep each custom trainer in an importable Python module in this folder.
"""

from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class nnUNetTrainer_200epochs(nnUNetTrainer):
    """Run nnU-Net with a fixed 200-epoch training schedule."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device | None = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device or default_device())
        self.num_epochs = 200
