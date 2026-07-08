"""Case-level split helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CaseSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    @property
    def train_val(self) -> list[str]:
        return [*self.train, *self.val]


def make_case_split(
    case_ids: Sequence[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> CaseSplit:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    ids = list(case_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Case ids must be unique")
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_cases = len(ids)
    n_test = max(1, round(n_cases * test_ratio / total)) if n_cases >= 3 else 0
    n_val = max(1, round(n_cases * val_ratio / total)) if n_cases >= 3 else 0
    if n_val + n_test >= n_cases:
        n_val = 1 if n_cases >= 3 else 0
        n_test = 1 if n_cases >= 3 else 0
    n_train = n_cases - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough cases to create a non-empty train split")

    train = sorted(ids[:n_train])
    val = sorted(ids[n_train : n_train + n_val])
    test = sorted(ids[n_train + n_val :])
    return CaseSplit(train=train, val=val, test=test)
