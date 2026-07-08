from __future__ import annotations

from pathlib import Path

import pytest

from img_seg.data.cases import discover_cases
from img_seg.data.splits import make_case_split


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def test_discover_cases_requires_mask_override_for_duplicate_masks(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_a"
    touch(case_dir / "image.nii")
    touch(case_dir / "image_Seg.nii")
    touch(case_dir / "image_Seg(1).nii")

    with pytest.raises(ValueError, match="Multiple mask files"):
        discover_cases(tmp_path)

    cases = discover_cases(tmp_path, mask_overrides={"case_a": "image_Seg.nii"})

    assert len(cases) == 1
    assert cases[0].case_id == "case_a"
    assert cases[0].image_path.name == "image.nii"
    assert cases[0].mask_path is not None
    assert cases[0].mask_path.name == "image_Seg.nii"


def test_make_case_split_is_case_level_and_reproducible() -> None:
    case_ids = [f"case_{idx}" for idx in range(8)]

    split_a = make_case_split(case_ids, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    split_b = make_case_split(case_ids, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert split_a == split_b
    assert len(split_a.train) == 6
    assert len(split_a.val) == 1
    assert len(split_a.test) == 1
    assert set(split_a.train).isdisjoint(split_a.val)
    assert set(split_a.train).isdisjoint(split_a.test)
    assert set(split_a.val).isdisjoint(split_a.test)


def test_discover_cases_ignores_processed_dir(tmp_path: Path) -> None:
    touch(tmp_path / "case_a" / "image.nii")
    touch(tmp_path / "case_a" / "image_Seg.nii")
    (tmp_path / "processed" / "nnunet_raw").mkdir(parents=True)

    cases = discover_cases(tmp_path)

    assert [case.case_id for case in cases] == ["case_a"]
