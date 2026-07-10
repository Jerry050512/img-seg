from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "utils" / "nnunet_lowmem_preprocess.py"
SPEC = importlib.util.spec_from_file_location("nnunet_lowmem_preprocess", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LOWMEM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOWMEM)

sample_foreground_locations_lowmem = LOWMEM.sample_foreground_locations_lowmem
sync_gt_segmentations = LOWMEM.sync_gt_segmentations


def test_sample_foreground_locations_lowmem_returns_valid_coordinates() -> None:
    seg = np.zeros((1, 20, 20), dtype=np.uint8)
    seg[:, 2:18, 3:17] = 1

    locations = sample_foreground_locations_lowmem(
        seg,
        [1, 2],
        min_num_samples=25,
        min_percent_coverage=0.01,
    )

    assert locations[1].shape == (25, 3)
    assert np.all(seg[tuple(locations[1].T)] == 1)
    assert np.unique(locations[1], axis=0).shape == locations[1].shape
    assert locations[2] == []


def test_sample_foreground_locations_lowmem_covers_small_sparse_masks() -> None:
    seg = np.zeros((1, 100, 100), dtype=np.uint8)
    foreground = np.array([[0, 1, 1], [0, 25, 40], [0, 99, 99]])
    seg[tuple(foreground.T)] = 1

    locations = sample_foreground_locations_lowmem(
        seg,
        [1],
        min_num_samples=10,
        min_percent_coverage=0.01,
    )[1]

    assert locations.shape == (3, 3)
    assert sorted(map(tuple, locations.tolist())) == sorted(map(tuple, foreground.tolist()))


def test_sync_gt_segmentations_copies_training_labels(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    labels_dir = raw_dir / "labelsTr"
    labels_dir.mkdir(parents=True)
    (raw_dir / "dataset.json").write_text(
        json.dumps({"file_ending": ".nii.gz"}),
        encoding="utf-8",
    )
    (labels_dir / "case_a.nii.gz").write_bytes(b"label-a")
    (labels_dir / "case_b.nii.gz").write_bytes(b"label-b")
    preprocessed_dir = tmp_path / "preprocessed"

    assert sync_gt_segmentations(raw_dir, preprocessed_dir) == 2
    assert (preprocessed_dir / "gt_segmentations" / "case_a.nii.gz").read_bytes() == b"label-a"
    assert sync_gt_segmentations(raw_dir, preprocessed_dir) == 0
