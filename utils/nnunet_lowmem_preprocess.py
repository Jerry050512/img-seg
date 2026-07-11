"""Low-memory nnU-Net 2D preprocessing helper for large foreground masks."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from img_seg.config import deep_get
from img_seg.models.nnunet_v2 import (
    dataset_folder_name,
    load_nnunet_config,
    nnunet_paths_from_config,
)


def _as_labels(item: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(item, (tuple, list)):
        return tuple(int(label) for label in item)
    return (int(item),)


def _sample_offsets_without_replacement(
    count: int,
    target: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample unique ordinal offsets with bounded memory and guaranteed progress."""

    if target == count:
        return np.arange(count, dtype=np.int64)

    selected: set[int] = set()
    # Floyd's algorithm: exactly target unique integers from range(count), no rejection loop.
    for value in range(count - target, count):
        candidate = int(rng.randint(0, value + 1))
        if candidate in selected:
            selected.add(value)
        else:
            selected.add(candidate)
    return np.fromiter(selected, dtype=np.int64, count=target)


def _sample_flat_indices_for_labels(
    flat_seg: np.ndarray,
    labels: tuple[int, ...],
    valid_count: int,
    target: int,
    rng: np.random.RandomState,
    chunk_size: int = 5_000_000,
) -> np.ndarray:
    selected_offsets = np.sort(
        _sample_offsets_without_replacement(
            valid_count,
            target,
            rng,
        )
    )
    picked_chunks: list[np.ndarray] = []
    seen_valid = 0
    selected_cursor = 0

    for chunk_start in range(0, flat_seg.size, chunk_size):
        if selected_cursor >= selected_offsets.size:
            break
        chunk = flat_seg[chunk_start : chunk_start + chunk_size]
        valid_mask = np.isin(chunk, labels)
        valid_in_chunk = int(np.count_nonzero(valid_mask))
        if valid_in_chunk == 0:
            continue

        chunk_end_offset = seen_valid + valid_in_chunk
        next_cursor = int(
            np.searchsorted(selected_offsets, chunk_end_offset, side="left")
        )
        if next_cursor > selected_cursor:
            local_offsets = selected_offsets[selected_cursor:next_cursor] - seen_valid
            valid_local_indices = np.flatnonzero(valid_mask)
            picked_chunks.append(valid_local_indices[local_offsets] + chunk_start)
            selected_cursor = next_cursor
        seen_valid = chunk_end_offset

    if selected_cursor != selected_offsets.size:
        raise RuntimeError(
            f"Sampled {selected_cursor}/{selected_offsets.size} foreground locations"
        )
    return np.concatenate(picked_chunks).astype(np.int64, copy=False)


def sample_foreground_locations_lowmem(
    seg: np.ndarray,
    classes_or_regions: list[int] | list[tuple[int, ...]],
    seed: int = 1234,
    verbose: bool = False,
    min_num_samples: int = 10000,
    min_percent_coverage: float = 0.01,
) -> dict[int | tuple[int, ...], np.ndarray | list]:
    """Sample class locations without materializing all foreground coordinates."""

    rng = np.random.RandomState(seed)
    flat_seg = np.ravel(seg)
    class_locs: dict[int | tuple[int, ...], np.ndarray | list] = {}

    for item in classes_or_regions:
        labels = _as_labels(item)
        key = labels if isinstance(item, (tuple, list)) else labels[0]
        valid_count = int(np.count_nonzero(np.isin(flat_seg, labels)))
        if valid_count == 0:
            class_locs[key] = []
            continue

        target = min(min_num_samples, valid_count)
        target = max(target, int(np.ceil(valid_count * min_percent_coverage)))
        flat_indices = _sample_flat_indices_for_labels(
            flat_seg,
            labels,
            valid_count,
            target,
            rng,
        )
        coords = np.column_stack(np.unravel_index(flat_indices, seg.shape)).astype(np.int64)
        class_locs[key] = coords
        if verbose:
            print(item, coords.shape[0])

    return class_locs


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cases_from_split(preprocessed_dataset_dir: Path) -> list[str]:
    split_text = (preprocessed_dataset_dir / "splits_final.json").read_text(encoding="utf-8")
    splits = json.loads(split_text)
    first = splits[0]
    return [*first["train"], *first["val"]]


def expected_outputs(output_prefix: Path) -> list[Path]:
    return [
        output_prefix.with_suffix(".b2nd"),
        output_prefix.with_name(f"{output_prefix.name}_seg.b2nd"),
        output_prefix.with_suffix(".pkl"),
    ]


def remove_existing_outputs(output_prefix: Path) -> None:
    for path in expected_outputs(output_prefix):
        if path.exists():
            path.unlink()


def sync_gt_segmentations(raw_dataset_dir: Path, preprocessed_dataset_dir: Path) -> int:
    """Mirror training labels required by nnU-Net's official validation step."""

    dataset_json = load_json(raw_dataset_dir / "dataset.json")
    file_ending = str(dataset_json["file_ending"])
    labels_dir = raw_dataset_dir / "labelsTr"
    output_dir = preprocessed_dataset_dir / "gt_segmentations"
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in sorted(labels_dir.glob(f"*{file_ending}")):
        destination = output_dir / source.name
        if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            continue
        shutil.copy2(source, destination)
        copied += 1
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nnunet_v2/base.yaml")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_nnunet_config(args.config)
    paths = nnunet_paths_from_config(config)
    dataset_dir_name = dataset_folder_name(
        int(deep_get(config, "dataset.id")),
        str(deep_get(config, "dataset.name")),
    )
    raw_dataset_dir = paths.raw / dataset_dir_name
    preprocessed_dataset_dir = paths.preprocessed / dataset_dir_name
    output_dir = preprocessed_dataset_dir / f"nnUNetPlans_{args.configuration}"

    plans_manager = PlansManager(load_json(preprocessed_dataset_dir / "nnUNetPlans.json"))
    configuration_manager = plans_manager.get_configuration(args.configuration)
    dataset_json = load_json(preprocessed_dataset_dir / "dataset.json")

    copied = sync_gt_segmentations(raw_dataset_dir, preprocessed_dataset_dir)
    print(f"synced gt_segmentations: {copied} copied")

    DefaultPreprocessor._sample_foreground_locations = staticmethod(
        sample_foreground_locations_lowmem
    )
    preprocessor = DefaultPreprocessor(verbose=True)
    cases = args.cases or cases_from_split(preprocessed_dataset_dir)

    for case_id in cases:
        image = raw_dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        label = raw_dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
        output_prefix = output_dir / case_id
        if not args.overwrite and all(path.exists() for path in expected_outputs(output_prefix)):
            print(f"skip {case_id}: already preprocessed")
            continue
        remove_existing_outputs(output_prefix)
        print(f"preprocess {case_id}")
        preprocessor.run_case_save(
            str(output_prefix),
            [str(image)],
            str(label),
            plans_manager,
            configuration_manager,
            dataset_json,
        )


if __name__ == "__main__":
    main()
