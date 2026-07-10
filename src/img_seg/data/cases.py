"""Dataset case discovery and manifest loading for local NIfTI folders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from img_seg.config import resolve_project_path

NIFTI_SUFFIXES = (".nii", ".nii.gz")
DEFAULT_IGNORED_DIRS = {"processed", "splits", "_standardizing_tmp"}


@dataclass(frozen=True)
class SegmentationCase:
    case_id: str
    image_path: Path
    mask_path: Path | None = None
    source_id: str | None = None


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def is_mask_file(path: Path) -> bool:
    name = path.name.lower()
    return is_nifti(path) and ("seg" in name or "mask" in name or "label" in name)


def normalize_case_id(value: str) -> str:
    """Normalize external case names to stable file/config identifiers."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        raise ValueError(f"Invalid empty case id after normalization: {value!r}")
    return normalized


def case_id_from_dir(path: Path) -> str:
    return path.name.replace(" ", "_")


def case_id_from_value(value: str, policy: str) -> str:
    if policy == "directory_name":
        case_id = value
    elif policy == "ascii_alnum_underscore":
        case_id = normalize_case_id(value)
    else:
        raise ValueError(f"Unsupported case_id_policy: {policy}")
    if not case_id:
        raise ValueError(f"Invalid empty case id: {value!r}")
    return case_id


def _should_skip(path: Path, ignored_dirs: set[str] | frozenset[str]) -> bool:
    return any(part in ignored_dirs or part.startswith(".") for part in path.parts)


def find_case_files(
    case_dir: Path,
    *,
    recursive: bool = False,
    ignored_dirs: set[str] | frozenset[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    ignored_dirs = DEFAULT_IGNORED_DIRS if ignored_dirs is None else ignored_dirs
    iterator = case_dir.rglob("*") if recursive else case_dir.iterdir()
    files = sorted(
        path
        for path in iterator
        if path.is_file()
        and is_nifti(path)
        and not _should_skip(path.relative_to(case_dir), ignored_dirs)
    )
    masks = [path for path in files if is_mask_file(path)]
    images = [path for path in files if path not in masks]
    return images, masks


def choose_single_file(candidates: list[Path], kind: str, case_id: str) -> Path:
    if not candidates:
        raise ValueError(f"No {kind} file found for case {case_id}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple {kind} files found for case {case_id}: {names}")
    return candidates[0]


def discover_cases(
    dataset_dir: str | Path,
    *,
    image_overrides: dict[str, str] | None = None,
    mask_overrides: dict[str, str] | None = None,
    ignored_dirs: set[str] | frozenset[str] | None = None,
    require_masks: bool = True,
    recursive: bool = False,
) -> list[SegmentationCase]:
    """Discover one image and one optional mask per dataset case directory.

    By default this scans immediate child directories only. Set ``recursive`` to
    search nested files inside each case, and use overrides to select a specific
    image or mask when a case contains multiple candidates.
    """

    dataset_dir = Path(dataset_dir)
    image_overrides = image_overrides or {}
    mask_overrides = mask_overrides or {}
    ignored_dirs = DEFAULT_IGNORED_DIRS if ignored_dirs is None else ignored_dirs
    cases: list[SegmentationCase] = []
    case_dirs = (
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in ignored_dirs
    )
    for case_dir in sorted(case_dirs):
        case_id = case_id_from_dir(case_dir)
        images, masks = find_case_files(case_dir, recursive=recursive, ignored_dirs=ignored_dirs)

        if case_id in image_overrides:
            image_path = case_dir / image_overrides[case_id]
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image override does not exist for {case_id}: {image_path}"
                )
        else:
            image_path = choose_single_file(images, "image", case_id)

        mask_path: Path | None = None
        if case_id in mask_overrides:
            mask_path = case_dir / mask_overrides[case_id]
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask override does not exist for {case_id}: {mask_path}")
        elif masks:
            mask_path = choose_single_file(masks, "mask", case_id)
        elif require_masks:
            raise ValueError(f"No mask file found for case {case_id}")

        cases.append(
            SegmentationCase(
                case_id=case_id,
                image_path=image_path,
                mask_path=mask_path,
                source_id=case_dir.name,
            )
        )
    if not cases:
        raise ValueError(f"No case directories found in {dataset_dir}")
    return cases


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return data


def _manifest_path(root: Path, value: str, *, case_id: str, kind: str) -> Path:
    candidate = root / Path(value)
    if not candidate.exists():
        raise FileNotFoundError(f"{kind} path does not exist for {case_id}: {candidate}")
    if not is_nifti(candidate):
        raise ValueError(f"{kind} path is not a NIfTI file for {case_id}: {candidate}")
    return candidate.resolve()


def _load_explicit_cases(
    manifest: dict[str, Any], root: Path, path: str | Path
) -> list[SegmentationCase]:
    entries = manifest.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Dataset manifest must contain a non-empty cases list: {path}")

    case_id_policy = str(manifest.get("case_id_policy", "ascii_alnum_underscore"))

    seen_ids: set[str] = set()
    cases: list[SegmentationCase] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid case entry in {path}: {entry!r}")
        case_id = case_id_from_value(str(entry.get("id", "")), case_id_policy)
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id in dataset manifest: {case_id}")
        seen_ids.add(case_id)

        image_value = entry.get("image")
        if not image_value:
            raise ValueError(f"Missing image path for case {case_id}")
        image_path = _manifest_path(root, str(image_value), case_id=case_id, kind="image")

        mask_path = None
        mask_value = entry.get("mask")
        if mask_value:
            mask_path = _manifest_path(root, str(mask_value), case_id=case_id, kind="mask")

        source_id = str(entry.get("source_id", case_id))
        cases.append(
            SegmentationCase(
                case_id=case_id,
                image_path=image_path,
                mask_path=mask_path,
                source_id=source_id,
            )
        )
    return cases


def _excluded_dirs(manifest: dict[str, Any]) -> set[str]:
    excluded = set(DEFAULT_IGNORED_DIRS)
    configured = manifest.get("exclude", [])
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, list):
        raise ValueError("Dataset manifest exclude must be a list of directory names")
    excluded.update(str(item) for item in configured)
    return excluded


def _validate_mask_policy(manifest: dict[str, Any]) -> None:
    policy = manifest.get("mask_policy")
    if policy is not None and policy != "nonzero_to_one":
        raise ValueError(f"Unsupported mask_policy: {policy}")


def _load_case_dir_pairs(manifest: dict[str, Any], root: Path) -> list[SegmentationCase]:
    _validate_mask_policy(manifest)
    image_name = str(manifest.get("image_name", "image.nii.gz"))
    mask_name = str(manifest.get("mask_name", "mask.nii.gz"))
    case_id_policy = str(manifest.get("case_id_policy", "directory_name"))
    excluded = _excluded_dirs(manifest)

    cases: list[SegmentationCase] = []
    seen_ids: set[str] = set()
    case_dirs = (
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in excluded
    )
    for case_dir in sorted(case_dirs):
        case_id = case_id_from_value(case_dir.name, case_id_policy)
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id from dataset layout: {case_id}")
        seen_ids.add(case_id)
        image_path = case_dir / image_name
        mask_path = case_dir / mask_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist for {case_id}: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist for {case_id}: {mask_path}")
        if not is_nifti(image_path):
            raise ValueError(f"Image path is not a NIfTI file for {case_id}: {image_path}")
        if not is_nifti(mask_path):
            raise ValueError(f"Mask path is not a NIfTI file for {case_id}: {mask_path}")
        cases.append(
            SegmentationCase(
                case_id=case_id,
                image_path=image_path.resolve(),
                mask_path=mask_path.resolve(),
                source_id=case_dir.name,
            )
        )
    if not cases:
        raise ValueError(f"No case directories found in {root}")
    return cases


def load_cases_from_manifest(path: str | Path) -> list[SegmentationCase]:
    """Load image/mask pairs from an explicit manifest or a layout config."""

    manifest = _load_manifest(path)
    root = resolve_project_path(manifest.get("root", "dataset"))
    if "cases" in manifest:
        return _load_explicit_cases(manifest, root, path)

    layout = str(manifest.get("layout", ""))
    if layout == "case_dir_pair":
        return _load_case_dir_pairs(manifest, root)
    raise ValueError(f"Unsupported dataset manifest layout: {layout!r}")
