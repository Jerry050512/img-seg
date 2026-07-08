"""Dataset case discovery for local NIfTI folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass(frozen=True)
class SegmentationCase:
    case_id: str
    image_path: Path
    mask_path: Path | None = None


def is_nifti(path: Path) -> bool:
    return path.name.endswith(NIFTI_SUFFIXES)


def is_mask_file(path: Path) -> bool:
    name = path.name.lower()
    return is_nifti(path) and ("seg" in name or "mask" in name or "label" in name)


def case_id_from_dir(path: Path) -> str:
    return path.name.replace(" ", "_")


def find_case_files(case_dir: Path) -> tuple[list[Path], list[Path]]:
    files = sorted(path for path in case_dir.iterdir() if path.is_file() and is_nifti(path))
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
    mask_overrides: dict[str, str] | None = None,
    require_masks: bool = True,
) -> list[SegmentationCase]:
    """Discover one image and one optional mask per immediate dataset child directory."""

    dataset_dir = Path(dataset_dir)
    mask_overrides = mask_overrides or {}
    cases: list[SegmentationCase] = []
    for case_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        case_id = case_id_from_dir(case_dir)
        images, masks = find_case_files(case_dir)
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

        cases.append(SegmentationCase(case_id=case_id, image_path=image_path, mask_path=mask_path))
    if not cases:
        raise ValueError(f"No case directories found in {dataset_dir}")
    return cases
