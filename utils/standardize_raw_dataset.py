"""Standardize raw NIfTI dataset folders to image.nii.gz/mask.nii.gz pairs.

This utility is intentionally conservative:
- it builds a complete standardized dataset in a temporary directory first;
- it validates image/mask shape and affine before moving raw folders;
- it moves old raw folders to a backup directory instead of deleting them;
- it leaves dataset/processed untouched.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from img_seg.config import resolve_project_path
from img_seg.data.cases import load_cases_from_manifest
from img_seg.io.nifti import require_nibabel


@dataclass(frozen=True)
class RawCase:
    case_id: str
    image_path: Path
    mask_path: Path
    source_top_dir: Path


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return data


def require_inside(child: Path, parent: Path) -> None:
    child.resolve().relative_to(parent.resolve())


def load_raw_cases(manifest_path: Path) -> tuple[Path, list[RawCase]]:
    manifest = load_manifest(manifest_path)
    root = resolve_project_path(manifest.get("root", "dataset"))
    cases: list[RawCase] = []
    seen_ids: set[str] = set()
    for case in load_cases_from_manifest(manifest_path):
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate case id in dataset manifest: {case.case_id}")
        seen_ids.add(case.case_id)
        if case.mask_path is None:
            raise ValueError(f"Missing mask path for case {case.case_id}")
        for path in (case.image_path, case.mask_path):
            if not path.exists():
                raise FileNotFoundError(path)
            require_inside(path, root)
        source_top_dir = root / case.image_path.relative_to(root).parts[0]
        cases.append(
            RawCase(
                case_id=case.case_id,
                image_path=case.image_path,
                mask_path=case.mask_path,
                source_top_dir=source_top_dir,
            )
        )
    if not cases:
        raise ValueError(f"No cases found in manifest: {manifest_path}")
    return root, cases


def save_image(nib: Any, src: Path, dst: Path) -> None:
    image = nib.load(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(dst))


def save_binary_mask(nib: Any, src: Path, dst: Path) -> None:
    image = nib.load(str(src))
    data = (np.asanyarray(image.dataobj) > 0).astype(np.uint8)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    output = nib.Nifti1Image(data, image.affine, header=header)
    output.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output.set_sform(image.get_sform(), int(image.header["sform_code"]))
    nib.save(output, str(dst))


def validate_pair(nib: Any, case: RawCase) -> None:
    image = nib.load(str(case.image_path))
    mask = nib.load(str(case.mask_path))
    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {case.case_id}: {image.shape} vs {mask.shape}"
        )
    if not np.allclose(image.affine, mask.affine):
        raise ValueError(f"Image/mask affine mismatch for {case.case_id}")


def build_standardized(root: Path, cases: list[RawCase], temp_dir: Path) -> None:
    nib = require_nibabel()
    if temp_dir.exists():
        raise FileExistsError(f"Temporary directory already exists: {temp_dir}")
    temp_dir.mkdir(parents=True)
    for case in cases:
        validate_pair(nib, case)
        case_dir = temp_dir / case.case_id
        save_image(nib, case.image_path, case_dir / "image.nii.gz")
        save_binary_mask(nib, case.mask_path, case_dir / "mask.nii.gz")

    expected = {case.case_id for case in cases}
    actual = {path.name for path in temp_dir.iterdir() if path.is_dir()}
    if actual != expected:
        raise RuntimeError(f"Unexpected standardized case directories: {sorted(actual)}")


def apply_standardized(root: Path, cases: list[RawCase], temp_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)

    source_dirs = sorted({case.source_top_dir for case in cases}, key=lambda path: path.name)
    target_dirs = [root / case.case_id for case in cases]
    for target in target_dirs:
        if target.exists() and target not in source_dirs:
            raise FileExistsError(f"Target directory already exists and is not a source: {target}")

    for source in source_dirs:
        if not source.is_dir():
            raise FileNotFoundError(source)
        require_inside(source, root)
        shutil.move(str(source), str(backup_dir / source.name))

    for case in cases:
        shutil.move(str(temp_dir / case.case_id), str(root / case.case_id))

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/data/dataset.yaml"),
        help="Current dataset manifest pointing to the raw source files.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Directory where old raw folders are moved.",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Directory where temporary standardized files are built.",
    )
    parser.add_argument("--apply", action="store_true", help="Move standardized data into place.")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    root, cases = load_raw_cases(manifest_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = args.temp_root / f"img_seg_standardizing_tmp_{stamp}"
    backup_dir = args.backup_root / f"img_seg_raw_before_standardize_{stamp}"

    print(f"root: {root}")
    print(f"cases: {len(cases)}")
    print(f"temp: {temp_dir}")
    print(f"backup: {backup_dir}")
    build_standardized(root, cases, temp_dir)

    if not args.apply:
        print("dry run complete; standardized files were built but not applied")
        print(f"review temp directory: {temp_dir}")
        return

    apply_standardized(root, cases, temp_dir, backup_dir)
    print("standardized raw dataset applied")
    print(f"old raw folders moved to: {backup_dir}")


if __name__ == "__main__":
    main()
