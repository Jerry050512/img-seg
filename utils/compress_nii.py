"""Compress NIfTI .nii files to .nii.gz without changing voxel data."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def iter_nii_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.name.endswith(".nii") else []
    pattern = "**/*.nii" if recursive else "*.nii"
    return sorted(path.glob(pattern))


def compress_one(src: Path, overwrite: bool, remove_source: bool) -> Path | None:
    dst = src.with_suffix(src.suffix + ".gz")
    if dst.exists() and not overwrite:
        print(f"skip existing: {dst}")
        return None

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as fin, gzip.open(tmp, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)
    tmp.replace(dst)

    if remove_source:
        src.unlink()
    print(f"compressed: {src} -> {dst}")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A .nii file or a directory.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .nii.gz files.")
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete the original .nii after successful compression.",
    )
    args = parser.parse_args()

    files = iter_nii_files(args.path, args.recursive)
    if not files:
        print("no .nii files found")
        return

    for src in files:
        compress_one(src, overwrite=args.overwrite, remove_source=args.remove_source)


if __name__ == "__main__":
    main()

