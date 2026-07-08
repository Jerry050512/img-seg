"""Audit NIfTI headers and, optionally, segmentation labels.

This script intentionally avoids nibabel so it can run before the project
environment is fully installed. It supports common single-file NIfTI .nii and
.nii.gz files used in this project.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import struct
from pathlib import Path
from typing import BinaryIO


DTYPES = {
    2: ("uint8", 1),
    4: ("int16", 2),
    8: ("int32", 4),
    16: ("float32", 4),
    64: ("float64", 8),
    256: ("int8", 1),
    512: ("uint16", 2),
    768: ("uint32", 4),
}


def open_maybe_gzip(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.name.endswith(".gz") else path.open("rb")


def read_header(path: Path) -> dict:
    with open_maybe_gzip(path) as f:
        header = f.read(348)
    if len(header) < 348:
        raise ValueError(f"{path} is too small to be a NIfTI file")

    endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
    dims = struct.unpack(endian + "8h", header[40:56])
    ndim = max(0, dims[0])
    shape = tuple(int(x) for x in dims[1 : 1 + ndim])
    datatype = struct.unpack(endian + "h", header[70:72])[0]
    bitpix = struct.unpack(endian + "h", header[72:74])[0]
    pixdim = struct.unpack(endian + "8f", header[76:108])
    spacing = tuple(round(float(x), 5) for x in pixdim[1 : 1 + ndim])
    vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
    magic = header[344:348].decode("latin1", "ignore").strip("\x00")

    dtype, bytes_per_voxel = DTYPES.get(datatype, (str(datatype), max(1, bitpix // 8)))
    voxel_count = 1
    for dim in shape:
        voxel_count *= dim
    return {
        "shape": "x".join(str(x) for x in shape),
        "spacing": "x".join(str(x) for x in spacing),
        "dtype": dtype,
        "bytes_per_voxel": bytes_per_voxel,
        "vox_offset": vox_offset,
        "magic": magic,
        "voxel_count": voxel_count,
    }


def label_summary(path: Path, header: dict) -> tuple[str, str]:
    try:
        import numpy as np
    except ImportError:
        return "", ""

    dtype = header["dtype"]
    if dtype not in {"uint8", "int8", "uint16", "int16", "uint32", "int32"}:
        return "", ""

    np_dtype = getattr(np, dtype)
    offset = int(header["vox_offset"])
    voxel_count = int(header["voxel_count"])
    values: set[int] = set()
    nonzero = 0

    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            f.seek(offset)
            raw = f.read()
        array = np.frombuffer(raw, dtype=np_dtype, count=voxel_count)
        chunks = [array]
    else:
        array = np.memmap(path, dtype=np_dtype, mode="r", offset=offset, shape=(voxel_count,))
        chunks = (array[i : i + 10_000_000] for i in range(0, voxel_count, 10_000_000))

    for chunk in chunks:
        values.update(int(x) for x in np.unique(chunk))
        nonzero += int(np.count_nonzero(chunk))
        if len(values) > 32:
            break

    labels = ",".join(str(x) for x in sorted(values))
    foreground = f"{nonzero / voxel_count * 100:.4f}%"
    return labels, foreground


def iter_nifti(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--labels", action="store_true", help="Compute labels for files containing 'seg'.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    rows = []
    for path in iter_nifti(args.path):
        header = read_header(path)
        labels, foreground = ("", "")
        if args.labels and "seg" in path.name.lower():
            labels, foreground = label_summary(path, header)
        rows.append(
            {
                "path": str(path),
                "size_mb": f"{os.path.getsize(path) / 1024 / 1024:.2f}",
                "shape": header["shape"],
                "spacing": header["spacing"],
                "dtype": header["dtype"],
                "labels": labels,
                "foreground": foreground,
            }
        )

    fieldnames = ["path", "size_mb", "shape", "spacing", "dtype", "labels", "foreground"]
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

