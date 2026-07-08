"""Project configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a user/config path relative to the project root by default."""

    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir or PROJECT_ROOT / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty files."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return data


def deep_get(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted config key, returning default when any level is missing."""

    current: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def project_path_from_config(config: Mapping[str, Any], dotted_key: str) -> Path:
    value = deep_get(config, dotted_key)
    if value is None:
        raise KeyError(f"Missing config key: {dotted_key}")
    return resolve_project_path(value)
