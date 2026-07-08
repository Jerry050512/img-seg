"""Small subprocess helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    print("+ " + " ".join(command))
    if dry_run:
        return None
    return subprocess.run(command, cwd=cwd, env=env, check=True, text=True)
