"""Source-tree imports for CLI subprocesses without an editable installation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def source_subprocess_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Add this checkout's src while preserving external dependency selections."""
    env = dict(os.environ if base_env is None else base_env)
    source = str(Path(__file__).resolve().parents[1] / "src")
    existing = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry and entry != source]
    env["PYTHONPATH"] = os.pathsep.join([source, *existing])
    return env
