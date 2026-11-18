"""The README demo runs and shows the guard stopping the exploded input."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "guard_demo.py"


def test_guard_demo_shows_the_explosion_and_the_stop(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(DEMO), run_name="__main__")
    without, with_guard = capsys.readouterr().out.split("With the guard")
    commands = [float(line.split()[-1]) for line in without.splitlines() if "largest joint command" in line]
    assert len(commands) == 5 and min(commands) > 10
    assert "guard stopped the input" in with_guard
    assert "flattened indices [2]" in with_guard
