"""Published LeRobot release compatibility for the package metadata."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_counterexample_a_published_lerobot_070_satisfies_dependency():
    project = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    dependency = next(
        Requirement(item)
        for item in metadata["project"]["dependencies"]
        if Requirement(item).name == "lerobot"
    )
    assert dependency.specifier.contains("0.7.0")
    assert metadata["project"]["scripts"]["flinch"] == "lerobot_flinch.cli:main"
