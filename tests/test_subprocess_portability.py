"""CLI subprocesses must work from a checkout without editable import hooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import lerobot

from tests.fixtures.make_bug_fixtures import make_bug_fixtures
from tests.subprocess_env import source_subprocess_env


def test_source_subprocess_env_preserves_dependency_pythonpath(tmp_path):
    incoming = {
        "PYTHONPATH": os.pathsep.join([str(tmp_path / "released-lerobot"), str(tmp_path / "another")]),
        "HF_HUB_OFFLINE": "1",
    }
    unchanged = dict(incoming)
    env = source_subprocess_env(incoming)
    source = str(Path(__file__).resolve().parents[1] / "src")
    assert env["PYTHONPATH"].split(os.pathsep) == [source, *incoming["PYTHONPATH"].split(os.pathsep)]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert incoming == unchanged
    assert source_subprocess_env({})["PYTHONPATH"] == source


def test_cli_source_subprocess_detects_bypass_without_editable_install(tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["bug_4415_silent_bypass"]
    source = Path(__file__).resolve().parents[1] / "src"
    expected_lerobot = Path(lerobot.__file__).resolve()
    env = source_subprocess_env()
    env["PREFLIGHT_TEST_DEPENDENCY_PATHS"] = json.dumps(
        list(
            dict.fromkeys(
                [
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                    str(expected_lerobot.parents[1]),
                ]
            )
        )
    )
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    script = """
import json, os, sys
from pathlib import Path
# -S disables site initialization and every .pth editable-install hook. Add only
# dependency directories, as plain paths, without processing their .pth files.
sys.path.extend(json.loads(os.environ["PREFLIGHT_TEST_DEPENDENCY_PATHS"]))
assert not any(type(finder).__module__.startswith("__editable") or
               getattr(finder, "__module__", "").startswith("__editable")
               for finder in sys.meta_path)
import lerobot
import lerobot.processor
assert Path(lerobot.__file__).resolve() == Path(sys.argv[2])
import socket
def no_network(*args, **kwargs):
    raise AssertionError("offline CLI opened a network socket")
socket.socket = socket.create_connection = no_network
import lerobot_preflight
from lerobot_preflight.cli import main
assert Path(lerobot_preflight.__file__).resolve().is_relative_to(Path(sys.argv[1]))
checkpoint = Path(sys.argv[3])
result = main(["--policy", str(checkpoint), "--sample", str(checkpoint / "sample_observation.npz")])
assert result == 1
print("source CLI detected SILENT_STEP_BYPASS without editable installation")
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(source), str(expected_lerobot), str(checkpoint)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SILENT_STEP_BYPASS" in result.stdout
    assert "without editable installation" in result.stdout
