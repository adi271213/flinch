"""The graph export is local, preserves artifacts, and retains diagnostic exits."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from lerobot_preflight.cli import main
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path, monkeypatch):
    checkpoints = make_bug_fixtures(tmp_path / "fixtures")

    def no_network(*args, **kwargs):
        pytest.fail("Graph CLI attempted network access")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    return checkpoints


@pytest.mark.parametrize("suffix", [".json", ".mmd", ".md"])
def test_cli_exports_graph_and_preserves_checkpoint(checkpoints, tmp_path, suffix):
    checkpoint = checkpoints["clean_act_modern"]
    original = {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()}
    output = tmp_path / f"processor-graph{suffix}"
    report_path = tmp_path / "report.json"
    args = ["--policy", str(checkpoint), "--graph", str(output), "--report", str(report_path)]
    assert main(args) == 0
    text = output.read_text()
    if suffix == ".json":
        graph = json.loads(text)
        assert graph["nodes"] and graph["edges"] and graph["summary"]
    elif suffix == ".mmd":
        assert text.startswith("flowchart LR")
    else:
        assert "```mermaid\nflowchart LR" in text
    assert json.loads(report_path.read_text())["has_failures"] is False
    # A repeat invocation can replace its own export, never an audited artifact.
    assert main(args) == 0
    assert output.read_text() == text
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()} == original


def test_cli_graph_does_not_hide_silent_normalization_bypass(checkpoints, tmp_path):
    output = tmp_path / "bypass.json"
    assert main(["--policy", str(checkpoints["bug_4415_silent_bypass"]), "--graph", str(output)]) == 1
    graph = json.loads(output.read_text())
    assert any(edge["type"] == "UNBOUND_BYPASS" for edge in graph["edges"])


def test_cli_graph_refuses_to_overwrite_checkpoint_artifact(checkpoints, capsys):
    checkpoint = checkpoints["clean_act_modern"]
    artifact = checkpoint / "preprocessor_config.json"
    original = artifact.read_bytes()
    assert main(["--policy", str(checkpoint), "--graph", str(artifact)]) == 1
    assert artifact.read_bytes() == original
    assert "Could not write graph" in capsys.readouterr().err


def test_cli_graph_rejects_unsupported_export_format(checkpoints, tmp_path, capsys):
    output = tmp_path / "graph.svg"
    assert main(["--policy", str(checkpoints["clean_act_modern"]), "--graph", str(output)]) == 1
    assert not output.exists()
    assert "Could not write graph" in capsys.readouterr().err


def test_cli_forwards_explicit_chronology_to_graph(checkpoints, tmp_path, monkeypatch):
    from lerobot_preflight import graph

    checkpoint = checkpoints["clean_act_modern"]
    metadata = tmp_path / "chronology.json"
    metadata.write_text(json.dumps({"n_obs_steps": 1, "current_index": 0}))
    original_builder = graph.build_processor_graph
    calls = []

    def build(path: Path, metadata_path=None):
        calls.append((path, metadata_path))
        return original_builder(path, metadata_path=metadata_path)

    monkeypatch.setattr(graph, "build_processor_graph", build)
    assert (
        main(
            [
                "--policy",
                str(checkpoint),
                "--metadata",
                str(metadata),
                "--graph",
                str(tmp_path / "graph.json"),
            ]
        )
        == 0
    )
    assert calls == [(checkpoint, metadata)]
