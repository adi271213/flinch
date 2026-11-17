"""The public release name must coexist with the separate dataset doctor."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from lerobot_preflight import __version__
from lerobot_preflight.cli import main
from lerobot_preflight.graph import build_processor_graph, write_processor_graph
from tests.fixtures.make_bug_fixtures import make_bug_fixtures
from tests.subprocess_env import source_subprocess_env


def test_distribution_and_console_entrypoint_use_distinct_release_name():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "lerobot-preflight"
    assert metadata["project"]["scripts"] == {"lerobot-preflight": "lerobot_preflight.cli:main"}
    assert (root / "src/lerobot_preflight/__init__.py").is_file()
    assert not list((root / "src/lerobot_doctor").glob("*.py"))


def test_help_and_version_identify_the_new_console_program(capsys):
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: lerobot-preflight" in help_text
    assert "lerobot-doctor" not in help_text
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == f"lerobot-preflight {__version__}"


def test_new_namespace_does_not_shadow_a_separately_installed_dataset_doctor(tmp_path):
    dataset_root = tmp_path / "dataset-distribution"
    dataset_package = dataset_root / "lerobot_doctor"
    dataset_package.mkdir(parents=True)
    (dataset_package / "__init__.py").write_text("DATASET_LINTER = True\n")
    source = Path(__file__).resolve().parents[1] / "src"
    script = '''
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import lerobot_doctor
import lerobot_preflight
assert lerobot_doctor.DATASET_LINTER is True
assert Path(lerobot_doctor.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert Path(lerobot_preflight.__file__).resolve().is_relative_to(Path(sys.argv[2]))
print("dataset doctor and checkpoint preflight coexist")
'''
    result = subprocess.run([sys.executable, "-S", "-c", script, str(dataset_root), str(source)],
                            cwd=tmp_path, env=source_subprocess_env(), capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr
    assert "coexist" in result.stdout


def test_renamed_cli_still_diagnoses_native_silent_bypass(tmp_path, capsys):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["bug_4415_silent_bypass"]
    assert main(["--policy", str(checkpoint), "--sample", str(checkpoint / "sample_observation.npz")]) == 1
    output = capsys.readouterr().out
    assert output.startswith("lerobot-preflight:")
    assert "SILENT_STEP_BYPASS" in output


@pytest.mark.parametrize("suffix", [".json", ".mmd", ".md"])
def test_new_graph_exports_can_replace_recognized_previous_release_outputs(tmp_path, suffix):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]
    graph = build_processor_graph(checkpoint)
    output = tmp_path / ("lineage" + suffix)
    write_processor_graph(graph, output)
    legacy = output.read_text().replace("lerobot-preflight-processor-lineage-v1", "lerobot-doctor-processor-lineage-v1")
    output.write_text(legacy)
    write_processor_graph(graph, output)
    assert "lerobot-preflight-processor-lineage-v1" in output.read_text()
    if suffix == ".json":
        assert json.loads(output.read_text())["format"] == "lerobot-preflight-processor-lineage-v1"
