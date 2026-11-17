"""Action un-normalization that disagrees with training, and path typos as usage errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

from lerobot_flinch.cli import main, run_checks
from lerobot_flinch.models import DiagnosticStatus, Severity
from tests.fixtures.make_bug_fixtures import make_bug_fixtures

PRE = "policy_preprocessor_step_0_normalizer_processor.safetensors"
POST = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


@pytest.fixture
def clean(tmp_path: Path) -> Path:
    return make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]


def _edit(path: Path, **changes: tuple[int, float]) -> None:
    state = load_file(path)
    for key, (index, value) in changes.items():
        key = key.replace("__", ".")
        tensor = state[key].clone()
        tensor[index] = value
        state[key] = tensor
    save_file(state, path)


def _findings(checkpoint: Path, check_id: str) -> list:
    return [f for f in run_checks(checkpoint).findings if f.check_id == check_id]


def test_postprocessor_action_std_that_differs_from_training_is_a_conflict(clean: Path) -> None:
    trained = load_file(clean / PRE)["action.std"][3].item()
    _edit(clean / POST, action__std=(3, 1e9))

    (conflict,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert conflict.status is DiagnosticStatus.CONFLICT
    assert conflict.provenance.violating_indices == [3]
    assert conflict.details["preprocessor_values"] == [pytest.approx(trained)]
    assert conflict.details["postprocessor_values"] == [1e9]
    assert conflict.details["statistic"] == "std"
    assert main(["--policy", str(clean)]) == 1


def test_extreme_action_scale_is_an_advisory_not_a_failure(clean: Path) -> None:
    _edit(clean / POST, action__std=(3, 1e9))
    state = load_file(clean / PRE)
    save_file({k: v for k, v in state.items() if not k.startswith("action.")}, clean / PRE)
    config_path = clean / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    config["steps"][0]["config"]["norm_map"].pop("ACTION")
    config_path.write_text(json.dumps(config))

    assert not _findings(clean, "ACTION_STATS_CONFLICT")  # Nothing to compare against.
    (extreme,) = _findings(clean, "EXTREME_ACTION_SCALE")
    assert extreme.status is DiagnosticStatus.UNKNOWN
    assert extreme.severity is Severity.WARNING
    assert extreme.provenance.violating_indices == [3]
    assert main(["--policy", str(clean)]) == 0
    assert main(["--policy", str(clean), "--strict"]) == 1


def test_small_but_real_action_mean_drift_is_reported(clean: Path) -> None:
    trained = load_file(clean / PRE)["action.mean"][0].item()
    _edit(clean / POST, action__mean=(0, trained + 0.01))
    (conflict,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert conflict.details["statistic"] == "mean"
    assert conflict.provenance.violating_indices == [0]


def test_float_rounding_between_identical_stats_is_not_a_conflict(clean: Path) -> None:
    trained = load_file(clean / PRE)["action.std"][2].item()
    _edit(clean / POST, action__std=(2, trained * (1 + 1e-6)))
    assert not _findings(clean, "ACTION_STATS_CONFLICT")


def test_healthy_checkpoint_has_no_action_findings(clean: Path) -> None:
    assert not _findings(clean, "ACTION_STATS_CONFLICT")
    assert not _findings(clean, "EXTREME_ACTION_SCALE")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--policy", "{tmp}/missing"], "does not exist"),
        (["--policy", "{tmp}/file.txt"], "is a file, not a checkpoint directory"),
        (["--policy", "{clean}", "--baseline", "{tmp}/missing"], "--baseline"),
        (["--policy", "{clean}", "--sample", "{tmp}/missing.npz"], "--sample"),
        (["--policy", "{clean}", "--metadata", "{tmp}/missing.json"], "--metadata"),
    ],
)
def test_path_typos_are_usage_errors(
    clean: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str], message: str
) -> None:
    (tmp_path / "file.txt").write_text("not a checkpoint")
    argv = [arg.format(tmp=tmp_path, clean=clean) for arg in args]
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err
