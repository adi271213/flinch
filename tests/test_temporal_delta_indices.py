"""Declared delta-zero chronology determines whether native index zero is valid."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from lerobot_preflight.models import DiagnosticStatus
from lerobot_preflight.temporal import run_temporal_audit
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


def _checkpoint(tmp_path: Path, indices, *, count=None) -> Path:
    checkpoint = make_bug_fixtures(tmp_path)["bug_5315_temporal_anchor"]
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["observation_delta_indices"] = indices
    if count is not None:
        config["n_obs_steps"] = count
    path.write_text(json.dumps(config))
    return checkpoint


@pytest.mark.parametrize("indices,order,status,expected_index", [
    ([-1, 0], "oldest_first", DiagnosticStatus.FAIL, 1),
    ([-2, -1, 0], "oldest_first", DiagnosticStatus.FAIL, 2),
    ([0, 1], "current_first", DiagnosticStatus.PASS, 0),
    ([0, 1, 2], "current_first", DiagnosticStatus.PASS, 0),
    ([0, -1], "current_first", DiagnosticStatus.PASS, 0),
])
def test_delta_zero_chronology_runs_native_anchor_without_metadata(
    tmp_path, monkeypatch, indices, order, status, expected_index,
):
    checkpoint = _checkpoint(tmp_path, indices, count=len(indices))
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()}

    def no_network(*args, **kwargs):
        pytest.fail("Temporal audit attempted network access")

    monkeypatch.setattr(socket, "socket", no_network)
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is status
    assert finding.details["chronology_source"] == "config.observation_delta_indices"
    assert finding.details["observation_delta_indices"] == indices
    assert finding.details["observation_order"] == order
    assert finding.details["actual_anchor_index"] == 0
    assert finding.details["expected_anchor_index"] == expected_index
    assert finding.details["native_roundtrip_passed"] is True
    provenance = finding.details["chronology_provenance"]
    assert provenance["file_path"] == str(checkpoint / "config.json")
    assert provenance["tensor_key"] == "observation_delta_indices"
    assert len(provenance["sha256_prefix"]) == 12
    assert before == {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()}


@pytest.mark.parametrize("indices,order,current,status", [
    ([-2, -1, 0], "current_first", 0, DiagnosticStatus.PASS),
    ([0, 1, 2], "oldest_first", -1, DiagnosticStatus.FAIL),
])
def test_explicit_metadata_overrides_inferred_delta_chronology(tmp_path, indices, order, current, status):
    checkpoint = _checkpoint(tmp_path, indices)
    sidecar = checkpoint / "caller_metadata.json"
    sidecar.write_text(json.dumps({"observation_order": order, "current_index": current}))
    finding, = run_temporal_audit(checkpoint, metadata_path=sidecar)
    assert finding.status is status
    assert finding.details["chronology_source"] == "explicit_metadata"
    assert finding.details["observation_order"] == order
    assert finding.details["observation_delta_indices"] == indices
    assert finding.details["n_obs_steps"] == len(indices)


@pytest.mark.parametrize("indices", [[-2, 0, 1], [0, 0], [-2, -1], [1, 0], [False, 1], [0, "1"]])
def test_ambiguous_or_invalid_deltas_do_not_invent_chronology(tmp_path, indices):
    checkpoint = _checkpoint(tmp_path, indices, count=len(indices))
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert finding.details["missing_declarations"] == ["temporal_observation_ordering"]
    assert finding.details["observation_delta_indices"] == indices
    assert finding.provenance.tensor_key == "observation_delta_indices"


@pytest.mark.parametrize("indices,status", [
    ([0, 1], DiagnosticStatus.PASS), ([-1, 0], DiagnosticStatus.FAIL),
])
@pytest.mark.parametrize("count", [1, 3])
def test_delta_window_length_takes_precedence_over_policy_n_obs_steps(tmp_path, count, indices, status):
    checkpoint = _checkpoint(tmp_path, indices, count=count)
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is status
    assert finding.details["n_obs_steps"] == 2
    assert finding.details["declared_n_obs_steps"] == count
    assert finding.details["window_length_source"] == "config.observation_delta_indices"
    assert finding.details["synthetic_state_shape"] == [1, 2, 14]
    assert finding.details["actual_anchor_index"] == 0


@pytest.mark.parametrize("window", [8, 16])
def test_current_first_world_model_windows_match_native_anchor_despite_n_obs_steps_one(tmp_path, window):
    checkpoint = _checkpoint(tmp_path, list(range(window)), count=1)
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["n_obs_steps"] == window
    assert finding.details["declared_n_obs_steps"] == 1
    assert finding.details["actual_anchor_index"] == finding.details["expected_anchor_index"] == 0
    assert finding.details["native_roundtrip_passed"] is True


def test_delta_sequence_declares_window_length_when_n_obs_steps_absent(tmp_path):
    checkpoint = _checkpoint(tmp_path, [0, 1])
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["n_obs_steps"] == 2
    assert finding.details["synthetic_state_shape"] == [1, 2, 14]


@pytest.mark.parametrize("generic,state_indices,status,effective_count,expected_anchor", [
    ([0, 1], [-1, 0], DiagnosticStatus.FAIL, 2, 1),
    ([-1, 0], [0], DiagnosticStatus.PASS, 1, 0),
])
def test_state_specific_delta_contract_overrides_generic_observation_window(
    tmp_path, generic, state_indices, status, effective_count, expected_anchor,
):
    checkpoint = _checkpoint(tmp_path, generic, count=1)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["state_observation_delta_indices"] = state_indices
    path.write_text(json.dumps(config))
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is status
    assert finding.details["n_obs_steps"] == effective_count
    assert finding.details["declared_n_obs_steps"] == 1
    assert finding.details["observation_delta_indices"] == generic
    assert finding.details["effective_state_delta_indices"] == state_indices
    assert finding.details["chronology_source"] == "config.state_observation_delta_indices"
    assert finding.details["window_length_source"] == "config.state_observation_delta_indices"
    assert finding.details["chronology_provenance"]["tensor_key"] == "state_observation_delta_indices"
    assert finding.details["expected_anchor_index"] == expected_anchor
    assert finding.details["actual_anchor_index"] == 0


@pytest.mark.parametrize("state_indices", [[], "[0]", [False], [1], [0, "1"]])
def test_invalid_state_specific_deltas_never_fall_back_to_generic_current_first(tmp_path, state_indices):
    checkpoint = _checkpoint(tmp_path, [0, 1], count=1)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["state_observation_delta_indices"] = state_indices
    path.write_text(json.dumps(config))
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert finding.details["required_coverage"] is True
    assert finding.details["state_delta_indices_source"] == "config.state_observation_delta_indices"
    assert finding.provenance.tensor_key == "state_observation_delta_indices"


def test_null_state_override_uses_generic_delta_contract_like_native_factory(tmp_path):
    checkpoint = _checkpoint(tmp_path, [0, 1], count=1)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["state_observation_delta_indices"] = None
    path.write_text(json.dumps(config))
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["effective_state_delta_indices"] == [0, 1]
    assert finding.details["state_delta_indices_source"] == "config.observation_delta_indices"


@pytest.mark.parametrize("indices", [[], [0], "[-1, 0]"])
def test_short_or_non_sequence_deltas_cannot_supply_missing_window(tmp_path, indices):
    checkpoint = _checkpoint(tmp_path, indices)
    finding, = run_temporal_audit(checkpoint)
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert "n_obs_steps" in finding.details["missing_declarations"]
