"""Reject ambiguous and malformed checkpoint artifacts before numeric dispatch."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from lerobot_preflight.models import DiagnosticStatus
from lerobot_preflight.preflight import run_preflight
from lerobot_preflight.privilege_delta import run_privilege_delta
from lerobot_preflight.trace import run_offline_trace
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path / "checkpoints")


@pytest.mark.parametrize("entry", [
    None,
    {"registry_name": [], "config": {}},
    {"registry_name": {}, "config": {}},
    {"class": [], "config": {}},
    {"config": {}},
    {"registry_name": "normalizer_processor", "config": []},
    {"registry_name": "normalizer_processor", "config": None},
    {"registry_name": "normalizer_processor", "state_file": []},
    {"registry_name": "normalizer_processor", "artifacts": []},
    {"registry_name": "normalizer_processor", "artifacts": {"tokenizer": {}}},
])
def test_malformed_step_declarations_are_blocking_findings(checkpoints, entry):
    root = checkpoints["clean_act_modern"]
    path = root / "preprocessor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["steps"][0] = entry
    path.write_text(json.dumps(config), encoding="utf-8")

    report = run_preflight(root)

    failures = [finding for finding in report.findings
                if finding.check_id == "SCHEMA_GENERATION" and finding.status is DiagnosticStatus.FAIL]
    assert report.has_failures()
    assert any(finding.details.get("step_index") == 0 for finding in failures)
    assert not any(finding.check_id == "SCHEMA_GENERATION" and finding.status is DiagnosticStatus.PASS
                   for finding in report.findings)


def test_unknown_string_constructor_remains_uninspected(checkpoints):
    root = checkpoints["clean_act_modern"]
    path = root / "preprocessor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["steps"].append({"registry_name": "custom_uninspected_processor", "config": {}})
    path.write_text(json.dumps(config), encoding="utf-8")

    report = run_preflight(root)

    assert not report.has_failures()
    assert any(finding.check_id == "FEATURE_CONTRACT" and finding.status is DiagnosticStatus.UNKNOWN
               and finding.details.get("step_index") == 1 for finding in report.findings)


def test_valid_pair_does_not_hide_a_malformed_processor_alias(checkpoints):
    root = checkpoints["clean_act_modern"]
    alias = root / "policy_preprocessor.json"
    alias.write_text(json.dumps({"steps": [{"registry_name": []}]}), encoding="utf-8")

    report = run_preflight(root)

    assert report.has_failures()
    assert not any(finding.check_id == "SCHEMA_GENERATION" and finding.status is DiagnosticStatus.PASS
                   for finding in report.findings)


@pytest.mark.parametrize("entry", [
    None,
    {"registry_name": [], "config": {}},
    {"registry_name": "normalizer_processor", "config": []},
])
def test_direct_trace_and_privilege_apis_report_malformed_steps(checkpoints, tmp_path, entry):
    baseline = checkpoints["clean_act_modern"]
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path = candidate / "preprocessor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["steps"][0] = entry
    path.write_text(json.dumps(config), encoding="utf-8")

    trace, = run_offline_trace(candidate, candidate / "sample_observation.npz")
    assert trace.check_id == "OFFLINE_TRACE"
    assert trace.status is DiagnosticStatus.FAIL
    assert trace.details["step_index"] == 0
    assert "Invalid step declaration" in trace.summary

    delta = run_privilege_delta(baseline, candidate)
    assert any(finding.check_id == "PRIVILEGE_OBSERVATION_STAT_DRIFT"
               and finding.status is DiagnosticStatus.FAIL
               and "comparison coverage is unavailable" in finding.summary for finding in delta)
    assert any(finding.check_id == "COMPATIBILITY_STATUS"
               and finding.status is DiagnosticStatus.UNKNOWN for finding in delta)


def test_nonobject_policy_config_returns_findings_across_direct_apis(checkpoints):
    root = checkpoints["clean_act_modern"]
    (root / "config.json").write_text("[]", encoding="utf-8")

    preflight = run_preflight(root)
    trace = run_offline_trace(root, root / "sample_observation.npz")
    delta = run_privilege_delta(root, root)

    assert preflight.has_failures()
    assert any(finding.check_id == "FEATURE_CONTRACT" and finding.status is DiagnosticStatus.FAIL
               and "Policy config must be a JSON object" in finding.summary for finding in preflight.findings)
    assert any(finding.check_id == "INVALID_SAMPLE_SHAPE" and finding.status is DiagnosticStatus.FAIL
               for finding in trace)
    assert any(finding.check_id == "COMPATIBILITY_STATUS" and finding.status is DiagnosticStatus.CONFLICT
               and "Policy config must be a JSON object" in finding.summary for finding in delta)


def test_escaping_policy_config_is_neither_read_nor_hashed(checkpoints, tmp_path, monkeypatch):
    root = checkpoints["clean_act_modern"]
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({
        "input_features": {}, "output_features": {}, "joint_angle_units": "outside-sentinel",
    }), encoding="utf-8")
    (root / "config.json").symlink_to(outside)
    original_open = Path.open

    def contained_open(path, *args, **kwargs):
        if path.resolve() == outside.resolve():
            raise AssertionError("An out-of-checkpoint config must not be opened or hashed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", contained_open)
    report = run_preflight(root)

    assert report.has_failures()
    assert "outside-sentinel" not in json.dumps(report.to_dict())
    policy_failures = [finding for finding in report.findings
                       if finding.status is DiagnosticStatus.FAIL and finding.provenance
                       and finding.provenance.file_path == str(root / "config.json")]
    assert {finding.check_id for finding in policy_failures} == {"FEATURE_CONTRACT", "PHYSICAL_SEMANTICS"}
    assert all(finding.provenance.sha256_prefix == "" for finding in policy_failures)


def test_privilege_error_provenance_never_hashes_escaping_policy_config(checkpoints, tmp_path, monkeypatch):
    root = checkpoints["clean_act_modern"]
    outside = tmp_path / "outside-policy.json"
    outside.write_text(json.dumps({"joint_angle_units": "outside-sentinel"}), encoding="utf-8")
    (root / "config.json").symlink_to(outside)
    original_open = Path.open

    def contained_open(path, *args, **kwargs):
        if path.resolve() == outside.resolve():
            raise AssertionError("Rejected policy-config links must not be opened for error provenance")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", contained_open)
    findings = run_privilege_delta(root, root)

    assert any(finding.check_id == "COMPATIBILITY_STATUS" and finding.status is DiagnosticStatus.CONFLICT
               for finding in findings)
    assert "outside-sentinel" not in json.dumps([finding.to_dict() for finding in findings])
    config_evidence = [finding.provenance for finding in findings if finding.provenance
                       and finding.provenance.file_path == str(root / "config.json")]
    assert config_evidence
    assert all(provenance.sha256_prefix == "" for provenance in config_evidence)


def test_trace_error_provenance_never_hashes_escaping_processor_config(checkpoints, tmp_path, monkeypatch):
    root = checkpoints["clean_act_modern"]
    path = root / "preprocessor_config.json"
    outside = tmp_path / "outside-processor.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    original_open = Path.open

    def contained_open(candidate, *args, **kwargs):
        if candidate.resolve() == outside.resolve():
            raise AssertionError("Rejected processor links must not be opened for error provenance")
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", contained_open)
    finding, = run_offline_trace(root, root / "sample_observation.npz")

    assert finding.check_id == "OFFLINE_TRACE"
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.provenance.file_path == str(path)
    assert finding.provenance.sha256_prefix == ""


def test_explicit_external_metadata_remains_supported(checkpoints, tmp_path):
    root = checkpoints["clean_act_modern"]
    metadata = tmp_path / "physical-metadata.json"
    metadata.write_text(json.dumps({"joint_angle_units": "radians"}), encoding="utf-8")

    report = run_preflight(root, metadata_path=metadata)

    assert not report.has_failures()
    finding, = [finding for finding in report.findings
                if finding.check_id == "PHYSICAL_SEMANTICS"
                and finding.details.get("field") == "joint_angle_units"]
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["value"] == "radians"
    assert finding.provenance.file_path == str(metadata)
    assert finding.provenance.sha256_prefix


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_processor_roles_require_matching_configs(checkpoints, conflicting):
    root = checkpoints["clean_act_modern"]
    original = root / "preprocessor_config.json"
    config = json.loads(original.read_text(encoding="utf-8"))
    alternate = deepcopy(config)
    if conflicting:
        alternate["steps"][0]["config"]["eps"] = 1e-6
    alias = root / "policy_preprocessor.json"
    alias.write_text(json.dumps(alternate), encoding="utf-8")

    report = run_preflight(root)

    schema, = [finding for finding in report.findings if finding.check_id == "SCHEMA_GENERATION"]
    assert report.has_failures() is conflicting
    assert schema.status is (DiagnosticStatus.CONFLICT if conflicting else DiagnosticStatus.PASS)
    if conflicting:
        role, = schema.details["conflicting_roles"]
        assert role["pipeline_role"] == "pre"
        assert set(role["processor_files"]) == {str(original), str(alias)}
        assert len({source["sha256_prefix"] for source in role["sources"]}) == 2


def test_artifact_validation_preserves_low_std_ground_truth(checkpoints):
    report = run_preflight(checkpoints["bug_5289_pathological_gain"])

    finding, = [finding for finding in report.findings
                if finding.check_id == "PATHOLOGICAL_STD_GAIN" and finding.status is DiagnosticStatus.FAIL]
    assert finding.provenance.tensor_key == "observation.state.std"
    assert finding.provenance.violating_indices == [9]
    assert finding.details["effective_gains_with_eps"][0] > 1e5
