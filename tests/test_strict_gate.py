"""Fail-closed comparison coverage and faithful non-finite JSON evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot_preflight import cli, privilege_delta
from lerobot_preflight.models import DiagnosticFinding, DiagnosticStatus, PreflightReport, Severity
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.mark.parametrize("check", ["PRIVILEGE_ACTION_GAIN_ESCALATION", "COMPATIBILITY_STATUS"])
def test_baseline_unknown_required_comparison_fails(check, monkeypatch, tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]
    monkeypatch.setattr(privilege_delta, "run_privilege_delta", lambda *args: [DiagnosticFinding(
        check, DiagnosticStatus.UNKNOWN, Severity.WARNING, "Comparison evidence unavailable.",
    )])
    output = tmp_path / "report.json"
    assert cli.main(["--policy", str(checkpoint), "--baseline", str(checkpoint),
                     "--report", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["has_failures"] is True
    assert report["gate_policy"]["baseline_requested"] is True
    assert any(f["details"].get("required_coverage") for f in report["findings"])


def test_baseline_empty_comparison_cannot_pass(monkeypatch, tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]
    monkeypatch.setattr(privilege_delta, "run_privilege_delta", lambda *args: [])
    assert cli.main(["--policy", str(checkpoint), "--baseline", str(checkpoint)]) == 1


def test_strict_rejects_unknown_while_default_preserves_status(tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]
    assert cli.main(["--policy", str(checkpoint)]) == 0
    assert cli.main(["--policy", str(checkpoint), "--strict"]) == 1


def test_round4_strict_clean_native_checkpoint_does_not_require_custom_physical_fields(tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]
    (checkpoint / "config.json").write_text(json.dumps({
        "type": "act", "n_obs_steps": 1, "n_action_steps": 50, "chunk_size": 50,
        "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    }))
    report = cli.run_doctor(checkpoint, strict=True)
    semantics = [f for f in report.findings if f.check_id == "PHYSICAL_SEMANTICS"]
    assert len(semantics) == 3
    assert all(f.status is DiagnosticStatus.UNKNOWN and f.severity is Severity.INFO
               and f.details.get("strict_exempt") for f in semantics)
    assert report.has_failures() is False
    assert cli.main(["--policy", str(checkpoint), "--strict"]) == 0


def test_round4_explicit_metadata_flag_controls_temporal_probe(tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["bug_5315_temporal_anchor"]
    without = cli.run_doctor(checkpoint)
    assert not any(f.check_id == "TEMPORAL_ANCHOR_MISMATCH" and f.status is DiagnosticStatus.FAIL
                   for f in without.findings)
    output = tmp_path / "temporal.json"
    assert cli.main(["--policy", str(checkpoint), "--metadata", str(checkpoint / "metadata.json"),
                     "--report", str(output)]) == 1
    document = json.loads(output.read_text())
    assert any(f["check_id"] == "TEMPORAL_ANCHOR_MISMATCH" and f["status"] == "FAIL"
               for f in document["findings"])


def test_round4_informational_exemption_never_hides_required_coverage_or_warnings():
    for severity, details in (
        (Severity.INFO, {"strict_exempt": True, "required_coverage": True}),
        (Severity.WARNING, {"strict_exempt": True}),
    ):
        finding = DiagnosticFinding("COVERAGE", DiagnosticStatus.UNKNOWN, severity,
                                    "Unavailable evidence.", details)
        assert PreflightReport("checkpoint", [finding], strict=True, baseline_requested=True).has_failures()


def test_missing_real_action_buffers_fail_baseline_closed(tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]
    path = checkpoint / "postprocessor_config.json"
    config = json.loads(path.read_text())
    for entry in config["steps"]:
        entry.pop("state_file", None)
    path.write_text(json.dumps(config))
    result = cli.run_doctor(checkpoint, baseline=checkpoint)
    assert result.has_failures()
    assert any(f.status is DiagnosticStatus.UNKNOWN and f.details.get("required_coverage")
               for f in result.findings)


def test_non_finite_ieee_evidence_survives_strict_json(tmp_path):
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["bug_4973_fp16_nan"]
    report = cli.run_doctor(checkpoint, sample=checkpoint / "sample_observation.npz")
    runtime = next(f for f in report.findings if f.check_id == "RUNTIME_NON_FINITE")
    data = runtime.to_dict()
    assert data["details"]["tensor_slice"] == ["NaN"]
    assert data["non_finite_values"]
    assert any(item["ieee754"] == "NaN" for item in data["non_finite_values"])
    explicit = DiagnosticFinding("NONFINITE", DiagnosticStatus.FAIL, Severity.CRITICAL, "Bad stats.",
                                 {"array/key": np.array([np.nan, np.inf, -np.inf]), "literal": "NaN"})
    encoded = json.loads(json.dumps(explicit.to_dict(), allow_nan=False))
    assert encoded["non_finite_values"] == [
        {"path": "/details/array~1key/0", "ieee754": "NaN"},
        {"path": "/details/array~1key/1", "ieee754": "Infinity"},
        {"path": "/details/array~1key/2", "ieee754": "-Infinity"},
    ]


def test_gate_unknown_does_not_change_finding_status():
    finding = DiagnosticFinding("COVERAGE", DiagnosticStatus.UNKNOWN, Severity.WARNING,
                                "Missing stats.", {"required_coverage": True})
    report = PreflightReport("checkpoint", [finding], baseline_requested=True)
    assert report.has_failures()
    assert report.to_dict()["findings"][0]["status"] == "UNKNOWN"


@pytest.mark.parametrize("check", ["FP16_NON_FINITE_RISK", "CAMERA_KEY_CONTRACT_MISMATCH",
                                  "INLINE_STAT_OVERRIDE_ADVISORY"])
def test_round3_pass_warning_advisory_only_fails_strict(check, monkeypatch, tmp_path, capsys):
    finding = DiagnosticFinding(check, DiagnosticStatus.PASS, Severity.WARNING,
                                "Conditional risk; the declared float32 runtime is verified finite.",
                                {"non_blocking_advisory": True})
    # Isolate advisory exit semantics from unrelated physical UNKNOWNs/schema failures.
    monkeypatch.setattr(cli, "run_preflight", lambda *args, **kwargs: PreflightReport(str(tmp_path), [finding]))
    monkeypatch.setattr(cli, "run_temporal_audit", lambda *args, **kwargs: [])
    standard = tmp_path / "standard.json"
    strict = tmp_path / "strict.json"
    assert cli.main(["--policy", str(tmp_path), "--report", str(standard)]) == 0
    assert cli.main(["--policy", str(tmp_path), "--strict", "--report", str(strict)]) == 1
    assert "(WARNING)" in capsys.readouterr().out
    assert json.loads(standard.read_text())["has_failures"] is False
    strict_report = json.loads(strict.read_text())
    assert strict_report["has_failures"] is True
    assert strict_report["gate_policy"]["warning_advisories_fail"] is True
    assert strict_report["findings"][0]["status"] == "PASS"


def test_round3_warning_severity_does_not_hide_a_blocking_failure():
    finding = DiagnosticFinding("DEAD_ACTION_DIMENSION", DiagnosticStatus.FAIL, Severity.WARNING,
                                "A declared action coordinate is frozen.")
    assert PreflightReport("checkpoint", [finding]).has_failures()
    assert PreflightReport("checkpoint", [finding], strict=True).has_failures()
