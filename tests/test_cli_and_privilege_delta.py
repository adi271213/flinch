"""Checkpoint privilege comparisons and preflight CLI integration."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DataProcessorPipeline,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    UnnormalizerProcessorStep,
)
from safetensors.torch import load_file, save_file

from lerobot_flinch.cli import main, run_checks
from lerobot_flinch.models import DiagnosticStatus, PreflightReport
from lerobot_flinch.privilege_delta import run_privilege_delta
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path / "fixtures")


def _find(findings, check_id: str, status: DiagnosticStatus | None = None):
    return [
        finding
        for finding in findings
        if finding.check_id == check_id and (status is None or finding.status is status)
    ]


def _post_state(checkpoint: Path) -> Path:
    config = json.loads((checkpoint / "postprocessor_config.json").read_text())
    step = next(entry for entry in config["steps"] if entry.get("registry_name") == "unnormalizer_processor")
    return checkpoint / step["state_file"]


def _pre_state(checkpoint: Path) -> Path:
    config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    step = next(entry for entry in config["steps"] if entry.get("registry_name") == "normalizer_processor")
    return checkpoint / step["state_file"]


def _rewrite_std(path: Path, indices: list[int], value: float) -> None:
    tensors = load_file(path)
    updated = tensors["action.std"].clone()
    updated[indices] = value
    tensors["action.std"] = updated
    save_file(tensors, path)


def _write_policy_config(checkpoint: Path, *, n_action_steps: int, chunk_size: int = 100) -> None:
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "n_obs_steps": 1,
                "n_action_steps": n_action_steps,
                "chunk_size": chunk_size,
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [14]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [14]}},
            }
        )
    )


@pytest.fixture
def comparison_pair(tmp_path: Path, checkpoints: dict[str, Path]) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    shutil.copytree(checkpoints["clean_act_modern"], baseline)
    shutil.copytree(checkpoints["clean_act_modern"], candidate)
    _write_policy_config(baseline, n_action_steps=50)
    _write_policy_config(candidate, n_action_steps=50)
    return baseline, candidate


def test_same_clean_checkpoint_has_no_privilege_escalation(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    findings = run_privilege_delta(checkpoint, checkpoint)
    assert findings
    assert all(
        finding.status not in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT} for finding in findings
    )
    assert _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.PASS)
    assert _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.PASS)
    assert _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.PASS)


def test_action_scale_and_executed_horizon_expansion_are_both_failures(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _rewrite_std(_post_state(candidate), [2, 6], 0.75)
    _write_policy_config(candidate, n_action_steps=100)

    findings = run_privilege_delta(baseline, candidate)
    gain_failures = _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)
    horizon_failures = _find(findings, "PRIVILEGE_HORIZON_ESCALATION", DiagnosticStatus.FAIL)
    assert len(gain_failures) == len(horizon_failures) == 1

    gain = gain_failures[0]
    assert gain.details["feature_key"] == "action"
    assert gain.details["step_index"] == 0
    assert gain.details["baseline_scale"] == pytest.approx([0.5, 0.5])
    assert gain.details["candidate_scale"] == pytest.approx([0.75, 0.75])
    assert gain.details["expansion_ratio"] == pytest.approx([1.5, 1.5])
    assert gain.details["tensor_slice"] == pytest.approx([0.75, 0.75])
    assert gain.provenance.violating_indices == [2, 6]
    assert gain.provenance.tensor_key == "action.std"
    assert Path(gain.provenance.file_path) == _post_state(candidate)
    assert (
        gain.provenance.sha256_prefix == hashlib.sha256(_post_state(candidate).read_bytes()).hexdigest()[:12]
    )
    baseline_provenance = gain.details["baseline_provenance"]
    assert Path(baseline_provenance["file_path"]) == _post_state(baseline)
    assert baseline_provenance["tensor_key"] == "action.std"
    assert (
        baseline_provenance["sha256_prefix"]
        == hashlib.sha256(_post_state(baseline).read_bytes()).hexdigest()[:12]
    )

    horizon = horizon_failures[0]
    assert Path(horizon.provenance.file_path) == candidate / "config.json"
    assert horizon.provenance.tensor_key == "n_action_steps"
    assert (
        horizon.provenance.sha256_prefix
        == hashlib.sha256((candidate / "config.json").read_bytes()).hexdigest()[:12]
    )
    assert horizon.details["baseline_n_action_steps"] == 50
    assert horizon.details["candidate_n_action_steps"] == 100
    assert horizon.details["baseline_chunk_size"] == 100
    assert horizon.details["candidate_chunk_size"] == 100
    baseline_horizon_provenance = horizon.details["baseline_provenance"]
    assert Path(baseline_horizon_provenance["file_path"]) == baseline / "config.json"
    assert baseline_horizon_provenance["tensor_key"] == "n_action_steps"


def test_scale_threshold_suppresses_subthreshold_expansion(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _rewrite_std(_post_state(candidate), [2, 6], 0.75)
    assert _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL
    )
    assert not _find(
        run_privilege_delta(baseline, candidate, scale_expansion_ratio=2.0),
        "PRIVILEGE_ACTION_GAIN_ESCALATION",
        DiagnosticStatus.FAIL,
    )


def test_missing_horizon_metadata_is_unknown_not_escalation(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    findings = run_privilege_delta(checkpoint, checkpoint)
    assert _find(findings, "PRIVILEGE_HORIZON_ESCALATION", DiagnosticStatus.UNKNOWN)
    assert not _find(findings, "PRIVILEGE_HORIZON_ESCALATION", DiagnosticStatus.FAIL)


def test_chunk_size_increase_alone_is_not_executed_horizon_escalation(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _write_policy_config(candidate, n_action_steps=50, chunk_size=200)
    findings = run_privilege_delta(baseline, candidate)
    assert not _find(findings, "PRIVILEGE_HORIZON_ESCALATION", DiagnosticStatus.FAIL)


def test_clean_to_4415_bypass_is_a_privilege_guard_removal(
    checkpoints: dict[str, Path],
) -> None:
    baseline = checkpoints["clean_act_modern"]
    candidate = checkpoints["bug_4415_silent_bypass"]
    findings = run_privilege_delta(baseline, candidate)
    failures = _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
    assert any(finding.details["feature_key"] == "observation.state" for finding in failures)
    assert any(finding.details["step_index"] == 0 for finding in failures)
    assert any(Path(finding.provenance.file_path) == _pre_state(candidate) for finding in failures)


def test_identity_mode_removes_observation_guard(comparison_pair: tuple[Path, Path]) -> None:
    baseline, candidate = comparison_pair
    config_path = candidate / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    config["steps"][0]["config"]["norm_map"]["STATE"] = "IDENTITY"
    config_path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    assert any(
        finding.details["feature_key"] == "observation.state"
        for finding in _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
    )


def test_postprocessor_stat_key_bypass_removes_action_guard(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    state_path = _post_state(candidate)
    tensors = load_file(state_path)
    save_file(
        {key.replace("action.", "wrong_dataset/action."): value for key, value in tensors.items()},
        state_path,
    )
    findings = run_privilege_delta(baseline, candidate)
    assert any(
        finding.details["feature_key"] == "action"
        for finding in _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
    )


def test_serialized_clipping_guard_removal_is_found_without_loading_its_constructor(
    comparison_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline, candidate = comparison_pair
    post_config_path = baseline / "postprocessor_config.json"
    config = json.loads(post_config_path.read_text())
    # The inspected native vla_jepa clip step clamps to [-1, 1] before unnormalization.
    # Insert only its serialized declaration; the preflight must not import its policy module.
    config["steps"].insert(0, {"registry_name": "vla_jepa_clip_actions", "config": {}})
    post_config_path.write_text(json.dumps(config))

    def forbid_socket(*args, **kwargs):
        pytest.fail("clipping guard comparison attempted network access")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    findings = run_privilege_delta(baseline, candidate)
    guard_failures = _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
    assert any(
        finding.details["feature_key"] == "action"
        and finding.details["pipeline_role"] == "postprocessor"
        and finding.details["baseline_step_index"] == 0
        and finding.details["reason"] == "clipping_step_removed_or_disabled"
        for finding in guard_failures
    )
    assert _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.UNKNOWN)


def _write_minmax_postprocessor(checkpoint: Path, *, lower: float, upper: float) -> None:
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    postprocessor = PolicyProcessorPipeline(
        name="policy_postprocessor",
        steps=[
            UnnormalizerProcessorStep(
                features=features,
                norm_map={FeatureType.ACTION: NormalizationMode.MIN_MAX},
                stats={
                    "action": {
                        "min": torch.full((14,), lower),
                        "max": torch.full((14,), upper),
                    }
                },
            )
        ],
    )
    postprocessor.save_pretrained(checkpoint, config_filename="postprocessor_config.json")


def test_minmax_inverse_span_expansion_is_compared(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _write_minmax_postprocessor(baseline, lower=-0.5, upper=0.5)
    _write_minmax_postprocessor(candidate, lower=-0.75, upper=0.75)
    findings = run_privilege_delta(baseline, candidate)
    failures = _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)
    assert failures
    assert failures[0].provenance.tensor_key in {"action.min", "action.max"}
    assert all(value == pytest.approx(1.5) for value in failures[0].details["expansion_ratio"])


def test_zero_inverse_scale_to_positive_is_escalation(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _rewrite_std(_post_state(baseline), [3], 0.0)
    findings = run_privilege_delta(baseline, candidate)
    failures = _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)
    assert failures
    assert 3 in failures[0].provenance.violating_indices


@pytest.mark.parametrize("change", ["mode", "dimension", "feature_key"])
def test_incompatible_action_contract_does_not_claim_numeric_comparison(
    comparison_pair: tuple[Path, Path], change: str
) -> None:
    baseline, candidate = comparison_pair
    config_path = candidate / "postprocessor_config.json"
    config = json.loads(config_path.read_text())
    step_config = config["steps"][0]["config"]
    if change == "mode":
        step_config["norm_map"]["ACTION"] = "MIN_MAX"
    elif change == "dimension":
        step_config["features"]["action"]["shape"] = [7]
    else:
        step_config["features"]["alternate_action"] = step_config["features"].pop("action")
    config_path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    compatibility = _find(findings, "COMPATIBILITY_STATUS")
    assert any(
        finding.status in {DiagnosticStatus.CONFLICT, DiagnosticStatus.UNKNOWN} for finding in compatibility
    )
    assert not _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)


def test_candidate_json_stats_disagree_with_saved_tensor_source(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    config_path = candidate / "postprocessor_config.json"
    config = json.loads(config_path.read_text())
    config["steps"][0]["config"]["stats"] = {"action": {"mean": [0.0] * 14, "std": [0.9] * 14}}
    config_path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    advisory = _find(findings, "INLINE_STAT_OVERRIDE_ADVISORY", DiagnosticStatus.PASS)
    assert advisory and all(f.severity.value == "WARNING" for f in advisory)
    assert not _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.CONFLICT)
    assert _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)


def test_policy_normalization_mapping_disagreeing_with_native_postprocessor_is_conflict(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, action_mode in ((baseline, "MEAN_STD"), (candidate, "IDENTITY")):
        config_path = checkpoint / "config.json"
        config = json.loads(config_path.read_text())
        config["normalization_mapping"] = {"STATE": "MEAN_STD", "ACTION": action_mode}
        config_path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    assert _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.CONFLICT)
    assert not _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)


def test_run_checks_aggregates_static_trace_temporal_and_baseline_checks(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _rewrite_std(_post_state(candidate), [2, 6], 0.75)
    _write_policy_config(candidate, n_action_steps=100)
    report = run_checks(
        candidate,
        sample=candidate / "sample_observation.npz",
        baseline=baseline,
    )
    assert isinstance(report, PreflightReport)
    assert report.has_failures()
    assert _find(report.findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)
    assert _find(report.findings, "PRIVILEGE_HORIZON_ESCALATION", DiagnosticStatus.FAIL)
    assert _find(report.findings, "TEMPORAL_ANCHOR_MISMATCH")
    assert _find(report.findings, "SILENT_STEP_BYPASS")
    json.dumps(report.to_dict(), allow_nan=False)


def test_runner_records_unexpected_stage_error_and_continues_to_temporal(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lerobot_flinch.cli as cli_module

    checkpoint = checkpoints["clean_act_modern"]

    def injected_preflight_error(*args, **kwargs):
        raise RuntimeError("injected preflight failure")

    monkeypatch.setattr(cli_module, "run_preflight", injected_preflight_error)
    report = run_checks(checkpoint)
    runner_failures = _find(report.findings, "RUNNER_ERROR", DiagnosticStatus.FAIL)
    assert len(runner_failures) == 1
    assert runner_failures[0].details["stage"] == "preflight"
    assert _find(report.findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.PASS)
    assert report.has_failures()

    report_path = tmp_path / "runner_error.json"
    assert main(["--policy", str(checkpoint), "--report", str(report_path)]) == 1
    saved = json.loads(report_path.read_text())
    assert any(finding["check_id"] == "RUNNER_ERROR" for finding in saved["findings"])


def test_cli_writes_strict_json_and_returns_zero_for_clean_checkpoint(
    checkpoints: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    report_path = tmp_path / "clean_report.json"
    exit_code = main(
        [
            "--policy",
            str(checkpoint),
            "--sample",
            str(checkpoint / "sample_observation.npz"),
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["has_failures"] is False
    assert isinstance(report["findings"], list)
    assert report["checkpoint_dir"] == str(checkpoint.resolve())
    json.dumps(report, allow_nan=False)
    assert capsys.readouterr().out


@pytest.mark.parametrize(
    ("fixture_name", "expected_check_id"),
    [
        ("bug_4415_silent_bypass", "SILENT_STEP_BYPASS"),
        ("bug_5289_pathological_gain", "EMPIRICAL_GAIN_EXPLOSION"),
        ("bug_5315_temporal_anchor", "TEMPORAL_ANCHOR_MISMATCH"),
    ],
)
def test_cli_returns_one_and_reports_each_ground_truth_bug(
    checkpoints: dict[str, Path], tmp_path: Path, fixture_name: str, expected_check_id: str
) -> None:
    checkpoint = checkpoints[fixture_name]
    report_path = tmp_path / f"{fixture_name}.json"
    metadata_args = (
        ["--metadata", str(checkpoint / "metadata.json")]
        if fixture_name == "bug_5315_temporal_anchor"
        else []
    )
    exit_code = main(
        [
            "--policy",
            str(checkpoint),
            "--sample",
            str(checkpoint / "sample_observation.npz"),
            "--report",
            str(report_path),
            *metadata_args,
        ]
    )
    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["has_failures"] is True
    assert any(
        finding["check_id"] == expected_check_id and finding["status"] == "FAIL"
        for finding in report["findings"]
    )
    json.dumps(report, allow_nan=False)


def test_cli_fp16_nonfinite_report_is_strict_json_without_numeric_nan(
    checkpoints: dict[str, Path], tmp_path: Path
) -> None:
    checkpoint = checkpoints["bug_4973_fp16_nan"]
    report_path = tmp_path / "fp16_report.json"
    exit_code = main(
        [
            "--policy",
            str(checkpoint),
            "--sample",
            str(checkpoint / "sample_observation.npz"),
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 1

    def reject_nonstandard_constant(value: str):
        pytest.fail(f"JSON contained a numeric nonfinite literal: {value}")

    report = json.loads(report_path.read_text(), parse_constant=reject_nonstandard_constant)
    runtime = [
        finding
        for finding in report["findings"]
        if finding["check_id"] == "RUNTIME_NON_FINITE" and finding["status"] == "FAIL"
    ]
    assert runtime
    assert runtime[0]["details"]["tensor_slice"] == ["NaN"]


def test_cli_human_output_identifies_step_dimension_and_sha(
    checkpoints: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    exit_code = main(["--policy", str(checkpoint), "--sample", str(checkpoint / "sample_observation.npz")])
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "PATHOLOGICAL_STD_GAIN" in output
    assert "step" in output.lower()
    assert "9" in output
    assert hashlib.sha256(_pre_state(checkpoint).read_bytes()).hexdigest()[:12] in output


def test_cli_baseline_option_includes_privilege_failures_in_json(
    comparison_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    baseline, candidate = comparison_pair
    _rewrite_std(_post_state(candidate), [2, 6], 0.75)
    _write_policy_config(candidate, n_action_steps=100)
    report_path = tmp_path / "privilege_report.json"
    exit_code = main(["--policy", str(candidate), "--baseline", str(baseline), "--report", str(report_path)])
    assert exit_code == 1
    report = json.loads(report_path.read_text())
    failures = {finding["check_id"] for finding in report["findings"] if finding["status"] == "FAIL"}
    assert {"PRIVILEGE_ACTION_GAIN_ESCALATION", "PRIVILEGE_HORIZON_ESCALATION"} <= failures


def test_cli_refuses_to_overwrite_an_audited_artifact(
    checkpoints: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    artifact = checkpoint / "preprocessor_config.json"
    original = artifact.read_bytes()
    exit_code = main(["--policy", str(checkpoint), "--report", str(artifact)])
    assert exit_code == 1
    assert artifact.read_bytes() == original
    assert "Refusing to overwrite an audited artifact" in capsys.readouterr().err


def test_privilege_delta_and_run_checks_do_not_open_network_or_mutate_artifacts(
    comparison_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline, candidate = comparison_pair
    _rewrite_std(_post_state(candidate), [2, 6], 0.75)

    def snapshot(checkpoint: Path) -> dict[str, bytes]:
        return {
            path.relative_to(checkpoint).as_posix(): path.read_bytes()
            for path in checkpoint.rglob("*")
            if path.is_file()
        }

    before_baseline, before_candidate = snapshot(baseline), snapshot(candidate)

    def forbid_socket(*args, **kwargs):
        pytest.fail("preflight attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    run_privilege_delta(baseline, candidate)
    run_checks(candidate, baseline=baseline)
    assert snapshot(baseline) == before_baseline
    assert snapshot(candidate) == before_candidate


def _add_serialized_vla_language_steps(checkpoint: Path, family: str = "smolvla") -> None:
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    prompt = {
        "smolvla": "smolvla_new_line_processor",
        "pi0": "pi0_new_line_processor",
        "pi05": "pi05_prepare_state_tokenizer_processor_step",
    }[family]
    config["steps"][:0] = [
        {"registry_name": "to_batch_processor", "config": {}},
        {"registry_name": prompt, "config": {}},
        {
            "registry_name": "tokenizer_processor",
            "config": {"tokenizer_name": "must-not-download-for-static-privilege", "max_length": 48},
        },
    ]
    path.write_text(json.dumps(config))


@pytest.mark.parametrize("family", ["smolvla", "pi0", "pi05"])
def test_vla_static_action_and_horizon_checks_never_construct_language_or_pipeline(
    comparison_pair: tuple[Path, Path],
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint in (baseline, candidate):
        _add_serialized_vla_language_steps(checkpoint, family)
    _rewrite_std(_post_state(candidate), [2, 6], 0.75)
    _write_policy_config(candidate, n_action_steps=100)

    def forbidden(*args, **kwargs):
        pytest.fail("static privilege comparison constructed a pipeline/tokenizer or opened the network")

    from lerobot.processor import TokenizerProcessorStep

    monkeypatch.setattr(TokenizerProcessorStep, "__post_init__", forbidden)
    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", forbidden)
    monkeypatch.setattr(DataProcessorPipeline, "from_config", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    findings = run_privilege_delta(baseline, candidate)
    gains = _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)
    assert len(gains) == 1 and gains[0].provenance.violating_indices == [2, 6]
    assert gains[0].details["processor_execution_required"] is False
    assert _find(findings, "PRIVILEGE_HORIZON_ESCALATION", DiagnosticStatus.FAIL)
    assert _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.PASS)
    assert not any("uninspected" in finding.summary.lower() for finding in findings)


def test_postprocessor_scale_comparison_survives_unreadable_preprocessor_buffers(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _add_serialized_vla_language_steps(baseline)
    _add_serialized_vla_language_steps(candidate)
    _rewrite_std(_post_state(candidate), [2, 6], 0.75)
    _pre_state(candidate).unlink()
    findings = run_privilege_delta(baseline, candidate)
    assert _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.UNKNOWN)
    assert _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)


def test_language_step_cannot_hide_removed_normalization_guard(
    comparison_pair: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, candidate = comparison_pair
    _add_serialized_vla_language_steps(baseline)
    _add_serialized_vla_language_steps(candidate)
    path = candidate / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"] = [
        step for step in config["steps"] if step.get("registry_name") != "normalizer_processor"
    ]
    path.write_text(json.dumps(config))

    def forbidden(*args, **kwargs):
        pytest.fail("static guard comparison constructed a pipeline/tokenizer or opened the network")

    from lerobot.processor import TokenizerProcessorStep

    monkeypatch.setattr(TokenizerProcessorStep, "__post_init__", forbidden)
    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", forbidden)
    monkeypatch.setattr(DataProcessorPipeline, "from_config", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    findings = run_privilege_delta(baseline, candidate)
    failures = _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
    assert any(finding.details["feature_key"] == "observation.state" for finding in failures)
    assert not _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.UNKNOWN)


@pytest.mark.parametrize(
    "mode,bounds",
    [
        (NormalizationMode.QUANTILES, ("q01", "q99")),
        (NormalizationMode.QUANTILE10, ("q10", "q90")),
    ],
)
def test_static_quantile_action_envelope_expansion(
    comparison_pair: tuple[Path, Path],
    mode: NormalizationMode,
    bounds: tuple[str, str],
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, span in ((baseline, 1.0), (candidate, 1.5)):
        pipeline = PolicyProcessorPipeline(
            name="policy_postprocessor",
            steps=[
                UnnormalizerProcessorStep(
                    features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
                    norm_map={FeatureType.ACTION: mode},
                    stats={
                        "action": {
                            bounds[0]: torch.full((14,), -span / 2),
                            bounds[1]: torch.full((14,), span / 2),
                        }
                    },
                )
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    gains = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL
    )
    assert gains and gains[0].details["expansion_ratio"] == pytest.approx([1.5] * 14)
    assert gains[0].provenance.tensor_key == f"action.{bounds[1]}"


def test_static_legacy_action_buffers_keep_original_key_and_payload_provenance(
    checkpoints: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, candidate = tmp_path / "legacy_baseline", tmp_path / "legacy_candidate"
    for checkpoint in (baseline, candidate):
        shutil.copytree(checkpoints["legacy_act_embedded"], checkpoint)
        _write_policy_config(checkpoint, n_action_steps=50)
        path = checkpoint / "config.json"
        config = json.loads(path.read_text())
        config["normalization_mapping"] = {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
        path.write_text(json.dumps(config))
    model = candidate / "model.safetensors"
    state = load_file(model)
    key = "unnormalize_outputs.buffer_action.std"
    state[key][[2, 6]] = 0.75
    save_file(state, model)

    def forbidden(*args, **kwargs):
        pytest.fail("legacy comparison constructed a pipeline or attempted network access")

    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    findings = run_privilege_delta(baseline, candidate)
    gains = _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)
    assert gains and gains[0].details["step_index"] is None
    assert gains[0].provenance.tensor_key == key
    assert gains[0].provenance.violating_indices == [2, 6]
    assert gains[0].details["provenance_digest_scope"] == "tensor_payload"
    assert Path(gains[0].provenance.file_path) == model
    assert (
        gains[0].provenance.sha256_prefix
        == hashlib.sha256(state[key].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]
    )


@pytest.mark.parametrize(
    ("field", "candidate_value"),
    [
        ("input_shapes", {"observation.state": [13]}),
        ("input_normalization_modes", {"observation.state": "MIN_MAX"}),
    ],
)
def test_legacy_input_contract_change_is_conflict_even_with_identical_action_buffers(
    checkpoints: dict[str, Path],
    tmp_path: Path,
    field: str,
    candidate_value: dict,
) -> None:
    baseline, candidate = tmp_path / "legacy_baseline", tmp_path / "legacy_candidate"
    config = {
        "type": "act",
        "n_action_steps": 50,
        "chunk_size": 100,
        "input_shapes": {"observation.state": [14]},
        "output_shapes": {"action": [14]},
        "input_normalization_modes": {"observation.state": "MEAN_STD"},
        "output_normalization_modes": {"action": "MEAN_STD"},
    }
    for checkpoint in (baseline, candidate):
        shutil.copytree(checkpoints["legacy_act_embedded"], checkpoint)
        (checkpoint / "config.json").write_text(json.dumps(config))
    config[field] = candidate_value
    (candidate / "config.json").write_text(json.dumps(config))

    findings = run_privilege_delta(baseline, candidate)
    conflicts = _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.CONFLICT)
    assert any(finding.details.get("field") == field for finding in conflicts)
    assert not _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.PASS)
    assert next(f for f in conflicts if f.details.get("field") == field).provenance.tensor_key == field


def test_action_mean_translation_fails_with_unchanged_std(comparison_pair: tuple[Path, Path]) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, mean in ((baseline, 0.0), (candidate, 1000.0)):
        path = _post_state(checkpoint)
        tensors = load_file(path)
        tensors["action.mean"] = torch.full((14,), mean)
        tensors["action.std"] = torch.ones(14)
        save_file(tensors, path)
    findings = run_privilege_delta(baseline, candidate)
    offsets = _find(findings, "PRIVILEGE_ACTION_OFFSET_ESCALATION", DiagnosticStatus.FAIL)
    envelopes = _find(findings, "PRIVILEGE_ACTION_ENVELOPE_SHIFT", DiagnosticStatus.FAIL)
    assert len(offsets) == len(envelopes) == 1
    assert offsets[0].severity.value == "CRITICAL"
    assert offsets[0].provenance.tensor_key == "action.mean"
    assert offsets[0].provenance.violating_indices == list(range(14))
    assert offsets[0].details["absolute_origin_shift"] == [1000.0] * 14
    assert offsets[0].details["relative_origin_shift"] == [1000.0] * 14
    assert envelopes[0].details["baseline_lower"] == [-3.0] * 14
    assert envelopes[0].details["candidate_lower"] == [997.0] * 14
    assert envelopes[0].details["candidate_upper"] == [1003.0] * 14
    assert not _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)


@pytest.mark.parametrize(
    "mode,names",
    [
        (NormalizationMode.MIN_MAX, ("min", "max")),
        (NormalizationMode.QUANTILES, ("q01", "q99")),
        (NormalizationMode.QUANTILE10, ("q10", "q90")),
    ],
)
def test_action_interval_translation_fails_with_unchanged_span(
    comparison_pair: tuple[Path, Path],
    mode: NormalizationMode,
    names: tuple[str, str],
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, low, high in ((baseline, -1.0, 1.0), (candidate, 999.0, 1001.0)):
        pipeline = PolicyProcessorPipeline(
            name="policy_postprocessor",
            steps=[
                UnnormalizerProcessorStep(
                    features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
                    norm_map={FeatureType.ACTION: mode},
                    stats={"action": {names[0]: torch.full((14,), low), names[1]: torch.full((14,), high)}},
                )
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    findings = run_privilege_delta(baseline, candidate)
    envelope = _find(findings, "PRIVILEGE_ACTION_ENVELOPE_SHIFT", DiagnosticStatus.FAIL)[0]
    assert envelope.provenance.violating_indices == list(range(14))
    assert envelope.provenance.tensor_key == f"action.{names[1]}"
    assert envelope.details["tensor_slice"] == [1001.0] * 14
    assert envelope.details["bound_side"] == "upper"
    assert envelope.details["baseline_lower"] == [-1.0] * 14
    assert envelope.details["baseline_upper"] == [1.0] * 14
    assert envelope.details["candidate_lower"] == [999.0] * 14
    assert envelope.details["candidate_upper"] == [1001.0] * 14
    assert not _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)


def test_ordered_guard_comparison_detects_reordered_native_steps(comparison_pair: tuple[Path, Path]) -> None:
    baseline, candidate = comparison_pair
    for checkpoint in (baseline, candidate):
        path = checkpoint / "postprocessor_config.json"
        config = json.loads(path.read_text())
        config["steps"].append({"registry_name": "device_processor", "config": {"device": "cpu"}})
        if checkpoint == candidate:
            config["steps"].reverse()
        path.write_text(json.dumps(config))
    failures = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL
    )
    assert any(f.details.get("reason") == "processor_step_order_changed" for f in failures)


@pytest.mark.parametrize(
    "bounds",
    [
        ({"minimum": -5.0, "maximum": 5.0}, {"minimum": -10.0, "maximum": 5.0}),
        ({"min": -1.0, "max": 1.0}, {"min": -1.0, "max": 2.0}),
        ({"clip": [-1.0, 1.0]}, {"clip": [999.0, 1001.0]}),
        ({"clip": 1.0}, {"clip": 2.0}),
    ],
)
def test_declared_clamp_expansion_is_flagged_without_constructing_step(
    comparison_pair: tuple[Path, Path],
    bounds: tuple[dict, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, config_bounds in zip((baseline, candidate), bounds):
        path = checkpoint / "postprocessor_config.json"
        config = json.loads(path.read_text())
        config["steps"].append({"registry_name": "custom_declared_action_clamp", "config": config_bounds})
        path.write_text(json.dumps(config))
    monkeypatch.setattr(
        DataProcessorPipeline, "from_config", lambda *a, **kw: pytest.fail("constructed a pipeline")
    )
    failures = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL
    )
    assert any(f.details.get("reason") == "clipping_bounds_widened" for f in failures)


def test_required_comparison_unknowns_are_explicitly_gate_blocking(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    _post_state(candidate).unlink()
    unknowns = [f for f in run_privilege_delta(baseline, candidate) if f.status == DiagnosticStatus.UNKNOWN]
    assert unknowns
    for finding in unknowns:
        assert finding.details["required_coverage"] is (finding.details.get("scope") != "physical_semantics")
    assert any(
        f.check_id == "PRIVILEGE_ACTION_GAIN_ESCALATION" and f.details["required_coverage"] for f in unknowns
    )


def test_composed_affine_origin_matches_native_unnormalizers(comparison_pair: tuple[Path, Path]) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, first_mean in ((baseline, 1.0), (candidate, 1000.0)):
        features = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
        pipeline = PolicyProcessorPipeline(
            name="policy_postprocessor",
            steps=[
                UnnormalizerProcessorStep(
                    features=features,
                    norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                    stats={"action": {"mean": torch.full((14,), first_mean), "std": torch.full((14,), 2.0)}},
                ),
                UnnormalizerProcessorStep(
                    features=features,
                    norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                    stats={"action": {"mean": torch.full((14,), 4.0), "std": torch.full((14,), 3.0)}},
                ),
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
        native_zero = pipeline({"action": torch.zeros(14)})["action"]
        assert native_zero.tolist() == [3 * first_mean + 4] * 14
    finding = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_ACTION_OFFSET_ESCALATION", DiagnosticStatus.FAIL
    )[0]
    assert finding.details["baseline_origin"] == [7.0] * 14
    assert finding.details["candidate_origin"] == [3004.0] * 14
    assert finding.provenance.tensor_key == "action.mean"
    assert finding.details["tensor_slice"] == [4.0] * 14


@pytest.mark.parametrize("threshold", [-1, float("inf"), float("nan")])
def test_offset_threshold_must_be_finite_nonnegative(
    comparison_pair: tuple[Path, Path], threshold: float
) -> None:
    with pytest.raises(ValueError, match="offset_shift_threshold"):
        run_privilege_delta(*comparison_pair, offset_shift_threshold=threshold)


def test_reordered_duplicate_normalizers_are_detected_from_stat_payloads(
    comparison_pair: tuple[Path, Path],
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint in (baseline, candidate):
        pipeline = PolicyProcessorPipeline(
            name="policy_postprocessor",
            steps=[
                UnnormalizerProcessorStep(
                    features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
                    norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                    stats={"action": {"mean": torch.full((14,), origin), "std": torch.ones(14)}},
                )
                for origin in (1.0, 2.0)
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    path = candidate / "postprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].reverse()
    path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    # Pure translations commute: the full final affine mapping is unchanged,
    # but artifact order drift must still be reported explicitly.
    assert not _find(findings, "PRIVILEGE_ACTION_OFFSET_ESCALATION", DiagnosticStatus.FAIL)
    assert any(
        f.details.get("reason") == "processor_step_order_changed"
        for f in _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
    )


def test_clamp_reordered_across_unnormalizer_is_a_guard_failure(comparison_pair: tuple[Path, Path]) -> None:
    baseline, candidate = comparison_pair
    for checkpoint in (baseline, candidate):
        path = checkpoint / "postprocessor_config.json"
        config = json.loads(path.read_text())
        config["steps"].insert(
            0, {"registry_name": "g05_normalization_clamp", "config": {"minimum": -1, "maximum": 1}}
        )
        if checkpoint == candidate:
            config["steps"].reverse()
        path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    assert any(
        f.details.get("reason") == "processor_step_order_changed"
        for f in _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
    )


@pytest.mark.parametrize(
    "role,key,registry",
    [
        ("pre", "observation.state", "molmoact2_masked_normalizer"),
        ("post", "action", "molmoact2_masked_unnormalizer"),
    ],
)
@pytest.mark.parametrize("baseline_has_mask", [False, True])
def test_masked_normalization_bypass_is_guard_removal_even_when_action_scale_shrinks(
    comparison_pair: tuple[Path, Path],
    role: str,
    key: str,
    registry: str,
    baseline_has_mask: bool,
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint in (baseline, candidate):
        path = checkpoint / f"{role}processor_config.json"
        config = json.loads(path.read_text())
        config["steps"][0]["registry_name"] = registry
        path.write_text(json.dumps(config))
        state_path = checkpoint / config["steps"][0]["state_file"]
        state = load_file(state_path)
        state[f"{key}.mean"] = torch.zeros(14)
        state[f"{key}.std"] = torch.full((14,), 2.0)
        if checkpoint == candidate or baseline_has_mask:
            mask = torch.ones(14, dtype=torch.bool)
            if checkpoint == candidate:
                mask[2] = False
            state[f"{key}.mask"] = mask
        save_file(state, state_path)
    findings = run_privilege_delta(baseline, candidate)
    failure = next(
        f
        for f in _find(findings, "PRIVILEGE_GUARD_REMOVAL", DiagnosticStatus.FAIL)
        if f.details.get("reason") == "normalization_mask_bypass"
    )
    assert failure.provenance.tensor_key == f"{key}.mask"
    assert failure.provenance.violating_indices == [2]
    assert failure.details["step_index"] == 0
    assert failure.details["tensor_slice"] == [False]
    assert failure.details["baseline_mask"] == [True]
    assert _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.CONFLICT)
    assert not _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)


def test_observation_mean_shift_fails_baseline_cli_with_identical_action_mapping(
    comparison_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, mean in ((baseline, 0.0), (candidate, 1000.0)):
        path = _pre_state(checkpoint)
        state = load_file(path)
        state["observation.state.mean"] = torch.full((14,), mean)
        state["observation.state.std"] = torch.ones(14)
        save_file(state, path)
    report_path = tmp_path / "observation_shift.json"
    assert main(["--policy", str(candidate), "--baseline", str(baseline), "--report", str(report_path)]) == 1
    findings = run_privilege_delta(baseline, candidate)
    drift = _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL)[0]
    assert drift.provenance.tensor_key == "observation.state.mean"
    assert drift.provenance.violating_indices == list(range(14))
    assert drift.details["candidate_stats"]["mean"] == [1000.0] * 14
    assert drift.details["candidate_forward_offset"] == pytest.approx([-1000.0] * 14)
    assert _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.PASS)


@pytest.mark.parametrize(
    "feature_key,feature_type",
    [
        ("observation.state", FeatureType.STATE),
        ("observation.environment_state", FeatureType.ENV),
    ],
)
@pytest.mark.parametrize(
    "mode,names",
    [
        (NormalizationMode.MEAN_STD, ("mean", "std")),
        (NormalizationMode.MIN_MAX, ("min", "max")),
        (NormalizationMode.QUANTILES, ("q01", "q99")),
        (NormalizationMode.QUANTILE10, ("q10", "q90")),
    ],
)
def test_observation_forward_drift_screens_all_numeric_features_and_modes(
    comparison_pair,
    feature_key,
    feature_type,
    mode,
    names,
    monkeypatch,
) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, upper in ((baseline, 1.0), (candidate, 0.5)):
        pipeline = PolicyProcessorPipeline(
            name="policy_preprocessor",
            steps=[
                NormalizerProcessorStep(
                    features={feature_key: PolicyFeature(feature_type, (14,))},
                    norm_map={feature_type: mode},
                    stats={
                        feature_key: {
                            names[0]: torch.zeros(14),
                            names[1]: torch.full((14,), upper),
                        }
                    },
                )
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    monkeypatch.setattr(
        DataProcessorPipeline, "from_config", lambda *a, **k: pytest.fail("full pipeline constructed")
    )
    monkeypatch.setattr(
        DataProcessorPipeline, "from_pretrained", lambda *a, **k: pytest.fail("full pipeline constructed")
    )
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    drift = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL
    )[0]
    assert drift.details["feature_key"] == feature_key
    assert drift.details["forward_slope_ratio"] == pytest.approx([2.0] * 14)
    assert drift.provenance.violating_indices == list(range(14))
    assert drift.provenance.tensor_key == f"{feature_key}.{names[1]}"


def test_observation_epsilon_drift_changes_native_forward_mapping(comparison_pair) -> None:
    baseline, candidate = comparison_pair
    for checkpoint, eps in ((baseline, 1e-8), (candidate, 1e-4)):
        config_path = checkpoint / "preprocessor_config.json"
        config = json.loads(config_path.read_text())
        config["steps"][0]["config"]["eps"] = eps
        config_path.write_text(json.dumps(config))
        path = _pre_state(checkpoint)
        state = load_file(path)
        state["observation.state.std"] = torch.zeros(14)
        save_file(state, path)
    drift = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL
    )[0]
    assert drift.details["baseline_eps"] == 1e-8
    assert drift.details["candidate_eps"] == 1e-4
    assert drift.details["forward_slope_ratio"] == pytest.approx([1e-4] * 14)
    assert drift.provenance.tensor_key == "config.eps"
    assert Path(drift.provenance.file_path).name == "preprocessor_config.json"
    assert drift.details["tensor_slice"] == [1e-4]


@pytest.mark.parametrize("change", ["missing_stat", "identity", "uninspected"])
def test_observation_drift_gate_never_passes_missing_or_changed_active_coverage(
    comparison_pair, change
) -> None:
    baseline, candidate = comparison_pair
    if change == "missing_stat":
        path = _pre_state(candidate)
        state = load_file(path)
        state.pop("observation.state.mean")
        save_file(state, path)
    else:
        path = candidate / "preprocessor_config.json"
        config = json.loads(path.read_text())
        if change == "identity":
            config["steps"][0]["config"]["norm_map"]["STATE"] = "IDENTITY"
        else:
            config["steps"][0]["registry_name"] = "uninspected_custom_observation_normalizer"
        path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    drift = _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL)
    assert drift and drift[0].details["required_coverage"] is True
    assert not _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.PASS)
    assert main(["--policy", str(candidate), "--baseline", str(baseline)]) == 1


def test_legacy_observation_drift_uses_embedded_forward_buffers(checkpoints, tmp_path) -> None:
    baseline, candidate = tmp_path / "legacy0", tmp_path / "legacy1"
    for checkpoint, mean in ((baseline, 0.0), (candidate, 1000.0)):
        shutil.copytree(checkpoints["legacy_act_embedded"], checkpoint)
        (checkpoint / "config.json").write_text(
            json.dumps(
                {
                    "input_shapes": {"observation.state": [14]},
                    "output_shapes": {"action": [14]},
                    "input_normalization_modes": {"observation.state": "MEAN_STD"},
                    "output_normalization_modes": {"action": "MEAN_STD"},
                    "n_action_steps": 50,
                }
            )
        )
        path = checkpoint / "model.safetensors"
        state = load_file(path)
        state["normalize_inputs.buffer_observation_state.mean"] = torch.full((14,), mean)
        state["normalize_inputs.buffer_observation_state.std"] = torch.ones(14)
        save_file(state, path)
    drift = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL
    )[0]
    assert drift.provenance.tensor_key == "normalize_inputs.buffer_observation_state.mean"
    assert drift.provenance.step_name == "legacy_normalize_inputs"
    assert drift.details["step_index"] is None


@pytest.mark.parametrize(
    "mode,names",
    [
        (NormalizationMode.MIN_MAX, ("min", "max")),
        (NormalizationMode.QUANTILES, ("q01", "q99")),
        (NormalizationMode.QUANTILE10, ("q10", "q90")),
    ],
)
def test_native_zero_span_eps_replacement_catches_small_scale_action_expansion(
    comparison_pair,
    mode,
    names,
) -> None:
    baseline, candidate = comparison_pair
    endpoints = {}
    for checkpoint, lower, upper in ((baseline, -5e-10, 5e-10), (candidate, -5e-9, -5e-9)):
        step = UnnormalizerProcessorStep(
            features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
            norm_map={FeatureType.ACTION: mode},
            eps=1e-8,
            stats={"action": {names[0]: torch.full((14,), lower), names[1]: torch.full((14,), upper)}},
        )
        pipeline = PolicyProcessorPipeline(name="policy_postprocessor", steps=[step])
        endpoints[checkpoint] = tuple(
            pipeline({"action": torch.full((14,), value)})["action"] for value in (-1.0, 1.0)
        )
        pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    native_ratio = (endpoints[candidate][1] - endpoints[candidate][0]) / (
        endpoints[baseline][1] - endpoints[baseline][0]
    )
    assert native_ratio.tolist() == pytest.approx([10.0] * 14)
    findings = run_privilege_delta(baseline, candidate)
    gain = _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL)[0]
    assert gain.details["expansion_ratio"] == pytest.approx(native_ratio.tolist())
    assert gain.details["candidate_scale"] == pytest.approx([5e-9] * 14)
    assert gain.details["baseline_scale"] == pytest.approx([5e-10] * 14)
    assert gain.provenance.tensor_key == f"action.{names[1]}"
    assert gain.details["tensor_slice"] == pytest.approx([-5e-9] * 14)
    envelopes = _find(findings, "PRIVILEGE_ACTION_ENVELOPE_SHIFT", DiagnosticStatus.FAIL)
    assert {finding.details["bound_side"] for finding in envelopes} == {"lower", "upper"}
    assert all(finding.details["envelope_tolerance"] == [1e-12] * 14 for finding in envelopes)
    for finding in envelopes:
        expected = endpoints[candidate][0 if finding.details["bound_side"] == "lower" else 1]
        assert finding.details["candidate_" + finding.details["bound_side"]] == pytest.approx(
            expected.tolist(), abs=1e-15
        )


def test_benign_inline_action_override_is_advisory_and_compares_effective_stats(comparison_pair) -> None:
    baseline, candidate = comparison_pair
    config_path = candidate / "postprocessor_config.json"
    config = json.loads(config_path.read_text())
    config["steps"][0]["config"]["stats"] = {"action": {"mean": [0.0] * 14, "std": [0.5] * 14}}
    config_path.write_text(json.dumps(config))
    state_path = _post_state(candidate)
    state = load_file(state_path)
    state["action.std"] = torch.full((14,), 999.0)
    save_file(state, state_path)
    findings = run_privilege_delta(baseline, candidate)
    advisory = _find(findings, "INLINE_STAT_OVERRIDE_ADVISORY", DiagnosticStatus.PASS)
    assert advisory and all(f.severity.value == "WARNING" for f in advisory)
    assert not any(f.status in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT} for f in findings)
    assert _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.PASS)


def _save_action_boundary_chain(
    checkpoint: Path, dtype: str, *, pathological: bool
) -> PolicyProcessorPipeline:
    feature = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    parameters = ((50000.0, 50000.0), (-0.995, 1.99e-5)) if pathological else ((0.0, 1.0), (0.0, 1.0))
    steps = [DeviceProcessorStep(device="cpu", float_dtype=dtype)]
    steps += [
        UnnormalizerProcessorStep(
            features=feature,
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={"action": {"mean": torch.full((14,), mean), "std": torch.full((14,), std)}},
        )
        for mean, std in parameters
    ]
    pipeline = PolicyProcessorPipeline(name="policy_postprocessor", steps=steps)
    pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    return pipeline


def test_round3_fp16_intermediate_overflow_is_not_hidden_by_finite_composed_action_map(
    comparison_pair,
    monkeypatch,
):
    baseline, candidate = comparison_pair
    _save_action_boundary_chain(baseline, "float16", pathological=False)
    native = _save_action_boundary_chain(candidate, "float16", pathological=True)
    stage1 = list(native.step_through({"action": torch.full((5, 14), 3.0)}))[2]["action"]
    assert stage1.dtype is torch.float16
    assert torch.isinf(stage1).all()
    # Both unnormalizers have a finite, near-identity ideal composed map.
    from lerobot_flinch.privilege_delta import _action_affine, _policy, _static_pair

    composed = _action_affine(_static_pair(candidate, _policy(candidate))[1])
    assert torch.isfinite(composed["slope"]).all() and torch.isfinite(composed["origin"]).all()
    assert composed["slope"].tolist() == pytest.approx([0.995] * 14, abs=0.002)
    monkeypatch.setattr(
        DataProcessorPipeline, "from_pretrained", lambda *a, **k: pytest.fail("full pipeline loader")
    )
    monkeypatch.setattr(
        DataProcessorPipeline, "from_config", lambda *a, **k: pytest.fail("full pipeline loader")
    )
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    findings = run_privilege_delta(baseline, candidate)
    failure = next(
        f
        for f in _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)
        if f.details["checkpoint_role"] == "candidate"
    )
    assert failure.details["step_index"] == 1
    assert failure.details["step_class"] == "UnnormalizerProcessorStep"
    assert failure.details["input_dtype"] == failure.details["output_dtype"] == "float16"
    assert failure.details["offending_dimensions"] == list(range(14))
    assert any(torch.isinf(torch.tensor(failure.details["tensor_slice"])))
    assert failure.provenance.tensor_key == "action"
    assert failure.provenance.sha256_prefix


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_round3_finite_runtime_dtype_chain_is_not_an_fp16_false_alarm(comparison_pair, dtype):
    baseline, candidate = comparison_pair
    _save_action_boundary_chain(baseline, dtype, pathological=False)
    native = _save_action_boundary_chain(candidate, dtype, pathological=True)
    assert torch.isfinite(native({"action": torch.full((5, 14), 3.0)})["action"]).all()
    findings = run_privilege_delta(baseline, candidate)
    assert not _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)


def test_round3_bad_baseline_also_reports_its_intermediate_dtype_failure(comparison_pair):
    baseline, candidate = comparison_pair
    _save_action_boundary_chain(baseline, "float16", pathological=True)
    _save_action_boundary_chain(candidate, "float16", pathological=True)
    failures = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL
    )
    assert {finding.details["checkpoint_role"] for finding in failures} == {"baseline", "candidate"}


@pytest.mark.parametrize(
    "feature_key,feature_type",
    [
        ("observation.state", FeatureType.STATE),
        ("observation.environment_state", FeatureType.ENV),
    ],
)
def test_round3_subthreshold_stage_offset_is_caught_after_composed_observation_gain(
    comparison_pair,
    feature_key,
    feature_type,
):
    baseline, candidate = comparison_pair
    for checkpoint, origin in ((baseline, 0.0), (candidate, 0.004)):
        pipeline = PolicyProcessorPipeline(
            name="policy_preprocessor",
            steps=[
                NormalizerProcessorStep(
                    features={feature_key: PolicyFeature(feature_type, (14,))},
                    norm_map={feature_type: NormalizationMode.MEAN_STD},
                    stats={
                        feature_key: {
                            "mean": torch.full((14,), origin if index == 0 else 0.0),
                            "std": torch.full((14,), 0.1),
                        }
                    },
                )
                for index in range(3)
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    findings = run_privilege_delta(baseline, candidate)
    drifts = _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL)
    assert len(drifts) == 1, "Each individual stage is below its drift threshold"
    drift = drifts[0]
    assert drift.details["comparison_scope"] == "composed_policy_input"
    assert drift.details["candidate_step_indices"] == [0, 1, 2]
    assert drift.details["policy_input_shift"] == pytest.approx([4.0] * 14, rel=1e-5)
    assert drift.details["composed_offset_shift"] == pytest.approx([4.0] * 14, rel=1e-5)
    assert drift.details["candidate_composed_offset"] == pytest.approx([-4.0] * 14, rel=1e-5)
    assert drift.details["candidate_composed_slope"] == pytest.approx([1000.0] * 14, rel=1e-5)
    assert drift.provenance.violating_indices == list(range(14))
    assert main(["--policy", str(candidate), "--baseline", str(baseline)]) == 1


def test_round3_identical_composed_observation_chains_pass_without_dtype_failure(comparison_pair):
    baseline, candidate = comparison_pair
    for checkpoint in (baseline, candidate):
        pipeline = PolicyProcessorPipeline(
            name="policy_preprocessor",
            steps=[
                NormalizerProcessorStep(
                    features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
                    norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                    stats={"observation.state": {"mean": torch.zeros(14), "std": torch.full((14,), 0.1)}},
                )
                for _ in range(3)
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    findings = run_privilege_delta(baseline, candidate)
    passed = _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.PASS)
    assert passed
    assert passed[0].details["processor_execution_required"] is True
    assert passed[0].details["native_boundary_probes_executed"] is True
    assert not _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)


def test_round3_single_stage_fp16_observation_boundary_overflow_has_exact_step(comparison_pair):
    baseline, candidate = comparison_pair
    for checkpoint, mean in ((baseline, 0.0), (candidate, 50000.0)):
        pipeline = PolicyProcessorPipeline(
            name="policy_preprocessor",
            steps=[
                DeviceProcessorStep(device="cpu", float_dtype="float16"),
                NormalizerProcessorStep(
                    features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
                    norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                    stats={
                        "observation.state": {"mean": torch.full((14,), mean), "std": torch.full((14,), 0.5)}
                    },
                ),
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    findings = run_privilege_delta(baseline, candidate)
    failure = next(
        f
        for f in _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)
        if f.details["checkpoint_role"] == "candidate"
    )
    assert failure.details["feature_key"] == "observation.state"
    assert failure.details["step_index"] == 1
    assert failure.details["output_dtype"] == "float16"


def test_round3_later_float32_promotion_does_not_hide_earlier_fp16_overflow(comparison_pair):
    baseline, candidate = comparison_pair
    for checkpoint, pathological in ((baseline, False), (candidate, True)):
        pipeline = _save_action_boundary_chain(checkpoint, "float16", pathological=pathological)
        pipeline.steps.insert(2, DeviceProcessorStep(device="cpu", float_dtype="float32"))
        pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    findings = run_privilege_delta(baseline, candidate)
    failure = next(
        f
        for f in _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)
        if f.details["checkpoint_role"] == "candidate"
    )
    assert failure.details["step_index"] == 1
    assert failure.details["output_dtype"] == "float16"


def test_round3_unexecuted_intervening_clamp_prevents_definite_probe_overflow_claim(comparison_pair):
    baseline, candidate = comparison_pair
    for checkpoint, pathological in ((baseline, False), (candidate, True)):
        _save_action_boundary_chain(checkpoint, "float16", pathological=pathological)
        path = checkpoint / "postprocessor_config.json"
        config = json.loads(path.read_text())
        config["steps"].insert(
            1, {"registry_name": "custom_declared_action_clamp", "config": {"min": 0.0, "max": 0.0}}
        )
        path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    assert not _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)
    unknown = _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.UNKNOWN)
    assert unknown and all(f.details["required_coverage"] for f in unknown)
    assert main(["--policy", str(candidate), "--baseline", str(baseline)]) == 1


def test_round3_observation_probes_use_saved_one_std_boundaries_not_invented_three_std(comparison_pair):
    baseline, candidate = comparison_pair
    for checkpoint in (baseline, candidate):
        pipeline = PolicyProcessorPipeline(
            name="policy_preprocessor",
            steps=[
                DeviceProcessorStep(device="cpu", float_dtype="float16"),
                NormalizerProcessorStep(
                    features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
                    norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                    stats={"observation.state": {"mean": torch.zeros(14), "std": torch.full((14,), 30000.0)}},
                ),
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    findings = run_privilege_delta(baseline, candidate)
    assert _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.PASS)
    assert not _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)


def test_round3_broadcast_leading_stat_axes_do_not_crash_nonfinite_probe_attribution(comparison_pair):
    baseline, candidate = comparison_pair
    _save_action_boundary_chain(baseline, "float16", pathological=False)
    _save_action_boundary_chain(candidate, "float16", pathological=True)
    path = _post_state(candidate)
    tensors = {key: value.reshape(1, 1, 14) for key, value in load_file(path).items()}
    save_file(tensors, path)
    failures = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL
    )
    failure = next(f for f in failures if f.details["checkpoint_role"] == "candidate")
    assert failure.details["step_index"] == 1
    assert failure.details["input_shape"] == [5, 14]
    assert failure.details["output_shape"] == [1, 5, 14]
    assert len(failure.details["input_values"]) == len(failure.details["output_values"])
    assert failure.provenance.violating_indices == list(range(14))


def test_round3_composed_observation_probes_preserve_leading_singleton_stat_axes(comparison_pair):
    baseline, candidate = comparison_pair
    for checkpoint, origin in ((baseline, 0.0), (candidate, 0.004)):
        pipeline = PolicyProcessorPipeline(
            name="policy_preprocessor",
            steps=[
                NormalizerProcessorStep(
                    features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
                    norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                    stats={
                        "observation.state": {
                            "mean": torch.full((1, 1, 14), origin if index == 0 else 0.0),
                            "std": torch.full((1, 1, 14), 0.1),
                        }
                    },
                )
                for index in range(3)
            ],
        )
        pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    findings = run_privilege_delta(baseline, candidate)
    drift = next(
        f
        for f in _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL)
        if f.details.get("comparison_scope") == "composed_policy_input"
    )
    assert drift.details["policy_input_shift"] == pytest.approx([4.0] * 14, rel=1e-5)
    assert len(drift.details["candidate_probe_outputs"]) == 3


def _round4_empty_action_checkpoint(checkpoint, mode, *, scalar=False, declare_action=True):
    checkpoint.mkdir(parents=True, exist_ok=True)
    state = PolicyFeature(FeatureType.STATE, (6,))
    PolicyProcessorPipeline(
        name="policy_preprocessor",
        steps=[
            NormalizerProcessorStep(
                features={"observation.state": state},
                norm_map={FeatureType.STATE: NormalizationMode.IDENTITY},
            )
        ],
    ).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    shape = () if scalar else (6,)
    names = {
        NormalizationMode.MEAN_STD: ("mean", "std"),
        NormalizationMode.MIN_MAX: ("min", "max"),
        NormalizationMode.QUANTILES: ("q01", "q99"),
        NormalizationMode.QUANTILE10: ("q10", "q90"),
    }[mode]
    stats = {
        names[0]: torch.zeros(shape),
        names[1]: torch.full(shape, 0.5 if mode is NormalizationMode.MEAN_STD else 1.0),
    }
    PolicyProcessorPipeline(
        name="policy_postprocessor",
        steps=[
            UnnormalizerProcessorStep(
                features={},
                norm_map={FeatureType.ACTION: mode},
                stats={"action": stats},
            )
        ],
    ).save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "n_obs_steps": 1,
                "n_action_steps": 50,
                "chunk_size": 100,
                "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
                "output_features": {"action": {"type": "ACTION", "shape": [6]}} if declare_action else {},
            }
        )
    )
    return checkpoint


def test_round4_b_implicit_action_invalid_stats_fail_native_comparison(tmp_path):
    baseline = _round4_empty_action_checkpoint(tmp_path / "baseline", NormalizationMode.MEAN_STD)
    candidate = _round4_empty_action_checkpoint(tmp_path / "candidate", NormalizationMode.MEAN_STD)
    tensors = load_file(_post_state(candidate))
    tensors["action.std"][2:4] = torch.tensor([float("nan"), float("inf")])
    save_file(tensors, _post_state(candidate))
    config = json.loads((candidate / "postprocessor_config.json").read_text())
    assert config["steps"][0]["config"]["features"] == {}
    # The actual native class applies ACTION even without a feature declaration.
    native = PolicyProcessorPipeline.from_pretrained(candidate, config_filename="postprocessor_config.json")
    result = native({"action": torch.ones(1, 6)})["action"]
    assert not torch.isfinite(result[:, 2:4]).any()
    findings = run_privilege_delta(baseline, candidate)
    failure = next(
        f
        for f in _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)
        if f.details["checkpoint_role"] == "candidate"
    )
    assert failure.provenance.violating_indices == [2, 3]
    assert not _find(findings, "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.PASS)
    assert main(["--policy", str(candidate), "--baseline", str(baseline)]) == 1


@pytest.mark.parametrize(
    "mode",
    [
        NormalizationMode.MEAN_STD,
        NormalizationMode.MIN_MAX,
        NormalizationMode.QUANTILES,
        NormalizationMode.QUANTILE10,
    ],
)
def test_round4_b_implicit_action_scale_is_compared_in_every_active_mode(tmp_path, mode):
    baseline = _round4_empty_action_checkpoint(tmp_path / "baseline", mode)
    candidate = _round4_empty_action_checkpoint(tmp_path / "candidate", mode)
    tensors = load_file(_post_state(candidate))
    high = {
        NormalizationMode.MEAN_STD: "std",
        NormalizationMode.MIN_MAX: "max",
        NormalizationMode.QUANTILES: "q99",
        NormalizationMode.QUANTILE10: "q90",
    }[mode]
    tensors[f"action.{high}"][2] *= 2
    save_file(tensors, _post_state(candidate))
    finding = _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.FAIL
    )[0]
    assert finding.provenance.violating_indices == [2]
    assert finding.details["expansion_ratio"] == pytest.approx([2.0])
    assert finding.details["action_shape_source"] == "policy_output_features"
    assert finding.details["physical_actuator_count_inferred"] is False


@pytest.mark.parametrize(
    "scalar,declare_action,source",
    [(True, True, "policy_output_features"), (False, False, "saved_stat_broadcast")],
)
def test_round4_b_implicit_action_width_provenance_remains_explicit(tmp_path, scalar, declare_action, source):
    checkpoint = _round4_empty_action_checkpoint(
        tmp_path / "checkpoint", NormalizationMode.MEAN_STD, scalar=scalar, declare_action=declare_action
    )
    finding = _find(
        run_privilege_delta(checkpoint, checkpoint), "PRIVILEGE_ACTION_GAIN_ESCALATION", DiagnosticStatus.PASS
    )[0]
    assert finding.details["action_shape_source"] == source
    assert len(finding.details["baseline_scale"]) == 6
    assert finding.details["physical_actuator_count_inferred"] is False


def _round4_visual_checkpoint(checkpoint, mode, *, changed=False, scalar=False, large=False):
    checkpoint.mkdir(parents=True, exist_ok=True)
    image_shape = (3, 1024, 1024) if large else (3, 32, 32)
    names = {
        NormalizationMode.MEAN_STD: ("mean", "std"),
        NormalizationMode.MIN_MAX: ("min", "max"),
        NormalizationMode.QUANTILES: ("q01", "q99"),
        NormalizationMode.QUANTILE10: ("q10", "q90"),
        NormalizationMode.IDENTITY: ("mean", "std"),
    }[mode]
    shape = () if scalar else (3,)
    upper = torch.full(
        shape, 0.25 if mode in (NormalizationMode.MEAN_STD, NormalizationMode.IDENTITY) else 1.0
    )
    if changed:
        upper[1] *= 0.01
    stats = {
        names[0]: torch.full(
            shape, 0.5 if mode in (NormalizationMode.MEAN_STD, NormalizationMode.IDENTITY) else 0.0
        ),
        names[1]: upper,
    }
    PolicyProcessorPipeline(
        name="policy_preprocessor",
        steps=[
            NormalizerProcessorStep(
                features={"observation.images.top": PolicyFeature(FeatureType.VISUAL, image_shape)},
                norm_map={FeatureType.VISUAL: mode},
                stats={"observation.images.top": stats},
            )
        ],
    ).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    PolicyProcessorPipeline(
        name="policy_postprocessor",
        steps=[
            UnnormalizerProcessorStep(
                features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={"action": {"mean": torch.zeros(6), "std": torch.ones(6)}},
            )
        ],
    ).save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "n_obs_steps": 1,
                "n_action_steps": 50,
                "chunk_size": 100,
                "input_features": {"observation.images.top": {"type": "VISUAL", "shape": list(image_shape)}},
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            }
        )
    )
    return checkpoint


@pytest.mark.parametrize(
    "mode",
    [
        NormalizationMode.MEAN_STD,
        NormalizationMode.MIN_MAX,
        NormalizationMode.QUANTILES,
        NormalizationMode.QUANTILE10,
    ],
)
def test_round4_d_visual_stat_drift_is_compared_without_full_image_probes(tmp_path, monkeypatch, mode):
    import lerobot_flinch.preflight as preflight

    baseline = _round4_visual_checkpoint(tmp_path / "baseline", mode, large=True)
    candidate = _round4_visual_checkpoint(tmp_path / "candidate", mode, changed=True, large=True)
    probe = preflight._native_normalization_probe
    seen = []

    def bounded(entries, normalizers, key, points):
        if key == "observation.images.top":
            seen.append(tuple(points.shape))
            assert points.numel() == 9, "Channel statistics must not allocate megapixel probe tensors"
        return probe(entries, normalizers, key, points)

    monkeypatch.setattr(preflight, "_native_normalization_probe", bounded)
    findings = run_privilege_delta(baseline, candidate)
    drifts = _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL)
    stage = next(f for f in drifts if f.details.get("comparison_scope") == "individual_stage")
    assert stage.details["feature_key"] == "observation.images.top"
    assert stage.details["feature_type"] == "VISUAL"
    assert stage.details["declared_feature_shape"] == [3, 1024, 1024]
    assert stage.details["statistic_shape"] == [3, 1, 1]
    assert stage.provenance.violating_indices == [1]
    assert stage.details["forward_slope_ratio"] == pytest.approx([100.0], rel=1e-5)
    composed = next(f for f in drifts if f.details.get("comparison_scope") == "composed_policy_input")
    assert composed.provenance.violating_indices == [1]
    assert seen == [(3, 3, 1, 1), (3, 3, 1, 1)]


def test_round4_d_real_native_visual_pipeline_confirms_the_reported_gain(tmp_path):
    baseline = _round4_visual_checkpoint(tmp_path / "baseline", NormalizationMode.MEAN_STD)
    candidate = _round4_visual_checkpoint(tmp_path / "candidate", NormalizationMode.MEAN_STD, changed=True)
    sample = {"observation.images.top": torch.full((1, 3, 32, 32), 0.6)}
    old = PolicyProcessorPipeline.from_pretrained(baseline, config_filename="preprocessor_config.json")(
        sample
    )
    new = PolicyProcessorPipeline.from_pretrained(candidate, config_filename="preprocessor_config.json")(
        sample
    )
    assert float(
        new["observation.images.top"][0, 1, 0, 0] / old["observation.images.top"][0, 1, 0, 0]
    ) == pytest.approx(100.0, rel=1e-5)
    assert _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL
    )


def test_round4_d_scalar_and_channel_visual_stats_with_same_mapping_are_compatible(tmp_path):
    baseline = _round4_visual_checkpoint(tmp_path / "baseline", NormalizationMode.MEAN_STD, scalar=True)
    candidate = _round4_visual_checkpoint(tmp_path / "candidate", NormalizationMode.MEAN_STD)
    findings = run_privilege_delta(baseline, candidate)
    assert not [f for f in findings if f.status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)]
    assert _find(findings, "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.PASS)


def test_round4_d_identity_visual_statistics_are_not_an_active_contract(tmp_path):
    baseline = _round4_visual_checkpoint(tmp_path / "baseline", NormalizationMode.IDENTITY)
    candidate = _round4_visual_checkpoint(tmp_path / "candidate", NormalizationMode.IDENTITY, changed=True)
    assert not _find(
        run_privilege_delta(baseline, candidate), "PRIVILEGE_OBSERVATION_STAT_DRIFT", DiagnosticStatus.FAIL
    )


def test_round4_b_declared_wrong_action_type_cannot_hide_native_action_stats(tmp_path):
    baseline = _round4_empty_action_checkpoint(tmp_path / "baseline", NormalizationMode.MEAN_STD)
    candidate = _round4_empty_action_checkpoint(tmp_path / "candidate", NormalizationMode.MEAN_STD)
    for checkpoint in (baseline, candidate):
        path = checkpoint / "postprocessor_config.json"
        config = json.loads(path.read_text())
        config["steps"][0]["config"]["features"] = {"action": {"type": "STATE", "shape": [6]}}
        path.write_text(json.dumps(config))
    tensors = load_file(_post_state(candidate))
    tensors["action.std"][2] = float("nan")
    save_file(tensors, _post_state(candidate))
    findings = run_privilege_delta(baseline, candidate)
    assert _find(findings, "PRIVILEGE_INTERMEDIATE_NON_FINITE", DiagnosticStatus.FAIL)
    assert _find(findings, "COMPATIBILITY_STATUS", DiagnosticStatus.CONFLICT)


def test_round4_d_legacy_explicit_visual_stats_are_compared_when_available(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for checkpoint, std in ((baseline, 0.25), (candidate, 0.0025)):
        checkpoint.mkdir()
        save_file(
            {
                "normalize_inputs.buffer_observation_images_top.mean": torch.full((3, 1, 1), 0.5),
                "normalize_inputs.buffer_observation_images_top.std": torch.full((3, 1, 1), std),
                "unnormalize_outputs.buffer_action.mean": torch.zeros(6),
                "unnormalize_outputs.buffer_action.std": torch.ones(6),
                "model.weight": torch.zeros(1024),
            },
            checkpoint / "model.safetensors",
        )
        (checkpoint / "config.json").write_text(
            json.dumps(
                {
                    "type": "act",
                    "n_obs_steps": 1,
                    "n_action_steps": 50,
                    "chunk_size": 100,
                    "input_features": {"observation.images.top": {"type": "VISUAL", "shape": [3, 32, 32]}},
                    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
                    "normalization_mapping": {"VISUAL": "MEAN_STD", "ACTION": "MEAN_STD"},
                }
            )
        )
    finding = next(
        f
        for f in _find(
            run_privilege_delta(baseline, candidate),
            "PRIVILEGE_OBSERVATION_STAT_DRIFT",
            DiagnosticStatus.FAIL,
        )
        if f.details.get("comparison_scope") == "individual_stage"
    )
    assert finding.details["feature_type"] == "VISUAL"
    assert finding.details["forward_slope_ratio"] == pytest.approx([100.0] * 3, rel=1e-5)
    assert finding.provenance.tensor_key == "normalize_inputs.buffer_observation_images_top.std"
