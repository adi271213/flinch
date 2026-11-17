"""Observed native processor behavior and declared temporal-anchor checks."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DataProcessorPipeline,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.relative_action_processor import to_absolute_actions, to_relative_actions

from lerobot_preflight.models import DiagnosticStatus
from lerobot_preflight.temporal import run_temporal_audit
from lerobot_preflight.trace import run_offline_trace
from tests.subprocess_env import source_subprocess_env
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path)


def _find(findings, check_id: str, status: DiagnosticStatus | None = None):
    return [
        finding
        for finding in findings
        if finding.check_id == check_id and (status is None or finding.status is status)
    ]


def _audit_temporal(checkpoint: Path):
    metadata = checkpoint / "metadata.json"
    return run_temporal_audit(checkpoint, metadata_path=metadata if metadata.is_file() else None)


def _observation_sample(checkpoint: Path) -> Path:
    return checkpoint / "sample_observation.npz"


def _flat_numbers(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def test_clean_sample_has_no_observed_trace_failure(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert findings
    assert all(finding.status not in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT} for finding in findings)
    assert not _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.FAIL)
    assert not _find(findings, "EMPIRICAL_GAIN_EXPLOSION", DiagnosticStatus.FAIL)
    assert not _find(findings, "RUNTIME_NON_FINITE", DiagnosticStatus.FAIL)
    json.dumps([finding.to_dict() for finding in findings], allow_nan=False)


def test_absent_real_action_marks_postprocessor_trace_unknown(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    (checkpoint / "sample_action.npz").unlink()
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert _find(findings, "POSTPROCESSOR_TRACE", DiagnosticStatus.UNKNOWN)
    assert not any(finding.status is DiagnosticStatus.FAIL for finding in findings)


def test_uninspected_step_blocks_native_loading_before_any_constructor(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    config_path = checkpoint / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    config["steps"].insert(0, {"registry_name": "untrusted_custom_step", "config": {}})
    config_path.write_text(json.dumps(config))

    def forbid_load(cls, *args, **kwargs):
        pytest.fail("an unsupported step reached native from_pretrained")

    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", classmethod(forbid_load))
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    unknowns = _find(findings, "OFFLINE_TRACE", DiagnosticStatus.UNKNOWN)
    assert len(unknowns) == 1
    assert unknowns[0].details["step_index"] == 0


def test_native_hooks_are_unregistered_after_offline_trace(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import lerobot_preflight.trace as trace_module

    checkpoint = checkpoints["clean_act_modern"]
    loaded = []

    class TrackingLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pipeline = PolicyProcessorPipeline.from_pretrained(*args, **kwargs)
            loaded.append(pipeline)
            return pipeline

    monkeypatch.setattr(trace_module, "PolicyProcessorPipeline", TrackingLoader)
    run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert len(loaded) == 2
    assert all(not pipeline.before_step_hooks and not pipeline.after_step_hooks for pipeline in loaded)


def test_4415_silent_bypass_identifies_native_step_and_missing_stat_keys(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_4415_silent_bypass"]
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    finding = failures[0]
    assert finding.details["pipeline_role"] == "preprocessor"
    assert finding.details["step_index"] == 0
    assert finding.details["step_class"] == "NormalizerProcessorStep"
    assert "observation.state" in {
        finding.details.get("tensor_key"), finding.details.get("feature_key")
    }
    assert {"observation.state.mean", "observation.state.std"} <= set(finding.details["expected_keys"])
    assert "observation.state" in finding.details["expected_feature_keys"]
    assert "wrong_dataset/observation.state.mean" in finding.details["actual_stat_keys"]
    assert "wrong_dataset/observation.state.std" in finding.details["actual_stat_keys"]
    assert np.allclose(_flat_numbers(finding.details["input_values"]), 0.20)
    assert np.allclose(_flat_numbers(finding.details["output_values"]), 0.20)
    assert np.allclose(_flat_numbers(finding.details["tensor_slice"]), 0.20)
    assert Path(finding.provenance.file_path).suffix == ".safetensors"
    assert finding.provenance.step_name == "NormalizerProcessorStep"
    assert finding.provenance.tensor_key == "observation.state"
    assert finding.provenance.sha256_prefix == hashlib.sha256(
        Path(finding.provenance.file_path).read_bytes()
    ).hexdigest()[:12]


def test_matching_stats_with_silent_native_step_are_still_detected(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]

    def silently_unchanged(self, transition):
        return transition.copy()

    monkeypatch.setattr(NormalizerProcessorStep, "__call__", silently_unchanged)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    bypasses = [
        finding
        for finding in _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.FAIL)
        if finding.details["feature_key"] == "observation.state"
    ]
    assert bypasses
    assert bypasses[0].details["reason"] == "unchanged_nonidentity_transform"
    assert bypasses[0].details["step_index"] == 0
    assert bypasses[0].details["input_values"] == bypasses[0].details["output_values"]


def test_matching_action_stats_with_silent_postprocessor_are_detected(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]

    def silently_unchanged(self, transition):
        return transition.copy()

    monkeypatch.setattr(UnnormalizerProcessorStep, "__call__", silently_unchanged)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    bypasses = [
        finding
        for finding in _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.FAIL)
        if finding.details["pipeline_role"] == "postprocessor"
        and finding.details["feature_key"] == "action"
    ]
    assert bypasses
    assert bypasses[0].details["step_index"] == 0
    assert bypasses[0].details["step_class"] == "UnnormalizerProcessorStep"
    assert bypasses[0].details["reason"] == "unchanged_nonidentity_transform"
    assert bypasses[0].details["input_values"] == bypasses[0].details["output_values"]


def test_5289_trace_locates_empirical_gain_explosion_coordinate(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = _find(findings, "EMPIRICAL_GAIN_EXPLOSION", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    finding = failures[0]
    assert finding.details["pipeline_role"] == "preprocessor"
    assert finding.details["step_index"] == 0
    assert finding.details["step_class"] == "NormalizerProcessorStep"
    assert "observation.state" in {
        finding.details.get("tensor_key"), finding.details.get("feature_key")
    }
    assert finding.provenance.violating_indices == [9]
    assert [0, 9] in finding.details["tensor_coordinates"]
    assert np.allclose(_flat_numbers(finding.details["input_values"]), 0.20)
    assert np.max(np.abs(_flat_numbers(finding.details["output_values"]))) > 1e5
    assert np.max(np.abs(_flat_numbers(finding.details["tensor_slice"]))) > 1e5
    assert not _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.FAIL)
    assert not _find(
        run_offline_trace(checkpoint, _observation_sample(checkpoint), explosion_threshold=2e5),
        "EMPIRICAL_GAIN_EXPLOSION",
        DiagnosticStatus.FAIL,
    )


def test_4973_trace_reports_first_nonfinite_normalizer_output(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_4973_fp16_nan"]
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = _find(findings, "RUNTIME_NON_FINITE", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    finding = failures[0]
    assert finding.details["pipeline_role"] == "preprocessor"
    assert finding.details["step_index"] == 0
    assert finding.details["step_class"] == "NormalizerProcessorStep"
    assert "observation.state" in {
        finding.details.get("tensor_key"), finding.details.get("feature_key")
    }
    assert finding.provenance.violating_indices == [4]
    assert [0, 4] in finding.details["tensor_coordinates"]
    assert np.isnan(_flat_numbers(finding.details["output_values"])).any()
    assert np.isnan(_flat_numbers(finding.details["tensor_slice"])).any()


def test_first_nonfinite_step_is_reported_after_a_native_identity_step(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_4973_fp16_nan"]
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True
    )
    preprocessor.steps = [IdentityProcessorStep(), *preprocessor.steps]
    preprocessor.save_pretrained(checkpoint, config_filename="preprocessor_config.json")

    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = _find(findings, "RUNTIME_NON_FINITE", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    assert failures[0].details["step_index"] == 1
    assert failures[0].details["step_class"] == "NormalizerProcessorStep"


@pytest.mark.parametrize("variant", ["identity_mode", "excluded_key", "fixed_point"])
def test_legitimate_unchanged_observation_is_not_silent_bypass(
    checkpoints: dict[str, Path], variant: str
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True
    )
    normalizer = preprocessor.steps[0]
    if variant == "identity_mode":
        normalizer.norm_map[FeatureType.STATE] = NormalizationMode.IDENTITY
    elif variant == "excluded_key":
        normalizer.normalize_observation_keys = {"observation.other"}
    elif variant == "fixed_point":
        with np.load(_observation_sample(checkpoint), allow_pickle=False) as sample:
            state = np.zeros_like(sample["observation.state"])
        np.savez(_observation_sample(checkpoint), **{"observation.state": state})
    preprocessor.save_pretrained(checkpoint, config_filename="preprocessor_config.json")

    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert not _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.FAIL)


def test_trace_accepts_native_default_processor_config_filenames(
    checkpoints: dict[str, Path],
) -> None:
    source = checkpoints["clean_act_modern"]
    checkpoint = source.parent / "default_names_trace"
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        source, config_filename="preprocessor_config.json", local_files_only=True
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        source, config_filename="postprocessor_config.json", local_files_only=True
    )
    preprocessor.name = "policy_preprocessor"
    postprocessor.name = "policy_postprocessor"
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    shutil.copyfile(source / "sample_observation.npz", checkpoint / "sample_observation.npz")
    shutil.copyfile(source / "sample_action.npz", checkpoint / "sample_action.npz")

    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert findings
    assert all(finding.status not in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT} for finding in findings)


def test_5315_temporal_audit_exposes_wrong_current_frame_anchor(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    findings = _audit_temporal(checkpoint)
    failures = _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    finding = failures[0]
    assert finding.details["step_index"] == 0
    assert finding.details["feature_key"] == "observation.state"
    assert finding.details["action_key"] == "action"
    assert finding.details["n_obs_steps"] == 3
    assert finding.details["observation_order"] == "oldest_first"
    assert finding.details["expected_anchor_index"] == 2
    assert finding.details["actual_anchor_index"] == 0
    assert finding.details["probe_state_slice"] == pytest.approx([0.20, 0.45, 0.70])
    assert finding.details["observed_relative_action_slice"] == pytest.approx([0.70])
    assert finding.details["expected_relative_action_slice"] == pytest.approx([0.20])
    assert finding.details["tensor_slice"] == pytest.approx([0.70])
    assert finding.details["native_roundtrip_passed"] is True
    assert finding.details["native_roundtrip_action_slice"] == pytest.approx([0.90])
    assert Path(finding.provenance.file_path).name == "preprocessor_config.json"
    assert finding.provenance.step_name == "RelativeActionsProcessorStep"
    assert finding.provenance.tensor_key == "observation.state"
    assert finding.provenance.violating_indices == [0]

    with np.load(checkpoint / "sample_observation.npz", allow_pickle=False) as sample:
        state = torch.from_numpy(sample["observation.state"].copy())
    with np.load(checkpoint / "sample_action.npz", allow_pickle=False) as sample:
        action = torch.from_numpy(sample["action"].copy())
    relative = to_relative_actions(action, state, [True] * 14)
    restored = to_absolute_actions(relative, state, [True] * 14)
    assert relative[0, 0, 0].item() == pytest.approx(0.70)
    torch.testing.assert_close(restored, action)


def test_temporal_audit_requires_explicit_chronology(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    # The sample still has three frames. Shape alone must not name the current one.
    (checkpoint / "metadata.json").write_text(json.dumps({"n_obs_steps": 3}))
    findings = _audit_temporal(checkpoint)
    unknowns = _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.UNKNOWN)
    assert unknowns
    assert unknowns[0].details["missing_declarations"] == ["temporal_observation_ordering"]
    assert not _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.FAIL)


def test_temporal_audit_passes_when_no_relative_anchor_is_enabled(
    checkpoints: dict[str, Path],
) -> None:
    findings = _audit_temporal(checkpoints["clean_act_modern"])
    assert _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.PASS)
    assert not _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.FAIL)


def test_explicit_newest_first_order_matches_native_anchor(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    (checkpoint / "metadata.json").write_text(
        json.dumps({"n_obs_steps": 3, "observation_order": "newest_first", "current_index": 0})
    )
    findings = _audit_temporal(checkpoint)
    passes = _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.PASS)
    assert len(passes) == 1
    assert passes[0].details["expected_anchor_index"] == 0
    assert passes[0].details["actual_anchor_index"] == 0


def test_single_observation_step_needs_no_chronology_declaration(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    (checkpoint / "metadata.json").write_text(json.dumps({"n_obs_steps": 1}))
    findings = _audit_temporal(checkpoint)
    passes = _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.PASS)
    assert len(passes) == 1
    assert passes[0].details["expected_anchor_index"] == 0
    assert passes[0].details["n_obs_steps"] == 1
    assert "Single-frame relative-action compatibility was verified" in passes[0].summary
    assert passes[0].details["dimension_compatibility_verified"] is True
    assert passes[0].details["native_roundtrip_passed"] is True


def test_temporal_audit_reports_conflicting_chronology_sources(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["temporal_observation_ordering"] = "newest_first"
    path.write_text(json.dumps(config))
    findings = _audit_temporal(checkpoint)
    conflicts = _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.CONFLICT)
    assert conflicts
    assert conflicts[0].details["field"] == "temporal_observation_ordering"
    assert len(conflicts[0].details["declarations"]) >= 2


def test_trace_and_temporal_audit_are_local_and_read_only(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = checkpoints["clean_act_modern"]
    temporal = checkpoints["bug_5315_temporal_anchor"]

    def snapshot(checkpoint: Path) -> dict[str, bytes]:
        return {
            path.relative_to(checkpoint).as_posix(): path.read_bytes()
            for path in checkpoint.rglob("*")
            if path.is_file()
        }

    clean_before = snapshot(clean)
    temporal_before = snapshot(temporal)

    def forbid_socket(*args, **kwargs):
        pytest.fail("diagnostic attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    run_offline_trace(clean, _observation_sample(clean))
    _audit_temporal(temporal)
    assert snapshot(clean) == clean_before
    assert snapshot(temporal) == temporal_before


def _add_vla_language_steps(checkpoint: Path) -> None:
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"] = [
        {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
        {"registry_name": "to_batch_processor", "config": {}},
        {"registry_name": "smolvla_new_line_processor", "config": {}},
        {"registry_name": "tokenizer_processor", "config": {
            "tokenizer_name": "never-download-this-tokenizer", "max_length": 48,
        }},
        {"registry_name": "device_processor", "config": {"device": "cpu"}},
        *config["steps"],
    ]
    path.write_text(json.dumps(config))


def _deny_tokenizer_and_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("numeric trace constructed a tokenizer or opened a network socket")
    monkeypatch.setattr(TokenizerProcessorStep, "__post_init__", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_vla_trace_stubs_only_language_and_executes_native_numeric_steps_read_only(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lerobot_preflight.trace as trace_module
    checkpoint = checkpoints["clean_act_modern"]
    _add_vla_language_steps(checkpoint)
    before = {str(path.relative_to(checkpoint)): path.read_bytes()
              for path in checkpoint.rglob("*") if path.is_file()}
    _deny_tokenizer_and_network(monkeypatch)
    pre, _ = trace_module._load_processor_pair(checkpoint, numeric_only=True)
    assert [type(step).__name__ for step in pre[0].steps] == [
        "RenameObservationsProcessorStep", "AddBatchDimensionProcessorStep", "NewLineTaskProcessorStep",
        "IdentityProcessorStep", "DeviceProcessorStep", "NormalizerProcessorStep",
    ]
    batch = {"observation.state": torch.full((14,), 0.20), "task": "Pick the cube"}
    output = pre[0](batch)
    assert output["observation.state"].shape == (1, 14)
    assert not torch.equal(output["observation.state"], batch["observation.state"].unsqueeze(0))
    assert output["task"] == ["Pick the cube\n"]
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert not any(f.status in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT} for f in findings)
    notes = _find(findings, "OFFLINE_LANGUAGE_TOKENIZATION", DiagnosticStatus.UNKNOWN)
    assert len(notes) == 1 and notes[0].severity.value == "INFO"
    assert notes[0].details["step_index"] == 3
    assert notes[0].details["language_coverage_verified"] is False
    assert notes[0].details["numeric_steps_real"] is True
    assert notes[0].details["downstream_token_consumers_checked"] is True
    assert notes[0].details["downstream_token_consumers"] == []
    assert "downstream token-dependent behavior remain unverified" in notes[0].summary
    assert not _find(findings, "OFFLINE_LANGUAGE_TOKENIZATION", DiagnosticStatus.PASS)
    assert notes[0].provenance.step_name == "TokenizerProcessorStep"
    assert not _find(findings, "OFFLINE_TRACE")
    for check in ("SILENT_STEP_BYPASS", "EMPIRICAL_GAIN_EXPLOSION", "RUNTIME_NON_FINITE"):
        assert _find(findings, check, DiagnosticStatus.PASS)
    assert before == {str(path.relative_to(checkpoint)): path.read_bytes()
                      for path in checkpoint.rglob("*") if path.is_file()}


@pytest.mark.parametrize("fixture,check,coordinate", [
    ("bug_4415_silent_bypass", "SILENT_STEP_BYPASS", 0),
    ("bug_5289_pathological_gain", "EMPIRICAL_GAIN_EXPLOSION", 9),
    ("bug_4973_fp16_nan", "RUNTIME_NON_FINITE", 4),
])
def test_vla_language_stub_preserves_bug_detection_and_original_step_indices(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    fixture: str, check: str, coordinate: int,
) -> None:
    checkpoint = checkpoints[fixture]
    _add_vla_language_steps(checkpoint)
    _deny_tokenizer_and_network(monkeypatch)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    matches = _find(findings, check, DiagnosticStatus.FAIL)
    assert matches
    assert matches[0].details["step_index"] == 5
    assert matches[0].details["step_class"] == "NormalizerProcessorStep"
    assert coordinate in matches[0].provenance.violating_indices
    assert matches[0].provenance.step_name == "NormalizerProcessorStep"


def test_vla_language_stub_does_not_whitelist_unknown_numeric_steps(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _add_vla_language_steps(checkpoint)
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(4, {"registry_name": "uninspected_numeric_transform", "config": {}})
    path.write_text(json.dumps(config))
    _deny_tokenizer_and_network(monkeypatch)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    unknown = _find(findings, "OFFLINE_TRACE", DiagnosticStatus.UNKNOWN)
    assert len(unknown) == 1 and unknown[0].details["step_index"] == 4


def test_large_raw_levels_are_not_gain_explosions_at_structural_vla_steps(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (14,)),
                "action": PolicyFeature(FeatureType.ACTION, (14,))}
    stats = {key: {"mean": torch.full((14,), 100.0), "std": torch.full((14,), 10.0)} for key in features}
    mapping = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        NormalizerProcessorStep(features=features, norm_map=mapping, stats=stats),
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    PolicyProcessorPipeline(name="policy_postprocessor", steps=[
        UnnormalizerProcessorStep(features={"action": features["action"]},
                                  norm_map=mapping, stats={"action": stats["action"]}),
    ]).save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    _add_vla_language_steps(checkpoint)
    np.savez(_observation_sample(checkpoint), **{
        "observation.state": np.full((14,), 110.0, dtype=np.float32),
        "observation.language.tokens": np.array([[99999]], dtype=np.int64),
    })
    np.savez(checkpoint / "sample_action.npz", action=np.full((14,), 110.0, dtype=np.float32))
    _deny_tokenizer_and_network(monkeypatch)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = [finding for finding in findings if finding.status is DiagnosticStatus.FAIL]
    # Structural/raw preprocessor values remain exempt. The actual inverse
    # numeric step now has its own coverage and grows normalized action 1 to 110.
    assert failures and all(finding.check_id == "EMPIRICAL_GAIN_EXPLOSION"
                            and finding.details["pipeline_role"] == "postprocessor"
                            and finding.details["step_index"] == 0 for finding in failures)
    assert not any(finding.check_id == "EMPIRICAL_GAIN_EXPLOSION"
                   and finding.details.get("pipeline_role") == "preprocessor" for finding in findings)


def test_strict_shared_loader_does_not_silently_stub_language_for_temporal_audits(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lerobot_preflight.trace as trace_module
    checkpoint = checkpoints["clean_act_modern"]
    _add_vla_language_steps(checkpoint)
    _deny_tokenizer_and_network(monkeypatch)
    with pytest.raises(trace_module._TraceLoadError) as exc:
        trace_module._load_processor_pair(checkpoint)
    assert exc.value.status is DiagnosticStatus.UNKNOWN


@pytest.mark.parametrize("consumer", ["active_normalizer", "rename", "uninspected"])
def test_stubbed_token_outputs_never_reach_declared_downstream_consumers(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch, consumer: str,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _add_vla_language_steps(checkpoint)
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    token_key = "observation.language.tokens"
    if consumer == "active_normalizer":
        step = {"registry_name": "normalizer_processor", "config": {
            "features": {token_key: {"type": "LANGUAGE", "shape": [48]}},
            "norm_map": {"LANGUAGE": "MEAN_STD"},
            "stats": {token_key: {"mean": [0.0], "std": [1.0]}},
        }}
    elif consumer == "rename":
        step = {"registry_name": "rename_observations_processor", "config": {
            "rename_map": {token_key: "observation.required_tokens"},
        }}
    else:
        step = {"registry_name": "uninspected_token_consumer", "config": {"input_key": token_key}}
    config["steps"].insert(4, step)
    path.write_text(json.dumps(config))
    _deny_tokenizer_and_network(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("a token-dependent pipeline was constructed")
    monkeypatch.setattr(DataProcessorPipeline, "from_config", forbidden)
    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", forbidden)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    unknown = _find(findings, "OFFLINE_TRACE", DiagnosticStatus.UNKNOWN)
    assert len(unknown) == 1
    assert unknown[0].details["step_index"] == 4
    assert unknown[0].details["token_keys"] == [token_key]
    assert unknown[0].details["stubbed_tokenizer_step_indices"] == [3]
    assert unknown[0].details["language_coverage_verified"] is False


def test_vla_tokenizer_accepts_contained_saved_artifact_directory(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _add_vla_language_steps(checkpoint)
    directory = checkpoint / "tokenizer"
    directory.mkdir()
    (directory / "tokenizer_config.json").write_text("{}")
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][3]["artifacts"] = {"tokenizer_name": "tokenizer"}
    path.write_text(json.dumps(config))
    _deny_tokenizer_and_network(monkeypatch)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert not _find(findings, "OFFLINE_TRACE")
    assert _find(findings, "OFFLINE_LANGUAGE_TOKENIZATION", DiagnosticStatus.UNKNOWN)
    assert _find(findings, "RUNTIME_NON_FINITE", DiagnosticStatus.PASS)


@pytest.mark.parametrize("location,value", [("observation", np.nan), ("observation", np.inf),
                                             ("action", -np.inf)])
def test_invalid_sample_nonfinite_is_attributed_before_first_processor_step(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch, location: str, value: float,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    sample = _observation_sample(checkpoint) if location == "observation" else checkpoint / "sample_action.npz"
    key = "observation.state" if location == "observation" else "action"
    tensor = np.full((1, 14), 0.20, dtype=np.float32)
    tensor[0, 7] = value
    np.savez(sample, **{key: tensor})
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input was executed by a native processor")
    monkeypatch.setattr(NormalizerProcessorStep, "__call__", forbidden)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    invalid = _find(findings, "INVALID_SAMPLE_INPUT", DiagnosticStatus.FAIL)
    assert len(invalid) == 1 and len(findings) == 1
    assert invalid[0].details["step_index"] is None
    assert invalid[0].details["stage"] == "sample_input"
    assert invalid[0].details["processor_steps_executed"] == 0
    assert invalid[0].details["tensor_coordinates"] == [[0, 7]]
    assert invalid[0].provenance.file_path == str(sample.resolve())
    assert invalid[0].provenance.sha256_prefix == hashlib.sha256(sample.read_bytes()).hexdigest()[:12]
    assert invalid[0].provenance.violating_indices == [7]
    assert not _find(findings, "RUNTIME_NON_FINITE")
    json.dumps(invalid[0].to_dict(), allow_nan=False)


@pytest.mark.parametrize("raw,mean,std,expected_failure", [
    (1000.0, 0.0, 10.0, False),  # 100 is large, but attenuated tenfold.
    (100.0, 0.0, 1.0, False),    # Already large input; no step amplification.
    (10.0, 0.0, 0.1, True),     # Both output > 50 and > 2x actual input.
])
def test_empirical_gain_requires_absolute_magnitude_and_actual_step_amplification(
    checkpoints: dict[str, Path], raw: float, mean: float, std: float, expected_failure: bool,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    pre = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True,
    )
    normalizer = pre.steps[0]
    normalizer.stats = {"observation.state": {"mean": [mean] * 14, "std": [std] * 14}}
    normalizer.to(dtype=torch.float32)
    pre.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    np.savez(_observation_sample(checkpoint), **{"observation.state": np.full((1, 14), raw, dtype=np.float32)})
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = _find(findings, "EMPIRICAL_GAIN_EXPLOSION", DiagnosticStatus.FAIL)
    assert bool(failures) is expected_failure
    if expected_failure:
        assert failures[0].details["amplification_factor"] == 2.0
        assert failures[0].details["step_index"] == 0
        assert failures[0].details["input_values"] == [raw] * 14
    else:
        assert _find(findings, "EMPIRICAL_GAIN_EXPLOSION", DiagnosticStatus.PASS)


def test_temporal_probe_preserves_exact_narrow_declared_state_width(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["input_features"]["observation.state"]["shape"] = [2]
    path.write_text(json.dumps(config))
    findings = _audit_temporal(checkpoint)
    failures = _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    details = failures[0].details
    assert details["declared_state_dim"] == 2
    assert details["declared_action_dim"] == 14
    assert details["synthetic_state_shape"] == [1, 3, 2]
    assert details["required_state_width"] == 14
    assert details["processor_steps_executed"] == 0
    assert not _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.PASS)


def test_temporal_probe_refuses_to_infer_undeclared_state_width_from_action(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    del config["input_features"]
    path.write_text(json.dumps(config))
    findings = _audit_temporal(checkpoint)
    unknowns = _find(findings, "TEMPORAL_ANCHOR_MISMATCH", DiagnosticStatus.UNKNOWN)
    assert len(unknowns) == 1
    assert "width is undeclared" in unknowns[0].summary
    assert unknowns[0].details["action_dim"] == 14
    assert "synthetic_state_shape" not in unknowns[0].details


def test_single_frame_relative_action_mask_rejects_narrow_state(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    (checkpoint / "metadata.json").write_text(json.dumps({"n_obs_steps": 1}))
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["input_features"]["observation.state"]["shape"] = [3]
    config["output_features"]["action"]["shape"] = [4]
    path.write_text(json.dumps(config))
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["n_obs_steps"] == 1
    assert finding.details["declared_state_dim"] == 3
    assert finding.details["declared_action_dim"] == 4
    assert finding.details["relative_action_mask"] == [True] * 4
    assert finding.details["missing_state_coordinates"] == [3]
    assert finding.provenance.violating_indices == [3]


@pytest.mark.parametrize("fault", ["nonzero_error", "exception", "wrong_shape", "nan"])
def test_temporal_roundtrip_failure_never_passes(checkpoints, monkeypatch, fault) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    (checkpoint / "metadata.json").write_text(json.dumps({
        "n_obs_steps": 3, "observation_order": "newest_first", "current_index": 0,
    }))
    native_inverse = AbsoluteActionsProcessorStep.__call__

    def damaged_inverse(self, transition):
        if fault == "exception":
            raise RuntimeError("inverse deliberately failed")
        result = native_inverse(self, transition)
        action = result["action"].clone()
        if fault == "wrong_shape":
            action = action[..., :1]
        else:
            action[..., 0] += {"nonzero_error": 0.25, "nan": float("nan")}[fault]
        return {**result, "action": action}

    monkeypatch.setattr(AbsoluteActionsProcessorStep, "__call__", damaged_inverse)
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["native_roundtrip_passed"] is False
    assert finding.details["roundtrip_error_tolerance"] == {"atol": 1e-5, "rtol": 1e-5}
    if fault == "nonzero_error":
        assert finding.details["native_roundtrip_max_abs_error"] > 0
    if fault == "exception":
        assert finding.details["roundtrip_error"] == "inverse deliberately failed"


def test_temporal_forward_probe_exception_is_failure(checkpoints, monkeypatch) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]

    def damaged_forward(self, transition):
        raise RuntimeError("forward deliberately failed")

    monkeypatch.setattr(RelativeActionsProcessorStep, "__call__", damaged_forward)
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["pipeline_error"] == "forward deliberately failed"


def test_counterexample_a_cli_import_uses_native_rename_when_helper_is_absent(tmp_path):
    script = '''
import socket
import lerobot.processor.rename_processor as renames
from lerobot.processor import RenameObservationsProcessorStep
# PyPI 0.7.0 exposes the native step while newer helper exports are optional.
def released_observation(self, observation):
    return {renames._rename_key_with_metadata(key, self.rename_map): value for key, value in observation.items()}
RenameObservationsProcessorStep.observation = released_observation
if hasattr(renames, "rename_keys"):
    del renames.rename_keys
def no_network(*args, **kwargs):
    raise AssertionError("compatibility import attempted network access")
socket.socket = socket.create_connection = no_network
from lerobot_preflight.cli import main
from lerobot_preflight.trace import rename_keys
marker = object()
assert rename_keys({"raw": marker}, {"raw": "observation.state"}) == {"observation.state": marker}
assert rename_keys({"raw_is_pad": marker}, {"raw": "observation.state"}) == {"observation.state_is_pad": marker}
assert callable(main)
print("CLI import and native rename fallback passed")
'''
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                               env=source_subprocess_env(), text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "native rename fallback passed" in completed.stdout


def test_counterexample_c_real_float32_one_ulp_roundtrip_passes(checkpoints):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    (checkpoint / "metadata.json").write_text(json.dumps({
        "n_obs_steps": 3, "observation_order": "newest_first", "current_index": 0,
    }))
    features = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {"action": {"mean": [-0.411523] * 14, "std": [0.168827] * 14}}
    relative = RelativeActionsProcessorStep(enabled=True)
    pre = PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats), relative,
    ])
    post = PolicyProcessorPipeline(name="policy_postprocessor", steps=[
        AbsoluteActionsProcessorStep(enabled=True, relative_step=relative),
        UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
    ])
    pre.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    post.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["native_roundtrip_passed"] is True
    assert finding.details["native_roundtrip_max_abs_error"] == pytest.approx(1.1920928955078125e-7)
    assert finding.details["roundtrip_error_tolerance"] == {"atol": 1e-5, "rtol": 1e-5}


def _large_offset_temporal_checkpoint(checkpoints, *, count=1):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    metadata = {"n_obs_steps": count}
    if count > 1:
        metadata.update(observation_order="oldest_first", current_index=-1)
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    features = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {"action": {"mean": [2048.0] * 14, "std": [0.168827] * 14}}
    relative = RelativeActionsProcessorStep(enabled=True)
    pre = PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats), relative,
    ])
    post = PolicyProcessorPipeline(name="policy_postprocessor", steps=[
        AbsoluteActionsProcessorStep(enabled=True, relative_step=relative),
        UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
    ])
    pre.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    post.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    return checkpoint


def test_large_action_offset_roundtrip_uses_native_intermediate_precision_scale(checkpoints):
    checkpoint = _large_offset_temporal_checkpoint(checkpoints)
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["native_roundtrip_passed"] is True
    assert finding.details["native_roundtrip_max_abs_error"] > 1e-5 + .9e-5
    budget = finding.details["roundtrip_precision_budget"]
    assert budget["max_intermediate_scale"] > 2048
    assert budget["machine_epsilon"] == torch.finfo(torch.float32).eps
    assert finding.details["roundtrip_error_tolerance"]["atol"] == pytest.approx(budget["roundoff_atol"])
    assert any(source["pipeline_role"] == "postprocessor" and source["source"] == "step_output"
               for source in budget["sources"])
    assert any(source["tensor_key"] == "action.mean" and source["max_absolute_value"] == 2048
               for source in budget["sources"])
    assert finding.details["actual_anchor_index"] == 0


def test_scale_aware_tolerance_does_not_hide_a_genuine_large_offset_inverse_error(checkpoints, monkeypatch):
    checkpoint = _large_offset_temporal_checkpoint(checkpoints)
    native = UnnormalizerProcessorStep.__call__

    def bad_inverse(self, transition):
        result = native(self, transition)
        return {**result, "action": result["action"] + 0.1}

    monkeypatch.setattr(UnnormalizerProcessorStep, "__call__", bad_inverse)
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["native_roundtrip_passed"] is False
    assert finding.details["native_roundtrip_max_abs_error"] > finding.details["roundtrip_error_tolerance"]["atol"]


def test_scale_aware_tolerance_preserves_wrong_history_anchor_detection(checkpoints):
    checkpoint = _large_offset_temporal_checkpoint(checkpoints, count=3)
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["native_roundtrip_passed"] is True
    assert finding.details["actual_anchor_index"] == 0
    assert finding.details["expected_anchor_index"] == 2


def test_counterexample_g_postprocessor_action_gain_is_observed(checkpoints):
    from safetensors.torch import load_file, save_file

    checkpoint = checkpoints["clean_act_modern"]
    pre_path = checkpoint / "preprocessor_config.json"
    config = json.loads(pre_path.read_text())
    # This observation-only preprocessor leaves the recorded normalized action at 0.8.
    config["steps"][0]["config"]["norm_map"]["ACTION"] = "IDENTITY"
    pre_path.write_text(json.dumps(config))
    post = json.loads((checkpoint / "postprocessor_config.json").read_text())
    state_path = checkpoint / post["steps"][0]["state_file"]
    state = load_file(state_path)
    state["action.std"][3] = 6e4
    save_file(state, state_path)
    np.savez(checkpoint / "sample_action.npz", action=np.full((1, 14), 0.8, np.float32))
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = _find(findings, "EMPIRICAL_GAIN_EXPLOSION", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    finding = failures[0]
    assert finding.details["pipeline_role"] == "postprocessor"
    assert finding.details["step_index"] == 0
    assert finding.provenance.violating_indices == [3]
    assert finding.details["input_values"] == pytest.approx([0.8])
    assert finding.details["output_values"] == pytest.approx([48000])


def test_counterexample_g_postprocessor_bypass_is_observed(checkpoints):
    from safetensors.torch import load_file, save_file

    checkpoint = checkpoints["clean_act_modern"]
    config = json.loads((checkpoint / "postprocessor_config.json").read_text())
    state_path = checkpoint / config["steps"][0]["state_file"]
    state = load_file(state_path)
    save_file({key.replace("action.", "wrong_dataset.action."): value for key, value in state.items()}, state_path)
    findings = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    failures = _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.FAIL)
    assert len(failures) == 1
    assert failures[0].details["pipeline_role"] == "postprocessor"
    assert failures[0].details["step_index"] == 0
    assert "wrong_dataset.action.std" in failures[0].details["actual_stat_keys"]


def test_test_only_fixture_sidecar_is_never_read_as_production_chronology(checkpoints):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    (checkpoint / "metadata.json").unlink()
    # A test-only file cannot manufacture certainty about the current frame.
    (checkpoint / ("fixture_" + "metadata.json")).write_text(json.dumps({
        "n_obs_steps": 3, "observation_order": "newest_first", "current_index": 0,
    }))
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert "n_obs_steps" in finding.details["missing_declarations"]


def test_chronology_sidecars_are_explicit_opt_in(checkpoints):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    # A plausible generic sidecar is ignored until supplied by API/CLI caller.
    implicit, = run_temporal_audit(checkpoint)
    assert implicit.status is DiagnosticStatus.UNKNOWN
    assert "n_obs_steps" in implicit.details["missing_declarations"]
    explicit, = run_temporal_audit(checkpoint, metadata_path=checkpoint / "metadata.json")
    assert explicit.status is DiagnosticStatus.FAIL
    assert explicit.details["actual_anchor_index"] == 0
    assert explicit.details["expected_anchor_index"] == 2


@pytest.mark.parametrize("extra_observation", [False, True])
def test_missing_active_state_sample_cannot_borrow_action_coverage(
    checkpoints, tmp_path, monkeypatch, extra_observation,
) -> None:
    import lerobot_preflight.trace as trace_module

    checkpoint = checkpoints["clean_act_modern"]
    sample = {"action": np.full((1, 14), 0.4, dtype=np.float32)}
    if extra_observation:
        sample["observation.unrelated"] = np.ones((1, 14), dtype=np.float32)
    path = tmp_path / "missing_state.npz"
    np.savez(path, **sample)
    monkeypatch.setattr(trace_module, "_trace_pipeline", lambda *args, **kwargs: pytest.fail("step 0 executed"))
    findings = run_offline_trace(checkpoint, path)
    finding, = _find(findings, "MISSING_REQUIRED_SAMPLE_FEATURE", DiagnosticStatus.FAIL)
    assert finding.details["feature_key"] == "observation.state"
    assert finding.details["step_index"] == 0
    assert finding.details["required_coverage"] is True
    assert finding.details["processor_steps_executed"] == 0
    assert finding.provenance.file_path == str(path)
    assert not _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.PASS)


def test_missing_policy_state_required_even_when_normalizer_is_identity(checkpoints, tmp_path) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["norm_map"]["STATE"] = "IDENTITY"
    path.write_text(json.dumps(config))
    (checkpoint / "config.json").write_text(json.dumps({
        "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
    }))
    sample = tmp_path / "action_only.npz"
    np.savez(sample, action=np.zeros((1, 14), dtype=np.float32))
    finding, = run_offline_trace(checkpoint, sample)
    assert finding.check_id == "MISSING_REQUIRED_SAMPLE_FEATURE"
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["feature_key"] == "observation.state"


@pytest.mark.parametrize("missing_environment", [False, True])
def test_active_environment_sample_coverage_follows_native_rename(
    checkpoints, tmp_path, missing_environment,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }
    stats = {key: {"mean": torch.zeros(feature.shape), "std": torch.ones(feature.shape) * 0.5}
             for key, feature in features.items()}
    pipeline = PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        RenameObservationsProcessorStep(rename_map={
            "observation.raw_state": "observation.state",
            "observation.raw_environment": "observation.environment_state",
        }),
        NormalizerProcessorStep(features=features, norm_map={
            FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ENV: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }, stats=stats),
    ])
    pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    (checkpoint / "config.json").write_text(json.dumps({"input_features": {
        key: {"type": feature.type.value, "shape": list(feature.shape)}
        for key, feature in features.items() if feature.type != FeatureType.ACTION
    }}))
    sample = {"observation.raw_state": np.ones((1, 14), dtype=np.float32) * 0.2,
              "action": np.ones((1, 14), dtype=np.float32) * 0.4}
    if not missing_environment:
        sample["observation.raw_environment"] = np.ones((1, 2), dtype=np.float32) * 0.2
    path = tmp_path / "renamed_environment.npz"
    np.savez(path, **sample)
    findings = run_offline_trace(checkpoint, path)
    if missing_environment:
        finding, = _find(findings, "MISSING_REQUIRED_SAMPLE_FEATURE", DiagnosticStatus.FAIL)
        assert finding.details["feature_key"] == "observation.environment_state"
        assert finding.details["step_index"] == 1
        assert not _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.PASS)
    else:
        assert not _find(findings, "MISSING_REQUIRED_SAMPLE_FEATURE")
        assert _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.PASS)
        assert all(f.status not in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT} for f in findings)


@pytest.mark.parametrize("n_obs_steps", [1, 3])
def test_undeclared_action_width_never_invents_one_coordinate(checkpoints, n_obs_steps, monkeypatch):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    del config["output_features"]
    path.write_text(json.dumps(config))
    metadata = {"n_obs_steps": n_obs_steps}
    if n_obs_steps > 1:
        metadata.update(observation_order="newest_first", current_index=0)
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))

    def forbidden_probe(*args, **kwargs):
        pytest.fail("A synthetic action width was invented before compatibility was established")

    monkeypatch.setattr(RelativeActionsProcessorStep, "_build_mask", forbidden_probe)
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert "Undeclared action dimension prevents temporal compatibility verification" in finding.summary
    assert finding.details["declared_action_dim"] is None
    assert finding.details["processor_steps_executed"] == 0
    assert "synthetic_state_shape" not in finding.details


def test_explicit_action_names_can_supply_temporal_action_width(checkpoints):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    del config["output_features"]
    path.write_text(json.dumps(config))
    (checkpoint / "metadata.json").write_text(json.dumps({"n_obs_steps": 1}))
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["action_names"] = [f"joint_{index}" for index in range(14)]
    path.write_text(json.dumps(config))
    finding, = _audit_temporal(checkpoint)
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["declared_action_dim"] == 14
    assert finding.details["dimension_compatibility_verified"] is True


@pytest.mark.parametrize("shape", [(1, 1), (1,), (), (2, 13), (14, 1)])
def test_sample_shape_cannot_broadcast_a_narrow_state_across_declared_features(
    checkpoints, tmp_path, monkeypatch, shape,
):
    import lerobot_preflight.trace as trace_module

    checkpoint = checkpoints["clean_act_modern"]
    path = tmp_path / "bad_shape.npz"
    np.savez(path, **{"observation.state": np.full(shape, 0.2, dtype=np.float32),
                     "action": np.full((1, 14), 0.4, dtype=np.float32)})
    monkeypatch.setattr(trace_module, "_trace_pipeline", lambda *args, **kwargs: pytest.fail("step 0 executed"))
    finding, = run_offline_trace(checkpoint, path)
    assert finding.check_id == "INVALID_SAMPLE_SHAPE"
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["expected_feature_shape"] == [14]
    assert finding.details["actual_shape"] == list(shape)
    assert finding.details["sample_feature_key"] == "observation.state"
    assert finding.details["processor_steps_executed"] == 0
    assert finding.provenance.file_path == str(path)
    assert finding.provenance.sha256_prefix == hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    assert finding.provenance.tensor_key == "observation.state"
    assert finding.provenance.violating_indices


@pytest.mark.parametrize("shape", [(14,), (1, 14), (2, 3, 14)])
def test_sample_shape_accepts_exact_features_with_optional_batch_and_time(checkpoints, tmp_path, shape):
    checkpoint = checkpoints["clean_act_modern"]
    path = tmp_path / "valid_shape.npz"
    np.savez(path, **{"observation.state": np.full(shape, 0.2, dtype=np.float32),
                     "action": np.full(shape, 0.4, dtype=np.float32)})
    findings = run_offline_trace(checkpoint, path)
    assert not _find(findings, "INVALID_SAMPLE_SHAPE")
    assert _find(findings, "SILENT_STEP_BYPASS", DiagnosticStatus.PASS)
    assert not any(finding.status in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT} for finding in findings)


def test_action_shape_failure_points_to_adjacent_action_npz(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "sample_action.npz"
    np.savez(path, action=np.full((1, 1), 0.4, dtype=np.float32))
    finding, = run_offline_trace(checkpoint, _observation_sample(checkpoint))
    assert finding.check_id == "INVALID_SAMPLE_SHAPE"
    assert finding.details["expected_feature_shape"] == [14]
    assert finding.details["actual_shape"] == [1, 1]
    assert finding.provenance.file_path == str(path)
    assert finding.provenance.tensor_key == "action"


@pytest.mark.parametrize("bad_key,bad_shape,expected", [
    ("observation.raw_environment", (1, 1), [2]),
    ("observation.images.top", (1, 1, 4, 5), [3, 4, 5]),
])
def test_present_environment_and_visual_shapes_are_checked_through_native_renames(
    checkpoints, tmp_path, bad_key, bad_shape, expected,
):
    checkpoint = checkpoints["clean_act_modern"]
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
        "observation.environment_state": PolicyFeature(FeatureType.ENV, (2,)),
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 4, 5)),
        "action": PolicyFeature(FeatureType.ACTION, (14,)),
    }
    stats = {key: {"mean": torch.zeros(feature.shape), "std": torch.ones(feature.shape) * 0.5}
             for key, feature in features.items() if feature.type is not FeatureType.VISUAL}
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        RenameObservationsProcessorStep(rename_map={"observation.raw_environment": "observation.environment_state"}),
        NormalizerProcessorStep(features=features, norm_map={
            FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ENV: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD, FeatureType.VISUAL: NormalizationMode.IDENTITY,
        }, stats=stats),
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    sample = {"observation.state": np.full((1, 14), 0.2, dtype=np.float32),
              "observation.raw_environment": np.full((1, 2), 0.2, dtype=np.float32),
              "observation.images.top": np.zeros((1, 3, 4, 5), dtype=np.float32),
              "action": np.full((1, 14), 0.4, dtype=np.float32)}
    sample[bad_key] = np.zeros(bad_shape, dtype=np.float32)
    path = tmp_path / "bad_environment_or_visual.npz"
    np.savez(path, **sample)
    finding, = run_offline_trace(checkpoint, path)
    assert finding.check_id == "INVALID_SAMPLE_SHAPE"
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["sample_feature_key"] == bad_key
    assert finding.details["expected_feature_shape"] == expected
    assert finding.details["actual_shape"] == list(bad_shape)
    assert finding.details["step_index"] == 1


def test_identity_policy_feature_shape_is_still_enforced(checkpoints, tmp_path):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["norm_map"]["STATE"] = "IDENTITY"
    path.write_text(json.dumps(config))
    (checkpoint / "config.json").write_text(json.dumps({"input_features": {
        "observation.state": {"type": "STATE", "shape": [14]},
    }}))
    sample = tmp_path / "bad_identity_shape.npz"
    np.savez(sample, **{"observation.state": np.zeros((1, 1), dtype=np.float32)})
    finding, = run_offline_trace(checkpoint, sample)
    assert finding.check_id == "INVALID_SAMPLE_SHAPE"
    assert finding.status is DiagnosticStatus.FAIL
