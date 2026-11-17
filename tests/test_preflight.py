"""Read-only checkpoint preflight tests against real LeRobot artifacts."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from safetensors.torch import load_file, save_file

from lerobot_preflight.models import (
    DiagnosticFinding,
    DiagnosticStatus,
    PreflightReport,
    Provenance,
    Severity,
)
from lerobot_preflight.preflight import run_preflight
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path)


def _matching(report, check_id: str):
    return [finding for finding in report.findings if finding.check_id == check_id]


def test_inspected_relative_only_pair_needs_no_normalization_buffers(checkpoints):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    report = run_preflight(checkpoint)
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        finding, = _matching(report, check)
        assert finding.status is DiagnosticStatus.PASS
        assert finding.details["all_active_steps_screened"] is True
        assert finding.details["normalization_statistics_required"] is False
        assert finding.details["screened_files"] == []
    assert not PreflightReport(str(checkpoint), report.findings, strict=True).has_failures()


def test_uninspected_numeric_pair_does_not_gain_a_no_stats_pass(checkpoints):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].append({"registry_name": "uninspected_numeric_processor", "config": {}})
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        finding, = _matching(report, check)
        assert finding.status is DiagnosticStatus.UNKNOWN
        assert finding.details["all_active_steps_screened"] is False


def _status(finding) -> str:
    return finding.status.value


def test_clean_checkpoint_has_no_failures_and_explicit_semantic_unknowns(
    checkpoints: dict[str, Path],
) -> None:
    report = run_preflight(checkpoints["clean_act_modern"])
    assert not report.has_failures()
    assert all(_status(finding) not in {"FAIL", "CONFLICT"} for finding in report.findings)
    unknowns = [
        finding
        for finding in _matching(report, "PHYSICAL_SEMANTICS")
        if _status(finding) == "UNKNOWN"
    ]
    assert len(unknowns) == 3
    assert {finding.details["field"] for finding in unknowns} == {
        "joint_angle_units",
        "action_coordinate_frame",
        "temporal_observation_ordering",
    }
    assert any(
        _status(finding) == "UNKNOWN" for finding in _matching(report, "FEATURE_CONTRACT")
    )

    serialized = report.to_dict()
    assert isinstance(serialized["findings"], list)
    assert len(serialized["findings"]) == len(report.findings)
    json.dumps(serialized, allow_nan=False)
    assert all(isinstance(finding.status, DiagnosticStatus) for finding in report.findings)
    assert all(isinstance(finding.severity, Severity) for finding in report.findings)


def test_declared_physical_semantics_remove_the_three_unknowns(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "physical_semantics": {
                    "joint_angle_units": "radians",
                    "action_coordinate_frame": "robot_base",
                    "temporal_observation_ordering": "oldest_first",
                }
            }
        )
    )
    report = run_preflight(checkpoint, metadata_path=checkpoint / "metadata.json")
    semantics = _matching(report, "PHYSICAL_SEMANTICS")
    assert {finding.details["field"] for finding in semantics} == {
        "joint_angle_units",
        "action_coordinate_frame",
        "temporal_observation_ordering",
    }
    assert all(_status(finding) == "PASS" for finding in semantics)
    assert not report.has_failures()


def test_explicit_metadata_retains_utf8_declarations_under_ansi_locale(checkpoints, tmp_path, monkeypatch):
    checkpoint = checkpoints["bug_4415_silent_bypass"]
    metadata = tmp_path / "physical_metadata.json"
    metadata.write_text(json.dumps({"joint_angle_units": "弧度", "action_coordinate_frame": "base_é"},
                                   ensure_ascii=False), encoding="utf-8")
    original = Path.open

    def ansi_open(path, mode="r", *args, **kwargs):
        if path == metadata and "b" not in mode:
            kwargs.setdefault("encoding", "cp1252")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", ansi_open)
    report = run_preflight(checkpoint, metadata_path=metadata)
    assert any(finding.check_id == "STAT_KEY_NAMESPACE_MISMATCH" and finding.status is DiagnosticStatus.FAIL
               for finding in report.findings)
    values = {finding.details["field"]: finding.details.get("value")
              for finding in report.findings if finding.check_id == "PHYSICAL_SEMANTICS"}
    assert values["joint_angle_units"] == "弧度"
    assert values["action_coordinate_frame"] == "base_é"


def test_report_model_has_failures_and_strict_json_serialization() -> None:
    provenance = Provenance(
        file_path="model.safetensors",
        step_name="normalizer_processor",
        tensor_key="observation.state.std",
        sha256_prefix="0123456789ab",
        violating_indices=[4],
    )
    unknown = DiagnosticFinding(
        check_id="PHYSICAL_SEMANTICS",
        status=DiagnosticStatus.UNKNOWN,
        severity=Severity.INFO,
        summary="Unit semantics are undeclared",
    )
    failure = DiagnosticFinding(
        check_id="FP16_NON_FINITE_RISK",
        status=DiagnosticStatus.FAIL,
        severity=Severity.CRITICAL,
        summary="Zero denominator in FP16",
        details={"value": torch.tensor(float("nan")), "gain": float("inf")},
        provenance=provenance,
    )
    conflict = DiagnosticFinding(
        check_id="FEATURE_CONTRACT",
        status=DiagnosticStatus.CONFLICT,
        severity=Severity.WARNING,
        summary="Feature shapes differ",
    )
    assert not PreflightReport(checkpoint_dir="unused", findings=[unknown]).has_failures()
    assert PreflightReport(checkpoint_dir="unused", findings=[failure]).has_failures()
    assert PreflightReport(checkpoint_dir="unused", findings=[conflict]).has_failures()
    report_dict = PreflightReport(checkpoint_dir="unused", findings=[unknown, failure]).to_dict()
    assert report_dict["findings"][1]["details"] == {"value": "NaN", "gain": "Infinity"}
    json.dumps(report_dict, allow_nan=False)


def test_legacy_embedded_stats_report_schema_generation_mismatch(
    checkpoints: dict[str, Path],
) -> None:
    report = run_preflight(checkpoints["legacy_act_embedded"])
    findings = _matching(report, "SCHEMA_GENERATION_MISMATCH")
    assert len(findings) == 1
    finding = findings[0]
    assert _status(finding) in {"FAIL", "CONFLICT"}
    assert Path(finding.provenance.file_path).name == "model.safetensors"
    assert finding.provenance.tensor_key == "unnormalize_outputs.buffer_action.mean"
    assert finding.provenance.sha256_prefix
    assert report.has_failures()


def test_silent_bypass_is_reported_as_observation_contract_failure(
    checkpoints: dict[str, Path],
) -> None:
    report = run_preflight(checkpoints["bug_4415_silent_bypass"])
    failures = [
        finding
        for finding in _matching(report, "FEATURE_CONTRACT")
        if _status(finding) == "FAIL" and finding.details.get("feature_key") == "observation.state"
    ]
    assert failures
    assert any(finding.details.get("step_index") == 0 for finding in failures)
    assert any(Path(finding.provenance.file_path).suffix == ".safetensors" for finding in failures)
    assert report.has_failures()

    namespace = [
        finding for finding in _matching(report, "STAT_KEY_NAMESPACE_MISMATCH")
        if finding.status is DiagnosticStatus.FAIL
    ]
    assert len(namespace) == 1
    mismatch = namespace[0]
    assert mismatch.details["feature_key"] == "observation.state"
    assert mismatch.details["step_index"] == 0
    assert set(mismatch.details["expected_buffer_keys"]) == {
        "observation.state.mean", "observation.state.std",
    }
    assert set(mismatch.details["matching_prefixed_keys"]) == {
        "wrong_dataset/observation.state.mean",
        "wrong_dataset/observation.state.std",
    }
    assert Path(mismatch.provenance.file_path).suffix == ".safetensors"
    assert mismatch.provenance.violating_indices == []


def test_pathological_std_reports_coordinate_and_saved_gain(
    checkpoints: dict[str, Path],
) -> None:
    report = run_preflight(checkpoints["bug_5289_pathological_gain"])
    findings = _matching(report, "PATHOLOGICAL_STD_GAIN")
    assert len(findings) == 1
    finding = findings[0]
    assert _status(finding) == "FAIL"
    assert finding.details["feature_key"] == "observation.state"
    assert finding.details["step_index"] == 0
    assert "saved_values" in finding.details
    assert "amplification_factors" in finding.details
    assert "tensor_slice" in finding.details
    assert finding.provenance.violating_indices == [9]
    assert finding.details["saved_values"] == pytest.approx([1.48e-6])
    assert finding.details["amplification_factors"] == pytest.approx([1 / 1.48e-6], rel=1e-5)
    assert finding.details["tensor_slice"] == pytest.approx([1.48e-6])
    assert finding.severity is Severity.CRITICAL
    assert Path(finding.provenance.file_path).suffix == ".safetensors"
    assert finding.provenance.step_name == "normalizer_processor"
    assert finding.provenance.tensor_key == "observation.state.std"
    saved_bytes = Path(finding.provenance.file_path).read_bytes()
    assert finding.provenance.sha256_prefix == hashlib.sha256(saved_bytes).hexdigest()[:12]
    assert report.has_failures()


def test_std_threshold_controls_pathological_gain_finding(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    assert any(
        _status(finding) == "FAIL"
        for finding in _matching(run_preflight(checkpoint), "PATHOLOGICAL_STD_GAIN")
    )
    assert not any(
        _status(finding) == "FAIL"
        for finding in _matching(
            run_preflight(checkpoint, std_warn_threshold=1e-7), "PATHOLOGICAL_STD_GAIN"
        )
    )


def test_fp16_zero_std_reports_nonfinite_risk_and_coordinate(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_4973_fp16_nan"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(0, {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": "float16"}})
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    findings = _matching(report, "FP16_NON_FINITE_RISK")
    assert len(findings) == 1
    finding = findings[0]
    assert _status(finding) == "FAIL"
    assert finding.details["feature_key"] == "observation.state"
    assert finding.provenance.violating_indices == [4]
    assert finding.provenance.tensor_key == "observation.state.std"
    assert finding.severity is Severity.CRITICAL
    assert "auto-casts statistics to the input dtype" in finding.summary
    assert finding.details["requires_legacy_buffer_half_cast_or_modern_migration"] is False
    assert report.has_failures()


def test_preflight_ignores_sample_npz_and_never_opens_a_socket(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    for path in checkpoint.glob("*.npz"):
        path.unlink()
    before = {p.relative_to(checkpoint).as_posix(): p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}

    def forbid_socket(*args, **kwargs):
        pytest.fail("preflight attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    report = run_preflight(checkpoint)
    after = {p.relative_to(checkpoint).as_posix(): p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    assert not report.has_failures()
    assert after == before


def test_native_pipeline_state_files_are_found_with_custom_pipeline_names(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom_names"
    checkpoint.mkdir()
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }
    stats = {
        key: {"mean": torch.zeros(14), "std": torch.full((14,), 0.5)} for key in features
    }
    preprocessor = PolicyProcessorPipeline(
        name="arbitrary input pipeline",
        steps=[
            NormalizerProcessorStep(
                features=features,
                norm_map={
                    FeatureType.STATE: NormalizationMode.MEAN_STD,
                    FeatureType.ACTION: NormalizationMode.MEAN_STD,
                },
                stats=stats,
            )
        ],
    )
    postprocessor = PolicyProcessorPipeline(
        name="arbitrary output pipeline",
        steps=[
            UnnormalizerProcessorStep(
                features={"action": features["action"]},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={"action": stats["action"]},
            )
        ],
    )
    preprocessor.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    postprocessor.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    pre_config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    post_config = json.loads((checkpoint / "postprocessor_config.json").read_text())
    assert pre_config["steps"][0]["state_file"].startswith("arbitrary_input_pipeline_")
    assert post_config["steps"][0]["state_file"].startswith("arbitrary_output_pipeline_")
    assert not run_preflight(checkpoint).has_failures()


def test_native_default_processor_json_names_are_discovered(
    checkpoints: dict[str, Path],
) -> None:
    source = checkpoints["clean_act_modern"]
    checkpoint = source.parent / "canonical_native_names"
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
    assert (checkpoint / "policy_preprocessor.json").is_file()
    assert (checkpoint / "policy_postprocessor.json").is_file()
    assert not run_preflight(checkpoint).has_failures()


def test_unsupported_native_step_reports_unknown_without_loading_it(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    for filename in ("preprocessor_config.json", "postprocessor_config.json"):
        path = checkpoint / filename
        config = json.loads(path.read_text())
        config["steps"][0]["registry_name"] = "untrusted_custom_step"
        path.write_text(json.dumps(config))

    def forbid_socket(*args, **kwargs):
        pytest.fail("unsupported step triggered a network socket")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    report = run_preflight(checkpoint)
    contract = _matching(report, "FEATURE_CONTRACT")
    assert sum(_status(finding) == "UNKNOWN" for finding in contract) >= 2
    assert not report.has_failures()


def test_unsupported_earlier_step_does_not_hide_later_normalizer_bypass(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["bug_4415_silent_bypass"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(0, {"registry_name": "untrusted_custom_step", "config": {}})
    path.write_text(json.dumps(config))

    def forbid_socket(*args, **kwargs):
        pytest.fail("unsupported step triggered a network socket")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    report = run_preflight(checkpoint)
    assert any(
        _status(finding) == "FAIL"
        and finding.details.get("feature_key") == "observation.state"
        and finding.details.get("step_index") == 1
        for finding in _matching(report, "FEATURE_CONTRACT")
    )
    assert report.has_failures()


def test_missing_local_state_file_fails_without_network_fallback(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    (checkpoint / config["steps"][0]["state_file"]).unlink()

    def forbid_socket(*args, **kwargs):
        pytest.fail("missing local state triggered a network socket")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    report = run_preflight(checkpoint)
    assert any(_status(finding) == "FAIL" for finding in _matching(report, "FEATURE_CONTRACT"))
    assert report.has_failures()


def test_policy_config_feature_contract_disagreement_is_conflict(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [7]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [14]}},
            }
        )
    )
    report = run_preflight(checkpoint)
    conflicts = [
        finding
        for finding in _matching(report, "FEATURE_CONTRACT")
        if _status(finding) == "CONFLICT" and finding.details.get("feature_key") == "observation.state"
    ]
    assert conflicts
    assert report.has_failures()


def _overwrite_observation_normalizer(
    checkpoint: Path,
    *,
    mode: NormalizationMode,
    observation_stats: dict[str, torch.Tensor],
    eps: float = 1e-8,
    extra_features: dict[str, PolicyFeature] | None = None,
    extra_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    extra_norm_map: dict[FeatureType, NormalizationMode] | None = None,
    action_stat_key: str = "action",
) -> None:
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }
    features.update(extra_features or {})
    stats = {
        "observation.state": observation_stats,
        action_stat_key: {"mean": torch.zeros(14), "std": torch.full((14,), 0.5)},
    }
    stats.update(extra_stats or {})
    norm_map = {FeatureType.STATE: mode, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    norm_map.update(extra_norm_map or {})
    pipeline = PolicyProcessorPipeline(
        name="policy_preprocessor",
        steps=[
            NormalizerProcessorStep(
                features=features,
                norm_map=norm_map,
                stats=stats,
                eps=eps,
            )
        ],
    )
    pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")


def _overwrite_action_unnormalizer(
    checkpoint: Path,
    action_std: torch.Tensor,
    *,
    eps: float = 1e-8,
    extra_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    stats = {"action": {"mean": torch.zeros(14), "std": action_std}}
    stats.update(extra_stats or {})
    pipeline = PolicyProcessorPipeline(
        name="policy_postprocessor",
        steps=[
            UnnormalizerProcessorStep(
                features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats=stats,
                eps=eps,
            )
        ],
    )
    pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")


def test_prefixed_action_buffers_silently_bypass_native_step_and_expose_namespace(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": torch.full((14,), 0.5)},
        action_stat_key="so100-blue.buffer.action",
    )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True
    )
    batch = {
        "observation.state": torch.full((1, 14), 0.2),
        "action": torch.full((1, 14), 0.4),
    }
    output = preprocessor(batch)
    torch.testing.assert_close(output["observation.state"], torch.full((1, 14), 0.4))
    torch.testing.assert_close(output["action"], batch["action"])

    report = run_preflight(checkpoint)
    namespace = [
        finding for finding in _matching(report, "STAT_KEY_NAMESPACE_MISMATCH")
        if finding.status is DiagnosticStatus.FAIL and finding.details.get("feature_key") == "action"
    ]
    assert len(namespace) == 1
    mismatch = namespace[0]
    assert mismatch.details["step_index"] == 0
    assert set(mismatch.details["expected_buffer_keys"]) == {"action.mean", "action.std"}
    assert set(mismatch.details["matching_prefixed_keys"]) == {
        "so100-blue.buffer.action.mean", "so100-blue.buffer.action.std",
    }
    assert {"observation.state", "action"} <= set(mismatch.details["declared_feature_keys"])
    assert "so100-blue.buffer.action" in mismatch.details["actual_feature_keys"]
    assert mismatch.provenance.violating_indices == []
    assert mismatch.provenance.tensor_key in {"action.mean", "action.std"}
    assert Path(mismatch.provenance.file_path).suffix == ".safetensors"


def test_min_max_equal_bounds_are_fp16_nonfinite_risk(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    max_values = torch.ones(14)
    max_values[4] = 0
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.MIN_MAX,
        observation_stats={"min": torch.zeros(14), "max": max_values},
    )
    report = run_preflight(checkpoint)
    findings = _matching(report, "FP16_NON_FINITE_RISK")
    assert any(
        finding.details.get("feature_key") == "observation.state"
        and 4 in (finding.provenance.violating_indices or [])
        for finding in findings
    )
    gain, = [finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN")
             if finding.status is DiagnosticStatus.FAIL]
    assert gain.provenance.violating_indices == [4]
    assert gain.severity is Severity.CRITICAL
    assert gain.details["gain_formula"] == "2 / where(max - min == 0, eps, max - min)"
    assert gain.details["baseline_output_values"] == [-1.0]
    assert gain.details["probe_output_values"] == [40_000_000.0]
    assert gain.details["effective_gains_with_eps"] == pytest.approx([2e8])


def test_nonzero_std_that_underflows_in_fp16_is_reported(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    std_values = torch.full((14,), 0.5)
    std_values[3] = 1e-8
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": std_values},
    )
    findings = _matching(run_preflight(checkpoint), "FP16_NON_FINITE_RISK")
    assert any(
        finding.details.get("feature_key") == "observation.state"
        and 3 in (finding.provenance.violating_indices or [])
        for finding in findings
    )


def test_unreferenced_metadata_stats_do_not_create_fp16_or_gain_failures(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    extra_stats = {
        "task_index": {
            "mean": torch.zeros(1), "std": torch.zeros(1),
            "min": torch.zeros(1), "max": torch.zeros(1),
        },
        "index": {
            "mean": torch.tensor([92_767_248.0]),
            "std": torch.tensor([1e-12]),
            "min": torch.zeros(1), "max": torch.tensor([185_534_496.0]),
        },
        "frame_index": {
            "mean": torch.tensor([35_559.0]),
            "std": torch.tensor([1e-12]),
            "min": torch.zeros(1), "max": torch.tensor([88_049.0]),
        },
    }
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": torch.full((14,), 0.5)},
        extra_stats=extra_stats,
    )
    _overwrite_action_unnormalizer(
        checkpoint, torch.full((14,), 0.5), extra_stats=extra_stats
    )
    report = run_preflight(checkpoint)
    assert not any(
        finding.status is DiagnosticStatus.FAIL
        for check in ("FP16_NON_FINITE_RISK", "PATHOLOGICAL_STD_GAIN")
        for finding in _matching(report, check)
    )
    assert not report.has_failures()


def test_identity_language_stats_are_not_normalized_even_when_declared(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    encoding = "observation.task_encoding"
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": torch.full((14,), 0.5)},
        extra_features={encoding: PolicyFeature(type=FeatureType.LANGUAGE, shape=(6,))},
        extra_stats={
            encoding: {
                "mean": torch.zeros(6), "std": torch.zeros(6),
                "min": torch.zeros(6), "max": torch.zeros(6),
            }
        },
        extra_norm_map={FeatureType.LANGUAGE: NormalizationMode.IDENTITY},
    )
    report = run_preflight(checkpoint)
    assert not any(
        finding.status is DiagnosticStatus.FAIL and finding.details.get("feature_key") == encoding
        for check in ("FP16_NON_FINITE_RISK", "PATHOLOGICAL_STD_GAIN")
        for finding in _matching(report, check)
    )
    assert not report.has_failures()


def test_inverse_zero_action_std_is_dead_dimension_without_denominator_risk(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    std = torch.full((14,), 0.5)
    std[4] = 0
    _overwrite_action_unnormalizer(checkpoint, std)
    report = run_preflight(checkpoint)
    dead = [
        finding for finding in _matching(report, "DEAD_ACTION_DIMENSION")
        if finding.status is DiagnosticStatus.FAIL
    ]
    assert len(dead) == 1
    assert dead[0].details["feature_key"] == "action"
    assert dead[0].details["step_index"] == 0
    assert dead[0].details["tensor_slice"] == [0.0]
    assert dead[0].provenance.tensor_key == "action.std"
    assert dead[0].provenance.violating_indices == [4]
    assert not any(
        finding.status is DiagnosticStatus.FAIL and finding.details.get("feature_key") == "action"
        for check in ("FP16_NON_FINITE_RISK", "PATHOLOGICAL_STD_GAIN")
        for finding in _matching(report, check)
    )


def test_effective_forward_gain_respects_saved_epsilon(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    std = torch.full((14,), 0.5)
    std[9] = 1.48e-6
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": std},
        eps=1e-3,
    )
    report = run_preflight(checkpoint, std_warn_threshold=1e-4)
    assert not any(
        finding.status is DiagnosticStatus.FAIL
        and finding.details.get("feature_key") == "observation.state"
        for finding in _matching(report, "PATHOLOGICAL_STD_GAIN")
    )


def test_min_max_ignores_unselected_std_and_mean_stats(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.MIN_MAX,
        observation_stats={
            "min": torch.zeros(14), "max": torch.ones(14),
            "mean": torch.full((14,), 1e6), "std": torch.zeros(14),
        },
    )
    report = run_preflight(checkpoint)
    assert not any(
        finding.status is DiagnosticStatus.FAIL
        and finding.details.get("feature_key") == "observation.state"
        for check in ("FP16_NON_FINITE_RISK", "PATHOLOGICAL_STD_GAIN")
        for finding in _matching(report, check)
    )


def test_quantiles_screen_only_active_quantile_pair(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    q99 = torch.ones(14)
    q99[7] = 0
    _overwrite_observation_normalizer(
        checkpoint,
        mode=NormalizationMode.QUANTILES,
        observation_stats={
            "q01": torch.zeros(14), "q99": q99,
            "min": torch.zeros(14), "max": torch.zeros(14),
            "mean": torch.full((14,), 1e6), "std": torch.zeros(14),
        },
    )
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(0, {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": "float16"}})
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    failures = [
        finding for finding in _matching(report, "FP16_NON_FINITE_RISK")
        if finding.status is DiagnosticStatus.FAIL
        and finding.details.get("feature_key") == "observation.state"
    ]
    assert failures
    assert {finding.provenance.tensor_key for finding in failures} <= {
        "observation.state.q01", "observation.state.q99",
    }
    assert any(7 in (finding.provenance.violating_indices or []) for finding in failures)
    gain, = [finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN")
             if finding.status is DiagnosticStatus.FAIL
             and finding.details.get("feature_key") == "observation.state"]
    assert gain.provenance.tensor_key == "observation.state.q01"
    assert gain.provenance.violating_indices == [7]


def test_known_molmo_masked_steps_keep_native_indices_and_inverse_direction(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    std = torch.full((14,), 0.5)
    std[4] = 0
    _overwrite_action_unnormalizer(checkpoint, std)
    for filename, registry_name in (
        ("preprocessor_config.json", "molmoact2_masked_normalizer"),
        ("postprocessor_config.json", "molmoact2_masked_unnormalizer"),
    ):
        path = checkpoint / filename
        config = json.loads(path.read_text())
        config["steps"][0]["registry_name"] = registry_name
        path.write_text(json.dumps(config))

    def deny_policy_import(name: str, *args, **kwargs):
        if "molmoact2" in name:
            pytest.fail("preflight imported policy code from a checkpoint registry name")
        return original_import(name, *args, **kwargs)

    import builtins
    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", deny_policy_import)
    report = run_preflight(checkpoint)
    gain = [
        finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN")
        if finding.status is DiagnosticStatus.FAIL
    ]
    dead = [
        finding for finding in _matching(report, "DEAD_ACTION_DIMENSION")
        if finding.status is DiagnosticStatus.FAIL
    ]
    assert len(gain) == 1 and gain[0].details["step_index"] == 0
    assert gain[0].details["step_direction"] == "forward"
    assert gain[0].provenance.violating_indices == [9]
    assert len(dead) == 1 and dead[0].details["step_index"] == 0
    assert dead[0].details["step_direction"] == "inverse"
    assert dead[0].provenance.violating_indices == [4]


@pytest.mark.parametrize(
    ("dtype", "eps", "expected_risk"),
    [("float32", 1e-6, False), ("float32", 0.0, False), ("bfloat16", 1e-8, False)],
)
def test_zero_variance_screen_uses_explicit_input_dtype_and_epsilon(
    checkpoints: dict[str, Path], dtype: str, eps: float, expected_risk: bool,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    std = torch.full((14,), 0.5)
    std[4] = 0
    _overwrite_observation_normalizer(
        checkpoint, mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": std}, eps=eps,
    )
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(0, {
        "registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": dtype},
    })
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    risks = [finding for finding in _matching(report, "FP16_NON_FINITE_RISK")
             if finding.status is DiagnosticStatus.FAIL]
    assert bool(risks) == expected_risk
    if eps == 0.0:
        assert any(f.status is DiagnosticStatus.FAIL for f in _matching(report, "NORMALIZATION_CHAIN_NON_FINITE"))
    if risks:
        assert risks[0].details["step_index"] == 1
        assert risks[0].details["screened_dtype"] == dtype
        assert risks[0].provenance.violating_indices == [4]
    gains = [finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN")
             if finding.status is DiagnosticStatus.FAIL]
    assert gains and gains[0].provenance.violating_indices == [4]


def test_observation_selection_excludes_unused_zero_variance_feature(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_4973_fp16_nan"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["normalize_observation_keys"] = ["observation.other"]
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    assert not report.has_failures()


def test_molmo_mask_suppresses_hazard_in_broadcast_scalar_stat(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _overwrite_observation_normalizer(
        checkpoint, mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.tensor(0.0), "std": torch.tensor(0.0)},
    )
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["registry_name"] = "molmoact2_masked_normalizer"
    state_path = checkpoint / config["steps"][0]["state_file"]
    state = load_file(state_path)
    state["observation.state.mask"] = torch.zeros(14)
    save_file(state, state_path)
    path.write_text(json.dumps(config))
    assert not run_preflight(checkpoint).has_failures()


def _write_multi_dataset_buffers(checkpoint: Path, *, bound_state: bool = False) -> None:
    stats = {
        f"{dataset}.buffer.action.{stat}": torch.full((14,), value)
        for dataset in ("so100-blue", "so100-red", "so100")
        for stat, value in (("mean", 0.2), ("std", 0.5))
    }
    for filename in ("preprocessor_config.json", "postprocessor_config.json"):
        config = json.loads((checkpoint / filename).read_text())
        state = dict(stats)
        if bound_state and filename == "preprocessor_config.json":
            state.update({"observation.state.mean": torch.zeros(14),
                          "observation.state.std": torch.full((14,), 0.5)})
        save_file(state, checkpoint / config["steps"][0]["state_file"])


def test_multi_dataset_base_statistics_require_target_dataset_binding(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _write_multi_dataset_buffers(checkpoint)
    native = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True,
    )
    batch = {"observation.state": torch.full((1, 14), 0.7), "action": torch.full((1, 14), 0.4)}
    output = native(batch)
    torch.testing.assert_close(output["observation.state"], batch["observation.state"])
    torch.testing.assert_close(output["action"], batch["action"])
    report = run_preflight(checkpoint)
    findings = _matching(report, "UNBOUND_BASE_MODEL_STATS")
    assert len(findings) == 2  # Both saved/default processor directions need binding.
    assert not _matching(report, "STAT_KEY_NAMESPACE_MISMATCH")
    assert all(finding.status is DiagnosticStatus.FAIL for finding in findings)
    assert all(finding.details["dataset_namespaces"] == ["so100", "so100-blue", "so100-red"] for finding in findings)
    assert all("zero-shot execution" in finding.summary for finding in findings)
    assert all(finding.details["training_provenance_verified"] is False for finding in findings)
    assert {Path(finding.provenance.file_path).suffix for finding in findings} == {".safetensors"}
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        assert any(finding.status is DiagnosticStatus.UNKNOWN for finding in _matching(report, check))
        assert not any(finding.status is DiagnosticStatus.PASS for finding in _matching(report, check))


def test_multi_dataset_action_prefixes_with_bound_state_remain_namespace_mismatch(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _write_multi_dataset_buffers(checkpoint, bound_state=True)
    report = run_preflight(checkpoint)
    assert not _matching(report, "UNBOUND_BASE_MODEL_STATS")
    assert len(_matching(report, "STAT_KEY_NAMESPACE_MISMATCH")) == 2


def test_camera_key_contract_is_advisory_for_identity_passthrough_and_exact_keys(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _overwrite_observation_normalizer(
        checkpoint, mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": torch.full((14,), 0.5)},
        extra_features={"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
        extra_norm_map={FeatureType.VISUAL: NormalizationMode.IDENTITY},
    )
    policy_path = checkpoint / "config.json"
    policy = {"input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
              "output_features": {"action": {"type": "ACTION", "shape": [14]}}}
    policy["input_features"]["observation.images.front"] = {"type": "VISUAL", "shape": [3, 32, 32]}
    policy_path.write_text(json.dumps(policy))
    report = run_preflight(checkpoint)
    finding, = _matching(report, "CAMERA_KEY_CONTRACT_MISMATCH")
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert finding.severity is Severity.WARNING
    assert not report.has_failures()
    assert finding.details["step_index"] == 0
    assert finding.details["processor_visual_keys"] == ["observation.image"]
    assert finding.details["policy_visual_keys"] == ["observation.images.front"]
    assert finding.details["missing_policy_keys"] == ["observation.images.front"]
    assert finding.details["normalizer_visual_mode"] == "IDENTITY"
    assert finding.details["identity_camera_keys_pass_through"] is True
    assert finding.details["preprocessor_camera_keyerror_expected"] is False
    assert finding.details["downstream_contract_scope"] == "policy backbone input_features lookup"
    assert finding.details["downstream_failure_verified"] is False
    assert "without a preprocessor KeyError" in finding.summary


@pytest.mark.parametrize("rename_before", [False, True])
def test_resolving_camera_rename_does_not_create_dedicated_conflict(
    checkpoints: dict[str, Path], rename_before: bool,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _overwrite_observation_normalizer(
        checkpoint, mode=NormalizationMode.MEAN_STD,
        observation_stats={"mean": torch.zeros(14), "std": torch.full((14,), 0.5)},
        extra_features={"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
        extra_norm_map={FeatureType.VISUAL: NormalizationMode.IDENTITY},
    )
    policy_path = checkpoint / "config.json"
    policy = {"input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
              "output_features": {"action": {"type": "ACTION", "shape": [14]}}}
    policy["input_features"]["observation.images.front"] = {"type": "VISUAL", "shape": [3, 32, 32]}
    policy_path.write_text(json.dumps(policy))
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    rename = {"registry_name": "rename_observations_processor",
              "config": {"rename_map": {"observation.image": "observation.images.front"}}}
    config["steps"].insert(0 if rename_before else len(config["steps"]), rename)
    path.write_text(json.dumps(config))
    assert not _matching(run_preflight(checkpoint), "CAMERA_KEY_CONTRACT_MISMATCH")


def test_bound_action_does_not_hide_unbound_multi_dataset_observation_state(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _write_multi_dataset_buffers(checkpoint)
    for filename in ("preprocessor_config.json", "postprocessor_config.json"):
        config = json.loads((checkpoint / filename).read_text())
        path = checkpoint / config["steps"][0]["state_file"]
        state = load_file(path)
        state.update({"action.mean": torch.zeros(14), "action.std": torch.full((14,), 0.5)})
        save_file(state, path)
    finding, = _matching(run_preflight(checkpoint), "UNBOUND_BASE_MODEL_STATS")
    assert finding.details["feature_key"] == "observation.state"


@pytest.mark.parametrize("location", ["step", "model", "orphan"])
def test_git_lfs_pointer_is_specific_failure_before_safetensors_parse(
    checkpoints: dict[str, Path], location: str,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    if location == "step":
        config = json.loads((checkpoint / "preprocessor_config.json").read_text())
        pointer = checkpoint / config["steps"][0]["state_file"]
    else:
        pointer = checkpoint / ("model.safetensors" if location == "model" else "orphan.safetensors")
    payload = ("version https://git-lfs.github.com/spec/v1\n"
               f"oid sha256:{'a' * 64}\nsize 640\n").encode()
    pointer.write_bytes(payload)
    report = run_preflight(checkpoint)
    finding, = _matching(report, "GIT_LFS_POINTER_ARTIFACT")
    assert finding.status is DiagnosticStatus.FAIL and finding.severity is Severity.CRITICAL
    assert Path(finding.provenance.file_path) == pointer
    assert finding.provenance.sha256_prefix == hashlib.sha256(payload).hexdigest()[:12]
    assert finding.details["lfs_oid_sha256"] == "a" * 64
    assert finding.details["declared_object_size"] == 640
    assert not any("header too large" in item.summary or "Cannot screen step buffers" in item.summary
                   for item in report.findings)
    if location == "step":
        assert finding.details["step_index"] == 0
        assert finding.provenance.step_name == "normalizer_processor"


def _legacy_range_checkpoint(checkpoint: Path, tensors: dict[str, torch.Tensor], *, old_schema: bool = False) -> None:
    model = checkpoint / "model.safetensors"
    save_file(tensors, model)
    payload = model.read_bytes()
    header_len = int.from_bytes(payload[:8], "little")
    descriptors = json.loads(payload[8:8 + header_len])
    header = {
        "format": "safetensors-header-v1", "source_file": "model.safetensors", "repo_id": "org/legacy",
        "revision": "a" * 40, "total_size": len(payload), "header_len": header_len,
        "header_sha256_prefix": hashlib.sha256(payload[:8 + header_len]).hexdigest()[:12],
        "complete": True, "tensor_values_available": False, "tensors": descriptors,
    }
    (checkpoint / "model.safetensors.header.json").write_text(json.dumps(header))
    state_path = checkpoint / "legacy_normalization.safetensors"
    save_file(tensors, state_path)
    metadata = {
        "format": "legacy-normalization-ranges-v1", "source_file": "model.safetensors",
        "repo_id": header["repo_id"], "revision": header["revision"], "total_size": len(payload),
        "header_len": header_len, "header_sha256_prefix": header["header_sha256_prefix"],
        "tensor_file": state_path.name, "tensor_file_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "tensor_values_available": True, "complete": True,
        "bytes_fetched": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
        "errors": [], "skipped_tensors": {}, "tensors": {},
    }
    for key, descriptor in descriptors.items():
        start, end = descriptor["data_offsets"]
        metadata["tensors"][key] = {
            **descriptor, "absolute_range": [8 + header_len + start, 8 + header_len + end],
            "payload_sha256": hashlib.sha256(payload[8 + header_len + start:8 + header_len + end]).hexdigest(),
        }
    (checkpoint / "legacy_normalization.json").write_text(json.dumps(metadata))
    (checkpoint / "download_manifest.json").write_text(json.dumps({"model_weights_skipped": True}))
    model.unlink()  # The real scanner retains only the header plus tiny buffers.
    if old_schema:
        config = {"input_shapes": {"observation.state": [14]}, "output_shapes": {"action": [14]},
                  "input_normalization_modes": {"observation.state": "mean_std"},
                  "output_normalization_modes": {"action": "mean_std"}}
    else:
        config = {"input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
                  "output_features": {"action": {"type": "ACTION", "shape": [14]}},
                  "normalization_mapping": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}}
    (checkpoint / "config.json").write_text(json.dumps(config))


def _legacy_mean_std_tensors(*, zero_state: bool = False) -> dict[str, torch.Tensor]:
    std = torch.full((14,), 0.5)
    std[4 if zero_state else 9] = 0 if zero_state else 1.48e-6
    action_std = torch.full((14,), 0.5)
    action_std[6] = 0
    return {
        "normalize_inputs.buffer_observation_state.mean": torch.zeros(14),
        "normalize_inputs.buffer_observation_state.std": std,
        "unnormalize_outputs.buffer_action.mean": torch.zeros(14),
        "unnormalize_outputs.buffer_action.std": action_std,
    }


@pytest.mark.parametrize("old_schema", [False, True])
def test_verified_legacy_slices_screen_native_direction_and_keep_original_provenance(
    checkpoints: dict[str, Path], old_schema: bool,
) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    _legacy_range_checkpoint(checkpoint, _legacy_mean_std_tensors(), old_schema=old_schema)
    report = run_preflight(checkpoint)
    gain, = [finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN") if finding.status is DiagnosticStatus.FAIL]
    assert gain.provenance.violating_indices == [9]
    assert gain.provenance.tensor_key == "normalize_inputs.buffer_observation_state.std"
    assert gain.provenance.step_name == "legacy_normalize_inputs"
    assert gain.details["step_index"] is None
    assert gain.details["screening_semantics"] == "legacy_normalize_formulas"
    assert gain.details["eps_source"] == "inspected_legacy_Normalize_constant"
    assert gain.details["legacy_execution_configuration_known"] is False
    assert gain.details["tensor_values_verified"] is True
    assert len(gain.details["absolute_range"]) == 2
    dead, = [finding for finding in _matching(report, "DEAD_ACTION_DIMENSION") if finding.status is DiagnosticStatus.FAIL]
    assert dead.provenance.violating_indices == [6]
    assert dead.provenance.tensor_key == "unnormalize_outputs.buffer_action.std"
    assert dead.details["step_direction"] == "inverse"
    assert not any(finding.status is DiagnosticStatus.FAIL for finding in _matching(report, "FP16_NON_FINITE_RISK"))
    assert not (checkpoint / "model.safetensors").exists()


def test_legacy_zero_variance_is_forward_fp16_risk_but_inverse_dead_dimension(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    _legacy_range_checkpoint(checkpoint, _legacy_mean_std_tensors(zero_state=True))
    report = run_preflight(checkpoint)
    risk, = [finding for finding in _matching(report, "FP16_NON_FINITE_RISK") if finding.details.get("advisory")]
    assert risk.status is DiagnosticStatus.PASS
    assert risk.provenance.tensor_key == "normalize_inputs.buffer_observation_state.std"
    assert risk.provenance.violating_indices == [4]
    assert risk.details["fp16_execution_conditional"] is True
    assert risk.details["screened_dtype"] == "float16"
    assert risk.severity is Severity.WARNING
    assert risk.details["requires_legacy_buffer_half_cast_or_modern_migration"] is True
    assert risk.details["legacy_half_input_only_promotes_to_float32"] is True
    assert "FP16 input alone promotes arithmetic to float32" in risk.summary
    assert ".half()" in risk.summary
    gain, = [finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN")
             if finding.status is DiagnosticStatus.FAIL]
    assert gain.severity is Severity.CRITICAL
    assert gain.provenance.violating_indices == [4]
    assert gain.details["baseline_output_values"] == [0.0]
    assert gain.details["probe_output_values"] == [20_000_000.0]
    assert gain.details["effective_gains_with_eps"] == pytest.approx([1e8])


def test_legacy_constant_min_max_has_critical_float32_gain_and_conditional_half_risk(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    lower = torch.zeros(18)
    upper = torch.ones(18)
    constants = torch.tensor([-0.0127, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    lower[11:] = constants
    upper[11:] = constants
    tensors = {
        "normalize_inputs.buffer_observation_environment_state.min": lower,
        "normalize_inputs.buffer_observation_environment_state.max": upper,
        "unnormalize_outputs.buffer_action.mean": torch.zeros(14),
        "unnormalize_outputs.buffer_action.std": torch.full((14,), 0.5),
    }
    _legacy_range_checkpoint(checkpoint, tensors, old_schema=True)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["input_shapes"] = {"observation.environment_state": [18]}
    config["input_normalization_modes"] = {"observation.environment_state": "min_max"}
    config_path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    gain, = [finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN")
             if finding.status is DiagnosticStatus.FAIL]
    assert gain.severity is Severity.CRITICAL
    assert gain.provenance.violating_indices == list(range(11, 18))
    assert gain.provenance.tensor_key == "normalize_inputs.buffer_observation_environment_state.min"
    assert gain.details["upper_buffer_key"] == "normalize_inputs.buffer_observation_environment_state.max"
    assert gain.details["gain_formula"] == "2 / (max - min + eps)"
    assert gain.details["baseline_output_values"] == [-1.0] * 7
    assert gain.details["probe_output_values"] == [40_000_000.0, 40_000_008.0, 40_000_000.0,
                                                   40_000_000.0, 40_000_000.0, 40_000_008.0, 40_000_000.0]
    assert gain.details["effective_gains_with_eps"] == pytest.approx([2e8] * 7)
    risk, = [finding for finding in _matching(report, "FP16_NON_FINITE_RISK")
             if finding.details.get("advisory")]
    assert risk.status is DiagnosticStatus.PASS
    assert risk.severity is Severity.WARNING
    assert risk.provenance.violating_indices == list(range(11, 18))
    assert risk.details["configured_runtime_nonfinite_observed"] is False


def test_inverse_constant_min_max_does_not_report_division_gain_or_half_risk(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    upper = torch.ones(14)
    upper[2] = 0
    pipeline = PolicyProcessorPipeline(name="policy_postprocessor", steps=[UnnormalizerProcessorStep(
        features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MIN_MAX},
        stats={"action": {"min": torch.zeros(14), "max": upper}},
    )])
    pipeline.save_pretrained(checkpoint, config_filename="postprocessor_config.json")
    report = run_preflight(checkpoint)
    assert not any(finding.status is DiagnosticStatus.FAIL
                   for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK")
                   for finding in _matching(report, check))


def test_legacy_identity_mode_does_not_screen_unselected_state_variance(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    tensors = _legacy_mean_std_tensors(zero_state=True)
    tensors["unnormalize_outputs.buffer_action.std"][6] = 0.5
    _legacy_range_checkpoint(checkpoint, tensors)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["normalization_mapping"]["STATE"] = "IDENTITY"
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    assert not any(finding.status is DiagnosticStatus.FAIL
                   for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "DEAD_ACTION_DIMENSION")
                   for finding in _matching(report, check))


@pytest.mark.parametrize("tamper", ["file_hash", "payload_hash", "range", "revision"])
def test_unverified_legacy_slices_are_coverage_unknown_not_checkpoint_numeric_failure(
    checkpoints: dict[str, Path], tamper: str,
) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    _legacy_range_checkpoint(checkpoint, _legacy_mean_std_tensors())
    path = checkpoint / "legacy_normalization.json"
    metadata = json.loads(path.read_text())
    original = "normalize_inputs.buffer_observation_state.std"
    if tamper == "file_hash":
        metadata["tensor_file_sha256"] = "0" * 64
    elif tamper == "payload_hash":
        metadata["tensors"][original]["payload_sha256"] = "0" * 64
    elif tamper == "range":
        metadata["tensors"][original]["absolute_range"][0] += 1
    else:
        metadata["revision"] = "b" * 40
    path.write_text(json.dumps(metadata))
    report = run_preflight(checkpoint)
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "DEAD_ACTION_DIMENSION"):
        assert any(finding.status is DiagnosticStatus.UNKNOWN for finding in _matching(report, check))
        assert not any(finding.status is DiagnosticStatus.FAIL for finding in _matching(report, check))


def test_partial_legacy_download_does_not_turn_missing_payload_into_policy_defect(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    _legacy_range_checkpoint(checkpoint, _legacy_mean_std_tensors())
    state_path = checkpoint / "legacy_normalization.safetensors"
    tensors = load_file(state_path)
    original = "normalize_inputs.buffer_observation_state.std"
    tensors.pop(original)
    save_file(tensors, state_path)
    path = checkpoint / "legacy_normalization.json"
    metadata = json.loads(path.read_text())
    metadata["tensors"].pop(original)
    metadata["tensor_file_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    metadata["complete"] = False
    metadata["errors"] = [{"tensor_keys": [original], "error": "HTTP range unavailable"}]
    path.write_text(json.dumps(metadata))
    report = run_preflight(checkpoint)
    assert any(finding.status is DiagnosticStatus.UNKNOWN for finding in _matching(report, "PATHOLOGICAL_STD_GAIN"))
    assert not any(finding.status is DiagnosticStatus.FAIL for finding in _matching(report, "PATHOLOGICAL_STD_GAIN"))
    assert not any(finding.status is DiagnosticStatus.FAIL for finding in _matching(report, "FEATURE_CONTRACT"))


def test_legacy_multi_dataset_statistics_are_unbound_and_not_arbitrarily_screened(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    tensors = {
        f"{group}.{dataset}_buffer_{feature}.{stat}": torch.full((14,), value)
        for group, feature in (("normalize_inputs", "observation_state"), ("unnormalize_outputs", "action"))
        for dataset in ("so100-blue", "so100-red", "so100")
        for stat, value in (("mean", 0.0), ("std", 1e-12))
    }
    _legacy_range_checkpoint(checkpoint, tensors)
    report = run_preflight(checkpoint)
    finding, = _matching(report, "UNBOUND_BASE_MODEL_STATS")
    assert finding.details["legacy_embedded"] is True
    assert finding.details["dataset_namespaces"] == ["so100", "so100-blue", "so100-red"]
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "DEAD_ACTION_DIMENSION"):
        assert any(item.status is DiagnosticStatus.UNKNOWN for item in _matching(report, check))
        assert not any(item.status is DiagnosticStatus.FAIL for item in _matching(report, check))


def test_old_legacy_modes_are_per_feature_and_do_not_screen_unselected_std(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    tensors = _legacy_mean_std_tensors(zero_state=True)
    tensors["normalize_inputs.buffer_observation_environment_state.mean"] = torch.zeros(18)
    environment_std = torch.full((18,), 0.5)
    environment_std[12] = 1.48e-6
    tensors["normalize_inputs.buffer_observation_environment_state.std"] = environment_std
    tensors["unnormalize_outputs.buffer_action.min"] = torch.zeros(14)
    tensors["unnormalize_outputs.buffer_action.max"] = torch.ones(14)
    _legacy_range_checkpoint(checkpoint, tensors, old_schema=True)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["input_shapes"]["observation.environment_state"] = [18]
    config["input_normalization_modes"] = {"observation.state": "identity", "observation.environment_state": "mean_std"}
    config["output_normalization_modes"]["action"] = "min_max"
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    gain, = [finding for finding in _matching(report, "PATHOLOGICAL_STD_GAIN") if finding.status is DiagnosticStatus.FAIL]
    assert gain.details["feature_key"] == "observation.environment_state"
    assert gain.provenance.violating_indices == [12]
    assert gain.provenance.tensor_key == "normalize_inputs.buffer_observation_environment_state.std"
    assert not any(finding.status is DiagnosticStatus.FAIL for finding in _matching(report, "FP16_NON_FINITE_RISK"))
    assert not any(finding.status is DiagnosticStatus.FAIL for finding in _matching(report, "DEAD_ACTION_DIMENSION"))


def test_bound_legacy_min_max_does_not_treat_unused_multi_dataset_mean_std_as_unbound(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    tensors = {
        f"unnormalize_outputs.{dataset}_buffer_action.{stat}": torch.full((14,), value)
        for dataset in ("so100-blue", "so100-red", "so100")
        for stat, value in (("mean", 0.0), ("std", 1e-12))
    }
    for prefix in ("normalize_inputs.buffer_observation_state", "unnormalize_outputs.buffer_action"):
        tensors[f"{prefix}.min"] = torch.zeros(14)
        tensors[f"{prefix}.max"] = torch.ones(14)
    _legacy_range_checkpoint(checkpoint, tensors)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["normalization_mapping"] = {"STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    assert not _matching(report, "UNBOUND_BASE_MODEL_STATS")
    assert not any(finding.status is DiagnosticStatus.FAIL
                   for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "DEAD_ACTION_DIMENSION")
                   for finding in _matching(report, check))


def test_legacy_bound_action_does_not_hide_unbound_observation_statistics(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    tensors = {
        f"{group}.{dataset}_buffer_{feature}.{stat}": torch.full((14,), value)
        for group, feature in (("normalize_inputs", "observation_state"), ("unnormalize_outputs", "action"))
        for dataset in ("so100-blue", "so100-red", "so100")
        for stat, value in (("mean", 0.0), ("std", 0.5))
    }
    tensors.update({"unnormalize_outputs.buffer_action.mean": torch.zeros(14),
                    "unnormalize_outputs.buffer_action.std": torch.full((14,), 0.5)})
    _legacy_range_checkpoint(checkpoint, tensors)
    report = run_preflight(checkpoint)
    finding, = _matching(report, "UNBOUND_BASE_MODEL_STATS")
    assert finding.details["legacy_embedded"] is True
    assert finding.details["missing_observation_state_stats"] == [
        "normalize_inputs.buffer_observation_state.mean", "normalize_inputs.buffer_observation_state.std",
    ]
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "DEAD_ACTION_DIMENSION"):
        assert any(item.status is DiagnosticStatus.UNKNOWN for item in _matching(report, check))
        assert not any(item.status is DiagnosticStatus.FAIL for item in _matching(report, check))


@pytest.mark.parametrize("mode,stat,value", [
    (NormalizationMode.MEAN_STD, "std", -0.5),
    (NormalizationMode.MEAN_STD, "mean", float("nan")),
    (NormalizationMode.MEAN_STD, "std", float("inf")),
    (NormalizationMode.MIN_MAX, "min", float("-inf")),
    (NormalizationMode.MIN_MAX, "max", float("nan")),
    (NormalizationMode.QUANTILES, "q01", float("inf")),
    (NormalizationMode.QUANTILES, "q99", float("nan")),
    (NormalizationMode.QUANTILE10, "q10", float("nan")),
    (NormalizationMode.QUANTILE10, "q90", float("inf")),
])
def test_invalid_active_statistics_have_exact_coordinate_and_saved_value(
    checkpoints: dict[str, Path], mode: NormalizationMode, stat: str, value: float,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    names = {NormalizationMode.MEAN_STD: ("mean", "std"),
             NormalizationMode.MIN_MAX: ("min", "max"),
             NormalizationMode.QUANTILES: ("q01", "q99"),
             NormalizationMode.QUANTILE10: ("q10", "q90")}[mode]
    stats = {names[0]: torch.zeros(14), names[1]: torch.ones(14)}
    stats[stat][3] = value
    _overwrite_observation_normalizer(checkpoint, mode=mode, observation_stats=stats)
    failures = [f for f in _matching(run_preflight(checkpoint), "INVALID_NORMALIZATION_STATS")
                if f.provenance.tensor_key == f"observation.state.{stat}"]
    assert len(failures) == 1
    finding = failures[0]
    assert finding.status is DiagnosticStatus.FAIL and finding.severity is Severity.CRITICAL
    assert finding.details["step_index"] == 0
    assert finding.provenance.violating_indices == [3]
    saved = finding.details["saved_values"][0]
    assert saved == value or (torch.isnan(torch.tensor(saved)) and torch.isnan(torch.tensor(value)))
    json.dumps(finding.to_dict(), allow_nan=False)


@pytest.mark.parametrize("mode,lower,upper", [
    (NormalizationMode.MIN_MAX, "min", "max"),
    (NormalizationMode.QUANTILES, "q01", "q99"),
    (NormalizationMode.QUANTILE10, "q10", "q90"),
])
def test_reversed_active_intervals_are_invalid(
    checkpoints: dict[str, Path], mode: NormalizationMode, lower: str, upper: str,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    low, high = torch.zeros(14), torch.ones(14)
    low[4] = 2
    _overwrite_observation_normalizer(checkpoint, mode=mode, observation_stats={lower: low, upper: high})
    finding, = _matching(run_preflight(checkpoint), "INVALID_NORMALIZATION_STATS")
    assert finding.provenance.tensor_key == f"observation.state.{lower}"
    assert finding.provenance.violating_indices == [4]
    assert finding.details["saved_values"] == [2]
    assert "reversed_interval" in finding.details["invalid_reasons"]


@pytest.mark.parametrize("external_state", [False, True])
def test_inline_stats_are_screened_and_override_external_state_like_native(
    checkpoints: dict[str, Path], external_state: bool,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    entry = config["steps"][0]
    inline = {"observation.state": {"mean": [0.0] * 14, "std": [0.5] * 14},
              "action": {"mean": [0.0] * 14, "std": [0.5] * 14}}
    inline["observation.state"]["std"][2] = 3.1e-6
    entry["config"]["stats"] = inline
    if not external_state:
        (checkpoint / entry.pop("state_file")).unlink()
    path.write_text(json.dumps(config))
    native = PolicyProcessorPipeline.from_pretrained(checkpoint, config_filename=path.name, local_files_only=True)
    assert native.steps[0].state_dict()["observation.state.std"][2] == torch.tensor(3.1e-6)
    finding, = [f for f in _matching(run_preflight(checkpoint), "PATHOLOGICAL_STD_GAIN")
                if f.status is DiagnosticStatus.FAIL]
    assert finding.provenance.violating_indices == [2]
    assert finding.details["saved_values"] == pytest.approx([3.1e-6])
    assert Path(finding.provenance.file_path) == path


@pytest.mark.parametrize("missing", ["state_file", "one_buffer", "no_stats"])
def test_partial_active_normalization_coverage_never_emits_numeric_pass(
    checkpoints: dict[str, Path], missing: str,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    entry = config["steps"][0]
    state_path = checkpoint / entry["state_file"]
    if missing == "state_file":
        state_path.unlink()
    elif missing == "no_stats":
        state_path.unlink()
        entry.pop("state_file")
        path.write_text(json.dumps(config))
    else:
        state = load_file(state_path)
        state.pop("observation.state.std")
        save_file(state, state_path)
    report = run_preflight(checkpoint)
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        findings = _matching(report, check)
        assert not any(f.status is DiagnosticStatus.PASS for f in findings)
        assert any(f.status is DiagnosticStatus.UNKNOWN and f.details.get("coverage_gaps") for f in findings)


def test_all_inline_active_normalizers_can_establish_complete_numeric_coverage(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    for filename in ("preprocessor_config.json", "postprocessor_config.json"):
        path = checkpoint / filename
        config = json.loads(path.read_text())
        for entry in config["steps"]:
            state_path = checkpoint / entry.pop("state_file")
            inline = {}
            for key, tensor in load_file(state_path).items():
                feature, stat = key.rsplit(".", 1)
                inline.setdefault(feature, {})[stat] = tensor.tolist()
            entry["config"]["stats"] = inline
            state_path.unlink()
        path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    assert not report.has_failures()
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        finding, = _matching(report, check)
        assert finding.status is DiagnosticStatus.PASS
        assert finding.details["all_active_steps_screened"] is True


def test_contained_artifact_directory_is_allowed_but_state_stays_file_only(tmp_path: Path) -> None:
    from lerobot_preflight.preflight import _local_file

    root = tmp_path / "checkpoint"
    artifact = root / "tokenizer"
    artifact.mkdir(parents=True)
    (artifact / "tokenizer.json").write_text("{}")
    assert _local_file(root, "tokenizer", allow_directory=True) == artifact.resolve()
    with pytest.raises(ValueError):
        _local_file(root, "tokenizer")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    for value in ("../outside", "tokenizer/../tokenizer", "escape"):
        with pytest.raises(ValueError):
            _local_file(root, value, allow_directory=True)
    (artifact / "nested_escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside symlink"):
        _local_file(root, "tokenizer", allow_directory=True)


def test_crlf_lfs_pointer_preserves_exact_oid_and_size(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    path = checkpoint / config["steps"][0]["state_file"]
    oid = "0123456789abcdef" * 4
    path.write_bytes(f"version https://git-lfs.github.com/spec/v1\r\noid sha256:{oid}\r\nsize 940\r\n".encode())
    finding, = _matching(run_preflight(checkpoint), "GIT_LFS_POINTER_ARTIFACT")
    assert finding.details["lfs_oid_sha256"] == oid
    assert finding.details["declared_object_size"] == 940


@pytest.mark.parametrize("mode,lower,upper", [
    (NormalizationMode.MIN_MAX, "min", "max"),
    (NormalizationMode.QUANTILES, "q01", "q99"),
    (NormalizationMode.QUANTILE10, "q10", "q90"),
])
@pytest.mark.parametrize("scalar_lower", [False, True])
def test_broadcastable_scalar_vector_reversed_bounds_preserve_runtime_coordinates(
    checkpoints: dict[str, Path], mode: NormalizationMode, lower: str, upper: str, scalar_lower: bool,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    low, high = torch.ones(14), torch.full((14,), 2.0)
    if scalar_lower:
        low = torch.tensor(1.0)
        high[0] = 0.0
    else:
        high = torch.tensor(2.0)
        low[0] = 3.0
    _overwrite_observation_normalizer(checkpoint, mode=mode, observation_stats={lower: low, upper: high})
    report = run_preflight(checkpoint)
    finding, = _matching(report, "INVALID_NORMALIZATION_STATS")
    assert finding.provenance.violating_indices == [0]
    assert finding.provenance.tensor_key == f"observation.state.{lower}"
    assert finding.details["saved_values"] == [1.0 if scalar_lower else 3.0]
    assert finding.details["upper_values"][0] == (0.0 if scalar_lower else 2.0)
    if scalar_lower:
        assert finding.details["source_tensor_shape"] == []
        assert finding.details["source_saved_values"] == [1.0]
        assert finding.details["broadcast_tensor_shape"] == [14]


@pytest.mark.parametrize("mode,lower,upper", [
    (NormalizationMode.MIN_MAX, "min", "max"),
    (NormalizationMode.QUANTILES, "q01", "q99"),
    (NormalizationMode.QUANTILE10, "q10", "q90"),
])
@pytest.mark.parametrize("scalar_lower", [False, True])
def test_broadcastable_scalar_vector_equal_ranges_screen_fp16_and_minmax_gain(
    checkpoints: dict[str, Path], mode: NormalizationMode, lower: str, upper: str, scalar_lower: bool,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    low, high = torch.ones(14), torch.full((14,), 2.0)
    if scalar_lower:
        low = torch.tensor(1.0)
        high[5] = 1.0
    else:
        high = torch.tensor(2.0)
        low[5] = 2.0
    _overwrite_observation_normalizer(checkpoint, mode=mode, observation_stats={lower: low, upper: high})
    native = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True,
    )
    # The same broadcastable scalar/vector pair is valid native arithmetic.
    output = native({"observation.state": torch.full((1, 14), 2.2, dtype=torch.float16)})["observation.state"]
    assert not torch.isfinite(output[0, 5])
    report = run_preflight(checkpoint)
    risk, = [f for f in _matching(report, "FP16_NON_FINITE_RISK")
             if f.status is DiagnosticStatus.PASS and f.details.get("feature_key") == "observation.state"]
    assert risk.severity is Severity.WARNING and risk.details["fp16_execution_configured"] is False
    assert risk.provenance.violating_indices == [5]
    assert risk.details["saved_values"] == [1.0 if scalar_lower else 2.0]
    assert not _matching(report, "INVALID_NORMALIZATION_STATS")
    gain, = [f for f in _matching(report, "PATHOLOGICAL_STD_GAIN") if f.status is DiagnosticStatus.FAIL]
    assert gain.provenance.violating_indices == [5]
    assert gain.details["effective_gains_with_eps"] == pytest.approx([2e8])
    assert gain.details["upper_values"] == [1.0 if scalar_lower else 2.0]


@pytest.mark.parametrize("std", [0.0, -0.5])
def test_scalar_std_broadcast_against_vector_mean_has_exact_active_coordinate_indices(
    checkpoints: dict[str, Path], std: float,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _overwrite_observation_normalizer(checkpoint, mode=NormalizationMode.MEAN_STD,
                                     observation_stats={"mean": torch.zeros(14), "std": torch.tensor(std)})
    report = run_preflight(checkpoint)
    check = "FP16_NON_FINITE_RISK" if std == 0 else "INVALID_NORMALIZATION_STATS"
    finding, = [f for f in _matching(report, check) if f.details.get("feature_key") == "observation.state"]
    assert finding.provenance.violating_indices == list(range(14))
    assert finding.details["saved_values"] == [std] * 14
    assert finding.details["source_tensor_shape"] == []
    assert finding.details["source_saved_values"] == [std]
    if std == 0:
        gain, = [f for f in _matching(report, "PATHOLOGICAL_STD_GAIN") if f.status is DiagnosticStatus.FAIL]
        assert gain.provenance.violating_indices == list(range(14))


@pytest.mark.parametrize("mode,lower,upper", [
    (NormalizationMode.QUANTILES, "q01", "q99"),
    (NormalizationMode.QUANTILE10, "q10", "q90"),
])
@pytest.mark.parametrize("span", [1e-5, 0.0])
def test_quantile_pathological_gain_matches_native_forward_affine(
    checkpoints: dict[str, Path], mode: NormalizationMode, lower: str, upper: str, span: float,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    low, high = torch.zeros(14), torch.ones(14)
    high[9] = span
    _overwrite_observation_normalizer(checkpoint, mode=mode, observation_stats={lower: low, upper: high})
    native = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True,
    )
    initial = native({"observation.state": torch.zeros(1, 14)})["observation.state"]
    shifted = native({"observation.state": torch.ones(1, 14)})["observation.state"]
    native_gain = (shifted - initial)[0, 9].item()
    expected = 2 / (span if span else 1e-8)
    assert native_gain == pytest.approx(expected)
    report = run_preflight(checkpoint)
    finding, = [f for f in _matching(report, "PATHOLOGICAL_STD_GAIN") if f.status is DiagnosticStatus.FAIL]
    assert finding.provenance.violating_indices == [9]
    assert finding.provenance.tensor_key == f"observation.state.{lower}"
    assert finding.details["upper_buffer_key"] == f"observation.state.{upper}"
    assert finding.details["effective_gains_with_eps"] == pytest.approx([native_gain])
    assert finding.details["gain_formula"] == f"2 / where({upper} - {lower} == 0, eps, {upper} - {lower})"


@pytest.mark.parametrize("mode,lower,upper", [
    (NormalizationMode.QUANTILES, "q01", "q99"),
    (NormalizationMode.QUANTILE10, "q10", "q90"),
])
def test_quantile_zero_span_respects_effective_epsilon_gain(
    checkpoints: dict[str, Path], mode: NormalizationMode, lower: str, upper: str,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _overwrite_observation_normalizer(
        checkpoint, mode=mode, observation_stats={lower: torch.zeros(14), upper: torch.zeros(14)}, eps=1e-3,
    )
    report = run_preflight(checkpoint)
    assert not [f for f in _matching(report, "PATHOLOGICAL_STD_GAIN") if f.status is DiagnosticStatus.FAIL]
    assert any(f.status is DiagnosticStatus.PASS for f in _matching(report, "PATHOLOGICAL_STD_GAIN"))


@pytest.mark.parametrize("external_value,inline_value,invalid", [
    (0.5, 1.0, False),
    (float("nan"), 1.0, False),
    (-1.0, 1.0, False),
    (1.0, -1.0, True),
    (1.0, float("nan"), True),
])
def test_inline_runtime_override_is_advisory_and_only_effective_values_are_screened(
    checkpoints: dict[str, Path], external_value: float, inline_value: float, invalid: bool,
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    config_path = checkpoint / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    entry = config["steps"][0]
    state_path = checkpoint / entry["state_file"]
    state = load_file(state_path)
    state["observation.state.std"][4] = external_value
    save_file(state, state_path)
    inline = {}
    for key, tensor in state.items():
        feature, name = key.rsplit(".", 1)
        inline.setdefault(feature, {})[name] = tensor.tolist()
    inline["observation.state"]["std"][4] = inline_value
    entry["config"]["stats"] = inline
    config_path.write_text(json.dumps(config))
    native = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename=config_path.name, local_files_only=True,
    )
    torch.testing.assert_close(native.steps[0].state_dict()["observation.state.std"][4],
                               torch.tensor(inline_value), equal_nan=True)
    report = run_preflight(checkpoint)
    advisory, = _matching(report, "INLINE_STAT_OVERRIDE_ADVISORY")
    assert advisory.status is DiagnosticStatus.PASS
    assert advisory.severity is Severity.WARNING
    assert advisory.details["effective_source"] == "config.stats"
    assert advisory.details["state_file_ignored"] is True
    assert Path(advisory.provenance.file_path) == config_path
    assert not [finding for finding in report.findings if finding.status is DiagnosticStatus.CONFLICT]
    invalid_findings = _matching(report, "INVALID_NORMALIZATION_STATS")
    assert bool(invalid_findings) is invalid
    if invalid:
        finding, = invalid_findings
        assert finding.provenance.violating_indices == [4]
        assert Path(finding.provenance.file_path) == config_path
    else:
        assert not report.has_failures()
        assert all(f.status is DiagnosticStatus.PASS for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK")
                   for f in _matching(report, check))


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("declaration", ["registry_name", "class"])
def test_uninspected_active_normalizer_cannot_borrow_known_step_numeric_pass(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch, identity: bool, declaration: str,
) -> None:
    from copy import deepcopy

    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    unknown = deepcopy(config["steps"][0])
    unknown.pop("registry_name", None)
    unknown.pop("class", None)
    unknown[declaration] = ("uninspected_normalizer_processor" if declaration == "registry_name"
                            else "untrusted_remote_package.UninspectedNormalizer")
    if identity:
        unknown["config"]["norm_map"] = {"STATE": "IDENTITY", "ACTION": "IDENTITY"}
    config["steps"].append(unknown)
    path.write_text(json.dumps(config))

    def deny_network(*args, **kwargs):
        pytest.fail("Uninspected normalizer attempted a network request")

    monkeypatch.setattr(socket, "socket", deny_network)
    report = run_preflight(checkpoint)
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        findings = _matching(report, check)
        if identity:
            assert all(f.status is DiagnosticStatus.PASS for f in findings)
        else:
            finding, = findings
            assert finding.status is DiagnosticStatus.UNKNOWN
            assert finding.details["all_active_steps_screened"] is False
            gap, = finding.details["coverage_gaps"]
            assert gap["step_index"] == 1
            assert gap["reason"] == "uninspected_active_normalization_contract"
            assert gap["constructor_invoked"] is False
            assert "observation.state.std" in gap["declared_required_buffer_keys"]


def _round3_action_chain(checkpoint: Path, dtype: str) -> None:
    feature = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    steps = [DeviceProcessorStep(device="cpu", float_dtype=dtype)]
    for mean, std in ((50_000., 50_000.), (-0.995, 1.99e-5)):
        steps.append(UnnormalizerProcessorStep(
            features=feature, norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={"action": {"mean": torch.full((14,), mean), "std": torch.full((14,), std)}},
        ))
    PolicyProcessorPipeline(name="policy_postprocessor", steps=steps).save_pretrained(
        checkpoint, config_filename="postprocessor_config.json")


def test_round3_preflight_reports_intermediate_fp16_overflow_before_composed_cancellation(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    _round3_action_chain(checkpoint, "float16")
    # A float64 affine composition hides the actual half-precision overflow.
    assert 50_000 * 1.99e-5 == pytest.approx(0.995)
    report = run_preflight(checkpoint)
    finding, = _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.details["step_index"] == 1
    assert finding.details["step_class"] == "UnnormalizerProcessorStep"
    assert finding.details["output_dtype"] == "float16"
    assert finding.details["source_stat_values"]["action.std"] == [50_000.] * 14
    assert finding.provenance.tensor_key == "action.std"
    assert finding.provenance.violating_indices == list(range(14))
    assert any(not torch.isfinite(torch.tensor(value)) for value in finding.details["tensor_slice"])
    assert any(f.status is DiagnosticStatus.FAIL and f.details.get("risk_reasons") == ["native_intermediate_arithmetic_nonfinite"]
               for f in _matching(report, "FP16_NON_FINITE_RISK"))


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_round3_native_probe_respects_wider_intermediate_dtypes(checkpoints, dtype):
    checkpoint = checkpoints["clean_act_modern"]
    _round3_action_chain(checkpoint, dtype)
    report = run_preflight(checkpoint)
    assert not _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    assert not any(f.status is DiagnosticStatus.FAIL for f in _matching(report, "FP16_NON_FINITE_RISK"))


def test_round3_single_forward_normalizer_arithmetic_overflow_is_detected(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    normalizer = NormalizerProcessorStep(features=feature,
        norm_map={FeatureType.STATE: NormalizationMode.MIN_MAX},
        stats={"observation.state": {"min": torch.full((14,), -50_000.), "max": torch.full((14,), 50_000.)}})
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        DeviceProcessorStep(device="cpu", float_dtype="float16"), normalizer,
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    report = run_preflight(checkpoint)
    finding, = _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    assert finding.details["step_index"] == 1
    assert finding.details["feature_key"] == "observation.state"
    assert finding.details["output_dtype"] == "float16"


@pytest.mark.parametrize("barrier_before_overflow", [False, True])
def test_round3_native_prefix_failure_is_not_hidden_by_later_uninspected_step(checkpoints, barrier_before_overflow):
    checkpoint = checkpoints["clean_act_modern"]
    _round3_action_chain(checkpoint, "float16")
    path = checkpoint / "postprocessor_config.json"
    config = json.loads(path.read_text())
    barrier = {"registry_name": "uninspected_clip_processor", "config": {"min": -1., "max": 1.}}
    config["steps"].insert(1 if barrier_before_overflow else 2, barrier)
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    failures = _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    if barrier_before_overflow:
        assert not failures
        assert any(f.status is DiagnosticStatus.UNKNOWN for f in _matching(report, "NORMALIZATION_CHAIN_PROBE"))
    else:
        assert len(failures) == 1
        assert failures[0].details["step_index"] == 1


def test_round3_masked_custom_steps_are_not_executed_as_unmasked_base_steps(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "postprocessor_config.json"
    config = json.loads(path.read_text())
    entry = config["steps"][0]
    entry["registry_name"] = "molmoact2_masked_unnormalizer"
    state_path = checkpoint / entry["state_file"]
    state = load_file(state_path)
    state["action.std"].zero_()
    state["action.mask"] = torch.zeros(14, dtype=torch.bool)
    save_file(state, state_path)
    config["steps"].insert(0, {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": "float16"}})
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    assert not _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    assert any(f.status is DiagnosticStatus.UNKNOWN for f in _matching(report, "NORMALIZATION_CHAIN_PROBE"))
    assert not any(f.status is DiagnosticStatus.FAIL for f in _matching(report, "FP16_NON_FINITE_RISK"))


def test_round3_conditional_legacy_cpu_float32_risk_is_nonblocking_advisory(checkpoints):
    checkpoint = checkpoints["legacy_act_embedded"]
    tensors = _legacy_mean_std_tensors()
    tensors["normalize_inputs.buffer_observation_state.mean"].fill_(70_000.)
    tensors["normalize_inputs.buffer_observation_state.std"].fill_(1.)
    tensors["unnormalize_outputs.buffer_action.std"].fill_(1.)
    _legacy_range_checkpoint(checkpoint, tensors)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config.update(device="cpu", dtype="float32", use_amp=False)
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    risk, = [f for f in _matching(report, "FP16_NON_FINITE_RISK") if f.details.get("advisory")]
    assert risk.status is DiagnosticStatus.PASS
    assert risk.severity is Severity.WARNING
    assert risk.details["blocking"] is False
    assert risk.details["checkpoint_device"] == "cpu"
    assert risk.details["configured_runtime_nonfinite_observed"] is False
    assert not PreflightReport(str(checkpoint), findings=[risk]).has_failures()
    assert PreflightReport(str(checkpoint), findings=[risk], strict=True).has_failures()
    assert not any(f.status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
                   for f in _matching(report, "FP16_NON_FINITE_RISK"))


def test_round3_forward_action_probe_uses_saved_physical_training_boundaries(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    feature = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    def normalizer(mean, std):
        return NormalizerProcessorStep(features=feature,
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={"action": {"mean": torch.full((14,), mean), "std": torch.full((14,), std)}})
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        DeviceProcessorStep(device="cpu", float_dtype="float16"), normalizer(50_000., 1.), normalizer(0., .1),
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    report = run_preflight(checkpoint)
    assert not _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")


def test_round3_broadcastable_leading_stat_dimensions_preserve_probe_attribution(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    _round3_action_chain(checkpoint, "float16")
    config = json.loads((checkpoint / "postprocessor_config.json").read_text())
    path = checkpoint / config["steps"][1]["state_file"]
    state = load_file(path)
    state["action.mean"] = state["action.mean"].reshape(1, 1, 14)
    state["action.std"] = state["action.std"].reshape(1, 1, 14)
    save_file(state, path)
    report = run_preflight(checkpoint)
    finding, = _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    assert finding.details["step_index"] == 1
    assert finding.details["input_tensor_shape"] == [5, 14]
    assert finding.details["output_tensor_shape"] == [1, 5, 14]
    assert len(finding.details["input_values"]) == len(finding.details["output_values"])
    assert finding.provenance.violating_indices == list(range(14))


def test_round3_forward_probe_accepts_native_leading_singleton_stat_axes(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    step = NormalizerProcessorStep(features=feature,
        norm_map={FeatureType.STATE: NormalizationMode.MIN_MAX},
        stats={"observation.state": {"min": torch.full((1, 1, 14), -50_000.),
                                      "max": torch.full((1, 1, 14), 50_000.)}})
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        DeviceProcessorStep(device="cpu", float_dtype="float16"), step,
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    report = run_preflight(checkpoint)
    finding, = _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    assert finding.details["step_index"] == 1
    assert finding.details["input_tensor_shape"] == [3, 14]
    assert finding.details["output_tensor_shape"] == [1, 3, 14]
    assert finding.provenance.violating_indices == list(range(14))
    assert not _matching(report, "NORMALIZATION_CHAIN_PROBE")


@pytest.mark.parametrize("step_class", [NormalizerProcessorStep, UnnormalizerProcessorStep])
@pytest.mark.parametrize("mode,first,last", [
    (NormalizationMode.MEAN_STD, "mean", "std"),
    (NormalizationMode.MIN_MAX, "min", "max"),
    (NormalizationMode.QUANTILES, "q01", "q99"),
    (NormalizationMode.QUANTILE10, "q10", "q90"),
])
def test_counterexample_b_implicit_action_is_screened_with_empty_features(
    checkpoints, step_class, mode, first, last,
):
    checkpoint = checkpoints["clean_act_modern"]
    saved = torch.tensor([.5, .5, float("nan"), float("inf"), .5, .5])
    step = step_class(features={}, norm_map={FeatureType.ACTION: mode},
                      stats={"action": {first: torch.zeros(6), last: saved}})
    native = PolicyProcessorPipeline(name="policy_preprocessor", steps=[step])
    native.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    # Action processing is native and independent of the serialized features map.
    result = native({"action": torch.full((1, 6), .25)})["action"]
    assert torch.isnan(result[0, 2])
    report = run_preflight(checkpoint)
    finding, = [f for f in _matching(report, "INVALID_NORMALIZATION_STATS")
                if f.provenance.tensor_key == f"action.{last}"]
    assert finding.status is DiagnosticStatus.FAIL
    assert finding.provenance.violating_indices == [2, 3]
    assert finding.details["feature_shape_declared"] is False
    assert finding.details["feature_shape_source"] == "saved_stat_broadcast"
    assert not any(f.status is DiagnosticStatus.FAIL for f in _matching(report, "FP16_NON_FINITE_RISK")), \
        "An unconfigured hypothetical FP16 check cannot be the blocking finding"


@pytest.mark.parametrize("dtype", [None, "float32", "bfloat16", "float16"])
def test_counterexample_e_modern_half_overflow_is_advisory_without_half_execution(checkpoints, dtype):
    checkpoint = checkpoints["clean_act_modern"]
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    step = NormalizerProcessorStep(features=feature, norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                                   stats={"observation.state": {"mean": torch.full((14,), 70_000.), "std": torch.ones(14)}})
    steps = ([] if dtype is None else [DeviceProcessorStep(device="cpu", float_dtype=dtype)]) + [step]
    PolicyProcessorPipeline(name="policy_preprocessor", steps=steps).save_pretrained(
        checkpoint, config_filename="preprocessor_config.json")
    report = run_preflight(checkpoint)
    risks = [f for f in _matching(report, "FP16_NON_FINITE_RISK")
             if f.details.get("feature_key") == "observation.state"]
    if dtype == "bfloat16":
        assert not risks
        assert not _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
        assert not report.has_failures()
        return
    assert risks
    if dtype == "float16":
        assert any(f.status is DiagnosticStatus.FAIL and f.severity is Severity.CRITICAL for f in risks)
        assert _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
    else:
        assert all(f.status is DiagnosticStatus.PASS and f.severity is Severity.WARNING for f in risks)
        assert all(f.details["advisory"] and not f.details["blocking"] for f in risks)
        assert not _matching(report, "NORMALIZATION_CHAIN_NON_FINITE")
        assert not report.has_failures()


def test_counterexample_e_saved_half_buffer_is_explicit_precision_evidence(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    state_path = checkpoint / config["steps"][0]["state_file"]
    state = load_file(state_path)
    state["observation.state.std"][4] = 0.
    save_file({key: tensor.half() for key, tensor in state.items()}, state_path)
    risk, = [f for f in _matching(run_preflight(checkpoint), "FP16_NON_FINITE_RISK")
             if f.details.get("feature_key") == "observation.state"]
    assert risk.status is DiagnosticStatus.FAIL and risk.severity is Severity.CRITICAL
    assert risk.details["serialized_fp16_buffers"] is True
    assert risk.provenance.violating_indices == [4]


def test_unreferenced_half_metadata_does_not_configure_active_feature_execution(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    path = checkpoint / config["steps"][0]["state_file"]
    state = load_file(path)
    state["observation.state.std"][4] = 0.
    state["task_index.std"] = torch.zeros(1, dtype=torch.float16)
    save_file(state, path)
    risk, = [f for f in _matching(run_preflight(checkpoint), "FP16_NON_FINITE_RISK")
             if f.details.get("feature_key") == "observation.state"]
    assert risk.status is DiagnosticStatus.PASS and risk.severity is Severity.WARNING
    assert risk.details["serialized_fp16_buffers"] is False


def test_counterexample_f_composed_forward_gain_is_screened_beyond_individual_steps(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    def step():
        return NormalizerProcessorStep(features=feature, norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
            stats={"observation.state": {"mean": torch.zeros(14), "std": torch.full((14,), 1.01e-4)}})
    native = PolicyProcessorPipeline(name="policy_preprocessor", steps=[step(), step()])
    native.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    actual = native({"observation.state": torch.full((1, 14), .001)})["observation.state"]
    assert actual[0, 0].item() == pytest.approx(98_010., rel=1e-4)
    report = run_preflight(checkpoint)
    gain, = [f for f in _matching(report, "PATHOLOGICAL_STD_GAIN") if f.status is DiagnosticStatus.FAIL]
    assert gain.details["comparison_scope"] == "composed_forward_normalization"
    assert gain.details["contributing_step_indices"] == [0, 1]
    assert gain.details["step_index"] == 1
    assert gain.details["composed_gains"] == pytest.approx([actual[0, 0].item() / .001] * 14, rel=1e-5)
    assert gain.provenance.tensor_key == "observation.state.std"
    assert gain.provenance.violating_indices == list(range(14))


def test_composed_forward_gain_respects_actual_later_inverse_cancellation(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    def step(cls, scale):
        return cls(features=feature, norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                   stats={"observation.state": {"mean": torch.zeros(14), "std": torch.full((14,), scale)}})
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        step(NormalizerProcessorStep, .01), step(NormalizerProcessorStep, .01), step(UnnormalizerProcessorStep, 1e-6),
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    report = run_preflight(checkpoint)
    assert not any(f.status is DiagnosticStatus.FAIL for f in _matching(report, "PATHOLOGICAL_STD_GAIN"))


@pytest.mark.parametrize("scale,expected_gain_failure", [(1.01e-4, True), (0.5, False)])
def test_composed_gain_tracks_native_rename_between_normalizers(checkpoints, scale, expected_gain_failure):
    checkpoint = checkpoints["clean_act_modern"]
    first, final = "observation.state", "observation.renamed_state"

    def normalizer(key):
        return NormalizerProcessorStep(features={key: PolicyFeature(FeatureType.STATE, (14,))},
            norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
            stats={key: {"mean": torch.zeros(14), "std": torch.full((14,), scale)}})

    native = PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        normalizer(first), RenameObservationsProcessorStep(rename_map={first: final}), normalizer(final),
    ])
    native.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    output = native({first: torch.full((1, 14), .001)})[final]
    report = run_preflight(checkpoint)
    gains = [f for f in _matching(report, "PATHOLOGICAL_STD_GAIN") if f.status is DiagnosticStatus.FAIL]
    assert bool(gains) is expected_gain_failure
    assert not _matching(report, "NORMALIZATION_CHAIN_PROBE")
    if expected_gain_failure:
        gain, = gains
        assert gain.details["source_feature_key"] == first
        assert gain.details["feature_key"] == final
        assert gain.details["contributing_step_indices"] == [0, 2]
        assert [stage["feature_key"] for stage in gain.details["contributing_stages"]] == [first, final]
        assert gain.details["composed_gains"] == pytest.approx([float(output[0, 0]) / .001] * 14, rel=1e-5)
        assert gain.provenance.tensor_key == final + ".std"
    else:
        assert any(f.status is DiagnosticStatus.PASS for f in _matching(report, "PATHOLOGICAL_STD_GAIN"))


def test_uninspected_step_between_normalizers_keeps_composed_gain_unknown(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    def normalizer():
        return NormalizerProcessorStep(features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
            norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
            stats={"observation.state": {"mean": torch.zeros(14), "std": torch.full((14,), 1.01e-4)}})
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[normalizer(), IdentityProcessorStep(), normalizer()]).save_pretrained(
        checkpoint, config_filename="preprocessor_config.json")
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][1] = {"registry_name": "uninspected_gain_or_clamp", "config": {}}
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    gains = _matching(report, "PATHOLOGICAL_STD_GAIN")
    assert any(f.status is DiagnosticStatus.UNKNOWN and f.details.get("comparison_scope") == "composed_forward_normalization"
               for f in gains)
    assert not any(f.status is DiagnosticStatus.PASS for f in gains)


def test_observation_rename_does_not_require_observations_for_action_subpath_probe(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    def normalizer(state_key):
        return NormalizerProcessorStep(features={state_key: PolicyFeature(FeatureType.STATE, (14,)),
                "action": PolicyFeature(FeatureType.ACTION, (14,))},
            norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={key: {"mean": torch.zeros(14), "std": torch.full((14,), .5)}
                   for key in (state_key, "action")})
    native = PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        normalizer("observation.state"),
        RenameObservationsProcessorStep(rename_map={"observation.state": "observation.renamed_state"}),
        normalizer("observation.renamed_state"),
    ])
    native.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    output = native({"observation.state": torch.full((1, 14), .2), "action": torch.full((1, 14), .2)})
    torch.testing.assert_close(output["action"], torch.full((1, 14), .8))
    report = run_preflight(checkpoint)
    assert not _matching(report, "NORMALIZATION_CHAIN_PROBE")
    assert not any(f.status is DiagnosticStatus.UNKNOWN for f in _matching(report, "PATHOLOGICAL_STD_GAIN"))


def test_explicit_bfloat16_70000_mean_does_not_fail_strict_as_hypothetical_half(checkpoints):
    from lerobot_preflight.cli import main
    checkpoint = checkpoints["clean_act_modern"]
    old_config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    old_state = checkpoint / old_config["steps"][0]["state_file"]
    step = NormalizerProcessorStep(features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"observation.state": {"mean": torch.full((14,), 70_000.), "std": torch.ones(14)}})
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        DeviceProcessorStep(device="cpu", float_dtype="bfloat16"), step,
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    old_state.unlink()  # This test is a clean graph, without the replaced fixture's orphan state.
    (checkpoint / "config.json").write_text(json.dumps({
        "type": "act", "n_obs_steps": 1,
        "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    }))
    report = run_preflight(checkpoint)
    assert not any(f.severity is Severity.WARNING for f in report.findings)
    assert main(["--policy", str(checkpoint), "--strict"]) == 0


def test_action_mode_without_feature_or_stats_has_unknown_numeric_coverage(checkpoints):
    from lerobot_preflight.cli import main
    checkpoint = checkpoints["clean_act_modern"]
    old_config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    old_state = checkpoint / old_config["steps"][0]["state_file"]
    PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        NormalizerProcessorStep(features={}, norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD}),
    ]).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    old_state.unlink()
    (checkpoint / "config.json").write_text(json.dumps({
        "type": "act", "n_obs_steps": 1,
        "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    }))
    report = run_preflight(checkpoint)
    gains = _matching(report, "PATHOLOGICAL_STD_GAIN")
    assert any(f.status is DiagnosticStatus.UNKNOWN for f in gains)
    assert not any(f.status is DiagnosticStatus.PASS for f in gains)
    assert not any(f.status is DiagnosticStatus.UNKNOWN for f in _matching(report, "FEATURE_CONTRACT"))
    assert main(["--policy", str(checkpoint), "--strict"]) == 1


def test_explicit_bfloat16_actual_overflow_is_reported_in_bfloat16_scope(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    step = NormalizerProcessorStep(features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"observation.state": {"mean": torch.full((14,), 3.4e38), "std": torch.ones(14)}})
    pipeline = PolicyProcessorPipeline(name="policy_preprocessor", steps=[
        DeviceProcessorStep(device="cpu", float_dtype="bfloat16"), step,
    ])
    pipeline.save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    assert not torch.isfinite(pipeline({"observation.state": torch.full((1, 14), 3.4e38)})["observation.state"]).all()
    report = run_preflight(checkpoint)
    risk, = _matching(report, "BF16_NON_FINITE_RISK")
    assert risk.status is DiagnosticStatus.FAIL and risk.severity is Severity.CRITICAL
    assert risk.details["screened_dtype"] == "bfloat16"
    assert risk.provenance.violating_indices == list(range(14))
    assert not any(f.details.get("feature_key") == "observation.state"
                   for f in _matching(report, "FP16_NON_FINITE_RISK"))


def test_undeclared_non_native_physical_semantics_are_informational(checkpoints):
    report = run_preflight(checkpoints["clean_act_modern"])
    semantics = _matching(report, "PHYSICAL_SEMANTICS")
    assert len(semantics) == 3
    assert all(f.status is DiagnosticStatus.UNKNOWN and f.severity is Severity.INFO
               and f.details["strict_exempt"] and not f.details["required_coverage"] for f in semantics)


def test_native_action_path_cannot_be_hidden_by_incorrect_declared_feature_type(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "postprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["features"]["action"]["type"] = "STATE"
    state_path = checkpoint / config["steps"][0]["state_file"]
    state = load_file(state_path)
    state["action.std"][2] = float("nan")
    save_file(state, state_path)
    path.write_text(json.dumps(config))
    report = run_preflight(checkpoint)
    assert any(f.status is DiagnosticStatus.FAIL and f.provenance.tensor_key == "action.std"
               and f.provenance.violating_indices == [2] for f in _matching(report, "INVALID_NORMALIZATION_STATS"))
    assert any(f.status is DiagnosticStatus.CONFLICT and f.details.get("effective_feature_type") == "ACTION"
               for f in _matching(report, "FEATURE_CONTRACT"))


def test_physical_sidecar_is_only_read_when_explicitly_selected(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "metadata.json"
    path.write_text(json.dumps({"joint_angle_units": "radians"}))
    before = _matching(run_preflight(checkpoint), "PHYSICAL_SEMANTICS")
    assert all(f.status is DiagnosticStatus.UNKNOWN for f in before)
    after = _matching(run_preflight(checkpoint, metadata_path=path), "PHYSICAL_SEMANTICS")
    assert any(f.status is DiagnosticStatus.PASS and f.details["field"] == "joint_angle_units" for f in after)


@pytest.mark.parametrize("content", [None, "invalid-json", "[]"])
def test_explicit_missing_or_invalid_metadata_is_a_requested_coverage_failure(checkpoints, content):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "metadata.json"
    if content is not None:
        path.write_text(content)
    findings = _matching(run_preflight(checkpoint, metadata_path=path), "PHYSICAL_SEMANTICS")
    assert any(f.status is DiagnosticStatus.FAIL and f.details.get("required_coverage") for f in findings)
