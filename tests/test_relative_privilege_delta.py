"""Relative action offsets must be compared, not rejected or assumed invariant."""

from __future__ import annotations

import json
import shutil
import socket

import pytest
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from safetensors.torch import load_file, save_file

from lerobot_preflight.cli import main
from lerobot_preflight.models import DiagnosticStatus
from lerobot_preflight.preflight import run_preflight
from lerobot_preflight.privilege_delta import run_privilege_delta


def _checkpoint(path, *, normalization=False, enabled=True, mask=None):
    path.mkdir()
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (14,)),
                "action": PolicyFeature(FeatureType.ACTION, (14,))}
    mapping = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    names = [f"joint{index}" for index in range(14)]
    relative = RelativeActionsProcessorStep(enabled=enabled, action_names=names,
                                            exclude_joints=[] if mask is None else mask)
    pre, post = [], [AbsoluteActionsProcessorStep(enabled=enabled, relative_step=relative)]
    if normalization:
        stats = {"observation.state": {"mean": [10.] * 14, "std": [2.] * 14},
                 "action": {"mean": [5.] * 14, "std": [2.] * 14}}
        pre.append(NormalizerProcessorStep(features=features, norm_map=mapping, stats=stats))
        post.append(UnnormalizerProcessorStep(features={"action": features["action"]},
                    norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD}, stats={"action": stats["action"]}))
    pre.append(relative)
    PolicyProcessorPipeline(name="policy_preprocessor", steps=pre).save_pretrained(path, config_filename="policy_preprocessor.json")
    PolicyProcessorPipeline(name="policy_postprocessor", steps=post).save_pretrained(path, config_filename="policy_postprocessor.json")
    (path / "config.json").write_text(json.dumps({"type": "custom_relative_fixture", "n_obs_steps": 3,
        "observation_delta_indices": [0, 1, 2], "n_action_steps": 50, "chunk_size": 100,
        "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        "normalization_mapping": {"STATE": "MEAN_STD" if normalization else "IDENTITY",
                                  "ACTION": "MEAN_STD" if normalization else "IDENTITY"}}))
    return path


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Relative baseline comparison attempted network access")
    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _action_findings(findings):
    return [finding for finding in findings if finding.check_id in {
        "PRIVILEGE_ACTION_GAIN_ESCALATION", "PRIVILEGE_ACTION_OFFSET_ESCALATION", "PRIVILEGE_ACTION_ENVELOPE_SHIFT"}]


@pytest.mark.parametrize("normalization", [False, True])
def test_matching_relative_pipeline_compares_against_itself_without_unknown(tmp_path, normalization):
    checkpoint = _checkpoint(tmp_path / "checkpoint", normalization=normalization)
    findings = run_privilege_delta(checkpoint, checkpoint)
    action = _action_findings(findings)
    assert len(action) == 3
    assert all(finding.status is DiagnosticStatus.PASS for finding in action)
    assert all(finding.details["processor_execution_required"] is True for finding in action)
    assert not [finding for finding in findings if finding.details.get("required_coverage") and finding.status is DiagnosticStatus.UNKNOWN]


def test_paired_preflight_proof_retains_exact_composed_action_gain_failure(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    action = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    steps = [NormalizerProcessorStep(features=action,
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": {"mean": [0.] * 14, "std": [.1] * 14}}) for _ in range(5)]
    steps.append(RelativeActionsProcessorStep(enabled=True))
    PolicyProcessorPipeline(name="policy_preprocessor", steps=steps).save_pretrained(
        checkpoint, config_filename="policy_preprocessor.json")
    report = run_preflight(checkpoint)
    failure, = [finding for finding in report.findings
                 if finding.check_id == "PATHOLOGICAL_STD_GAIN" and finding.status is DiagnosticStatus.FAIL]
    assert failure.details["comparison_scope"] == "composed_forward_normalization"
    assert failure.details["step_index"] == 4
    assert failure.provenance.violating_indices == list(range(14))
    assert failure.provenance.tensor_key == "action.std"
    assert failure.provenance.file_path.endswith("step_4_normalizer_processor.safetensors")
    assert failure.details["saved_values"] == pytest.approx([.1] * 14)
    assert all(gain > 99_999 for gain in failure.details["composed_gains"])


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("normalization", [False, True])
def test_current_first_relative_cli_self_baseline_passes(tmp_path, strict, normalization):
    checkpoint = _checkpoint(tmp_path / "checkpoint", normalization=normalization)
    args = ["--policy", str(checkpoint), "--baseline", str(checkpoint)]
    if strict:
        args.append("--strict")
    assert main(args) == 0


def test_copied_relative_checkpoints_have_matching_state_dependent_maps(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", normalization=True)
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    findings = _action_findings(run_privilege_delta(baseline, candidate))
    assert all(finding.status is DiagnosticStatus.PASS for finding in findings)
    maps = findings[0].details["affine_maps"]
    assert maps["postprocessor"]["candidate"]["state_coefficient"] == [1.] * 14
    assert maps["postprocessor"]["candidate"]["action_slope"] == [2.] * 14


@pytest.mark.parametrize("case", ["observation_mean", "observation_std", "action_mean", "action_std"])
def test_relative_offsets_cannot_hide_surrounding_affine_stat_drift(tmp_path, case):
    baseline = _checkpoint(tmp_path / "baseline", normalization=True)
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    role = "pre" if case.startswith("observation") else "post"
    config = json.loads((candidate / f"policy_{role}processor.json").read_text())
    entry = next(entry for entry in config["steps"] if "state_file" in entry)
    path = candidate / entry["state_file"]
    values = load_file(path)
    key = ("observation.state." if role == "pre" else "action.") + case.rsplit("_", 1)[1]
    values[key][4] += 1.
    save_file(values, path)
    findings = _action_findings(run_privilege_delta(baseline, candidate))
    drift = [finding for finding in findings if finding.status is DiagnosticStatus.FAIL]
    assert drift
    assert all(finding.provenance.violating_indices == [4] for finding in drift)
    assert all(finding.details["scope"] == "paired_relative_action_affine_mapping" for finding in drift)
    assert all(finding.provenance.tensor_key == key for finding in drift)
    assert all(finding.provenance.file_path == str(path) for finding in drift)


def test_relative_effective_mask_drift_has_exact_coordinate(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline")
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path = candidate / "policy_preprocessor.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["exclude_joints"] = ["joint4"]
    path.write_text(json.dumps(config))
    drift = [finding for finding in _action_findings(run_privilege_delta(baseline, candidate))
             if finding.status is DiagnosticStatus.FAIL]
    assert drift
    assert all(finding.provenance.violating_indices == [4] for finding in drift)
    assert any(finding.details["mask_changed"] for finding in drift)


def test_equivalent_mask_spelling_does_not_create_incompatible_contract(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", mask=["Joint4"])
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path = candidate / "policy_preprocessor.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["exclude_joints"] = ["jOiNt4"]
    path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    assert not [finding for finding in findings if finding.status in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT}]
    assert all(finding.status is DiagnosticStatus.PASS for finding in _action_findings(findings))


def test_anchor_contract_change_fails_even_with_identical_native_processors(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline")
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path = candidate / "config.json"
    config = json.loads(path.read_text())
    config["observation_delta_indices"] = [-2, -1, 0]
    path.write_text(json.dumps(config))
    findings = _action_findings(run_privilege_delta(baseline, candidate))
    assert any(finding.status is DiagnosticStatus.FAIL and finding.details["anchor_changed"] for finding in findings)
    assert main(["--policy", str(candidate), "--baseline", str(candidate)]) == 1


def test_uninspected_step_keeps_relative_comparison_unknown(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    path = checkpoint / "policy_postprocessor.json"
    config = json.loads(path.read_text())
    config["steps"].insert(0, {"registry_name": "uninspected_custom_action_step", "config": {}})
    path.write_text(json.dumps(config))
    assert all(finding.status is DiagnosticStatus.UNKNOWN for finding in _action_findings(run_privilege_delta(checkpoint, checkpoint)))


def test_relative_intermediate_fp16_overflow_cannot_cancel_in_affine_composition(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    feature = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    post = PolicyProcessorPipeline.from_pretrained(checkpoint, config_filename="policy_postprocessor.json", local_files_only=True)
    absolute = post.steps[0]
    steps = [DeviceProcessorStep(device="cpu", float_dtype="float16"), absolute]
    for mean, std in ((50_000., 50_000.), (-0.995, 1.99e-5)):
        steps.append(UnnormalizerProcessorStep(features=feature,
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={"action": {"mean": [mean] * 14, "std": [std] * 14}}))
    PolicyProcessorPipeline(name="policy_postprocessor", steps=steps).save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["normalization_mapping"]["ACTION"] = "MEAN_STD"
    config_path.write_text(json.dumps(config))
    failures = [finding for finding in run_privilege_delta(checkpoint, checkpoint)
                if finding.check_id == "PRIVILEGE_INTERMEDIATE_NON_FINITE" and finding.status is DiagnosticStatus.FAIL]
    assert failures
    assert failures[0].details["step_index"] == 2
    assert failures[0].provenance.violating_indices == list(range(14))


def test_small_relative_offset_drift_is_not_hidden_by_ordinary_origin_threshold(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", normalization=True)
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    config = json.loads((candidate / "policy_preprocessor.json").read_text())
    path = candidate / config["steps"][0]["state_file"]
    state = load_file(path)
    state["observation.state.mean"][2] += .001
    save_file(state, path)
    failures = [finding for finding in run_privilege_delta(baseline, candidate)
                if finding.check_id == "PRIVILEGE_ACTION_OFFSET_ESCALATION" and finding.status is DiagnosticStatus.FAIL]
    assert len(failures) == 1
    assert failures[0].provenance.violating_indices == [2]
    assert failures[0].details["relative_offset_invariance_atol"] == 1e-12


@pytest.mark.parametrize("enabled", [False, True])
def test_subthreshold_gain_still_flags_expanded_envelope_when_relative_mask_is_inactive(tmp_path, enabled):
    baseline = _checkpoint(tmp_path / "baseline", normalization=True, enabled=enabled,
                           mask=[f"joint{index}" for index in range(14)])
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    config = json.loads((candidate / "policy_postprocessor.json").read_text())
    path = candidate / next(entry["state_file"] for entry in config["steps"] if "state_file" in entry)
    state = load_file(path)
    state["action.std"][4] *= 1.01
    save_file(state, path)
    findings = run_privilege_delta(baseline, candidate)
    assert not [finding for finding in findings if finding.check_id == "PRIVILEGE_ACTION_GAIN_ESCALATION"
                and finding.status is DiagnosticStatus.FAIL]
    envelope = [finding for finding in findings if finding.check_id == "PRIVILEGE_ACTION_ENVELOPE_SHIFT"
                and finding.status is DiagnosticStatus.FAIL]
    assert len(envelope) == 1
    assert envelope[0].provenance.violating_indices == [4]


def test_disabled_relative_and_absolute_steps_are_verified_identity_conversions(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint", normalization=True, enabled=False)
    findings = _action_findings(run_privilege_delta(checkpoint, checkpoint))
    assert len(findings) == 3
    assert all(finding.status is DiagnosticStatus.PASS for finding in findings)
    assert all(finding.details["baseline_enabled"] == [False, False] for finding in findings)


def test_disabled_relative_mask_metadata_does_not_change_identity_mapping(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", enabled=False)
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path = candidate / "policy_preprocessor.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["exclude_joints"] = ["joint4"]
    path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    assert all(finding.status is DiagnosticStatus.PASS for finding in _action_findings(findings))
    assert not [finding for finding in findings if finding.status in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT}]


@pytest.mark.parametrize("enabled", [False, True])
def test_inactive_relative_mask_strict_cli_self_baseline_passes(tmp_path, enabled):
    checkpoint = _checkpoint(tmp_path / "checkpoint", normalization=True, enabled=enabled,
                             mask=[f"joint{index}" for index in range(14)])
    assert main(["--policy", str(checkpoint), "--baseline", str(checkpoint), "--strict"]) == 0


def test_small_pre_action_scale_contraction_is_exact_coordinate_drift(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", normalization=True)
    config = json.loads((baseline / "policy_preprocessor.json").read_text())
    filename = config["steps"][0]["state_file"]
    state = load_file(baseline / filename)
    state["action.mean"].zero_()
    save_file(state, baseline / filename)
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    state["action.std"][4] *= 1.001
    save_file(state, candidate / filename)
    findings = run_privilege_delta(baseline, candidate)
    assert not [finding for finding in findings if finding.check_id == "PRIVILEGE_ACTION_GAIN_ESCALATION"
                and finding.status is DiagnosticStatus.FAIL]
    drift = [finding for finding in findings if finding.check_id == "PRIVILEGE_ACTION_OFFSET_ESCALATION"
             and finding.status is DiagnosticStatus.FAIL]
    assert len(drift) == 1
    assert drift[0].provenance.violating_indices == [4]
    assert drift[0].provenance.tensor_key == "action.std"


def test_equivalent_observation_reparameterization_uses_identical_native_probe_inputs(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", normalization=True)
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    normalizers = []
    for cls in (NormalizerProcessorStep, UnnormalizerProcessorStep):
        normalizers.append(cls(features=feature, norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                               stats={"observation.state": {"mean": [10.] * 14, "std": [2.] * 14}}, eps=0.))
    relative = RelativeActionsProcessorStep(enabled=True, action_names=[f"joint{index}" for index in range(14)])
    PolicyProcessorPipeline(name="policy_preprocessor", steps=normalizers + [relative]).save_pretrained(
        baseline, config_filename="policy_preprocessor.json")
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    config = json.loads((candidate / "policy_preprocessor.json").read_text())
    for entry in config["steps"][:2]:
        path = candidate / entry["state_file"]
        state = load_file(path)
        state["observation.state.mean"].fill_(20.)
        save_file(state, path)
    findings = _action_findings(run_privilege_delta(baseline, candidate))
    assert len(findings) == 3
    assert all(finding.status is DiagnosticStatus.PASS for finding in findings)
    assert all(finding.details["affine_maps"]["preprocessor"]["native_probe_shift"] == [0.] * 14
               for finding in findings)
