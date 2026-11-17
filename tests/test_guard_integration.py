"""Runtime guards remain real native steps throughout offline audits and diffs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)

from lerobot_flinch.cli import main
from lerobot_flinch.guard import InputEnvelopeGuardStep, add_input_guard
from lerobot_flinch.models import DiagnosticStatus
from lerobot_flinch.preflight import _native_normalization_probe, run_preflight
from lerobot_flinch.privilege_delta import run_privilege_delta


def _checkpoint(root: Path, *, relative: bool = False, mode: str = "raise", limit: float = 10.0) -> Path:
    root.mkdir()
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
        "action": PolicyFeature(FeatureType.ACTION, (14,)),
    }
    mapping = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {key: {"mean": torch.zeros(14), "std": torch.ones(14)} for key in features}
    normalizer = NormalizerProcessorStep(features=features, norm_map=mapping, stats=stats)
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=[f"joint{i}" for i in range(14)])
    pre = PolicyProcessorPipeline(steps=[normalizer, *([relative_step] if relative else [])])
    add_input_guard(pre, max_abs=limit, on_violation=mode)
    post = PolicyProcessorPipeline(
        steps=[
            *([AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)] if relative else []),
            UnnormalizerProcessorStep(
                features={"action": features["action"]},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={"action": stats["action"]},
            ),
        ]
    )
    pre.save_pretrained(root, config_filename="policy_preprocessor.json")
    post.save_pretrained(root, config_filename="policy_postprocessor.json")
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "custom_guard_fixture",
                "n_obs_steps": 3 if relative else 1,
                "observation_delta_indices": [0, 1, 2] if relative else [0],
                "n_action_steps": 50,
                "chunk_size": 100,
                "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
                "output_features": {"action": {"type": "ACTION", "shape": [14]}},
                "normalization_mapping": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
            }
        )
    )
    return root


def _config(root: Path) -> tuple[Path, dict]:
    path = root / "policy_preprocessor.json"
    return path, json.loads(path.read_text())


@pytest.mark.parametrize("mode", ["raise", "clamp"])
@pytest.mark.parametrize("relative", [False, True])
def test_guarded_self_baseline_is_strict_clean(tmp_path: Path, mode: str, relative: bool) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint", relative=relative, mode=mode)
    assert main(["--policy", str(checkpoint), "--baseline", str(checkpoint), "--strict"]) == 0
    findings = run_privilege_delta(checkpoint, checkpoint)
    assert not any(
        finding.status in {DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT}
        or (finding.status is DiagnosticStatus.UNKNOWN and not finding.details.get("strict_exempt"))
        for finding in findings
    )
    if relative and mode == "clamp":
        action = [f for f in findings if f.check_id == "PRIVILEGE_ACTION_GAIN_ESCALATION"]
        assert action and action[0].details["affine_mapping_claimed"] is False
        assert action[0].details["exact_processor_contracts_verified"] is True


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("remove", "input_guard_removed"),
        ("widen", "input_guard_envelope_widened"),
        ("keys", "input_guard_key_coverage_reduced"),
        ("clamp", "input_guard_raise_downgraded_to_clamp"),
    ],
)
def test_guard_weakening_is_a_proven_privilege_failure(tmp_path: Path, mutation: str, reason: str) -> None:
    baseline = _checkpoint(tmp_path / "baseline")
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path, config = _config(candidate)
    if mutation == "remove":
        config["steps"].pop(1)
    elif mutation == "widen":
        config["steps"][1]["config"]["max_abs"] = 100.0
    elif mutation == "keys":
        config["steps"][1]["config"]["keys"] = ["observation.other"]
    else:
        config["steps"][1]["config"]["on_violation"] = "clamp"
    path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    (guard,) = [
        f for f in findings if f.check_id == "PRIVILEGE_GUARD_REMOVAL" and f.details.get("reason") == reason
    ]
    assert guard.status is DiagnosticStatus.FAIL
    assert guard.details["baseline_step_index"] == 1
    assert guard.details["step_index"] == (None if mutation == "remove" else 1)
    assert guard.provenance is not None
    assert guard.provenance.file_path == str(path)
    assert guard.provenance.sha256_prefix == hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    assert main(["--policy", str(candidate), "--baseline", str(baseline)]) == 1


def test_narrower_envelope_does_not_invent_a_compatibility_conflict(tmp_path: Path) -> None:
    baseline = _checkpoint(tmp_path / "baseline")
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path, config = _config(candidate)
    config["steps"][1]["config"]["max_abs"] = 5.0
    path.write_text(json.dumps(config))
    assert main(["--policy", str(candidate), "--baseline", str(baseline), "--strict"]) == 0


def test_numeric_probe_executes_real_clamp_and_keeps_normalizer_prefix() -> None:
    feature = PolicyFeature(FeatureType.STATE, (2,))
    normalizer = NormalizerProcessorStep(
        features={"observation.state": feature},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"observation.state": {"mean": torch.zeros(2), "std": torch.ones(2)}},
    )
    entries = [
        {"registry_name": "normalizer_processor", "config": normalizer.get_config()},
        {
            "registry_name": "flinch_input_guard",
            "config": {"max_abs": 0.5, "on_violation": "clamp", "keys": ["observation.state"]},
        },
    ]
    snapshots = _native_normalization_probe(
        entries, {0: normalizer}, "observation.state", torch.tensor([[2.0, -2.0]])
    )
    assert [s["step_class"] for s in snapshots] == ["NormalizerProcessorStep", "InputEnvelopeGuardStep"]
    torch.testing.assert_close(snapshots[0]["output"], torch.tensor([[2.0, -2.0]]))
    torch.testing.assert_close(snapshots[1]["output"], torch.tensor([[0.5, -0.5]]))


def test_guard_cannot_hide_composed_gain_before_its_rejection(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    steps = [
        NormalizerProcessorStep(
            features=feature,
            norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
            stats={"observation.state": {"mean": torch.zeros(14), "std": torch.full((14,), 0.1)}},
        )
        for _ in range(5)
    ]
    steps.append(InputEnvelopeGuardStep(keys=["observation.state"]))
    PolicyProcessorPipeline(steps=steps).save_pretrained(
        checkpoint, config_filename="policy_preprocessor.json"
    )
    report = run_preflight(checkpoint)
    assert any(
        f.check_id == "PATHOLOGICAL_STD_GAIN"
        and f.status is DiagnosticStatus.FAIL
        and f.details.get("comparison_scope") == "composed_forward_normalization"
        for f in report.findings
    )
    (guard,) = [f for f in report.findings if f.check_id == "INPUT_ENVELOPE_GUARD"]
    assert guard.details["step_index"] == 5
    assert guard.details["guard_executed"] is True
    assert guard.details["offending_indices"]


def test_clamp_separates_affine_segments_instead_of_becoming_identity(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    feature = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    stages = [
        NormalizerProcessorStep(
            features=feature,
            norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
            stats={"observation.state": {"mean": torch.zeros(14), "std": torch.full((14,), 0.001)}},
        )
        for _ in range(2)
    ]
    steps = [stages[0], InputEnvelopeGuardStep(max_abs=0.001, on_violation="clamp"), stages[1]]
    PolicyProcessorPipeline(steps=steps).save_pretrained(
        checkpoint, config_filename="policy_preprocessor.json"
    )
    report = run_preflight(checkpoint)
    assert not any(
        f.check_id == "PATHOLOGICAL_STD_GAIN"
        and f.status is DiagnosticStatus.FAIL
        and f.details.get("comparison_scope") == "composed_forward_normalization"
        for f in report.findings
    )


def test_changed_nonlinear_relative_mapping_is_not_claimed_affine(tmp_path: Path) -> None:
    baseline = _checkpoint(tmp_path / "baseline", relative=True, mode="clamp", limit=0.5)
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline, candidate)
    path, config = _config(candidate)
    config["steps"][1]["config"]["max_abs"] = 1.0
    path.write_text(json.dumps(config))
    findings = run_privilege_delta(baseline, candidate)
    action = [f for f in findings if f.check_id == "PRIVILEGE_ACTION_OFFSET_ESCALATION"]
    assert action and action[0].status is DiagnosticStatus.FAIL
    assert action[0].details["affine_mapping_claimed"] is False
