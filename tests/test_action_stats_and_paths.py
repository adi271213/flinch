"""Action un-normalization that disagrees with training, and path typos as usage errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    UnnormalizerProcessorStep,
)
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


def _action_step(cls, std=1.0, mean=0.0, *, mode=NormalizationMode.MEAN_STD, eps=1e-8):
    stats = (
        {"mean": torch.full((14,), float(mean)), "std": torch.full((14,), float(std))}
        if mode is NormalizationMode.MEAN_STD
        else {
            name: torch.zeros(14)
            for name in {
                NormalizationMode.MIN_MAX: ("min", "max"),
                NormalizationMode.QUANTILES: ("q01", "q99"),
                NormalizationMode.QUANTILE10: ("q10", "q90"),
            }[mode]
        }
    )
    return cls(
        features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
        norm_map={FeatureType.ACTION: mode},
        stats={"action": stats},
        eps=eps,
    )


def _save_action_chains(clean, before, after):
    pre = PolicyProcessorPipeline(name="policy_preprocessor", steps=before)
    post = PolicyProcessorPipeline(name="policy_postprocessor", steps=after)
    pre.save_pretrained(clean, config_filename="preprocessor_config.json")
    post.save_pretrained(clean, config_filename="postprocessor_config.json")
    return pre, post


def test_native_multistage_inverse_is_not_compared_as_a_cross_product(clean):
    pre, post = _save_action_chains(
        clean,
        [_action_step(NormalizerProcessorStep, 2), _action_step(NormalizerProcessorStep, 3)],
        [_action_step(UnnormalizerProcessorStep, 3), _action_step(UnnormalizerProcessorStep, 2)],
    )
    action = torch.full((1, 14), 0.9)
    assert torch.allclose(post(pre({"action": action}))["action"], action)
    (finding,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert finding.status is DiagnosticStatus.PASS
    assert finding.details["comparison_scope"] == "composed_native_action_inverse"


def test_native_multistage_inverse_drift_keeps_exact_coordinates(clean):
    altered = _action_step(UnnormalizerProcessorStep, 3)
    altered.stats["action"]["std"][2] = 3.1
    altered.to(dtype=torch.float32)
    pre, post = _save_action_chains(
        clean,
        [_action_step(NormalizerProcessorStep, 2), _action_step(NormalizerProcessorStep, 3)],
        [altered, _action_step(UnnormalizerProcessorStep, 2)],
    )
    action = torch.full((1, 14), 0.9)
    assert not torch.allclose(post(pre({"action": action}))["action"], action)
    (finding,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert finding.status is DiagnosticStatus.CONFLICT
    assert finding.provenance.violating_indices == [2]
    assert len(finding.details["contributing_stages"]) == 4


def test_native_same_stats_epsilon_loss_is_not_invented_stat_drift(clean):
    pre, post = _save_action_chains(
        clean,
        [_action_step(NormalizerProcessorStep, 0.5, eps=1)],
        [_action_step(UnnormalizerProcessorStep, 0.5)],
    )
    action = torch.full((1, 14), 0.9)
    assert torch.allclose(post(pre({"action": action}))["action"], action / 3)
    assert not _findings(clean, "ACTION_STATS_CONFLICT")


@pytest.mark.parametrize("stat,baseline,candidate", [("mean", 2048.0, 2048.1), ("std", 1e-8, 2e-8)])
def test_action_stat_tolerance_does_not_hide_offset_or_tiny_scale_drift(clean, stat, baseline, candidate):
    before = _action_step(NormalizerProcessorStep, 0.168827, 2048.0)
    after = _action_step(UnnormalizerProcessorStep, 0.168827, 2048.0)
    before.stats["action"][stat][2] = baseline
    after.stats["action"][stat][2] = candidate
    before.to(dtype=torch.float32)
    after.to(dtype=torch.float32)
    _save_action_chains(clean, [before], [after])
    (finding,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert finding.status is DiagnosticStatus.CONFLICT
    assert finding.provenance.violating_indices == [2]


def test_stat_layouts_can_broadcast_within_a_matching_declared_action_shape(clean):
    state = load_file(clean / POST)
    state["action.std"] = state["action.std"][:1].clone()
    save_file(state, clean / POST)
    assert not _findings(clean, "ACTION_STATS_CONFLICT")
    state["action.std"] = torch.ones(2)
    save_file(state, clean / POST)
    failures = _findings(clean, "FEATURE_CONTRACT")
    assert any(finding.status is DiagnosticStatus.FAIL for finding in failures)
    assert run_checks(clean).has_failures()


def test_action_conflict_provenance_hashes_the_active_safetensors(clean):
    import hashlib

    _edit(clean / POST, action__std=(3, 1.5))
    (finding,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert finding.provenance.file_path == str(clean / POST)
    assert finding.provenance.sha256_prefix == hashlib.sha256((clean / POST).read_bytes()).hexdigest()[:12]


def test_native_action_path_is_active_without_a_feature_declaration(clean):
    path = clean / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["features"].pop("action")
    path.write_text(json.dumps(config))
    _edit(clean / POST, action__std=(3, 1.5))
    (finding,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert finding.status is DiagnosticStatus.CONFLICT
    assert finding.provenance.violating_indices == [3]


def test_identity_action_buffers_are_not_compared_or_screened_as_inverse_scale(clean):
    for filename in ("preprocessor_config.json", "postprocessor_config.json"):
        path = clean / filename
        config = json.loads(path.read_text())
        config["steps"][0]["config"]["norm_map"]["ACTION"] = "IDENTITY"
        path.write_text(json.dumps(config))
    _edit(clean / POST, action__std=(3, 1e9))
    assert not _findings(clean, "ACTION_STATS_CONFLICT")
    assert not _findings(clean, "EXTREME_ACTION_SCALE")


def test_effective_fp16_stat_rounding_is_not_a_conflict(clean):
    before = _action_step(NormalizerProcessorStep, 50000)
    after = _action_step(UnnormalizerProcessorStep, 49984)
    _save_action_chains(
        clean,
        [DeviceProcessorStep(device="cpu", float_dtype="float16"), before],
        [DeviceProcessorStep(device="cpu", float_dtype="float16"), after],
    )
    assert not _findings(clean, "ACTION_STATS_CONFLICT")


@pytest.mark.parametrize(
    "mode", [NormalizationMode.MIN_MAX, NormalizationMode.QUANTILES, NormalizationMode.QUANTILE10]
)
def test_inverse_extreme_scale_uses_native_zero_span_epsilon(clean, mode):
    post_step = _action_step(UnnormalizerProcessorStep, mode=mode, eps=1e9)
    _, post = _save_action_chains(clean, [], [post_step])
    observed = post({"action": torch.ones((1, 14))})["action"]
    assert observed[0, 2] == 1e9
    (finding,) = _findings(clean, "EXTREME_ACTION_SCALE")
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert finding.details["action_scale_values"][2] == 5e8
    assert finding.provenance.violating_indices == list(range(14))


def test_uninspectable_multistage_action_mapping_is_unknown(clean):
    _save_action_chains(
        clean,
        [_action_step(NormalizerProcessorStep, 2), _action_step(NormalizerProcessorStep, 3)],
        [_action_step(UnnormalizerProcessorStep, 3), _action_step(UnnormalizerProcessorStep, 2)],
    )
    path = clean / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(1, {"registry_name": "unknown_action_step", "config": {}})
    path.write_text(json.dumps(config))
    (finding,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert finding.status is DiagnosticStatus.UNKNOWN
    assert finding.details["required_coverage"] is True


@pytest.mark.parametrize(
    "mode",
    [
        NormalizationMode.MIN_MAX,
        NormalizationMode.QUANTILES,
        NormalizationMode.QUANTILE10,
        NormalizationMode.MEAN_STD,
    ],
)
def test_inverse_epsilon_contract_is_checked_only_where_native_arithmetic_uses_it(clean, mode):
    before = _action_step(NormalizerProcessorStep, mode=mode, eps=0.001)
    after = _action_step(UnnormalizerProcessorStep, mode=mode, eps=1.0)
    _save_action_chains(clean, [before], [after])
    findings = _findings(clean, "ACTION_STATS_CONFLICT")
    if mode is NormalizationMode.MEAN_STD:
        assert not findings
    else:
        (finding,) = findings
        assert finding.status is DiagnosticStatus.CONFLICT
        assert finding.details["statistic"] == "effective_span"
        assert finding.provenance.violating_indices == list(range(14))
        expected = UnnormalizerProcessorStep(**before.get_config())
        expected.load_state_dict(before.state_dict())
        expected_output = PolicyProcessorPipeline(steps=[expected])({"action": torch.zeros(1, 14)})["action"]
        actual_output = PolicyProcessorPipeline(steps=[after])({"action": torch.zeros(1, 14)})["action"]
        assert torch.allclose(actual_output / expected_output, torch.full((1, 14), 1000.0))


@pytest.mark.parametrize("drift_mask", [False, True])
def test_masked_action_coordinates_follow_native_passthrough_semantics(clean, drift_mask):
    for filename, statefile, registry in (
        ("preprocessor_config.json", PRE, "molmoact2_masked_normalizer"),
        ("postprocessor_config.json", POST, "molmoact2_masked_unnormalizer"),
    ):
        path = clean / filename
        config = json.loads(path.read_text())
        config["steps"][0]["registry_name"] = registry
        path.write_text(json.dumps(config))
        state = load_file(clean / statefile)
        state["action.mask"] = torch.ones(14, dtype=torch.bool)
        state["action.mask"][3] = False
        if statefile == POST:
            state["action.std"][3] = 1e9
            if drift_mask:
                state["action.mask"][3] = True
        save_file(state, clean / statefile)
    findings = _findings(clean, "ACTION_STATS_CONFLICT")
    if drift_mask:
        assert any(
            f.details.get("statistic") == "mask" and f.provenance.violating_indices == [3] for f in findings
        )
        assert _findings(clean, "EXTREME_ACTION_SCALE")
    else:
        assert not findings
        assert not _findings(clean, "EXTREME_ACTION_SCALE")


def test_multistage_comparison_allows_equivalent_stage_factorization(clean):
    _save_action_chains(
        clean,
        [_action_step(NormalizerProcessorStep, 2), _action_step(NormalizerProcessorStep, 3)],
        [_action_step(UnnormalizerProcessorStep, 6)],
    )
    (finding,) = _findings(clean, "ACTION_STATS_CONFLICT")
    assert finding.status is DiagnosticStatus.PASS


def test_scalar_and_vector_action_stats_define_the_same_native_inverse(clean):
    before = _action_step(NormalizerProcessorStep, 0.5)
    before.stats["action"] = {"mean": torch.tensor(0.0), "std": torch.tensor(0.5)}
    before.to(dtype=torch.float32)
    after = _action_step(UnnormalizerProcessorStep, 0.5)
    pre, post = _save_action_chains(clean, [before], [after])
    action = torch.full((1, 14), 0.9)
    assert torch.equal(post(pre({"action": action}))["action"], action)
    assert not _findings(clean, "ACTION_STATS_CONFLICT")
