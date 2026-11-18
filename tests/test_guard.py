"""The runtime input guard on real LeRobot preprocessors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    batch_to_transition,
    transition_to_batch,
)

from lerobot_flinch.cli import run_checks
from lerobot_flinch.guard import InputEnvelopeError, InputEnvelopeGuardStep, add_input_guard
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def fixtures(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path / "fixtures")


def _preprocessor(checkpoint: Path) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="preprocessor_config.json",
        local_files_only=True,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )


def _sample(checkpoint: Path) -> dict[str, torch.Tensor]:
    with np.load(checkpoint / "sample_observation.npz") as sample:
        return {"observation.state": torch.from_numpy(sample["observation.state"]).float()}


def test_guard_stops_the_near_zero_std_explosion_on_a_recorded_sample(fixtures: dict[str, Path]) -> None:
    preprocessor = add_input_guard(_preprocessor(fixtures["bug_5289_pathological_gain"]))

    with pytest.raises(InputEnvelopeError, match=r"observation\.state .*indices \[9\]"):
        preprocessor(_sample(fixtures["bug_5289_pathological_gain"]))


def test_guard_passes_healthy_inputs_through_unchanged(fixtures: dict[str, Path]) -> None:
    checkpoint = fixtures["clean_act_modern"]
    expected = _preprocessor(checkpoint)(_sample(checkpoint))["observation.state"]
    guarded = add_input_guard(_preprocessor(checkpoint))(_sample(checkpoint))["observation.state"]
    torch.testing.assert_close(guarded, expected, rtol=0, atol=0)


def test_clamp_mode_bounds_only_the_offending_values(fixtures: dict[str, Path]) -> None:
    checkpoint = fixtures["bug_5289_pathological_gain"]
    raw = _preprocessor(checkpoint)(_sample(checkpoint))["observation.state"]
    clamped = add_input_guard(_preprocessor(checkpoint), max_abs=5.0, on_violation="clamp")(
        _sample(checkpoint)
    )
    out = clamped["observation.state"]
    assert out.abs().max() <= 5.0
    inside = raw.abs() <= 5.0
    torch.testing.assert_close(out[inside], raw[inside], rtol=0, atol=0)


def test_non_finite_values_raise_even_in_clamp_mode() -> None:
    guard = InputEnvelopeGuardStep(on_violation="clamp")
    with pytest.raises(InputEnvelopeError, match="indices \\[1\\]"):
        guard.observation({"observation.state": torch.tensor([0.1, float("nan"), 0.2])})


def test_guard_survives_save_and_load(fixtures: dict[str, Path], tmp_path: Path) -> None:
    preprocessor = add_input_guard(_preprocessor(fixtures["clean_act_modern"]), max_abs=7.5)
    preprocessor.save_pretrained(tmp_path / "saved", config_filename="preprocessor_config.json")

    reloaded = _preprocessor(tmp_path / "saved")
    (guard,) = [step for step in reloaded.steps if isinstance(step, InputEnvelopeGuardStep)]
    assert guard.get_config() == {
        "max_abs": 7.5,
        "on_violation": "raise",
        "keys": ["observation.state"],
        "managed_by_helper": True,
    }


def test_flinch_audits_a_guarded_checkpoint_like_an_unguarded_one(
    fixtures: dict[str, Path], tmp_path: Path
) -> None:
    checkpoint = fixtures["clean_act_modern"]
    before = sorted((f.check_id, f.status.value) for f in run_checks(checkpoint).findings)
    add_input_guard(_preprocessor(checkpoint)).save_pretrained(
        checkpoint, config_filename="preprocessor_config.json"
    )
    after = sorted((f.check_id, f.status.value) for f in run_checks(checkpoint).findings)
    assert after == before


def test_add_input_guard_goes_right_after_the_normalizer(fixtures: dict[str, Path]) -> None:
    steps = add_input_guard(_preprocessor(fixtures["clean_act_modern"])).steps
    position = next(i for i, step in enumerate(steps) if isinstance(step, InputEnvelopeGuardStep))
    assert type(steps[position - 1]).__name__ == "NormalizerProcessorStep"

    with pytest.raises(ValueError, match="no NormalizerProcessorStep"):
        add_input_guard(PolicyProcessorPipeline(steps=[]))


@pytest.mark.parametrize("config", [{"max_abs": 0}, {"max_abs": float("inf")}, {"on_violation": "ignore"}])
def test_invalid_guard_config_is_rejected(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        InputEnvelopeGuardStep(**config)  # type: ignore[arg-type]


def _mixed_normalizer(*, state_mode=NormalizationMode.MEAN_STD, selected=None, state_stats=True):
    return NormalizerProcessorStep(
        features={
            "observation.state": PolicyFeature(FeatureType.STATE, (1,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (1,)),
            "observation.raw": PolicyFeature(FeatureType.ENV, (1,)),
        },
        norm_map={
            FeatureType.STATE: state_mode,
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
            FeatureType.ENV: NormalizationMode.IDENTITY,
        },
        normalize_observation_keys=selected,
        stats={"observation.state": {"mean": torch.zeros(1), "std": torch.ones(1)}} if state_stats else {},
    )


def test_default_helper_preserves_healthy_identity_camera_and_raw_observations():
    pre = PolicyProcessorPipeline(steps=[_mixed_normalizer()])
    returned = add_input_guard(pre, max_abs=10.0)
    (guard,) = [step for step in pre.steps if isinstance(step, InputEnvelopeGuardStep)]
    assert returned is pre
    assert guard.keys == ["observation.state"]
    batch = {
        "observation.state": torch.tensor([0.5]),
        "observation.images.front": torch.tensor([255.0]),
        "observation.raw": torch.tensor([2048.0]),
    }
    output = pre(batch)
    assert output["observation.images.front"] is batch["observation.images.front"]
    assert output["observation.raw"] is batch["observation.raw"]
    assert output["observation.state"].item() == 0.5


@pytest.mark.parametrize("normalizer", ["identity", "excluded", "unbound"])
def test_default_helper_does_not_label_untouched_observations_as_normalized(normalizer):
    kwargs = {
        "identity": {"state_mode": NormalizationMode.IDENTITY},
        "excluded": {"selected": set()},
        "unbound": {"state_stats": False},
    }[normalizer]
    pre = PolicyProcessorPipeline(steps=[_mixed_normalizer(**kwargs)])
    original = pre.steps
    with pytest.raises(ValueError, match="no complete active normalized observation"):
        add_input_guard(pre)
    assert pre.steps is original
    add_input_guard(pre, keys=["observation.raw"])
    with pytest.raises(InputEnvelopeError) as error:
        pre({"observation.raw": torch.tensor([2048.0])})
    assert error.value.key == "observation.raw"


def test_helper_tracks_native_renames_before_the_last_normalizer():
    first = _mixed_normalizer()
    rename = RenameObservationsProcessorStep(rename_map={"observation.state": "observation.renamed"})
    action = NormalizerProcessorStep(
        features={},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": {"mean": torch.zeros(1), "std": torch.ones(1)}},
    )
    pre = add_input_guard(PolicyProcessorPipeline(steps=[first, rename, action]))
    assert pre.steps[-1].keys == ["observation.renamed"]
    with pytest.raises(InputEnvelopeError) as error:
        pre({"observation.state": torch.tensor([20.0])})
    assert error.value.key == "observation.renamed"


def test_reconfiguration_is_idempotent_and_preserves_native_hooks_and_converters(fixtures):
    pre = _preprocessor(fixtures["clean_act_modern"])
    before, after = [], []
    pre.before_step_hooks.append(lambda index, transition: before.append(index))
    pre.after_step_hooks.append(lambda index, transition: after.append(index))
    to_transition, to_output = pre.to_transition, pre.to_output
    before_hooks, after_hooks = pre.before_step_hooks, pre.after_step_hooks
    add_input_guard(pre, max_abs=5.0)
    guard = pre.steps[-1]
    add_input_guard(pre, max_abs=10.0)
    assert [step for step in pre.steps if isinstance(step, InputEnvelopeGuardStep)] == [guard]
    assert guard.max_abs == 10.0
    assert pre.to_transition is to_transition and pre.to_output is to_output
    assert pre.before_step_hooks is before_hooks and pre.after_step_hooks is after_hooks
    assert pre._serialized_state_filenames is None
    pre({"observation.state": torch.full((1, 14), 0.2)})
    assert before == after == [0, 1]


def test_reloaded_managed_guard_can_be_reconfigured_without_stale_limits(fixtures, tmp_path):
    pre = add_input_guard(_preprocessor(fixtures["clean_act_modern"]), max_abs=5.0)
    pre.save_pretrained(tmp_path / "saved", config_filename="preprocessor_config.json")
    loaded = _preprocessor(tmp_path / "saved")
    add_input_guard(loaded, max_abs=10.0)
    assert sum(isinstance(step, InputEnvelopeGuardStep) for step in loaded.steps) == 1
    guard = loaded.steps[-1]
    assert guard.max_abs == 10.0
    guard.observation({"observation.state": torch.tensor([6.0])})


@pytest.mark.parametrize("multiple", [False, True])
def test_helper_refuses_ambiguous_or_manual_existing_guards(multiple):
    first = InputEnvelopeGuardStep(max_abs=5.0, managed_by_helper=multiple)
    steps = [_mixed_normalizer(), first]
    if multiple:
        steps.append(InputEnvelopeGuardStep(max_abs=3.0))
    pre = PolicyProcessorPipeline(steps=steps)
    original = list(pre.steps)
    with pytest.raises(ValueError, match="ambiguous"):
        add_input_guard(pre, max_abs=10.0)
    assert list(pre.steps) == original
    assert first.max_abs == 5.0


def test_clamp_copies_dictionary_and_preserves_original_tensors():
    value = torch.tensor([0.1, 20.0, -0.2])
    original = {"observation.state": value}
    result = InputEnvelopeGuardStep(on_violation="clamp").observation(original)
    assert result is not original
    assert original["observation.state"] is value
    assert value.tolist() == pytest.approx([0.1, 20.0, -0.2])
    assert result["observation.state"].tolist() == pytest.approx([0.1, 10.0, -0.2])


def test_guard_error_has_bounded_exact_structured_evidence():
    with pytest.raises(InputEnvelopeError) as error:
        InputEnvelopeGuardStep().observation({"observation.state": torch.full((2, 100), 100_000.0)})
    failure = error.value
    assert failure.key == "observation.state"
    assert failure.total_count == 200
    assert failure.indices == list(range(8))
    assert failure.values == [100_000.0] * 8
    assert len(str(failure)) < 600


def test_guard_catches_actual_fp16_normalizer_nan_and_clamp_does_not_conceal_it(fixtures):
    pre = add_input_guard(_preprocessor(fixtures["bug_4973_fp16_nan"]), on_violation="clamp")
    with np.load(fixtures["bug_4973_fp16_nan"] / "sample_observation.npz") as sample:
        value = torch.from_numpy(sample["observation.state"]).half()
    with pytest.raises(InputEnvelopeError) as error:
        pre({"observation.state": value})
    assert error.value.key == "observation.state"
    assert 4 in error.value.indices
    assert error.value.total_count >= 1
    assert any(not np.isfinite(value) for value in error.value.values)


@pytest.mark.parametrize(
    "config",
    [
        {"max_abs": True},
        {"max_abs": "10"},
        {"max_abs": None},
        {"max_abs": float("nan")},
        {"max_abs": float("-inf")},
        {"on_violation": None},
        {"keys": []},
        {"keys": "observation.state"},
        {"keys": [1]},
        {"keys": [""]},
        {"keys": [" observation.state"]},
        {"keys": ["observation.\nstate"]},
        {"keys": ["observation.state", "observation.state"]},
        {"managed_by_helper": "yes"},
    ],
)
def test_malformed_guard_configuration_is_rejected_with_value_error(config):
    with pytest.raises(ValueError):
        InputEnvelopeGuardStep(**config)


def test_configuration_lists_are_defensive_copies():
    keys = ["observation.state"]
    guard = InputEnvelopeGuardStep(keys=keys)
    keys.append("observation.raw")
    saved = guard.get_config()
    saved["keys"].append("observation.camera")
    assert guard.keys == ["observation.state"]


def test_partial_mask_requires_explicit_tensor_envelope_selection():
    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2MaskedNormalizerProcessorStep

    normalizer = MolmoAct2MaskedNormalizerProcessorStep(
        features={"observation.state": PolicyFeature(FeatureType.STATE, (2,))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={
            "observation.state": {
                "mean": torch.zeros(2),
                "std": torch.ones(2),
                "mask": torch.tensor([True, False]),
            }
        },
    )
    pre = PolicyProcessorPipeline(steps=[normalizer])
    with pytest.raises(ValueError, match="partially normalized"):
        add_input_guard(pre)
    assert len(pre.steps) == 1
    add_input_guard(pre, keys=["observation.state"])
    assert pre.steps[-1].keys == ["observation.state"]
    result = pre({"observation.state": torch.tensor([0.2, 0.8])})["observation.state"]
    torch.testing.assert_close(result, torch.tensor([0.2, 0.8]))


def test_repositioned_guard_invalidates_native_cached_state_filenames(fixtures, tmp_path):
    first = _mixed_normalizer()
    inverse = UnnormalizerProcessorStep(features=first.features, norm_map=first.norm_map, stats=first.stats)
    last = _mixed_normalizer()
    original = PolicyProcessorPipeline(
        name="policy_preprocessor",
        steps=[
            first,
            InputEnvelopeGuardStep(keys=["observation.state"], managed_by_helper=True),
            inverse,
            last,
        ],
    )
    original.save_pretrained(tmp_path / "source", config_filename="preprocessor_config.json")
    loaded = _preprocessor(tmp_path / "source")
    assert len(loaded._serialized_state_filenames) == 4
    add_input_guard(loaded)
    assert len(loaded.steps) == 4 and isinstance(loaded.steps[-1], InputEnvelopeGuardStep)
    assert loaded._serialized_state_filenames is None
    state = loaded.state_dict()
    assert len(state) == 3
    for index, values in enumerate(state.values()):
        values["observation.state.mean"].fill_(index)
    loaded.load_state_dict(state)
    assert [step.state_dict()["observation.state.mean"].item() for step in loaded.steps[:-1]] == [
        0.0,
        1.0,
        2.0,
    ]
    loaded.save_pretrained(tmp_path / "saved", config_filename="preprocessor_config.json")
    reloaded = _preprocessor(tmp_path / "saved")
    batch = {"observation.state": torch.tensor([0.5])}
    torch.testing.assert_close(
        loaded(batch)["observation.state"], reloaded(batch)["observation.state"], rtol=0, atol=0
    )


@pytest.mark.parametrize(("dtype", "limit"), [(torch.float16, 3.3), (torch.bfloat16, 3.31)])
def test_guard_real_limit_does_not_round_outward_in_low_precision(dtype, limit):
    sample = torch.tensor([4.0], dtype=dtype)
    guard = InputEnvelopeGuardStep(max_abs=limit, on_violation="clamp")
    output = guard.observation({"observation.state": sample})["observation.state"]
    assert float(output[0]) <= limit
    boundary = torch.tensor([limit], dtype=dtype)
    assert float(boundary[0]) > limit
    with pytest.raises(InputEnvelopeError):
        InputEnvelopeGuardStep(max_abs=limit).observation({"observation.state": boundary})


def test_clamps_are_counted_and_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    guard = InputEnvelopeGuardStep(max_abs=1.0, on_violation="clamp")
    with caplog.at_level("WARNING"):
        for _ in range(3):
            guard.observation({"observation.state": torch.tensor([0.5, 4.0, -9.0])})
    assert guard.clamp_count == 6
    assert caplog.text.count("Input guard clamped observation.state") == 1
    assert "clamp_count" not in guard.get_config()


def test_saved_guard_loads_without_importing_flinch(fixtures: dict[str, Path], tmp_path: Path) -> None:
    import subprocess
    import sys

    from tests.subprocess_env import source_subprocess_env

    add_input_guard(_preprocessor(fixtures["clean_act_modern"])).save_pretrained(
        tmp_path / "saved", config_filename="preprocessor_config.json"
    )
    script = (
        "import sys; from lerobot.processor import PolicyProcessorPipeline\n"
        "p = PolicyProcessorPipeline.from_pretrained(sys.argv[1], config_filename='preprocessor_config.json',"
        " local_files_only=True)\n"
        "print([type(s).__name__ for s in p.steps])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "saved")],
        env=source_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "InputEnvelopeGuardStep" in result.stdout
