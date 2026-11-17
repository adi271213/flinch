"""The runtime input guard on real LeRobot preprocessors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch

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
    assert guard.get_config() == {"max_abs": 7.5, "on_violation": "raise", "keys": None}


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
