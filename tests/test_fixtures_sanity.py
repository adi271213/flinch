"""Ground-truth behavior checks for the checkpoint fixture generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from safetensors.torch import load_file

from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path)


def _pipeline(checkpoint: Path, filename: str = "preprocessor_config.json"):
    return PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename=filename, local_files_only=True
    )


def _sample(checkpoint: Path, filename: str, key: str) -> torch.Tensor:
    with np.load(checkpoint / filename, allow_pickle=False) as sample:
        assert key in sample.files
        return torch.from_numpy(sample[key].copy())


def _state_files(checkpoint: Path, filename: str) -> list[Path]:
    config = json.loads((checkpoint / filename).read_text())
    assert config["steps"]
    return [checkpoint / step["state_file"] for step in config["steps"] if "state_file" in step]


def test_clean_modern_checkpoint_round_trips_and_normalizes(checkpoints: dict[str, Path]) -> None:
    assert set(checkpoints) == {
        "clean_act_modern",
        "bug_4415_silent_bypass",
        "bug_5289_pathological_gain",
        "bug_4973_fp16_nan",
        "bug_5315_temporal_anchor",
        "legacy_act_embedded",
    }
    checkpoint = checkpoints["clean_act_modern"]
    pre_state_files = _state_files(checkpoint, "preprocessor_config.json")
    post_state_files = _state_files(checkpoint, "postprocessor_config.json")
    assert len(pre_state_files) == len(post_state_files) == 1
    assert all(path.is_file() for path in pre_state_files + post_state_files)

    pre_state = load_file(pre_state_files[0])
    assert {"observation.state.mean", "observation.state.std", "action.mean", "action.std"} <= set(
        pre_state
    )
    preprocessor = _pipeline(checkpoint)
    postprocessor = _pipeline(checkpoint, "postprocessor_config.json")
    assert isinstance(preprocessor.steps[0], NormalizerProcessorStep)
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert preprocessor.steps[0].norm_map[FeatureType.STATE] == NormalizationMode.MEAN_STD
    assert preprocessor.steps[0].norm_map[FeatureType.ACTION] == NormalizationMode.MEAN_STD

    state = _sample(checkpoint, "sample_observation.npz", "observation.state")
    action = _sample(checkpoint, "sample_action.npz", "action")
    assert state.shape[-1] == action.shape[-1] == 14
    normalized = preprocessor({"observation.state": state, "action": action})
    torch.testing.assert_close(normalized["observation.state"], torch.full_like(state, 0.4))
    torch.testing.assert_close(normalized["action"], torch.full_like(action, 0.8))
    restored = postprocessor({"action": normalized["action"]})
    torch.testing.assert_close(restored["action"], action)


def test_4415_missing_stat_key_silently_bypasses_observation(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_4415_silent_bypass"]
    state_files = _state_files(checkpoint, "preprocessor_config.json")
    assert len(state_files) == 1
    saved = load_file(state_files[0])
    assert "wrong_dataset/observation.state.mean" in saved
    assert "wrong_dataset/observation.state.std" in saved
    assert "observation.state.mean" not in saved
    assert "observation.state.std" not in saved

    state = _sample(checkpoint, "sample_observation.npz", "observation.state")
    action = _sample(checkpoint, "sample_action.npz", "action")
    processed = _pipeline(checkpoint)({"observation.state": state, "action": action})
    torch.testing.assert_close(processed["observation.state"], state)
    torch.testing.assert_close(processed["action"], torch.full_like(action, 0.8))


def test_5289_near_constant_coordinate_has_pathological_gain(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    saved = load_file(_state_files(checkpoint, "preprocessor_config.json")[0])
    assert saved["observation.state.std"][9].item() == pytest.approx(1.48e-6)
    state = _sample(checkpoint, "sample_observation.npz", "observation.state")
    assert state[0, 9].item() == pytest.approx(0.20)

    processed = _pipeline(checkpoint)({"observation.state": state})["observation.state"]
    expected = 0.20 / (1.48e-6 + 1e-8)
    assert processed[0, 9].item() == pytest.approx(expected, rel=1e-5)
    assert processed[0, 9].item() > 1e5
    torch.testing.assert_close(processed[0, :9], torch.full_like(processed[0, :9], 0.4))


def test_4973_fp16_constant_coordinate_produces_nonfinite_value(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_4973_fp16_nan"]
    saved = load_file(_state_files(checkpoint, "preprocessor_config.json")[0])
    assert saved["observation.state.std"][4].item() == 0.0
    state = _sample(checkpoint, "sample_observation.npz", "observation.state")
    assert state.dtype == torch.float16

    processed = _pipeline(checkpoint)({"observation.state": state})["observation.state"]
    assert processed.dtype == torch.float16
    assert not torch.isfinite(processed[0, 4])
    assert torch.isfinite(torch.cat((processed[0, :4], processed[0, 5:]))).all()


def test_5315_oldest_first_stack_uses_wrong_relative_anchor(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["n_obs_steps"] == 3
    assert metadata["observation_order"] == "oldest_first"
    assert metadata["current_index"] == -1
    state = _sample(checkpoint, "sample_observation.npz", "observation.state")
    action = _sample(checkpoint, "sample_action.npz", "action")
    assert state.shape[-2:] == (3, 14)
    torch.testing.assert_close(state[0, :, 0], torch.tensor([0.20, 0.45, 0.70]))
    assert action[0, 0, 0].item() == pytest.approx(0.90)

    preprocessor = _pipeline(checkpoint)
    assert isinstance(preprocessor.steps[0], RelativeActionsProcessorStep)
    assert preprocessor.steps[0].enabled
    relative = preprocessor({"observation.state": state, "action": action})["action"]
    assert relative[0, 0, 0].item() == pytest.approx(0.70)
    current_frame_delta = action[0, 0, 0] - state[0, -1, 0]
    assert current_frame_delta.item() == pytest.approx(0.20)
    assert relative[0, 0, 0].item() != pytest.approx(current_frame_delta.item())


def test_legacy_act_stores_stats_in_model_file_without_processor_config(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["legacy_act_embedded"]
    assert not (checkpoint / "preprocessor_config.json").exists()
    assert not (checkpoint / "postprocessor_config.json").exists()
    model_file = checkpoint / "model.safetensors"
    assert model_file.is_file()
    saved = load_file(model_file)
    assert "unnormalize_outputs.buffer_action.mean" in saved
    assert "unnormalize_outputs.buffer_action.std" in saved
    assert saved["unnormalize_outputs.buffer_action.mean"].shape == (14,)
    assert saved["unnormalize_outputs.buffer_action.std"].shape == (14,)
