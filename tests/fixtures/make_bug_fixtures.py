"""Generate small, local checkpoints that reproduce known processor failures.

All modern artifacts are written by LeRobot's own processor serialization. The
legacy artifact deliberately uses the older embedded model-weight key format.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from safetensors.torch import save_file

OBSERVATION_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM = 14


def _features() -> dict[str, PolicyFeature]:
    return {
        OBSERVATION_KEY: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,)),
    }


def _stats(
    *,
    observation_stat_key: str = OBSERVATION_KEY,
    observation_std: torch.Tensor | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    if observation_std is None:
        observation_std = torch.full((STATE_DIM,), 0.5, dtype=torch.float32)
    return {
        observation_stat_key: {
            "mean": torch.zeros(STATE_DIM, dtype=torch.float32),
            "std": observation_std,
        },
        ACTION_KEY: {
            "mean": torch.zeros(STATE_DIM, dtype=torch.float32),
            "std": torch.full((STATE_DIM,), 0.5, dtype=torch.float32),
        },
    }


def _write_normalization_checkpoint(
    directory: Path,
    stats: dict[str, dict[str, torch.Tensor]],
    observation: np.ndarray,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    features = _features()
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    preprocessor = PolicyProcessorPipeline(
        name="policy_preprocessor",
        steps=[NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)],
    )
    postprocessor = PolicyProcessorPipeline(
        name="policy_postprocessor",
        steps=[
            UnnormalizerProcessorStep(
                features={ACTION_KEY: features[ACTION_KEY]},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={ACTION_KEY: stats[ACTION_KEY]},
            )
        ],
    )
    preprocessor.save_pretrained(directory, config_filename="preprocessor_config.json")
    postprocessor.save_pretrained(directory, config_filename="postprocessor_config.json")
    np.savez(directory / "sample_observation.npz", **{OBSERVATION_KEY: observation})
    np.savez(
        directory / "sample_action.npz",
        **{ACTION_KEY: np.full((1, STATE_DIM), 0.40, dtype=np.float32)},
    )
    return directory


def _write_temporal_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Probe widths are declared contracts, never inferred from action dimensions.
    (directory / "config.json").write_text(
        json.dumps(
            {
                "input_features": {OBSERVATION_KEY: {"type": "STATE", "shape": [STATE_DIM]}},
                "output_features": {ACTION_KEY: {"type": "ACTION", "shape": [STATE_DIM]}},
            },
            indent=2,
        )
        + "\n"
    )
    relative_step = RelativeActionsProcessorStep(enabled=True)
    preprocessor = PolicyProcessorPipeline(name="policy_preprocessor", steps=[relative_step])
    postprocessor = PolicyProcessorPipeline(
        name="policy_postprocessor",
        steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)],
    )
    preprocessor.save_pretrained(directory, config_filename="preprocessor_config.json")
    postprocessor.save_pretrained(directory, config_filename="postprocessor_config.json")

    observation = np.zeros((1, 3, STATE_DIM), dtype=np.float32)
    observation[0, :, 0] = [0.20, 0.45, 0.70]
    action = np.zeros((1, 1, STATE_DIM), dtype=np.float32)
    action[0, 0, 0] = 0.90
    np.savez(directory / "sample_observation.npz", **{OBSERVATION_KEY: observation})
    np.savez(directory / "sample_action.npz", **{ACTION_KEY: action})
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "n_obs_steps": 3,
                "observation_order": "oldest_first",
                "current_index": -1,
            },
            indent=2,
        )
        + "\n"
    )
    return directory


def _write_legacy_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    legacy_action_stats = _stats()[ACTION_KEY]
    legacy_step = UnnormalizerProcessorStep(
        features={ACTION_KEY: _features()[ACTION_KEY]},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={ACTION_KEY: legacy_action_stats},
    )
    embedded_state = {
        f"unnormalize_outputs.buffer_{key}": tensor for key, tensor in legacy_step.state_dict().items()
    }
    save_file(embedded_state, directory / "model.safetensors")
    return directory


def make_bug_fixtures(output_root: str | Path) -> dict[str, Path]:
    """Write six local fixture directories: clean, four bugs, and legacy ACT."""
    root = Path(output_root)
    sample_state = np.full((1, STATE_DIM), 0.20, dtype=np.float32)
    fixtures = {
        "clean_act_modern": _write_normalization_checkpoint(
            root / "clean_act_modern", _stats(), sample_state
        ),
        "bug_4415_silent_bypass": _write_normalization_checkpoint(
            root / "bug_4415_silent_bypass",
            _stats(observation_stat_key="wrong_dataset/observation.state"),
            sample_state,
        ),
    }

    low_std = torch.full((STATE_DIM,), 0.5, dtype=torch.float32)
    low_std[9] = 1.48e-6
    fixtures["bug_5289_pathological_gain"] = _write_normalization_checkpoint(
        root / "bug_5289_pathological_gain",
        _stats(observation_std=low_std),
        sample_state,
    )

    fp16_std = torch.full((STATE_DIM,), 0.5, dtype=torch.float32)
    fp16_std[4] = 0.0
    fixtures["bug_4973_fp16_nan"] = _write_normalization_checkpoint(
        root / "bug_4973_fp16_nan",
        _stats(observation_std=fp16_std),
        np.zeros((1, STATE_DIM), dtype=np.float16),
    )
    fixtures["bug_5315_temporal_anchor"] = _write_temporal_checkpoint(root / "bug_5315_temporal_anchor")
    fixtures["legacy_act_embedded"] = _write_legacy_checkpoint(root / "legacy_act_embedded")
    return fixtures
