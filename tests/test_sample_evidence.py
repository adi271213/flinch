"""A recorded rollout passed as --sample shows how close real data came to a near-zero std."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file

from lerobot_flinch.cli import run_checks
from lerobot_flinch.models import DiagnosticStatus
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    return make_bug_fixtures(tmp_path / "fixtures")["bug_5289_pathological_gain"]


def _rollout(checkpoint: Path, path: Path, coordinate_9_noise: float) -> Path:
    stats = load_file(next(checkpoint.glob("*preprocessor*normalizer*.safetensors")))
    mean = stats["observation.state.mean"].numpy()
    rng = np.random.default_rng(0)
    frames = np.repeat(mean[None], 50, axis=0) + rng.normal(0, 0.1, (50, mean.size))
    frames[:, 9] = mean[9] + rng.normal(0, coordinate_9_noise, 50) if coordinate_9_noise else mean[9]
    np.savez(path, **{"observation.state": frames.astype(np.float32)})
    return path


def _gain(checkpoint: Path, sample: Path):
    (finding,) = [
        f
        for f in run_checks(checkpoint, sample=sample).findings
        if f.check_id == "PATHOLOGICAL_STD_GAIN" and f.status is DiagnosticStatus.FAIL
    ]
    return finding


def test_constant_coordinate_in_rollout_is_reported_but_still_fails(checkpoint: Path, tmp_path: Path) -> None:
    finding = _gain(checkpoint, _rollout(checkpoint, tmp_path / "steady.npz", 0.0))
    assert "Across the 50 frames of your --sample they never move." in finding.summary
    assert finding.details["sample_normalized_spread"] == [0.0]


def test_moving_coordinate_in_rollout_shows_the_real_blow_up(checkpoint: Path, tmp_path: Path) -> None:
    finding = _gain(checkpoint, _rollout(checkpoint, tmp_path / "noisy.npz", 1e-3))
    assert "they move by up to" in finding.summary
    (normalized,) = finding.details["sample_normalized_spread"]
    assert normalized > 1000  # ~1e-3 of sensor noise against std 1.48e-6.


def test_single_frame_sample_makes_no_claim(checkpoint: Path) -> None:
    finding = _gain(checkpoint, checkpoint / "sample_observation.npz")
    assert "has one frame" in finding.summary


def test_rollout_trace_reports_the_coordinate_not_flattened_frames(checkpoint: Path, tmp_path: Path) -> None:
    sample = _rollout(checkpoint, tmp_path / "noisy.npz", 1e-3)
    (finding,) = [
        f
        for f in run_checks(checkpoint, sample=sample).findings
        if f.check_id == "EMPIRICAL_GAIN_EXPLOSION" and f.status is DiagnosticStatus.FAIL
    ]
    assert finding.provenance.violating_indices == [9]
    assert finding.details["feature_coordinates"] == [9]
    assert finding.details["frames_total"] == 50
    assert 0 < finding.details["frames_affected"] <= 50
    assert f"coordinate(s) [9] in {finding.details['frames_affected']} of 50 frames" in finding.summary
