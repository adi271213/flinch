"""Checkpoint vs env/dataset checks, each built from a mismatch LeRobot users reported."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_flinch.cli import main
from lerobot_flinch.eval_contract import (
    check_dataset,
    check_env,
    check_processor_steps,
    load_env_config,
    read_dataset_meta,
)
from lerobot_flinch.models import DiagnosticStatus


def _images(*names: str, size: int = 256) -> dict[str, dict[str, object]]:
    return {f"observation.images.{name}": {"type": "VISUAL", "shape": [3, size, size]} for name in names}


SMOLVLA_LIBERO = {  # Shape of HuggingFaceVLA/smolvla_libero's config.json.
    "type": "smolvla",
    "input_features": {**_images("image", "image2"), "observation.state": {"type": "STATE", "shape": [8]}},
    "output_features": {"action": {"type": "ACTION", "shape": [7]}},
}
PI05_BASE = {  # Shape of lerobot/pi05_base's config.json: state/action padded to 32 on purpose.
    "type": "pi05",
    "max_state_dim": 32,
    "max_action_dim": 32,
    "input_features": {
        **_images("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", size=224),
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
}


def _status(findings, check):
    (finding,) = [f for f in findings if f.check_id == check]
    return finding


# --- #3674 / #2954: saved processor missing steps the factory adds ----------------------------


def _xvla_checkpoint(root: Path, steps: list[str]) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"type": "xvla"}))
    (root / "policy_preprocessor.json").write_text(
        json.dumps({"name": "policy_preprocessor", "steps": [{"registry_name": s} for s in steps]})
    )
    return root


def test_missing_factory_steps_are_a_conflict(tmp_path: Path) -> None:
    # The steps lerobot/xvla-libero ships (#3674): no image-to-float or ImageNet normalization.
    shipped = ["rename_observations_processor", "to_batch_processor", "tokenizer_processor"]
    checkpoint = _xvla_checkpoint(tmp_path / "xvla", [*shipped, "xvla_add_domain_id", "normalizer_processor"])
    finding = _status(check_processor_steps(checkpoint, {"type": "xvla"}), "PROCESSOR_STEPS_MISSING")
    assert finding.status is DiagnosticStatus.CONFLICT
    assert finding.details["missing_steps"] == ["xvla_image_to_float", "xvla_imagenet_normalize"]


def test_complete_processor_passes(tmp_path: Path) -> None:
    steps = ["tokenizer_processor", "xvla_image_to_float", "xvla_imagenet_normalize", "xvla_add_domain_id"]
    checkpoint = _xvla_checkpoint(tmp_path / "xvla", steps)
    finding = _status(check_processor_steps(checkpoint, {"type": "xvla"}), "PROCESSOR_STEPS_MISSING")
    assert finding.status is DiagnosticStatus.PASS


def test_steps_check_skips_custom_policies_and_old_checkpoints(tmp_path: Path) -> None:
    custom = _xvla_checkpoint(tmp_path / "custom", ["normalizer_processor"])
    assert check_processor_steps(custom, {"type": "my_plugin_policy"}) == []
    old = tmp_path / "old"
    old.mkdir()
    assert check_processor_steps(old, {"type": "xvla"}) == []


# --- #5185 image size, #5118 / #4245 cameras ----------------------------------------------------


def test_libero_default_size_conflicts_with_256_px_checkpoint() -> None:
    finding = _status(check_env(SMOLVLA_LIBERO, load_env_config("libero", {})), "ENV_IMAGE_SIZE_MISMATCH")
    assert finding.status is DiagnosticStatus.CONFLICT
    assert finding.details["env_size"] == [360, 360]


def test_matching_size_and_cameras_pass() -> None:
    env = load_env_config("libero", {"observation_height": "256", "observation_width": "256"})
    findings = check_env(SMOLVLA_LIBERO, env)
    assert {f.check_id: f.status for f in findings} == {
        "ENV_IMAGE_SIZE_MISMATCH": DiagnosticStatus.PASS,
        "ENV_CAMERA_COUNT_MISMATCH": DiagnosticStatus.PASS,
    }


def test_too_few_env_cameras_is_a_conflict_unless_declared_empty() -> None:
    env = load_env_config("libero", {"camera_name": "agentview_image"})
    assert (
        _status(check_env(SMOLVLA_LIBERO, env), "ENV_CAMERA_COUNT_MISMATCH").status
        is DiagnosticStatus.CONFLICT
    )
    with_placeholder = {**SMOLVLA_LIBERO, "empty_cameras": 1}
    assert (
        _status(check_env(with_placeholder, env), "ENV_CAMERA_COUNT_MISMATCH").status is DiagnosticStatus.PASS
    )


def test_unknown_env_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="no_such_field"):
        load_env_config("libero", {"no_such_field": "1"})


# --- #4517 / #2963 dataset features, #4045 / #4548 task text ----------------------------------


def _info(state: int, action: int, cameras: int) -> dict[str, object]:
    features = {
        f"observation.images.cam{i}": {"dtype": "video", "shape": [3, 256, 256]} for i in range(cameras)
    }
    features["observation.state"] = {"dtype": "float32", "shape": [state]}
    features["action"] = {"dtype": "float32", "shape": [action]}
    return {"features": features}


def test_padded_state_and_action_are_not_a_mismatch() -> None:
    finding = _status(check_dataset(PI05_BASE, _info(8, 7, cameras=3), None), "DATASET_FEATURE_MISMATCH")
    assert finding.status is DiagnosticStatus.PASS  # #2963: 32 is deliberate padding.


def test_camera_shortfall_and_unpadded_shape_mismatch_conflict() -> None:
    finding = _status(check_dataset(PI05_BASE, _info(8, 7, cameras=2), None), "DATASET_FEATURE_MISMATCH")
    assert finding.status is DiagnosticStatus.CONFLICT
    assert finding.details["problems"] == ["the checkpoint expects 3 camera(s), the dataset has 2"]
    finding = _status(check_dataset(SMOLVLA_LIBERO, _info(6, 7, cameras=2), None), "DATASET_FEATURE_MISMATCH")
    assert finding.details["problems"] == [
        "observation.state is [8] in the checkpoint but [6] in the dataset"
    ]


def test_camera_names_are_not_compared() -> None:
    renamed = {
        **SMOLVLA_LIBERO,
        "input_features": {
            **_images("camera1", "camera2"),
            "observation.state": {"type": "STATE", "shape": [8]},
        },
    }
    assert (
        _status(check_dataset(renamed, _info(8, 7, 2), None), "DATASET_FEATURE_MISMATCH").status
        is DiagnosticStatus.PASS
    )


def test_task_identifiers_warn_and_sentences_do_not(tmp_path: Path) -> None:
    ids = check_dataset(SMOLVLA_LIBERO, _info(8, 7, 2), ["CloseFridge", "OpenDrawer"])
    finding = _status(ids, "TASK_TEXT_FORMAT")
    assert finding.status is DiagnosticStatus.PASS and finding.details["advisory"] is True
    sentences = check_dataset(SMOLVLA_LIBERO, _info(8, 7, 2), ["put the bowl on the plate"])
    assert not [f for f in sentences if f.check_id == "TASK_TEXT_FORMAT"]
    unreadable = check_dataset(SMOLVLA_LIBERO, _info(8, 7, 2), None)
    assert _status(unreadable, "TASK_TEXT_FORMAT").status is DiagnosticStatus.UNKNOWN


def _dataset(root: Path, tasks: list[str]) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(_info(8, 7, 2)))
    (root / "meta" / "tasks.jsonl").write_text("".join(json.dumps({"task": t}) + "\n" for t in tasks))
    return root


def test_reads_tasks_jsonl(tmp_path: Path) -> None:
    info, tasks = read_dataset_meta(_dataset(tmp_path / "ds", ["CloseFridge"]))
    assert tasks == ["CloseFridge"] and "action" in info["features"]


def test_reads_tasks_parquet_from_either_column(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for column in ("task", "__index_level_0__"):  # RoboCasa vs LIBERO layouts on the Hub.
        root = tmp_path / column
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}")
        pq.write_table(pa.table({"task_index": [0], column: ["OpenDrawer"]}), root / "meta" / "tasks.parquet")
        assert read_dataset_meta(root)[1] == ["OpenDrawer"]


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_flags_the_libero_size_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(SMOLVLA_LIBERO))
    assert main(["--policy", str(checkpoint), "--env", "libero"]) == 1
    assert "renders 360x360 images" in capsys.readouterr().out
    size = ["--env-option", "observation_height=256", "--env-option", "observation_width=256"]
    main(["--policy", str(checkpoint), "--env", "libero", *size])
    assert "ENV_IMAGE_SIZE_MISMATCH] The libero env renders 256x256" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--env-option", "observation_height=256"], "--env-option needs --env"),
        (["--env", "libero", "--env-option", "oops"], "KEY=VALUE"),
        (["--env", "no_such_env"], "--env no_such_env"),
        (["--dataset", "{tmp}"], "has no meta/info.json"),
    ],
)
def test_cli_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    with pytest.raises(SystemExit) as exit_info:
        main(["--policy", str(checkpoint), *[arg.format(tmp=tmp_path) for arg in extra]])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --- False positives found by scanning all 71 official checkpoints with processor files ------


def _checkpoint(root: Path, config: dict[str, object], steps: list[str]) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    (root / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": s} for s in steps]})
    )
    return root


def _always(policy_type: str) -> set[str]:
    from lerobot_flinch.eval_contract import _factory_steps

    expected = _factory_steps(policy_type)
    assert expected is not None
    return expected[0]


def test_switched_off_relative_actions_are_not_required(tmp_path: Path) -> None:
    # pi05's factory builds RelativeActionsProcessorStep(enabled=config.use_relative_actions).
    assert "relative_actions_processor" not in _always("pi05")
    checkpoint = _checkpoint(tmp_path / "pi05", {"type": "pi05"}, sorted(_always("pi05")))
    assert _status(check_processor_steps(checkpoint, {"type": "pi05"}), "PROCESSOR_STEPS_MISSING").status is (
        DiagnosticStatus.PASS
    )


def test_vla_jepa_image_prep_is_done_by_the_model_when_absent(tmp_path: Path) -> None:
    saved = sorted(_always("vla_jepa") - {"vla_jepa_image_prep"})
    checkpoint = _checkpoint(tmp_path / "jepa", {"type": "vla_jepa"}, saved)
    assert _status(
        check_processor_steps(checkpoint, {"type": "vla_jepa"}), "PROCESSOR_STEPS_MISSING"
    ).status is (DiagnosticStatus.PASS)


@pytest.mark.parametrize(("n_obs_steps", "required"), [(1, False), (2, True)])
def test_g05_history_crop_only_matters_with_several_observation_steps(
    tmp_path: Path, n_obs_steps: int, required: bool
) -> None:
    config = {"type": "g05", "n_obs_steps": n_obs_steps}
    checkpoint = _checkpoint(tmp_path / "g05", config, sorted(_always("g05") - {"g05_action_history_crop"}))
    finding = _status(check_processor_steps(checkpoint, config), "PROCESSOR_STEPS_MISSING")
    assert ("g05_action_history_crop" in finding.details.get("missing_steps", [])) is required
