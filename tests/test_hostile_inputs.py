"""Regression tests for hostile inputs: corrupt files, huge tensors, local legacy
checkpoints, exit codes and Hub retries."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
import torch
from safetensors.torch import load_file, save_file

import lerobot_flinch.hub_scanner as hub_scanner
import lerobot_flinch.preflight as preflight
from lerobot_flinch.cli import main, run_checks
from lerobot_flinch.models import MAX_EVIDENCE_ITEMS, DiagnosticStatus
from lerobot_flinch.preflight import run_preflight
from tests.fixtures.make_bug_fixtures import make_bug_fixtures

PRE = "policy_preprocessor_step_0_normalizer_processor.safetensors"


@pytest.fixture
def fixtures(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path / "fixtures")


def _ids(findings) -> set[str]:
    return {finding.check_id for finding in findings}


# --- One corrupt file degrades its own findings, not the whole report ----------------------


@pytest.mark.parametrize("corruption", ["truncated", "random"])
def test_corrupt_stats_file_keeps_every_other_check(fixtures: dict[str, Path], corruption: str) -> None:
    checkpoint = fixtures["clean_act_modern"]
    clean_ids = _ids(run_checks(checkpoint).findings)
    path = checkpoint / PRE
    path.write_bytes(path.read_bytes()[:50] if corruption == "truncated" else bytes(range(256)) * 16)

    report = run_checks(checkpoint)

    assert "RUNNER_ERROR" not in _ids(report.findings)
    corrupt = [f for f in report.findings if "Corrupt or truncated safetensors file" in f.summary]
    assert corrupt and all(f.status is DiagnosticStatus.FAIL for f in corrupt)
    # Checks that don't need the broken file still report.
    assert {"TEMPORAL_ANCHOR_MISMATCH", "PHYSICAL_SEMANTICS", "SCHEMA_GENERATION"} <= _ids(report.findings)
    assert len(report.findings) >= len(clean_ids)
    assert report.exit_code() == 1


def test_one_failing_stage_does_not_erase_the_others(
    fixtures: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = fixtures["bug_5289_pathological_gain"]

    def broken(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(preflight, "_compare_action_stats", broken)
    report = run_preflight(checkpoint)

    (incomplete,) = [f for f in report.findings if f.check_id == "CHECK_INCOMPLETE"]
    assert incomplete.status is DiagnosticStatus.UNKNOWN
    assert incomplete.details["tool_error"] is True
    assert any(
        f.check_id == "PATHOLOGICAL_STD_GAIN" and f.status is DiagnosticStatus.FAIL for f in report.findings
    )
    assert report.exit_code() == 1  # A proven finding outranks a tool error.


# --- Bounded evidence ----------------------------------------------------------------------


def test_huge_stats_tensor_produces_a_small_report(fixtures: dict[str, Path], tmp_path: Path) -> None:
    checkpoint = fixtures["clean_act_modern"]
    state = load_file(checkpoint / PRE)
    key = "observation.images.front"
    state[f"{key}.mean"] = torch.full((3, 480, 640), 0.5)
    state[f"{key}.std"] = torch.full((3, 480, 640), 1e-7)
    save_file(state, checkpoint / PRE)
    config_path = checkpoint / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    step = config["steps"][0]["config"]
    step["features"][key] = {"type": "VISUAL", "shape": [3, 480, 640]}
    step["norm_map"]["VISUAL"] = "MEAN_STD"
    config_path.write_text(json.dumps(config))
    report_path = tmp_path / "report.json"

    assert main(["--policy", str(checkpoint), "--report", str(report_path)]) == 1

    assert report_path.stat().st_size < 200_000
    document = json.loads(report_path.read_text())
    (gain,) = [
        f
        for f in document["findings"]
        if f["check_id"] == "PATHOLOGICAL_STD_GAIN" and f["provenance"]["tensor_key"] == f"{key}.std"
    ]
    assert "(921600 in total)" in gain["summary"]
    assert gain["details"]["violating_count"] == 921600
    assert gain["details"]["maximum_gain"] == pytest.approx(1 / (1e-7 + 1e-8), rel=1e-3)
    assert len(gain["provenance"]["violating_indices"]) == MAX_EVIDENCE_ITEMS
    assert gain["provenance"]["violating_index_count"] == 921600

    def longest_list(value) -> int:
        if isinstance(value, dict):
            return max((longest_list(item) for item in value.values()), default=0)
        if isinstance(value, list):
            return max([len(value), *(longest_list(item) for item in value)])
        return 0

    assert longest_list(document) <= MAX_EVIDENCE_ITEMS


# --- Local legacy checkpoints ----------------------------------------------------------------


def _legacy_checkpoint(root: Path, std: torch.Tensor) -> Path:
    root.mkdir(parents=True)
    save_file(
        {
            "model.weight": torch.zeros(2 * 1024 * 1024),  # 8 MB of "weights" that must not be read.
            "normalize_inputs.buffer_observation_state.mean": torch.zeros(7),
            "normalize_inputs.buffer_observation_state.std": std,
            "unnormalize_outputs.buffer_action.mean": torch.zeros(7),
            "unnormalize_outputs.buffer_action.std": torch.ones(7),
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "vqbet",
                "input_shapes": {"observation.state": [7]},
                "output_shapes": {"action": [7]},
                "input_normalization_modes": {"observation.state": "mean_std"},
                "output_normalization_modes": {"action": "mean_std"},
            }
        )
    )
    return root


def test_local_legacy_checkpoint_is_screened_like_the_hub_path(tmp_path: Path) -> None:
    std = torch.full((7,), 0.5)
    std[4] = 6.6630414e-06
    checkpoint = _legacy_checkpoint(tmp_path / "xarm_like", std)

    report = run_preflight(checkpoint)

    (gain,) = [
        f
        for f in report.findings
        if f.check_id == "PATHOLOGICAL_STD_GAIN" and f.status is DiagnosticStatus.FAIL
    ]
    assert gain.provenance.violating_indices == [4]
    assert gain.provenance.tensor_key == "normalize_inputs.buffer_observation_state.std"
    assert "149856" in gain.summary


def test_local_legacy_reader_reads_buffers_not_weights(tmp_path: Path) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "legacy", torch.full((7,), 0.5))
    values = preflight._local_legacy_values(checkpoint)
    assert values is not None
    _, state, metadata, _ = values
    assert "model.weight" not in state
    assert metadata["bytes_read"] < (checkpoint / "model.safetensors").stat().st_size / 100


def test_healthy_local_legacy_checkpoint_passes_the_gain_check(tmp_path: Path) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "legacy", torch.full((7,), 0.5))
    report = run_preflight(checkpoint)
    gains = [f for f in report.findings if f.check_id == "PATHOLOGICAL_STD_GAIN"]
    assert gains and all(f.status is DiagnosticStatus.PASS for f in gains)


def test_modern_weights_file_is_not_treated_as_legacy(fixtures: dict[str, Path]) -> None:
    checkpoint = fixtures["clean_act_modern"]
    before = [f.to_dict() for f in run_preflight(checkpoint).findings]
    save_file({"model.weight": torch.zeros(1024)}, checkpoint / "model.safetensors")
    after = [f.to_dict() for f in run_preflight(checkpoint).findings]
    assert not [f for f in after if f["check_id"] == "CHECK_INCOMPLETE"]
    assert [
        f["check_id"] for f in after if f["check_id"] in {"PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"}
    ] == [f["check_id"] for f in before if f["check_id"] in {"PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"}]


# --- Exit codes ------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc"])
def test_bad_std_threshold_is_a_usage_error(
    fixtures: dict[str, Path], capsys: pytest.CaptureFixture[str], value: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--policy", str(fixtures["clean_act_modern"]), "--std-threshold", value])
    assert exit_info.value.code == 2
    assert "--std-threshold" in capsys.readouterr().err


def test_exit_codes_distinguish_pass_finding_and_tool_error(
    fixtures: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import lerobot_flinch.cli as cli_module

    assert main(["--policy", str(fixtures["clean_act_modern"])]) == 0
    assert main(["--policy", str(fixtures["bug_5289_pathological_gain"])]) == 1

    def crash(*args, **kwargs):
        raise RuntimeError("injected crash")

    monkeypatch.setattr(cli_module, "run_preflight", crash)
    assert main(["--policy", str(fixtures["clean_act_modern"])]) == 3


# --- Hub retries -----------------------------------------------------------------------------


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(hub_scanner, "_sleep", delays.append)
    return delays


def _http_error(code: int, retry_after: str | None = None) -> HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return HTTPError("https://huggingface.co/x", code, "error", headers, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "transient", [URLError("connection reset"), TimeoutError("timed out"), _http_error(503), _http_error(429)]
)
def test_transient_hub_errors_are_retried(no_sleep: list[float], transient: Exception) -> None:
    calls = iter([transient, transient, "ok"])

    def flaky() -> str:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    assert hub_scanner.with_retries(flaky) == "ok"
    assert len(no_sleep) == 2
    assert all(0 <= delay <= hub_scanner.RETRY_MAX_SECONDS for delay in no_sleep)


@pytest.mark.parametrize("permanent", [_http_error(404), _http_error(401), ValueError("bad data")])
def test_permanent_errors_are_not_retried(no_sleep: list[float], permanent: Exception) -> None:
    calls = []

    def fails() -> None:
        calls.append(1)
        raise permanent

    with pytest.raises(type(permanent)):
        hub_scanner.with_retries(fails)
    assert len(calls) == 1 and not no_sleep


def test_retry_after_header_is_honoured_and_attempts_are_bounded(no_sleep: list[float]) -> None:
    calls = []

    def always_busy() -> None:
        calls.append(1)
        raise _http_error(429, retry_after="2")

    with pytest.raises(HTTPError):
        hub_scanner.with_retries(always_busy)
    assert len(calls) == hub_scanner.RETRY_ATTEMPTS
    assert no_sleep == [2.0] * (hub_scanner.RETRY_ATTEMPTS - 1)


def test_offline_mode_is_never_retried(no_sleep: list[float]) -> None:
    class OfflineModeIsEnabled(ConnectionError):
        pass

    with pytest.raises(OfflineModeIsEnabled):
        hub_scanner.with_retries(lambda: (_ for _ in ()).throw(OfflineModeIsEnabled("offline")))
    assert not no_sleep
