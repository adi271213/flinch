"""Terminal and JSON presentation of CLI findings."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline
from safetensors.torch import load_file

import lerobot_flinch.hub_scanner as hub_scanner
from lerobot_flinch.cli import main, run_checks
from lerobot_flinch.hub_source import _first_line
from lerobot_flinch.models import DiagnosticStatus
from lerobot_flinch.preflight import run_preflight
from lerobot_flinch.privilege_delta import run_privilege_delta
from tests.fixtures.make_bug_fixtures import make_bug_fixtures
from tests.test_preflight import _write_multi_dataset_buffers


def _base_model_checkpoint(tmp_path: Path) -> Path:
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["clean_act_modern"]
    _write_multi_dataset_buffers(checkpoint)
    return checkpoint


def test_unbound_base_model_is_one_actionable_failure(tmp_path: Path) -> None:
    checkpoint = _base_model_checkpoint(tmp_path)
    raw = run_preflight(checkpoint)
    raw_unbound = [f for f in raw.findings if f.check_id == "UNBOUND_BASE_MODEL_STATS"]
    raw_missing = [
        f
        for f in raw.findings
        if f.check_id == "FEATURE_CONTRACT" and f.summary.startswith("Required normalization buffer ")
    ]
    assert len(raw_unbound) == 2 and raw_missing  # The Hub scanner still sees every raw finding.

    report = run_checks(checkpoint)

    failures = [f for f in report.findings if f.status is DiagnosticStatus.FAIL]
    assert [f.check_id for f in failures] == ["UNBOUND_BASE_MODEL_STATS"]
    (finding,) = failures
    assert finding.summary.startswith(
        "Base model: the target dataset's normalization statistics are not bound"
    )
    assert "Fine-tune on your dataset or bind its statistics" in finding.summary
    missing = finding.details["missing_buffers"]
    assert len(missing) == len(raw_missing)
    assert len({(m["processor_file"], m["tensor_key"]) for m in missing}) == len(missing)
    assert f"({len(missing)} required buffers missing across 2 processor files)" in finding.summary
    assert len(finding.details["folded_findings"]) == len(raw_unbound) - 1 + len(raw_missing)
    assert report.has_failures()


def test_unrelated_missing_buffer_is_not_folded(tmp_path: Path) -> None:
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["bug_4415_silent_bypass"]
    raw = run_preflight(checkpoint)
    assert [f.to_dict() for f in run_checks(checkpoint).findings[: len(raw.findings)]] == [
        f.to_dict() for f in raw.findings
    ]


def test_empty_fields_are_not_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--policy", str(_base_model_checkpoint(tmp_path))]) == 1
    output = capsys.readouterr().out
    assert "dims=[]" not in output
    assert "values=[]" not in output


def test_base_model_json_report_keeps_folded_evidence(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    main(["--policy", str(_base_model_checkpoint(tmp_path)), "--report", str(report_path)])
    findings = json.loads(report_path.read_text(encoding="utf-8"))["findings"]
    (unbound,) = [f for f in findings if f["check_id"] == "UNBOUND_BASE_MODEL_STATS"]
    folded = unbound["details"]["folded_findings"]
    assert {f["check_id"] for f in folded} == {"UNBOUND_BASE_MODEL_STATS", "FEATURE_CONTRACT"}


@pytest.mark.parametrize("same_namespaces", [False, True])
def test_unrelated_multistage_buffer_failure_stays_separate(tmp_path: Path, same_namespaces: bool) -> None:
    checkpoint = _base_model_checkpoint(tmp_path)
    original = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="preprocessor_config.json", local_files_only=True
    )
    first = deepcopy(original.steps[0])
    name = original.name
    del original
    namespaces = ("so100-blue", "so100-red", "so100") if same_namespaces else ("factoryA", "factoryB")
    stats = {
        f"{namespace}.buffer.action": {"mean": torch.zeros(14), "std": torch.ones(14)}
        for namespace in namespaces
    }
    stats["observation.environment_state"] = {"mean": torch.zeros(14)}
    unrelated = NormalizerProcessorStep(
        features={"observation.environment_state": PolicyFeature(FeatureType.ENV, (14,))},
        norm_map={FeatureType.ENV: NormalizationMode.MEAN_STD},
        stats=stats,
    )
    PolicyProcessorPipeline(name=name, steps=[first, unrelated]).save_pretrained(
        checkpoint, config_filename="preprocessor_config.json"
    )
    raw = run_preflight(checkpoint)
    expected = [
        finding.to_dict()
        for finding in raw.findings
        if finding.check_id == "FEATURE_CONTRACT"
        and finding.status is DiagnosticStatus.FAIL
        and finding.details.get("feature_key") == "observation.environment_state"
    ]
    assert len(expected) == 1

    report = run_checks(checkpoint)

    actual = [
        finding.to_dict()
        for finding in report.findings
        if finding.check_id == "FEATURE_CONTRACT"
        and finding.status is DiagnosticStatus.FAIL
        and finding.details.get("feature_key") == "observation.environment_state"
    ]
    assert actual == expected
    assert (
        len([finding for finding in report.findings if finding.check_id == "UNBOUND_BASE_MODEL_STATS"]) == 1
    )
    assert report.has_failures()


def test_distinct_stat_paths_with_same_basename_keep_all_missing_buffers(tmp_path: Path) -> None:
    checkpoint = _base_model_checkpoint(tmp_path)
    for role, filename in (("pre", "preprocessor_config.json"), ("post", "postprocessor_config.json")):
        path = checkpoint / filename
        config = json.loads(path.read_text(encoding="utf-8"))
        old = checkpoint / config["steps"][0]["state_file"]
        target = checkpoint / role / "normalization.safetensors"
        target.parent.mkdir()
        old.rename(target)
        config["steps"][0]["state_file"] = target.relative_to(checkpoint).as_posix()
        path.write_text(json.dumps(config), encoding="utf-8")
    raw_missing = [
        finding
        for finding in run_preflight(checkpoint).findings
        if finding.check_id == "FEATURE_CONTRACT"
        and finding.summary.startswith("Required normalization buffer ")
    ]

    report = run_checks(checkpoint)

    (base,) = [finding for finding in report.findings if finding.check_id == "UNBOUND_BASE_MODEL_STATS"]
    assert len(base.details["missing_buffers"]) == len(raw_missing) == 6
    assert {entry["processor_file"] for entry in base.details["missing_buffers"]} == {
        "pre/normalization.safetensors",
        "post/normalization.safetensors",
    }
    assert "6 required buffers missing across 2 processor files" in base.summary
    expected = [finding.to_dict() for finding in raw_missing]
    actual = [
        finding for finding in base.details["folded_findings"] if finding["check_id"] == "FEATURE_CONTRACT"
    ]
    assert sorted(expected, key=str) == sorted(actual, key=str)
    assert report.has_failures()


def test_inline_normalizers_in_same_file_keep_distinct_step_evidence(tmp_path: Path) -> None:
    checkpoint = _base_model_checkpoint(tmp_path)
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    step = config["steps"][0]
    tensors = load_file(checkpoint / step.pop("state_file"))
    stats: dict[str, dict[str, object]] = {}
    for key, value in tensors.items():
        feature, _, stat = key.rpartition(".")
        stats.setdefault(feature, {})[stat] = value.tolist()
    step["config"]["stats"] = stats
    config["steps"].append(deepcopy(step))
    path.write_text(json.dumps(config), encoding="utf-8")
    raw_missing = [
        finding
        for finding in run_preflight(checkpoint).findings
        if finding.check_id == "FEATURE_CONTRACT"
        and finding.summary.startswith("Required normalization buffer ")
    ]

    report = run_checks(checkpoint)

    (base,) = [finding for finding in report.findings if finding.check_id == "UNBOUND_BASE_MODEL_STATS"]
    missing = base.details["missing_buffers"]
    assert len(missing) == len(raw_missing) == 10
    assert (
        len({(entry["processor_file"], entry["step_index"], entry["tensor_key"]) for entry in missing}) == 10
    )
    inline = [entry for entry in missing if entry["processor_file"] == "preprocessor_config.json"]
    assert len(inline) == 8 and {entry["step_index"] for entry in inline} == {0, 1}
    assert "10 required buffers missing across 2 processor files" in base.summary
    assert report.has_failures()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("compare_baseline", [False, True])
def test_folding_preserves_strict_and_required_baseline_gate_findings(
    tmp_path: Path, strict: bool, compare_baseline: bool
) -> None:
    checkpoint = _base_model_checkpoint(tmp_path)
    baseline_findings = run_privilege_delta(checkpoint, checkpoint) if compare_baseline else []

    report = run_checks(checkpoint, strict=strict, baseline=checkpoint if compare_baseline else None)

    required = {
        (finding.check_id, finding.status)
        for finding in baseline_findings
        if finding.details.get("required_coverage")
    }
    retained = {
        (finding.check_id, finding.status)
        for finding in report.findings
        if finding.details.get("required_coverage")
    }
    assert required <= retained
    assert report.strict is strict and report.baseline_requested is compare_baseline
    assert report.has_failures()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (None, "unknown error"),
        ("", "unknown error"),
        (" \n\t ", "unknown error"),
        ("cause\nhelp text", "cause"),
        ("\n cause \n\nhelp text", "cause "),
    ],
)
def test_first_error_line_handles_empty_and_whitespace_messages(message: object, expected: str) -> None:
    assert _first_line(message) == expected


@pytest.mark.parametrize(
    ("error_type", "status"),
    [("RepositoryNotFoundError", 401), ("GatedRepoError", 403), ("RevisionNotFoundError", 404)],
)
def test_unavailable_repository_gets_one_line_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error_type: str, status: int
) -> None:
    response = type("Response", (), {"status_code": status})()
    verbose = (
        f"{status} Client Error. (Request ID: Root=1-abc)\n\nRepository Not Found for url: https://x\n"
        "Invalid username or password."
    )
    error = type(error_type, (OSError,), {})(verbose)
    error.response = response  # type: ignore[attr-defined]

    class UnavailableApi:
        def model_info(self, repo_id: str, **kwargs: object) -> object:
            raise error

    monkeypatch.setattr(hub_scanner, "HfApi", lambda **kwargs: UnavailableApi())

    assert main(["--policy", "hf://owner/missing@v2"]) == 3
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "Repository hf://owner/missing@v2 was not found" in err
    assert "private or gated" in err
    assert "Invalid username or password" not in err
    assert "Request ID" not in err
