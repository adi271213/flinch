"""Terminal and JSON presentation of CLI findings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lerobot_preflight.hub_scanner as hub_scanner
from lerobot_preflight.cli import main, run_checks
from lerobot_preflight.models import DiagnosticStatus
from lerobot_preflight.preflight import run_preflight
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

    assert main(["--policy", "hf://owner/missing@v2"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "Repository hf://owner/missing@v2 was not found" in err
    assert "private or gated" in err
    assert "Invalid username or password" not in err
    assert "Request ID" not in err
