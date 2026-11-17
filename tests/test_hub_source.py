"""`--policy hf://owner/name[@revision]` audits a bounded, pinned Hub snapshot."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

import lerobot_flinch.hub_scanner as hub_scanner
from lerobot_flinch import cli, hub_source
from lerobot_flinch.cli import main
from lerobot_flinch.hub_source import HubReference, HubSourceError, parse_hub_reference
from lerobot_flinch.models import PreflightReport
from tests.fixtures.make_bug_fixtures import make_bug_fixtures
from tests.test_hub_scanner import _FakeHubApi, _mock_hub_downloads, _repo_files

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def fake_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[_FakeHubApi, list[tuple[str, str, str]]]:
    checkpoints = make_bug_fixtures(tmp_path / "fixtures")
    repositories: dict[str, dict[str, object]] = {
        f"owner/{name}": {"sha": SHA, "source": path, "files": _repo_files(path)}
        for name, path in checkpoints.items()
        if name in {"clean_act_modern", "bug_5289_pathological_gain"}
    }
    api = _FakeHubApi(repositories)
    monkeypatch.setattr(hub_scanner, "HfApi", lambda **kwargs: api)
    downloads = _mock_hub_downloads(monkeypatch, repositories)
    return api, downloads


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("./ckpt", None),
        ("owner/name", None),
        ("hf://owner/name", HubReference("owner/name")),
        ("hf://lerobot/smolvla_base@main", HubReference("lerobot/smolvla_base", "main")),
        ("hf://o-1/n.2_x@" + SHA, HubReference("o-1/n.2_x", SHA)),
    ],
)
def test_parse_hub_reference(value: str, expected: HubReference | None) -> None:
    assert parse_hub_reference(value) == expected


@pytest.mark.parametrize(
    "value", ["hf://owner", "hf://owner/name/extra", "hf://../name", "hf://owner/..", "hf://owner/name@"]
)
def test_invalid_hub_reference_is_rejected(value: str) -> None:
    with pytest.raises(HubSourceError):
        parse_hub_reference(value)


def test_hub_policy_reports_pinned_defect_without_weights(
    fake_hub: tuple[_FakeHubApi, list[tuple[str, str, str]]], tmp_path: Path
) -> None:
    api, downloads = fake_hub
    report_path = tmp_path / "report.json"

    code = main(["--policy", "hf://owner/bug_5289_pathological_gain", "--report", str(report_path)])

    assert code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["checkpoint_dir"] == f"hf://owner/bug_5289_pathological_gain@{SHA}"
    by_id = {finding["check_id"]: finding for finding in report["findings"]}
    assert by_id["HUB_SNAPSHOT"]["status"] == "PASS"
    assert by_id["HUB_SNAPSHOT"]["details"]["revision"] == SHA
    assert by_id["HUB_SNAPSHOT"]["details"]["neural_weights_downloaded"] is False
    assert by_id["PATHOLOGICAL_STD_GAIN"]["status"] == "FAIL"
    downloaded = {filename for _, filename, _ in downloads}
    assert not any(name.endswith(".npz") for name in downloaded)
    assert "preprocessor_config.json" in downloaded
    assert api.model_info_calls[0][1].get("revision") is None


def test_hub_policy_clean_checkpoint_passes_and_snapshot_is_removed(
    fake_hub: tuple[_FakeHubApi, list[tuple[str, str, str]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Path] = []
    original = hub_scanner.download_lightweight_snapshot

    def recording(repo_id: str, snapshot_dir: Path, *args: object, **kwargs: object) -> dict[str, object]:
        created.append(Path(snapshot_dir))
        return original(repo_id, snapshot_dir, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(hub_scanner, "download_lightweight_snapshot", recording)

    assert main(["--policy", "hf://owner/clean_act_modern"]) == 0
    assert created and not created[0].exists()


def test_hub_revision_is_forwarded_and_baseline_accepts_hub_reference(
    fake_hub: tuple[_FakeHubApi, list[tuple[str, str, str]]], capsys: pytest.CaptureFixture[str]
) -> None:
    api, _ = fake_hub

    main(
        [
            "--policy",
            "hf://owner/bug_5289_pathological_gain@v1.0",
            "--baseline",
            "hf://owner/clean_act_modern",
        ]
    )

    revisions = {repo_id: kwargs.get("revision") for repo_id, kwargs in api.model_info_calls}
    assert revisions == {"owner/bug_5289_pathological_gain": "v1.0", "owner/clean_act_modern": None}
    output = capsys.readouterr().out
    assert f"flinch: hf://owner/bug_5289_pathological_gain@{SHA}" in output
    assert "baseline snapshot hf://owner/clean_act_modern" in output


def test_bare_repo_id_gets_actionable_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["--policy", "lerobot/smolvla_base"])
    assert exit_info.value.code == 2
    assert "--policy hf://lerobot/smolvla_base" in capsys.readouterr().err


def test_existing_local_directory_shaped_like_repo_id_is_still_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = make_bug_fixtures(tmp_path / "owner")["clean_act_modern"]
    monkeypatch.chdir(tmp_path)
    assert main(["--policy", f"owner/{checkpoint.name}"]) == 0


def test_unreachable_hub_repository_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class MissingRepoApi:
        def model_info(self, repo_id: str, **kwargs: object) -> object:
            raise OSError("404 Repository Not Found")

    monkeypatch.setattr(hub_scanner, "HfApi", lambda **kwargs: MissingRepoApi())

    assert main(["--policy", "hf://owner/missing"]) == 1
    captured = capsys.readouterr()
    assert "Could not fetch the processor artifacts of hf://owner/missing" in captured.err
    assert "404 Repository Not Found" in captured.err
    assert "Traceback" not in captured.err + captured.out


@pytest.mark.parametrize(
    "access", [{"gated": True}, {"gated": "auto"}, {"gated": "manual"}, {"private": True}]
)
def test_gated_and_private_hub_snapshots_never_reach_diagnostics(
    fake_hub: tuple[_FakeHubApi, list[tuple[str, str, str]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    access: dict[str, object],
) -> None:
    api, _ = fake_hub
    original_info = api.model_info
    original_download = hub_scanner.download_lightweight_snapshot
    snapshots: list[Path] = []

    def model_info(repo_id: str, **kwargs: Any) -> Any:
        info = original_info(repo_id, **kwargs)
        for field, value in access.items():
            setattr(info, field, value)
        return info

    def download(repo_id: str, snapshot_dir: Path, **kwargs: Any) -> dict[str, Any]:
        snapshots.append(snapshot_dir)
        return original_download(repo_id, snapshot_dir, **kwargs)

    def no_diagnostics(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An unsupported Hub snapshot reached diagnostics")

    monkeypatch.setattr(api, "model_info", model_info)
    monkeypatch.setattr(hub_scanner, "download_lightweight_snapshot", download)
    monkeypatch.setattr(cli, "run_checks", no_diagnostics)

    assert main(["--policy", "hf://owner/clean_act_modern"]) == 1

    captured = capsys.readouterr()
    assert "Gated or private repositories are not supported" in captured.err
    assert "Traceback" not in captured.err + captured.out
    assert snapshots and all(not snapshot.parent.exists() for snapshot in snapshots)


@pytest.mark.parametrize("stage", ["setup", "download"])
@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_hub_snapshot_resource_failures_are_clean_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    error_type: type[Exception],
) -> None:
    snapshots: list[Path] = []

    def fail_setup(*args: Any, **kwargs: Any) -> None:
        raise error_type("injected snapshot setup failure")

    def fail_download(repo_id: str, snapshot_dir: Path, **kwargs: Any) -> dict[str, Any]:
        snapshots.append(snapshot_dir)
        snapshot_dir.mkdir()
        (snapshot_dir / "partial.json").write_text("{}", encoding="utf-8")
        raise error_type("injected snapshot download failure")

    def no_diagnostics(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Failed snapshot acquisition reached diagnostics")

    if stage == "setup":
        monkeypatch.setattr(hub_source.tempfile, "TemporaryDirectory", fail_setup)
    monkeypatch.setattr(hub_scanner, "download_lightweight_snapshot", fail_download)
    monkeypatch.setattr(cli, "run_checks", no_diagnostics)

    assert main(["--policy", "hf://owner/checkpoint"]) == 1

    captured = capsys.readouterr()
    assert "Could not fetch the processor artifacts of hf://owner/checkpoint" in captured.err
    assert f"{error_type.__name__}: injected snapshot {stage} failure" in captured.err
    assert "Traceback" not in captured.err + captured.out
    assert all(not snapshot.parent.exists() for snapshot in snapshots)
    assert bool(snapshots) is (stage == "download")


def test_hub_sample_trace_executes_native_numeric_steps_offline(
    fake_hub: tuple[_FakeHubApi, list[tuple[str, str, str]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, downloads = fake_hub
    local = make_bug_fixtures(tmp_path / "samples")["bug_5289_pathological_gain"]
    report_path = tmp_path / "trace.json"
    original_checks = cli.run_checks

    def no_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Offline numeric diagnostics attempted network access")

    def offline_checks(*args: Any, **kwargs: Any) -> PreflightReport:
        assert downloads
        with monkeypatch.context() as runtime:
            runtime.setattr(socket, "socket", no_network)
            runtime.setattr(socket, "create_connection", no_network)
            return original_checks(*args, **kwargs)

    monkeypatch.setattr(cli, "run_checks", offline_checks)

    code = main(
        [
            "--policy",
            "hf://owner/bug_5289_pathological_gain",
            "--sample",
            str(local / "sample_observation.npz"),
            "--report",
            str(report_path),
        ]
    )

    assert code == 1
    findings = json.loads(report_path.read_text(encoding="utf-8"))["findings"]
    explosion = next(finding for finding in findings if finding["check_id"] == "EMPIRICAL_GAIN_EXPLOSION")
    assert explosion["status"] == "FAIL"
    assert explosion["provenance"]["violating_indices"] == [9]
    assert abs(explosion["details"]["output_values"][0]) > 100_000
    assert not any(filename.endswith(".npz") for _, filename, _ in downloads)


@pytest.mark.parametrize("suffix", [".json", ".mmd"])
def test_hub_graph_is_exported_before_snapshot_cleanup(
    fake_hub: tuple[_FakeHubApi, list[tuple[str, str, str]]], tmp_path: Path, suffix: str
) -> None:
    graph_path = tmp_path / ("lineage" + suffix)

    assert main(["--policy", "hf://owner/clean_act_modern", "--graph", str(graph_path)]) == 0

    text = graph_path.read_text(encoding="utf-8")
    assert "observation.state" in text
    if suffix == ".json":
        graph = json.loads(text)
        assert graph["nodes"] and any(edge["type"] == "BINDS_STATS" for edge in graph["edges"])
    else:
        assert "flowchart LR" in text
