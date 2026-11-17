"""`--policy hf://owner/name[@revision]` audits a bounded, pinned Hub snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lerobot_preflight.hub_scanner as hub_scanner
from lerobot_preflight.cli import main
from lerobot_preflight.hub_source import HubReference, HubSourceError, parse_hub_reference
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
    assert f"lerobot-preflight: hf://owner/bug_5289_pathological_gain@{SHA}" in output
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
