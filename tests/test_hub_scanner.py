"""Offline tests for bounded Hub checkpoint metadata scanning."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import lerobot_preflight.hub_scanner as hub_scanner
from lerobot_preflight.hub_scanner import (
    download_lightweight_snapshot,
    fetch_safetensors_header,
    parse_safetensors_header,
    scan_hub,
    write_hub_reports,
)
from lerobot_preflight.preflight import run_preflight
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path / "fixtures")


def _header_only(path: Path) -> bytes:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        assert len(length_bytes) == 8
        length = int.from_bytes(length_bytes, "little")
        header = stream.read(length)
    return length_bytes + header


def _encoded_header(tensors: dict[str, dict], *, metadata: dict[str, str] | None = None) -> bytes:
    data = dict(tensors)
    if metadata:
        data["__metadata__"] = metadata
    header = json.dumps(data, separators=(",", ":")).encode()
    return len(header).to_bytes(8, "little") + header


def test_parse_real_native_and_legacy_safetensors_headers_without_payload(
    checkpoints: dict[str, Path],
) -> None:
    modern = checkpoints["bug_5289_pathological_gain"]
    config = json.loads((modern / "preprocessor_config.json").read_text())
    modern_path = modern / config["steps"][0]["state_file"]
    modern_prefix = _header_only(modern_path)
    assert len(modern_prefix) < modern_path.stat().st_size
    tensors = parse_safetensors_header(modern_prefix, total_size=modern_path.stat().st_size)
    assert {"observation.state.mean", "observation.state.std", "action.mean", "action.std"} <= set(
        tensors
    )
    assert tensors["observation.state.std"]["dtype"] == "F32"
    assert tensors["observation.state.std"]["shape"] == [14]
    assert len(tensors["observation.state.std"]["data_offsets"]) == 2

    legacy_path = checkpoints["legacy_act_embedded"] / "model.safetensors"
    legacy_prefix = _header_only(legacy_path)
    assert len(legacy_prefix) < legacy_path.stat().st_size
    legacy_tensors = parse_safetensors_header(legacy_prefix, total_size=legacy_path.stat().st_size)
    assert set(legacy_tensors) == {
        "unnormalize_outputs.buffer_action.mean",
        "unnormalize_outputs.buffer_action.std",
    }
    assert all(value["shape"] == [14] for value in legacy_tensors.values())


def test_parse_extracts_both_legacy_input_and_output_buffer_prefixes(tmp_path: Path) -> None:
    model_file = tmp_path / "model.safetensors"
    save_file(
        {
            "normalize_inputs.buffer_observation.state.mean": torch.zeros(14),
            "unnormalize_outputs.buffer_action.std": torch.ones(14),
        },
        model_file,
    )
    tensors = parse_safetensors_header(
        _header_only(model_file), total_size=model_file.stat().st_size
    )
    assert set(tensors) == {
        "normalize_inputs.buffer_observation.state.mean",
        "unnormalize_outputs.buffer_action.std",
    }
    assert tensors["normalize_inputs.buffer_observation.state.mean"]["dtype"] == "F32"
    assert tensors["normalize_inputs.buffer_observation.state.mean"]["shape"] == [14]
    assert tensors["unnormalize_outputs.buffer_action.std"]["dtype"] == "F32"
    assert tensors["unnormalize_outputs.buffer_action.std"]["shape"] == [14]


@pytest.mark.parametrize(
    "prefix,total_size",
    [
        (b"short", None),
        ((2**63).to_bytes(8, "little"), 100),
        ((1).to_bytes(8, "little") + b"{", 100),
        ((20).to_bytes(8, "little") + b"{}", 100),
        (
            _encoded_header(
                {"bad": {"dtype": "F32", "shape": [1], "data_offsets": [0, 999999]}}
            ),
            100,
        ),
    ],
)
def test_parse_rejects_truncated_oversized_malformed_or_out_of_bounds_header(
    prefix: bytes, total_size: int | None
) -> None:
    with pytest.raises(ValueError):
        parse_safetensors_header(prefix, total_size=total_size)


class _FakeRangeResponse:
    def __init__(self, body: bytes, status: int, headers: dict[str, str], reads: list[int]) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = headers
        self._reads = reads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        assert 0 <= size <= 65536, "scanner attempted an unbounded HTTP read"
        self._reads.append(size)
        result = self._body[self._offset : self._offset + size]
        self._offset += len(result)
        return result


def _fake_opener(data: bytes, *, ignore_range: bool = False):
    requested_ranges: list[str] = []
    reads: list[int] = []

    def opener(request, *args, **kwargs):
        range_value = request.get_header("Range")
        assert range_value is not None, "scanner omitted the Range header"
        requested_ranges.append(range_value)
        start_text, end_text = range_value.removeprefix("bytes=").split("-", 1)
        start, end = int(start_text), int(end_text)
        if ignore_range:
            return _FakeRangeResponse(data, 200, {"Content-Length": str(len(data))}, reads)
        body = data[start : min(end + 1, len(data))]
        headers = {
            "Content-Range": f"bytes {start}-{start + len(body) - 1}/{len(data)}",
            "Content-Length": str(len(body)),
        }
        return _FakeRangeResponse(body, 206, headers, reads)

    return opener, requested_ranges, reads


def _normalization_header(path: Path, repo_id: str = "org/legacy", revision: str = "a" * 40) -> dict:
    prefix = _header_only(path)
    return {
        "format": "safetensors-header-v1", "source_file": "model.safetensors",
        "repo_id": repo_id, "revision": revision, "header_len": len(prefix) - 8,
        "header_sha256_prefix": hashlib.sha256(prefix).hexdigest()[:12],
        "total_size": path.stat().st_size, "complete": True,
        "tensors": parse_safetensors_header(prefix, path.stat().st_size),
    }


def test_legacy_ranges_fetch_only_state_action_values_and_preserve_saved_bits(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    expected = {
        "normalize_inputs.buffer_observation_state.mean": torch.tensor([0.2, 0.3]),
        "normalize_inputs.buffer_observation_state.std": torch.tensor([1.48e-6, float("nan")]),
        "unnormalize_outputs.buffer_action.mean": torch.tensor(0.5),
        "unnormalize_outputs.buffer_action.std": torch.tensor([0.0, 0.5]),
    }
    save_file({**expected, "model.weight": torch.zeros(256 * 1024),
               "normalize_inputs.buffer_observation_images_top.mean": torch.zeros(3, 1, 1),
               "normalize_inputs.buffer_index.std": torch.zeros(1)}, path)
    header = _normalization_header(path)
    opener, ranges, reads = _fake_opener(path.read_bytes())
    state, metadata = hub_scanner.fetch_legacy_normalization_tensors(
        "org/legacy", header, revision="a" * 40, opener=opener,
    )
    assert set(state) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(state[key], value, equal_nan=True, rtol=0, atol=0)
        descriptor = header["tensors"][key]
        start, end = [8 + header["header_len"] + offset for offset in descriptor["data_offsets"]]
        assert metadata["tensors"][key]["absolute_range"] == [start, end]
        assert metadata["tensors"][key]["payload_sha256"] == hashlib.sha256(path.read_bytes()[start:end]).hexdigest()
    assert metadata["complete"] is True
    assert metadata["bytes_fetched"] == sum(tensor.numel() * tensor.element_size() for tensor in expected.values())
    assert sum(reads) == metadata["bytes_fetched"] < 100
    intervals = [(int(value.split("=")[1].split("-")[0]), int(value.rsplit("-", 1)[1]) + 1) for value in ranges]
    selected = [tuple(item["absolute_range"]) for item in metadata["tensors"].values()]
    assert sum(end - start for start, end in intervals) == sum(end - start for start, end in selected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_legacy_range_reconstruction_supports_saved_float_dtypes(tmp_path: Path, dtype: torch.dtype) -> None:
    path = tmp_path / "model.safetensors"
    expected = torch.tensor([0.0, 0.2, 1.5], dtype=dtype)
    key = "normalize_inputs.buffer_observation_state.std"
    save_file({key: expected}, path)
    opener, _, _ = _fake_opener(path.read_bytes())
    state, metadata = hub_scanner.fetch_legacy_normalization_tensors(
        "org/legacy", _normalization_header(path), revision="a" * 40, opener=opener,
    )
    assert state[key].dtype == dtype
    torch.testing.assert_close(state[key], expected, rtol=0, atol=0)
    assert metadata["complete"] is True


def test_legacy_tensor_ranges_reject_http_200_before_any_body_read(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    save_file({"normalize_inputs.buffer_observation_state.std": torch.tensor([0.5])}, path)
    opener, _, reads = _fake_opener(path.read_bytes(), ignore_range=True)
    state, metadata = hub_scanner.fetch_legacy_normalization_tensors(
        "org/legacy", _normalization_header(path), revision="a" * 40, opener=opener,
    )
    assert state == {} and reads == []
    assert metadata["complete"] is False
    assert metadata["bytes_fetched"] == 0
    assert "HTTP 206" in metadata["errors"][0]["error"]


@pytest.mark.parametrize("failure", ["range", "size", "encoding", "truncated"])
def test_legacy_tensor_ranges_validate_response_and_truncation(tmp_path: Path, failure: str) -> None:
    path = tmp_path / "model.safetensors"
    save_file({"unnormalize_outputs.buffer_action.std": torch.tensor([0.5])}, path)
    source = path.read_bytes()
    reads = []
    def opener(request, **kwargs):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        headers = {"Content-Range": f"bytes {start}-{end}/{len(source)}", "Content-Length": str(end - start + 1)}
        if failure == "range": headers["Content-Range"] = f"bytes 0-3/{len(source)}"
        if failure == "size": headers["Content-Range"] = f"bytes {start}-{end}/{len(source) + 1}"
        if failure == "encoding": headers["Content-Encoding"] = "gzip"
        body = source[start:end + 1]
        if failure == "truncated": body = body[:-1]
        return _FakeRangeResponse(body, 206, headers, reads)
    state, metadata = hub_scanner.fetch_legacy_normalization_tensors(
        "org/legacy", _normalization_header(path), revision="a" * 40, opener=opener,
    )
    assert not state and metadata["errors"]
    if failure != "truncated": assert reads == []
    else: assert metadata["bytes_fetched"] == 3


@pytest.mark.parametrize("mutation", ["shape", "dtype", "offsets", "oversized"])
def test_legacy_descriptor_validation_happens_before_network(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "model.safetensors"
    key = "normalize_inputs.buffer_observation_state.std"
    save_file({key: torch.tensor([0.5])}, path)
    header = _normalization_header(path)
    descriptor = header["tensors"][key]
    if mutation == "shape": descriptor["shape"] = [2]
    if mutation == "dtype": descriptor["dtype"] = "I32"
    if mutation == "offsets": descriptor["data_offsets"] = [header["total_size"], header["total_size"] + 4]
    if mutation == "oversized":
        descriptor["shape"] = [hub_scanner.MAX_LEGACY_TENSOR_BYTES]
        descriptor["data_offsets"] = [0, hub_scanner.MAX_LEGACY_TENSOR_BYTES * 4]
    def deny_network(*args, **kwargs):
        pytest.fail("invalid tensor descriptor reached HTTP")
    state, metadata = hub_scanner.fetch_legacy_normalization_tensors(
        "org/legacy", header, revision="a" * 40, opener=deny_network,
    )
    assert not state and key in metadata["skipped_tensors"]


def test_cached_scan_enriches_legacy_ranges_before_local_numeric_screening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    remote = tmp_path / "remote_model.safetensors"
    std = torch.full((14,), 0.5)
    std[9] = 1.48e-6
    save_file({
        "model.weight": torch.zeros(256 * 1024),
        "normalize_inputs.buffer_observation_state.mean": torch.zeros(14),
        "normalize_inputs.buffer_observation_state.std": std,
        "unnormalize_outputs.buffer_action.mean": torch.zeros(14),
        "unnormalize_outputs.buffer_action.std": torch.ones(14),
    }, remote)
    header = _normalization_header(remote)
    (checkpoint / "model.safetensors.header.json").write_text(json.dumps(header))
    (checkpoint / "config.json").write_text(json.dumps({
        "type": "act", "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        "normalization_mapping": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
    }))
    manifest = {
        "repo_id": "org/legacy", "revision": "a" * 40, "checkpoint_dir": str(checkpoint),
        "snapshot_dir": str(checkpoint), "checkpoint_subdir": ".", "downloaded_files": ["config.json"],
        "errors": [], "critical_download_errors": False, "skipped_files": [],
        "artifact_bytes_downloaded": 100, "header_bytes_fetched": len(_header_only(remote)),
        "model_weights_skipped": True, "full_model_weights_downloaded": False,
    }
    source_report = tmp_path / "old_report.json"
    source_report.write_text(json.dumps({"generated_at_utc": "2026-11-14T00:00:00Z", "repos": [manifest]}))
    opener, ranges, _ = _fake_opener(remote.read_bytes())
    monkeypatch.setattr(hub_scanner, "urlopen", opener)
    def deny_client(*args, **kwargs):
        pytest.fail("cached enrichment attempted a Hub client/full download")
    monkeypatch.setattr(hub_scanner, "HfApi", deny_client)
    monkeypatch.setattr(hub_scanner, "hf_hub_download", deny_client)
    report = scan_hub(cached_reports=[source_report], limit=100, fetch_legacy_stats=True)
    assert report["selection"]["network_used"] is True
    assert report["summary"]["normalization_bytes_fetched"] == 4 * 14 * 4
    assert report["summary"]["cached_legacy_range_evidence"]["tensors"] == 4
    assert report["summary"]["cached_legacy_range_evidence"]["payload_bytes"] == 4 * 14 * 4
    assert report["summary"]["header_bytes_fetched"] == report["summary"]["artifact_bytes_downloaded"] == 0
    assert report["summary"]["issues"]["PATHOLOGICAL_STD_GAIN"] == ["org/legacy"]
    finding = next(f for f in report["repos"][0]["findings"]
                   if f["check_id"] == "PATHOLOGICAL_STD_GAIN" and f["status"] == "FAIL")
    assert finding["provenance"]["violating_indices"] == [9]
    assert finding["provenance"]["tensor_key"] == "normalize_inputs.buffer_observation_state.std"
    assert ranges and not (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "legacy_normalization.safetensors").is_file()
    cached_path, _ = write_hub_reports(report, tmp_path / "reports")
    monkeypatch.setattr(hub_scanner, "urlopen", deny_client)
    repeat = scan_hub(cached_reports=[cached_path], limit=100, fetch_legacy_stats=True)
    assert repeat["summary"]["cached_legacy_range_evidence"]["new_payload_bytes_this_run"] == 0
    assert repeat["summary"]["cached_legacy_range_evidence"]["payload_bytes"] == 4 * 14 * 4
    assert repeat["summary"]["normalization_bytes_fetched"] == 0
    assert repeat["summary"]["issues"]["PATHOLOGICAL_STD_GAIN"] == ["org/legacy"]


def test_fetch_header_uses_bounded_range_reads_for_large_weight_payload() -> None:
    payload_size = 6 * 1024 * 1024
    prefix = _encoded_header(
        {"model.weight": {"dtype": "F32", "shape": [payload_size // 4], "data_offsets": [0, payload_size]}}
    )
    remote_file = prefix + b"\0" * payload_size
    opener, ranges, reads = _fake_opener(remote_file)
    result = fetch_safetensors_header(
        "org/heavy", filename="model.safetensors", revision="sha123",
        total_size=len(remote_file), opener=opener,
    )
    assert result["format"] == "safetensors-header-v1"
    assert result["repo_id"] == "org/heavy"
    assert result["revision"] == "sha123"
    assert result["source_file"] == "model.safetensors"
    assert result["tensors"]["model.weight"]["shape"] == [payload_size // 4]
    assert result["header_sha256_prefix"] == hashlib.sha256(prefix).hexdigest()[:12]
    assert result["bytes_fetched"] < 2 * 65536
    assert sum(reads) < 2 * 65536
    assert ranges[0] == "bytes=0-65535"


@pytest.mark.parametrize("ignore_range", [False, True])
def test_fetch_extended_header_handles_range_and_range_ignored_servers(
    ignore_range: bool,
) -> None:
    payload_size = 6 * 1024 * 1024
    prefix = _encoded_header(
        {"model.weight": {"dtype": "F32", "shape": [payload_size // 4], "data_offsets": [0, payload_size]}},
        metadata={"long_note": "x" * 70000},
    )
    remote_file = prefix + b"\0" * payload_size
    opener, ranges, reads = _fake_opener(remote_file, ignore_range=ignore_range)
    result = fetch_safetensors_header(
        "org/heavy", filename="model.safetensors", revision="sha123",
        total_size=len(remote_file), opener=opener,
    )
    assert result["tensors"]["model.weight"]["dtype"] == "F32"
    assert len(ranges) >= 2
    assert ranges[0] == "bytes=0-65535"
    assert sum(reads) < 200000
    assert result["bytes_fetched"] < 200000


def test_fetch_rejects_oversized_header_before_downloading_payload() -> None:
    declared_header_size = 4 * 1024 * 1024 + 1
    remote_file = declared_header_size.to_bytes(8, "little") + b"x" * 65536
    opener, ranges, reads = _fake_opener(remote_file)
    with pytest.raises(ValueError):
        fetch_safetensors_header(
            "org/oversized", filename="model.safetensors", revision="sha123",
            total_size=len(remote_file), opener=opener,
        )
    assert ranges == ["bytes=0-65535"]
    assert sum(reads) <= 65536


class _FakeHubApi:
    def __init__(self, repositories: dict[str, dict[str, object]]) -> None:
        self.repositories = repositories
        self.model_info_calls: list[tuple[str, dict]] = []
        self.list_repo_files_calls: list[tuple[str, str | None]] = []
        self.list_models_calls: list[dict] = []

    def model_info(self, repo_id: str, **kwargs):
        self.model_info_calls.append((repo_id, kwargs))
        repo = self.repositories[repo_id]
        siblings = [
            SimpleNamespace(rfilename=filename, size=size)
            for filename, size in repo["files"].items()
        ]
        return SimpleNamespace(sha=repo["sha"], siblings=siblings, private=False)

    def list_repo_files(self, repo_id: str, *, revision: str | None = None, **kwargs):
        self.list_repo_files_calls.append((repo_id, revision))
        return list(self.repositories[repo_id]["files"])

    def list_models(self, **kwargs):
        self.list_models_calls.append(kwargs)
        for repo_id in list(self.repositories)[: kwargs.get("limit") or len(self.repositories)]:
            yield SimpleNamespace(id=repo_id)


def _repo_files(source: Path, *, large_model: bool = False) -> dict[str, int]:
    files = {
        path.name: path.stat().st_size
        for path in source.iterdir()
        if path.is_file()
    }
    if large_model:
        files["model.safetensors"] = 8 * 1024 * 1024
    return files


def _mock_hub_downloads(
    monkeypatch: pytest.MonkeyPatch,
    repositories: dict[str, dict[str, object]],
    *,
    legacy_header_source: Path | None = None,
) -> list[tuple[str, str, str]]:
    downloads: list[tuple[str, str, str]] = []

    def fake_hf_hub_download(*args, **kwargs):
        repo_id = kwargs.get("repo_id", args[0] if args else None)
        filename = kwargs.get("filename", args[1] if len(args) > 1 else None)
        revision = kwargs.get("revision")
        assert repo_id in repositories
        assert filename is not None
        assert revision == repositories[repo_id]["sha"]
        assert kwargs.get("token") is False
        assert filename != "model.safetensors", "model weights must never be downloaded"
        source = Path(repositories[repo_id]["source"]) / filename
        assert source.is_file(), f"unexpected Hub download: {filename}"
        local_dir = Path(kwargs["local_dir"])
        destination = local_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        downloads.append((repo_id, filename, revision))
        return str(destination)

    def fake_fetch(repo_id: str, filename: str = "model.safetensors", revision: str = "main", **kwargs):
        assert legacy_header_source is not None
        assert filename == "model.safetensors"
        assert revision == repositories[repo_id]["sha"]
        header = _header_only(legacy_header_source)
        return {
            "format": "safetensors-header-v1",
            "source_file": filename,
            "tensors": parse_safetensors_header(
                header, total_size=legacy_header_source.stat().st_size
            ),
            "repo_id": repo_id,
            "revision": revision,
            "header_sha256_prefix": hashlib.sha256(header).hexdigest()[:12],
            "header_len": len(header) - 8,
            "complete": True,
            "total_size": kwargs.get("total_size"),
            "bytes_fetched": len(header),
        }

    monkeypatch.setattr(hub_scanner, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(hub_scanner, "fetch_safetensors_header", fake_fetch)
    def fake_tensor_range(request, **kwargs):
        assert legacy_header_source is not None
        repo_id = next(repo_id for repo_id in repositories if f"/{repo_id}/resolve/" in request.full_url)
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        body = legacy_header_source.read_bytes()[start:end + 1]
        return _FakeRangeResponse(body, 206, {
            "Content-Range": f"bytes {start}-{end}/{repositories[repo_id]['files']['model.safetensors']}",
            "Content-Length": str(end - start + 1),
        }, [])
    monkeypatch.setattr(hub_scanner, "urlopen", fake_tensor_range)
    return downloads


def test_lightweight_snapshot_skips_large_model_and_indexes_legacy_header(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = checkpoints["legacy_act_embedded"]
    repo_id = "org/legacy"
    repositories = {
        repo_id: {
            "source": source,
            "sha": "a" * 40,
            "files": _repo_files(source, large_model=True),
        }
    }
    api = _FakeHubApi(repositories)
    downloads = _mock_hub_downloads(
        monkeypatch, repositories, legacy_header_source=source / "model.safetensors"
    )
    snapshot = tmp_path / "legacy_snapshot"
    manifest = download_lightweight_snapshot(repo_id, snapshot, api=api)
    assert manifest["repo_id"] == repo_id
    assert manifest["revision"] == "a" * 40
    assert Path(manifest["snapshot_dir"]) == snapshot
    assert manifest["model_weights_skipped"] is True
    assert not (snapshot / "model.safetensors").exists()
    assert (snapshot / "model.safetensors.header.json").is_file()
    assert manifest["header_metadata"] == {
        "model.safetensors": "model.safetensors.header.json"
    }
    assert manifest["header_bytes_fetched"] == len(_header_only(source / "model.safetensors"))
    assert any(item["file_path"] == "model.safetensors" for item in manifest["skipped_files"])
    assert not downloads
    assert api.model_info_calls[0][1].get("files_metadata") is True
    assert api.list_repo_files_calls == [(repo_id, "a" * 40)]

    report = run_preflight(snapshot)
    legacy = [
        finding for finding in report.findings
        if finding.check_id == "SCHEMA_GENERATION_MISMATCH" and finding.status.value == "FAIL"
    ]
    assert legacy
    assert legacy[0].details["evidence_source"] == "safetensors_header_metadata"
    assert legacy[0].details["tensor_values_available"] is False
    assert set(legacy[0].details["legacy_buffer_keys"]) == {
        "unnormalize_outputs.buffer_action.mean",
        "unnormalize_outputs.buffer_action.std",
    }


def test_small_model_safetensors_is_also_header_only(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = checkpoints["legacy_act_embedded"]
    repo_id = "org/small_legacy"
    repositories = {
        repo_id: {
            "source": source,
            "sha": "9" * 40,
            "files": _repo_files(source),
        }
    }
    api = _FakeHubApi(repositories)
    downloads = _mock_hub_downloads(
        monkeypatch, repositories, legacy_header_source=source / "model.safetensors"
    )
    snapshot = tmp_path / "small_legacy_snapshot"
    manifest = download_lightweight_snapshot(repo_id, snapshot, api=api)
    assert manifest["model_weights_skipped"] is True
    assert manifest["full_model_weights_downloaded"] is False
    assert not (snapshot / "model.safetensors").exists()
    assert "model.safetensors" not in manifest["downloaded_files"]
    assert manifest["header_metadata"]
    assert not any(filename == "model.safetensors" for _, filename, _ in downloads)

    report = run_preflight(snapshot)
    legacy = [
        finding for finding in report.findings
        if finding.check_id == "SCHEMA_GENERATION_MISMATCH" and finding.status.value == "FAIL"
    ]
    assert len(legacy) == 1
    assert legacy[0].details["evidence_source"] == "safetensors_header_metadata"
    assert legacy[0].details["tensor_values_available"] is False


def test_lightweight_snapshot_downloads_only_small_processor_artifacts(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = checkpoints["bug_5289_pathological_gain"]
    repo_id = "org/lowstd"
    repositories = {
        repo_id: {"source": source, "sha": "b" * 40, "files": _repo_files(source)}
    }
    api = _FakeHubApi(repositories)
    downloads = _mock_hub_downloads(monkeypatch, repositories)
    snapshot = tmp_path / "modern_snapshot"
    manifest = download_lightweight_snapshot(repo_id, snapshot, api=api)
    downloaded_names = {Path(name).name for name in manifest["downloaded_files"]}
    assert {"preprocessor_config.json", "postprocessor_config.json"} <= downloaded_names
    assert any(name.endswith("normalizer_processor.safetensors") for name in downloaded_names)
    assert any(name.endswith("unnormalizer_processor.safetensors") for name in downloaded_names)
    assert not any(name.endswith(".npz") for name in downloaded_names)
    assert all(filename not in {"sample_observation.npz", "sample_action.npz"} for _, filename, _ in downloads)
    assert any(item["file_path"].endswith(".npz") for item in manifest["skipped_files"])
    assert not manifest["errors"]

    report = run_preflight(snapshot)
    assert any(
        finding.check_id == "PATHOLOGICAL_STD_GAIN" and finding.status.value == "FAIL"
        for finding in report.findings
    )


def test_nested_processor_pair_wins_over_root_policy_config(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "nested_source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"type": "act"}))
    shutil.copytree(checkpoints["clean_act_modern"], source / "processor")
    files = {
        path.relative_to(source).as_posix(): path.stat().st_size
        for path in source.rglob("*")
        if path.is_file()
    }
    repo_id = "org/nested"
    repositories = {
        repo_id: {"source": source, "sha": "1" * 40, "files": files}
    }
    api = _FakeHubApi(repositories)
    _mock_hub_downloads(monkeypatch, repositories)
    snapshot = tmp_path / "nested_snapshot"
    manifest = download_lightweight_snapshot(repo_id, snapshot, api=api)
    assert Path(manifest["checkpoint_dir"]) == snapshot / "processor"
    assert (snapshot / "processor" / "preprocessor_config.json").is_file()
    assert (snapshot / "processor" / "postprocessor_config.json").is_file()
    assert not run_preflight(snapshot / "processor").has_failures()


def test_legacy_weight_index_sidecar_is_schema_evidence_without_tensor_values(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "index_only"
    snapshot.mkdir()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8 * 1024 * 1024},
                "weight_map": {
                    "unnormalize_outputs.buffer_action.mean": "model-00001-of-00001.safetensors",
                    "unnormalize_outputs.buffer_action.std": "model-00001-of-00001.safetensors",
                },
            }
        )
    )
    (snapshot / "download_manifest.json").write_text(
        json.dumps({"model_weights_skipped": True})
    )
    report = run_preflight(snapshot)
    legacy = [
        finding for finding in report.findings
        if finding.check_id == "SCHEMA_GENERATION_MISMATCH" and finding.status.value == "FAIL"
    ]
    assert len(legacy) == 1
    assert legacy[0].details["evidence_source"] == "safetensors_weight_index"
    assert legacy[0].details["digest_scope"] == "local_weight_index_file"
    assert legacy[0].details["tensor_values_available"] is False
    assert legacy[0].details["tensor_shape_dtype_available"] is False
    assert Path(legacy[0].provenance.file_path) == snapshot / "model.safetensors.index.json"
    assert legacy[0].provenance.sha256_prefix == hashlib.sha256(
        (snapshot / "model.safetensors.index.json").read_bytes()
    ).hexdigest()[:12]
    assert not (snapshot / "model-00001-of-00001.safetensors").exists()


@pytest.mark.parametrize("sidecar", ["missing", "invalid"])
def test_skipped_model_with_no_usable_header_is_unknown_not_schema_failure(
    tmp_path: Path, sidecar: str
) -> None:
    snapshot = tmp_path / sidecar
    snapshot.mkdir()
    (snapshot / "download_manifest.json").write_text(
        json.dumps({"model_weights_skipped": True})
    )
    if sidecar == "invalid":
        (snapshot / "model.safetensors.header.json").write_text("{broken")
    report = run_preflight(snapshot)
    assert any(
        finding.check_id == "SCHEMA_GENERATION" and finding.status.value == "UNKNOWN"
        for finding in report.findings
    )
    assert not any(
        finding.check_id == "SCHEMA_GENERATION" and finding.status.value == "FAIL"
        for finding in report.findings
    )
    assert not any(
        finding.check_id == "FEATURE_CONTRACT" and finding.status.value == "FAIL"
        for finding in report.findings
    )


def test_oversized_processor_config_is_skipped_and_does_not_create_false_issue(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = checkpoints["clean_act_modern"]
    repo_id = "org/incomplete"
    files = _repo_files(source)
    files["preprocessor_config.json"] = 6 * 1024 * 1024
    repositories = {
        repo_id: {"source": source, "sha": "e" * 40, "files": files}
    }
    api = _FakeHubApi(repositories)
    downloads = _mock_hub_downloads(monkeypatch, repositories)
    original_download = hub_scanner.hf_hub_download

    def forbid_large_config(*args, **kwargs):
        filename = kwargs.get("filename", args[1] if len(args) > 1 else None)
        assert filename != "preprocessor_config.json", "oversized config was fully downloaded"
        return original_download(*args, **kwargs)

    monkeypatch.setattr(hub_scanner, "hf_hub_download", forbid_large_config)
    manifest = download_lightweight_snapshot(repo_id, tmp_path / "incomplete_snapshot", api=api)
    assert manifest["critical_download_errors"] is True
    assert not (tmp_path / "incomplete_snapshot" / "preprocessor_config.json").exists()
    assert any(
        item["file_path"] == "preprocessor_config.json"
        and item["size"] == 6 * 1024 * 1024
        for item in manifest["skipped_files"]
    )
    assert all(filename != "preprocessor_config.json" for _, filename, _ in downloads)

    report = scan_hub(repo_ids=[repo_id], snapshot_dir=tmp_path / "incomplete_scan", api=api)
    assert report["summary"]["errors"] == 1
    assert report["summary"]["unknown"] == 1
    assert report["repos"][0]["classification"] == "UNKNOWN"
    assert report["summary"]["issues"]["FEATURE_CONTRACT"] == []
    assert any(
        finding["check_id"] == "HUB_SNAPSHOT" and finding["status"] == "UNKNOWN"
        for finding in report["repos"][0]["findings"]
    )


def test_scan_three_mock_repositories_aggregates_only_confirmed_issues(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "org/clean": checkpoints["clean_act_modern"],
        "org/lowstd": checkpoints["bug_5289_pathological_gain"],
        "org/legacy": checkpoints["legacy_act_embedded"],
    }
    repositories = {
        repo_id: {
            "source": source,
            "sha": letter * 40,
            "files": _repo_files(source, large_model=(repo_id == "org/legacy")),
        }
        for (repo_id, source), letter in zip(sources.items(), "abc", strict=True)
    }
    api = _FakeHubApi(repositories)
    downloads = _mock_hub_downloads(
        monkeypatch,
        repositories,
        legacy_header_source=checkpoints["legacy_act_embedded"] / "model.safetensors",
    )
    progress: list[str] = []
    report = scan_hub(
        repo_ids=list(sources), limit=3, snapshot_dir=tmp_path / "scan_snapshots",
        std_threshold=1e-4, api=api, progress=progress.append,
    )
    summary = report["summary"]
    assert summary["total_repos"] == 3
    assert summary["modern"] == 2
    assert summary["legacy"] == 1
    assert summary["mixed"] == 0
    assert summary["unknown"] == 0
    assert summary["non_policy"] == 0
    assert summary["gated_or_private"] == 0
    assert summary["buckets"]["MODERN_POLICY"] == 2
    assert summary["buckets"]["LEGACY_POLICY"] == 1
    assert summary["errors"] == 0
    assert summary["issues"]["PATHOLOGICAL_STD_GAIN"] == ["org/lowstd"]
    assert summary["issues"]["SCHEMA_GENERATION_MISMATCH"] == ["org/legacy"]
    assert "PHYSICAL_SEMANTICS" not in summary["issues"]

    repos = {entry["repo_id"]: entry for entry in report["repos"]}
    assert set(repos) == set(sources)
    assert repos["org/clean"]["classification"] == "MODERN_POLICY"
    assert repos["org/lowstd"]["classification"] == "MODERN_POLICY"
    assert repos["org/legacy"]["classification"] == "LEGACY_POLICY"
    assert any(
        finding["check_id"] == "PATHOLOGICAL_STD_GAIN" and finding["status"] == "FAIL"
        for finding in repos["org/lowstd"]["findings"]
    )
    assert any(
        finding["check_id"] == "SCHEMA_GENERATION_MISMATCH" and finding["status"] == "FAIL"
        for finding in repos["org/legacy"]["findings"]
    )
    assert progress
    assert all(filename != "model.safetensors" for _, filename, _ in downloads)
    assert len({entry["snapshot_dir"] for entry in report["repos"]}) == 3


def test_modern_processors_plus_legacy_header_are_classified_mixed(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "org/mixed"
    source = checkpoints["clean_act_modern"]
    repositories = {
        repo_id: {
            "source": source,
            "sha": "f" * 40,
            "files": _repo_files(source, large_model=True),
        }
    }
    api = _FakeHubApi(repositories)
    _mock_hub_downloads(
        monkeypatch, repositories,
        legacy_header_source=checkpoints["legacy_act_embedded"] / "model.safetensors",
    )
    report = scan_hub(repo_ids=[repo_id], snapshot_dir=tmp_path / "mixed_scan", api=api)
    assert report["summary"]["mixed"] == 1
    assert report["summary"]["modern"] == 0
    assert report["summary"]["legacy"] == 0
    assert report["summary"]["errors"] == 0
    assert report["repos"][0]["classification"] == "MIXED_POLICY"


def test_asset_only_robot_repo_is_non_policy_from_file_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset_source"
    (source / "robots").mkdir(parents=True)
    (source / "meshes").mkdir()
    (source / "README.md").write_text("robot assets\n")
    (source / "robots" / "arm.urdf").write_text("<robot name='arm'/>\n")
    (source / "meshes" / "gripper.stl").write_text("solid gripper\nendsolid\n")
    repo_id = "org/robot-assets"
    repositories = {
        repo_id: {
            "source": source,
            "sha": "a" * 40,
            "files": {
                path.relative_to(source).as_posix(): path.stat().st_size
                for path in source.rglob("*") if path.is_file()
            },
        }
    }
    api = _FakeHubApi(repositories)
    downloads = _mock_hub_downloads(monkeypatch, repositories)
    report = scan_hub(repo_ids=[repo_id], snapshot_dir=tmp_path / "asset_scan", api=api)
    assert report["summary"]["non_policy"] == 1
    assert report["summary"]["modern"] == 0
    assert report["summary"]["legacy"] == 0
    assert report["repos"][0]["classification"] == "NON_POLICY_REPO"
    assert not any(
        finding["status"] in {"FAIL", "CONFLICT"}
        for finding in report["repos"][0]["findings"]
    )
    assert not downloads
    manifest = json.loads(
        Path(report["repos"][0]["snapshot_dir"]).joinpath("download_manifest.json").read_text()
    )
    assert {"robots/arm.urdf", "meshes/gripper.stl"} <= set(manifest["repo_files"])


def test_reward_output_config_is_non_policy_even_with_a_model_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "reward_source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "type": "robometer",
        "input_features": {"observation.images.top": {"type": "VISUAL", "shape": [3, 224, 224]}},
        "output_features": {
            "progress": {"type": "REWARD", "shape": [1]},
            "success": {"type": "REWARD", "shape": [1]},
        },
    }))
    repo_id = "org/reward-model"
    repositories = {
        repo_id: {"source": source, "sha": "b" * 40, "files": _repo_files(source)}
    }
    api = _FakeHubApi(repositories)
    _mock_hub_downloads(monkeypatch, repositories)
    report = scan_hub(repo_ids=[repo_id], snapshot_dir=tmp_path / "reward_scan", api=api)
    assert report["summary"]["non_policy"] == 1
    assert report["repos"][0]["classification"] == "NON_POLICY_REPO"
    assert not any(
        finding["check_id"] == "SCHEMA_GENERATION" and finding["status"] == "FAIL"
        for finding in report["repos"][0]["findings"]
    )


def test_documentation_only_repository_is_non_policy_from_complete_file_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documentation_source"
    source.mkdir()
    for name in (".gitattributes", "README.md", "LICENSE"):
        (source / name).write_text("repository documentation\n")
    repo_id = "org/ordinary-demo"
    repositories = {repo_id: {"source": source, "sha": "a" * 40, "files": _repo_files(source)}}
    downloads = _mock_hub_downloads(monkeypatch, repositories)
    report = scan_hub(repo_ids=[repo_id], snapshot_dir=tmp_path / "documentation_scan",
                      api=_FakeHubApi(repositories))
    repo = report["repos"][0]
    assert repo["classification"] == "NON_POLICY_REPO"
    assert report["summary"]["non_policy"] == 1 and report["summary"]["unknown"] == 0
    assert repo["classification_evidence"]["reason"] == "documentation_only_without_policy_artifacts"
    assert set(repo["classification_evidence"]["documentation_files"]) == {".gitattributes", "README.md", "LICENSE"}
    assert not any(finding["status"] in {"FAIL", "CONFLICT"} for finding in repo["findings"])
    assert not downloads
    incomplete = {**repo, "repo_files_complete": False}
    assert hub_scanner._repository_classification(incomplete) == (None, {})


def test_reward_output_label_does_not_hide_local_processor_pair(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = checkpoints["bug_5289_pathological_gain"]
    (source / "config.json").write_text(json.dumps({
        "type": "robometer",
        "output_features": {"progress": {"type": "REWARD", "shape": [1]}},
    }))
    repo_id = "org/reward-label-with-policy"
    repositories = {
        repo_id: {"source": source, "sha": "d" * 40, "files": _repo_files(source)}
    }
    _mock_hub_downloads(monkeypatch, repositories)
    report = scan_hub(
        repo_ids=[repo_id], snapshot_dir=tmp_path / "reward_label_scan",
        api=_FakeHubApi(repositories),
    )
    assert report["repos"][0]["classification"] == "MODERN_POLICY"
    assert report["summary"]["non_policy"] == 0
    assert report["summary"]["issues"]["PATHOLOGICAL_STD_GAIN"] == [repo_id]


def test_optional_model_header_denial_does_not_hide_local_processor_pair(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = checkpoints["bug_5289_pathological_gain"]
    repo_id = "org/optional-header-denied"
    repositories = {
        repo_id: {"source": source, "sha": "e" * 40,
                  "files": _repo_files(source, large_model=True)}
    }
    api = _FakeHubApi(repositories)
    original_model_info = api.model_info

    def gated_model_info(*args, **kwargs):
        info = original_model_info(*args, **kwargs)
        info.gated = "auto"
        return info

    monkeypatch.setattr(api, "model_info", gated_model_info)
    _mock_hub_downloads(monkeypatch, repositories)

    def denied_header(*args, **kwargs):
        raise PermissionError("401 Client Error: Unauthorized for optional model header")

    monkeypatch.setattr(hub_scanner, "fetch_safetensors_header", denied_header)
    report = scan_hub(
        repo_ids=[repo_id], snapshot_dir=tmp_path / "header_denied_scan", api=api,
    )
    audited = report["repos"][0]
    assert audited["classification"] == "MODERN_POLICY"
    assert report["summary"]["gated_or_private"] == 0
    assert report["summary"]["issues"]["PATHOLOGICAL_STD_GAIN"] == [repo_id]
    warnings = [finding for finding in audited["findings"] if finding["check_id"] == "HUB_SNAPSHOT"]
    assert len(warnings) == 1 and warnings[0]["status"] == "UNKNOWN"
    assert warnings[0]["details"]["repo_access"]["gated"] == "auto"
    assert warnings[0]["details"]["errors"][0]["file_path"] == "model.safetensors"
    assert not any(finding["check_id"] == "REPO_CLASSIFICATION" for finding in audited["findings"])


def test_access_denial_is_gated_bucket_not_checkpoint_failure(tmp_path: Path) -> None:
    class DeniedApi:
        def model_info(self, repo_id: str, **kwargs):
            raise PermissionError("401 Client Error: Unauthorized for url")

        def list_repo_files(self, *args, **kwargs):
            pytest.fail("access-denied repo should not list files")

    report = scan_hub(
        repo_ids=["org/gated"], snapshot_dir=tmp_path / "gated_scan", api=DeniedApi()
    )
    assert report["summary"]["gated_or_private"] == 1
    assert report["summary"]["errors"] == 1
    assert report["repos"][0]["classification"] == "GATED_OR_PRIVATE"
    assert not any(
        finding["check_id"] == "SCHEMA_GENERATION" and finding["status"] == "FAIL"
        for finding in report["repos"][0]["findings"]
    )


def test_cached_report_rescan_uses_no_hub_api_or_downloads(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_id = "org/cached-lowstd"
    source = checkpoints["bug_5289_pathological_gain"]
    repositories = {
        repo_id: {"source": source, "sha": "c" * 40, "files": _repo_files(source)}
    }
    api = _FakeHubApi(repositories)
    _mock_hub_downloads(monkeypatch, repositories)
    first = scan_hub(repo_ids=[repo_id], snapshot_dir=tmp_path / "first_scan", api=api)
    first_json, _ = write_hub_reports(first, tmp_path / "first_report")

    def forbid_network(*args, **kwargs):
        pytest.fail("cached rescan constructed a Hub client or opened the network")

    monkeypatch.setattr(hub_scanner, "HfApi", forbid_network)
    monkeypatch.setattr(hub_scanner, "hf_hub_download", forbid_network)
    monkeypatch.setattr(hub_scanner, "fetch_safetensors_header", forbid_network)
    second = scan_hub(cached_reports=[first_json])
    assert second["summary"]["total_repos"] == 1
    assert second["summary"]["modern"] == 1
    assert second["repos"][0]["classification"] == "MODERN_POLICY"
    assert second["summary"]["issues"]["PATHOLOGICAL_STD_GAIN"] == [repo_id]


def test_scan_without_explicit_repos_uses_default_top_thirty_limit(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = checkpoints["clean_act_modern"]
    repositories = {
        "org/clean": {"source": source, "sha": "d" * 40, "files": _repo_files(source)}
    }
    api = _FakeHubApi(repositories)
    _mock_hub_downloads(monkeypatch, repositories)
    report = scan_hub(snapshot_dir=tmp_path / "default_scan", api=api)
    assert api.list_models_calls
    assert api.list_models_calls[0]["limit"] == 30
    assert api.list_models_calls[0]["filter"] == "lerobot"
    assert report["summary"]["total_repos"] == 1
    assert report["repos"][0]["repo_id"] == "org/clean"


def test_targeted_discovery_balances_100_tagged_candidates_and_filters_asset_listings() -> None:
    models = [SimpleNamespace(id=f"org/{family}_{index}", tags=["lerobot"], private=False,
                              siblings=[SimpleNamespace(rfilename="config.json")])
              for family in hub_scanner.POLICY_FAMILIES for index in range(25)]
    seeds = [SimpleNamespace(id=repo_id, tags=["lerobot"], private=False, siblings=[])
             for repo_id in hub_scanner._TARGET_SEEDS]
    models += seeds + [SimpleNamespace(id="org/act_mesh", tags=["lerobot"], private=False,
                                     siblings=[SimpleNamespace(rfilename="robot.urdf")])]
    calls = []

    class DiscoveryApi:
        def list_models(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["filter"] == "lerobot"
            matches = [model for model in models if kwargs.get("search", "") in model.id]
            return iter(matches[:kwargs["limit"]])

    selected, evidence = hub_scanner.select_targeted_repositories(DiscoveryApi(), 100)
    assert len(selected) == len(set(selected)) == 100
    assert selected[:2] == list(hub_scanner._TARGET_SEEDS)
    assert "org/act_mesh" not in selected
    assert evidence["filtered_asset_repositories"] == ["org/act_mesh"]
    assert all(count >= 15 for count in evidence["family_hint_counts"].values())
    assert len(calls) == 9 and all(call["limit"] <= 400 for call in calls)
    assert evidence["verified_seed_repositories"] == list(hub_scanner._TARGET_SEEDS)


def test_scan_accepts_100_repositories_and_rejects_larger_limit(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = checkpoints["clean_act_modern"]
    repo_id = "org/clean"
    repositories = {repo_id: {"source": source, "sha": "a" * 40, "files": _repo_files(source)}}
    _mock_hub_downloads(monkeypatch, repositories)
    report = scan_hub(repo_ids=[repo_id], limit=100, snapshot_dir=tmp_path / "limit100",
                      api=_FakeHubApi(repositories))
    assert report["selection"]["limit"] == 100
    assert report["summary"]["total_repos"] == 1
    with pytest.raises(ValueError, match="100"):
        scan_hub(limit=101)


def test_processor_reference_cannot_download_a_small_model_shard(
    checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = checkpoints["clean_act_modern"]
    config_path = source / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    original_state = source / config["steps"][0]["state_file"]
    shard = "model-00001-of-00001.safetensors"
    shutil.copyfile(original_state, source / shard)
    config["steps"][0]["state_file"] = shard
    config_path.write_text(json.dumps(config))
    repo_id = "org/referenced-shard"
    repositories = {repo_id: {"source": source, "sha": "a" * 40, "files": _repo_files(source)}}
    downloads = _mock_hub_downloads(monkeypatch, repositories)
    manifest = download_lightweight_snapshot(repo_id, tmp_path / "reference_snapshot",
                                            api=_FakeHubApi(repositories))
    assert manifest["critical_download_errors"] is True
    assert not any(filename == shard for _, filename, _ in downloads)
    assert any(item["reason"] == "processor_references_neural_weights" for item in manifest["skipped_files"])


def test_write_hub_reports_exports_strict_json_and_repository_links(tmp_path: Path) -> None:
    report = {
        "report_version": 1,
        "generated_at_utc": "2026-11-13T00:00:00+00:00",
        "selection": {"filter": "lerobot", "limit": 30},
        "summary": {
            "total_repos": 1, "modern": 1, "legacy": 0, "mixed": 0,
            "unknown": 0, "non_policy": 0, "gated_or_private": 0, "errors": 0,
            "buckets": {
                "MODERN_POLICY": 1, "LEGACY_POLICY": 0, "MIXED_POLICY": 0,
                "NON_POLICY_REPO": 0, "GATED_OR_PRIVATE": 0, "UNKNOWN": 0,
            },
            "issues": {"PATHOLOGICAL_STD_GAIN": ["org/lowstd"]},
            "artifact_bytes_downloaded": 4096, "header_bytes_fetched": 0,
        },
        "repos": [
            {
                "repo_id": "org/lowstd", "revision": "a" * 40,
                "classification": "MODERN_POLICY", "errors": [],
                "findings": [{
                    "check_id": "PATHOLOGICAL_STD_GAIN", "status": "FAIL",
                    "summary": "Small active state standard deviation.",
                    "details": {"step_index": 5, "saved_values": [1.48e-6]},
                    "provenance": {"file_path": "/snapshot/policy_preprocessor_step_5.safetensors",
                                   "tensor_key": "observation.state.std", "sha256_prefix": "123456abcdef",
                                   "violating_indices": [9]},
                }, {"check_id": "FP16_NON_FINITE_RISK", "status": "UNKNOWN"}],
            }
        ],
    }
    json_path, markdown_path = write_hub_reports(report, tmp_path / "reports")
    assert json_path.name == "hub_audit_report.json"
    assert markdown_path.name == "hub_audit_report.md"
    assert json_path.is_file() and markdown_path.is_file()
    parsed = json.loads(json_path.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert parsed == hub_scanner._portable_report(report, report_dir=tmp_path / "reports")
    markdown = markdown_path.read_text()
    assert "PATHOLOGICAL_STD_GAIN" in markdown
    assert "[org/lowstd](https://huggingface.co/org/lowstd)" in markdown
    assert "[org/lowstd](https://huggingface.co/org/lowstd/tree/" in markdown
    assert "No policy inference, simulator rollout, or motor command runs." in markdown
    assert "Exact failure and conflict evidence" in markdown
    assert "policy_preprocessor_step_5.safetensors` / 5" in markdown
    assert "observation.state.std" in markdown
    assert "123456abcdef" in markdown and "[9]" in markdown and "1.48e-06" in markdown
    assert "`MODERN_POLICY` | 1" in markdown
    assert "Repositories with UNKNOWN numerical buffer screening: **1**" in markdown
    named_json, named_md = write_hub_reports(report, tmp_path / "reports", "hub_100_report")
    assert named_json.name == "hub_100_report.json" and named_md.name == "hub_100_report.md"
    with pytest.raises(ValueError, match="filename stem"):
        write_hub_reports(report, tmp_path / "reports", "../unsafe")


@pytest.mark.parametrize("dtype,raw,expected", [
    ("F16", bytes.fromhex("003c00c0"), [1.0, -2.0]),
    ("F32", bytes.fromhex("0000803f000000c0"), [1.0, -2.0]),
    ("F64", bytes.fromhex("000000000000f03f00000000000000c0"), [1.0, -2.0]),
    ("BF16", bytes.fromhex("803f00c0"), [1.0, -2.0]),
])
def test_tensor_slice_decode_uses_explicit_little_endian(dtype, raw, expected, monkeypatch):
    import numpy as np

    observed = []
    native_frombuffer = np.frombuffer

    def inspect_endian(data, *, dtype):
        observed.append(dtype.str)
        assert dtype.str.startswith("<")
        return native_frombuffer(data, dtype=dtype)

    monkeypatch.setattr(np, "frombuffer", inspect_endian)
    monkeypatch.setattr(torch, "frombuffer", lambda *args, **kwargs: pytest.fail("host-native decoding"))
    decoded = hub_scanner._decode_tensor_slice(raw, dtype, [2])
    assert decoded.tolist() == expected
    assert observed


def test_portable_cached_report_preserves_pins_coverage_and_can_rescan(
    checkpoints, tmp_path, monkeypatch,
):
    project = tmp_path / "portable_project"
    source = project / ".hub_audit_snapshots" / "org-lowstd"
    source.parent.mkdir(parents=True)
    shutil.copytree(checkpoints["bug_5289_pathological_gain"], source)
    monkeypatch.setattr(hub_scanner, "__file__", str(project / "src/lerobot_preflight/hub_scanner.py"))
    manifest = {"repo_id": "org/lowstd", "revision": "e" * 40,
                "snapshot_dir": str(source), "checkpoint_dir": str(source), "errors": [],
                "downloaded_files": _repo_files(source), "critical_download_errors": False}
    result = hub_scanner._audit_snapshot(manifest, 1e-4)
    report = hub_scanner._report([result], {"mode": "cached", "network_used": False,
        "recorded_error": "missing /Users/example/.cache/artifact", "original_discovery": {
            "strategy": "tagged_family_round_robin_then_tagged_fallback", "candidate_count": 724,
            "family_hint_counts": {"act": 1}, "queries": [{"filter": "lerobot"}],
        }})
    # Repeat payload under a second pinned repo: the narrative must reference shared evidence.
    import copy
    duplicate = copy.deepcopy(result)
    duplicate["repo_id"] = "org/lowstd-clone"
    report = hub_scanner._report([result, duplicate], report["selection"])
    json_path, markdown_path = write_hub_reports(report, project / "reports")
    exported, markdown = json_path.read_text(), markdown_path.read_text()
    assert "/Users/" not in exported + markdown
    assert str(project) not in exported + markdown
    assert str(source) == report["repos"][0]["checkpoint_dir"], "Export mutated active local manifest"
    parsed = json.loads(exported)
    assert parsed["repos"][0]["checkpoint_dir"] == ".hub_audit_snapshots/org-lowstd"
    assert parsed["repos"][0]["checkpoint_id"] == "org/lowstd@" + "e" * 40
    assert "UNKNOWN numeric coverage" in markdown and "UNKNOWN temporal coverage" in markdown
    assert "Pinned revisions (full commits in JSON) | 2/2" in markdown
    assert "not a standalone ISO safety-compliance guarantee" in markdown
    assert "## Shared stat payloads" in markdown and "shared_payload_evidence" in markdown
    assert "Candidate pool: 724" in markdown

    def no_network(*args, **kwargs):
        pytest.fail("portable cached scan attempted network access")

    monkeypatch.setattr(hub_scanner, "urlopen", no_network)
    monkeypatch.setattr(hub_scanner, "HfApi", no_network)
    monkeypatch.setattr(hub_scanner, "hf_hub_download", no_network)
    rescanned = scan_hub(cached_reports=[json_path])
    assert rescanned["summary"]["issues"]["PATHOLOGICAL_STD_GAIN"] == ["org/lowstd", "org/lowstd-clone"]
    assert rescanned["selection"]["original_discovery"]["candidate_count"] == 724


@pytest.mark.parametrize("fixture,classification", [
    ("bug_5289_pathological_gain", "MODERN_POLICY"),
    ("legacy_act_embedded", "LEGACY_POLICY"),
])
def test_external_home_snapshot_report_can_move_and_rescan_without_path_leaks(
    checkpoints, tmp_path, monkeypatch, fixture, classification,
):
    # This home-shaped path triggers the same redaction as Windows' native
    # C:/Users/runneradmin/AppData/Local/Temp even when the test host is POSIX.
    home = tmp_path / "Users" / "runneradmin" / "AppData" / "Local" / "Temp" / "audit-bundle"
    snapshot = home / "snapshot"
    checkpoint = snapshot / "pretrained_model"
    shutil.copytree(checkpoints[fixture], checkpoint)
    repo_id = "org/external-home"
    manifest = {"repo_id": repo_id, "revision": "e" * 40, "snapshot_dir": str(snapshot),
                "checkpoint_dir": str(checkpoint), "errors": [],
                "downloaded_files": _repo_files(checkpoint), "critical_download_errors": False}
    result = hub_scanner._audit_snapshot(manifest, 1e-4)
    assert result["classification"] == classification
    report = hub_scanner._report([result], {"mode": "cached", "network_used": False})
    path, markdown = write_hub_reports(report, home / "reports")
    exported = json.loads(path.read_text(encoding="utf-8"))
    location = exported["repos"][0]
    assert location["snapshot_dir"] == "../snapshot"
    assert location["checkpoint_dir"] == "../snapshot/pretrained_model"
    assert location["snapshot_path_base"] == location["checkpoint_path_base"] == "report_directory"
    text = path.read_text(encoding="utf-8") + markdown.read_text(encoding="utf-8")
    assert str(home) not in text and "/Users/runneradmin/" not in text and "\\Users\\runneradmin\\" not in text
    assert report["repos"][0]["checkpoint_dir"] == str(checkpoint), "Export mutated the live manifest"

    def forbid_network(*args, **kwargs):
        pytest.fail("portable cached rescan attempted network access")

    for name in ("HfApi", "hf_hub_download", "urlopen"):
        monkeypatch.setattr(hub_scanner, name, forbid_network)
    # The relative layout, rather than the old username or current working
    # directory, must determine the locations after moving the report bundle.
    moved = tmp_path / "moved-audit-bundle"
    shutil.move(home, moved)
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    rescanned = scan_hub(cached_reports=[moved / "reports" / path.name])
    assert rescanned["repos"][0]["classification"] == classification
    assert Path(rescanned["repos"][0]["snapshot_dir"]) == moved / "snapshot"
    assert Path(rescanned["repos"][0]["checkpoint_dir"]) == moved / "snapshot" / "pretrained_model"
    if fixture == "bug_5289_pathological_gain":
        assert rescanned["summary"]["issues"]["PATHOLOGICAL_STD_GAIN"] == [repo_id]
        failure = next(f for f in rescanned["repos"][0]["findings"]
                       if f["check_id"] == "PATHOLOGICAL_STD_GAIN" and f["status"] == "FAIL")
        assert failure["provenance"]["violating_indices"] == [9]


def test_windows_external_snapshot_locator_is_relative_to_the_report_directory():
    project = PureWindowsPath("D:/a/checkout/preflight")
    home = PureWindowsPath("C:/Users/runneradmin/AppData/Local/Temp/audit")
    location, base = hub_scanner._snapshot_report_location(home / "snapshot", project, home / "reports")
    assert location == "../snapshot" and base == "report_directory"
    assert "Users" not in location and "runneradmin" not in location
    assert hub_scanner._snapshot_report_location(project / ".hub_audit_snapshots/org-lowstd", project,
                                               home / "reports") == (".hub_audit_snapshots/org-lowstd", "project")
    with pytest.raises(ValueError, match="same filesystem drive"):
        hub_scanner._snapshot_report_location(home / "snapshot", project, PureWindowsPath("E:/reports"))


@pytest.mark.parametrize("checkpoint", [
    r"C:\Work Tree\preflight\.hub_audit_snapshots\org-lowstd",
    "C:/Work Tree/preflight/.hub_audit_snapshots/org-lowstd",
    r"c:\work tree\PREFLIGHT\.hub_audit_snapshots\org-lowstd",
])
def test_portable_report_exports_windows_paths_without_machine_prefix(checkpoint, monkeypatch):
    source = PureWindowsPath("C:/Work Tree/preflight/src/lerobot_preflight/hub_scanner.py")
    monkeypatch.setattr(hub_scanner, "Path", lambda _: SimpleNamespace(resolve=lambda: source))
    original = {"repos": [{"repo_id": "org/lowstd", "revision": "a" * 40,
                            "checkpoint_dir": checkpoint}],
                "project_root": "C:/Work Tree/preflight",
                "recorded_error": r"missing C:\Users\example\.cache\artifact",
                "unrelated_pattern": r"\d+\s"}
    result = hub_scanner._portable_report(original)
    assert result["repos"][0]["checkpoint_dir"] == ".hub_audit_snapshots/org-lowstd"
    assert result["project_root"] == "."
    assert "Users" not in result["recorded_error"]
    assert result["unrelated_pattern"] == original["unrelated_pattern"]
    assert original["repos"][0]["checkpoint_dir"] == checkpoint
    # A neighboring project sharing a string prefix must not be truncated.
    neighboring = r"C:\Work Tree\preflight-copy\stats.safetensors"
    assert hub_scanner._portable_report({"path": neighboring})["path"] == neighboring


def test_unicode_numeric_report_is_utf8_under_a_windows_ansi_default(checkpoints, tmp_path, monkeypatch):
    source = checkpoints["bug_5289_pathological_gain"]
    result = hub_scanner._audit_snapshot({
        "repo_id": "jhseon-holiday/vqbet_xarm_holi_test", "revision": "e" * 40,
        "snapshot_dir": str(source), "checkpoint_dir": str(source), "errors": [],
        "downloaded_files": _repo_files(source), "critical_download_errors": False,
    }, 1e-4)
    gain = next(finding for finding in result["findings"] if finding["check_id"] == "PATHOLOGICAL_STD_GAIN")
    gain["summary"] += " Approximation ≈ and perturbation → stay readable."
    report = hub_scanner._report([result], {"mode": "cached"})
    native_open = Path.open
    requested_encodings = []

    def ansi_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode:
            if "w" in mode:
                requested_encodings.append(encoding)
            if encoding is None or encoding == "locale":
                encoding = "cp1252"
        return native_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", ansi_open)
    json_path, markdown_path = write_hub_reports(report, tmp_path / "reports")
    exported = json.loads(json_path.read_bytes().decode("utf-8"))
    assert "Approximation ≈ and perturbation →" in next(
        finding["summary"] for finding in exported["repos"][0]["findings"]
        if finding["check_id"] == "PATHOLOGICAL_STD_GAIN")
    assert "gain ≈" in markdown_path.read_bytes().decode("utf-8")
    assert requested_encodings == ["utf-8", "utf-8"]


def test_round2_reports_separate_legacy_migration_and_conditional_fp16(tmp_path):
    def result(repo_id, findings):
        return {"repo_id": repo_id, "revision": "a" * 40, "classification": "LEGACY_POLICY",
                "checkpoint_dir": str(tmp_path / repo_id.split("/")[-1]), "errors": [],
                "findings": findings}

    schema = {"check_id": "SCHEMA_GENERATION_MISMATCH", "status": "FAIL", "severity": "CRITICAL",
              "summary": "Legacy embedded stats require migration.", "details": {}, "provenance": None}
    conditional = {"check_id": "FP16_NON_FINITE_RISK", "status": "FAIL", "severity": "WARNING",
                   "summary": "Conditional half-buffer risk.", "details": {
                       "fp16_execution_conditional": True, "normalization_implementation": "legacy Normalize"},
                   "provenance": None}
    modern = {**conditional, "severity": "CRITICAL", "details": {
        "normalization_implementation": "modern NormalizerProcessorStep"}}
    repos = [result("org/legacy", [schema, conditional]), result("org/legacy-only", [schema]),
             {**result("org/modern", [modern]), "classification": "MODERN_POLICY"}]
    report = hub_scanner._report(repos, {"mode": "cached", "network_used": False})
    groups = report["summary"]["finding_categories"]
    assert groups["artifact_contract"]["issues"] == {"FP16_NON_FINITE_RISK": ["org/modern"]}
    assert groups["version_compatibility"]["repository_count"] == 2
    assert groups["conditional_precision"]["issues"] == {"FP16_NON_FINITE_RISK": ["org/legacy"]}
    # Report framing retains the raw finding statuses used by existing CLI gates.
    assert report["summary"]["issues"]["FP16_NON_FINITE_RISK"] == ["org/legacy", "org/modern"]
    json_path, md_path = write_hub_reports(report, tmp_path / "reports")
    markdown = md_path.read_text()
    artifact_section = markdown.split("## Artifact / Processor-Contract Findings", 1)[1].split("## Version-Compatibility", 1)[0]
    assert "org/legacy" not in artifact_section
    assert "org/modern" in artifact_section
    assert "Version-Compatibility / Legacy Schema (Requires PR #1452 Migration)" in markdown
    assert "Conditional FP16 Precision Warnings" in markdown
    assert "Conditional FP16 warnings (unique repositories) | 1" in markdown
    assert "Legacy checkpoints may be valid for their original LeRobot version" in markdown
    assert "FAIL" == json.loads(json_path.read_text())["repos"][0]["findings"][1]["status"]


def test_round2_deduplicates_individual_tensors_across_different_stat_files(checkpoints, tmp_path):
    from safetensors.torch import load_file

    base = checkpoints["bug_5289_pathological_gain"]
    candidate = tmp_path / "different-file"
    shutil.copytree(base, candidate)
    config = json.loads((base / "preprocessor_config.json").read_text())
    filename = config["steps"][0]["state_file"]
    tensors = load_file(candidate / filename)
    tensors["action.mean"] += 123.0  # An unrelated tensor changes the whole file digest.
    save_file(tensors, candidate / filename)
    assert hashlib.sha256((base / filename).read_bytes()).digest() != hashlib.sha256((candidate / filename).read_bytes()).digest()
    repos = []
    for repo_id, checkpoint in (("org/base", base), ("org/candidate", candidate)):
        repos.append(hub_scanner._audit_snapshot({
            "repo_id": repo_id, "revision": "f" * 40, "checkpoint_dir": str(checkpoint),
            "snapshot_dir": str(checkpoint), "errors": [], "critical_download_errors": False,
        }, 1e-4))
    report = hub_scanner._report(repos, {"mode": "cached", "network_used": False})
    payload = next(t for t in repos[0]["stat_tensor_payloads"] if t["tensor_key"] == "observation.state.std")
    descriptor = parse_safetensors_header(_header_only(base / filename), (base / filename).stat().st_size)["observation.state.std"]
    data_start = len(_header_only(base / filename))
    start, end = descriptor["data_offsets"]
    assert payload["payload_sha256"] == hashlib.sha256((base / filename).read_bytes()[data_start+start:data_start+end]).hexdigest()
    references = report["tensor_payload_groups"][payload["payload_sha256"]]
    assert {r["repo_id"] for r in references} == {"org/base", "org/candidate"}
    assert all(r["digest_scope"] == "individual_tensor_payload" for r in references)
    _, markdown_path = write_hub_reports(report, tmp_path / "reports")
    markdown = markdown_path.read_text()
    row = next(line for line in markdown.splitlines() if "org/candidate" in line and "PATHOLOGICAL_STD_GAIN` | FAIL" in line)
    assert "shared_payload_evidence" in row and payload["payload_sha256"] in row


def test_round2_payload_inventory_never_hashes_full_model_weights(checkpoints, monkeypatch):
    checkpoint = checkpoints["clean_act_modern"]
    model = checkpoint / "model.safetensors"
    save_file({"normalize_inputs.buffer_observation_state.std": torch.ones(14),
               "model.weight": torch.zeros(1024)}, model)
    config_path = checkpoint / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    config["steps"].append({"registry_name": "normalizer_processor", "state_file": "model.safetensors", "config": {}})
    config_path.write_text(json.dumps(config))
    original_open = Path.open

    def deny_model_open(path, *args, **kwargs):
        assert path != model, "Payload inventory tried to read neural model weights"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_model_open)
    assert hub_scanner._stat_tensor_payloads(checkpoint)


def test_round2_dedup_preserves_different_native_formulas(checkpoints, tmp_path):
    import copy

    checkpoint = checkpoints["bug_5289_pathological_gain"]
    first = hub_scanner._audit_snapshot({"repo_id": "org/legacy-formula", "revision": "a"*40,
        "checkpoint_dir": str(checkpoint), "errors": []}, 1e-4)
    second = copy.deepcopy(first)
    second["repo_id"] = "org/modern-formula"
    for repo, implementation, formula in (
        (first, "legacy Normalize", "2 / (max - min + eps)"),
        (second, "modern NormalizerProcessorStep", "2 / where(max - min == 0, eps, max - min)"),
    ):
        finding = next(f for f in repo["findings"] if f["check_id"] == "PATHOLOGICAL_STD_GAIN")
        finding["details"].update(normalization_implementation=implementation, gain_formula=formula)
    report = hub_scanner._report([first, second], {"mode": "cached"})
    _, path = write_hub_reports(report, tmp_path / "reports")
    row = next(line for line in path.read_text().splitlines() if "org/modern-formula" in line and "PATHOLOGICAL_STD_GAIN` | FAIL" in line)
    assert "shared_payload_evidence" not in row
    assert "modern NormalizerProcessorStep" in row
    assert "where(max - min == 0" in row


def test_round2_dedup_preserves_distinct_stat_namespaces(checkpoints, tmp_path):
    from safetensors.torch import load_file

    first = checkpoints["bug_4415_silent_bypass"]
    second = tmp_path / "other-prefix"
    shutil.copytree(first, second)
    config = json.loads((second / "preprocessor_config.json").read_text())
    path = second / config["steps"][0]["state_file"]
    tensors = load_file(path)
    save_file({key.replace("wrong_dataset/", "another_dataset/"): value for key, value in tensors.items()}, path)
    repos = [hub_scanner._audit_snapshot({"repo_id": repo_id, "revision": "a"*40,
        "checkpoint_dir": str(checkpoint), "errors": []}, 1e-4)
        for repo_id, checkpoint in (("org/first-prefix", first), ("org/other-prefix", second))]
    _, output = write_hub_reports(hub_scanner._report(repos, {"mode": "cached"}), tmp_path / "reports")
    rows = [line for line in output.read_text().splitlines() if "org/other-prefix" in line and "STAT_KEY_NAMESPACE_MISMATCH` | FAIL" in line]
    assert rows and all("shared_payload_evidence" not in row for row in rows)
    assert any("another_dataset/" in row for row in rows)


def test_round3_nonblocking_legacy_precision_advice_stays_in_report(tmp_path):
    advisory = {"check_id": "FP16_NON_FINITE_RISK", "status": "PASS", "severity": "WARNING",
                "summary": "Declared CPU float32 is finite; half-buffer execution remains conditional.",
                "details": {"fp16_execution_conditional": True, "normalization_implementation": "legacy Normalize",
                            "checkpoint_device": "cpu", "checkpoint_use_amp": False,
                            "non_blocking_advisory": True}, "provenance": None}
    repo = {"repo_id": "org/float32-legacy", "revision": "a" * 40, "classification": "LEGACY_POLICY",
            "checkpoint_dir": str(tmp_path), "errors": [], "findings": [advisory]}
    report = hub_scanner._report([repo], {"mode": "cached", "network_used": False})
    assert report["summary"]["issues"]["FP16_NON_FINITE_RISK"] == []
    assert report["summary"]["finding_categories"]["artifact_contract"]["repository_count"] == 0
    assert report["summary"]["finding_categories"]["conditional_precision"]["repository_count"] == 1
    assert report["summary"]["finding_categories"]["conditional_precision"]["finding_count"] == 1
    assert report["summary"]["advisories"]["FP16_NON_FINITE_RISK"] == ["org/float32-legacy"]
    _, path = write_hub_reports(report, tmp_path / "reports")
    conditional = path.read_text().split("## Conditional FP16 Precision Warnings", 1)[1]
    assert "org/float32-legacy" in conditional and "PASS / WARNING" in conditional
    assert "do not independently fail standard CLI mode" in conditional
    assert "Strict mode rejects WARNING advisories" in conditional


def test_round4_report_counts_shared_signatures_prerequisites_and_discloses_seed_queries(tmp_path):
    def finding(check, *, key="action.mean", mode=None, digest="490ab239d96e", details=None):
        return {"check_id": check, "status": "FAIL", "severity": "CRITICAL",
                "summary": "Required normalization buffer action.mean is missing; normalization may be bypassed."
                if check == "FEATURE_CONTRACT" else "Verified artifact condition.",
                "details": {"feature_key": key.rsplit(".", 1)[0], "normalization_mode": mode, **(details or {})},
                "provenance": {"file_path": "stats.safetensors", "tensor_key": key,
                               "sha256_prefix": digest, "violating_indices": [0]}}

    def payload(key, sha):
        return {"file_path": "stats.safetensors", "tensor_key": key, "payload_sha256": sha,
                "dtype": "F32", "shape": [6], "digest_scope": "individual_tensor_payload"}

    repos = []
    unbound_ids = ["lerobot/smolvla_base", "y1y2y3/smolvla_base1"] + [f"org/base-copy-{i}" for i in range(8)]
    for repo_id in unbound_ids:
        fs = [finding("UNBOUND_BASE_MODEL_STATS"),
              finding("FEATURE_CONTRACT", details={"available_buffer_keys": []})]
        if repo_id == "y1y2y3/smolvla_base1":
            fs += [finding("GIT_LFS_POINTER_ARTIFACT", details={"lfs_oid_sha256": "490ab239d96e" + "0" * 52})] * 2
        hub_scanner._annotate_base_prerequisites(repo_id, fs)
        repos.append({"repo_id": repo_id, "revision": "f" * 40, "classification": "MODERN_POLICY",
                      "checkpoint_dir": str(tmp_path / repo_id.replace("/", "_")), "errors": [], "findings": fs,
                      "stat_tensor_payloads": [payload("so100-blue.buffer.action.mean", "f" * 64)]})
    fs = [finding("FEATURE_CONTRACT", details={"available_buffer_keys": []})]
    hub_scanner._annotate_base_prerequisites("lerobot/pi0_old", fs)
    repos.append({"repo_id": "lerobot/pi0_old", "revision": "f" * 40, "classification": "MODERN_POLICY",
                  "checkpoint_dir": str(tmp_path / "pi0-old"), "errors": [], "findings": fs})
    for repo_id, sha, mode, key in (
        ("org/xarm", "a" * 64, "MEAN_STD", "observation.state.std"),
        ("org/xarm-copy", "a" * 64, "MEAN_STD", "observation.state.std"),
        ("org/zero-std", "b" * 64, "MEAN_STD", "observation.state.std"),
        ("org/equal-bounds", "c" * 64, "MIN_MAX", "observation.environment_state.min"),
    ):
        repos.append({"repo_id": repo_id, "revision": "e" * 40, "classification": "LEGACY_POLICY",
                      "checkpoint_dir": str(tmp_path / repo_id.replace("/", "_")), "errors": [],
                      "findings": [finding("PATHOLOGICAL_STD_GAIN", key=key, mode=mode)],
                      "stat_tensor_payloads": [payload(key, sha)]})
    for i in range(85):
        repos.append({"repo_id": f"org/clean-{i}", "revision": "d" * 40, "classification": "MODERN_POLICY",
                      "checkpoint_dir": str(tmp_path / f"clean-{i}"), "errors": [], "findings": []})
    for i, repo in enumerate(repos):
        repo["findings"].append({"check_id": "TEMPORAL_ANCHOR_MISMATCH", "status": "PASS" if i < 24 else "UNKNOWN",
                                 "severity": "INFO", "details": {},
                                 "summary": "No enabled relative-action conversion requires a temporal anchor."
                                 if i < 24 else "Temporal probe unavailable."})
    discovery = {"queries": [
        {"filter": "lerobot", "limit": 25, "full": True, "search": "smolvla_base", "returned": 25},
        {"filter": "lerobot", "limit": 25, "full": True, "search": "act_aloha_sim_transfer_cube_human", "returned": 4},
    ]}
    report = hub_scanner._report(repos, {"mode": "cached", "original_discovery": discovery})
    counts = report["summary"]["artifact_finding_families"]
    assert (counts["family_count"], counts["repository_count"], counts["defect_or_numeric_risk_signatures"],
            counts["prerequisite_families"]) == (5, 15, 4, 1)
    assert report["summary"]["finding_categories"]["base_model_prerequisite"]["repository_count"] == 11
    assert report["summary"]["finding_categories"]["artifact_contract"]["repository_count"] == 5
    assert report["summary"]["temporal_coverage"] == {
        "single_frame_or_non_relative_repos": 24, "unknown_repos": 76,
        "not_evaluated_repos": 0, "multi_step_relative_repos_exercised": 0,
        "relative_action_policy_repos": 0, "non_relative_policy_repos": 24}
    assert len(report["summary"]["shared_base_stat_evidence"]["repo_ids"]) == 10
    _, path = write_hub_reports(report, tmp_path / "reports")
    text = path.read_text()
    assert "5 independent artifact finding families across 15 repositories" in text
    assert "4 defect/numeric-risk signatures and 1 dataset-stat binding prerequisite family" in text
    assert "Dataset-Stat Binding Prerequisites (Non-Defect)" in text
    assert '"search": "smolvla_base"' in text and '"search": "act_aloha_sim_transfer_cube_human"' in text
    assert "0 relative-action policies" in text and "24 non-relative policies" in text and "76 UNKNOWN" in text
    assert "0 multi-step relative policies exercised" in text
    assert "state[:, 0]" in text and "state[:, -1]" in text
    assert "current-first windows" in text and "oldest-first state history" in text
    assert "VLA-JEPA and LingBot-VA" in text and "`[-1, 0]`" in text
    assert "explicitly selected chronology metadata overrides the inference" in text
    prerequisite_section = text.split("## Intentional Base Models:", 1)[1].split("## Version-Compatibility", 1)[0]
    assert "### Exact base-model prerequisite evidence" in prerequisite_section
    assert "### Exact conditional precision evidence" not in prerequisite_section
    assert '_INTENTIONAL_BASE_REPOS = {"lerobot/smolvla_base", "lerobot/pi0_old"}' in prerequisite_section
    assert "repository ID" in prerequisite_section and "Shared buffer hashes" in prerequisite_section


def test_round4_temporal_scope_does_not_count_non_policy_repo_as_verified_policy(tmp_path):
    def repo(repo_id, classification, findings):
        return {"repo_id": repo_id, "classification": classification, "revision": "e" * 40,
                "checkpoint_dir": str(tmp_path), "errors": [], "findings": findings}
    report = hub_scanner._report([
        repo("org/policy", "MODERN_POLICY", [{"check_id": "TEMPORAL_ANCHOR_MISMATCH", "status": "PASS",
            "severity": "INFO", "summary": "No enabled relative-action conversion requires a temporal anchor.", "details": {}}]),
        repo("org/assets", "NON_POLICY_REPO", []),
    ], {"mode": "cached"})
    assert report["summary"]["temporal_coverage"] == {
        "single_frame_or_non_relative_repos": 1, "unknown_repos": 0,
        "not_evaluated_repos": 1, "multi_step_relative_repos_exercised": 0,
        "relative_action_policy_repos": 0, "non_relative_policy_repos": 1}


def test_relative_and_non_relative_temporal_policy_counts_are_distinct(tmp_path):
    def repo(repo_id, finding):
        return {"repo_id": repo_id, "classification": "MODERN_POLICY", "revision": "e" * 40,
                "checkpoint_dir": str(tmp_path), "errors": [], "findings": [finding]}
    report = hub_scanner._report([
        repo("org/relative", {"check_id": "TEMPORAL_ANCHOR_MISMATCH", "status": "PASS", "severity": "INFO",
             "summary": "Single-frame relative-action compatibility verified.",
             "details": {"step_class": "RelativeActionsProcessorStep", "n_obs_steps": 1}}),
        repo("org/non-relative", {"check_id": "TEMPORAL_ANCHOR_MISMATCH", "status": "PASS", "severity": "INFO",
             "summary": "No enabled relative-action conversion requires a temporal anchor.", "details": {}}),
        repo("org/unverified", {"check_id": "TEMPORAL_ANCHOR_MISMATCH", "status": "UNKNOWN", "severity": "INFO",
             "summary": "Unverified temporal coverage.", "details": {}}),
    ], {"mode": "cached"})
    coverage = report["summary"]["temporal_coverage"]
    assert coverage["relative_action_policy_repos"] == 1
    assert coverage["non_relative_policy_repos"] == 1
    assert coverage["unknown_repos"] == 1
    assert coverage["multi_step_relative_repos_exercised"] == 0
