"""Bounded local legacy evidence and independent-stage regression checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

import lerobot_flinch.preflight as preflight
from lerobot_flinch.hub_scanner import (
    MAX_HEADER_BYTES,
    MAX_LEGACY_NORMALIZATION_BYTES,
    MAX_LEGACY_TENSOR_BYTES,
)
from lerobot_flinch.models import DiagnosticStatus
from tests.fixtures.make_bug_fixtures import make_bug_fixtures

STATE = "normalize_inputs.buffer_observation_state"
ACTION = "unnormalize_outputs.buffer_action"


def _legacy(root: Path, *, mode: str = "mean_std", large: bool = False) -> Path:
    root.mkdir()
    width = MAX_LEGACY_TENSOR_BYTES // 4 + 1 if large else 7
    std = torch.full((width,), 0.5)
    std[4] = 6.6630414e-6
    save_file(
        {
            "model.weight": torch.zeros(262144),
            f"{STATE}.mean": torch.zeros(width),
            f"{STATE}.std": std,
            f"{STATE}.min": torch.full((width,), -1.0),
            f"{STATE}.max": torch.ones(width),
            f"{ACTION}.mean": torch.zeros(7),
            f"{ACTION}.std": torch.ones(7),
            # Unused metadata/image tensors must not be materialized as vectors.
            "normalize_inputs.buffer_observation_image.mean": torch.zeros(3, 16, 16),
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "vqbet",
                "input_shapes": {"observation.state": [width]},
                "output_shapes": {"action": [7]},
                "input_normalization_modes": {"observation.state": mode},
                "output_normalization_modes": {"action": "mean_std"},
            }
        )
    )
    return root


def test_local_legacy_evidence_has_bounded_header_and_exact_tensor_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _legacy(tmp_path / "legacy")
    model = checkpoint / "model.safetensors"
    values = preflight._local_legacy_values(checkpoint)
    assert values is not None
    _, state, metadata, header = values
    assert set(state) == {f"{STATE}.{name}" for name in ("mean", "std", "min", "max")} | {
        f"{ACTION}.{name}" for name in ("mean", "std")
    }
    assert metadata["bytes_read"] < 2048
    prefix = model.read_bytes()[: 8 + header["header_len"]]
    expected_hash = hashlib.sha256(prefix).hexdigest()[:12]
    allowed = [(0, len(prefix)), *(item["absolute_range"] for item in metadata["tensors"].values())]
    original_open = Path.open

    class BoundedReader:
        def __init__(self, stream: Any):
            self.stream = stream

        def __enter__(self) -> BoundedReader:
            self.stream.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.stream.__exit__(*args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.stream, name)

        def read(self, amount: int = -1) -> bytes:
            start = self.stream.tell()
            assert amount >= 0
            assert any(low <= start and start + amount <= high for low, high in allowed)
            return self.stream.read(amount)

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = original_open(path, *args, **kwargs)
        return BoundedReader(stream) if path.resolve() == model.resolve() else stream

    def no_whole_model_digest(stream: Any, algorithm: str) -> Any:
        assert Path(stream.name).name != "model.safetensors"
        return real_digest(stream, algorithm)

    real_digest = hashlib.file_digest
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(hashlib, "file_digest", no_whole_model_digest)
    preflight._file_digest.cache_clear()
    report = preflight.run_preflight(checkpoint)
    (gain,) = [
        finding
        for finding in report.findings
        if finding.check_id == "PATHOLOGICAL_STD_GAIN" and finding.status is DiagnosticStatus.FAIL
    ]
    assert gain.provenance is not None
    assert gain.provenance.violating_indices == [4]
    assert gain.provenance.sha256_prefix == expected_hash
    assert gain.provenance.digest_scope == "safetensors_header_bytes_including_length_prefix"
    descriptor = metadata["tensors"][f"{STATE}.std"]
    assert gain.details["payload_sha256"] == descriptor["payload_sha256"]
    assert gain.details["absolute_range"] == descriptor["absolute_range"]
    assert descriptor["absolute_range"][1] - descriptor["absolute_range"][0] == 28


@pytest.mark.parametrize("mode,expected", [("mean_std", True), ("min_max", False)])
def test_local_legacy_screens_only_the_active_mode(tmp_path: Path, mode: str, expected: bool) -> None:
    report = preflight.run_preflight(_legacy(tmp_path / "legacy", mode=mode))
    assert (
        any(
            finding.check_id == "PATHOLOGICAL_STD_GAIN" and finding.status is DiagnosticStatus.FAIL
            for finding in report.findings
        )
        is expected
    )


@pytest.mark.parametrize(
    "corruption",
    ["negative", "float", "bool", "overlap", "shape_bytes", "truncated", "duplicate", "oversized"],
)
def test_invalid_local_container_is_rejected_without_numeric_pass(tmp_path: Path, corruption: str) -> None:
    checkpoint = _legacy(tmp_path / "legacy")
    model = checkpoint / "model.safetensors"
    descriptor = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    entries = {f"{STATE}.std": descriptor}
    payload = b"\0" * 8
    if corruption == "negative":
        descriptor["data_offsets"] = [-4, 0]
    elif corruption == "float":
        descriptor["data_offsets"] = [0.0, 4.0]
    elif corruption == "bool":
        descriptor["data_offsets"] = [False, 4]
    elif corruption == "overlap":
        entries[f"{STATE}.mean"] = {**descriptor}
    elif corruption == "shape_bytes":
        descriptor["shape"] = [2]
    raw = json.dumps(entries).encode()
    if corruption == "duplicate":
        item = json.dumps(descriptor)
        raw = ('{"' + STATE + '.std":' + item + ',"' + STATE + '.std":' + item + "}").encode()
    length = len(raw)
    if corruption == "truncated":
        length += 100
    elif corruption == "oversized":
        length = MAX_HEADER_BYTES + 1
    model.write_bytes(length.to_bytes(8, "little") + raw + payload)
    with pytest.raises(ValueError):
        preflight._local_legacy_values(checkpoint)
    report = preflight.run_preflight(checkpoint)
    assert any(
        f.check_id == "SCHEMA_GENERATION" and f.status is DiagnosticStatus.FAIL for f in report.findings
    )
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        findings = [f for f in report.findings if f.check_id == check]
        assert findings and all(f.status is DiagnosticStatus.UNKNOWN for f in findings)
    assert len([f for f in report.findings if f.check_id == "PHYSICAL_SEMANTICS"]) == 3


def test_oversized_vector_is_skipped_with_unknown_coverage(tmp_path: Path) -> None:
    checkpoint = _legacy(tmp_path / "legacy", large=True)
    values = preflight._local_legacy_values(checkpoint)
    assert values is not None
    _, state, metadata, _ = values
    assert f"{STATE}.std" not in state
    assert "payload budget" in metadata["skipped_tensors"][f"{STATE}.std"]
    report = preflight.run_preflight(checkpoint)
    assert any(
        f.check_id == "PATHOLOGICAL_STD_GAIN" and f.status is DiagnosticStatus.UNKNOWN
        for f in report.findings
    )
    assert not any(
        f.check_id == "PATHOLOGICAL_STD_GAIN" and f.status is DiagnosticStatus.PASS for f in report.findings
    )


def test_local_legacy_aggregate_payload_budget_is_enforced(tmp_path: Path) -> None:
    checkpoint = _legacy(tmp_path / "legacy")
    tensors = {
        f"unnormalize_outputs.dataset{i}_buffer_action.std": torch.ones(MAX_LEGACY_TENSOR_BYTES // 4)
        for i in range(6)
    }
    save_file(tensors, checkpoint / "model.safetensors")
    values = preflight._local_legacy_values(checkpoint)
    assert values is not None
    _, state, metadata, header = values
    assert (
        sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        <= MAX_LEGACY_NORMALIZATION_BYTES
    )
    assert metadata["bytes_read"] <= 8 + header["header_len"] + MAX_LEGACY_NORMALIZATION_BYTES
    assert len(metadata["skipped_tensors"]) == 2


def test_provenance_cache_invalidates_when_timestamp_is_restored(tmp_path: Path) -> None:
    path = tmp_path / "normalizer.safetensors"
    save_file({"action.std": torch.ones(7)}, path)
    original = path.stat()
    first = preflight._provenance(path)
    save_file({"action.std": torch.full((7,), 2.0)}, path)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    second = preflight._provenance(path)
    assert first.sha256_prefix != second.sha256_prefix
    assert second.sha256_prefix == hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def test_windows_creation_time_does_not_hide_in_place_stat_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "normalizer.safetensors"
    save_file({"action.std": torch.ones(7)}, path)
    original_stat = path.stat()
    # Prime the persistent cache with the original bytes. Emulate Windows keeping
    # every cache-key field unchanged after an equal-length in-place rewrite.
    preflight._file_digest.cache_clear()
    original_hash = preflight._file_digest(
        str(path.resolve()),
        original_stat.st_size,
        original_stat.st_mtime_ns,
        original_stat.st_ctime_ns,
        original_stat.st_ino,
    )
    save_file({"action.std": torch.full((7,), 2.0)}, path)
    real_stat = Path.stat

    def unchanged_stat(artifact: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        return original_stat if artifact == path else real_stat(artifact, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unchanged_stat)
    monkeypatch.setattr(preflight, "_USE_FILE_DIGEST_CACHE", False)
    provenance = preflight._provenance(path)
    assert provenance.sha256_prefix != original_hash
    assert provenance.sha256_prefix == hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@pytest.mark.parametrize("broken_probe", [False, True])
def test_provenance_or_composed_probe_failure_retains_other_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken_probe: bool
) -> None:
    checkpoint = make_bug_fixtures(tmp_path / "fixtures")["bug_5289_pathological_gain"]

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected composed probe")

    def inaccessible_digest(*args: Any, **kwargs: Any) -> Any:
        raise OSError("injected provenance permission failure")

    if broken_probe:
        monkeypatch.setattr(preflight, "_screen_native_normalization_chain", broken)
    else:
        monkeypatch.setattr(preflight, "_file_digest", inaccessible_digest)
        monkeypatch.setattr(preflight, "_uncached_file_digest", inaccessible_digest)
    report = preflight.run_preflight(checkpoint)
    assert any(
        f.check_id == "PATHOLOGICAL_STD_GAIN" and f.status is DiagnosticStatus.FAIL for f in report.findings
    )
    assert len([f for f in report.findings if f.check_id == "PHYSICAL_SEMANTICS"]) == 3
    if broken_probe:
        assert any(f.check_id == "CHECK_INCOMPLETE" for f in report.findings)
        assert any(
            f.check_id == "FP16_NON_FINITE_RISK" and f.status is DiagnosticStatus.UNKNOWN
            for f in report.findings
        )
        assert not any(
            f.check_id == "FP16_NON_FINITE_RISK" and f.status is DiagnosticStatus.PASS
            for f in report.findings
        )
