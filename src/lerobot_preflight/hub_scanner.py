"""Download bounded public Hub metadata, then audit an offline checkpoint snapshot."""

from __future__ import annotations

import fnmatch
import hashlib
import html
import json
import math
import ntpath
import posixpath
import re
import tempfile
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.request import Request, urlopen
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
import torch
import numpy as np
from safetensors.torch import save_file

from .models import DiagnosticFinding, DiagnosticStatus, Severity
from .preflight import _local_file, _verified_legacy_values, run_preflight
from .temporal import run_temporal_audit

MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_HEADER_BYTES = 4 * 1024 * 1024
HEADER_PREFIX_BYTES = 64 * 1024
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_REPOSITORIES = 100
MAX_LEGACY_TENSOR_BYTES = 64 * 1024
MAX_LEGACY_NORMALIZATION_BYTES = 256 * 1024
MAX_LEGACY_TENSORS = 128
POLICY_FAMILIES = ("act", "diffusion", "smolvla", "pi0", "vqbet", "tdmpc")
_TARGET_SEEDS = ("lerobot/smolvla_base", "lerobot/act_aloha_sim_transfer_cube_human")
# These official base checkpoints are explicitly identified as such in this
# audit's scope. A repository name alone is never evidence of fine-tuning.
_INTENTIONAL_BASE_REPOS = {"lerobot/smolvla_base", "lerobot/pi0_old"}
ISSUE_CHECKS = (
    "PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "TEMPORAL_ANCHOR_MISMATCH",
    "SCHEMA_GENERATION_MISMATCH", "STAT_KEY_NAMESPACE_MISMATCH", "DEAD_ACTION_DIMENSION", "FEATURE_CONTRACT",
    "UNBOUND_BASE_MODEL_STATS", "CAMERA_KEY_CONTRACT_MISMATCH", "GIT_LFS_POINTER_ARTIFACT",
    "INVALID_NORMALIZATION_STATS",
    "NORMALIZATION_CHAIN_NON_FINITE",
    "BF16_NON_FINITE_RISK",
)
_BUCKETS = ("MODERN_POLICY", "LEGACY_POLICY", "NON_POLICY_REPO", "GATED_OR_PRIVATE", "MIXED_POLICY", "UNKNOWN")
_JSON_NAMES = {
    "config.json", "policy_preprocessor.json", "policy_postprocessor.json",
    "preprocessor_config.json", "postprocessor_config.json", "train_config.json",
    "metadata.json", "meta.json", "model.safetensors.index.json",
}
_PROCESSOR_CONFIGS = {
    "policy_preprocessor.json", "policy_postprocessor.json", "preprocessor_config.json", "postprocessor_config.json",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key: {key}")
        result[key] = value
    return result


def parse_safetensors_header(prefix: bytes, total_size: int | None = None) -> dict[str, dict[str, Any]]:
    """Parse only the 8-byte length prefix and complete UTF-8 JSON header.

    Tensor bodies are neither needed nor read. Returned dtype and shape values
    are exact header declarations, not validation of the absent tensor payload.
    """
    if len(prefix) < 8:
        raise ValueError("Truncated safetensors length prefix")
    length = int.from_bytes(prefix[:8], "little")
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError("Safetensors header length exceeds the bounded header budget")
    if len(prefix) < 8 + length:
        raise ValueError("Truncated safetensors JSON header")
    if total_size is not None and (type(total_size) is not int or total_size < 8 + length):
        raise ValueError("Safetensors header exceeds the declared file size")
    try:
        header = json.loads(prefix[8:8 + length].decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")
    tensors, intervals = {}, []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict) or not isinstance(value.get("dtype"), str) or not value["dtype"]:
            raise ValueError(f"Invalid tensor dtype declaration: {key}")
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {key}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets) or offsets[1] < offsets[0]):
            raise ValueError(f"Invalid tensor data offsets: {key}")
        if total_size is not None and offsets[1] > total_size - 8 - length:
            raise ValueError(f"Tensor offsets exceed the declared file size: {key}")
        tensors[key] = {"dtype": value["dtype"], "shape": shape, "data_offsets": offsets}
        if offsets[0] != offsets[1]:
            intervals.append(tuple(offsets))
    last_end = 0
    for start, end in sorted(intervals):
        if start < last_end:
            raise ValueError("Overlapping tensor offsets in safetensors header")
        last_end = end
    return tensors


def _read_bounded(stream: Any, amount: int) -> bytes:
    parts = []
    while amount > 0:
        requested = min(amount, HEADER_PREFIX_BYTES)
        block = stream.read(requested)
        if not block:
            break
        if len(block) > requested:
            raise ValueError("HTTP stream exceeded its requested read budget")
        parts.append(block)
        amount -= len(block)
    return b"".join(parts)


def _range_read(url: str, start: int, size: int, opener: Callable) -> tuple[bytes, int, int | None]:
    request = Request(url, headers={
        "Range": f"bytes={start}-{start + size - 1}", "Accept-Encoding": "identity",
        "User-Agent": "lerobot-preflight/metadata-scanner",
    })
    with opener(request, timeout=20) as response:
        status = response.status
        if status not in (200, 206):
            raise ValueError(f"Unexpected HTTP status for bounded header request: {status}")
        if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
            raise ValueError("Compressed responses cannot be treated as byte ranges")
        declared_size = None
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != start or int(match[2]) < start:
                raise ValueError("Server returned a different byte range than requested")
            declared_size = int(match[3]) if match[3] != "*" else None
        elif response.headers.get("Content-Length"):
            declared_size = int(response.headers["Content-Length"])
        # An HTTP 200 server ignored Range. Read only the bounded prefix needed,
        # then close it; never access response.content or copy the full body.
        skipped = _read_bounded(response, start) if status == 200 and start else b""
        if status == 200 and len(skipped) != start:
            raise ValueError("Truncated response before requested safetensors header offset")
        result = _read_bounded(response, size)
        return result, len(skipped) + len(result), declared_size


def fetch_safetensors_header(
    repo_id: str, filename: str = "model.safetensors", revision: str = "main",
    total_size: int | None = None, *, opener: Callable | None = None,
) -> dict[str, Any]:
    """Read a 64 KiB prefix, extending only to the bounded complete header."""
    open_stream = opener or urlopen
    url = hf_hub_url(repo_id, filename, revision=revision)
    prefix, consumed, response_size = _range_read(url, 0, HEADER_PREFIX_BYTES, open_stream)
    if total_size is not None and response_size is not None and total_size != response_size:
        raise ValueError("Hub metadata and Range response disagree on file size")
    total_size = total_size if total_size is not None else response_size
    if len(prefix) < 8:
        raise ValueError("Truncated safetensors length prefix")
    length = int.from_bytes(prefix[:8], "little")
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError("Safetensors header length exceeds the bounded header budget")
    required = 8 + length
    if len(prefix) < required:
        remaining, received, extended_size = _range_read(url, len(prefix), required - len(prefix), open_stream)
        if total_size is not None and extended_size is not None and total_size != extended_size:
            raise ValueError("Header byte ranges disagree on file size")
        prefix += remaining
        consumed += received
    tensors = parse_safetensors_header(prefix, total_size)
    return {
        "format": "safetensors-header-v1", "source_file": PurePosixPath(filename).name,
        "repo_id": repo_id, "revision": revision, "total_size": total_size,
        "header_len": length, "header_sha256_prefix": hashlib.sha256(prefix[:required]).hexdigest()[:12],
        "header_digest_scope": "safetensors_header_bytes_including_length_prefix",
        "bytes_fetched": consumed, "complete": True, "tensor_values_available": False, "tensors": tensors,
    }


def _legacy_state_action_key(key: str) -> bool:
    """Select embedded scalar/vector normalization buffers, never model parameters."""
    if not key.startswith(("normalize_inputs.", "unnormalize_outputs.")):
        return False
    namespace, separator, statistic = key.rpartition(".")
    if not separator or statistic not in {"mean", "std", "min", "max", "q01", "q99", "q10", "q90"}:
        return False
    namespace = namespace.lower()
    return ("buffer" in namespace and ("state" in namespace or "action" in namespace)
            and not any(token in namespace for token in ("image", "camera", "frame_index", "task_index")))


class _LegacyRangeError(ValueError):
    def __init__(self, message: str, bytes_fetched: int = 0):
        super().__init__(message)
        self.bytes_fetched = bytes_fetched


def _strict_tensor_range_read(
    url: str, start: int, size: int, total_size: int, opener: Callable,
) -> bytes:
    """Require an exact 206 response before reading any arbitrary-offset body.

    Unlike the bounded header reader, this must never skip an HTTP 200 body to
    reach a potentially gigabyte-scale offset. A deterministic query parameter
    also prevents a cached header-prefix response being reused for a new range.
    """
    end = start + size - 1
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query))
    query.update({"download": "true", "lerobot_preflight_range": f"{start}-{end}"})
    ranged_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
    request = Request(ranged_url, headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
        "User-Agent": "lerobot-preflight/normalization-range-scanner",
    })
    with opener(request, timeout=20) as response:
        if response.status != 206:
            raise _LegacyRangeError("Legacy normalization tensors require HTTP 206; the response body was not read")
        if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
            raise _LegacyRangeError("Compressed tensor range responses are unsupported")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (start, end, total_size):
            raise _LegacyRangeError("Server returned a different tensor byte range or file size; body was not read")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != size:
            raise _LegacyRangeError("Tensor range Content-Length disagrees with the requested byte count")
        payload = _read_bounded(response, size)
        if len(payload) != size:
            raise _LegacyRangeError("Truncated legacy normalization tensor payload", len(payload))
        return payload


def _decode_tensor_slice(raw: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
    """Safetensors payloads are little-endian, regardless of the reader's host."""
    if dtype == "BF16":
        words = np.frombuffer(raw, dtype=np.dtype("<u2")).astype(np.uint16, copy=True)
        return torch.from_numpy(words).view(torch.bfloat16).reshape(shape)
    little_dtype, native_dtype = {
        "F16": ("<f2", np.float16), "F32": ("<f4", np.float32), "F64": ("<f8", np.float64),
    }[dtype]
    values = np.frombuffer(raw, dtype=np.dtype(little_dtype)).astype(native_dtype, copy=True)
    return torch.from_numpy(values).reshape(shape)


def fetch_legacy_normalization_tensors(
    repo_id: str, header: dict[str, Any], *, filename: str = "model.safetensors",
    revision: str = "main", opener: Callable | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fetch only bounded scalar/vector state/action buffers from a pinned model.

    Original tensor keys are preserved. Header offsets are relative to the data
    section; recorded absolute ranges are [start, end), whereas HTTP ends are
    inclusive. Reconstruction is not a claim that a buffer is active: preflight
    resolves the checkpoint's feature/mode/direction contract separately.
    """
    if header.get("format") != "safetensors-header-v1" or not isinstance(header.get("tensors"), dict):
        raise ValueError("Expected parsed safetensors header metadata")
    if header.get("repo_id") != repo_id or header.get("revision") != revision:
        raise ValueError("Header repository/revision does not match the pinned tensor request")
    if header.get("source_file") != PurePosixPath(filename).name:
        raise ValueError("Header source_file does not match the model tensor request")
    _safe_name(filename)
    if header.get("complete") is False:
        raise ValueError("Incomplete model headers cannot authorize tensor payload requests")
    header_len, total_size = header.get("header_len"), header.get("total_size")
    if (type(header_len) is not int or not 0 < header_len <= MAX_HEADER_BYTES
            or type(total_size) is not int or total_size < 8 + header_len):
        raise ValueError("Header length and total model size must be bounded positive integers")
    dtype_map = {"F16": (torch.float16, 2), "F32": (torch.float32, 4),
                 "F64": (torch.float64, 8), "BF16": (torch.bfloat16, 2)}
    result: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {
        "format": "legacy-normalization-ranges-v1", "repo_id": repo_id, "revision": revision,
        "source_file": filename, "total_size": total_size, "header_len": header_len,
        "header_sha256_prefix": header.get("header_sha256_prefix"),
        "tensor_file": "legacy_normalization.safetensors", "tensor_file_sha256": None,
        "tensor_values_available": False, "complete": False, "bytes_fetched": 0,
        "tensors": {}, "skipped_tensors": {}, "errors": [],
    }
    selected, budget = [], 0
    candidates = [(key, descriptor) for key, descriptor in header["tensors"].items()
                  if isinstance(key, str) and _legacy_state_action_key(key)]
    if len(candidates) > MAX_LEGACY_TENSORS:
        raise ValueError("Legacy normalization tensor count exceeds the bounded request budget")
    for key, descriptor in candidates:
        try:
            if not isinstance(descriptor, dict):
                raise ValueError("tensor descriptor must be an object")
            shape, offsets, dtype = descriptor.get("shape"), descriptor.get("data_offsets"), descriptor.get("dtype")
            if (not isinstance(shape, list) or len(shape) > 1
                    or any(type(dim) is not int or dim <= 0 for dim in shape)):
                raise ValueError("only nonempty one-dimensional or scalar buffers are supported")
            if dtype not in dtype_map:
                raise ValueError(f"unsupported normalization dtype {dtype}")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int or offset < 0 for offset in offsets)
                    or offsets[1] <= offsets[0]):
                raise ValueError("invalid tensor data_offsets")
            amount = math.prod(shape) * dtype_map[dtype][1]
            if offsets[1] - offsets[0] != amount:
                raise ValueError("tensor byte count disagrees with dtype and shape")
            if amount > MAX_LEGACY_TENSOR_BYTES or budget + amount > MAX_LEGACY_NORMALIZATION_BYTES:
                raise ValueError("normalization buffer exceeds the bounded payload budget")
            start, end = 8 + header_len + offsets[0], 8 + header_len + offsets[1]
            if start < 8 + header_len or end > total_size:
                raise ValueError("tensor byte range exceeds the declared model file")
            selected.append((start, end, key, descriptor))
            budget += amount
        except (TypeError, ValueError) as exc:
            metadata["skipped_tensors"][key] = str(exc)
    selected.sort()
    last_end = 0
    for start, end, _, _ in selected:
        if start < last_end:
            raise ValueError("Overlapping legacy normalization tensor ranges")
        last_end = end
    # Coalesce only exactly adjacent selected tensors: no camera or weight bytes
    # between their intervals may be read to save a request.
    groups: list[list[tuple[int, int, str, dict[str, Any]]]] = []
    for item in selected:
        if groups and groups[-1][-1][1] == item[0]:
            groups[-1].append(item)
        else:
            groups.append([item])
    for group in groups:
        start, end = group[0][0], group[-1][1]
        try:
            payload = _strict_tensor_range_read(hf_hub_url(repo_id, filename, revision=revision),
                                                start, end - start, total_size, opener or urlopen)
            metadata["bytes_fetched"] += len(payload)
            for tensor_start, tensor_end, key, descriptor in group:
                raw = payload[tensor_start - start:tensor_end - start]
                result[key] = _decode_tensor_slice(raw, descriptor["dtype"], descriptor["shape"])
                metadata["tensors"][key] = {
                    **descriptor, "absolute_range": [tensor_start, tensor_end],
                    "payload_sha256": hashlib.sha256(raw).hexdigest(),
                }
        except Exception as exc:
            metadata["bytes_fetched"] += getattr(exc, "bytes_fetched", 0)
            metadata["errors"].append({"tensor_keys": [item[2] for item in group],
                                       "error": f"{type(exc).__name__}: {exc}"})
    metadata["tensor_values_available"] = bool(result)
    metadata["complete"] = bool(candidates) and len(result) == len(candidates)
    return result, metadata


def enrich_legacy_snapshot(manifest: dict[str, Any], *, opener: Callable | None = None) -> dict[str, Any]:
    """Fetch tiny embedded normalization buffers before local-only audits run."""
    updated = dict(manifest)
    updated.setdefault("normalization_bytes_fetched", 0)
    checkpoint = Path(updated["checkpoint_dir"]).resolve()
    header_path = checkpoint / "model.safetensors.header.json"
    if not header_path.is_file() or not header_path.resolve().is_relative_to(checkpoint):
        return updated
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
        if not any(_legacy_state_action_key(key) for key in header.get("tensors", {})):
            return updated
        state_path = checkpoint / "legacy_normalization.safetensors"
        metadata_path = checkpoint / "legacy_normalization.json"
        if any(path.is_symlink() or not path.resolve().is_relative_to(checkpoint)
               for path in (state_path, metadata_path)):
            raise ValueError("Legacy normalization artifacts must be local snapshot files")
        if state_path.is_file() and metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (previous.get("complete") is True and previous.get("repo_id") == updated["repo_id"]
                    and previous.get("revision") == updated["revision"]
                    and previous.get("header_sha256_prefix") == header.get("header_sha256_prefix")
                    and previous.get("tensor_file_sha256") == hashlib.sha256(state_path.read_bytes()).hexdigest()):
                updated.update({"legacy_normalization_metadata": str(metadata_path), "legacy_normalization_cached": True})
                return updated
        filename = (PurePosixPath(updated.get("checkpoint_subdir", ".")) / "model.safetensors").as_posix()
        state, metadata = fetch_legacy_normalization_tensors(updated["repo_id"], header, filename=filename,
                                                             revision=updated["revision"], opener=opener)
        if state:
            save_file(state, state_path)
            metadata["tensor_file_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        updated["normalization_bytes_fetched"] += metadata["bytes_fetched"]
        updated["legacy_normalization_metadata"] = str(metadata_path)
        if metadata["errors"] or metadata["skipped_tensors"]:
            updated["errors"] = [*updated.get("errors", []), {
                "file_path": filename, "error": "Legacy normalization range coverage is incomplete",
                "range_errors": metadata["errors"], "skipped_tensors": metadata["skipped_tensors"],
            }]
    except Exception as exc:
        updated["errors"] = [*updated.get("errors", []), _download_error("legacy_normalization.json", exc)]
    manifest_path = checkpoint / "download_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(checkpoint):
        raise ValueError("Download manifest must be a local snapshot file")
    manifest_path.write_text(json.dumps(updated, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return updated


def _safe_name(filename: str) -> PurePosixPath:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or "\\" in filename or not path.parts:
        raise ValueError(f"Invalid repository-relative filename: {filename}")
    return path


def _processor_tensor(filename: str) -> bool:
    return any(fnmatch.fnmatchcase(PurePosixPath(filename).name, pattern) for pattern in (
        "policy_preprocessor_step_*.safetensors", "policy_postprocessor_step_*.safetensors",
    ))


def _weight_file(filename: str) -> bool:
    """Never let a processor reference authorize a conventional model-weight file."""
    name = PurePosixPath(filename).name.lower()
    return (name in {"model.safetensors", "pytorch_model.bin"}
            or bool(re.fullmatch(r"(?:model|pytorch_model)[-_].*\.(?:safetensors|bin)", name)))


def _download_error(filename: str | None, exc: Exception) -> dict[str, Any]:
    status = getattr(getattr(exc, "response", None), "status_code", getattr(exc, "code", None))
    return {"file_path": filename, "error": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__, "http_status": status if isinstance(status, int) else None}


def download_lightweight_snapshot(repo_id: str, snapshot_dir: Path, api: Any | None = None) -> dict[str, Any]:
    """Populate a fresh snapshot using pinned, size-checked public Hub files."""
    api = api if api is not None else HfApi(token=False)
    root = Path(snapshot_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("A Hub snapshot directory must be empty to avoid mixing revisions")
    manifest: dict[str, Any] = {
        "repo_id": repo_id, "revision": None, "snapshot_dir": str(root), "checkpoint_dir": str(root),
        "downloaded_files": [], "skipped_files": [], "header_metadata": {}, "errors": [],
        "model_weights_skipped": False, "artifact_bytes_downloaded": 0, "header_bytes_fetched": 0,
        "critical_download_errors": False,
        "repo_files": [], "repo_files_complete": False,
        "full_model_weights_downloaded": False,
    }
    try:
        info = api.model_info(repo_id, files_metadata=True, timeout=20)
        manifest["repo_access"] = {"private": bool(getattr(info, "private", False)), "gated": getattr(info, "gated", False)}
        if getattr(info, "private", False):
            raise ValueError("Only public checkpoints are scanned")
        revision = info.sha
        if not isinstance(revision, str) or not revision:
            raise ValueError("The Hub did not return an immutable revision")
        manifest["revision"] = revision
        files = sorted({_safe_name(name).as_posix() for name in api.list_repo_files(repo_id, revision=revision)})
        sizes = {sibling.rfilename: sibling.size for sibling in info.siblings or []}
        manifest.update({"repo_files": files, "repo_files_complete": True, "repo_file_sizes": sizes})
        roots = {PurePosixPath(name).parent for name in files if PurePosixPath(name).name in _PROCESSOR_CONFIGS | {"config.json"}}

        def root_priority(path: PurePosixPath) -> tuple[int, int, str]:
            names = {PurePosixPath(name).name for name in files if PurePosixPath(name).parent == path}
            pair = bool(names & {"policy_preprocessor.json", "preprocessor_config.json"}) and bool(
                names & {"policy_postprocessor.json", "postprocessor_config.json"})
            return (0 if pair else 1 if names & _PROCESSOR_CONFIGS else 2, len(path.parts), str(path))

        selected_root = min(roots, key=root_priority) if roots else PurePosixPath(".")
        manifest["checkpoint_subdir"] = selected_root.as_posix()
        checkpoint = root / selected_root
        checkpoint.mkdir(parents=True, exist_ok=True)
        manifest["checkpoint_dir"] = str(checkpoint)
        manifest["checkpoint_selection"] = "processor_pair_first_then_shortest_lexicographic_config_directory"
    except Exception as exc:
        manifest["errors"].append(_download_error(None, exc))
        manifest["critical_download_errors"] = True
        (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def skip(filename: str, reason: str, size: int | None, critical: bool = False) -> None:
        manifest["skipped_files"].append({"file_path": filename, "reason": reason, "size": size})
        if critical:
            manifest["critical_download_errors"] = True

    def download(filename: str, critical: bool = True) -> None:
        if filename in manifest["downloaded_files"]:
            return
        if _weight_file(filename):
            skip(filename, "neural_weights_not_downloaded", sizes.get(filename), critical=critical)
            return
        size = sizes.get(filename)
        try:
            if size is None:
                size = get_hf_file_metadata(hf_hub_url(repo_id, filename, revision=revision), token=False, timeout=15).size
            if type(size) is not int or size < 0 or size > MAX_DOWNLOAD_BYTES:
                skip(filename, "unknown_size_or_exceeds_5_MiB_limit", size, critical=critical)
                return
            if manifest["artifact_bytes_downloaded"] + size > MAX_SNAPSHOT_BYTES:
                skip(filename, "snapshot_byte_budget", size, critical=critical)
                return
            downloaded = Path(hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision, local_dir=root, token=False,
            ))
            if not downloaded.is_file() or not downloaded.resolve().is_relative_to(root):
                raise ValueError("The Hub download did not produce the expected local snapshot file")
            actual_size = downloaded.stat().st_size
            manifest["artifact_bytes_downloaded"] += actual_size
            if actual_size != size or actual_size > MAX_DOWNLOAD_BYTES:
                raise ValueError("Downloaded size disagrees with the pinned Hub metadata or file byte limit")
            manifest["downloaded_files"].append(filename)
        except Exception as exc:
            manifest["critical_download_errors"] |= critical
            manifest["errors"].append(_download_error(filename, exc))

    selected_files = [name for name in files if PurePosixPath(name).parent == selected_root]
    for filename in files:
        if PurePosixPath(filename).name in _JSON_NAMES or filename.endswith(".safetensors.index.json"):
            required = PurePosixPath(filename).parent == selected_root and PurePosixPath(filename).name in _PROCESSOR_CONFIGS
            download(filename, critical=required)
    processor_files = {name for name in selected_files if _processor_tensor(name)}
    for filename in list(manifest["downloaded_files"]):
        if filename.endswith(".json") and PurePosixPath(filename).parent == selected_root:
            try:
                config = json.loads((root / filename).read_text(encoding="utf-8"))
                for step in config.get("steps", []) if isinstance(config, dict) else []:
                    state_file = step.get("state_file")
                    if isinstance(state_file, str) and state_file.endswith(".safetensors"):
                        referenced = (selected_root / _safe_name(state_file)).as_posix()
                        if referenced in files and not _weight_file(referenced):
                            processor_files.add(referenced)
                        elif referenced in files:
                            skip(referenced, "processor_references_neural_weights", sizes.get(referenced), critical=True)
            except (OSError, ValueError, AttributeError, TypeError):
                # Invalid processor JSON is evidence for preflight, not a network failure.
                pass
    # A processor cannot authorize a heavy model download through state_file.
    model_name = (selected_root / "model.safetensors").as_posix()
    processor_files.discard(model_name)
    for filename in sorted(processor_files):
        download(filename)
    for filename in selected_files:
        name = PurePosixPath(filename).name
        if filename.endswith(".npz"):
            skip(filename, "sample_data_not_requested", sizes.get(filename))
        if filename in processor_files or not (filename.endswith(".safetensors") or name == "pytorch_model.bin"):
            continue
        size = sizes.get(filename)
        manifest["model_weights_skipped"] = True
        skip(filename, "header_only" if name == "model.safetensors" else "neural_weights_not_downloaded", size)
        if name == "model.safetensors":
            try:
                metadata = fetch_safetensors_header(repo_id, filename, revision, total_size=size)
                sidecar = root / f"{filename}.header.json"
                sidecar.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                manifest["header_metadata"][filename] = str(sidecar.relative_to(root))
                manifest["header_bytes_fetched"] += metadata["bytes_fetched"]
            except Exception as exc:
                manifest["errors"].append(_download_error(filename, exc))
    manifest = enrich_legacy_snapshot(manifest)
    (checkpoint / "download_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _classification(findings: list[DiagnosticFinding]) -> str:
    modern = any(f.check_id == "SCHEMA_GENERATION" and f.details.get("processor_files") for f in findings)
    legacy = any(f.details.get("legacy_buffer_keys") for f in findings)
    return "MIXED_POLICY" if modern and legacy else "MODERN_POLICY" if modern else "LEGACY_POLICY" if legacy else "UNKNOWN"


def _has_local_processor_pair(checkpoint: Path) -> bool:
    """Require both locally readable native processor configs before bypassing repo labels."""
    root = checkpoint.resolve()

    def usable(filename: str) -> bool:
        path = checkpoint / filename
        if not path.is_file() or not path.resolve().is_relative_to(root):
            return False
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(config, dict) and isinstance(config.get("steps"), list)

    return (any(usable(name) for name in ("policy_preprocessor.json", "preprocessor_config.json"))
            and any(usable(name) for name in ("policy_postprocessor.json", "postprocessor_config.json")))


def _repository_classification(manifest: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Use access responses or positive artifact evidence, never repository names."""
    checkpoint = Path(manifest["checkpoint_dir"])
    if _has_local_processor_pair(checkpoint):
        return None, {}
    access = manifest.get("repo_access", {})
    denied = [error for error in manifest.get("errors", []) if (
        error.get("http_status") in (401, 403) or error.get("error_type") == "GatedRepoError"
        or re.search(r"(?:\b(?:401|403)\s+(?:Client Error|Unauthorized|Forbidden)|\bHTTP Error\s+(?:401|403)\b)",
                     error.get("error", ""), flags=re.IGNORECASE)
    )]
    if access.get("private") or access.get("gated") or denied:
        return "GATED_OR_PRIVATE", {"reason": "gated_or_private_access", "access": access,
                                     "access_errors": denied, "cached_evidence": manifest.get("snapshot_reused", False)}
    config_path = checkpoint / "config.json"
    if config_path.is_file() and config_path.resolve().is_relative_to(checkpoint.resolve()):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            outputs = config.get("output_features") if isinstance(config, dict) else None
            if isinstance(outputs, dict) and outputs and all(
                isinstance(feature, dict) and feature.get("type") == "REWARD" for feature in outputs.values()
            ):
                return "NON_POLICY_REPO", {"reason": "reward_outputs_without_action", "config_path": str(config_path),
                                           "model_type": config.get("type"), "output_features": outputs}
        except (OSError, ValueError):
            pass
    files = manifest.get("repo_files", [])
    if manifest.get("repo_files_complete") and isinstance(files, list):
        assets = [name for name in files if PurePosixPath(name).suffix.lower() in {".xml", ".urdf", ".mjcf", ".stl", ".obj"}]
        policy_names = _PROCESSOR_CONFIGS | {"config.json", "train_config.json"}
        weights = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pkl"}
        if assets and not any(PurePosixPath(name).name in policy_names or PurePosixPath(name).suffix.lower() in weights for name in files):
            return "NON_POLICY_REPO", {"reason": "simulation_assets_without_policy_artifacts", "file_count": len(files),
                                       "asset_file_count": len(assets), "example_asset_files": assets[:10],
                                       "revision": manifest.get("revision")}
        documentation_names = {
            ".gitattributes", "readme", "readme.md", "readme.rst", "readme.txt",
            "license", "license.md", "license.rst", "license.txt",
            "licence", "licence.md", "licence.rst", "licence.txt",
        }
        if files and all(PurePosixPath(name).name.lower() in documentation_names for name in files):
            return "NON_POLICY_REPO", {"reason": "documentation_only_without_policy_artifacts",
                                       "file_count": len(files), "documentation_files": files,
                                       "revision": manifest.get("revision")}
    return None, {}


def _stat_tensor_payloads(checkpoint: Path) -> list[dict[str, Any]]:
    """Hash individual saved stat bytes, never a full neural-model payload.

    Byte hashes describe serialized artifacts, including unused dataset stats;
    they do not establish that a normalization feature is active at runtime.
    """
    stats = {"mean", "std", "min", "max", "q01", "q99", "q10", "q90"}
    payloads: list[dict[str, Any]] = []
    state_paths: set[Path] = set()
    for config_path in sorted(checkpoint.glob("*.json")):
        if config_path.stat().st_size > MAX_DOWNLOAD_BYTES:
            continue
        try:
            config = json.loads(_local_file(checkpoint, config_path.name).read_text(encoding="utf-8"))
            for entry in config.get("steps", []) if isinstance(config, dict) else []:
                filename = entry.get("state_file") if isinstance(entry, dict) else None
                if isinstance(filename, str) and not _weight_file(PurePosixPath(filename).name):
                    state_paths.add(_local_file(checkpoint, filename))
        except (OSError, ValueError, TypeError):
            continue
    for path in sorted(state_paths):
        if path.stat().st_size > MAX_DOWNLOAD_BYTES:
            continue
        try:
            with path.open("rb") as stream:
                length_bytes = stream.read(8)
                header_len = int.from_bytes(length_bytes, "little")
                if len(length_bytes) != 8 or not 0 < header_len <= MAX_HEADER_BYTES:
                    continue
                tensors = parse_safetensors_header(length_bytes + stream.read(header_len), path.stat().st_size)
                for key, descriptor in tensors.items():
                    if key.rsplit(".", 1)[-1] not in stats:
                        continue
                    start, end = descriptor["data_offsets"]
                    if end - start > MAX_LEGACY_TENSOR_BYTES:
                        continue
                    stream.seek(8 + header_len + start)
                    raw = stream.read(end - start)
                    if len(raw) != end - start:
                        continue
                    payloads.append({"file_path": str(path), "tensor_key": key,
                                     "dtype": descriptor["dtype"], "shape": descriptor["shape"],
                                     "payload_sha256": hashlib.sha256(raw).hexdigest(),
                                     "digest_scope": "individual_tensor_payload", "bytes": len(raw),
                                     "absolute_range": [8 + header_len + start, 8 + header_len + end]})
        except (OSError, ValueError, TypeError):
            continue
    if (checkpoint / "legacy_normalization.json").is_file():
        try:
            path, state, metadata, _ = _verified_legacy_values(checkpoint)
            for key in sorted(state):
                descriptor = metadata["tensors"][key]
                payloads.append({"file_path": str(path), "tensor_key": key,
                                 "dtype": descriptor["dtype"], "shape": descriptor["shape"],
                                 "payload_sha256": descriptor["payload_sha256"],
                                 "digest_scope": "individual_tensor_payload",
                                 "bytes": descriptor["data_offsets"][1] - descriptor["data_offsets"][0],
                                 "source_file": metadata["source_file"],
                                 "absolute_range": descriptor["absolute_range"]})
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return payloads


def _finding_category(finding: dict[str, Any]) -> str:
    """Separate migration and conditional precision advice from artifact failures."""
    if finding.get("report_category") == "base_model_prerequisite":
        return "base_model_prerequisite"
    if finding.get("check_id") == "SCHEMA_GENERATION_MISMATCH":
        return "version_compatibility"
    details = finding.get("details") or {}
    if (finding.get("check_id") == "FP16_NON_FINITE_RISK" and finding.get("severity") == "WARNING"
            and (details.get("fp16_execution_conditional")
                 or details.get("requires_legacy_buffer_half_cast_or_modern_migration")
                 or details.get("normalization_implementation") == "legacy Normalize")):
        return "conditional_precision"
    return "artifact_contract"


def _included_report_finding(finding: dict[str, Any]) -> bool:
    return (finding.get("status") in {"FAIL", "CONFLICT"}
            or (_finding_category(finding) == "conditional_precision"
                and finding.get("severity") == "WARNING"))


def _finding_groups(results: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {
        "artifact_contract": {}, "version_compatibility": {}, "conditional_precision": {},
        "base_model_prerequisite": {},
    }
    for repo in results:
        for finding in repo["findings"]:
            if not _included_report_finding(finding):
                continue
            repos = groups[_finding_category(finding)].setdefault(finding["check_id"], [])
            if repo["repo_id"] not in repos:
                repos.append(repo["repo_id"])
    return groups


def _annotate_base_prerequisites(repo_id: str, findings: list[dict[str, Any]]) -> None:
    """Report unbound execution prerequisites without claiming corrupt base weights."""
    unbound = any(f["check_id"] == "UNBOUND_BASE_MODEL_STATS" for f in findings)
    for finding in findings:
        details = finding.get("details") or {}
        missing_stats = (finding["check_id"] == "FEATURE_CONTRACT"
                         and "available_buffer_keys" in details
                         and finding.get("summary", "").startswith("Required normalization buffer "))
        if finding["check_id"] == "UNBOUND_BASE_MODEL_STATS" or (
            missing_stats and (unbound or repo_id in _INTENTIONAL_BASE_REPOS)
        ):
            finding["report_category"] = "base_model_prerequisite"
            finding["report_interpretation"] = (
                "Target dataset statistics must be bound before execution; this is a deployment prerequisite, "
                "not evidence of defective pretrained weights. Fine-tuning history is unverified."
            )


def _artifact_families(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count payload-independent defect signatures separately from binding prerequisites."""
    families: dict[str, dict[str, Any]] = {}
    for repo in results:
        for finding in repo["findings"]:
            category = _finding_category(finding)
            if finding["status"] not in {"FAIL", "CONFLICT"} or category not in {
                "artifact_contract", "base_model_prerequisite",
            }:
                continue
            details = finding.get("details") or {}
            if category == "base_model_prerequisite":
                identity, label, kind = "dataset_stat_binding", "Dataset-stat binding prerequisite", "prerequisite"
            elif finding["check_id"] == "GIT_LFS_POINTER_ARTIFACT":
                identity = "git_lfs_pointer:" + str(details.get("lfs_oid_sha256") or "unavailable")
                label, kind = "Git LFS pointer in place of tensor data", "defect_or_numeric_risk"
            else:
                tensors = _finding_payloads(repo, finding)
                evidence = sorted({t["payload_sha256"] for t in tensors}) or [
                    (finding.get("provenance") or {}).get("sha256_prefix", "unavailable")]
                identity = json.dumps([finding["check_id"], details.get("normalization_mode"), evidence], sort_keys=True)
                label, kind = finding["check_id"], "defect_or_numeric_risk"
            family = families.setdefault(identity, {"label": label, "kind": kind, "check_ids": [],
                                                   "repo_ids": [], "evidence_identity": identity})
            if finding["check_id"] not in family["check_ids"]:
                family["check_ids"].append(finding["check_id"])
            if repo["repo_id"] not in family["repo_ids"]:
                family["repo_ids"].append(repo["repo_id"])
    return {"family_count": len(families),
            "defect_or_numeric_risk_signatures": sum(f["kind"] == "defect_or_numeric_risk" for f in families.values()),
            "prerequisite_families": sum(f["kind"] == "prerequisite" for f in families.values()),
            "repository_count": len({repo for f in families.values() for repo in f["repo_ids"]}),
            "families": list(families.values())}


def _individual_payload_groups(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for repo in results:
        for tensor in repo.get("stat_tensor_payloads", []):
            groups.setdefault(tensor["payload_sha256"], []).append({
                "repo_id": repo["repo_id"], "revision": repo.get("revision"), **tensor,
            })
    return groups


def _finding_payloads(repo: dict[str, Any], finding: dict[str, Any]) -> list[dict[str, Any]]:
    details, provenance = finding.get("details") or {}, finding.get("provenance") or {}
    primary = details.get("source_tensor_key") or provenance.get("tensor_key")
    keys = {primary} if primary else set()
    upper = details.get("upper_buffer_key")
    if upper:
        keys.add(upper)
        if primary:
            keys.add(primary.rsplit(".", 1)[0] + "." + upper.rsplit(".", 1)[-1])
    available = set(details.get("available_buffer_keys", [])) | set(details.get("matching_prefixed_keys", []))
    candidates = [tensor for tensor in repo.get("stat_tensor_payloads", [])
                  if tensor["file_path"] == provenance.get("file_path")]
    if finding.get("check_id") == "NORMALIZATION_CHAIN_NON_FINITE" and details.get("feature_key"):
        return [tensor for tensor in candidates
                if tensor["tensor_key"].startswith(details["feature_key"] + ".")]
    matching = [tensor for tensor in candidates if tensor["tensor_key"] in keys]
    return matching or [tensor for tensor in candidates if tensor["tensor_key"] in available]


def _audit_snapshot(manifest: dict[str, Any], std_threshold: float) -> dict[str, Any]:
    checkpoint = Path(manifest["checkpoint_dir"])
    classification, evidence = _repository_classification(manifest)
    findings: list[DiagnosticFinding] = []
    if classification is not None:
        findings.append(DiagnosticFinding(
            "REPO_CLASSIFICATION", DiagnosticStatus.PASS if classification == "NON_POLICY_REPO" else DiagnosticStatus.UNKNOWN,
            Severity.INFO, "Repository is outside action-policy checkpoint scope." if classification == "NON_POLICY_REPO" else
            "Gated/private access prevents an unauthenticated checkpoint audit.", evidence,
        ))
    elif manifest.get("critical_download_errors"):
        findings.append(DiagnosticFinding(
            "HUB_SNAPSHOT", DiagnosticStatus.UNKNOWN, Severity.WARNING,
            "Download coverage is incomplete; missing local files are not attributed to checkpoint defects.",
            {"errors": manifest.get("errors", []), "skipped_files": manifest.get("skipped_files", [])},
        ))
    else:
        access = manifest.get("repo_access", {})
        access_warning = _has_local_processor_pair(checkpoint) and (
            access.get("private") or access.get("gated")
        )
        if manifest.get("errors") or access_warning:
            findings.append(DiagnosticFinding(
                "HUB_SNAPSHOT", DiagnosticStatus.UNKNOWN, Severity.WARNING,
                "Hub access or optional file metadata is restricted; available processor artifacts are audited."
                if access_warning else
                "Some optional metadata or weight headers were unavailable; available processor artifacts are audited.",
                {"errors": manifest.get("errors", []), "repo_access": access} if access_warning else
                {"errors": manifest["errors"]},
            ))
        for stage, check in (("preflight", lambda: run_preflight(checkpoint, std_threshold)),
                             ("temporal", lambda: run_temporal_audit(checkpoint))):
            try:
                result = check()
                findings.extend(result.findings if hasattr(result, "findings") else result)
            except Exception as exc:
                findings.append(DiagnosticFinding(
                    "HUB_AUDIT", DiagnosticStatus.UNKNOWN, Severity.WARNING,
                    f"The {stage} audit could not complete: {type(exc).__name__}: {exc}", {"stage": stage},
                ))
    serialized = [finding.to_dict() for finding in findings]
    _annotate_base_prerequisites(manifest["repo_id"], serialized)
    return {**manifest, "classification": classification or _classification(findings),
            "classification_evidence": evidence, "findings": serialized,
            "stat_tensor_payloads": _stat_tensor_payloads(checkpoint)}


def _report(results: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    buckets = {category: sum(repo["classification"] == category for repo in results) for category in _BUCKETS}
    summary = {
        "total_repos": len(results), "buckets": buckets,
        **{key: buckets[category] for key, category in (
            ("modern", "MODERN_POLICY"), ("legacy", "LEGACY_POLICY"), ("mixed", "MIXED_POLICY"),
            ("non_policy", "NON_POLICY_REPO"), ("gated_or_private", "GATED_OR_PRIVATE"), ("unknown", "UNKNOWN"))},
        "errors": sum(bool(repo.get("errors")) or repo.get("critical_download_errors", False) for repo in results),
        "issues": {check: list(dict.fromkeys(repo["repo_id"] for repo in results if any(
            f["check_id"] == check and f["status"] in ("FAIL", "CONFLICT") for f in repo["findings"]))) for check in ISSUE_CHECKS},
        "advisories": {check: list(dict.fromkeys(repo["repo_id"] for repo in results if any(
            f["check_id"] == check and f["status"] in {"PASS", "UNKNOWN"} and f.get("severity") == "WARNING"
            for f in repo["findings"]))) for check in (
                "CAMERA_KEY_CONTRACT_MISMATCH", "FP16_NON_FINITE_RISK", "INLINE_STAT_OVERRIDE_ADVISORY")},
        "artifact_bytes_downloaded": sum(repo.get("artifact_bytes_downloaded", 0) for repo in results),
        "header_bytes_fetched": sum(repo.get("header_bytes_fetched", 0) for repo in results),
        "normalization_bytes_fetched": sum(repo.get("normalization_bytes_fetched", 0) for repo in results),
        "snapshots_reused": sum(repo.get("snapshot_reused", False) for repo in results),
        "numeric_unknown_repos": sum(any(f["status"] == "UNKNOWN" and f["check_id"] in {
            "PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "INVALID_NORMALIZATION_STATS",
        } for f in repo["findings"]) for repo in results),
        "temporal_unknown_repos": sum(any(f["status"] == "UNKNOWN" and
            f["check_id"] == "TEMPORAL_ANCHOR_MISMATCH" for f in repo["findings"]) for repo in results),
        "normalization_chain_unknown_repos": sum(any(f["status"] == "UNKNOWN" and
            f["check_id"] == "NORMALIZATION_CHAIN_PROBE" for f in repo["findings"]) for repo in results),
    }
    verified_snapshots = verified_tensors = verified_bytes = 0
    for repo in results:
        checkpoint = Path(repo["checkpoint_dir"])
        if not (checkpoint / "legacy_normalization.json").is_file():
            continue
        try:
            _, state, metadata, _ = _verified_legacy_values(checkpoint)
            verified_snapshots += 1
            verified_tensors += len(state)
            verified_bytes += sum(metadata["tensors"][key]["absolute_range"][1] -
                                  metadata["tensors"][key]["absolute_range"][0] for key in state)
        except (OSError, ValueError, KeyError, TypeError):
            # Incomplete/corrupt range coverage already has explicit local findings.
            continue
    summary["cached_legacy_range_evidence"] = {
        "snapshots": verified_snapshots, "tensors": verified_tensors, "payload_bytes": verified_bytes,
        "new_payload_bytes_this_run": summary["normalization_bytes_fetched"],
    }
    finding_groups = _finding_groups(results)
    summary["finding_categories"] = {category: {
        "issues": issues, "repository_count": len({repo for repos in issues.values() for repo in repos}),
        "finding_count": sum(_included_report_finding(f) and _finding_category(f) == category
                             for repo in results for f in repo["findings"]),
    } for category, issues in finding_groups.items()}
    summary["artifact_finding_families"] = _artifact_families(results)
    summary["temporal_coverage"] = {
        "single_frame_or_non_relative_repos": sum(any(
            f["check_id"] == "TEMPORAL_ANCHOR_MISMATCH" and f["status"] == "PASS" and (
                (f.get("details") or {}).get("n_obs_steps") == 1
                or f.get("summary", "").startswith("No enabled relative-action conversion")
            ) for f in repo["findings"]) for repo in results),
        "unknown_repos": summary["temporal_unknown_repos"],
        "not_evaluated_repos": sum(not any(f["check_id"] == "TEMPORAL_ANCHOR_MISMATCH"
                                          for f in repo["findings"]) for repo in results),
        "multi_step_relative_repos_exercised": sum(any(
            f["check_id"] == "TEMPORAL_ANCHOR_MISMATCH"
            and (f.get("details") or {}).get("n_obs_steps", 0) > 1
            and (f.get("details") or {}).get("probe_state_slice") is not None
            for f in repo["findings"]) for repo in results),
        "relative_action_policy_repos": sum(any(
            f["check_id"] == "TEMPORAL_ANCHOR_MISMATCH" and f["status"] != "UNKNOWN"
            and (f.get("details") or {}).get("step_class") == "RelativeActionsProcessorStep"
            for f in repo["findings"]) for repo in results),
        "non_relative_policy_repos": sum(any(
            f["check_id"] == "TEMPORAL_ANCHOR_MISMATCH" and f["status"] == "PASS"
            and f.get("summary", "").startswith("No enabled relative-action conversion")
            for f in repo["findings"]) for repo in results),
    }
    payloads = _individual_payload_groups(results)
    summary["individual_tensor_payloads"] = {
        "references": sum(len(refs) for refs in payloads.values()), "unique_payloads": len(payloads),
        "shared_across_repositories": sum(len({ref["repo_id"] for ref in refs}) > 1 for refs in payloads.values()),
        "digest_scope": "individual_tensor_payload",
    }
    canonical = next((repo for repo in results if repo["repo_id"] == "lerobot/smolvla_base"), None)
    if canonical is not None:
        canonical_payloads = {t["payload_sha256"] for t in canonical.get("stat_tensor_payloads", [])
                              if "action" in t["tensor_key"]}
        canonical_finding = next((f for f in canonical["findings"]
                                  if f["check_id"] == "UNBOUND_BASE_MODEL_STATS"), {})
        summary["shared_base_stat_evidence"] = {
            "canonical_repo_id": canonical["repo_id"],
            "canonical_container_sha256_prefix": (canonical_finding.get("provenance") or {}).get("sha256_prefix"),
            "canonical_action_payload_sha256": sorted(canonical_payloads),
            "repo_ids": [repo["repo_id"] for repo in results if canonical_payloads
                         and any(f["check_id"] == "UNBOUND_BASE_MODEL_STATS" for f in repo["findings"])
                         and canonical_payloads.issubset({t["payload_sha256"] for t in repo.get("stat_tensor_payloads", [])})],
            "fine_tuning_history_verified": False,
        }
    return {"report_version": 7, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": selection, "summary": summary, "repos": results,
            "tensor_payload_groups": payloads}


def rescan_cached_reports(
    report_paths: Iterable[Path], limit: int = 30, std_threshold: float = 1e-4,
    repo_file_evidence: Path | None = None, progress: Callable[[str], None] | None = None,
    fetch_legacy_stats: bool = False,
) -> dict[str, Any]:
    """Re-audit pinned snapshots; optional HTTP ranges run before local validators."""
    if type(limit) is not int or not 1 <= limit <= MAX_REPOSITORIES or not math.isfinite(std_threshold) or std_threshold <= 0:
        raise ValueError("limit must be 1..100 and std_threshold must be positive and finite")
    paths = [Path(path) for path in report_paths]
    evidence = {}
    if repo_file_evidence is not None:
        for item in json.loads(Path(repo_file_evidence).read_text(encoding="utf-8")):
            if (isinstance(item, dict) and isinstance(item.get("files"), list)
                    and item.get("file_count", len(item["files"])) == len(item["files"])):
                evidence[(item.get("repo_id"), item.get("revision"))] = item["files"]
    snapshots, seen, original_discovery = [], set(), None
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        source_selection = document.get("selection", {})
        if original_discovery is None:
            original_discovery = source_selection.get("original_discovery") or (
                source_selection if source_selection.get("strategy") else None)
        for repo in document["repos"]:
            identity = (repo["repo_id"], repo.get("revision"))
            if identity in seen or len(snapshots) >= limit:
                continue
            seen.add(identity)
            manifest = {key: value for key, value in repo.items() if key not in ("findings", "classification", "classification_evidence")}
            # Older reports use project-relative paths. External snapshots in
            # newer reports are relative to the report itself, never redacted
            # display strings or the caller's current working directory.
            for field in ("snapshot_dir", "checkpoint_dir"):
                cached = Path(manifest[field])
                if not cached.is_absolute():
                    base_kind = manifest.get(field.removesuffix("_dir") + "_path_base", "project")
                    if base_kind == "report_directory":
                        base = path.resolve().parent
                    elif base_kind == "project":
                        base = Path(__file__).resolve().parents[2]
                    else:
                        raise ValueError(f"Unsupported cached {field} path base: {base_kind}")
                    manifest[field] = str((base / cached).resolve())
            manifest.update({"snapshot_reused": True, "source_report": str(path.resolve()),
                             "source_report_generated_at_utc": document.get("generated_at_utc"),
                             "cached_artifact_bytes": repo.get("cached_artifact_bytes", repo.get("artifact_bytes_downloaded", 0)),
                             "cached_header_bytes": repo.get("cached_header_bytes", repo.get("header_bytes_fetched", 0)),
                             "cached_normalization_bytes": repo.get("cached_normalization_bytes", repo.get("normalization_bytes_fetched", 0)),
                             "artifact_bytes_downloaded": 0, "header_bytes_fetched": 0, "normalization_bytes_fetched": 0})
            if identity in evidence:
                manifest.update({"repo_files": evidence[identity], "repo_files_complete": True,
                                 "repo_file_evidence": str(Path(repo_file_evidence).resolve())})
            snapshots.append(manifest)
    results = []
    for index, manifest in enumerate(snapshots, 1):
        if progress:
            progress(f"[{index}/{len(snapshots)}] cached {manifest['repo_id']} @ {(manifest.get('revision') or 'unavailable')[:12]}")
        if fetch_legacy_stats:
            manifest = enrich_legacy_snapshot(manifest)
        results.append(_audit_snapshot(manifest, std_threshold))
        if progress:
            network = f"{manifest.get('normalization_bytes_fetched', 0)} normalization range bytes" if fetch_legacy_stats else "no network"
            progress(f"  {results[-1]['classification']}; {len(results[-1]['findings'])} findings; {network}")
    return _report(results, {"mode": "cached", "limit": limit, "source_reports": [str(path.resolve()) for path in paths],
                             "network_used": fetch_legacy_stats, "fetch_legacy_stats": fetch_legacy_stats,
                             "original_discovery": original_discovery})


def _family_hint(model: Any) -> str | None:
    """Rank discovery metadata; this is never a processor compatibility verdict."""
    text = " ".join([model.id, *(getattr(model, "tags", None) or [])]).lower()
    for family in POLICY_FAMILIES:
        pattern = {
            "act": r"(?<![a-z0-9])act\d*(?![a-z])",
            "pi0": r"(?<![a-z0-9])pi[-_]?0\d*(?![a-z])",
            "vqbet": r"(?<![a-z0-9])vq[-_]?bet(?![a-z])",
            "tdmpc": r"(?<![a-z0-9])td[-_]?mpc\d*(?![a-z])",
            "smolvla": r"smol[-_]?vla",
        }.get(family, family)
        if re.search(pattern, text):
            return family
    return None


def _asset_listing(model: Any) -> bool:
    names = [getattr(sibling, "rfilename", "") for sibling in (getattr(model, "siblings", None) or [])]
    assets = any(PurePosixPath(name).suffix.lower() in {".xml", ".urdf", ".mjcf", ".stl", ".obj"} for name in names)
    artifacts = any(PurePosixPath(name).name in _PROCESSOR_CONFIGS | {"config.json", "train_config.json"}
                    or PurePosixPath(name).suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
                    for name in names)
    return assets and not artifacts


def select_targeted_repositories(api: Any, limit: int) -> tuple[list[str], dict[str, Any]]:
    """Discover a bounded pool tagged lerobot and balance architecture search hints.

    Names and tags select candidates only. Downloaded artifacts determine their
    policy generation and findings. Mesh-only listing evidence is filtered here.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_REPOSITORIES:
        raise ValueError("limit must be between 1 and 100")
    queries = [(seed.split("/", 1)[1], 25) for seed in _TARGET_SEEDS]
    queries += [(family, 100) for family in POLICY_FAMILIES]
    queries += [(None, 400)]
    pool: dict[str, Any] = {}
    filtered, query_evidence = [], []
    for search, count in queries:
        arguments = {"filter": "lerobot", "limit": count, "full": True}
        if search is not None:
            arguments["search"] = search
        returned = 0
        for model in api.list_models(**arguments):
            returned += 1
            if getattr(model, "private", False):
                continue
            if _asset_listing(model):
                if model.id not in filtered:
                    filtered.append(model.id)
                continue
            pool.setdefault(model.id, model)
        query_evidence.append({**arguments, "returned": returned})
    selected = [seed for seed in _TARGET_SEEDS if seed in pool][:limit]
    queues = {family: [repo_id for repo_id, model in pool.items() if _family_hint(model) == family]
              for family in POLICY_FAMILIES}
    while len(selected) < limit and any(queues.values()):
        for family in POLICY_FAMILIES:
            while queues[family] and queues[family][0] in selected:
                queues[family].pop(0)
            if queues[family] and len(selected) < limit:
                selected.append(queues[family].pop(0))
    for repo_id in pool:
        if len(selected) >= limit:
            break
        if repo_id not in selected:
            selected.append(repo_id)
    hints = {repo_id: _family_hint(pool[repo_id]) for repo_id in selected}
    return selected, {
        "strategy": "tagged_family_round_robin_then_tagged_fallback",
        "queries": query_evidence, "candidate_count": len(pool),
        "filtered_asset_repositories": filtered, "verified_seed_repositories": [seed for seed in _TARGET_SEEDS if seed in pool],
        "family_hints": hints,
        "family_hint_counts": {family: sum(hint == family for hint in hints.values()) for family in POLICY_FAMILIES},
        "family_hint_scope": "Discovery names/tags only; checkpoint artifacts establish policy identity.",
    }


def scan_hub(
    repo_ids: Iterable[str] | None = None, limit: int = 30, snapshot_dir: Path | None = None,
    std_threshold: float = 1e-4, api: Any | None = None, progress: Callable[[str], None] | None = None,
    cached_reports: Iterable[Path] | None = None, repo_file_evidence: Path | None = None,
    targeted: bool = False,
    fetch_legacy_stats: bool = False,
) -> dict[str, Any]:
    """Scan at most one hundred public repos; network I/O finishes before local audits."""
    if type(limit) is not int or not 1 <= limit <= MAX_REPOSITORIES:
        raise ValueError("limit must be between 1 and 100")
    if not math.isfinite(std_threshold) or std_threshold <= 0:
        raise ValueError("std_threshold must be positive and finite")
    if cached_reports is not None:
        if repo_ids is not None or targeted:
            raise ValueError("Choose either cached reports or live discovery")
        return rescan_cached_reports(cached_reports, limit, std_threshold, repo_file_evidence, progress, fetch_legacy_stats)
    api = api if api is not None else HfApi(token=False)
    selection = {"mode": "live", "filter": "lerobot" if repo_ids is None else None, "limit": limit}
    if targeted:
        if repo_ids is not None:
            raise ValueError("Choose targeted discovery or explicit repository IDs")
        if progress:
            progress("Discovering tagged LeRobot candidates across ACT, diffusion, SmolVLA, pi0, VQ-BeT, and TD-MPC...")
        candidates, evidence = select_targeted_repositories(api, limit)
        selection.update(evidence)
    else:
        candidates = repo_ids if repo_ids is not None else (model.id for model in api.list_models(filter="lerobot", limit=limit))
    selected = []
    for repo_id in candidates:
        if repo_id not in selected:
            selected.append(repo_id)
        if len(selected) == limit:
            break
    cache = Path(snapshot_dir) if snapshot_dir is not None else Path.cwd() / ".hub_audit_snapshots"
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for index, repo_id in enumerate(selected, 1):
        if progress:
            progress(f"[{index}/{len(selected)}] {repo_id}")
        prefix = re.sub(r"[^a-zA-Z0-9_.-]", "_", repo_id)[:90] + "--"
        target = Path(tempfile.mkdtemp(prefix=prefix, dir=cache))
        manifest = download_lightweight_snapshot(repo_id, target, api=api)
        results.append(_audit_snapshot(manifest, std_threshold))
        if progress:
            progress(f"  {results[-1]['classification']}; {len(results[-1]['findings'])} findings; "
                     f"{manifest['artifact_bytes_downloaded']} artifact bytes + {manifest['header_bytes_fetched']} header bytes")
    return _report(results, selection)


def _snapshot_report_location(snapshot: PurePath, project: PurePath,
                              report_dir: PurePath) -> tuple[str, str]:
    """Give operational snapshot paths an explicit, portable resolution base."""
    try:
        return snapshot.relative_to(project).as_posix(), "project"
    except ValueError:
        pass
    windows = isinstance(snapshot, PureWindowsPath)
    path_module = ntpath if windows else posixpath
    path_type = PureWindowsPath if windows else PurePosixPath
    try:
        relative = path_module.relpath(str(snapshot), str(report_dir))
    except ValueError as exc:
        raise ValueError("Cannot export a cached snapshot location relative to the report directory; "
                         "place the report and external snapshot on the same filesystem drive.") from exc
    return path_type(relative).as_posix(), "report_directory"


def _portable_report(report: dict[str, Any], *, report_dir: Path | None = None) -> dict[str, Any]:
    """Export private display evidence and recoverable local snapshot locations."""
    project = Path(__file__).resolve().parents[2]
    # Serialized evidence paths use their host's separators. Match a complete
    # project component with either spelling, then emit project-relative POSIX
    # paths without changing arbitrary strings containing backslashes.
    project_pattern = re.compile(
        re.escape(project.as_posix()).replace("/", r"[\\/]") + r"(?:[\\/]|$)",
        re.IGNORECASE if project.drive else 0,
    )

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: convert(child) for key, child in value.items()}
        if isinstance(value, list):
            return [convert(child) for child in value]
        if isinstance(value, str):
            if project_pattern.fullmatch(value):
                return "."
            value, matches = project_pattern.subn("", value)
            if matches:
                value = value.replace("\\", "/")
            # Redact unrelated home paths that may occur inside recorded errors.
            value = re.sub(r"/Users/[^/\s\"'<>]+/", "local-artifacts/", value)
            return re.sub(r"(?:[A-Za-z]:)?[\\/]Users[\\/][^\\/\s\"'<>]+[\\/]", "local-artifacts/", value)
        return value

    exported = convert(report)
    for original, repo in zip(report.get("repos", []), exported.get("repos", []), strict=True):
        repo["checkpoint_id"] = f"{repo['repo_id']}@{repo.get('revision') or 'unavailable'}"
        if report_dir is None:
            continue
        for field in ("snapshot_dir", "checkpoint_dir"):
            original_path = original.get(field)
            if not isinstance(original_path, str) or not Path(original_path).is_absolute():
                continue
            # Generic home-path redaction is appropriate for displayed errors
            # and provenance, but must never destroy a cached snapshot locator.
            location, base_kind = _snapshot_report_location(
                Path(original_path).resolve(), project, Path(report_dir).resolve())
            repo[field] = location
            base_field = field.removesuffix("_dir") + "_path_base"
            if base_kind == "project":
                repo.pop(base_field, None)  # Preserve the legacy project-relative contract.
            else:
                repo[base_field] = base_kind
    return exported


def _maintainer_highlights(report: dict[str, Any], output: Path) -> list[str]:
    """Summarize verified headline evidence, never repository names alone."""
    repositories = {repo["repo_id"]: repo for repo in report["repos"]}

    def finding(repo_id: str, check: str) -> dict[str, Any] | None:
        return next((item for item in repositories.get(repo_id, {}).get("findings", [])
                     if item.get("check_id") == check and item.get("status") in {"FAIL", "CONFLICT"}), None)

    def link(repo_id: str) -> str:
        revision = repositories[repo_id].get("revision")
        return f"[{repo_id}](https://huggingface.co/{repo_id}" + (f"/tree/{revision})" if revision else ")")

    def number(value: Any, *, rounded: bool = False) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return f"{value:,.0f}" if rounded else f"{value:.8g}"
        return "not recorded"

    def values(item: dict[str, Any], key: str) -> list[Any]:
        value = item.get("details", {}).get(key)
        return value if isinstance(value, list) else []

    def label(value: str) -> str:
        return html.escape(value, quote=True).replace("|", "&#124;").replace("\n", " ")

    def artifact_link(filename: str) -> str:
        return f"[{filename}]({filename})" if (output / filename).is_file() else f"`{filename}`"

    xarm_id = "jhseon-holiday/vqbet_xarm_holi_test"
    numeric = [(xarm_id, "xArm"), ("HaomingSong/lerobot-pi0-bridge", "Pi0 Bridge"),
               ("iantc104/Diffusion-RPL-UR5-PegInsert-v0", "UR5 PegInsert")]
    rows = []
    for repo_id, name in numeric:
        item = finding(repo_id, "PATHOLOGICAL_STD_GAIN")
        if item is None:
            continue
        details, provenance = item.get("details", {}), item.get("provenance") or {}
        saved, gains = values(item, "saved_values"), values(item, "effective_gains_with_eps")
        indices = provenance.get("violating_indices") or []
        mode = details.get("normalization_mode", "undeclared mode")
        key = details.get("source_tensor_key") or provenance.get("tensor_key") or details.get("feature_key", "statistic")
        evidence = f"`{key}` at {indices}; `{mode}`"
        if saved:
            evidence += f"; saved value {number(saved[0])}"
        if mode == "MEAN_STD" and saved and all(value == 0 for value in saved):
            evidence += "; **zero std**"
        spans = values(item, "saved_span_values")
        if spans and all(value == 0 for value in spans):
            evidence += "; **zero span (min == max)**"
        if gains:
            evidence += f"; gain ≈ **{number(max(gains), rounded=True)}×**"
        outputs = values(item, "probe_output_values")
        if outputs:
            evidence += f"; synthetic +0.20 → ≈{number(outputs[0], rounded=True)} in float32"
        evidence += f"; SHA `{provenance.get('sha256_prefix', 'not recorded')}`"
        rows.append(f"| Numeric risk: {name} | {link(repo_id)} | {evidence} |")

    pointer_id = "y1y2y3/smolvla_base1"
    pointer = finding(pointer_id, "GIT_LFS_POINTER_ARTIFACT")
    if pointer is not None:
        size = pointer.get("details", {}).get("pointer_size_bytes")
        size_text = f"**{size}-byte** " if type(size) is int else ""
        rows.append(f"| Artifact: Git LFS pointer | {link(pointer_id)} | {size_text}Git LFS pointer text replaces processor tensor data; `GIT_LFS_POINTER_ARTIFACT`. |")
    bases = [repo_id for repo_id in ("lerobot/smolvla_base", "lerobot/pi0_old")
             if any(_included_report_finding(item) and _finding_category(item) == "base_model_prerequisite"
                    for item in repositories.get(repo_id, {}).get("findings", []))]
    if bases:
        rows.append(f"| Base-model prerequisite family | {', '.join(link(repo_id) for repo_id in bases)} | Bind target dataset statistics or fine-tune before execution; missing/unbound stats are not defective pretrained weights. |")
    if not rows:
        return []
    lines = ["## Findings at a glance", "", "Verified numeric/artifact signatures and base-model prerequisites from the pinned sample; only findings present in this report are listed.", "",
             "| Family | Pinned checkpoint | Verified artifact evidence |", "| --- | --- | --- |", *rows, ""]
    xarm = finding(xarm_id, "PATHOLOGICAL_STD_GAIN")
    if xarm is not None:
        details, provenance = xarm.get("details", {}), xarm.get("provenance") or {}
        gain = values(xarm, "effective_gains_with_eps")
        stat_key = details.get("source_tensor_key") or provenance.get("tensor_key") or "statistic"
        lines += [f"Compact xArm lineage ({artifact_link('xarm_processor_lineage.mmd')}; a static normalization view):", "", "```mermaid", "flowchart LR",
                  '  x_state["xArm observation.state"] --> x_norm["legacy Normalize: ' + label(details.get("normalization_mode", "undeclared")) + '"]',
                  '  x_norm -->|"BINDS_STATS"| x_stats["' + label(stat_key) + '; SHA ' + label(provenance.get("sha256_prefix", "not recorded")) + '"]',
                  '  x_norm -->|"gain ≈ ' + (number(max(gain), rounded=True) if gain else "not recorded") + 'x"| x_input["policy input"]', "```", ""]
    taek_id = "taekyukim/smolvla_base"
    unbound_candidates = [item for item in repositories.get(taek_id, {}).get("findings", [])
                          if item.get("check_id") == "UNBOUND_BASE_MODEL_STATS"
                          and item.get("status") in {"FAIL", "CONFLICT"}]
    # Reports enumerate postprocessing first. The state/action overview must
    # select the observation-bearing preprocessor, not relabel post step zero.
    unbound = next((item for item in unbound_candidates
                    if "observation.state" in item.get("details", {}).get("declared_feature_keys", [])),
                   unbound_candidates[0] if unbound_candidates else None)
    camera = next((item for item in repositories.get(taek_id, {}).get("findings", [])
                   if item.get("check_id") == "CAMERA_KEY_CONTRACT_MISMATCH"), None)
    if unbound is not None and camera is not None:
        details = unbound.get("details", {})
        camera_details = camera.get("details", {})
        source = ((unbound.get("provenance") or {}).get("file_path") or "").lower()
        role = "preprocessor" if "preprocessor" in source else "postprocessor" if "postprocessor" in source else "normalizer"
        targets = "state/action" if details.get("missing_observation_state_stats") else details.get("feature_key", "feature")
        prefixes = sorted({key.rsplit(".", 1)[0] + ".*" for key in
                           details.get("matching_prefixed_keys", details.get("available_buffer_keys", []))})
        lines += [f"Compact {taek_id} lineage ({artifact_link('taekyukim_smolvla_base_lineage.mmd')}):", "", "```mermaid", "flowchart LR",
                  '  t_stats["' + label(', '.join(prefixes) or "unbound dataset buffers") + '"] -.->|"target statistics unbound"| t_norm["taekyukim/smolvla_base ' + role + '[' + str(details.get("step_index", "?")) + ']"]',
                  '  t_norm -.->|"UNBOUND_BYPASS: ' + label(targets) + ' when present"| t_state["unnormalized tensors"]',
                  '  t_camera["processor: ' + label(', '.join(camera_details.get("processor_visual_keys", []))) + '"] -.->|"camera declaration drift; IDENTITY passes through"| t_policy["policy: ' + label(', '.join(camera_details.get("policy_visual_keys", []))) + '"]', "```", "",
                  "Camera drift is an advisory about downstream input mapping; IDENTITY preprocessing does not throw a camera KeyError.", ""]
    lines += ["Legacy numeric examples use inspected float32 buffer arithmetic; FP16 non-finite risks are conditional advisories. These artifact checks do not establish an observed robot failure.", "", "## Coverage and interpretation", ""]
    return lines


def _collapse_report_appendices(lines: list[str]) -> list[str]:
    """Preserve full evidence inside GitHub-compatible disclosure blocks."""
    appendices = {
        "Exact Hub discovery queries (including seeds)": "Expand discovery queries and seed evidence",
        "Targeted discovery coverage": "Expand policy-family selection details",
        "Shared stat payloads": "Expand individual tensor-payload SHA-256 inventory",
        "Artifact / Processor-Contract Findings": "Expand exact artifact and numeric-risk evidence",
        "Intentional Base Models: Dataset-Stat Binding Prerequisites (Non-Defect)": "Expand base-model prerequisite evidence",
        "Version-Compatibility / Legacy Schema (Requires PR #1452 Migration)": "Expand legacy migration evidence by repository",
        "Conditional FP16 Precision Warnings (Legacy or Unconfigured Modern Half Precision)": "Expand conditional precision advisories",
        "Camera declaration advisories": "Expand camera contract advisory evidence",
        "Repository coverage": "Expand all repository revisions, findings, and coverage gaps",
    }
    rendered, opened = [], False
    for line in lines:
        if line.startswith("## "):
            if opened:
                rendered += ["", "</details>", ""]
                opened = False
            rendered.append(line)
            if line[3:] in appendices:
                rendered += ["", "<details>", f"<summary>{appendices[line[3:]]}</summary>", ""]
                opened = True
        else:
            rendered.append(line)
    if opened:
        rendered += ["", "</details>", ""]
    return rendered


def write_hub_reports(report: dict[str, Any], output_dir: Path, report_name: str = "hub_audit_report") -> tuple[Path, Path]:
    """Write strict JSON and a portable Markdown summary with pinned repo links."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", report_name):
        raise ValueError("report_name must be a plain filename stem")
    output = Path(output_dir)
    report = _portable_report(report, report_dir=output)
    output.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output / f"{report_name}.json", output / f"{report_name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    numeric_unknown = sum(any(f.get("status") == "UNKNOWN" and f.get("check_id") in {
        "PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "INVALID_NORMALIZATION_STATS",
    } for f in repo["findings"]) for repo in report["repos"])
    temporal_unknown = sum(any(f.get("status") == "UNKNOWN" and f.get("check_id") ==
        "TEMPORAL_ANCHOR_MISMATCH" for f in repo["findings"]) for repo in report["repos"])
    chain_unknown = sum(any(f.get("status") == "UNKNOWN" and f.get("check_id") ==
        "NORMALIZATION_CHAIN_PROBE" for f in repo["findings"]) for repo in report["repos"])
    pinned = sum(bool(repo.get("revision")) for repo in report["repos"])
    lines = ["# lerobot-preflight: numerical processor and chronology audit", "", f"Generated: {report['generated_at_utc']}", "",
             "Reference-Free Numerical Processor Auditor & Chronology/Contract Linter for LeRobot Checkpoints. Findings describe serialized artifacts and inspected processor behavior; this is not a standalone ISO safety-compliance guarantee.", "",
             "| Metric | Count |", "| --- | ---: |"]
    families = summary.get("artifact_finding_families") or _artifact_families(report["repos"])
    if families["family_count"]:
        lines[6:6] = [
            f"**{families['family_count']} independent artifact finding families across {families['repository_count']} repositories**: "
            f"{families['defect_or_numeric_risk_signatures']} defect/numeric-risk signatures and "
            f"{families['prerequisite_families']} dataset-stat binding prerequisite family. "
            "Base-model binding requirements are not defective pretrained weights; repeated payloads are not independent defects.", "",
        ]
    for key, label in (("total_repos", "Total repos scanned"), ("modern", "Modern PolicyProcessorPipeline"),
                       ("legacy", "Legacy embedded normalization (PR #1452)"), ("mixed", "Modern + embedded legacy"),
                       ("non_policy", "Non-policy repositories"), ("gated_or_private", "Gated or private"),
                       ("snapshots_reused", "Cached snapshots reused"),
                       ("normalization_bytes_fetched", "Legacy normalization tensor bytes fetched"),
                       ("unknown", "Generation unknown"), ("errors", "Repos with download/coverage errors")):
        lines.append(f"| {label} | {summary.get(key, 0)} |")
    lines += [f"| UNKNOWN numeric coverage (repositories) | {numeric_unknown}/{len(report['repos'])} |",
              f"| UNKNOWN native normalization-chain probes (repositories) | {chain_unknown}/{len(report['repos'])} |",
              f"| UNKNOWN temporal coverage (repositories) | {temporal_unknown}/{len(report['repos'])} |",
              f"| Pinned revisions (full commits in JSON) | {pinned}/{len(report['repos'])} |"]
    cached_ranges = summary.get("cached_legacy_range_evidence")
    if cached_ranges:
        lines += [f"| Cached legacy normalization tensors reconstructed | {cached_ranges['tensors']} |",
                  f"| Cached tensor payload bytes (not new downloads) | {cached_ranges['payload_bytes']:,} |"]
    categories = _finding_groups(report["repos"])
    for category, label in (("artifact_contract", "Artifact / processor-contract findings (unique repositories)"),
                            ("base_model_prerequisite", "Base-model stat-binding prerequisites (unique repositories)"),
                            ("version_compatibility", "Legacy version-compatibility requirements (unique repositories)"),
                            ("conditional_precision", "Conditional FP16 warnings (unique repositories)")):
        lines.append(f"| {label} | {len({repo for repos in categories[category].values() for repo in repos})} |")
    lines += ["", "Base-model prerequisites, legacy migration requirements, and hypothetical FP16 warnings are separate from defect/numeric-risk findings. Category counts overlap. An unbound base checkpoint is not ready for zero-shot execution: its prerequisite findings retain deployment-gate FAIL status without claiming corrupt weights. Standard CLI mode gates on blocking FAIL/CONFLICT; strict mode additionally gates on coverage UNKNOWNs and WARNING advisories. Optional undeclared physical semantics remain informational, even in strict mode."]
    temporal = summary.get("temporal_coverage", {})
    if temporal:
        lines += ["", f"Temporal scope: **{temporal['relative_action_policy_repos']} relative-action policies**, "
                  f"**{temporal['non_relative_policy_repos']} non-relative policies**, "
                  f"**{temporal['not_evaluated_repos']} repositories not evaluated** (including non-policy scope), "
                  f"**{temporal['unknown_repos']} UNKNOWN**, and "
                  f"**{temporal['multi_step_relative_repos_exercised']} multi-step relative policies exercised**. "
                  "Zero TEMPORAL_ANCHOR_MISMATCH findings do not establish correct anchoring for unexercised policies."]
    lines += ["", "`TEMPORAL_ANCHOR_MISMATCH` (#5315) is a cross-policy chronology contract check for custom or migrated pipelines, not a reproduced upstream bug. "
              "The inspected native RelativeActionsProcessorStep and its inverse collapse 3D state with `state[:, 0]`. "
              "That choice is intentional and valid for current-first windows whose saved `observation_delta_indices` starts at 0, including VLA-JEPA and LingBot-VA. "
              "The check rejects a custom or migrated pairing with an oldest-first state history ending at delta 0, such as `[-1, 0]` or `[-k, ..., 0]`, whose current frame is `state[:, -1]`. "
              "The audit resolves these supported saved delta-index patterns automatically; explicitly selected chronology metadata overrides the inference. "
              "A saved non-null `state_observation_delta_indices` takes precedence for state, matching LeRobot's dataset factory. "
              "Delta-window length can exceed `n_obs_steps=1` in native VLA configurations. "
              "Policy names, tensor shapes, and separate image-history windows do not establish state chronology."]
    base_reuse = summary.get("shared_base_stat_evidence")
    if base_reuse:
        n = len(base_reuse["repo_ids"])
        lines += ["", f"The shared SmolVLA stat-binding family includes **{n} repositories**: the official base and "
                  f"{max(n - 1, 0)} community forks/migrations share its individual action-stat payloads. "
                  f"The canonical modern container is `{base_reuse['canonical_container_sha256_prefix']}`; "
                  "legacy containers and Git LFS pointers are not byte-identical tensor files. Payload matches do not verify fine-tuning history. "
                  "The two xArm repositories share identical saved statistics and contribute one numeric-risk signature."]
    selection = report.get("selection", {})
    discovery = selection.get("original_discovery") or selection
    if discovery.get("strategy") == "tagged_family_round_robin_then_tagged_fallback":
        method = "tagged `lerobot` policy-family queries, name/tag family hints, round-robin family balancing and tagged fallback; obvious asset listings were filtered"
    elif selection.get("mode") == "cached":
        method = "the same pinned repositories from the recorded input reports; no fresh Hub selection"
    elif discovery.get("filter"):
        method = f"Hub `{discovery['filter']}` filter, capped by the recorded limit"
    else:
        method = "explicit repository IDs supplied by the caller"
    lines += ["", f"Selection: {method}. This is a convenience sample, not a prevalence estimate.",
              "Every repository below identifies its pinned commit; JSON records full revisions, discovery queries when available, and repo-relative local evidence paths."]
    if discovery.get("queries"):
        lines += ["", "## Exact Hub discovery queries (including seeds)", "",
                  "Seed searches are not guaranteed inclusions; matching tagged candidates are selected before family round-robin and fallback. "
                  "No additional discovery query ran during this cached revalidation.", "",
                  "| Query purpose | HfApi.list_models arguments | Returned |", "| --- | --- | ---: |"]
        for query in discovery["queries"]:
            arguments = {key: value for key, value in query.items() if key != "returned"}
            purpose = ("seed search" if query.get("search") in {seed.split("/", 1)[1] for seed in _TARGET_SEEDS}
                       and query.get("limit") == 25 else "family search" if query.get("search") else "tagged fallback")
            lines.append(f"| {purpose} | `{json.dumps(arguments, sort_keys=True)}` | {query.get('returned', 'unavailable')} |")
        lines += ["", "Configured seed repository IDs: " + ", ".join(f"`{seed}`" for seed in _TARGET_SEEDS) + ".",
                  "Verified seed IDs in the original candidate pool: " +
                  (", ".join(f"`{repo}`" for repo in discovery.get("verified_seed_repositories", [])) or "none recorded") + "."]
    lines += ["", "## Repository classification", "", "| Classification | Count |", "| --- | ---: |"]
    for category in _BUCKETS:
        count = summary.get("buckets", {}).get(category)
        if count is None:
            count = sum(repo.get("classification") == category for repo in report["repos"])
        lines.append(f"| `{category}` | {count} |")
    if discovery.get("family_hint_counts"):
        lines += ["", "## Targeted discovery coverage", "",
                  "These are repository-name/tag hints used for balanced selection, not inferred architecture compatibility.",
                  "", "| Family hint | Repositories selected |", "| --- | ---: |"]
        for family, count in discovery["family_hint_counts"].items():
            lines.append(f"| `{family}` | {count} |")
        lines += ["", f"Candidate pool: {discovery['candidate_count']}; asset-only listings filtered before selection: "
                  f"{len(discovery.get('filtered_asset_repositories', []))}. Full discovery queries are recorded in JSON."]
    payload_groups = _individual_payload_groups(report["repos"])
    shared = {digest: refs for digest, refs in payload_groups.items()
              if len({ref["repo_id"] for ref in refs}) > 1}
    if shared:
        lines += ["", "## Shared stat payloads", "",
                  "Groups use the full SHA-256 of each individual tensor's raw payload bytes, independently of safetensors file layout, tensor order, or unrelated buffers. Byte equality alone does not establish matching dtype, shape, feature, normalization mode, or physical semantics; each reference retains that metadata. Unused serialized statistics are included in this artifact inventory.",
                  "", "| Individual tensor-payload SHA-256 | Repository / tensor references |", "| --- | --- |"]
        for digest, refs in sorted(shared.items()):
            references = sorted({f"`{ref['repo_id']}:{ref['tensor_key']}`" for ref in refs})
            lines.append(f"| `{digest}` | {', '.join(references)} |")

    def cell(value: Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return text.replace("|", "\\|").replace("\n", "<br>")

    for category, title in (
        ("artifact_contract", "Artifact / Processor-Contract Findings"),
        ("base_model_prerequisite", "Intentional Base Models: Dataset-Stat Binding Prerequisites (Non-Defect)"),
        ("version_compatibility", "Version-Compatibility / Legacy Schema (Requires PR #1452 Migration)"),
        ("conditional_precision", "Conditional FP16 Precision Warnings (Legacy or Unconfigured Modern Half Precision)"),
    ):
        lines += ["", f"## {title}", "", "| Check | Unique repositories | Repositories |", "| --- | ---: | --- |"]
        for check, repos in categories[category].items():
            links = ", ".join(f"[{repo}](https://huggingface.co/{repo})" for repo in repos)
            lines.append(f"| `{check}` | {len(repos)} | {links} |")
        if not categories[category]:
            lines.append("| — | 0 | — |")
        if category == "version_compatibility":
            lines += ["", "Legacy checkpoints may be valid for their original LeRobot version. These findings identify migration requirements for modern processor consumers, not standalone checkpoint defects."]
        elif category == "conditional_precision":
            lines += ["", "These WARNING advisories concern a hypothetical FP16 execution that is not configured, or casting legacy buffers/module to half. They are excluded from defect counts and do not independently fail standard CLI mode. Strict mode rejects WARNING advisories. Actual non-finite results under an active runtime dtype remain blocking FAIL findings."]
        elif category == "base_model_prerequisite":
            lines += ["", "Bind target dataset statistics or fine-tune before execution. These are non-defect prerequisites for base checkpoints such as smolvla_base and pi0_old; missing-stat FEATURE_CONTRACT symptoms are grouped with the same prerequisite. This report does not establish whether community weights were fine-tuned. Original deployment-gate statuses remain visible.", "",
                      'Classification is explicitly curated: `_INTENTIONAL_BASE_REPOS = {"lerobot/smolvla_base", "lerobot/pi0_old"}` identifies those two official upstream bases by repository ID. Shared buffer hashes (individual tensor-payload SHA-256) identify exact-stat copies/forks; unbound multi-dataset stat namespaces establish the binding prerequisite. Neither establishes neural-weight identity or fine-tuning history.']
        evidence_heading = {
            "artifact_contract": "Exact failure and conflict evidence",
            "version_compatibility": "Exact compatibility evidence",
            "conditional_precision": "Exact conditional precision evidence",
            "base_model_prerequisite": "Exact base-model prerequisite evidence",
        }[category]
        lines += ["", f"### {evidence_heading}", "",
                  "Step indices are zero-based within the named processor file. Embedded legacy buffers have no processor step index. File/header digests and individual tensor-payload digests have distinct scopes.", "",
                  "| Repository / check | Status / severity | File / step | Tensor key(s) | Dimensions | File SHA-256 prefix / scope | Tensor-payload SHA-256 | Exact evidence |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        evidence_rows = 0
        seen_payloads: dict[str, str] = {}
        for repo in report["repos"]:
            for finding in repo["findings"]:
                if not _included_report_finding(finding) or _finding_category(finding) != category:
                    continue
                evidence_rows += 1
                details, provenance = finding.get("details") or {}, finding.get("provenance") or {}
                step = details.get("step_index")
                if step is None:
                    step = "embedded (no processor step)" if details.get("legacy_buffer_keys") else "not available"
                keys = details.get("legacy_buffer_keys") or provenance.get("tensor_key") or details.get("feature_key") or "not available"
                source = PurePosixPath(provenance.get("file_path") or "not available").name
                digest = provenance.get("sha256_prefix") or "not available"
                scope = details.get("digest_scope") or "local_evidence_file"
                exact = {"summary": finding.get("summary", ""), **details}
                tensors = _finding_payloads(repo, finding)
                digests = sorted({tensor["payload_sha256"] for tensor in tensors})
                signature = json.dumps([finding["check_id"],
                    sorted((tensor["payload_sha256"], tensor["dtype"], tensor["shape"], tensor["tensor_key"]) for tensor in tensors),
                    details.get("feature_key"), details.get("normalization_mode"), details.get("step_direction"),
                    details.get("eps"), details.get("screened_dtype"), details.get("risk_reasons"),
                    {key: details.get(key) for key in (
                        "normalization_implementation", "gain_formula", "screening_semantics",
                        "legacy_min_max_epsilon_behavior", "fp16_execution_conditional",
                        "declared_feature_keys", "available_buffer_keys", "matching_prefixed_keys",
                        "expected_buffer_keys", "required_buffer_keys", "target_dataset_stats_bound",
                        "input_dtype", "output_dtype", "probe_scope", "probe_input_values",
                        "probe_coordinates", "input_values", "source_stat_values",
                    )},
                    provenance.get("violating_indices")], sort_keys=True)
                if digests and signature in seen_payloads:
                    exact = {"shared_payload_evidence": seen_payloads[signature], "step_index": step,
                             "note": "Same individual tensor payloads and screening configuration; evidence printed once."}
                elif digests:
                    seen_payloads[signature] = f"{repo['repo_id']} / {finding['check_id']} / {keys}"
                lines.append(f"| [{repo['repo_id']}](https://huggingface.co/{repo['repo_id']}) / `{finding['check_id']}` | "
                             f"{finding['status']} / {finding.get('severity', 'unspecified')} | `{cell(source)}` / {cell(step)} | {cell(keys)} | "
                             f"{cell(provenance.get('violating_indices'))} | `{cell(digest)}` / {cell(scope)} | "
                             f"{cell(digests) if digests else 'not available'} | {cell(exact)} |")
        if not evidence_rows:
            lines.append("| — | — | — | — | — | — | — | No findings in this category. |")
    cameras = [(repo, f) for repo in report["repos"] for f in repo["findings"]
               if f.get("check_id") == "CAMERA_KEY_CONTRACT_MISMATCH"]
    if cameras:
        lines += ["", "## Camera declaration advisories", "",
                  "IDENTITY visual normalization passes unmatched camera keys through without crashing preprocessing. These WARNING advisories describe declaration drift and downstream policy input-key lookup risk, not a blocking preprocessing failure.",
                  "", "| Repository | Status / severity | Evidence |", "| --- | --- | --- |"]
        for repo, finding in cameras:
            lines.append(f"| `{repo['repo_id']}` | {finding['status']} / {finding.get('severity', 'WARNING')} | {cell(finding.get('details', {}))} |")
    lines += ["", "## Repository coverage", "", "| Repository | Revision | Generation | Static findings | Coverage errors |",
              "| --- | --- | --- | --- | ---: |"]
    for repo in report["repos"]:
        revision = repo.get("revision")
        url = f"https://huggingface.co/{repo['repo_id']}" + (f"/tree/{revision}" if revision else "")
        issues = sorted({f"{f['check_id']} ({_finding_category(f)})" for f in repo["findings"] if f["status"] in ("FAIL", "CONFLICT")})
        lines.append(f"| [{repo['repo_id']}]({url}) | `{(revision or 'unavailable')[:12]}` | {repo['classification']} | "
                     f"{', '.join(f'`{check}`' for check in issues) or '—'} | {len(repo['errors'])} |")
    lines += ["", "## Scope", "",
              "Local CLI: `lerobot-preflight --policy ./ckpt --report ./preflight-report.json`. Add `--sample ./obs.npz` for an offline numeric trace or `--baseline ./prod` for a processor comparison.",
              "This is artifact screening plus native synthetic normalization-boundary and temporal-anchor probes. No policy inference, simulator rollout, or motor command runs.",
              "Neural model weights are never downloaded as full files, including models below 5 MiB. Safetensors models use bounded header reads. Selected scalar/vector legacy state and action normalization values are fetched through exact HTTP 206 byte ranges; no other tensor payloads are requested.",
              "When legacy epsilon or execution dtype is undeclared, numeric screening uses the inspected native migration defaults (epsilon 1e-8); FP16 findings are conditional risks, not proof of the legacy runtime dtype or an observed execution failure.",
              f"Repositories with UNKNOWN numerical buffer screening: **{numeric_unknown}**; with UNKNOWN temporal-anchor checking: **{temporal_unknown}**. Zero listed failures do not establish absence of hazards where evidence is UNKNOWN.",
              "UNKNOWN findings and download errors indicate incomplete evidence. Static risk findings do not establish an observed robot failure.",
              (("This cached run enriches the input reports' pinned revisions with bounded legacy normalization range requests before local validation." if
                report.get("selection", {}).get("fetch_legacy_stats") else
                "This cached run re-audits the input reports' pinned revisions without network access.") if
               report.get("selection", {}).get("mode") == "cached" else
               "Repository selection follows the Hub `lerobot` filter (or explicit repo IDs); this is not a representative estimate of all LeRobot checkpoints."),
              "Cached runs reuse pinned snapshots and recorded access errors; they do not recheck current Hub permissions. Non-policy classification uses file/config evidence, not repository names.",
              f"Downloaded artifact bytes: {summary['artifact_bytes_downloaded']:,}; header bytes read: {summary['header_bytes_fetched']:,}; normalization tensor bytes read: {summary.get('normalization_bytes_fetched', 0):,}.", ""]
    lines[6:6] = _maintainer_highlights(report, output)
    markdown_path.write_text("\n".join(_collapse_report_appendices(lines)), encoding="utf-8")
    return json_path, markdown_path
