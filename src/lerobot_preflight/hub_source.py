"""Resolve ``hf://owner/name[@revision]`` checkpoint references to bounded local snapshots.

Network access happens here, before any diagnostic runs. Only processor configs, processor
statistics, model headers and bounded legacy normalization slices are fetched; neural weights
are never downloaded. Diagnostics then run on the local snapshot exactly as for a local path.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HUB_PREFIX = "hf://"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class HubSourceError(RuntimeError):
    """A Hub reference could not be resolved to a complete processor snapshot."""


@dataclass(frozen=True)
class HubReference:
    repo_id: str
    revision: str | None = None

    def display(self, resolved_revision: str | None = None) -> str:
        revision = resolved_revision or self.revision
        return f"{HUB_PREFIX}{self.repo_id}" + (f"@{revision}" if revision else "")


@dataclass
class ResolvedCheckpoint:
    path: Path
    label: str
    manifest: dict[str, Any] | None = field(default=None)


def parse_hub_reference(value: str) -> HubReference | None:
    """Return the Hub reference for ``hf://owner/name[@revision]``, or ``None`` for a local path."""
    if not value.startswith(HUB_PREFIX):
        return None
    body = value[len(HUB_PREFIX) :]
    repo_id, separator, revision = body.partition("@")
    if not _REPO_ID.fullmatch(repo_id) or ".." in repo_id:
        raise HubSourceError(
            f"Invalid Hub repository reference {value!r}; expected hf://owner/name[@revision]"
        )
    if separator and (not revision or not _REVISION.fullmatch(revision) or ".." in revision):
        raise HubSourceError(f"Invalid Hub revision in {value!r}")
    return HubReference(repo_id, revision or None)


def looks_like_repo_id(value: str) -> bool:
    """True for a bare ``owner/name`` that is plausibly a Hub ID rather than a local path."""
    return bool(_REPO_ID.fullmatch(value)) and not Path(value).exists()


@contextmanager
def resolve_checkpoint(value: str | Path, *, api: Any | None = None) -> Iterator[ResolvedCheckpoint]:
    """Yield a local checkpoint directory for a local path or an ``hf://`` reference.

    Hub snapshots live in a temporary directory that is removed when the context exits.
    """
    reference = parse_hub_reference(str(value))
    if reference is None:
        path = Path(value)
        yield ResolvedCheckpoint(path, str(path.resolve()))
        return

    from .hub_scanner import download_lightweight_snapshot

    with tempfile.TemporaryDirectory(prefix="lerobot-preflight-hub-") as temporary:
        manifest = download_lightweight_snapshot(
            reference.repo_id, Path(temporary) / "snapshot", api=api, revision=reference.revision
        )
        if manifest.get("critical_download_errors"):
            errors = manifest.get("errors") or []
            skipped = [item for item in manifest.get("skipped_files", []) if item.get("reason")]
            reasons = "; ".join(
                f"{item.get('file_path') or 'repository'}: {item.get('error')}" for item in errors
            ) or "; ".join(f"{item['file_path']}: {item['reason']}" for item in skipped)
            raise HubSourceError(
                f"Could not fetch the processor artifacts of {reference.display()}"
                + (f" ({reasons})" if reasons else "")
                + ". Gated or private repositories are not supported; download the checkpoint and pass its local path."
            )
        yield ResolvedCheckpoint(
            Path(manifest["checkpoint_dir"]),
            reference.display(manifest.get("revision")),
            manifest,
        )
