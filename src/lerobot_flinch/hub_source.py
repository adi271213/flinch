"""Resolve ``hf://owner/name[@revision]`` checkpoint references to bounded local snapshots.

Network access happens here, before any diagnostic runs. Only processor configs, processor
statistics, model headers and bounded legacy normalization slices are fetched; neural weights
are never downloaded. Diagnostics then run on the local snapshot exactly as for a local path.
"""

from __future__ import annotations

import functools
import re
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
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


_UNAVAILABLE_ERRORS = {"RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError"}


def _repository_unavailable(error: dict[str, Any]) -> bool:
    """A repository-level 401/403/404 means missing, private or gated, never a credential problem."""
    return error.get("file_path") is None and (
        error.get("error_type") in _UNAVAILABLE_ERRORS or error.get("http_status") in {401, 403, 404}
    )


def _first_line(message: object) -> str:
    """Keep the cause of a Hub client error without its multi-line help text."""
    lines = str(message).strip().splitlines() if message is not None else []
    return lines[0] if lines else "unknown error"


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

    failure = f"Could not fetch the processor artifacts of {reference.display()}"
    with ExitStack() as stack:
        try:
            temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="flinch-hub-"))
            manifest = download_lightweight_snapshot(
                reference.repo_id, Path(temporary) / "snapshot", api=api, revision=reference.revision
            )
        except (OSError, ValueError) as exc:
            raise HubSourceError(f"{failure} ({type(exc).__name__}: {exc})") from exc
        access = manifest.get("repo_access", {})
        if access.get("private") or access.get("gated"):
            raise HubSourceError(
                f"{failure}. Gated or private repositories are not supported; "
                "download the checkpoint and pass its local path."
            )
        if manifest.get("critical_download_errors"):
            errors = manifest.get("errors") or []
            if any(_repository_unavailable(item) for item in errors):
                raise HubSourceError(
                    f"Repository {reference.display()} was not found (check the repository name and revision), or it is private or gated. "
                    "Only public, ungated repositories can be audited by reference; "
                    "download the checkpoint and pass its local path."
                )
            skipped = [item for item in manifest.get("skipped_files", []) if item.get("reason")]
            reasons = "; ".join(
                f"{item.get('file_path') or 'repository'}: {_first_line(item.get('error'))}"
                for item in errors
            ) or "; ".join(f"{item['file_path']}: {item['reason']}" for item in skipped)
            raise HubSourceError(failure + (f" ({reasons})" if reasons else "") + ".")
        yield ResolvedCheckpoint(
            Path(manifest["checkpoint_dir"]),
            reference.display(manifest.get("revision")),
            manifest,
        )


@contextmanager
def resolve_dataset(value: str | Path) -> Iterator[Path]:
    """A local dataset folder, or the metadata of a public Hub dataset in a temporary folder."""
    reference = parse_hub_reference(str(value))
    if reference is None:
        yield Path(value)
        return
    from huggingface_hub import hf_hub_download

    from .hub_scanner import with_retries

    with tempfile.TemporaryDirectory(prefix="flinch-dataset-") as temporary:
        root = Path(temporary)
        for name in ("meta/info.json", "meta/tasks.parquet", "meta/tasks.jsonl"):
            try:
                with_retries(
                    functools.partial(
                        hf_hub_download,
                        reference.repo_id,
                        name,
                        repo_type="dataset",
                        revision=reference.revision,
                        local_dir=root,
                        token=False,
                    )
                )
            except Exception as exc:
                if name == "meta/info.json":
                    raise HubSourceError(
                        f"Could not fetch the metadata of dataset {reference.display()} ({_first_line(exc)})."
                    ) from exc
        yield root
