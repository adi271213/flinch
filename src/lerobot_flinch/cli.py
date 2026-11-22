"""Command-line entry point for read-only LeRobot checkpoint diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .eval_contract import (
    check_dataset,
    check_env,
    check_processor_steps,
    load_env_config,
    read_dataset_meta,
)
from .hub_source import (
    HubSourceError,
    ResolvedCheckpoint,
    looks_like_repo_id,
    parse_hub_reference,
    resolve_checkpoint,
    resolve_dataset,
)
from .models import (
    DiagnosticFinding,
    DiagnosticStatus,
    PreflightReport,
    Severity,
    bounded_text,
    tool_error_finding,
)
from .preflight import _multi_dataset_action_prefixes, run_preflight
from .temporal import run_temporal_audit
from .trace import run_offline_trace


def _positive_finite(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive, finite number, not {value}")
    return number


def _runner_error(stage: str, exc: Exception) -> DiagnosticFinding:
    return tool_error_finding("RUNNER_ERROR", stage, exc, Severity.CRITICAL)


def _collect(
    report: PreflightReport,
    stage: str,
    check: Callable[[], PreflightReport | list[DiagnosticFinding]],
) -> None:
    try:
        result = check()
        findings = result.findings if isinstance(result, PreflightReport) else result
        if not isinstance(findings, list) or not all(
            isinstance(finding, DiagnosticFinding) for finding in findings
        ):
            raise TypeError(f"{stage} returned invalid findings")
        if stage == "privilege delta":
            for finding in findings:
                if finding.status is DiagnosticStatus.UNKNOWN:
                    finding.details.setdefault("required_coverage", True)
            if not findings:
                findings = [
                    DiagnosticFinding(
                        "COMPATIBILITY_STATUS",
                        DiagnosticStatus.UNKNOWN,
                        Severity.WARNING,
                        "Baseline comparison returned no coverage evidence.",
                        {"required_coverage": True},
                    )
                ]
        report.findings.extend(findings)
    except Exception as exc:
        report.findings.append(_runner_error(stage, exc))


def _add_sample_evidence(report: PreflightReport, sample: Path) -> None:
    """Say what the recorded sample did at each near-zero-spread coordinate.

    The finding stays a FAIL either way: a coordinate that is constant in one recording is
    still one sensor glitch away from the full gain. This only tells the user how close
    their own data came.
    """
    wanted = {
        str(finding.details.get("feature_key"))
        for finding in report.findings
        if finding.check_id == "PATHOLOGICAL_STD_GAIN" and finding.status is DiagnosticStatus.FAIL
    }
    if not wanted:
        return
    try:
        with np.load(sample, allow_pickle=False) as archive:  # Read only the arrays we need.
            data = {key: archive[key] for key in archive.files if key in wanted}
    except (OSError, ValueError):
        return  # The trace already reports an unreadable sample.
    for finding in report.findings:
        details, provenance = finding.details, finding.provenance
        values = data.get(str(details.get("feature_key")))
        gains = details.get("effective_gains_with_eps")
        if (
            finding.check_id != "PATHOLOGICAL_STD_GAIN"
            or finding.status is not DiagnosticStatus.FAIL
            or provenance is None
            or not provenance.violating_indices
            or values is None
            or not isinstance(gains, list)
        ):
            continue
        width = math.prod(details.get("tensor_shape") or [values.shape[-1]])
        if not width or values.size % width:
            continue
        indices = provenance.violating_indices[: len(gains)]
        frames = values.reshape(-1, width)
        spread = np.ptp(frames[:, indices].astype(np.float64), axis=0)
        normalized = spread * np.asarray(gains[: len(indices)], dtype=np.float64)
        details["sample_frames"] = len(frames)
        details["sample_raw_spread"] = spread.tolist()
        details["sample_normalized_spread"] = normalized.tolist()
        if len(frames) == 1:
            finding.summary += (
                " Your --sample has one frame, so it can't show whether these coordinates move."
            )
        elif not normalized.any():
            finding.summary += f" Across the {len(frames)} frames of your --sample they never move."
        else:
            finding.summary += (
                f" Across the {len(frames)} frames of your --sample they move by up to "
                f"{normalized.max():.4g} in normalized units."
            )


def run_checks(
    policy: Path,
    sample: Path | None = None,
    baseline: Path | None = None,
    std_threshold: float = 1e-4,
    strict: bool = False,
    metadata: Path | None = None,
    env: Any | None = None,
    dataset: Path | None = None,
) -> PreflightReport:
    """Run selected checks and collect failures without aborting later checks."""
    policy = Path(policy)
    report = PreflightReport(str(policy.resolve()), strict=strict, baseline_requested=baseline is not None)
    metadata_options = {"metadata_path": Path(metadata)} if metadata is not None else {}
    _collect(
        report,
        "preflight",
        lambda: run_preflight(policy, std_warn_threshold=std_threshold, **metadata_options),
    )
    _collect(report, "temporal audit", lambda: run_temporal_audit(policy, **metadata_options))
    _collect(report, "evaluation contract", lambda: _evaluation_contract(policy, env, dataset))
    if sample is not None:
        _collect(report, "offline trace", lambda: run_offline_trace(policy, Path(sample)))
        _add_sample_evidence(report, Path(sample))
    if baseline is not None:

        def compare() -> list[DiagnosticFinding]:
            from .privilege_delta import run_privilege_delta

            return run_privilege_delta(Path(baseline), policy)

        _collect(report, "privilege delta", compare)
    _fold_base_model_prerequisites(report)
    return report


def _evaluation_contract(policy: Path, env: Any | None, dataset: Path | None) -> list[DiagnosticFinding]:
    """Checks against the env and dataset the checkpoint will be used with (eval_contract.py)."""
    config_path = policy / "config.json"
    if not config_path.is_file():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return []
    findings = check_processor_steps(policy, config, env)
    if env is not None:
        findings += check_env(config, env)
    if dataset is not None:
        findings += check_dataset(config, *read_dataset_meta(dataset))
    return findings


def _is_unbound_missing_buffer(finding: DiagnosticFinding) -> bool:
    """A missing-buffer candidate with multi-dataset statistics to match by source."""
    return (
        finding.check_id == "FEATURE_CONTRACT"
        and finding.status is DiagnosticStatus.FAIL
        and finding.summary.startswith("Required normalization buffer ")
        and bool(_multi_dataset_action_prefixes(finding.details.get("available_buffer_keys", [])))
    )


def _fold_base_model_prerequisites(report: PreflightReport) -> None:
    """Present one actionable base-model finding instead of one failure per missing buffer.

    The status stays FAIL: a base model is not deployable as-is. Every folded finding is kept
    verbatim under ``details.folded_findings`` in the JSON report.
    """

    def source(finding: DiagnosticFinding) -> tuple[str, str | int | None, int | None] | None:
        provenance = finding.provenance
        if provenance is None:
            return None
        index = finding.details.get("step_index")
        return provenance.file_path, provenance.step_name, index if isinstance(index, int) else None

    groups: dict[tuple[str, ...], list[DiagnosticFinding]] = {}
    for finding in report.findings:
        if finding.check_id != "UNBOUND_BASE_MODEL_STATS" or finding.status is not DiagnosticStatus.FAIL:
            continue
        prefixes = _multi_dataset_action_prefixes(finding.details.get("available_buffer_keys", []))
        if prefixes and source(finding) is not None:
            groups.setdefault(prefixes, []).append(finding)

    folded_ids: set[int] = set()
    root = Path(report.checkpoint_dir)
    for prefixes, unbound in groups.items():
        required: dict[tuple[str, str | int | None, int | None], set[str]] = {}
        for finding in unbound:
            location = source(finding)
            if location is not None:
                required.setdefault(location, set()).update(
                    finding.details.get("required_buffer_keys", [])
                    + finding.details.get("missing_observation_state_stats", [])
                )
        related = []
        for finding in report.findings:
            location = source(finding)
            provenance = finding.provenance
            if (
                _is_unbound_missing_buffer(finding)
                and _multi_dataset_action_prefixes(finding.details.get("available_buffer_keys", []))
                == prefixes
                and location is not None
                and provenance is not None
                and provenance.tensor_key in required.get(location, set())
            ):
                related.append(finding)
        if not related:
            continue  # Do not invent a missing-buffer count from unavailable lineage.
        primary = unbound[0]
        folded = unbound[1:] + related
        missing: list[dict[str, Any]] = []
        for finding in related:
            provenance = finding.provenance
            assert provenance is not None
            path = Path(provenance.file_path)
            try:
                processor_file = path.relative_to(root).as_posix()
            except ValueError:
                processor_file = str(path)
            entry = {
                "processor_file": processor_file,
                "step_index": finding.details.get("step_index"),
                "step_name": provenance.step_name,
                "tensor_key": provenance.tensor_key,
            }
            if entry not in missing:
                missing.append(entry)
        files = {entry["processor_file"] for entry in missing}
        primary.summary = (
            "Base model: the target dataset's normalization statistics are not bound "
            f"({len(missing)} required buffers missing across {len(files)} processor files). "
            "Fine-tune on your dataset or bind its statistics before deploying; as-is, "
            "normalization is silently bypassed."
        )
        primary.details.update(
            {
                "missing_buffers": missing,
                "folded_findings": [finding.to_dict() for finding in folded],
            }
        )
        folded_ids.update(id(finding) for finding in folded if finding is not primary)
    report.findings[:] = [finding for finding in report.findings if id(finding) not in folded_ids]


def _compact(value: Any, limit: int = 160) -> str:
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _finding_context(finding: DiagnosticFinding) -> str:
    data = finding.to_dict()
    details = data["details"]
    provenance = data["provenance"] or {}
    context: list[str] = []

    step = details.get("step_index")
    if step is None:
        step = details.get("baseline_step_index")
    if step is None and isinstance(provenance.get("step_name"), int):
        step = provenance["step_name"]
    if step is not None:
        context.append(f"step={step}")

    key = provenance.get("tensor_key") or details.get("tensor_key") or details.get("feature_key")
    if key is not None:
        context.append(f"key={key}")

    dimensions = provenance.get("violating_indices")
    if dimensions is None and details.get("probe_coordinate") is not None:
        dimensions = [details["probe_coordinate"]]
    if dimensions not in (None, []):
        context.append(
            f"dims={json.dumps(dimensions, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}"
        )

    for field in ("tensor_slice", "saved_values", "observed_relative_action_slice", "value"):
        if field in details and details[field] not in (None, [], {}):
            context.append(f"values={_compact(details[field])}")
            break

    if provenance.get("sha256_prefix"):
        context.append(f"sha256={provenance['sha256_prefix']}")
        if provenance.get("digest_scope"):
            context.append(f"digest_scope={provenance['digest_scope']}")
    return " | ".join(context)


def _print_report(report: PreflightReport) -> None:
    print(f"flinch: {report.checkpoint_dir}")
    if report.strict or report.baseline_requested:
        print(
            "Gate policy: required UNKNOWN findings and WARNING advisories fail; informational physical notes are exempt (--strict)."
            if report.strict
            else "Gate policy: UNKNOWN required baseline comparison coverage fails."
        )
    for status in (
        DiagnosticStatus.FAIL,
        DiagnosticStatus.CONFLICT,
        DiagnosticStatus.UNKNOWN,
        DiagnosticStatus.PASS,
    ):
        findings = [finding for finding in report.findings if finding.status == status]
        print(f"{status.value} ({len(findings)})")
        for finding in findings:
            advisory = " (WARNING)" if finding.severity is Severity.WARNING else ""
            print(f"  [{finding.check_id}]{advisory} {bounded_text(finding.summary)}")
            context = _finding_context(finding)
            if context:
                print(f"    {context}")
    _print_coverage(report)


def _print_coverage(report: PreflightReport) -> None:
    """End with what the result does and doesn't cover, so exit 0 is never read as 'all verified'."""
    coverage = report.coverage()
    missing = coverage["not_verified"]
    print(f"COVERAGE: {coverage['checks_verified']} of {coverage['checks_run']} checks verified")
    if missing:
        print(f"  Not verified: {', '.join(missing)}")
    if coverage["not_declared"]:
        print(f"  Not declared (physical meaning unknown): {', '.join(coverage['not_declared'])}")
    if missing and report.exit_code() == 0:
        print(
            "  Exit 0 means no problem was found in the checks that ran, not that everything was verified."
            " Use --strict to fail on unverified checks."
        )


def _is_prior_report(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(document, dict) or not isinstance(document.get("checkpoint_dir"), str):
        return False
    if not isinstance(document.get("has_failures"), bool):
        return False
    findings = document.get("findings")
    return isinstance(findings, list) and all(
        isinstance(finding, dict)
        and isinstance(finding.get("check_id"), str)
        and finding.get("status") in {status.value for status in DiagnosticStatus}
        and isinstance(finding.get("summary"), str)
        for finding in findings
    )


def _write_report(path: Path, report: PreflightReport) -> None:
    """Write JSON only to a new path or a previous preflight report."""
    if path.exists() or path.is_symlink():
        if not _is_prior_report(path):
            raise FileExistsError(f"Refusing to overwrite an audited artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        path.write_text(document, encoding="utf-8")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flinch", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--policy",
        required=True,
        help="Local checkpoint directory, or a public Hub checkpoint as hf://owner/name[@revision]",
    )
    parser.add_argument("--sample", type=Path, help="Local observation NPZ for offline tracing")
    parser.add_argument(
        "--baseline",
        help="Baseline checkpoint for privilege comparison: local directory or hf://owner/name[@revision]",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on coverage UNKNOWNs or WARNING advisories; optional physical notes are exempt",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Explicit local JSON sidecar for physical semantics and observation chronology",
    )
    parser.add_argument("--report", type=Path, help="Write strict JSON to a new or prior report file")
    parser.add_argument(
        "--graph", type=Path, help="Export processor/feature lineage as .json, .mmd, or Markdown .md"
    )
    parser.add_argument(
        "--env",
        help="LeRobot env the checkpoint will be evaluated in (e.g. libero, robocasa); no simulator is started",
    )
    parser.add_argument(
        "--env-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Env setting as passed to lerobot-eval, e.g. observation_height=256 (repeatable)",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset the checkpoint will be fine-tuned on or was trained on: local folder or hf://owner/name",
    )
    parser.add_argument(
        "--std-threshold",
        type=_positive_finite,
        default=1e-4,
        help="Warn below this saved standard deviation (default: 1e-4)",
    )
    args = parser.parse_args(argv)
    if args.env_option and args.env is None:
        parser.error("--env-option needs --env")
    try:
        options = dict(option.split("=", 1) for option in args.env_option)
    except ValueError:
        parser.error("--env-option takes KEY=VALUE")
    try:
        args.env_config = load_env_config(args.env, options) if args.env else None
    except Exception as exc:  # Unknown env type or option, or a value LeRobot rejects.
        parser.error(f"--env {args.env}: {exc}")
    if args.dataset is not None and parse_hub_reference(args.dataset) is None:
        if not (Path(args.dataset) / "meta" / "info.json").is_file():
            parser.error(f"--dataset {args.dataset!r} has no meta/info.json")
    if args.graph is not None and args.graph.suffix.lower() not in {".json", ".mmd", ".md"}:
        parser.error(f"--graph must end in .json, .mmd or .md, not {args.graph.suffix or 'nothing'}")
    for option in ("policy", "baseline"):
        value = getattr(args, option)
        if value is None:
            continue
        try:
            parse_hub_reference(value)
        except HubSourceError as exc:
            parser.error(str(exc))
        if looks_like_repo_id(value):
            parser.error(
                f"--{option} {value!r} is not a local directory. "
                f"To audit a Hugging Face Hub checkpoint, use --{option} hf://{value}"
            )
        # A typo in a path is a usage error, not evidence of a broken checkpoint.
        if parse_hub_reference(value) is None and not Path(value).is_dir():
            problem = "is a file, not a checkpoint directory" if Path(value).exists() else "does not exist"
            parser.error(f"--{option} {value!r} {problem}")
    for option in ("sample", "metadata"):
        value = getattr(args, option)
        if value is not None and not value.is_file():
            parser.error(f"--{option} {str(value)!r} is not a file")

    try:
        with ExitStack() as stack:
            policy = stack.enter_context(resolve_checkpoint(args.policy))
            baseline = stack.enter_context(resolve_checkpoint(args.baseline)) if args.baseline else None
            args.dataset_path = stack.enter_context(resolve_dataset(args.dataset)) if args.dataset else None
            return _run_resolved(args, policy, baseline)
    except HubSourceError as exc:
        print(f"flinch: {exc}", file=sys.stderr)
        report = PreflightReport(
            str(args.policy), strict=args.strict, baseline_requested=args.baseline is not None
        )
        report.findings.append(_runner_error("Hub snapshot", exc))
        # The error is already on stderr; only write the JSON report (if asked) for CI.
        return _finish_report(args, report, print_terminal=False)


def _hub_snapshot_finding(resolved: ResolvedCheckpoint, role: str) -> DiagnosticFinding:
    manifest = resolved.manifest or {}
    details = {
        "role": role,
        "repo_id": manifest.get("repo_id"),
        "requested_revision": manifest.get("requested_revision"),
        "revision": manifest.get("revision"),
        "downloaded_files": manifest.get("downloaded_files", []),
        "artifact_bytes_downloaded": manifest.get("artifact_bytes_downloaded"),
        "neural_weights_downloaded": False,
        "errors": manifest.get("errors", []),
    }
    if manifest.get("errors"):
        return DiagnosticFinding(
            "HUB_SNAPSHOT",
            DiagnosticStatus.UNKNOWN,
            Severity.WARNING,
            f"Audited {resolved.label}; some optional Hub metadata or weight headers were unavailable.",
            details,
        )
    return DiagnosticFinding(
        "HUB_SNAPSHOT",
        DiagnosticStatus.PASS,
        Severity.INFO,
        f"Audited pinned {role} snapshot {resolved.label}; neural weights were not downloaded.",
        details,
    )


def _run_resolved(
    args: argparse.Namespace, policy: ResolvedCheckpoint, baseline: ResolvedCheckpoint | None
) -> int:
    try:
        report = run_checks(
            policy.path,
            args.sample,
            baseline.path if baseline is not None else None,
            args.std_threshold,
            strict=args.strict,
            metadata=args.metadata,
            env=args.env_config,
            dataset=args.dataset_path,
        )
    except Exception as exc:
        report = PreflightReport(
            str(args.policy), strict=args.strict, baseline_requested=args.baseline is not None
        )
        report.findings.append(_runner_error("preflight run", exc))
    report.checkpoint_dir = policy.label
    hub_findings = [
        _hub_snapshot_finding(resolved, role)
        for resolved, role in ((policy, "policy"), (baseline, "baseline"))
        if resolved is not None and resolved.manifest is not None
    ]
    report.findings[:0] = hub_findings
    return _finish_report(args, report, policy=policy)


def _finish_report(
    args: argparse.Namespace,
    report: PreflightReport,
    *,
    policy: ResolvedCheckpoint | None = None,
    print_terminal: bool = True,
) -> int:
    """Retain findings and output errors before determining the effective gate result."""
    graph_written = False
    if args.graph is not None and policy is not None:
        try:
            from .graph import build_processor_graph, write_processor_graph

            graph = build_processor_graph(policy.path, metadata_path=args.metadata)
            write_processor_graph(graph, args.graph)
            graph_written = True
        except Exception as exc:
            finding = _runner_error("graph output", exc)
            finding.details["output_path"] = str(args.graph)
            report.findings.append(finding)
            print(f"Could not write graph {args.graph}: {exc}", file=sys.stderr)
    try:
        if print_terminal:
            _print_report(report)
        if graph_written:
            print(f"Processor lineage graph: {args.graph}")
    except Exception as exc:
        report.findings.append(_runner_error("terminal output", exc))
        print(f"Could not print terminal report: {exc}", file=sys.stderr)
    if args.report is not None:
        try:
            _write_report(args.report, report)
        except Exception as exc:
            finding = _runner_error("report output", exc)
            finding.details["output_path"] = str(args.report)
            report.findings.append(finding)
            print(f"Could not write report {args.report}: {exc}", file=sys.stderr)
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
