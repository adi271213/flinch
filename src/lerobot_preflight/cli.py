"""Command-line entry point for read-only LeRobot checkpoint diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .models import DiagnosticFinding, DiagnosticStatus, PreflightReport, Severity
from .preflight import run_preflight
from .temporal import run_temporal_audit
from .trace import run_offline_trace


def _runner_error(stage: str, exc: Exception) -> DiagnosticFinding:
    return DiagnosticFinding(
        check_id="RUNNER_ERROR",
        status=DiagnosticStatus.FAIL,
        severity=Severity.CRITICAL,
        summary=f"{stage} failed: {type(exc).__name__}: {exc}",
        details={"stage": stage, "exception_type": type(exc).__name__},
    )


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


def run_doctor(
    policy: Path,
    sample: Path | None = None,
    baseline: Path | None = None,
    std_threshold: float = 1e-4,
    strict: bool = False,
    metadata: Path | None = None,
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
    if sample is not None:
        _collect(report, "offline trace", lambda: run_offline_trace(policy, Path(sample)))
    if baseline is not None:

        def compare() -> list[DiagnosticFinding]:
            from .privilege_delta import run_privilege_delta

            return run_privilege_delta(Path(baseline), policy)

        _collect(report, "privilege delta", compare)
    return report


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
    if dimensions is not None:
        context.append(
            f"dims={json.dumps(dimensions, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}"
        )

    for field in ("tensor_slice", "saved_values", "observed_relative_action_slice", "value"):
        if field in details:
            context.append(f"values={_compact(details[field])}")
            break

    if provenance.get("sha256_prefix"):
        context.append(f"sha256={provenance['sha256_prefix']}")
    return " | ".join(context)


def _print_report(report: PreflightReport) -> None:
    print(f"lerobot-preflight: {report.checkpoint_dir}")
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
            print(f"  [{finding.check_id}]{advisory} {finding.summary}")
            context = _finding_context(finding)
            if context:
                print(f"    {context}")


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
    """Write JSON only to a new path or a previous doctor report."""
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
    parser = argparse.ArgumentParser(prog="lerobot-preflight", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--policy", type=Path, required=True, help="Local checkpoint directory")
    parser.add_argument("--sample", type=Path, help="Local observation NPZ for offline tracing")
    parser.add_argument("--baseline", type=Path, help="Local baseline checkpoint for privilege comparison")
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
        "--std-threshold",
        type=float,
        default=1e-4,
        help="Warn below this saved standard deviation (default: 1e-4)",
    )
    args = parser.parse_args(argv)

    try:
        report = run_doctor(
            args.policy,
            args.sample,
            args.baseline,
            args.std_threshold,
            strict=args.strict,
            metadata=args.metadata,
        )
    except Exception as exc:
        report = PreflightReport(str(args.policy))
        report.findings.append(_runner_error("doctor", exc))
    _print_report(report)
    if args.report is not None:
        try:
            _write_report(args.report, report)
        except Exception as exc:
            print(f"Could not write report {args.report}: {exc}", file=sys.stderr)
            return 1
    if args.graph is not None:
        try:
            from .graph import build_processor_graph, write_processor_graph

            graph = build_processor_graph(args.policy, metadata_path=args.metadata)
            write_processor_graph(graph, args.graph)
            print(f"Processor lineage graph: {args.graph}")
        except Exception as exc:
            print(f"Could not write graph {args.graph}: {exc}", file=sys.stderr)
            return 1
    return 1 if report.has_failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
