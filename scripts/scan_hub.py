"""Scan a bounded set of LeRobot Hub checkpoints for processor hazards."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

# Support direct execution from a checkout without installing the package.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lerobot_flinch.hub_scanner import scan_hub, write_hub_reports  # noqa: E402


def _print_progress(*parts: Any) -> None:
    print(" ".join(str(part) for part in parts), flush=True)


def _print_summary(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    summary = report.get("summary", report)
    print("\nHub scan summary")
    for key, label in (
        ("total_repos", "Total repositories"),
        ("modern", "Modern"),
        ("legacy", "Legacy"),
        ("mixed", "Mixed"),
        ("non_policy", "Non-policy"),
        ("gated_or_private", "Gated or private"),
        ("unknown", "Unknown"),
        ("errors", "Errors"),
        ("normalization_bytes_fetched", "Legacy normalization bytes fetched"),
    ):
        print(f"  {label}: {summary.get(key, 0)}")

    issues = summary.get("issues", report.get("issues", {}))
    print("Issues:")
    if isinstance(issues, dict) and issues:
        for check_id in sorted(issues):
            repo_ids = issues[check_id]
            print(f"  {check_id} ({len(repo_ids)}):")
            for repo_id in sorted(repo_ids):
                print(f"    {repo_id}")
    else:
        print("  None")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=30, help="Maximum repositories to scan (1-100; default: 30)"
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Balance tagged policy-family candidates; filter asset-only repository listings",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        metavar="REPO_ID",
        help="Scan this repository; repeat for multiple repositories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for JSON and Markdown reports (default: current directory)",
    )
    parser.add_argument(
        "--report-name", default="hub_audit_report", help="Output filename stem, e.g. hub_100_report"
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, help="Optional local directory for downloaded checkpoint snapshots"
    )
    parser.add_argument(
        "--cached-report",
        action="append",
        type=Path,
        dest="cached_reports",
        help="Re-audit pinned snapshots from this prior JSON report; repeat to combine reports",
    )
    parser.add_argument(
        "--fetch-legacy-stats",
        action="store_true",
        help="Fetch tiny legacy normalization tensor ranges when re-auditing cached snapshots",
    )
    parser.add_argument(
        "--repo-file-evidence",
        type=Path,
        help="Pinned repository file lists to supplement older cached reports",
    )
    parser.add_argument(
        "--std-threshold", type=float, default=1e-4, help="Small standard-deviation threshold (default: 1e-4)"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    if args.repos is not None and len(args.repos) > 100:
        parser.error("no more than 100 --repo values are allowed")
    if args.cached_reports is not None and args.repos is not None:
        parser.error("choose --cached-report or --repo")
    if args.targeted and (args.cached_reports is not None or args.repos is not None):
        parser.error("choose --targeted, --cached-report, or --repo")
    if not math.isfinite(args.std_threshold) or args.std_threshold <= 0:
        parser.error("--std-threshold must be positive and finite")

    try:
        report = scan_hub(
            repo_ids=args.repos,
            limit=args.limit,
            snapshot_dir=args.snapshot_dir,
            std_threshold=args.std_threshold,
            progress=_print_progress,
            cached_reports=args.cached_reports,
            repo_file_evidence=args.repo_file_evidence,
            targeted=args.targeted,
            fetch_legacy_stats=args.fetch_legacy_stats,
        )
        json_path, markdown_path = write_hub_reports(report, args.output_dir, args.report_name)
    except Exception as exc:
        print(f"Hub scan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print_summary(report, json_path, markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
