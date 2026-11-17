"""Structured results for read-only checkpoint checks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_TOOL_ERROR = 3


class DiagnosticStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


def _json_value(value: Any) -> Any:
    """Convert common diagnostic values into strict JSON-compatible data."""
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(_json_value(key)): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    # NumPy arrays/scalars and torch tensors expose these conversion methods.
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _non_finite_values(value: Any, path: str = "/details") -> list[dict[str, str]]:
    """Record IEEE values separately so a JSON string cannot conceal their type."""
    if isinstance(value, float) and not math.isfinite(value):
        return [{"path": path, "ieee754": _json_value(value)}]
    if isinstance(value, Mapping):
        return [
            item
            for key, child in value.items()
            for item in _non_finite_values(child, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        ]
    if isinstance(value, (list, tuple)):
        return [
            item for index, child in enumerate(value) for item in _non_finite_values(child, f"{path}/{index}")
        ]
    if hasattr(value, "tolist"):
        return _non_finite_values(value.tolist(), path)
    return []


# Findings carry evidence, not whole tensors. A 3x480x640 image statistic would otherwise
# put ~1M numbers into every report. Longer lists keep their first entries plus a count.
MAX_EVIDENCE_ITEMS = 64
_NUMBER_LIST = re.compile(r"\[(?:\s*-?[\d.eE+-]+(?:\s*,\s*-?[\d.eE+-]+)*\s*)\]")


def _bounded(value: Any) -> Any:
    """Cap long lists recursively, recording the original length next to each capped list."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if hasattr(item, "tolist") and not isinstance(item, (str, bytes)):
                item = item.tolist()
            if isinstance(item, (list, tuple)) and len(item) > MAX_EVIDENCE_ITEMS:
                out[str(key)] = [_bounded(child) for child in item[:MAX_EVIDENCE_ITEMS]]
                out[f"{key}_total_count"] = len(item)
            else:
                out[str(key)] = _bounded(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_bounded(child) for child in value]
    return value


def bounded_text(text: str) -> str:
    """Shorten long numeric lists inside a human-readable summary."""

    def shorten(match: re.Match[str]) -> str:
        items = [item.strip() for item in match.group(0)[1:-1].split(",")]
        if len(items) <= MAX_EVIDENCE_ITEMS:
            return match.group(0)
        return "[" + ", ".join(items[:8]) + f", ... ({len(items)} in total)]"

    return _NUMBER_LIST.sub(shorten, text)


@dataclass
class Provenance:
    file_path: str
    step_name: str | int | None
    tensor_key: str | None
    sha256_prefix: str
    violating_indices: list[int] | None = None
    # Total number of offending coordinates when only the first ones are listed.
    violating_index_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "step_name": _json_value(self.step_name),
            "tensor_key": self.tensor_key,
            "sha256_prefix": self.sha256_prefix,
            "violating_indices": _json_value(
                self.violating_indices[:MAX_EVIDENCE_ITEMS]
                if self.violating_indices
                else self.violating_indices
            ),
            **(
                {"violating_index_count": count}
                if (count := self.violating_index_count or len(self.violating_indices or []))
                > MAX_EVIDENCE_ITEMS
                else {}
            ),
        }


@dataclass
class DiagnosticFinding:
    check_id: str
    status: DiagnosticStatus
    severity: Severity
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "check_id": self.check_id,
            "status": self.status.value,
            "severity": self.severity.value,
            "summary": bounded_text(self.summary),
            "details": _json_value(_bounded(self.details)),
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
        }
        non_finite = _non_finite_values(_bounded(self.details))
        if non_finite:
            result["non_finite_values"] = non_finite
        return result


@dataclass
class PreflightReport:
    checkpoint_dir: str
    findings: list[DiagnosticFinding] = field(default_factory=list)
    strict: bool = False
    baseline_requested: bool = False

    def __post_init__(self) -> None:
        self.checkpoint_dir = str(self.checkpoint_dir)

    def has_failures(self) -> bool:
        return any(
            finding.status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
            or (self.strict and finding.severity is Severity.WARNING)
            or (
                finding.status is DiagnosticStatus.UNKNOWN
                and (
                    (
                        self.strict
                        and not (
                            finding.severity is Severity.INFO and finding.details.get("strict_exempt", False)
                        )
                    )
                    or (self.baseline_requested and finding.details.get("required_coverage", False))
                )
            )
            for finding in self.findings
        )

    def has_tool_errors(self) -> bool:
        """Flinch itself could not finish part of the run (crash, I/O, network)."""
        return any(finding.details.get("tool_error") is True for finding in self.findings)

    def exit_code(self) -> int:
        """0 pass, 1 checkpoint finding, 3 Flinch could not finish. (2 is a usage error.)

        A proven FAIL or CONFLICT wins over a tool error: it is the most actionable answer.
        """
        if any(
            finding.status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
            and finding.details.get("tool_error") is not True
            for finding in self.findings
        ):
            return EXIT_FINDINGS
        if self.has_tool_errors():
            return EXIT_TOOL_ERROR
        return EXIT_FINDINGS if self.has_failures() else EXIT_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "has_failures": self.has_failures(),
            "exit_code": self.exit_code(),
            "gate_policy": {
                "strict": self.strict,
                "baseline_requested": self.baseline_requested,
                "required_comparison_unknowns_fail": self.baseline_requested,
                "warning_advisories_fail": self.strict,
                "informational_physical_unknowns_exempt": True,
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }
