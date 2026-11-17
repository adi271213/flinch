"""Structured results for read-only checkpoint checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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


@dataclass
class Provenance:
    file_path: str
    step_name: str | int | None
    tensor_key: str | None
    sha256_prefix: str
    violating_indices: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "step_name": _json_value(self.step_name),
            "tensor_key": self.tensor_key,
            "sha256_prefix": self.sha256_prefix,
            "violating_indices": _json_value(self.violating_indices),
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
            "summary": self.summary,
            "details": _json_value(self.details),
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
        }
        non_finite = _non_finite_values(self.details)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "has_failures": self.has_failures(),
            "gate_policy": {
                "strict": self.strict,
                "baseline_requested": self.baseline_requested,
                "required_comparison_unknowns_fail": self.baseline_requested,
                "warning_advisories_fail": self.strict,
                "informational_physical_unknowns_exempt": True,
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }
