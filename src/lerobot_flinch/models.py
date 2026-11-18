"""Structured results for read-only checkpoint checks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
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


def _sequence_leaves(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _sequence_leaves(child)
    else:
        yield value


def _plain_sequence(value: Any) -> bool:
    return all(
        child is None or isinstance(child, (str, bool, int, float, Enum, Path))
        for child in _sequence_leaves(value)
    )


_COUNTERS = ("total_count", "nan_count", "positive_infinity_count", "negative_infinity_count")


def _sequence_summary(value: Any) -> tuple[list[Any], dict[str, Any]]:
    sample: list[Any] = []
    summary: dict[str, Any] = dict.fromkeys(_COUNTERS, 0)
    low = high = None
    for index, child in enumerate(_sequence_leaves(value)):
        summary["total_count"] += 1
        if len(sample) < MAX_EVIDENCE_ITEMS:
            sample.append(child)
        if not isinstance(child, (int, float)) or isinstance(child, bool):
            continue
        if isinstance(child, float) and not math.isfinite(child):
            key = (
                "nan_count"
                if math.isnan(child)
                else "positive_infinity_count"
                if child > 0
                else "negative_infinity_count"
            )
            summary[key] += 1
            summary.setdefault("first_non_finite_flat_index", index)
            summary.setdefault("first_non_finite_value", _json_value(child))
        else:
            low = child if low is None else min(low, child)
            high = child if high is None else max(high, child)
    if low is not None:
        summary.update(finite_minimum=low, finite_maximum=high)
    return sample, summary


def _array_sample(value: Any) -> tuple[Any, dict[str, Any]]:
    """Reduce numeric arrays before converting their bounded sample to Python."""
    import numpy as np

    summary: dict[str, Any] = {"total_count": 0, "shape": list(value.shape)}
    if type(value).__module__.startswith("torch"):
        import torch

        summary["total_count"] = count = value.numel()
        if count <= MAX_EVIDENCE_ITEMS:
            return value.detach().cpu().tolist(), summary
        source = value.detach().reshape(-1).cpu()  # One host copy; sliced before .tolist().
        try:
            flat = source.numpy()
        except TypeError:
            # NumPy lacks bfloat16, float8 and complex32; these wider types hold them exactly.
            flat = source.to(torch.complex64 if source.is_complex() else torch.float32).numpy()
        numeric = True  # torch summarises bool and integer tensors too.
    else:
        flat = np.asarray(value).reshape(-1)
        summary["total_count"] = count = int(flat.size)
        if count <= MAX_EVIDENCE_ITEMS:
            return value.tolist(), summary
        numeric = bool(np.issubdtype(flat.dtype, np.number))
        source = flat
    if numeric:
        complex_values = np.iscomplexobj(flat)
        finite = np.isfinite(flat)
        finite_values = flat[finite]
        if finite_values.size and not complex_values:
            summary.update(
                finite_minimum=finite_values.min().item(), finite_maximum=finite_values.max().item()
            )
        summary.update(
            nan_count=int(np.isnan(flat).sum()),
            positive_infinity_count=0 if complex_values else int(np.isposinf(flat).sum()),
            negative_infinity_count=0 if complex_values else int(np.isneginf(flat).sum()),
        )
        if not bool(finite.all()):
            index = int((~finite).argmax())
            summary.update(
                first_non_finite_flat_index=index, first_non_finite_value=_json_value(flat[index].item())
            )
    return source[:MAX_EVIDENCE_ITEMS].tolist(), summary


def _complete_sequence_summary(value: Any) -> dict[str, Any] | None:
    """Summarize even numeric arrays in an omitted tail of a structured sequence."""
    combined: dict[str, Any] = dict.fromkeys(_COUNTERS, 0)
    for leaf in _sequence_leaves(value):
        if (
            hasattr(leaf, "tolist")
            and hasattr(leaf, "shape")
            and type(leaf).__module__.startswith(("torch", "numpy"))
        ):
            sample, child = _array_sample(leaf)
            if child["total_count"] <= MAX_EVIDENCE_ITEMS:
                _, child = _sequence_summary(sample)
        elif isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            _, child = _sequence_summary([leaf])
        else:
            return None
        if "first_non_finite_flat_index" in child and "first_non_finite_flat_index" not in combined:
            combined["first_non_finite_flat_index"] = (
                combined["total_count"] + child["first_non_finite_flat_index"]
            )
            combined["first_non_finite_value"] = child["first_non_finite_value"]
        for key in _COUNTERS:
            combined[key] += child.get(key, 0)
        if "finite_minimum" in child:
            combined["finite_minimum"] = min(
                combined.get("finite_minimum", child["finite_minimum"]), child["finite_minimum"]
            )
            combined["finite_maximum"] = max(
                combined.get("finite_maximum", child["finite_maximum"]), child["finite_maximum"]
            )
    return combined


def _record_truncation(
    truncations: list[dict[str, Any]], path: str, layout: str, summary: dict[str, Any]
) -> bool:
    """Note a truncated flat sample; return whether it was truncated."""
    if summary["total_count"] <= MAX_EVIDENCE_ITEMS:
        return False
    truncations.append({"path": path, "shown_count": MAX_EVIDENCE_ITEMS, "sample_layout": layout, **summary})
    return True


def _bounded(value: Any, path: str = "/details", truncations: list[dict[str, Any]] | None = None) -> Any:
    """Bound all sequence levels and retain summaries of the complete source."""
    if truncations is None:
        truncations = []
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key = str(_json_value(key))
            before = len(truncations)
            child_path = path + "/" + key.replace("~", "~0").replace("/", "~1")
            out[key] = _bounded(item, child_path, truncations)
            direct = next((entry for entry in truncations[before:] if entry["path"] == child_path), None)
            if direct is not None and f"{key}_total_count" not in value:
                out[f"{key}_total_count"] = direct["total_count"]
        return out
    if (
        hasattr(value, "tolist")
        and hasattr(value, "shape")
        and type(value).__module__.startswith(("torch", "numpy"))
    ):
        sample, summary = _array_sample(value)
        _record_truncation(truncations, path, "flattened row-major prefix", summary)
        return sample
    if isinstance(value, (list, tuple)):
        if _plain_sequence(value):
            sample, summary = _sequence_summary(value)
            if _record_truncation(truncations, path, "flattened sequence prefix", summary):
                return sample
        if len(value) > MAX_EVIDENCE_ITEMS:
            complete_summary = _complete_sequence_summary(value)
            truncations.append(
                {
                    "path": path,
                    "total_count": len(value),
                    "shown_count": MAX_EVIDENCE_ITEMS,
                    "sample_layout": "sequence prefix",
                    **({"numeric_summary": complete_summary} if complete_summary is not None else {}),
                }
            )
        return [
            _bounded(child, f"{path}/{index}", truncations)
            for index, child in enumerate(value[:MAX_EVIDENCE_ITEMS])
        ]
    return value


def index_list_text(items: Sequence[Any], total: int | None = None, *, limit: int = 16) -> str:
    """Render a list for a summary: in full up to ``limit`` items, else the first 8 and a total."""
    total = len(items) if total is None else total
    shown = items if total <= limit else items[:8]
    return "[" + ", ".join(map(str, shown)) + ("]" if total <= limit else f", ... ({total} in total)]")


def bounded_text(text: str) -> str:
    """Shorten long numeric lists inside a human-readable summary."""

    def shorten(match: re.Match[str]) -> str:
        items = [item.strip() for item in match.group(0)[1:-1].split(",")]
        if len(items) <= MAX_EVIDENCE_ITEMS:
            return match.group(0)
        return index_list_text(items, limit=MAX_EVIDENCE_ITEMS)

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
    digest_scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "step_name": _json_value(self.step_name),
            "tensor_key": self.tensor_key,
            "sha256_prefix": self.sha256_prefix,
            **({"digest_scope": self.digest_scope} if self.digest_scope is not None else {}),
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
        truncations: list[dict[str, Any]] = []
        bounded = _bounded(self.details, truncations=truncations)
        result = {
            "check_id": self.check_id,
            "status": self.status.value,
            "severity": self.severity.value,
            "summary": bounded_text(self.summary),
            "details": _json_value(bounded),
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
        }
        if truncations:
            result["evidence_truncation"] = truncations[:MAX_EVIDENCE_ITEMS]
            result["evidence_truncation_count"] = len(truncations)
        non_finite = _non_finite_values(bounded)
        if non_finite:
            result["non_finite_values"] = non_finite[:MAX_EVIDENCE_ITEMS]
            if len(non_finite) > MAX_EVIDENCE_ITEMS:
                result["non_finite_value_count"] = len(non_finite)
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

    def coverage(self) -> dict[str, Any]:
        """Which checks reached a verdict, and which could not be verified.

        Exit code 0 means no problem was found in the checks that could run. It does not
        mean every check ran; ``not_verified`` lists the ones that did not.
        """
        # Informational notes (undeclared units/frames) are listed separately, not counted as checks.
        checks = list(
            dict.fromkeys(f.check_id for f in self.findings if not f.details.get("strict_exempt", False))
        )
        unknown = [finding for finding in self.findings if finding.status is DiagnosticStatus.UNKNOWN]
        not_verified = list(
            dict.fromkeys(f.check_id for f in unknown if not f.details.get("strict_exempt", False))
        )
        not_declared = list(
            dict.fromkeys(
                str(f.details.get("field", f.check_id)) for f in unknown if f.details.get("strict_exempt")
            )
        )
        return {
            "checks_run": len(checks),
            "checks_verified": len([check for check in checks if check not in not_verified]),
            "not_verified": not_verified,
            "not_declared": not_declared,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "has_failures": self.has_failures(),
            "exit_code": self.exit_code(),
            "coverage": self.coverage(),
            "gate_policy": {
                "strict": self.strict,
                "baseline_requested": self.baseline_requested,
                "required_comparison_unknowns_fail": self.baseline_requested,
                "warning_advisories_fail": self.strict,
                "informational_physical_unknowns_exempt": True,
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }


def tool_error_finding(
    check_id: str, stage: str, exc: BaseException, severity: Severity, provenance: Provenance | None = None
) -> DiagnosticFinding:
    """A crash says nothing about the checkpoint, so it is UNKNOWN and marked as a tool error (exit 3)."""
    return DiagnosticFinding(
        check_id,
        DiagnosticStatus.UNKNOWN,
        severity,
        f"Flinch could not finish the {stage} check: {type(exc).__name__}: {exc}",
        {"stage": stage, "exception_type": type(exc).__name__, "tool_error": True},
        provenance,
    )
