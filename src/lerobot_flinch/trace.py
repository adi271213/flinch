"""Trace inspected native processor steps against local samples on the CPU."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    batch_to_transition,
    transition_to_batch,
)

try:
    from lerobot.processor.rename_processor import rename_keys
except ImportError:

    def rename_keys[T](mapping: Mapping[str, T], rename_map: Mapping[str, str] | None) -> dict[str, T]:
        """Use the installed native rename step when its helper is unavailable."""
        return cast(
            dict[str, T],
            RenameObservationsProcessorStep(rename_map=rename_map or {}).observation(dict(mapping)),
        )


from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)

from .guard import MODE_STATS, InputEnvelopeError, InputEnvelopeGuardStep
from .models import DiagnosticFinding, DiagnosticStatus, Provenance, Severity, index_list_text
from .preflight import (
    _CONFIG_NAMES,
    _local_file,
    _provenance,
    _read_tensors,
    _role,
    _step_class,
    _validate_step_entry,
)

_NORMALIZERS = (NormalizerProcessorStep, UnnormalizerProcessorStep)
_TRACE_CHECKS = ("SILENT_STEP_BYPASS", "EMPIRICAL_GAIN_EXPLOSION", "RUNTIME_NON_FINITE")
_LANGUAGE_OUTPUT_KEYS = frozenset(
    (
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_SUBTASK_TOKENS,
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    )
)


def _declares_class(entry: dict[str, Any], registry_name: str, cls: type) -> bool:
    """Match the same registry-first declaration precedence as the native loader."""
    if "registry_name" in entry:
        return cast(bool, entry["registry_name"] == registry_name)
    return entry.get("class") == f"{cls.__module__}.{cls.__name__}"


def _tokenizer_entry(entry: dict[str, Any]) -> bool:
    return _declares_class(entry, "tokenizer_processor", TokenizerProcessorStep)


def _trace_step_class(entry: dict[str, Any], numeric_only: bool) -> type | None:
    if numeric_only:
        if _tokenizer_entry(entry):
            return cast(type, IdentityProcessorStep)
        if _declares_class(entry, "smolvla_new_line_processor", NewLineTaskProcessorStep):
            return cast(type, NewLineTaskProcessorStep)
    return _step_class(entry)


def _token_dependencies(entry: dict[str, Any]) -> list[str]:
    """Inspect declared token consumers without importing checkpoint-specific code.

    Device/layout/identity steps only pass existing tensors through. Active
    normalization of language features, explicit rename sources, and other saved
    references to tokenizer output keys require real tokenization coverage.
    Unknown constructors are rejected independently even if they omit declarations.
    """
    cls = _trace_step_class(entry, True)
    config = entry.get("config", {})
    if cls in (
        IdentityProcessorStep,
        DeviceProcessorStep,
        AddBatchDimensionProcessorStep,
        NewLineTaskProcessorStep,
    ):
        return []
    if cls in _NORMALIZERS:
        modes = config.get("norm_map", {})
        return sorted(
            key
            for key, feature in config.get("features", {}).items()
            if key in _LANGUAGE_OUTPUT_KEYS and modes.get(feature.get("type"), "IDENTITY") != "IDENTITY"
        )
    if cls is RenameObservationsProcessorStep:
        return sorted(set(config.get("rename_map", {})) & _LANGUAGE_OUTPUT_KEYS)

    def references(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} & _LANGUAGE_OUTPUT_KEYS
        if isinstance(value, Mapping):
            return set().union(*(references(key) | references(child) for key, child in value.items()))
        if isinstance(value, (list, tuple)):
            return set().union(*(references(child) for child in value))
        return set()

    return sorted(references(config))


class _TraceLoadError(Exception):
    def __init__(
        self,
        status: DiagnosticStatus,
        summary: str,
        details: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        super().__init__(summary)
        self.status = status
        self.summary = summary
        self.details = details or {}
        self.provenance = provenance


def _step_provenance(
    config_path: Path,
    config: dict[str, Any],
    step_index: int,
    tensor_key: str,
    indices: list[int] | None = None,
) -> Provenance:
    entry = config["steps"][step_index]
    cls = _step_class(entry)
    path = config_path.parent / entry["state_file"] if "state_file" in entry else config_path
    return _provenance(path, cls.__name__ if cls else step_index, tensor_key, indices)


def _load_processor_pair(
    root: Path,
    *,
    numeric_only: bool = False,
) -> tuple[tuple[Any, Path, dict[str, Any]], tuple[Any, Path, dict[str, Any]]]:
    """Load a local native pair only after inspecting every constructor and path.

    Each returned tuple contains the native pipeline, JSON path, and saved config.
    Uninspected constructors are never invoked. Numeric-only tracing may replace
    the inspected language tokenizer with native IdentityProcessorStep in memory;
    the tokenizer constructor is never called. Other callers remain strict.
    """
    root = root.resolve()
    if not root.is_dir():
        raise _TraceLoadError(DiagnosticStatus.FAIL, "Checkpoint directory does not exist.")
    candidates: dict[str, list[tuple[Path, dict[str, Any]]]] = {"pre": [], "post": []}
    for path in sorted(root.glob("*.json")):
        try:
            _local_file(root, path.name)
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if path.name in _CONFIG_NAMES:
                raise _TraceLoadError(
                    DiagnosticStatus.FAIL, f"Cannot read processor JSON: {exc}", provenance=_provenance(path)
                ) from exc
            continue
        if isinstance(config, dict) and isinstance(config.get("steps"), list):
            role = _role(path, config)
            if role:
                candidates[role].append((path, config))
    selected = []
    for role in ("pre", "post"):
        entries = candidates[role]
        if not entries:
            raise _TraceLoadError(
                DiagnosticStatus.UNKNOWN, f"A modern {role}processor could not be identified locally."
            )
        if any(config != entries[0][1] for _, config in entries[1:]):
            raise _TraceLoadError(
                DiagnosticStatus.CONFLICT,
                f"Multiple {role}processor JSONs disagree.",
                {"processor_files": [str(path) for path, _ in entries]},
            )
        selected.append(entries[0])

    # Gate the entire pair before constructing even its first step.
    for path, config in selected:
        stubbed_indices: list[int] = []
        for index, entry in enumerate(config["steps"]):
            try:
                _validate_step_entry(entry)
                cls = _trace_step_class(entry, numeric_only)
            except (TypeError, ValueError) as exc:
                raise _TraceLoadError(
                    DiagnosticStatus.FAIL,
                    f"Invalid step declaration: {exc}",
                    {"step_index": index},
                    _provenance(path, index),
                ) from exc
            dependencies = (
                _token_dependencies(entry)
                if numeric_only and stubbed_indices and isinstance(entry, dict)
                else []
            )
            if dependencies:
                raise _TraceLoadError(
                    DiagnosticStatus.UNKNOWN,
                    "A downstream token-consuming step cannot be verified after language tokenization was stubbed.",
                    {
                        "step_index": index,
                        "step": entry,
                        "token_keys": dependencies,
                        "stubbed_tokenizer_step_indices": stubbed_indices,
                        "numeric_subpath_coverage": "not executed",
                        "language_coverage_verified": False,
                    },
                    _provenance(path, index),
                )
            if cls is None:
                raise _TraceLoadError(
                    DiagnosticStatus.UNKNOWN,
                    "An uninspected processor step prevents safe offline execution.",
                    {"step_index": index, "step": entry},
                    _provenance(path, index),
                )
            try:
                if entry.get("state_file") is not None:
                    _local_file(root, entry["state_file"])
                for filename in entry.get("artifacts", {}).values():
                    _local_file(root, filename, allow_directory=True)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                raise _TraceLoadError(
                    DiagnosticStatus.FAIL, str(exc), {"step_index": index}, _provenance(path, cls.__name__)
                ) from exc
            if numeric_only and _tokenizer_entry(entry):
                stubbed_indices.append(index)
    loaded = []
    for path, config in selected:
        overrides = {}
        sanitized = deepcopy(config)
        substituted = False
        for entry in config["steps"]:
            cls = _trace_step_class(entry, numeric_only)
            if cls is not None and cls in (*_NORMALIZERS, DeviceProcessorStep):
                overrides[entry.get("registry_name", cls.__name__)] = {"device": "cpu"}
        if numeric_only:
            for index, entry in enumerate(config["steps"]):
                if _tokenizer_entry(entry):
                    sanitized["steps"][index] = {
                        "class": f"{IdentityProcessorStep.__module__}.{IdentityProcessorStep.__name__}",
                        "config": {},
                    }
                    substituted = True
        try:
            if substituted:
                state = {
                    entry["state_file"].removesuffix(".safetensors"): _read_tensors(
                        _local_file(root, entry["state_file"])
                    )
                    for entry in sanitized["steps"]
                    if "state_file" in entry
                }
                pipeline = PolicyProcessorPipeline.from_config(
                    sanitized,
                    state_dict=state,
                    overrides=overrides,
                    to_transition=batch_to_transition,
                    to_output=transition_to_batch,
                )
            else:
                pipeline = PolicyProcessorPipeline.from_pretrained(
                    path,
                    config_filename=path.name,
                    local_files_only=True,
                    overrides=overrides,
                    to_transition=batch_to_transition,
                    to_output=transition_to_batch,
                )
        except Exception as exc:
            raise _TraceLoadError(
                DiagnosticStatus.FAIL, f"Native processor loading failed: {exc}", provenance=_provenance(path)
            ) from exc
        loaded.append((pipeline, path, config))
    relative = [step for step in loaded[0][0].steps if isinstance(step, RelativeActionsProcessorStep)]
    absolute = [step for step in loaded[1][0].steps if isinstance(step, AbsoluteActionsProcessorStep)]
    if len(relative) > 1 and any(step.enabled for step in absolute):
        raise _TraceLoadError(
            DiagnosticStatus.UNKNOWN, "Multiple relative steps make the absolute anchor ambiguous."
        )
    if len(relative) == 1:
        for step in absolute:
            step.relative_step = relative[0]
    return loaded[0], loaded[1]


def _language_stub_notes(artifact: tuple[Any, Path, dict[str, Any]], role: str) -> list[DiagnosticFinding]:
    pipeline, path, config = artifact
    return [
        DiagnosticFinding(
            "OFFLINE_LANGUAGE_TOKENIZATION",
            DiagnosticStatus.UNKNOWN,
            Severity.INFO,
            "Numeric state/action subpaths were traced with native IdentityProcessorStep replacing language tokenization; "
            "language tokenization and downstream token-dependent behavior remain unverified.",
            {
                "pipeline_role": role,
                "step_index": index,
                "original_step_class": "TokenizerProcessorStep",
                "original_registry_name": entry.get("registry_name"),
                "replacement_step_class": "IdentityProcessorStep",
                "numeric_only": True,
                "language_coverage_verified": False,
                "numeric_steps_real": True,
                "downstream_token_consumers_checked": True,
                "downstream_token_consumers": [],
                "coverage_scope": "numeric state/action subpaths only",
            },
            _provenance(path, "TokenizerProcessorStep"),
        )
        for index, entry in enumerate(config["steps"])
        if _tokenizer_entry(entry) and isinstance(pipeline.steps[index], IdentityProcessorStep)
    ]


def _sample(path: Path) -> dict[str, Any]:
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            if value.dtype.kind in "biuf":
                result[key] = torch.from_numpy(value.copy())
            elif value.dtype.kind in "SU":
                result[key] = value.tolist()
            else:
                raise ValueError(f"Unsupported sample dtype for {key}: {value.dtype}")
    return result


def _missing_sample_features(
    root: Path,
    artifact: tuple[Any, Path, dict[str, Any]],
    batch: dict[str, Any],
    sample_path: Path,
) -> list[DiagnosticFinding]:
    """Check numeric contracts before execution, following native key renames only.

    This is coverage bookkeeping, not a substitute for processor execution. Native
    ``rename_keys`` preserves the exact saved rename semantics, including collision
    checks. Tensors are neither normalized nor otherwise transformed here.
    """
    pipeline, config_path, config = artifact
    raw = {
        key: key
        for key, tensor in batch.items()
        if key.startswith("observation.") and isinstance(tensor, torch.Tensor)
    }
    available = raw.copy()
    missing: dict[str, dict[str, Any]] = {}

    def require(key: str, index: int | None, declaration: Path) -> None:
        if key in available:
            return
        missing.setdefault(
            key,
            {
                "feature_key": key,
                "step_index": index,
                "step_class": type(pipeline.steps[index]).__name__ if index is not None else None,
                "available_numeric_observation_keys_at_step": sorted(available),
                "declaration_provenance": _provenance(declaration, index, key).to_dict(),
            },
        )

    for index, step in enumerate(pipeline.steps):
        if isinstance(step, RenameObservationsProcessorStep):
            available = rename_keys(available, step.rename_map)
        elif isinstance(step, _NORMALIZERS):
            for key, feature in step.features.items():
                if feature.type not in (FeatureType.STATE, FeatureType.ENV):
                    continue
                if step.normalize_observation_keys is not None and key not in step.normalize_observation_keys:
                    continue
                if (
                    step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
                    is not NormalizationMode.IDENTITY
                ):
                    require(key, index, config_path)
        elif isinstance(step, RelativeActionsProcessorStep) and step.enabled:
            require("observation.state", index, config_path)

    policy_path = root / "config.json"
    if policy_path.is_file():
        policy = json.loads(_local_file(root, "config.json").read_text(encoding="utf-8"))
        for key, feature in policy.get("input_features", {}).items():
            if feature.get("type") in (FeatureType.STATE.value, FeatureType.ENV.value):
                # A config may describe raw dataset keys or the policy's canonical
                # keys; either declaration must be backed by an actual sample tensor.
                if key not in raw:
                    require(key, None, policy_path)

    return [
        DiagnosticFinding(
            "MISSING_REQUIRED_SAMPLE_FEATURE",
            DiagnosticStatus.FAIL,
            Severity.CRITICAL,
            f"The sample does not cover required numeric observation feature {key}.",
            {
                **details,
                "required_coverage": True,
                "stage": "sample_input",
                "processor_steps_executed": 0,
                "sample_numeric_observation_keys": sorted(raw),
                "sample_path": str(sample_path),
                "tensor_slice": [],
            },
            _provenance(sample_path, "sample_input", key, []),
        )
        for key, details in missing.items()
    ]


def _invalid_sample_shapes(
    root: Path,
    pre: tuple[Any, Path, dict[str, Any]],
    post: tuple[Any, Path, dict[str, Any]],
    batch: dict[str, Any],
    sources: dict[str, str],
) -> list[DiagnosticFinding]:
    """Validate saved feature shapes without executing or broadcasting any tensor.

    Only the leading batch/history axes may be absent or added. Native rename
    bookkeeping follows the source NPZ key while trailing feature axes stay exact.
    """
    raw = {key: (key, tensor) for key, tensor in batch.items() if isinstance(tensor, torch.Tensor)}
    observations = {key: item for key, item in raw.items() if key.startswith("observation.")}
    current = observations.copy()
    findings: dict[tuple[str, tuple[int, ...]], DiagnosticFinding] = {}

    def inspect(
        key: str,
        shape: Any,
        available: dict[str, tuple[str, torch.Tensor]],
        declaration: Path,
        index: int | None,
        step_class: str | None,
    ) -> None:
        if key not in available:
            return  # Missing required features have a separate coverage finding.
        if not isinstance(shape, (tuple, list)) or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"Invalid declared feature shape for {key}: {shape!r}")
        expected = tuple(shape)
        original, tensor = available[key]
        actual = tuple(tensor.shape)
        prefix = len(actual) - len(expected)
        trailing = actual[-len(expected) :] if expected else ()
        if 0 <= prefix <= 2 and trailing == expected and all(dim > 0 for dim in actual):
            return
        source = Path(sources["action"] if original == "action" else sources["observation"])
        unequal_axes = [
            axis
            for axis in range(max(len(expected), len(trailing)))
            if axis >= len(expected) or axis >= len(trailing) or expected[axis] != trailing[axis]
        ]
        findings.setdefault(
            (original, expected),
            DiagnosticFinding(
                "INVALID_SAMPLE_SHAPE",
                DiagnosticStatus.FAIL,
                Severity.CRITICAL,
                f"Sample feature {original} has shape {list(actual)}; trailing feature shape must be {list(expected)}.",
                {
                    "stage": "sample_input",
                    "step_index": index,
                    "step_class": step_class,
                    "feature_key": key,
                    "sample_feature_key": original,
                    "actual_shape": list(actual),
                    "expected_feature_shape": list(expected),
                    "actual_trailing_shape": list(trailing),
                    "leading_axis_count": prefix,
                    "allowed_leading_axes": ["batch", "time"],
                    "violating_feature_shape_axes": unequal_axes,
                    "indexing": "feature shape axes",
                    "processor_steps_executed": 0,
                    "tensor_slice": tensor.reshape(-1)[:16].tolist(),
                    "tensor_slice_truncated": tensor.numel() > 16,
                    "declaration_provenance": _provenance(declaration, index, key).to_dict(),
                },
                _provenance(source, "sample_input", original, unequal_axes),
            ),
        )

    pipeline, config_path, _ = pre
    for index, step in enumerate(pipeline.steps):
        if isinstance(step, RenameObservationsProcessorStep):
            current = rename_keys(current, step.rename_map)
        available = {**current, **({"action": raw["action"]} if "action" in raw else {})}
        for key, feature in getattr(step, "features", {}).items():
            inspect(key, feature.shape, available, config_path, index, type(step).__name__)

    post_pipeline, post_path, _ = post
    for index, step in enumerate(post_pipeline.steps):
        for key, feature in getattr(step, "features", {}).items():
            if key == "action":
                inspect(key, feature.shape, raw, post_path, index, type(step).__name__)

    policy_path = root / "config.json"
    if policy_path.is_file():
        policy = json.loads(_local_file(root, "config.json").read_text(encoding="utf-8"))
        for field in ("input_features", "output_features"):
            for key, feature in policy.get(field, {}).items():
                # Configs can describe raw dataset keys or canonical policy keys.
                available = raw if key in raw else current
                inspect(key, feature.get("shape"), available, policy_path, None, None)
    return list(findings.values())


def _tensors(transition: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    result = {}

    def visit(value: Any, key: str) -> None:
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, Mapping):
            for name, child in value.items():
                visit(child, f"{key}.{name}" if key else str(name))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{key}.{index}")

    for key, value in transition.items():
        visit(value, "" if key == "observation" else key)
    return result


def _slice_details(
    before: torch.Tensor | None,
    after: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[dict[str, Any], list[int]]:
    indices = mask.reshape(-1).nonzero().flatten().tolist()
    outputs = after.reshape(-1)[indices].tolist()
    inputs = (
        before.reshape(-1)[indices].tolist() if before is not None and before.shape == after.shape else None
    )
    return {
        "tensor_coordinates": mask.nonzero().tolist(),
        "input_values": inputs,
        "raw_input_values": inputs,
        "output_values": outputs,
        "tensor_slice": outputs,
        "tensor_shape": list(after.shape),
        "indexing": "flattened row-major",
        "input_dtype": str(before.dtype) if before is not None else None,
        "output_dtype": str(after.dtype),
    }, indices


def _nonidentity(state: dict[str, torch.Tensor], key: str, mode: NormalizationMode) -> bool:
    stats = {
        name: state.get(f"{key}.{name}") for name in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90")
    }
    if mode == NormalizationMode.MEAN_STD:
        return any(
            tensor is not None and bool(torch.any(tensor != identity))
            for tensor, identity in ((stats["mean"], 0), (stats["std"], 1))
        )
    bounds = MODE_STATS.get(mode)
    return bool(bounds) and any(
        stats[name] is not None and bool(torch.any(stats[name] != identity))
        for name, identity in zip(bounds or (), (-1, 1))
    )


def _bypasses(
    step: Any,
    state: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> tuple[list[tuple[str, dict[str, Any], list[int]]], int]:
    """Look for exact key misses and unchanged, non-fixed-point transformations."""
    if not isinstance(step, _NORMALIZERS):
        return [], 0
    expected = {key: ft.type for key, ft in step.features.items() if ft.type != FeatureType.ACTION}
    if "action" in before:
        expected["action"] = FeatureType.ACTION
    actual_keys = sorted({key.rsplit(".", 1)[0] for key in state})
    findings, checked = [], 0
    for key, feature_type in expected.items():
        mode = step.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY or key not in before or key not in after:
            continue
        if (
            key != "action"
            and step.normalize_observation_keys is not None
            and key not in step.normalize_observation_keys
        ):
            continue
        checked += 1
        missing = key not in actual_keys
        nonidentity = (
            _nonidentity(state, key, mode)
            if not missing
            else any(_nonidentity(state, actual, mode) for actual in actual_keys)
        )
        previous, current = before[key], after[key]
        required = MODE_STATS.get(mode, ())
        details = {
            "expected_keys": [f"{key}.{stat}" for stat in required],
            "expected_feature_keys": sorted(expected),
            "actual_stat_keys": sorted(state),
            "actual_feature_keys": actual_keys,
            "normalization_mode": mode.value,
        }
        if missing:
            mask = torch.ones_like(current, dtype=torch.bool)
            slices, indices = _slice_details(previous, current, mask)
            findings.append(
                (
                    key,
                    {
                        **details,
                        **slices,
                        "reason": "stat_key_mismatch",
                        "linf_delta": 0.0,
                        "inconclusive": not nonidentity,
                    },
                    indices,
                )
            )
        elif not nonidentity:
            continue
        elif (
            previous.shape == current.shape
            and previous.dtype == current.dtype
            and torch.equal(previous, current)
            and torch.isfinite(current).all()
        ):
            # Equality alone is insufficient: nonidentity affine transforms have
            # fixed points, and small changes can round away in the input dtype.
            if mode != NormalizationMode.MEAN_STD:
                continue
            mean, std = state.get(f"{key}.mean"), state.get(f"{key}.std")
            if mean is None or std is None:
                continue
            # Native normalization first casts buffer constants to the input
            # dtype. Reproduce that arithmetic for the diagnostic expectation,
            # including its intermediate rounding, rather than a float64 ideal.
            x, mean, std = previous, mean.to(previous.dtype), std.to(previous.dtype)
            target = (
                x * std + mean
                if isinstance(step, UnnormalizerProcessorStep)
                else (x - mean) / (std + step.eps)
            )
            changed = torch.isfinite(target) & (target != previous)
            # Requiring a representable expected change avoids rounding/fixed-point false positives.
            if changed.any():
                slices, indices = _slice_details(previous, current, changed)
                findings.append(
                    (
                        key,
                        {**details, **slices, "reason": "unchanged_nonidentity_transform", "linf_delta": 0.0},
                        indices,
                    )
                )
    return findings, checked


class _StopTrace(Exception):
    pass


def _gain_mask(
    step: Any,
    state: dict[str, torch.Tensor],
    key: str,
    previous: torch.Tensor | None,
    current: torch.Tensor,
    threshold: float,
    amplification_factor: float,
) -> torch.Tensor:
    """Require both a large output and measured elementwise magnitude growth."""
    large = torch.isfinite(current) & (current.double().abs() > threshold)
    if isinstance(
        step,
        (
            IdentityProcessorStep,
            AddBatchDimensionProcessorStep,
            DeviceProcessorStep,
            RenameObservationsProcessorStep,
            NewLineTaskProcessorStep,
            InputEnvelopeGuardStep,
        ),
    ):
        return torch.zeros_like(large)
    if isinstance(step, _NORMALIZERS):
        if key == "action":
            feature_type = FeatureType.ACTION
        else:
            feature = step.features.get(key)
            if feature is None or (
                step.normalize_observation_keys is not None and key not in step.normalize_observation_keys
            ):
                return torch.zeros_like(large)
            feature_type = feature.type
        mode = step.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        required = MODE_STATS.get(mode)
        if not required or any(f"{key}.{stat}" not in state for stat in required):
            return torch.zeros_like(large)
    if previous is None or previous.numel() != current.numel():
        return torch.zeros_like(large)
    # Relative/absolute action conversions can change values. A tensor that only
    # changes layout or preserves a pre-existing magnitude has not gained here.
    return large & (
        current.double().abs() > previous.reshape(current.shape).double().abs() * amplification_factor
    )


def _trace_pipeline(
    artifact: tuple[Any, Path, dict[str, Any]],
    data: dict[str, Any],
    role: str,
    threshold: float,
    findings: list[DiagnosticFinding],
    sources: dict[str, str],
    amplification_factor: float,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    pipeline, config_path, config = artifact
    before: dict[str, torch.Tensor] = {}
    observation_tensor_keys: set[str] = set()
    active_index: int | None = None
    coverage = {"steps": 0, "bypass_targets": 0, "preprocessor_tensors": 0, "postprocessor_tensors": 0}
    step_states = [
        {key: value.clone() for key, value in step.state_dict().items()} for step in pipeline.steps
    ]

    def emit(
        check: str,
        index: int,
        key: str,
        details: dict[str, Any],
        indices: list[int],
        status: DiagnosticStatus = DiagnosticStatus.FAIL,
    ) -> None:
        # The first axis is the batch/frame axis. For a multi-frame rollout, report the
        # feature coordinate and how many frames hit it, not positions in the stacked array.
        shape = details.get("tensor_shape") or []
        frames = shape[0] if len(shape) >= 2 else 1
        width = max(1, math.prod(shape[1:])) if len(shape) >= 2 else max(1, math.prod(shape))
        where = ""
        if frames > 1 and indices:
            coordinates = sorted({flat % width for flat in indices})
            details = {
                **details,
                "flattened_indices": indices,
                "feature_coordinates": coordinates,
                "frames_affected": len({flat // width for flat in indices}),
                "frames_total": frames,
            }
            where = (
                f", coordinate(s) {index_list_text(coordinates, limit=8)} "
                f"in {details['frames_affected']} of {frames} frames"
            )
            indices = coordinates
        findings.append(
            DiagnosticFinding(
                check,
                status,
                Severity.CRITICAL if status is DiagnosticStatus.FAIL else Severity.INFO,
                (
                    f"{check} at {role} step {index}, {key}{where}."
                    if status is DiagnosticStatus.FAIL
                    else f"Missing stat keys leave the intended nonidentity transform unknown at {role} step {index}, {key}."
                ),
                {
                    "pipeline_role": role,
                    "step_index": index,
                    "step_class": type(pipeline.steps[index]).__name__,
                    "tensor_key": key,
                    "feature_key": key,
                    "execution_device": "cpu",
                    "sample_sources": sources,
                    **details,
                },
                _step_provenance(config_path, config, index, key, indices),
            )
        )

    def before_hook(index: int, transition: dict[str, Any]) -> None:
        nonlocal before, active_index, observation_tensor_keys
        active_index = index
        before = _tensors(transition)
        observation = transition.get("observation", {})
        observation_tensor_keys = (
            {key for key, value in observation.items() if isinstance(value, torch.Tensor)}
            if isinstance(observation, dict)
            else set()
        )

    def after_hook(index: int, transition: dict[str, Any]) -> None:
        after = _tensors(transition)
        coverage["steps"] += 1
        step = pipeline.steps[index]
        if isinstance(step, InputEnvelopeGuardStep):
            checked_keys = sorted(
                key
                for key, value in before.items()
                if key in observation_tensor_keys and value.is_floating_point() and step.selects(key)
            )
            clamped_keys = [key for key in checked_keys if not torch.equal(before[key], after[key])]
            findings.append(
                DiagnosticFinding(
                    "INPUT_ENVELOPE_GUARD",
                    DiagnosticStatus.PASS if checked_keys else DiagnosticStatus.UNKNOWN,
                    Severity.INFO,
                    f"Input envelope guard executed at {role} step {index}; "
                    + (
                        "finite outliers were clamped in the selected sample tensors."
                        if clamped_keys
                        else "selected sample tensors passed the guard."
                        if checked_keys
                        else "no selected floating sample tensor was present."
                    ),
                    {
                        "pipeline_role": role,
                        "step_index": index,
                        "step_class": type(step).__name__,
                        "max_abs": step.max_abs,
                        "on_violation": step.on_violation,
                        "keys": step.keys,
                        "checked_tensor_keys": checked_keys,
                        "clamped_tensor_keys": clamped_keys,
                        "guard_scope": "selected floating observation tensors at this insertion point only",
                        "execution_device": "cpu",
                        "sample_sources": sources,
                    },
                    _step_provenance(
                        config_path, config, index, checked_keys[0] if checked_keys else "observation"
                    ),
                )
            )
        bypasses, checked = _bypasses(pipeline.steps[index], step_states[index], before, after)
        coverage["bypass_targets"] += checked
        for key, details, indices in bypasses:
            status = DiagnosticStatus.UNKNOWN if details.pop("inconclusive", False) else DiagnosticStatus.FAIL
            emit("SILENT_STEP_BYPASS", index, key, details, indices, status)
        for key, tensor in after.items():
            finite = torch.isfinite(tensor)
            coverage[f"{role}_tensors"] += 1
            mask = _gain_mask(
                pipeline.steps[index],
                step_states[index],
                key,
                before.get(key),
                tensor,
                threshold,
                amplification_factor,
            )
            if mask.any():
                details, indices = _slice_details(before.get(key), tensor, mask)
                previous = before.get(key)
                details["already_above_threshold_on_entry"] = (
                    (previous.double().abs() > threshold).reshape(-1)[indices].tolist()
                    if previous is not None and previous.shape == tensor.shape
                    else None
                )
                emit(
                    "EMPIRICAL_GAIN_EXPLOSION",
                    index,
                    key,
                    {
                        **details,
                        "explosion_threshold": threshold,
                        "amplification_factor": amplification_factor,
                        "amplification_rule": "abs(output) > abs(input) * amplification_factor",
                    },
                    indices,
                )
            if not finite.all():
                details, indices = _slice_details(before.get(key), tensor, ~finite)
                previous = before.get(key)
                details["nonfinite_on_entry"] = (
                    (~torch.isfinite(previous)).reshape(-1)[indices].tolist()
                    if previous is not None and previous.shape == tensor.shape
                    else None
                )
                emit("RUNTIME_NON_FINITE", index, key, details, indices)
                # Stop at the first offending step so downstream steps cannot obscure its origin.
                raise _StopTrace

    pipeline.register_before_step_hook(before_hook)
    pipeline.register_after_step_hook(after_hook)
    try:
        with torch.inference_mode():
            return pipeline(data), coverage
    except _StopTrace:
        return None, coverage
    except InputEnvelopeError as exc:
        assert active_index is not None
        step = pipeline.steps[active_index]
        tensor = before.get(exc.key)
        emit(
            "INPUT_ENVELOPE_VIOLATION",
            active_index,
            exc.key,
            {
                "max_abs": step.max_abs,
                "on_violation": step.on_violation,
                "keys": step.keys,
                "input_values": exc.values,
                "tensor_slice": exc.values,
                "violating_index_count": exc.total_count,
                "tensor_shape": list(tensor.shape) if tensor is not None else None,
                "indexing": "flattened row-major",
                "processor_output_available": False,
                "guard_scope": "selected floating observation tensors at this insertion point only",
            },
            exc.indices,
        )
        assert findings[-1].provenance is not None
        findings[-1].provenance.violating_index_count = exc.total_count
        return None, coverage
    except Exception as exc:
        key = next(iter(before), "transition")
        findings.append(
            DiagnosticFinding(
                "TRACE_EXECUTION",
                DiagnosticStatus.FAIL,
                Severity.CRITICAL,
                f"Native {role} {'step execution' if active_index is not None else 'input conversion'} failed: {exc}",
                {
                    "pipeline_role": role,
                    "step_index": active_index,
                    "step_class": type(pipeline.steps[active_index]).__name__
                    if active_index is not None
                    else None,
                    "stage": "processor_step" if active_index is not None else "input_conversion",
                    "tensor_key": key,
                    "input_tensor_keys": sorted(before),
                },
                _step_provenance(config_path, config, active_index, key)
                if active_index is not None
                else _provenance(config_path),
            )
        )
        return None, coverage
    finally:
        pipeline.unregister_before_step_hook(before_hook)
        pipeline.unregister_after_step_hook(after_hook)


def run_offline_trace(
    checkpoint_dir: Path,
    sample_npz_path: Path,
    explosion_threshold: float = 50.0,
    amplification_factor: float = 2.0,
) -> list[DiagnosticFinding]:
    """Run only native, inspected data processors against real local sample tensors.

    Numeric sample dtypes are preserved, including FP16. An adjacent
    sample_action.npz supplies action when it is not in the observation archive.
    Postprocessing receives the preprocessed sample action, without a policy call.
    Gain findings require an absolute output above explosion_threshold AND more
    than amplification_factor (default 2x) elementwise magnitude growth.
    """
    if not math.isfinite(explosion_threshold) or explosion_threshold <= 0:
        raise ValueError("explosion_threshold must be positive and finite")
    if not math.isfinite(amplification_factor) or amplification_factor <= 1:
        raise ValueError("amplification_factor must be finite and greater than one")
    findings: list[DiagnosticFinding] = []
    try:
        pre, post = _load_processor_pair(Path(checkpoint_dir), numeric_only=True)
    except _TraceLoadError as exc:
        return [
            DiagnosticFinding(
                "OFFLINE_TRACE",
                exc.status,
                Severity.CRITICAL
                if exc.status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
                else Severity.INFO,
                exc.summary,
                exc.details,
                exc.provenance,
            )
        ]
    sample_path = Path(sample_npz_path).resolve()
    sources = {"observation": str(sample_path)}
    try:
        batch = _sample(sample_path)
        if "action" in batch:
            sources["action"] = str(sample_path)
        else:
            action_path = sample_path.parent / "sample_action.npz"
            if action_path.is_file() and action_path != sample_path:
                action = _sample(action_path).get("action")
                if not isinstance(action, torch.Tensor):
                    raise ValueError("sample_action.npz does not contain a numeric action tensor")
                batch["action"] = action
                sources["action"] = str(action_path)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return [
            DiagnosticFinding(
                "TRACE_INPUT",
                DiagnosticStatus.FAIL,
                Severity.CRITICAL,
                f"Cannot load sample tensors: {exc}",
                provenance=_provenance(sample_path),
            )
        ]
    for key, tensor in _tensors(batch).items():
        nonfinite = ~torch.isfinite(tensor)
        if nonfinite.any():
            details, indices = _slice_details(tensor, tensor, nonfinite)
            source = Path(sources["action"] if key == "action" else sources["observation"])
            findings.append(
                DiagnosticFinding(
                    "INVALID_SAMPLE_INPUT",
                    DiagnosticStatus.FAIL,
                    Severity.CRITICAL,
                    f"Sample tensor {key} contains NaN or Inf before processor step 0.",
                    {
                        **details,
                        "stage": "sample_input",
                        "step_index": None,
                        "tensor_key": key,
                        "sample_sources": sources,
                        "processor_steps_executed": 0,
                    },
                    _provenance(source, "sample_input", key, indices),
                )
            )
    if findings:
        return findings
    try:
        findings.extend(_invalid_sample_shapes(Path(checkpoint_dir).resolve(), pre, post, batch, sources))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return [
            DiagnosticFinding(
                "INVALID_SAMPLE_SHAPE",
                DiagnosticStatus.FAIL,
                Severity.CRITICAL,
                f"Declared sample shape compatibility could not be established: {exc}",
                {"stage": "sample_input", "processor_steps_executed": 0},
                _provenance(sample_path, "sample_input"),
            )
        ]
    if findings:
        return findings
    try:
        findings.extend(_missing_sample_features(Path(checkpoint_dir).resolve(), pre, batch, sample_path))
        if not findings and not any(
            key.startswith("observation.") and isinstance(value, torch.Tensor) for key, value in batch.items()
        ):
            raise ValueError("The sample archive has no numeric observation tensors")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return [
            DiagnosticFinding(
                "MISSING_REQUIRED_SAMPLE_FEATURE",
                DiagnosticStatus.FAIL,
                Severity.CRITICAL,
                f"Required numeric sample coverage could not be established: {exc}",
                {"required_coverage": True, "stage": "sample_input", "processor_steps_executed": 0},
                _provenance(sample_path, "sample_input"),
            )
        ]
    if findings:
        return findings
    findings.extend(_language_stub_notes(pre, "preprocessor"))
    findings.extend(_language_stub_notes(post, "postprocessor"))
    output, coverage = _trace_pipeline(
        pre, batch, "preprocessor", explosion_threshold, findings, sources, amplification_factor
    )
    complete = output is not None
    if output is not None and isinstance(output.get("action"), torch.Tensor):
        post_output, post_coverage = _trace_pipeline(
            post,
            {"action": output["action"]},
            "postprocessor",
            explosion_threshold,
            findings,
            sources,
            amplification_factor,
        )
        complete = post_output is not None
        coverage["steps"] += post_coverage["steps"]
        coverage["bypass_targets"] += post_coverage["bypass_targets"]
        coverage["postprocessor_tensors"] += post_coverage["postprocessor_tensors"]
    else:
        complete = False
        findings.append(
            DiagnosticFinding(
                "POSTPROCESSOR_TRACE",
                DiagnosticStatus.UNKNOWN,
                Severity.INFO,
                "Postprocessing was not exercised because no valid preprocessed sample action is available.",
                {"sample_sources": sources},
                _provenance(post[1]),
            )
        )
    counts = {
        "SILENT_STEP_BYPASS": coverage["bypass_targets"],
        "EMPIRICAL_GAIN_EXPLOSION": coverage["preprocessor_tensors"] + coverage["postprocessor_tensors"],
        "RUNTIME_NON_FINITE": coverage["steps"],
    }
    for check in _TRACE_CHECKS:
        if not any(finding.check_id == check for finding in findings):
            observed = counts[check]
            findings.append(
                DiagnosticFinding(
                    check,
                    DiagnosticStatus.PASS if observed and complete else DiagnosticStatus.UNKNOWN,
                    Severity.INFO,
                    "No violation observed in the traced sample."
                    if observed and complete
                    else "The complete check could not be observed with the available sample and trace.",
                    {
                        "observed_count": observed,
                        "trace_complete": complete,
                        "sample_sources": sources,
                        "execution_device": "cpu",
                    },
                )
            )
    return findings
