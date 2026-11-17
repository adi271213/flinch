"""Read-only temporal anchor audit using LeRobot's native processor steps.

The probe uses synthetic tensors. Its chronology comes from declared frame
deltas or explicit metadata; tensor shape alone does not identify the current frame.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from lerobot.configs import FeatureType
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .models import DiagnosticFinding, DiagnosticStatus, Provenance, Severity
from .preflight import _MODE_STATS, _provenance

_CHECK_ID = "TEMPORAL_ANCHOR_MISMATCH"
_ROUNDTRIP_ATOL = 1e-5
_ROUNDTRIP_RTOL = 1e-5
_ROUNDTRIP_SAFETY_FACTOR = 4.0
_ORDER_CURRENT_INDEX = {
    "oldest_first": -1,
    "current_last": -1,
    "newest_first": 0,
    "current_first": 0,
}


def _delta_chronology(indices: Any) -> str | None:
    """Identify delta-zero's declared location, without using tensor shape.

    The native relative-action step deliberately collapses stacked state to
    index zero. That matches current-first windows, but not histories whose
    sole current (delta-zero) entry is at the end.
    """
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)) or len(indices) < 2:
        return None
    if not all(type(value) is int or (type(value) is float and math.isfinite(value)) for value in indices):
        return None
    if indices[-1] == 0 and all(value < 0 for value in indices[:-1]):
        return "oldest_first"
    if indices[0] == 0 and all(value != 0 for value in indices[1:]):
        return "current_first"
    return None


def _finding(
    status: DiagnosticStatus,
    summary: str,
    details: dict[str, Any] | None = None,
    provenance: Provenance | None = None,
) -> DiagnosticFinding:
    return DiagnosticFinding(
        check_id=_CHECK_ID,
        status=status,
        severity=(
            Severity.CRITICAL
            if status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
            else Severity.INFO
        ),
        summary=summary,
        details=details or {},
        provenance=provenance,
    )


def _declarations(
    root: Path, metadata_path: Path | None = None,
) -> tuple[dict[str, list[tuple[Path, Any]]], list[Path]]:
    """Read native config and only the caller's explicitly selected sidecar."""
    declared: dict[str, list[tuple[Path, Any]]] = {
        "n_obs_steps": [],
        "observation_order": [],
        "current_index": [],
        "observation_delta_indices": [],
        "state_observation_delta_indices": [],
    }
    invalid: list[Path] = []
    config_path = root / "config.json"
    paths = [config_path] if config_path.is_file() and config_path.resolve().is_relative_to(root) else []
    if metadata_path is not None:
        explicit = Path(metadata_path).resolve()
        if explicit not in paths:
            paths.append(explicit)
    for path in paths:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            invalid.append(path)
            continue
        if not isinstance(metadata, dict):
            invalid.append(path)
            continue
        sources = [metadata]
        for nested_key in ("physical_semantics", "metadata"):
            nested = metadata.get(nested_key)
            if isinstance(nested, dict):
                sources.append(nested)
        for source in sources:
            if path == config_path:
                for key in ("observation_delta_indices", "state_observation_delta_indices"):
                    if source.get(key) is not None:
                        declared[key].append((path, source[key]))
            for key in ("n_obs_steps", "current_index"):
                if source.get(key) is not None:
                    declared[key].append((path, source[key]))
            for key in ("observation_order", "temporal_observation_ordering"):
                if source.get(key) is not None:
                    declared["observation_order"].append((path, source[key]))
    return declared, invalid


def _resolve_metadata(
    root: Path, metadata_path: Path | None = None,
) -> tuple[dict[str, Any] | None, DiagnosticFinding | None]:
    declared, invalid = _declarations(root, metadata_path)
    if invalid:
        path = invalid[0]
        return None, _finding(
            DiagnosticStatus.UNKNOWN,
            "Temporal metadata could not be read.",
            {"invalid_metadata_files": [str(item) for item in invalid]},
            _provenance(path),
        )

    for field, entries in declared.items():
        if field == "current_index":
            # -1 and n_obs_steps - 1 can name the same frame.
            continue
        if entries and any(value != entries[0][1] for _, value in entries[1:]):
            return None, _finding(
                DiagnosticStatus.CONFLICT,
                f"Explicit temporal declarations disagree for {field}.",
                {
                    "field": "temporal_observation_ordering" if field == "observation_order" else field,
                    "declarations": [
                        {"file_path": str(path), "value": value} for path, value in entries
                    ],
                },
                _provenance(entries[0][0], tensor_key=field),
            )

    delta_entries = declared["observation_delta_indices"]
    deltas = delta_entries[0][1] if delta_entries else None
    state_entries = declared["state_observation_delta_indices"]
    state_deltas = state_entries[0][1] if state_entries else None
    effective_entries = state_entries or delta_entries
    effective_deltas = state_deltas if state_entries else deltas
    delta_key = "state_observation_delta_indices" if state_entries else "observation_delta_indices"
    delta_source = f"config.{delta_key}"
    delta_provenance = _provenance(effective_entries[0][0], tensor_key=delta_key) if effective_entries else None
    if state_entries and (
        not isinstance(state_deltas, (list, tuple)) or not state_deltas
        or not all(type(value) is int or (type(value) is float and math.isfinite(value)) for value in state_deltas)
        or (len(state_deltas) == 1 and state_deltas[0] != 0)
    ):
        return None, _finding(
            DiagnosticStatus.UNKNOWN,
            "State-specific delta indices are invalid; generic observation deltas cannot replace that contract.",
            {"observation_delta_indices": deltas, "state_observation_delta_indices": state_deltas,
             "effective_state_delta_indices": state_deltas, "state_delta_indices_source": delta_source,
             "required_coverage": True}, delta_provenance,
        )
    delta_count = (
        len(effective_deltas)
        if isinstance(effective_deltas, (list, tuple)) and (len(effective_deltas) >= 2 or state_entries)
        else None
    )
    inferred_order = _delta_chronology(effective_deltas)
    declared_count = declared["n_obs_steps"][0][1] if declared["n_obs_steps"] else None
    # Dataset timestamp construction follows observation delta indices, which
    # can span many frames even when policy n_obs_steps is one (VLA-JEPA,
    # LingBot-VA). Probe that declared window rather than inventing a conflict
    # between two config fields with different meanings.
    count = delta_count if delta_count is not None else declared_count
    order = declared["observation_order"][0][1] if declared["observation_order"] else None
    current = declared["current_index"][0][1] if declared["current_index"] else None
    explicit_chronology = bool(declared["observation_order"] or declared["current_index"])
    if not explicit_chronology and inferred_order is not None:
        order = inferred_order
    missing: list[str] = []
    if count is None:
        missing.append("n_obs_steps")
    if order is None and current is None and count != 1:
        missing.append("temporal_observation_ordering")
    if missing:
        return None, _finding(
            DiagnosticStatus.UNKNOWN,
            "The current observation frame cannot be identified from checkpoint metadata.",
            {"missing_declarations": missing, "observation_delta_indices": deltas,
             "state_observation_delta_indices": state_deltas, "effective_state_delta_indices": effective_deltas,
             "state_delta_indices_source": delta_source if effective_entries else None},
            delta_provenance,
        )
    if type(count) is not int or count < 1:
        return None, _finding(
            DiagnosticStatus.UNKNOWN,
            "n_obs_steps must be a positive integer for a temporal anchor probe.",
            {"n_obs_steps": count},
            _provenance(declared["n_obs_steps"][0][0], tensor_key="n_obs_steps") if declared["n_obs_steps"] else None,
        )
    if order is not None and (not isinstance(order, str) or order not in _ORDER_CURRENT_INDEX):
        return None, _finding(
            DiagnosticStatus.UNKNOWN,
            "Observation ordering is not one of the explicitly supported conventions.",
            {"observation_order": order, "supported_orders": sorted(_ORDER_CURRENT_INDEX)},
            _provenance(declared["observation_order"][0][0], tensor_key="observation_order"),
        )
    if current is not None and (type(current) is not int or not -count <= current < count):
        return None, _finding(
            DiagnosticStatus.UNKNOWN,
            "current_index is outside the declared observation window.",
            {"n_obs_steps": count, "current_index": current},
            _provenance(declared["current_index"][0][0], tensor_key="current_index"),
        )
    current_entries = declared["current_index"]
    for path, value in current_entries[1:]:
        if type(value) is not int or not -count <= value < count:
            return None, _finding(
                DiagnosticStatus.UNKNOWN,
                "A declared current_index is outside the observation window.",
                {"n_obs_steps": count, "current_index": value},
                _provenance(path, tensor_key="current_index"),
            )
        if value % count != current % count:
            return None, _finding(
                DiagnosticStatus.CONFLICT,
                "Explicit current_index declarations identify different frames.",
                {
                    "field": "current_index",
                    "n_obs_steps": count,
                    "declarations": [
                        {"file_path": str(source), "value": declared_value}
                        for source, declared_value in current_entries
                    ],
                },
                _provenance(current_entries[0][0], tensor_key="current_index"),
            )

    from_order = _ORDER_CURRENT_INDEX[order] % count if order is not None else None
    from_current = current % count if current is not None else None
    if from_order is not None and from_current is not None and from_order != from_current:
        path = declared["current_index"][0][0]
        return None, _finding(
            DiagnosticStatus.CONFLICT,
            "Observation order and current_index identify different current frames.",
            {
                "field": "current_index",
                "n_obs_steps": count,
                "observation_order": order,
                "current_index": current,
                "order_implied_index": from_order,
                "declarations": [
                    {"file_path": str(p), "value": v}
                    for p, v in declared["observation_order"] + declared["current_index"]
                ],
            },
            _provenance(path, tensor_key="current_index"),
        )
    chronology_source = (
        "explicit_metadata" if explicit_chronology and metadata_path is not None and any(
            path == Path(metadata_path).resolve()
            for path, _ in declared["observation_order"] + declared["current_index"]
        ) else "config_explicit_chronology" if explicit_chronology else
        delta_source if inferred_order is not None or (state_entries and state_deltas == [0]) else "single_frame"
    )
    return {
        "n_obs_steps": count,
        "declared_n_obs_steps": declared_count,
        "window_length_source": delta_source if delta_count is not None else "n_obs_steps",
        "observation_order": order,
        "declared_current_index": current,
        "observation_delta_indices": deltas,
        "state_observation_delta_indices": state_deltas,
        "effective_state_delta_indices": effective_deltas,
        "state_delta_indices_source": delta_source if effective_entries else None,
        "chronology_source": chronology_source,
        "chronology_provenance": (
            delta_provenance.to_dict() if chronology_source == delta_source and delta_provenance else None
        ),
        "expected_anchor_index": (
            from_current if from_current is not None else from_order if from_order is not None else 0
        ),
    }, None


def _feature_width(root: Path, pre: Any, post: Any, key: str) -> int | None:
    config_path = root / "config.json"
    if config_path.is_file() and config_path.resolve().is_relative_to(root):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            mapping = config.get("output_features" if key == ACTION else "input_features", {})
            shape = mapping.get(key, {}).get("shape", ())
            if shape and type(shape[-1]) is int and shape[-1] > 0:
                return shape[-1]
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    for pipeline in (pre, post):
        for step in pipeline.steps:
            feature = getattr(step, "features", {}).get(key)
            shape = getattr(feature, "shape", ())
            if shape and type(shape[-1]) is int and shape[-1] > 0:
                return shape[-1]
    return None


def _probe(
    root: Path,
    pre_triple: tuple[Any, Path, dict[str, Any]],
    post_triple: tuple[Any, Path, dict[str, Any]],
    relative_index: int,
    metadata: dict[str, Any],
) -> DiagnosticFinding:
    from .trace import _step_provenance

    pre, config_path, saved_config = pre_triple
    post, _, _ = post_triple
    relative = pre.steps[relative_index]
    count = metadata["n_obs_steps"]
    action_dim = _feature_width(root, pre, post, ACTION) or len(relative.action_names or [])
    if not action_dim:
        return _finding(
            DiagnosticStatus.UNKNOWN,
            "Undeclared action dimension prevents temporal compatibility verification.",
            {"step_index": relative_index, "feature_key": ACTION, "declared_action_dim": None,
             "dimension_compatibility_verified": False, "processor_steps_executed": 0, **metadata},
            _step_provenance(config_path, saved_config, relative_index, ACTION),
        )
    state_dim = _feature_width(root, pre, post, OBS_STATE)
    if state_dim is None:
        return _finding(
            DiagnosticStatus.UNKNOWN,
            "The state feature width is undeclared; no synthetic width was inferred from the action.",
            {"step_index": relative_index, "feature_key": OBS_STATE, "action_dim": action_dim, **metadata},
            _step_provenance(config_path, saved_config, relative_index, OBS_STATE),
        )
    mask = relative._build_mask(action_dim)
    if state_dim < len(mask):
        missing = list(range(state_dim, len(mask)))
        return _finding(
            DiagnosticStatus.FAIL,
            "The declared state width cannot satisfy the native relative-action mask.",
            {"step_index": relative_index, "step_class": type(relative).__name__,
             "pipeline_role": "preprocessor", "feature_key": OBS_STATE,
             "declared_state_dim": state_dim, "declared_action_dim": action_dim,
             "relative_action_mask": mask, "required_state_width": len(mask),
             "missing_state_coordinates": missing, "tensor_slice": [],
             "synthetic_state_shape": [1, count, state_dim],
             "dimension_compatibility_verified": False, "processor_steps_executed": 0, **metadata},
            _step_provenance(config_path, saved_config, relative_index, OBS_STATE, missing),
        )
    probe_dim = next((index for index, enabled in enumerate(mask) if enabled), None)
    inactive_mask = probe_dim is None
    probe_dim = 0 if inactive_mask else probe_dim
    provenance = _step_provenance(config_path, saved_config, relative_index, OBS_STATE,
                                  [] if inactive_mask else [probe_dim])

    state = (
        torch.arange(count, dtype=torch.float32).mul(0.25).add(0.20)
        .view(1, count, 1).expand(1, count, state_dim).clone()
    )
    action = torch.full((1, 1, action_dim), 0.90, dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}
    precision_sources: list[dict[str, Any]] = []
    dtype_epsilons: dict[str, float] = {}

    def observe_precision(tensor: Any, source: str, role: str, index: int | None = None,
                          *, arithmetic_dtype: bool = True, tensor_key: str = ACTION) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return
        values = tensor.detach().double()
        finite = values[torch.isfinite(values)]
        magnitude = float(finite.abs().max()) if finite.numel() else 0.0
        precision_sources.append({"source": source, "pipeline_role": role, "step_index": index,
                                  "tensor_key": tensor_key, "dtype": str(tensor.dtype),
                                  "max_absolute_value": magnitude})
        if arithmetic_dtype:
            dtype_epsilons[str(tensor.dtype)] = torch.finfo(tensor.dtype).eps

    observe_precision(action, "probe_input", "preprocessor")
    for role, (pipeline, _, _) in (("preprocessor", pre_triple), ("postprocessor", post_triple)):
        for index, step in enumerate(pipeline.steps):
            if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
                mode = step.norm_map.get(FeatureType.ACTION)
                for stat in _MODE_STATS.get(mode, ()):
                    key = f"{ACTION}.{stat}"
                    observe_precision(step.state_dict().get(key), "active_action_constant", role, index,
                                      arithmetic_dtype=False, tensor_key=key)

    def precision_budget() -> dict[str, Any]:
        scale = max((item["max_absolute_value"] for item in precision_sources), default=1.0)
        epsilon = max(dtype_epsilons.values(), default=torch.finfo(torch.float32).eps)
        return {"max_intermediate_scale": scale, "machine_epsilon": epsilon,
                "safety_factor": _ROUNDTRIP_SAFETY_FACTOR,
                "roundoff_atol": epsilon * scale * _ROUNDTRIP_SAFETY_FACTOR,
                "dtype_epsilons": dict(dtype_epsilons), "sources": list(precision_sources)}

    def effective_atol() -> float:
        return max(_ROUNDTRIP_ATOL, precision_budget()["roundoff_atol"])

    def before(index: int, transition: EnvTransition) -> None:
        observe_precision(transition.get(TransitionKey.ACTION), "step_input", "preprocessor", index)
        if index != relative_index:
            return
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        candidate_state = observation.get(OBS_STATE)
        candidate_action = transition.get(TransitionKey.ACTION)
        if isinstance(candidate_state, torch.Tensor):
            captured["state"] = candidate_state.detach().clone()
        if isinstance(candidate_action, torch.Tensor):
            captured["action_before"] = candidate_action.detach().clone()

    def after(index: int, transition: EnvTransition) -> None:
        observe_precision(transition.get(TransitionKey.ACTION), "step_output", "preprocessor", index)
        if index != relative_index:
            return
        candidate_action = transition.get(TransitionKey.ACTION)
        if isinstance(candidate_action, torch.Tensor):
            captured["action_after"] = candidate_action.detach().clone()

    pre.before_step_hooks.append(before)
    pre.after_step_hooks.append(after)
    processed: dict[str, Any] | None = None
    pipeline_error: str | None = None
    try:
        with torch.inference_mode():
            processed = pre({OBS_STATE: state, ACTION: action})
    except Exception as exc:
        pipeline_error = str(exc)
    finally:
        pre.before_step_hooks.remove(before)
        pre.after_step_hooks.remove(after)

    entered_state = captured.get("state")
    entered_action = captured.get("action_before")
    observed_action = captured.get("action_after")
    if (
        entered_state is None or entered_action is None or observed_action is None
        or entered_state.ndim != 3 or entered_state.shape[1] != count
        or entered_action.ndim != 3 or observed_action.shape != entered_action.shape
    ):
        return _finding(
            DiagnosticStatus.FAIL if pipeline_error is not None else DiagnosticStatus.UNKNOWN,
            "The native forward relative-action probe failed." if pipeline_error is not None else
            "The native relative step did not receive the declared temporal tensor shape.",
            {
                "step_index": relative_index,
                "step_class": type(relative).__name__,
                "pipeline_role": "preprocessor",
                "feature_key": OBS_STATE,
                "captured_state_shape": list(entered_state.shape) if entered_state is not None else None,
                "captured_action_shape": list(entered_action.shape) if entered_action is not None else None,
                "declared_state_dim": state_dim,
                "declared_action_dim": action_dim,
                "synthetic_state_shape": list(state.shape),
                "pipeline_error": pipeline_error,
                **metadata,
            },
            provenance,
        )

    action_before = entered_action[0, 0, probe_dim]
    actual_relative = observed_action[0, 0, probe_dim]
    candidate_anchors = entered_state[0, :, probe_dim]
    observed_anchor = action_before - actual_relative
    expected_index = metadata["expected_anchor_index"]
    expected_relative = action_before - candidate_anchors[expected_index]

    roundtrip: bool | None = None
    roundtrip_slice: list[float] | None = None
    roundtrip_error: str | None = None
    roundtrip_max_error: float | None = None
    if processed is not None and any(
        isinstance(step, AbsoluteActionsProcessorStep) and step.enabled for step in post.steps
    ):
        def inverse_before(index: int, transition: EnvTransition) -> None:
            observe_precision(transition.get(TransitionKey.ACTION), "step_input", "postprocessor", index)

        def inverse_after(index: int, transition: EnvTransition) -> None:
            observe_precision(transition.get(TransitionKey.ACTION), "step_output", "postprocessor", index)

        post.before_step_hooks.append(inverse_before)
        post.after_step_hooks.append(inverse_after)
        try:
            with torch.inference_mode():
                restored = post({ACTION: processed[ACTION]})
            result = restored[ACTION]
            if isinstance(result, torch.Tensor) and result.shape == action.shape:
                difference = (result.double() - action.double()).abs()
                roundtrip_max_error = float(difference.max())
                roundtrip = bool(torch.isfinite(result).all()) and bool(torch.allclose(
                    result.double(), action.double(), atol=effective_atol(), rtol=_ROUNDTRIP_RTOL,
                ))
                roundtrip_slice = [float(result[0, 0, probe_dim])]
            else:
                roundtrip = False
                roundtrip_error = "The inverse result is not a tensor with the declared action shape."
        except Exception as exc:
            roundtrip = False
            roundtrip_error = str(exc)
        finally:
            post.before_step_hooks.remove(inverse_before)
            post.after_step_hooks.remove(inverse_after)

    # Cancellation when recovering the anchor has the same intermediate-scale
    # rounding exposure as the full round trip, not just the final ~0.9 output.
    matches = torch.isclose(candidate_anchors, observed_anchor, rtol=_ROUNDTRIP_RTOL,
                            atol=effective_atol()).nonzero().flatten()
    actual_index = int(matches[0]) if matches.numel() == 1 else None

    details = {
        "step_index": relative_index,
        "step_class": type(relative).__name__,
        "pipeline_role": "preprocessor",
        "feature_key": OBS_STATE,
        "action_key": ACTION,
        "declared_state_dim": state_dim,
        "declared_action_dim": action_dim,
        "synthetic_state_shape": list(state.shape),
        **metadata,
        "actual_anchor_index": actual_index,
        "probe_coordinate": probe_dim,
        "probe_state_slice": [float(value) for value in candidate_anchors],
        "probe_action_slice": [float(action_before)],
        "observed_relative_action_slice": [float(actual_relative)],
        "expected_relative_action_slice": [float(expected_relative)],
        "tensor_slice": [float(actual_relative)],
        "native_roundtrip_passed": roundtrip,
        "native_roundtrip_action_slice": roundtrip_slice,
        "native_roundtrip_max_abs_error": roundtrip_max_error,
        "roundtrip_error_tolerance": {"atol": effective_atol(), "rtol": _ROUNDTRIP_RTOL},
        "roundtrip_precision_budget": precision_budget(),
        "dimension_compatibility_verified": True,
        "relative_action_mask": mask,
    }
    if pipeline_error is not None:
        details["later_pipeline_error"] = pipeline_error
    if roundtrip_error is not None:
        details["roundtrip_error"] = roundtrip_error
    if pipeline_error is not None or roundtrip is False:
        return _finding(
            DiagnosticStatus.FAIL,
            "The native forward/inverse relative-action round-trip failed or exceeded numerical tolerance.",
            details,
            provenance,
        )
    if inactive_mask:
        return _finding(
            DiagnosticStatus.PASS,
            "All action coordinates are excluded from relative conversion; native probes confirm no active anchor offset.",
            {**details, "actual_anchor_index": None, "temporal_anchor_applied": False}, provenance,
        )
    if actual_index is None:
        return _finding(
            DiagnosticStatus.UNKNOWN,
            "The native relative output does not identify a unique observation anchor.",
            details,
            provenance,
        )
    if actual_index != expected_index:
        return _finding(
            DiagnosticStatus.FAIL,
            "Relative action conversion anchors to a different observation frame than declared current.",
            details,
            provenance,
        )
    if count > 1 and roundtrip is None:
        return _finding(
            DiagnosticStatus.UNKNOWN,
            "The observation anchor was identified, but no paired inverse round-trip was verified.",
            details,
            provenance,
        )
    return _finding(
        DiagnosticStatus.PASS,
        "Single-frame relative-action compatibility was verified; no multi-step observation history is present."
        if count == 1 else "Native relative action conversion uses the declared current observation frame and has an inverse round-trip within numerical tolerance.",
        details,
        provenance,
    )


def run_temporal_audit(
    checkpoint_dir: Path, metadata_path: Path | None = None,
) -> list[DiagnosticFinding]:
    """Compare native anchoring with config declarations and an opt-in sidecar."""
    from .trace import _load_processor_pair, _step_provenance, _TraceLoadError

    root = Path(checkpoint_dir).resolve()
    try:
        pre_triple, post_triple = _load_processor_pair(root)
    except _TraceLoadError as exc:
        return [_finding(exc.status, exc.summary, exc.details, exc.provenance)]

    pre, config_path, saved_config = pre_triple
    post, post_config_path, post_saved_config = post_triple
    relative_indices = [
        index for index, step in enumerate(pre.steps)
        if isinstance(step, RelativeActionsProcessorStep) and step.enabled
    ]
    if not relative_indices:
        absolute_indices = [
            index for index, step in enumerate(post.steps)
            if isinstance(step, AbsoluteActionsProcessorStep) and step.enabled
        ]
        if absolute_indices:
            index = absolute_indices[0]
            return [_finding(
                DiagnosticStatus.UNKNOWN,
                "An enabled absolute-action step has no enabled relative-action anchor.",
                {"step_index": index, "feature_key": OBS_STATE},
                _step_provenance(post_config_path, post_saved_config, index, OBS_STATE),
            )]
        return [_finding(
            DiagnosticStatus.PASS,
            "No enabled relative-action conversion requires a temporal anchor.",
            {"feature_key": OBS_STATE},
            _provenance(config_path),
        )]
    if len(relative_indices) != 1:
        return [_finding(
            DiagnosticStatus.UNKNOWN,
            "Multiple enabled relative-action steps make the temporal anchor ambiguous.",
            {"step_indices": relative_indices, "feature_key": OBS_STATE},
            _provenance(config_path),
        )]
    metadata, problem = _resolve_metadata(root, metadata_path)
    if problem is not None:
        return [problem]
    assert metadata is not None
    return [_probe(root, pre_triple, post_triple, relative_indices[0], metadata)]
