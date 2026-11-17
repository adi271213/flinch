"""Compare serialized processor authority without executing a policy or robot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    create_transition,
)
from safetensors import safe_open

from .models import DiagnosticFinding, DiagnosticStatus, PreflightReport, Provenance, Severity
from .preflight import (
    _CONFIG_NAMES,
    _legacy_declared_features,
    _local_file,
    _normalization_base,
    _physical_semantics,
    _provenance,
    _read_tensors,
    _role,
    _step_class,
    _validate_step_entry,
    _verified_legacy_values,
)

type _Artifact = tuple[dict[int, Any], Path, dict[str, Any]]
type _ArtifactPair = tuple[_Artifact, _Artifact]

_GAIN = "PRIVILEGE_ACTION_GAIN_ESCALATION"
_OFFSET = "PRIVILEGE_ACTION_OFFSET_ESCALATION"
_ENVELOPE = "PRIVILEGE_ACTION_ENVELOPE_SHIFT"
_HORIZON = "PRIVILEGE_HORIZON_ESCALATION"
_GUARD = "PRIVILEGE_GUARD_REMOVAL"
_COMPAT = "COMPATIBILITY_STATUS"
_OBSERVATION = "PRIVILEGE_OBSERVATION_STAT_DRIFT"
_INTERMEDIATE = "PRIVILEGE_INTERMEDIATE_NON_FINITE"
_NORMALIZERS = (NormalizerProcessorStep, UnnormalizerProcessorStep)
_MASKED_NORMALIZERS = {"molmoact2_masked_normalizer", "molmoact2_masked_unnormalizer"}
_REQUIRED = {
    NormalizationMode.MEAN_STD: ("mean", "std"),
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}
# These registry identifiers and persisted options are inspected in the reference
# policy processors. We diff their JSON only; their constructors are not invoked.
_FIXED_ACTION_CLIPS = {
    "vla_jepa_clip_actions",
    "lawam_clip_actions",
    "molmoact2_clamp_action",
    "longwam_action_clamp",
}
_BOTH_CLIPS = {"wall_oss_05_clamp_normalized", "g05_normalization_clamp", "molmoact2_clamp_normalized"}
# Inspected steps that only create language prompts/tokens, not numeric actions.
_LANGUAGE_STEPS = {
    "tokenizer_processor",
    "smolvla_new_line_processor",
    "pi0_new_line_processor",
    "pi05_prepare_state_tokenizer_processor_step",
    "lerobot.processor.tokenizer_processor.TokenizerProcessorStep",
    "lerobot.processor.newline_task_processor.NewLineTaskProcessorStep",
    "lerobot.policies.pi0.processor_pi0.Pi0NewLineProcessor",
    "lerobot.policies.pi05.processor_pi05.Pi05PrepareStateTokenizerProcessorStep",
}


def _language_only(entry: dict[str, Any]) -> bool:
    return entry.get("registry_name", entry.get("class")) in _LANGUAGE_STEPS


def _step_provenance(
    path: Path,
    config: dict[str, Any],
    index: int,
    key: str,
    indices: list[int] | None = None,
) -> Provenance:
    entry = config["steps"][index]
    if entry.get("_legacy"):
        original = entry["_legacy_keys"].get(key, key)
        if entry.get("_legacy_raw_model") and key in entry["_legacy_state"]:
            tensor = entry["_legacy_state"][key]
            raw = tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
            return Provenance(
                str(path),
                "legacy_" + entry.get("_legacy_group", "unnormalize_outputs"),
                original,
                hashlib.sha256(raw).hexdigest()[:12],
                indices,
            )
        return _provenance(
            path, "legacy_" + entry.get("_legacy_group", "unnormalize_outputs"), original, indices
        )
    source = (
        _local_file(path.parent, entry["state_file"])
        if "state_file" in entry and not entry.get("config", {}).get("stats")
        else path
    )
    cls = _normalization_base(entry) or _step_class(entry)
    return _provenance(source, cls.__name__ if cls else entry.get("registry_name", index), key, indices)


def _artifact_provenance(artifact: _Artifact) -> Provenance:
    _, path, config = artifact
    for index, entry in enumerate(config["steps"]):
        if entry.get("_legacy"):
            return _step_provenance(path, config, index, next(iter(entry["_legacy_keys"])))
    return _provenance(path)


def _add(
    findings: list[DiagnosticFinding],
    check: str,
    status: DiagnosticStatus,
    summary: str,
    details: dict[str, Any] | None = None,
    provenance: Provenance | None = None,
) -> None:
    details = dict(details or {})
    if status == DiagnosticStatus.UNKNOWN:
        # Undeclared physical semantics remain explicit, separate unknowns. An
        # incomplete artifact comparison must fail a requested baseline gate.
        details.setdefault("required_coverage", details.get("scope") != "physical_semantics")
    findings.append(
        DiagnosticFinding(
            check,
            status,
            Severity.CRITICAL
            if status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
            else Severity.INFO,
            summary,
            details,
            provenance,
        )
    )


def _configs(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    if not root.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {root}")
    candidates: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        try:
            path = _local_file(root, path.name)
            config = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            if path.name in _CONFIG_NAMES:
                raise
            continue
        if isinstance(config, dict) and isinstance(config.get("steps"), list):
            role = _role(path, config)
            if role:
                candidates[role].append((path, config))
    selected = {}
    for role, entries in candidates.items():
        if any(config != entries[0][1] for _, config in entries[1:]):
            raise ValueError(f"Multiple {role}processor configurations disagree in {root}")
        selected[role] = entries[0]
    return selected


def _native_normalizer(root: Path, entry: dict[str, Any]) -> Any | None:
    cls = _normalization_base(entry)
    if cls is None or cls not in _NORMALIZERS:
        return None
    step = cls(**{**entry.get("config", {}), "device": "cpu"})
    if entry.get("_legacy"):
        step.load_state_dict(entry["_legacy_state"])
    elif "state_file" in entry:
        step.load_state_dict(_read_tensors(_local_file(root, entry["state_file"])))
    _bind_implicit_action_contract(step, entry)
    return step


def _bind_implicit_action_contract(
    step: Any, entry: dict[str, Any], policy: dict[str, Any] | None = None
) -> None:
    """Size diagnostics for native ACTION math even without serialized features.

    Native ``_normalize_action`` dispatches by norm_map independently of
    ``features``. Adding its diagnostic feature to this isolated native instance
    does not change action arithmetic or the checkpoint's serialized contract.
    Saved stat widths describe tensor coordinates, never physical actuators.
    """
    serialized_action = entry.get("config", {}).get("features", {}).get("action")
    if serialized_action is not None and serialized_action.get("type") == FeatureType.ACTION.value:
        return
    from .preflight import _normalizer_features

    effective = _normalizer_features(entry, step.state_dict()).get("action")
    if effective is None:
        return
    declared = (policy or {}).get("output_features", {}).get("action")
    if declared is not None and declared.get("type") == FeatureType.ACTION.value:
        shape, source = tuple(declared["shape"]), "policy_output_features"
    elif entry.get("_action_shape_source") == "policy_output_features":
        # Guard comparisons reload isolated steps without the policy argument.
        # Retain the explicit width already verified by _static_pair.
        shape, source = tuple(entry["_action_shape"]), "policy_output_features"
    else:
        shape, source = tuple(effective["shape"]), "saved_stat_broadcast"
    step.features = {**step.features, "action": PolicyFeature(FeatureType.ACTION, shape)}
    entry["_action_shape_source"] = source
    entry["_action_shape"] = list(shape)


def _static_pair(root: Path, policy: dict[str, Any]) -> _ArtifactPair:
    """Read artifact contracts without constructing any full processor pipeline."""
    configs = _configs(root)
    if not configs:
        all_declared = _legacy_declared_features(policy)
        existing_inputs = {key for group, key, _, _ in all_declared if group == "normalize_inputs"}
        for input_key, declaration in (policy.get("input_features") or {}).items():
            if declaration.get("type") != FeatureType.ACTION.value and input_key not in existing_inputs:
                feature_type = FeatureType(declaration["type"])
                all_declared.append(
                    (
                        "normalize_inputs",
                        input_key,
                        PolicyFeature(feature_type, tuple(declaration["shape"])),
                        NormalizationMode(
                            (policy.get("normalization_mapping") or {}).get(feature_type.value, "IDENTITY")
                        ),
                    )
                )
        declared = [item for item in all_declared if item[0] == "unnormalize_outputs" and item[1] == "action"]
        if len(declared) != 1 or declared[0][3] not in _REQUIRED:
            raise ValueError("No explicit active legacy action feature/mode contract is available")
        _, key, feature, mode = declared[0]
        aliases = {f"action.{stat}": f"unnormalize_outputs.buffer_action.{stat}" for stat in _REQUIRED[mode]}
        input_declared = [
            item for item in all_declared if item[0] == "normalize_inputs" and item[3] in _REQUIRED
        ]
        all_aliases = dict(aliases)
        for group, input_key, _, input_mode in input_declared:
            all_aliases.update(
                {
                    f"{input_key}.{stat}": f"{group}.buffer_{input_key.replace('.', '_')}.{stat}"
                    for stat in _REQUIRED[input_mode]
                }
            )
        raw_model = False
        if (root / "legacy_normalization.json").is_file():
            source, state, _, _ = _verified_legacy_values(root)
        elif (root / "model.safetensors").is_file():
            source = _local_file(root, "model.safetensors")
            # Memory-map only the tiny declared normalization tensors. Neural
            # weights are not materialized just to compare action envelopes.
            with safe_open(source, framework="pt", device="cpu") as model:
                state = {
                    original: model.get_tensor(original)
                    for original in all_aliases.values()
                    if original in model.keys()
                }
            raw_model = True
        else:
            raise ValueError("Verified legacy action buffer values are unavailable locally")
        if any(original not in state for original in aliases.values()):
            raise ValueError("Target-dataset legacy action buffers are unbound or incomplete")
        canonical = {alias: state[original] for alias, original in aliases.items()}
        native = UnnormalizerProcessorStep(features={key: feature}, norm_map={feature.type: mode})
        native.load_state_dict(canonical)
        entry = {
            "registry_name": "unnormalizer_processor",
            "config": native.get_config(),
            "_legacy": True,
            "_legacy_keys": aliases,
            "_legacy_state": canonical,
            "_legacy_raw_model": raw_model,
            "_legacy_group": "unnormalize_outputs",
        }
        # These in-memory declarations adapt the existing native step only;
        # nothing is saved and no replacement processor implementation exists.
        input_steps, input_normalizers, input_errors = [], {}, {}
        for index, (group, input_key, input_feature, input_mode) in enumerate(input_declared):
            input_aliases = {
                f"{input_key}.{stat}": all_aliases[f"{input_key}.{stat}"] for stat in _REQUIRED[input_mode]
            }
            input_state = {
                alias: state[original] for alias, original in input_aliases.items() if original in state
            }
            input_native = NormalizerProcessorStep(
                features={input_key: input_feature}, norm_map={input_feature.type: input_mode}, eps=1e-8
            )
            input_native.load_state_dict(input_state)
            input_steps.append(
                {
                    "registry_name": "normalizer_processor",
                    "config": input_native.get_config(),
                    "_legacy": True,
                    "_legacy_group": group,
                    "_legacy_keys": input_aliases,
                    "_legacy_state": input_state,
                    "_legacy_raw_model": raw_model,
                }
            )
            input_normalizers[index] = input_native
            if len(input_state) != len(input_aliases):
                input_errors[index] = f"Missing legacy observation buffers for {input_key}"
        return (
            (input_normalizers, source, {"steps": input_steps, "_normalizer_errors": input_errors}),
            ({0: native}, source, {"steps": [entry]}),
        )
    artifacts: list[_Artifact] = []
    for role in ("pre", "post"):
        path, config = configs.get(role, (root / f"policy_{role}processor.json", {"steps": []}))
        normalizers = {}
        errors = {}
        for index, entry in enumerate(config["steps"]):
            _validate_step_entry(entry)
            try:
                step = _native_normalizer(root, entry)
            except (ValueError, OSError, TypeError, KeyError, RuntimeError, AttributeError) as exc:
                errors[index] = str(exc)
                continue
            if step is not None:
                _bind_implicit_action_contract(step, entry, policy)
                normalizers[index] = step
        artifacts.append((normalizers, path, {**config, "_normalizer_errors": errors}))
    return artifacts[0], artifacts[1]


def _kind(entry: dict[str, Any]) -> str:
    cls = _step_class(entry)
    if cls is not None:
        return f"{cls.__module__}.{cls.__name__}"
    return cast(str, entry.get("registry_name", entry.get("class", "")))


def _clip_keys(entry: dict[str, Any]) -> list[str]:
    name, config = _kind(entry), entry.get("config", {})
    if name in _FIXED_ACTION_CLIPS:
        return ["action"]
    if name in _BOTH_CLIPS:
        return ["observation.state", "action"]
    if name == "dm05_clip_normalized_processor":
        return [
            key
            for key, flag in (("observation.state", "clip_state"), ("action", "clip_action"))
            if config.get(flag, False)
        ]
    if any(word in name.lower() for word in ("clip", "clamp")) and _declared_clip_bounds(config):
        return [config.get("feature_key", "action")]
    return []


def _declared_clip_bounds(config: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Read explicit serialized clamp bounds; never infer hardware limits."""
    bounds = {}
    for low, high in (("minimum", "maximum"), ("min", "max"), ("clip_min", "clip_max")):
        if low in config and high in config:
            bounds[f"{low}/{high}"] = (
                torch.as_tensor(config[low], dtype=torch.float64),
                torch.as_tensor(config[high], dtype=torch.float64),
            )
    if "clip" in config and not isinstance(config["clip"], bool):
        clip = config["clip"]
        if isinstance(clip, (float, int)):
            bounds["clip"] = (
                torch.tensor(-abs(clip), dtype=torch.float64),
                torch.tensor(abs(clip), dtype=torch.float64),
            )
        elif isinstance(clip, (list, tuple)) and len(clip) == 2:
            bounds["clip"] = (
                torch.as_tensor(clip[0], dtype=torch.float64),
                torch.as_tensor(clip[1], dtype=torch.float64),
            )
        elif isinstance(clip, dict):
            bounds.update({f"clip.{key}": value for key, value in _declared_clip_bounds(clip).items()})
    return bounds


def _guard_order(
    base_config: dict[str, Any], candidate_config: dict[str, Any], baseline: Path, candidate: Path
) -> tuple[list[str], list[str]]:
    before = [_kind(entry) for entry in base_config["steps"] if not _language_only(entry)]
    after = [_kind(entry) for entry in candidate_config["steps"] if not _language_only(entry)]
    # Compare the sequence of retained occurrences as well as duplicate steps.
    # Insertions/removals have independent guard/compatibility findings.
    from collections import Counter

    shared = Counter(before) & Counter(after)

    def retained(sequence: list[str]) -> list[str]:
        remaining = shared.copy()
        result = []
        for kind in sequence:
            if remaining[kind]:
                result.append(kind)
                remaining[kind] -= 1
        return result

    old_order, new_order = retained(before), retained(after)
    if old_order != new_order:
        return old_order, new_order

    def payload_order(root: Path, config: dict[str, Any]) -> list[str]:
        result = []
        for entry in config["steps"]:
            if _language_only(entry):
                continue
            if entry.get("_legacy"):
                return []
            declared = {key: value for key, value in entry.items() if key != "state_file"}
            digest = hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()[:12]
            if "state_file" in entry:
                digest += (
                    ":" + hashlib.sha256(_local_file(root, entry["state_file"]).read_bytes()).hexdigest()[:12]
                )
            result.append(f"{_kind(entry)}:{digest}")
        return result

    old_payloads, new_payloads = (
        payload_order(baseline, base_config),
        payload_order(candidate, candidate_config),
    )
    if Counter(old_payloads) == Counter(new_payloads) and old_payloads != new_payloads:
        return old_payloads, new_payloads
    return old_order, new_order


def _guard_delta(
    baseline: Path,
    candidate: Path,
    base_configs: dict[str, tuple[Path, dict[str, Any]]],
    candidate_configs: dict[str, tuple[Path, dict[str, Any]]],
    findings: list[DiagnosticFinding],
) -> None:
    for role, (base_path, base_config) in base_configs.items():
        candidate_path, candidate_config = candidate_configs.get(
            role, (candidate / base_path.name, {"steps": []})
        )
        old_order, new_order = _guard_order(base_config, candidate_config, baseline, candidate)
        if old_order != new_order:
            first = next(
                index
                for index, (old_kind, new_kind) in enumerate(zip(old_order, new_order))
                if old_kind != new_kind
            )
            old_indices = [i for i, item in enumerate(base_config["steps"]) if not _language_only(item)]
            new_indices = [i for i, item in enumerate(candidate_config["steps"]) if not _language_only(item)]
            reordered_base_index, reordered_candidate_index = old_indices[first], new_indices[first]
            _add(
                findings,
                _GUARD,
                DiagnosticStatus.FAIL,
                "The candidate reordered serialized numeric processor steps; guard placement is no longer equivalent.",
                {
                    "pipeline_role": "preprocessor" if role == "pre" else "postprocessor",
                    "step_index": reordered_candidate_index,
                    "baseline_step_index": reordered_base_index,
                    "reason": "processor_step_order_changed",
                    "baseline_order": old_order,
                    "candidate_order": new_order,
                    "baseline_provenance": _provenance(base_path, reordered_base_index).to_dict(),
                },
                _provenance(candidate_path, reordered_candidate_index),
            )
        by_kind: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, entry in enumerate(candidate_config["steps"]):
            by_kind[_kind(entry)].append((index, entry))
        occurrence: dict[str, int] = defaultdict(int)
        for base_index, entry in enumerate(base_config["steps"]):
            kind = _kind(entry)
            ordinal = occurrence[kind]
            occurrence[kind] += 1
            matches = by_kind[kind]
            candidate_index, candidate_entry = matches[ordinal] if ordinal < len(matches) else (None, None)
            base_step = _native_normalizer(baseline, entry)
            candidate_step = (
                _native_normalizer(candidate, candidate_entry) if candidate_entry is not None else None
            )
            if base_step is not None:
                base_state = base_step.state_dict()
                for key, feature in base_step.features.items():
                    mode = base_step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
                    required = _REQUIRED.get(mode, ())
                    if not required or (
                        feature.type != FeatureType.ACTION
                        and base_step.normalize_observation_keys is not None
                        and key not in base_step.normalize_observation_keys
                    ):
                        continue
                    if any(f"{key}.{stat}" not in base_state for stat in required):
                        _add(
                            findings,
                            _GUARD,
                            DiagnosticStatus.UNKNOWN,
                            "The baseline normalization requirement is already missing buffers.",
                            {"feature_key": key, "baseline_step_index": base_index, "pipeline_role": role},
                            _step_provenance(base_path, base_config, base_index, key),
                        )
                        continue
                    reason = None
                    if candidate_step is None:
                        reason = "normalization_step_removed"
                    elif key not in candidate_step.features:
                        reason = "feature_contract_removed"
                    elif (
                        candidate_step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
                        == NormalizationMode.IDENTITY
                    ):
                        reason = "normalization_changed_to_identity"
                    elif (
                        feature.type != FeatureType.ACTION
                        and candidate_step.normalize_observation_keys is not None
                        and key not in candidate_step.normalize_observation_keys
                    ):
                        reason = "feature_excluded_from_normalization"
                    else:
                        candidate_mode = candidate_step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
                        needed = _REQUIRED.get(candidate_mode, ())
                        if any(f"{key}.{stat}" not in candidate_step.state_dict() for stat in needed):
                            reason = "normalization_stat_key_bypass"
                        if kind in _MASKED_NORMALIZERS and feature.shape:
                            assert candidate_index is not None
                            width = feature.shape[-1]

                            def effective_mask(state: dict[str, torch.Tensor]) -> torch.Tensor:
                                value = state.get(f"{key}.mask")
                                # The inspected native mixin ignores absent or
                                # unbroadcastable masks and normalizes all coords.
                                return (
                                    value.bool()
                                    if value is not None and value.ndim == 1 and value.numel() == width
                                    else torch.ones(width, dtype=torch.bool)
                                )

                            old_mask = effective_mask(base_state)
                            new_mask = effective_mask(candidate_step.state_dict())
                            removed = old_mask & ~new_mask
                            if bool(removed.any()):
                                indices = removed.nonzero().flatten().tolist()
                                mask_key = f"{key}.mask"
                                _add(
                                    findings,
                                    _GUARD,
                                    DiagnosticStatus.FAIL,
                                    f"The candidate bypasses baseline normalization on masked {key} coordinates.",
                                    {
                                        "feature_key": key,
                                        "step_index": candidate_index,
                                        "baseline_step_index": base_index,
                                        "reason": "normalization_mask_bypass",
                                        "pipeline_role": "preprocessor" if role == "pre" else "postprocessor",
                                        "baseline_mask": old_mask[removed].tolist(),
                                        "candidate_mask": new_mask[removed].tolist(),
                                        "saved_values": new_mask[removed].tolist(),
                                        "tensor_slice": new_mask[removed].tolist(),
                                        "baseline_provenance": _step_provenance(
                                            base_path, base_config, base_index, mask_key, indices
                                        ).to_dict(),
                                    },
                                    _step_provenance(
                                        candidate_path, candidate_config, candidate_index, mask_key, indices
                                    ),
                                )
                    if reason:
                        unverified_replacement = candidate_step is None and any(
                            _step_class(item) is None and not _language_only(item)
                            for item in candidate_config["steps"]
                        )
                        provenance = (
                            _step_provenance(candidate_path, candidate_config, candidate_index, key)
                            if candidate_index is not None
                            else _provenance(candidate_path, tensor_key=key)
                        )
                        _add(
                            findings,
                            _GUARD,
                            DiagnosticStatus.UNKNOWN if unverified_replacement else DiagnosticStatus.FAIL,
                            f"A replacement normalization requirement for {key} cannot be verified."
                            if unverified_replacement
                            else f"The candidate removed or bypassed a baseline normalization requirement for {key}.",
                            {
                                "feature_key": key,
                                "step_index": candidate_index,
                                "baseline_step_index": base_index,
                                "pipeline_role": "preprocessor" if role == "pre" else "postprocessor",
                                "reason": reason,
                                "expected_buffer_keys": [f"{key}.{stat}" for stat in required],
                                "actual_buffer_keys": sorted(candidate_step.state_dict())
                                if candidate_step
                                else [],
                                "tensor_slice": [],
                                "replacement_equivalence_verified": False if unverified_replacement else None,
                                "baseline_provenance": _step_provenance(
                                    base_path, base_config, base_index, key
                                ).to_dict(),
                            },
                            provenance,
                        )
            for key in _clip_keys(entry):
                reason = None
                before_bounds, after_bounds = {}, {}
                if candidate_entry is None or key not in _clip_keys(candidate_entry):
                    reason = "clipping_step_removed_or_disabled"
                else:
                    old, new = dict(entry.get("config", {})), dict(candidate_entry.get("config", {}))
                    if kind == "g05_normalization_clamp":
                        old = {"minimum": -5.0, "maximum": 5.0, **old}
                        new = {"minimum": -5.0, "maximum": 5.0, **new}
                    before_bounds, after_bounds = _declared_clip_bounds(old), _declared_clip_bounds(new)
                    for field, (low, high) in before_bounds.items():
                        if field not in after_bounds:
                            reason = "clipping_bounds_removed_or_changed"
                            break
                        new_low, new_high = after_bounds[field]
                        if (
                            not torch.isfinite(new_low).all()
                            or not torch.isfinite(new_high).all()
                            or (new_low > new_high).any()
                        ):
                            reason = "invalid_clipping_bounds"
                            break
                        if (new_low < low).any() or (new_high > high).any():
                            reason = "clipping_bounds_widened"
                            break
                if reason:
                    _add(
                        findings,
                        _GUARD,
                        DiagnosticStatus.FAIL,
                        f"A baseline serialized {role}processor clipping step was removed or weakened.",
                        {
                            "feature_key": key,
                            "step_index": candidate_index,
                            "baseline_step_index": base_index,
                            "pipeline_role": "preprocessor" if role == "pre" else "postprocessor",
                            "reason": reason,
                            "baseline_config": entry.get("config", {}),
                            "scope": "serialized_policy_pipeline",
                            "baseline_declared_bounds": {
                                field: [lo.tolist(), hi.tolist()] for field, (lo, hi) in before_bounds.items()
                            },
                            "candidate_declared_bounds": {
                                field: [lo.tolist(), hi.tolist()] for field, (lo, hi) in after_bounds.items()
                            },
                            "candidate_config": candidate_entry.get("config", {})
                            if candidate_entry
                            else None,
                            "baseline_provenance": _provenance(base_path, kind, key).to_dict(),
                        },
                        _provenance(candidate_path, kind, key),
                    )


def _policy(root: Path) -> dict[str, Any]:
    path = root / "config.json"
    if not path.is_file():
        return {}
    config = json.loads(_local_file(root, path.name).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Policy config must be a JSON object: {path}")
    return config


def _horizon(
    baseline: Path,
    candidate: Path,
    old: dict[str, Any],
    new: dict[str, Any],
    findings: list[DiagnosticFinding],
) -> None:
    values = {
        "baseline_n_action_steps": old.get("n_action_steps"),
        "candidate_n_action_steps": new.get("n_action_steps"),
        "baseline_chunk_size": old.get("chunk_size"),
        "candidate_chunk_size": new.get("chunk_size"),
        "scope": "declared_execution_steps",
        "controller_timing_inferred": False,
        "baseline_provenance": _provenance(baseline / "config.json", tensor_key="n_action_steps").to_dict(),
    }
    provenance = _provenance(candidate / "config.json", tensor_key="n_action_steps")
    for label, config, root in (("baseline", old, baseline), ("candidate", new, candidate)):
        steps, chunk = config.get("n_action_steps"), config.get("chunk_size")
        if steps is not None and (type(steps) is not int or steps < 1):
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"{label} n_action_steps is not a positive integer.",
                values,
                _provenance(root / "config.json", tensor_key="n_action_steps"),
            )
        if chunk is not None and (type(chunk) is not int or chunk < 1):
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"{label} chunk_size is not a positive integer.",
                values,
                _provenance(root / "config.json", tensor_key="chunk_size"),
            )
        if type(steps) is int and type(chunk) is int and steps > chunk:
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"{label} executes more steps than its declared prediction chunk.",
                values,
                _provenance(root / "config.json", tensor_key="n_action_steps"),
            )
    before, after = values["baseline_n_action_steps"], values["candidate_n_action_steps"]
    if type(before) is not int or before < 1 or type(after) is not int or after < 1:
        _add(
            findings,
            _HORIZON,
            DiagnosticStatus.UNKNOWN,
            "Both checkpoints must declare n_action_steps; chunk_size alone does not specify execution commitment.",
            values,
            provenance,
        )
        return
    # The inspected ACT ensemble predicts on every observation instead of draining a chunk.
    effective_before = (
        1 if old.get("type") == "act" and old.get("temporal_ensemble_coeff") is not None else before
    )
    effective_after = (
        1 if new.get("type") == "act" and new.get("temporal_ensemble_coeff") is not None else after
    )
    values.update(
        {"baseline_effective_steps": effective_before, "candidate_effective_steps": effective_after}
    )
    for label, config, steps, root in (
        ("baseline", old, before, baseline),
        ("candidate", new, after, candidate),
    ):
        if config.get("type") == "act" and config.get("temporal_ensemble_coeff") is not None and steps != 1:
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"{label} ACT temporal ensembling requires n_action_steps=1.",
                values,
                _provenance(root / "config.json", tensor_key="n_action_steps"),
            )
    expansion = effective_after > effective_before
    _add(
        findings,
        _HORIZON,
        DiagnosticStatus.FAIL if expansion else DiagnosticStatus.PASS,
        "The candidate increases the declared execution commitment between feedback updates."
        if expansion
        else "The candidate does not increase the declared execution commitment.",
        values,
        provenance,
    )


def _signature(artifact: _Artifact, action_dim: int | None = None) -> list[dict[str, Any]]:
    normalizers, _, config = artifact
    result = []
    for index, entry in enumerate(config["steps"]):
        step = normalizers.get(index)
        cls = _step_class(entry)
        if step is not None:
            result.append(
                {
                    "step_class": type(step).__name__,
                    "features": {
                        key: {"type": ft.type.value, "shape": list(ft.shape)}
                        for key, ft in step.features.items()
                    },
                    "norm_map": {key.value: value.value for key, value in step.norm_map.items()},
                    "eps": step.eps,
                    "normalization_masks": {
                        key: value.tolist()
                        for key, value in step.state_dict().items()
                        if key.endswith(".mask")
                    }
                    if _kind(entry) in _MASKED_NORMALIZERS
                    else None,
                    "normalize_observation_keys": sorted(step.normalize_observation_keys)
                    if step.normalize_observation_keys is not None
                    else None,
                }
            )
        elif cls is RelativeActionsProcessorStep and action_dim is not None:
            native = cls(**entry.get("config", {}))
            result.append(
                {
                    "step_class": cls.__name__,
                    "enabled": native.enabled,
                    "effective_mask": native._build_mask(action_dim) if native.enabled else None,
                }
            )
        elif cls is not None and cls in (
            AbsoluteActionsProcessorStep,
            RenameObservationsProcessorStep,
            AddBatchDimensionProcessorStep,
        ):
            result.append({"step_class": cls.__name__, "config": entry.get("config", {})})
        elif cls is RelativeActionsProcessorStep:
            result.append({"step_class": cls.__name__, "config": entry.get("config", {})})
        elif cls is DeviceProcessorStep:
            result.append(
                {"step_class": cls.__name__, "float_dtype": entry.get("config", {}).get("float_dtype")}
            )
    return result


def _compatible(
    baseline: Path,
    candidate: Path,
    old: dict[str, Any],
    new: dict[str, Any],
    base_pair: _ArtifactPair,
    candidate_pair: _ArtifactPair,
    findings: list[DiagnosticFinding],
) -> bool:
    compatible = True
    unknown_steps: list[dict[str, Any]] = []
    for root, policy, pair in ((baseline, old, base_pair), (candidate, new, candidate_pair)):
        for normalizers, path, config in pair:
            unknown_steps.extend(
                {"processor_file": str(path), "step_index": index, "error": message}
                for index, message in config.get("_normalizer_errors", {}).items()
            )
            for index, entry in enumerate(config["steps"]):
                step = normalizers.get(index)
                if step is None:
                    if _step_class(entry) is None and not _language_only(entry):
                        unknown_steps.append(
                            {"processor_file": str(path), "step_index": index, "step": _kind(entry)}
                        )
                    continue
                if (
                    not entry.get("_legacy")
                    and entry.get("config", {}).get("stats")
                    and "state_file" in entry
                ):
                    saved = _read_tensors(_local_file(root, entry["state_file"]))
                    active = step.state_dict()
                    for key in sorted(saved.keys() | active.keys()):
                        a, b = saved.get(key), active.get(key)
                        if (
                            a is not None
                            and b is not None
                            and a.shape == b.shape
                            and torch.allclose(a.double(), b.double(), rtol=0, atol=0, equal_nan=True)
                        ):
                            continue
                        _add(
                            findings,
                            "INLINE_STAT_OVERRIDE_ADVISORY",
                            DiagnosticStatus.PASS,
                            f"Inline JSON statistics override the inactive safetensors value for {key} under native precedence.",
                            {
                                "step_index": index,
                                "feature_key": key.rsplit(".", 1)[0],
                                "saved_values": a.reshape(-1).tolist() if a is not None else None,
                                "active_json_values": b.reshape(-1).tolist() if b is not None else None,
                                "tensor_slice": b.reshape(-1).tolist() if b is not None else [],
                                "config_path": str(path),
                                "required_coverage": False,
                                "effective_statistics_source": "inline_config_stats",
                            },
                            _step_provenance(path, config, index, key),
                        )
                        findings[-1].severity = Severity.WARNING
                mapping = policy.get("normalization_mapping") or {}
                declared_action = entry.get("config", {}).get("features", {}).get("action")
                if (
                    declared_action is not None
                    and declared_action.get("type") != FeatureType.ACTION.value
                    and step.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)
                    is not NormalizationMode.IDENTITY
                ):
                    compatible = False
                    _add(
                        findings,
                        _COMPAT,
                        DiagnosticStatus.CONFLICT,
                        "The action feature type disagrees with native ACTION normalization dispatch.",
                        {
                            "step_index": index,
                            "declared_feature_type": declared_action.get("type"),
                            "effective_feature_type": FeatureType.ACTION.value,
                        },
                        _provenance(path, type(step).__name__, "action"),
                    )
                for feature_type, mode in step.norm_map.items():
                    if feature_type.value in mapping and mapping[feature_type.value] != mode.value:
                        compatible = False
                        _add(
                            findings,
                            _COMPAT,
                            DiagnosticStatus.CONFLICT,
                            "Policy and processor normalization modes disagree.",
                            {
                                "step_index": index,
                                "feature_type": feature_type.value,
                                "policy_mode": mapping[feature_type.value],
                                "processor_mode": mode.value,
                                "policy_config": str(root / "config.json"),
                            },
                            _provenance(path, type(step).__name__),
                        )
    for index, role in enumerate(("preprocessor", "postprocessor")):
        if base_pair[index][2].get("_normalizer_errors") or candidate_pair[index][2].get(
            "_normalizer_errors"
        ):
            continue  # An unreadable artifact is incomplete evidence, not proof of a changed contract.
        before_width = (old.get("output_features") or {}).get("action", {}).get("shape", [])
        after_width = (new.get("output_features") or {}).get("action", {}).get("shape", [])
        before, after = (
            _signature(base_pair[index], before_width[-1] if before_width else None),
            _signature(candidate_pair[index], after_width[-1] if after_width else None),
        )
        if before != after:
            compatible = False
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"{role} feature keys, dimensions, modes, or action parameterization changed.",
                {
                    "pipeline_role": role,
                    "baseline_contract": before,
                    "candidate_contract": after,
                    "baseline_provenance": _artifact_provenance(base_pair[index]).to_dict(),
                },
                _artifact_provenance(candidate_pair[index]),
            )
    # Legacy policies persist the same contract in per-feature shape/mode maps.
    # Comparing only modern feature objects would silently omit legacy inputs,
    # since only the independent action unnormalizer is constructed above.
    for field in (
        "input_features",
        "output_features",
        "input_shapes",
        "output_shapes",
        "input_normalization_modes",
        "output_normalization_modes",
        "normalization_mapping",
    ):
        if old.get(field) != new.get(field):
            compatible = False
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"Policy {field} contracts disagree.",
                {"field": field, "baseline_features": old.get(field), "candidate_features": new.get(field)},
                _provenance(candidate / "config.json", tensor_key=field),
            )
            removed_keys = (
                (old.get(field) or {}).keys() - (new.get(field) or {}).keys()
                if field in {"input_features", "output_features", "input_shapes", "output_shapes"}
                else set()
            )
            for key in removed_keys:
                _add(
                    findings,
                    _GUARD,
                    DiagnosticStatus.FAIL,
                    f"The candidate removed the policy feature contract for {key}.",
                    {
                        "feature_key": key,
                        "field": field,
                        "baseline_provenance": _provenance(
                            baseline / "config.json", tensor_key=key
                        ).to_dict(),
                    },
                    _provenance(candidate / "config.json", tensor_key=key),
                )
    semantic_reports = [PreflightReport(str(root)) for root in (baseline, candidate)]
    for root, report in zip((baseline, candidate), semantic_reports):
        _physical_semantics(root, report)
    for field in ("joint_angle_units", "action_coordinate_frame"):
        sources = [
            next(f for f in report.findings if f.details["field"] == field) for report in semantic_reports
        ]
        if any(f.status == DiagnosticStatus.CONFLICT for f in sources):
            compatible = False
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"Physical metadata for {field} is internally inconsistent.",
                {"field": field, "declarations": [f.to_dict() for f in sources]},
            )
        elif (
            all(f.status == DiagnosticStatus.PASS for f in sources)
            and sources[0].details["value"] != sources[1].details["value"]
        ):
            compatible = False
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                f"Explicit {field} declarations disagree.",
                {
                    "field": field,
                    "baseline_value": sources[0].details["value"],
                    "candidate_value": sources[1].details["value"],
                },
                sources[1].provenance,
            )
        elif not all(f.status == DiagnosticStatus.PASS for f in sources):
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.UNKNOWN,
                f"Physical equivalence of {field} is undeclared; only numeric buffer ratios can be compared.",
                {"field": field, "scope": "physical_semantics", "strict_exempt": True},
            )
    if unknown_steps:
        _add(
            findings,
            _COMPAT,
            DiagnosticStatus.UNKNOWN,
            "Some serialized numeric processor steps are uninspected; their complete behavior is not verified.",
            {"uninspected_steps": unknown_steps, "scope": "uninspected_processor_behavior"},
        )
    if compatible and not unknown_steps:
        _add(
            findings,
            _COMPAT,
            DiagnosticStatus.PASS,
            "Serialized processor feature and normalization contracts are compatible; no language constructor was invoked.",
            {
                "scope": "processor_contract",
                "physical_units_inferred": False,
                "processor_execution_required": False,
                "language_constructors_invoked": False,
            },
        )
    return compatible


def _action_scale(artifact: _Artifact) -> tuple[torch.Tensor, list[tuple[int, str]], list[str]]:
    """The scale is the native inverse affine slope, including zero-span eps."""
    mapping = _action_affine(artifact)
    sources = [
        (index, "action.std" if mode == "MEAN_STD" else "action." + _REQUIRED[NormalizationMode(mode)][1])
        for (index, _), mode in zip(mapping["sources"], mapping["modes"])
    ]
    return mapping["slope"], sources, mapping["modes"]


def _scale_delta(
    base_pair: _ArtifactPair,
    candidate_pair: _ArtifactPair,
    threshold: float,
    findings: list[DiagnosticFinding],
) -> None:
    try:
        old, old_sources, old_modes = _action_scale(base_pair[1])
        new, new_sources, new_modes = _action_scale(candidate_pair[1])
        if old.shape != new.shape or old_modes != new_modes:
            raise ValueError("The action scales have incompatible dimensions or normalization stages")
    except (ValueError, RuntimeError) as exc:
        _add(findings, _GAIN, DiagnosticStatus.UNKNOWN, str(exc))
        return
    ratio = torch.where(
        old == 0, torch.where(new == 0, torch.ones_like(old), torch.full_like(old, torch.inf)), new / old
    )
    mask = ratio > threshold
    indices = mask.nonzero().flatten().tolist()
    index, key = new_sources[-1]
    old_index, old_key = old_sources[-1]
    selected = mask if indices else torch.ones_like(mask)
    source_tensor = candidate_pair[1][0][index].state_dict()[key]
    saved_slice = torch.broadcast_to(source_tensor.squeeze(), new.shape)[selected].tolist()
    _add(
        findings,
        _GAIN,
        DiagnosticStatus.FAIL if indices else DiagnosticStatus.PASS,
        "The candidate expands serialized action unnormalization scale."
        if indices
        else "The candidate does not expand serialized action unnormalization scale beyond the threshold.",
        {
            "feature_key": "action",
            "pipeline_role": "postprocessor",
            "step_index": None if candidate_pair[1][2]["steps"][index].get("_legacy") else index,
            "baseline_scale": old[selected].tolist(),
            "candidate_scale": new[selected].tolist(),
            "expansion_ratio": ratio[selected].tolist(),
            "scale_expansion_ratio": threshold,
            "tensor_slice": saved_slice,
            "saved_values": saved_slice,
            "effective_runtime_inverse_slope": new[selected].tolist(),
            "normalization_modes": new_modes,
            "scale_definitions": [
                "effective_runtime_std"
                if mode == "MEAN_STD"
                else "effective_runtime_bound_denominator_divided_by_two"
                for mode in new_modes
            ],
            "composition": "product_of_serialized_scales"
            if len(new_sources) > 1
            else "single_serialized_scale",
            "scope": "serialized_unnormalization_scales",
            "indexing": "action coordinate",
            "baseline_step_indices": [
                None if base_pair[1][2]["steps"][i].get("_legacy") else i for i, _ in old_sources
            ],
            "candidate_step_indices": [
                None if candidate_pair[1][2]["steps"][i].get("_legacy") else i for i, _ in new_sources
            ],
            "processor_execution_required": False,
            "language_constructors_invoked": False,
            "action_shape_source": candidate_pair[1][2]["steps"][index].get(
                "_action_shape_source", "processor_features"
            ),
            "physical_actuator_count_inferred": False,
            "provenance_digest_scope": "tensor_payload"
            if candidate_pair[1][2]["steps"][index].get("_legacy_raw_model")
            else "local_evidence_file",
            "baseline_provenance_digest_scope": "tensor_payload"
            if base_pair[1][2]["steps"][old_index].get("_legacy_raw_model")
            else "local_evidence_file",
            "baseline_contributors": [
                _step_provenance(base_pair[1][1], base_pair[1][2], i, k, indices).to_dict()
                for i, k in old_sources
            ],
            "candidate_contributors": [
                _step_provenance(candidate_pair[1][1], candidate_pair[1][2], i, k, indices).to_dict()
                for i, k in new_sources
            ],
            "baseline_provenance": _step_provenance(
                base_pair[1][1], base_pair[1][2], old_index, old_key, indices
            ).to_dict(),
        },
        _step_provenance(candidate_pair[1][1], candidate_pair[1][2], index, key, indices),
    )


def _runtime_dtype(config: dict[str, Any], index: int, default: torch.dtype = torch.float32) -> torch.dtype:
    dtype = default
    dtype_mapping: dict[str, torch.dtype] = DeviceProcessorStep.DTYPE_MAPPING
    for entry in config["steps"][:index]:
        if _step_class(entry) is DeviceProcessorStep:
            named = entry.get("config", {}).get("float_dtype")
            if named in dtype_mapping:
                dtype = dtype_mapping[named]
    return dtype


def _observation_maps(artifact: _Artifact, policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Inspect every active forward observation mapping, including image stats.

    Visual maps use the broadcast grid of saved statistics (usually C×1×1),
    avoiding full image allocations when each pixel shares a channel statistic.
    """
    normalizers, _, config = artifact
    if config.get("_normalizer_errors"):
        raise ValueError("Some preprocessor normalization buffers are unavailable")
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric_types = set(FeatureType) - {FeatureType.ACTION}
    for index, entry in enumerate(config["steps"]):
        step = normalizers.get(index)
        if step is None:
            if _language_only(entry):
                continue
            declared = entry.get("config", {})
            for key, feature in declared.get("features", {}).items():
                if (
                    feature.get("type") in {kind.value for kind in numeric_types}
                    and declared.get("norm_map", {}).get(feature["type"], "IDENTITY") != "IDENTITY"
                ):
                    raise ValueError(f"Uninspected active observation normalization for {key}")
            continue
        state = step.state_dict()
        for key, feature in step.features.items():
            if feature.type not in numeric_types:
                continue
            if step.normalize_observation_keys is not None and key not in step.normalize_observation_keys:
                continue
            mode = step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
            if mode is NormalizationMode.IDENTITY:
                continue
            if not isinstance(step, NormalizerProcessorStep) or mode not in _REQUIRED:
                raise ValueError(f"Unsupported observation normalization direction or mode for {key}")
            if len(feature.shape) != 1 and feature.type is not FeatureType.VISUAL:
                raise ValueError(f"Observation normalization requires a declared vector shape for {key}")
            if not math.isfinite(step.eps) or step.eps < 0:
                raise ValueError(f"Invalid observation normalization epsilon for {key}")
            dtype = _runtime_dtype(config, index, step.dtype or torch.float32)
            values = {}
            raw = {name: state.get(f"{key}.{name}") for name in _REQUIRED[mode]}
            if any(tensor is None for tensor in raw.values()):
                raise ValueError(f"Active observation buffers are incomplete for {key}")
            if feature.type is FeatureType.VISUAL:
                stat_shape = torch.broadcast_shapes(*(tensor.shape for tensor in raw.values()))
                full_broadcast = torch.broadcast_shapes(stat_shape, feature.shape)
                if full_broadcast[-len(feature.shape) :] != feature.shape or any(
                    dim != 1 for dim in full_broadcast[: -len(feature.shape)]
                ):
                    raise ValueError(f"Visual statistics do not broadcast to the declared image for {key}")
                # Native visual stats already reshape flat C to (C,1,1).
                # Scalar stats share one spatial/channel cell for probe purposes.
                stat_shape = stat_shape or (1,) * len(feature.shape)
            else:
                stat_shape = feature.shape
            for stat in _REQUIRED[mode]:
                tensor = state.get(f"{key}.{stat}")
                if tensor is None:
                    raise ValueError(f"Active observation buffer {key}.{stat} is missing")
                source = tensor if feature.type is FeatureType.VISUAL else tensor.squeeze()
                values[stat] = torch.broadcast_to(source.to(dtype), stat_shape)
                if not torch.isfinite(values[stat]).all():
                    raise ValueError(f"Active observation buffer {key}.{stat} is non-finite")
            if mode is NormalizationMode.MEAN_STD:
                if (values["std"] < 0).any():
                    raise ValueError(f"Negative observation standard deviation for {key}")
                denom = values["std"] + step.eps
                input_center = values["mean"].double()
                input_scale = denom.double()
                slope = input_scale.reciprocal()
                offset = -input_center / input_scale
                domain = (-3.0, 3.0)
            else:
                low_name, high_name = _REQUIRED[mode]
                span = values[high_name] - values[low_name]
                if (span < 0).any():
                    raise ValueError(f"Reversed observation interval for {key}")
                # Historical Normalize adds eps to every MIN_MAX span. Native
                # modern processors replace eps only when the span is zero.
                denom = (
                    span + step.eps
                    if entry.get("_legacy") and mode is NormalizationMode.MIN_MAX
                    else torch.where(span == 0, torch.full_like(span, step.eps), span)
                )
                input_scale = denom.double() / 2
                input_center = values[low_name].double() + input_scale
                slope = input_scale.reciprocal()
                offset = -input_center / input_scale
                domain = (-1.0, 1.0)
            if (
                not torch.isfinite(slope).all()
                or not torch.isfinite(offset).all()
                or (input_scale <= 0).any()
            ):
                raise ValueError(f"Observation affine mapping is non-finite or singular for {key}")
            mask = state.get(f"{key}.mask") if _kind(entry) in _MASKED_NORMALIZERS else None
            if mask is not None and mask.ndim == 1 and tuple(mask.shape) == feature.shape:
                slope, offset = torch.where(mask.bool(), slope, 1.0), torch.where(mask.bool(), offset, 0.0)
                input_center = torch.where(mask.bool(), input_center, 0.0)
                input_scale = torch.where(mask.bool(), input_scale, 1.0)
            result[key].append(
                {
                    "step_index": index,
                    "mode": mode.value,
                    "dtype": str(dtype),
                    "eps": step.eps,
                    "feature_type": feature.type.value,
                    "feature_shape": list(feature.shape),
                    "statistic_shape": list(stat_shape),
                    "slope": slope,
                    "offset": offset,
                    "input_center": input_center,
                    "input_scale": input_scale,
                    "input_lower": input_center + domain[0] * input_scale,
                    "input_upper": input_center + domain[1] * input_scale,
                    "domain": domain,
                    "stats": values,
                }
            )
    declared = policy.get("input_features") or {}
    mapping = policy.get("normalization_mapping") or {}
    for key, feature in declared.items():
        if (
            feature.get("type") in {kind.value for kind in numeric_types}
            and mapping.get(feature["type"], "IDENTITY") != "IDENTITY"
            and key not in result
        ):
            raise ValueError(
                f"Policy declares active observation normalization but no screened stage binds {key}"
            )
    return dict(result)


def _observation_delta(
    base_pair: _ArtifactPair,
    candidate_pair: _ArtifactPair,
    policies: list[dict[str, Any]],
    shift_threshold: float,
    scale_ratio: float,
    findings: list[DiagnosticFinding],
) -> None:
    try:
        old = _observation_maps(base_pair[0], policies[0])
        new = _observation_maps(candidate_pair[0], policies[1])
    except (ValueError, TypeError, KeyError, RuntimeError, AttributeError) as exc:
        _add(
            findings,
            _OBSERVATION,
            DiagnosticStatus.FAIL,
            f"Required observation normalization comparison coverage is unavailable: {exc}",
            {"required_coverage": True, "coverage_status": "UNKNOWN", "processor_execution_required": False},
            _artifact_provenance(candidate_pair[0]),
        )
        return
    for key in sorted(old.keys() | new.keys()):
        before, after = old.get(key, []), new.get(key, [])
        if len(before) != len(after):
            _add(
                findings,
                _OBSERVATION,
                DiagnosticStatus.FAIL,
                f"The number of active observation normalization stages changed for {key}.",
                {
                    "feature_key": key,
                    "baseline_active_stages": len(before),
                    "candidate_active_stages": len(after),
                    "required_coverage": True,
                },
                _artifact_provenance(candidate_pair[0]),
            )
            continue
        for previous, current in zip(before, after):
            try:
                shape = torch.broadcast_shapes(previous["slope"].shape, current["slope"].shape)

                def expanded(stage: dict[str, Any]) -> dict[str, Any]:
                    return {
                        **stage,
                        **{
                            name: torch.broadcast_to(stage[name], shape)
                            for name in (
                                "slope",
                                "offset",
                                "input_center",
                                "input_scale",
                                "input_lower",
                                "input_upper",
                            )
                        },
                        "stats": {
                            name: torch.broadcast_to(value, shape) for name, value in stage["stats"].items()
                        },
                        "statistic_shape": list(shape),
                    }

                previous, current = expanded(previous), expanded(current)
            except RuntimeError:
                _add(
                    findings,
                    _OBSERVATION,
                    DiagnosticStatus.FAIL,
                    f"Active observation shape changed for {key}.",
                    {"feature_key": key, "required_coverage": True},
                    _artifact_provenance(candidate_pair[0]),
                )
                continue
            center_shift = (current["input_center"] - previous["input_center"]).abs()
            relative_shift = center_shift / previous["input_scale"].abs().clamp_min(1e-12)
            ratio = current["slope"].abs() / previous["slope"].abs().clamp_min(1e-30)
            mask = (relative_shift > shift_threshold) | (ratio > scale_ratio) | (ratio < 1 / scale_ratio)
            if previous["mode"] != current["mode"] or previous["dtype"] != current["dtype"]:
                mask = torch.ones_like(mask)
            if not bool(mask.any()):
                continue
            index, previous_index = current["step_index"], previous["step_index"]
            stat = next(
                (
                    name
                    for name, values in current["stats"].items()
                    if name not in previous["stats"] or not torch.equal(values, previous["stats"][name])
                ),
                _REQUIRED[NormalizationMode(current["mode"])][0],
            )
            tensor_key = f"{key}.{stat}"
            indices = mask.reshape(-1).nonzero().flatten().tolist()
            provenance = _step_provenance(
                candidate_pair[0][1], candidate_pair[0][2], index, tensor_key, indices
            )
            baseline_provenance = _step_provenance(
                base_pair[0][1],
                base_pair[0][2],
                previous_index,
                f"{key}.{stat}"
                if stat in previous["stats"]
                else f"{key}.{_REQUIRED[NormalizationMode(previous['mode'])][0]}",
                indices,
            )
            source_is_eps = previous["eps"] != current["eps"] and all(
                name in previous["stats"] and torch.equal(values, previous["stats"][name])
                for name, values in current["stats"].items()
            )
            saved_slice = current["stats"][stat][mask].tolist()
            if source_is_eps:
                tensor_key = "config.eps"
                provenance = _provenance(candidate_pair[0][1], index, tensor_key, indices)
                baseline_provenance = _provenance(base_pair[0][1], previous_index, tensor_key, indices)
                saved_slice = [current["eps"]]
            _add(
                findings,
                _OBSERVATION,
                DiagnosticStatus.FAIL,
                f"The candidate changed the effective forward observation normalization mapping for {key}.",
                {
                    "feature_key": key,
                    "step_index": None if candidate_pair[0][2]["steps"][index].get("_legacy") else index,
                    "baseline_step_index": previous_index,
                    "required_coverage": True,
                    "baseline_mode": previous["mode"],
                    "candidate_mode": current["mode"],
                    "feature_type": current["feature_type"],
                    "declared_feature_shape": current["feature_shape"],
                    "statistic_shape": current["statistic_shape"],
                    "indexing": "flattened statistic broadcast grid",
                    "baseline_eps": previous["eps"],
                    "candidate_eps": current["eps"],
                    "baseline_input_center": previous["input_center"][mask].tolist(),
                    "candidate_input_center": current["input_center"][mask].tolist(),
                    "relative_input_offset_shift": relative_shift[mask].tolist(),
                    "baseline_forward_slope": previous["slope"][mask].tolist(),
                    "candidate_forward_slope": current["slope"][mask].tolist(),
                    "forward_slope_ratio": ratio[mask].tolist(),
                    "baseline_forward_offset": previous["offset"][mask].tolist(),
                    "candidate_forward_offset": current["offset"][mask].tolist(),
                    "baseline_input_lower": previous["input_lower"][mask].tolist(),
                    "baseline_input_upper": previous["input_upper"][mask].tolist(),
                    "candidate_input_lower": current["input_lower"][mask].tolist(),
                    "candidate_input_upper": current["input_upper"][mask].tolist(),
                    "baseline_stats": {
                        name: tensor[mask].tolist() for name, tensor in previous["stats"].items()
                    },
                    "candidate_stats": {
                        name: tensor[mask].tolist() for name, tensor in current["stats"].items()
                    },
                    "observation_shift_threshold": shift_threshold,
                    "observation_scale_ratio": scale_ratio,
                    "tensor_slice": saved_slice,
                    "source_field": tensor_key,
                    "comparison_scope": "individual_stage",
                    "processor_execution_required": False,
                    "baseline_provenance": baseline_provenance.to_dict(),
                },
                provenance,
            )
        _composed_observation_delta(
            base_pair[0], candidate_pair[0], key, before, after, shift_threshold, scale_ratio, findings
        )
    if not any(f.check_id == _OBSERVATION for f in findings):
        modern_features = [
            key
            for key in old
            if not any(
                entry.get("_legacy") for entry in base_pair[0][2]["steps"] + candidate_pair[0][2]["steps"]
            )
        ]
        probe_coverage_unknown = any(
            f.check_id == _INTERMEDIATE
            and f.status is DiagnosticStatus.UNKNOWN
            and f.details.get("feature_key") in modern_features
            for f in findings
        )
        _add(
            findings,
            _OBSERVATION,
            DiagnosticStatus.PASS,
            "All declared active numeric observation normalization stages agree within the configured thresholds.",
            {
                "screened_features": sorted(old),
                "required_coverage": True,
                "observation_shift_threshold": shift_threshold,
                "observation_scale_ratio": scale_ratio,
                "processor_execution_required": bool(modern_features),
                "native_boundary_probes_executed": bool(modern_features) and not probe_coverage_unknown,
                "native_boundary_probe_features": modern_features,
                "comparison_scope": "individual_stages_and_composed_policy_input",
            },
        )


def _observation_probe_points(stages: list[dict[str, Any]]) -> torch.Tensor:
    """Use the first stage's declared training boundaries, without inferred units."""
    first = stages[0]
    if first["mode"] == NormalizationMode.MEAN_STD.value:
        mean, std = first["stats"]["mean"].double(), first["stats"]["std"].double()
        return torch.stack((mean - std, mean, mean + std))
    low, high = _REQUIRED[NormalizationMode(first["mode"])]
    lower, upper = first["stats"][low].double(), first["stats"][high].double()
    return torch.stack((lower, (lower + upper) / 2, upper))


def _probe_native_artifact(
    artifact: _Artifact,
    feature_key: str,
    probes: torch.Tensor,
    checkpoint_role: str,
    findings: list[DiagnosticFinding],
) -> torch.Tensor | None:
    """Execute only the real native numeric subpath and retain every stage dtype."""
    from .preflight import _native_normalization_probe

    normalizers, path, config = artifact
    if any(entry.get("_legacy") for entry in config["steps"]):
        # Historical Normalize has different promotion and MIN_MAX epsilon rules;
        # modern synthetic steps cannot establish its actual execution behavior.
        return None
    try:
        snapshots = _native_normalization_probe(config["steps"], normalizers, feature_key, probes.float())
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
        _add(
            findings,
            _INTERMEDIATE,
            DiagnosticStatus.UNKNOWN,
            f"Native numeric boundary probes could not complete: {exc}",
            {
                "checkpoint_role": checkpoint_role,
                "feature_key": feature_key,
                "required_coverage": True,
                "probe_scope": "native_normalization_numeric_subpath",
            },
            _artifact_provenance(artifact),
        )
        return None
    for snapshot in snapshots:
        output = snapshot["output"]
        bad = ~torch.isfinite(output)
        if not bool(bad.any()):
            continue
        index = snapshot["step_index"]
        coordinates = bad.nonzero().tolist()
        coordinate_count = math.prod(probes.shape[1:])
        dimensions = bad.reshape(-1, coordinate_count).any(dim=0).nonzero().flatten().tolist()
        entered = torch.broadcast_to(snapshot["input"], output.shape)
        _add(
            findings,
            _INTERMEDIATE,
            DiagnosticStatus.FAIL,
            f"A native {checkpoint_role} processor produces NaN or Inf at an intermediate boundary-probe step.",
            {
                "checkpoint_role": checkpoint_role,
                "pipeline_role": "postprocessor" if feature_key == "action" else "preprocessor",
                "step_index": index,
                "step_class": snapshot["step_class"],
                "feature_key": feature_key,
                "tensor_key": feature_key,
                "input_dtype": snapshot["input_dtype"],
                "output_dtype": snapshot["output_dtype"],
                "input_shape": list(snapshot["input"].shape),
                "output_shape": list(output.shape),
                "probe_coordinates": coordinates,
                "offending_dimensions": dimensions,
                "input_values": entered[bad].tolist(),
                "output_values": output[bad].tolist(),
                "tensor_slice": output[bad].tolist(),
                "probe_inputs": probes.tolist(),
                "synthetic_boundary_probe": True,
                "required_coverage": True,
                "processor_execution_required": True,
                "execution_device": "cpu",
                "probe_scope": "native_normalization_numeric_subpath",
            },
            _step_provenance(path, config, index, feature_key, dimensions),
        )
        return None
    result = snapshots[-1]["output"].double() if snapshots else probes.double()
    if result.numel() != probes.numel() or result.shape[-1] != probes.shape[-1]:
        _add(
            findings,
            _INTERMEDIATE,
            DiagnosticStatus.UNKNOWN,
            "Native normalization changed the boundary-probe shape incompatibly with the declared feature.",
            {
                "checkpoint_role": checkpoint_role,
                "feature_key": feature_key,
                "required_coverage": True,
                "probe_input_shape": list(probes.shape),
                "probe_output_shape": list(result.shape),
            },
            _artifact_provenance(artifact),
        )
        return None
    # Native broadcastable (1, 1, D) stats can prepend singleton axes. Preserve
    # their execution above, then restore the equivalent probe-coordinate grid.
    return result.reshape_as(probes)


def _compose_observation_stages(stages: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    slope = torch.ones_like(stages[0]["slope"])
    offset = torch.zeros_like(slope)
    for stage in stages:
        slope, offset = stage["slope"] * slope, stage["slope"] * offset + stage["offset"]
    return slope, offset


def _composed_observation_delta(
    baseline: _Artifact,
    candidate: _Artifact,
    key: str,
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    shift_threshold: float,
    scale_ratio: float,
    findings: list[DiagnosticFinding],
) -> None:
    """Compare complete policy inputs, so small stage shifts cannot compound unseen."""
    if not previous or not current:
        return
    try:
        old_slope, old_offset = _compose_observation_stages(previous)
        new_slope, new_offset = _compose_observation_stages(current)
        common_shape = torch.broadcast_shapes(*(stage["slope"].shape for stage in previous + current))
        old_slope, old_offset = (
            torch.broadcast_to(old_slope, common_shape),
            torch.broadcast_to(old_offset, common_shape),
        )
        new_slope, new_offset = (
            torch.broadcast_to(new_slope, common_shape),
            torch.broadcast_to(new_offset, common_shape),
        )
        initial = {
            **previous[0],
            "stats": {
                name: torch.broadcast_to(value, common_shape) for name, value in previous[0]["stats"].items()
            },
        }
        probes = _observation_probe_points([initial])
    except RuntimeError as exc:
        _add(
            findings,
            _OBSERVATION,
            DiagnosticStatus.FAIL,
            f"Observation statistic grids cannot be composed for {key}: {exc}",
            {"feature_key": key, "required_coverage": True},
            _artifact_provenance(candidate),
        )
        return
    old_output = _probe_native_artifact(baseline, key, probes, "baseline", findings)
    new_output = _probe_native_artifact(candidate, key, probes, "candidate", findings)
    legacy = any(entry.get("_legacy") for entry in baseline[2]["steps"] + candidate[2]["steps"])
    if old_output is None or new_output is None:
        if not legacy:
            return  # A native probe emitted explicit failure/incomplete coverage.
        old_output, new_output = probes * old_slope + old_offset, probes * new_slope + new_offset
    ratio = new_slope.abs() / old_slope.abs().clamp_min(1e-30)
    output_shift = (new_output - old_output).abs().amax(dim=0)
    offset_shift = (new_offset - old_offset).abs()
    mask = (
        (output_shift > shift_threshold)
        | (offset_shift > shift_threshold)
        | (ratio > scale_ratio)
        | (ratio < 1 / scale_ratio)
    )
    mask |= (
        ~torch.isfinite(new_slope)
        | ~torch.isfinite(new_offset)
        | ~torch.isfinite(old_slope)
        | ~torch.isfinite(old_offset)
    )
    if not bool(mask.any()):
        return
    index, previous_index = current[-1]["step_index"], previous[-1]["step_index"]
    indices = mask.reshape(-1).nonzero().flatten().tolist()
    _add(
        findings,
        _OBSERVATION,
        DiagnosticStatus.FAIL,
        f"The candidate changes the composed policy-input observation normalization for {key}.",
        {
            "comparison_scope": "composed_policy_input",
            "feature_key": key,
            "required_coverage": True,
            "step_index": None if candidate[2]["steps"][index].get("_legacy") else index,
            "feature_type": current[-1]["feature_type"],
            "declared_feature_shape": current[-1]["feature_shape"],
            "statistic_shape": list(common_shape),
            "indexing": "flattened statistic broadcast grid",
            "baseline_step_indices": [stage["step_index"] for stage in previous],
            "candidate_step_indices": [stage["step_index"] for stage in current],
            "baseline_composed_slope": old_slope[mask].tolist(),
            "candidate_composed_slope": new_slope[mask].tolist(),
            "baseline_composed_offset": old_offset[mask].tolist(),
            "candidate_composed_offset": new_offset[mask].tolist(),
            "composed_slope_ratio": ratio[mask].tolist(),
            "composed_offset_shift": offset_shift[mask].tolist(),
            "policy_input_shift": output_shift[mask].tolist(),
            "probe_inputs": probes[:, mask].tolist(),
            "baseline_probe_outputs": old_output[:, mask].tolist(),
            "candidate_probe_outputs": new_output[:, mask].tolist(),
            "tensor_slice": new_output[:, mask].tolist(),
            "observation_shift_threshold": shift_threshold,
            "observation_scale_ratio": scale_ratio,
            "processor_execution_required": not legacy,
            "execution_device": "cpu",
            "baseline_provenance": _step_provenance(
                baseline[1], baseline[2], previous_index, key, indices
            ).to_dict(),
        },
        _step_provenance(candidate[1], candidate[2], index, key, indices),
    )


def _has_relative_contract(pair: _ArtifactPair) -> bool:
    return any(
        _step_class(entry) in (RelativeActionsProcessorStep, AbsoluteActionsProcessorStep)
        for _, _, config in pair
        for entry in config["steps"]
    )


def _relative_stage_affine(
    step: Any, key: str, kind: FeatureType, width: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inspect native affine constants; actual arithmetic is probed natively below."""
    unit, zero = torch.ones(width, dtype=torch.float64), torch.zeros(width, dtype=torch.float64)
    mode = step.norm_map.get(kind, NormalizationMode.IDENTITY)
    if mode is NormalizationMode.IDENTITY:
        return unit, zero
    state = step.state_dict()
    if mode not in _REQUIRED or any(f"{key}.{stat}" not in state for stat in _REQUIRED[mode]):
        raise ValueError(f"Relative-action mapping requires complete active statistics for {key}")
    values = {
        stat: torch.broadcast_to(state[f"{key}.{stat}"].squeeze().to(dtype), (width,))
        for stat in _REQUIRED[mode]
    }
    inverse = isinstance(step, UnnormalizerProcessorStep)
    if mode is NormalizationMode.MEAN_STD:
        if (values["std"] < 0).any():
            raise ValueError(f"Negative standard deviation in relative-action mapping for {key}")
        if inverse:
            return values["std"].double(), values["mean"].double()
        denominator = (values["std"] + step.eps).double()
        return denominator.reciprocal(), -values["mean"].double() / denominator
    low, high = _REQUIRED[mode]
    span = values[high] - values[low]
    if (span < 0).any():
        raise ValueError(f"Reversed bounds in relative-action mapping for {key}")
    denominator = torch.where(span == 0, torch.full_like(span, step.eps), span).double()
    return (
        (denominator / 2, values[low].double() + denominator / 2)
        if inverse
        else (2 / denominator, -2 * values[low].double() / denominator - 1)
    )


def _relative_action_mapping(root: Path, policy: dict[str, Any], pair: _ArtifactPair) -> dict[str, Any]:
    """Compose native relative/absolute offsets as A*action + S*raw_state + b.

    S includes the observation mapping at the exact cache point. Later action
    unnormalizers scale that cached offset; normalizers after the cache do not
    retroactively change it. No tensor shape is treated as physical metadata.
    """
    from .temporal import _resolve_metadata

    dtype_mapping: dict[str, torch.dtype] = DeviceProcessorStep.DTYPE_MAPPING
    action_shape = (policy.get("output_features") or {}).get("action", {}).get("shape", [])
    state_shape = (policy.get("input_features") or {}).get("observation.state", {}).get("shape", [])
    if (
        len(action_shape) != 1
        or len(state_shape) != 1
        or any(type(width) is not int or width < 1 for width in action_shape + state_shape)
    ):
        raise ValueError(
            "Relative-action comparison requires declared vector action and observation.state widths"
        )
    width, state_width = action_shape[0], state_shape[0]
    relative_indices = [
        i for i, entry in enumerate(pair[0][2]["steps"]) if _step_class(entry) is RelativeActionsProcessorStep
    ]
    absolute_indices = [
        i for i, entry in enumerate(pair[1][2]["steps"]) if _step_class(entry) is AbsoluteActionsProcessorStep
    ]
    if len(relative_indices) != 1 or len(absolute_indices) != 1:
        raise ValueError("One paired relative preprocessor and absolute postprocessor is required")
    relative_index, absolute_index = relative_indices[0], absolute_indices[0]
    relative = RelativeActionsProcessorStep(**pair[0][2]["steps"][relative_index].get("config", {}))
    absolute = AbsoluteActionsProcessorStep(
        **pair[1][2]["steps"][absolute_index].get("config", {}), relative_step=relative
    )
    mask = torch.tensor(relative._build_mask(width), dtype=torch.float64)
    if not relative.enabled and not absolute.enabled:
        mask.zero_()
    if len(mask) > state_width:
        raise ValueError("The declared state cannot satisfy the native relative-action mask width")
    metadata, problem = _resolve_metadata(root)
    if (relative.enabled or absolute.enabled) and problem is not None:
        raise ValueError(f"Relative-action anchor cannot be resolved: {problem.summary}")
    metadata = metadata or {
        "n_obs_steps": 1,
        "expected_anchor_index": 0,
        "chronology_source": "disabled_conversion",
    }
    maps, native_pipelines, contributors = {}, {}, []
    state_probe_points = None
    state_slope, state_offset = (
        torch.ones(state_width, dtype=torch.float64),
        torch.zeros(state_width, dtype=torch.float64),
    )
    cached_slope = cached_offset = None
    current_key = "observation.state"
    for role, (normalizers, path, config) in zip(("preprocessor", "postprocessor"), pair):
        if config.get("_normalizer_errors"):
            raise ValueError("Relative-action mapping requires readable normalization buffers")
        action_slope, state_coefficient, offset = (
            torch.ones(width, dtype=torch.float64),
            torch.zeros(width, dtype=torch.float64),
            torch.zeros(width, dtype=torch.float64),
        )
        dtype = torch.float32
        native_steps, serialized_indices = [], []
        for index, entry in enumerate(config["steps"]):
            cls, step = _step_class(entry), normalizers.get(index)
            if _language_only(entry):
                continue
            if _kind(entry) in _MASKED_NORMALIZERS or _clip_keys(entry):
                raise ValueError("Masked or clipped relative-action maps are not verified affine chains")
            if step is not None:
                native = deepcopy(step)
                native.to(device="cpu")
                if role == "preprocessor" and current_key in step.features:
                    feature = step.features[current_key]
                    if (
                        step.normalize_observation_keys is None
                        or current_key in step.normalize_observation_keys
                    ):
                        mode = step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
                        if state_probe_points is None and mode in _REQUIRED:
                            raw = {
                                name: torch.broadcast_to(
                                    step.state_dict()[f"{current_key}.{name}"].squeeze().float(),
                                    (state_width,),
                                )
                                for name in _REQUIRED[mode]
                            }
                            if mode is NormalizationMode.MEAN_STD:
                                state_probe_points = torch.stack(
                                    (raw["mean"] - raw["std"], raw["mean"], raw["mean"] + raw["std"])
                                )
                            else:
                                lower, upper = (raw[name] for name in _REQUIRED[mode])
                                state_probe_points = torch.stack((lower, (lower + upper) / 2, upper))
                        a, b = _relative_stage_affine(step, current_key, feature.type, state_width, dtype)
                        state_slope, state_offset = a * state_slope, a * state_offset + b
                a, b = _relative_stage_affine(step, "action", FeatureType.ACTION, width, dtype)
                action_slope, state_coefficient, offset = (
                    a * action_slope,
                    a * state_coefficient,
                    a * offset + b,
                )
                contributors.append(
                    {
                        "role": role,
                        "step_index": index,
                        "provenance": _step_provenance(path, config, index, "action").to_dict(),
                    }
                )
            elif cls is DeviceProcessorStep:
                named = entry.get("config", {}).get("float_dtype")
                if named is not None:
                    dtype = dtype_mapping[named]
                native = cls(**{**entry.get("config", {}), "device": "cpu"})
            elif cls is RenameObservationsProcessorStep:
                native = cls(**entry.get("config", {}))
                if role == "preprocessor":
                    current_key = next(iter(native.observation({current_key: None})))
            elif cls is RelativeActionsProcessorStep and role == "preprocessor" and index == relative_index:
                if current_key != "observation.state":
                    raise ValueError(
                        "The relative step's observation.state cache is not available after renaming"
                    )
                native = relative
                cached_slope, cached_offset = state_slope[:width].clone(), state_offset[:width].clone()
                if relative.enabled:
                    state_coefficient -= mask * cached_slope
                    offset -= mask * cached_offset
            elif cls is AbsoluteActionsProcessorStep and role == "postprocessor" and index == absolute_index:
                native = absolute
                if cached_slope is None:
                    raise ValueError("Absolute conversion has no inspectable native relative-state cache")
                if absolute.enabled:
                    state_coefficient += mask * cached_slope
                    offset += mask * cached_offset
            elif cls is not None and cls in (IdentityProcessorStep, AddBatchDimensionProcessorStep):
                native = cls(**entry.get("config", {}))
            else:
                raise ValueError(f"Uninspected relative-action subpath step {index}: {_kind(entry)}")
            native_steps.append(native)
            serialized_indices.append(index)
        maps[role] = {"action_slope": action_slope, "state_coefficient": state_coefficient, "offset": offset}
        native_pipelines[role] = (
            PolicyProcessorPipeline(
                steps=native_steps, to_transition=lambda value: value, to_output=lambda value: value
            ),
            serialized_indices,
            path,
            config,
        )
    return {
        "maps": maps,
        "native_pipelines": native_pipelines,
        "width": width,
        "state_width": state_width,
        "metadata": metadata,
        "mask": mask,
        "relative_index": relative_index,
        "absolute_index": absolute_index,
        "state_probe_points": state_probe_points,
        "enabled": [relative.enabled, absolute.enabled],
        "contributors": contributors,
        "provenance": _provenance(pair[0][1], "RelativeActionsProcessorStep", "observation.state"),
    }


def _relative_native_probes(
    mapping: dict[str, Any],
    role: str,
    findings: list[DiagnosticFinding],
    state_points: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Probe every intermediate of paired real native pipelines, never a surrogate."""
    count, width, state_width = mapping["metadata"]["n_obs_steps"], mapping["width"], mapping["state_width"]
    # Independent action/state combinations expose offset changes and casts.
    points = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    if state_points is None:
        state_points = mapping["state_probe_points"]
    if state_points is None:
        state_points = torch.tensor([-1.0, 0.0, 1.0]).reshape(3, 1).expand(3, state_width)
    action = points.repeat_interleave(len(state_points)).reshape(-1, 1, 1).expand(-1, 1, width).clone()
    state = state_points.repeat(5, 1).reshape(-1, 1, state_width).expand(-1, count, state_width).clone()
    state += torch.arange(count).reshape(1, -1, 1) * 0.25
    transition = create_transition(action=action, observation={"observation.state": state})
    outputs = {}
    for pipeline_role in ("preprocessor", "postprocessor"):
        pipeline, indices, path, config = mapping["native_pipelines"][pipeline_role]
        # Postprocessing consumes policy outputs independently of training
        # action normalization, while reading the native cache populated above.
        if pipeline_role == "postprocessor":
            transition = create_transition(action=action.clone())
        iterator = iter(pipeline.step_through(transition))
        next(iterator)
        try:
            with torch.inference_mode():
                for index in indices:
                    transition = next(iterator)
                    for key, tensor in (
                        ("action", transition.get("action")),
                        ("observation.state", (transition.get("observation") or {}).get("observation.state")),
                    ):
                        if not isinstance(tensor, torch.Tensor) or torch.isfinite(tensor).all():
                            continue
                        bad = ~torch.isfinite(tensor)
                        dimensions = bad.reshape(-1, tensor.shape[-1]).any(0).nonzero().flatten().tolist()
                        _add(
                            findings,
                            _INTERMEDIATE,
                            DiagnosticStatus.FAIL,
                            f"Native {role} relative-action boundary probes produce NaN or Inf.",
                            {
                                "checkpoint_role": role,
                                "pipeline_role": pipeline_role,
                                "step_index": index,
                                "feature_key": key,
                                "output_dtype": str(tensor.dtype),
                                "tensor_slice": tensor[bad].tolist(),
                                "offending_dimensions": dimensions,
                                "required_coverage": True,
                                "processor_execution_required": True,
                                "scope": "paired_native_relative_action_chain",
                                "probe_action_inputs": action.tolist(),
                                "probe_state_inputs": state.tolist(),
                            },
                            _step_provenance(path, config, index, key, dimensions),
                        )
                        return None
        except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
            _add(
                findings,
                _INTERMEDIATE,
                DiagnosticStatus.UNKNOWN,
                f"Native paired relative-action probes could not complete: {exc}",
                {
                    "checkpoint_role": role,
                    "required_coverage": True,
                    "scope": "paired_native_relative_action_chain",
                },
                _provenance(path),
            )
            return None
        outputs[pipeline_role] = transition["action"].detach().double().clone()
    return outputs


def _relative_action_delta(
    baseline: Path,
    candidate: Path,
    policies: list[dict[str, Any]],
    pairs: tuple[_ArtifactPair, _ArtifactPair],
    scale_ratio: float,
    offset_threshold: float,
    findings: list[DiagnosticFinding],
    envelope_abs_tol: float = 1e-6,
) -> None:
    """Check relative offset invariance alongside the ordinary action slope."""
    try:
        old, new = (
            _relative_action_mapping(root, policy, pair)
            for root, policy, pair in zip((baseline, candidate), policies, pairs)
        )
        if old["width"] != new["width"] or old["state_width"] != new["state_width"]:
            raise ValueError("Relative-action state/action widths differ")
    except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
        for check in (_GAIN, _OFFSET, _ENVELOPE):
            _add(findings, check, DiagnosticStatus.UNKNOWN, str(exc), {"scope": "relative_action_mapping"})
        return
    # Compare identical raw observations. The union still screens both saved
    # boundary sets for overflow when two equivalent chains reparameterize them.
    state_points = torch.unique(
        torch.cat(
            [
                mapping["state_probe_points"]
                if mapping["state_probe_points"] is not None
                else torch.tensor([-1.0, 0.0, 1.0]).reshape(3, 1).expand(3, mapping["state_width"])
                for mapping in (old, new)
            ]
        ),
        dim=0,
    )
    old_probe = _relative_native_probes(old, "baseline", findings, state_points)
    new_probe = _relative_native_probes(new, "candidate", findings, state_points)
    if old_probe is None or new_probe is None:
        for check in (_GAIN, _OFFSET, _ENVELOPE):
            _add(
                findings,
                check,
                DiagnosticStatus.UNKNOWN,
                "Paired native relative-action boundary probes did not complete.",
                {"scope": "relative_action_mapping"},
            )
        return
    width = new["width"]
    mask_change = old["mask"] != new["mask"]
    enabled_change = old["enabled"] != new["enabled"]
    anchor_change = old["metadata"]["expected_anchor_index"] != new["metadata"]["expected_anchor_index"]
    invalid_anchor = any(
        mapping["metadata"]["expected_anchor_index"] != 0 and any(mapping["enabled"])
        for mapping in (old, new)
    )
    structural = mask_change.clone()
    if enabled_change or anchor_change or invalid_anchor:
        structural |= old["mask"].bool() | new["mask"].bool()
    gain_mask, offset_mask = torch.zeros(width, dtype=torch.bool), structural.clone()
    metrics = {}
    for pipeline_role in ("preprocessor", "postprocessor"):
        before, after = old["maps"][pipeline_role], new["maps"][pipeline_role]
        ratio = torch.where(
            before["action_slope"] == 0,
            torch.where(after["action_slope"] == 0, torch.ones(width), torch.full((width,), torch.inf)),
            after["action_slope"].abs() / before["action_slope"].abs(),
        )
        gain_mask |= ratio > scale_ratio
        # Changing the anchor's state coefficient is a coordinate-map change,
        # even if a probe happens to use zero state or a round trip cancels it.
        offset_mask |= (after["state_coefficient"] - before["state_coefficient"]).abs() > 1e-12
        offset_mask |= (after["offset"] - before["offset"]).abs() > 1e-12
        if pipeline_role == "preprocessor":
            # Training-action coordinates are part of the relative contract,
            # including scale contractions below the output-gain threshold.
            offset_mask |= (after["action_slope"] - before["action_slope"]).abs() > 1e-12
        normalized_offset = (after["offset"] - before["offset"]).abs() / before[
            "action_slope"
        ].abs().clamp_min(1e-12)
        shift = (new_probe[pipeline_role] - old_probe[pipeline_role]).abs().reshape(-1, width).amax(0)
        offset_mask |= normalized_offset > offset_threshold
        offset_mask |= shift > offset_threshold * before["action_slope"].abs().clamp_min(1e-12)
        offset_mask |= ~torch.isfinite(after["offset"]) | ~torch.isfinite(after["state_coefficient"])
        metrics[pipeline_role] = {
            "baseline": {key: value.tolist() for key, value in before.items()},
            "candidate": {key: value.tolist() for key, value in after.items()},
            "action_slope_ratio": ratio.tolist(),
            "native_probe_shift": shift.tolist(),
        }
    # Envelope escapes are independent of the configurable gain-warning ratio.
    # Equal state coefficients cancel the unknown raw-state term on both sides.
    old_post, new_post = old["maps"]["postprocessor"], new["maps"]["postprocessor"]
    old_low = old_post["offset"] - 3 * old_post["action_slope"].abs()
    old_high = old_post["offset"] + 3 * old_post["action_slope"].abs()
    new_low = new_post["offset"] - 3 * new_post["action_slope"].abs()
    new_high = new_post["offset"] + 3 * new_post["action_slope"].abs()
    tolerance = (old_post["action_slope"].abs() * envelope_abs_tol).clamp_min(1e-12)
    envelope_mask = offset_mask | (new_low < old_low - tolerance) | (new_high > old_high + tolerance)
    changes = []
    for artifact_index, pipeline_role in ((1, "postprocessor"), (0, "preprocessor")):
        previous_artifact, current_artifact = pairs[0][artifact_index], pairs[1][artifact_index]
        for index, step in current_artifact[0].items():
            prior = previous_artifact[0].get(index)
            if prior is None:
                continue
            before_state, after_state = prior.state_dict(), step.state_dict()
            for key, tensor in sorted(after_state.items()):
                if not key.startswith(("action.", "observation.state.")):
                    continue
                previous = before_state.get(key)
                if previous is None or not torch.equal(previous, tensor):
                    changes.append(
                        {
                            "pipeline_role": pipeline_role,
                            "artifact_index": artifact_index,
                            "step_index": index,
                            "tensor_key": key,
                            "baseline_provenance": _step_provenance(
                                previous_artifact[1], previous_artifact[2], index, key
                            ).to_dict(),
                            "candidate_provenance": _step_provenance(
                                current_artifact[1], current_artifact[2], index, key
                            ).to_dict(),
                        }
                    )
    for check, mask in ((_GAIN, gain_mask), (_OFFSET, offset_mask), (_ENVELOPE, envelope_mask)):
        indices = mask.nonzero().flatten().tolist()
        step_index, pipeline_role = new["absolute_index"], "postprocessor"
        provenance = _provenance(pairs[1][1][1], "AbsoluteActionsProcessorStep", "action", indices)
        if changes:
            change = changes[0]
            artifact = pairs[1][change["artifact_index"]]
            step_index, pipeline_role = change["step_index"], change["pipeline_role"]
            provenance = _step_provenance(artifact[1], artifact[2], step_index, change["tensor_key"], indices)
        elif mask_change.any() or enabled_change:
            step_index, pipeline_role = new["relative_index"], "preprocessor"
            provenance = _provenance(
                pairs[1][0][1], "RelativeActionsProcessorStep", "config.effective_mask", indices
            )
        elif anchor_change or invalid_anchor:
            step_index, pipeline_role = new["relative_index"], "preprocessor"
            provenance = _provenance(
                candidate / "config.json",
                "RelativeActionsProcessorStep",
                "state_observation_delta_indices"
                if new["metadata"].get("state_observation_delta_indices") is not None
                else "observation_delta_indices",
                indices,
            )
        _add(
            findings,
            check,
            DiagnosticStatus.FAIL if indices else DiagnosticStatus.PASS,
            "The candidate changes a verified relative-action affine/anchor contract."
            if indices
            else "Paired relative/absolute action offsets and surrounding affine stages are invariant within configured thresholds.",
            {
                "scope": "paired_relative_action_affine_mapping",
                "feature_key": "action",
                "step_index": step_index,
                "pipeline_role": pipeline_role,
                "changed_statistics": changes,
                "baseline_mask": old["mask"].bool().tolist(),
                "candidate_mask": new["mask"].bool().tolist(),
                "baseline_enabled": old["enabled"],
                "candidate_enabled": new["enabled"],
                "baseline_anchor_index": old["metadata"]["expected_anchor_index"],
                "candidate_anchor_index": new["metadata"]["expected_anchor_index"],
                "native_anchor_index": 0,
                "invalid_native_anchor_contract": invalid_anchor,
                "mask_changed": bool(mask_change.any()),
                "enabled_changed": enabled_change,
                "anchor_changed": anchor_change,
                "affine_maps": metrics,
                "tensor_slice": new_probe["postprocessor"].reshape(-1, width)[:, indices].tolist()
                if indices
                else [],
                "required_coverage": True,
                "processor_execution_required": True,
                "execution_device": "cpu",
                "scale_expansion_ratio": scale_ratio,
                "offset_shift_threshold": offset_threshold,
                "relative_offset_invariance_atol": 1e-12,
                "envelope_abs_tol": envelope_abs_tol,
                "normalized_comparison_interval": [-3.0, 3.0],
                "hardware_limits_inferred": False,
                "baseline_lower_without_state_term": old_low.tolist(),
                "baseline_upper_without_state_term": old_high.tolist(),
                "candidate_lower_without_state_term": new_low.tolist(),
                "candidate_upper_without_state_term": new_high.tolist(),
                "baseline_provenance": old["provenance"].to_dict(),
                "candidate_contributors": new["contributors"],
            },
            provenance,
        )


def _action_boundary_probes(
    pair: _ArtifactPair, checkpoint_role: str, findings: list[DiagnosticFinding]
) -> None:
    artifact = pair[1]
    for step in artifact[0].values():
        feature = step.features.get("action")
        if feature is not None and feature.type == FeatureType.ACTION and len(feature.shape) == 1:
            probes = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]).view(-1, 1).expand(5, feature.shape[0]).clone()
            _probe_native_artifact(artifact, "action", probes, checkpoint_role, findings)
            return


def _action_affine(artifact: _Artifact) -> dict[str, Any]:
    """Compose the exact native inverse-normalization affine formulas.

    The probe interval is [-3, 3] for an initial MEAN_STD stage and [-1, 1]
    for bounded modes. These are diagnostic comparison domains, not hardware
    limits or an assertion that a policy's outputs remain inside them.
    """
    normalizers, _, config = artifact
    if config.get("_normalizer_errors"):
        raise ValueError("Action affine comparison requires readable postprocessor normalization buffers")
    slope = origin = None
    sources, modes = [], []
    domain = None
    for index, entry in enumerate(config["steps"]):
        step = normalizers.get(index)
        if step is None:
            cls = _step_class(entry)
            if cls in (RelativeActionsProcessorStep, AbsoluteActionsProcessorStep):
                raise ValueError(
                    "State-dependent action conversion prevents a complete static affine-origin comparison"
                )
            if cls is None and not _language_only(entry) and not _clip_keys(entry):
                raise ValueError(
                    f"Uninspected postprocessor step prevents a complete affine comparison: {_kind(entry)}"
                )
            continue
        mode = step.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY:
            continue
        if not isinstance(step, UnnormalizerProcessorStep):
            raise ValueError(
                "Mixed-direction postprocessor action normalization is not covered by this affine comparison"
            )
        feature = step.features.get("action")
        if (
            feature is None
            or feature.type != FeatureType.ACTION
            or len(feature.shape) != 1
            or mode not in _REQUIRED
        ):
            raise ValueError("A supported one-dimensional action contract is required for affine comparison")
        state = step.state_dict()
        dtype = _runtime_dtype(config, index, step.dtype or torch.float32)
        if not math.isfinite(step.eps) or step.eps < 0:
            raise ValueError("Action affine epsilon must be finite and nonnegative")
        values = {}
        for name in _REQUIRED[mode]:
            tensor = state.get(f"action.{name}")
            if tensor is None:
                raise ValueError(f"The action affine buffer action.{name} is missing")
            values[name] = torch.broadcast_to(tensor.squeeze().to(dtype), feature.shape)
            if not torch.isfinite(values[name]).all():
                raise ValueError(f"The action affine buffer action.{name} is non-finite")
        if mode == NormalizationMode.MEAN_STD:
            a, b, origin_key = values["std"].double(), values["mean"].double(), "action.mean"
            if (a < 0).any():
                raise ValueError("Negative action std prevents a valid affine-envelope comparison")
            initial_domain = (-3.0, 3.0)
        else:
            low_name, high_name = _REQUIRED[mode]
            span = values[high_name] - values[low_name]
            if (span < 0).any():
                raise ValueError("Reversed action bounds prevent a valid affine-envelope comparison")
            # Native inverse bounds modes substitute eps only for a zero span.
            denom = torch.where(span == 0, torch.full_like(span, step.eps), span).double()
            a, b, origin_key = denom / 2, values[low_name].double() + denom / 2, f"action.{low_name}"
            initial_domain = (-1.0, 1.0)
        if "action.mask" in state and _kind(entry) == "molmoact2_masked_unnormalizer":
            mask = state["action.mask"]
            if mask.ndim != 1 or tuple(mask.shape) != feature.shape:
                raise ValueError("The action normalization mask has an incompatible shape")
            a, b = torch.where(mask.bool(), a, 1.0), torch.where(mask.bool(), b, 0.0)
        if slope is None:
            slope, origin, domain = a, b, initial_domain
        else:
            slope, origin = a * slope, a * origin + b
        if not torch.isfinite(slope).all() or not torch.isfinite(origin).all():
            raise ValueError("Composed action affine parameters are non-finite")
        sources.append((index, origin_key))
        modes.append(mode.value)
    if slope is None:
        raise ValueError("No active action affine unnormalizer is available")
    assert domain is not None
    return {
        "slope": slope,
        "origin": origin,
        "lower": slope * domain[0] + origin,
        "upper": slope * domain[1] + origin,
        "domain": domain,
        "sources": sources,
        "modes": modes,
    }


def _affine_delta(
    base_pair: _ArtifactPair,
    candidate_pair: _ArtifactPair,
    threshold: float,
    findings: list[DiagnosticFinding],
    envelope_abs_tol: float = 1e-6,
) -> None:
    try:
        old, new = _action_affine(base_pair[1]), _action_affine(candidate_pair[1])
        if old["slope"].shape != new["slope"].shape or old["modes"] != new["modes"]:
            raise ValueError("Action affine mappings have incompatible dimensions or normalization stages")
    except (ValueError, RuntimeError) as exc:
        for check in (_OFFSET, _ENVELOPE):
            _add(
                findings,
                check,
                DiagnosticStatus.UNKNOWN,
                str(exc),
                {"scope": "serialized_action_affine_mapping"},
            )
        return
    displacement = (new["origin"] - old["origin"]).abs()
    relative = displacement / old["slope"].abs().clamp_min(1e-12)
    offset_mask = relative > threshold
    tolerance = (envelope_abs_tol * old["slope"].abs()).clamp_min(1e-12)
    escape_mask = (new["lower"] < old["lower"] - tolerance) | (new["upper"] > old["upper"] + tolerance)
    index, key = new["sources"][-1]
    old_index, old_key = old["sources"][-1]
    checks: list[tuple[str, torch.Tensor, str, str, str | None]] = [
        (
            _OFFSET,
            offset_mask,
            "The candidate shifts the serialized action origin beyond the allowed baseline-scale fraction.",
            "The serialized action origin does not shift beyond the configured threshold.",
            None,
        ),
    ]
    if new["modes"][-1] != "MEAN_STD" and bool(escape_mask.any()):
        for side, escaped in (
            ("lower", new["lower"] < old["lower"] - tolerance),
            ("upper", new["upper"] > old["upper"] + tolerance),
        ):
            if bool(escaped.any()):
                checks.append(
                    (
                        _ENVELOPE,
                        escaped,
                        f"The candidate's implied action {side} bound extends outside the baseline interval.",
                        "",
                        side,
                    )
                )
    else:
        checks.append(
            (
                _ENVELOPE,
                escape_mask,
                "The candidate's implied action interval extends outside the baseline interval.",
                "The candidate's implied action interval remains within the baseline interval.",
                None,
            )
        )
    for check, mask, failure, success, bound_side in checks:
        indices = mask.nonzero().flatten().tolist()
        selected = mask if indices else torch.ones_like(mask)
        source_key, old_source_key = key, old_key
        if bound_side is not None:
            stat_index = 0 if bound_side == "lower" else 1
            source_key = "action." + _REQUIRED[NormalizationMode(new["modes"][-1])][stat_index]
            old_source_key = "action." + _REQUIRED[NormalizationMode(old["modes"][-1])][stat_index]
        elif check == _ENVELOPE and new["modes"][-1] == "MEAN_STD" and not bool(displacement[selected].any()):
            source_key = old_source_key = "action.std"
        source_tensor = candidate_pair[1][0][index].state_dict()[source_key]
        saved_slice = torch.broadcast_to(source_tensor.squeeze(), new["origin"].shape)[selected].tolist()
        _add(
            findings,
            check,
            DiagnosticStatus.FAIL if indices else DiagnosticStatus.PASS,
            failure if indices else success,
            {
                "feature_key": "action",
                "pipeline_role": "postprocessor",
                "step_index": None if candidate_pair[1][2]["steps"][index].get("_legacy") else index,
                "baseline_origin": old["origin"][selected].tolist(),
                "candidate_origin": new["origin"][selected].tolist(),
                "absolute_origin_shift": displacement[selected].tolist(),
                "relative_origin_shift": relative[selected].tolist(),
                "offset_shift_threshold": threshold,
                "offset_threshold_units": "baseline_affine_scale_fraction",
                "envelope_abs_tol": envelope_abs_tol,
                "envelope_tolerance": tolerance[selected].tolist(),
                "envelope_tolerance_formula": "max(envelope_abs_tol * abs(baseline_inverse_slope), 1e-12)",
                "baseline_lower": old["lower"][selected].tolist(),
                "baseline_upper": old["upper"][selected].tolist(),
                "candidate_lower": new["lower"][selected].tolist(),
                "candidate_upper": new["upper"][selected].tolist(),
                "normalized_comparison_interval": list(new["domain"]),
                "normalization_modes": new["modes"],
                "tensor_slice": saved_slice,
                "saved_values": saved_slice,
                "saved_buffer_shape": list(source_tensor.shape),
                "slice_indexing": "action coordinate after broadcasting the saved buffer",
                "bound_side": bound_side,
                "hardware_limits_inferred": False,
                "scope": "serialized_action_affine_mapping_before_clipping",
                "processor_execution_required": False,
                "baseline_provenance": _step_provenance(
                    base_pair[1][1], base_pair[1][2], old_index, old_source_key, indices
                ).to_dict(),
                "candidate_contributors": [
                    _step_provenance(candidate_pair[1][1], candidate_pair[1][2], i, k, indices).to_dict()
                    for i, k in new["sources"]
                ],
            },
            _step_provenance(candidate_pair[1][1], candidate_pair[1][2], index, source_key, indices),
        )


def run_privilege_delta(
    baseline_dir: Path,
    candidate_dir: Path,
    scale_expansion_ratio: float = 1.05,
    offset_shift_threshold: float = 0.05,
    observation_shift_threshold: float = 0.05,
    observation_scale_ratio: float = 1.05,
    envelope_abs_tol: float = 1e-6,
) -> list[DiagnosticFinding]:
    """Compare numeric affine maps, explicit execution horizons and retained guards.

    ``offset_shift_threshold`` is the maximum absolute origin displacement as
    a fraction of the baseline affine scale (default 0.05). Envelope escapes
    are also reported independently, even when that fraction is not exceeded.

    No units, coordinate frames, controller timing or external hardware safety
    limits are inferred. Language and policy-specific processor constructors are
    never needed: inspected native normalization steps read local buffers and
    evaluate bounded synthetic transitions on CPU, retaining intermediate dtype.
    """
    if not math.isfinite(scale_expansion_ratio) or scale_expansion_ratio < 1:
        raise ValueError("scale_expansion_ratio must be finite and at least 1.0")
    if not math.isfinite(offset_shift_threshold) or offset_shift_threshold < 0:
        raise ValueError("offset_shift_threshold must be finite and nonnegative")
    if not math.isfinite(observation_shift_threshold) or observation_shift_threshold < 0:
        raise ValueError("observation_shift_threshold must be finite and nonnegative")
    if not math.isfinite(observation_scale_ratio) or observation_scale_ratio < 1:
        raise ValueError("observation_scale_ratio must be finite and at least 1.0")
    if not math.isfinite(envelope_abs_tol) or envelope_abs_tol < 0:
        raise ValueError("envelope_abs_tol must be finite and nonnegative")
    baseline, candidate = Path(baseline_dir).resolve(), Path(candidate_dir).resolve()
    findings: list[DiagnosticFinding] = []
    policies = []
    for root in (baseline, candidate):
        try:
            policies.append(_policy(root))
        except (ValueError, OSError) as exc:
            policies.append({})
            _add(
                findings,
                _COMPAT,
                DiagnosticStatus.CONFLICT,
                str(exc),
                provenance=_provenance(root / "config.json"),
            )
    _horizon(baseline, candidate, policies[0], policies[1], findings)
    try:
        base_pair = _static_pair(baseline, policies[0])
        candidate_pair = _static_pair(candidate, policies[1])
    except (ValueError, OSError, TypeError, KeyError, RuntimeError, AttributeError) as exc:
        _add(
            findings,
            _OBSERVATION,
            DiagnosticStatus.FAIL,
            f"Required observation normalization comparison coverage is unavailable: {exc}",
            {"required_coverage": True, "coverage_status": "UNKNOWN"},
        )
        _add(
            findings,
            _COMPAT,
            DiagnosticStatus.UNKNOWN,
            f"Serialized processor contracts could not be read: {exc}",
        )
        _add(
            findings,
            _GAIN,
            DiagnosticStatus.UNKNOWN,
            "Local active action unnormalization buffers are unavailable or unbound.",
        )
        try:
            _guard_delta(baseline, candidate, _configs(baseline), _configs(candidate), findings)
        except (ValueError, OSError, TypeError, KeyError, RuntimeError, AttributeError) as guard_exc:
            _add(
                findings,
                _GUARD,
                DiagnosticStatus.UNKNOWN,
                f"Guard retention could not be fully determined: {guard_exc}",
            )
        if not any(f.check_id == _GUARD for f in findings):
            _add(
                findings,
                _GUARD,
                DiagnosticStatus.UNKNOWN,
                "Guard retention cannot be established without readable serialized normalization contracts.",
            )
    else:
        relative_contract = _has_relative_contract(base_pair) or _has_relative_contract(candidate_pair)
        _observation_delta(
            base_pair,
            candidate_pair,
            policies,
            observation_shift_threshold,
            observation_scale_ratio,
            findings,
        )
        if not relative_contract:
            _action_boundary_probes(base_pair, "baseline", findings)
            _action_boundary_probes(candidate_pair, "candidate", findings)
        try:
            base_configs = {
                role: (artifact[1], artifact[2]) for role, artifact in zip(("pre", "post"), base_pair)
            }
            candidate_configs = {
                role: (artifact[1], artifact[2]) for role, artifact in zip(("pre", "post"), candidate_pair)
            }
            _guard_delta(baseline, candidate, base_configs, candidate_configs, findings)
        except (ValueError, OSError, TypeError, KeyError, RuntimeError, AttributeError) as exc:
            _add(
                findings,
                _GUARD,
                DiagnosticStatus.UNKNOWN,
                f"Guard retention could not be fully determined: {exc}",
            )
        try:
            compatible = _compatible(
                baseline, candidate, policies[0], policies[1], base_pair, candidate_pair, findings
            )
            if relative_contract:
                _relative_action_delta(
                    baseline,
                    candidate,
                    policies,
                    (base_pair, candidate_pair),
                    scale_expansion_ratio,
                    offset_shift_threshold,
                    findings,
                    envelope_abs_tol,
                )
            elif compatible:
                _scale_delta(base_pair, candidate_pair, scale_expansion_ratio, findings)
                _affine_delta(base_pair, candidate_pair, offset_shift_threshold, findings, envelope_abs_tol)
            else:
                _add(
                    findings,
                    _GAIN,
                    DiagnosticStatus.UNKNOWN,
                    "Incompatible contracts prevent a meaningful per-coordinate scale ratio.",
                )
        except (ValueError, TypeError, KeyError, RuntimeError, AttributeError) as exc:
            _add(
                findings, _COMPAT, DiagnosticStatus.CONFLICT, f"Native contracts could not be compared: {exc}"
            )
            _add(findings, _GAIN, DiagnosticStatus.UNKNOWN, "Action scale comparison is unavailable.")
    if not any(f.check_id == _GUARD for f in findings):
        _add(
            findings,
            _GUARD,
            DiagnosticStatus.PASS,
            "No baseline serialized normalization or inspected clipping requirement was removed.",
        )
    for check in (_OFFSET, _ENVELOPE):
        if not any(f.check_id == check for f in findings):
            _add(
                findings,
                check,
                DiagnosticStatus.UNKNOWN,
                "A complete compatible serialized action affine mapping was not available.",
                {"scope": "serialized_action_affine_mapping"},
            )
    return findings
