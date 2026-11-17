"""Deterministic, local processor lineage exports using existing dependencies only.

This is an artifact graph, not a second processor implementation. Inspected
native normalizers supply effective ``state_dict()`` values, including inline
override precedence. No policy, tokenizer, remote code or model weights execute.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.configs import FeatureType, NormalizationMode
from lerobot.processor import (
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from safetensors import SafetensorError

from .models import _json_value
from .preflight import (
    _MASKED_NORMALIZERS,
    _MODE_STATS,
    _legacy_declared_features,
    _local_file,
    _normalization_base,
    _normalizer_features,
    _read_tensors,
    _role,
    _step_class,
    _validate_step_entry,
    _verified_legacy_values,
)

_FORMAT = "flinch-processor-lineage-v1"
# Exports from earlier releases (lerobot-doctor, lerobot-preflight) remain readable;
# new exports always use _FORMAT.
_LEGACY_FORMATS = ("lerobot-preflight-processor-lineage-v1", "lerobot-doctor-processor-lineage-v1")
_MARKER = "%% " + _FORMAT
_BUFFER_BUDGET = 5 * 1024 * 1024


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(kind: str, *parts: Any) -> str:
    return kind + "_" + hashlib.sha256(_canonical(parts).encode()).hexdigest()[:16]


def _gain(
    state: dict[str, torch.Tensor],
    key: str,
    mode: NormalizationMode,
    inverse: bool,
    eps: float,
    dtype: str | None,
    *,
    legacy: bool = False,
) -> dict[str, Any]:
    """Describe the inspected native affine slope; never infer input dtype."""
    required = _MODE_STATS.get(mode, ())
    if not required:
        return {"max_forward_gain": 1.0, "max_inverse_gain": 1.0}
    if any(f"{key}.{stat}" not in state for stat in required):
        return {"max_forward_gain": None, "max_inverse_gain": None, "gain_status": "UNKNOWN"}
    if dtype is not None and dtype not in DeviceProcessorStep.DTYPE_MAPPING:
        raise ValueError(f"Unsupported floating execution dtype: {dtype}")
    evaluation_dtype = DeviceProcessorStep.DTYPE_MAPPING.get(dtype, torch.float32)
    values = {stat: state[f"{key}.{stat}"].to(evaluation_dtype) for stat in required}
    if mode is NormalizationMode.MEAN_STD:
        forward = (values["std"] + eps).reciprocal()
        backward = values["std"]
        formula = "std" if inverse else "1 / (std + eps)"
    else:
        lower, upper = required
        span = values[upper] - values[lower]
        denom = span + eps if legacy else torch.where(span == 0, torch.full_like(span, eps), span)
        forward, backward = 2 / denom, denom / 2
        formula = (
            ("(span + eps) / 2" if inverse else "2 / (span + eps)")
            if legacy
            else ("where(span == 0, eps, span) / 2" if inverse else "2 / where(span == 0, eps, span)")
        )
    return {
        "max_forward_gain": float(forward.abs().max()) if forward.numel() else None,
        "max_inverse_gain": float(backward.abs().max()) if backward.numel() else None,
        "active_max_gain": float((backward if inverse else forward).abs().max()),
        "gain_formula": formula,
        "gain_status": "PASS",
        "gain_evaluation_dtype": str(evaluation_dtype).removeprefix("torch."),
        "gain_dtype_conditional": dtype is None,
        "eps": eps,
    }


def _effective_dtype(requested: str | None, declared: str | None) -> str | None:
    """Native Device steps leave explicitly integer/bool tensors uncast."""
    if declared is not None and declared not in DeviceProcessorStep.DTYPE_MAPPING:
        return declared
    return requested or declared


def build_processor_graph(checkpoint_dir: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    """Build a stable property graph from local serialized processor contracts.

    Missing/uninspectable evidence is retained as UNKNOWN. Input dtypes that are
    not declared remain unknown; a float32 gain reference is explicitly marked
    conditional until an effective dtype is configured. Neural model bodies are
    never read: legacy graphs use verified tiny range artifacts when available.
    """
    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {root}")
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    unknown_steps: list[dict[str, Any]] = []
    artifacts: set[str] = set()

    def node(kind: str, identity: Any, label: str, **properties: Any) -> str:
        identifier = _identifier(kind, identity)
        nodes[identifier] = {
            "id": identifier,
            "type": kind,
            "label": label,
            "properties": _json_value(properties),
        }
        return identifier

    def edge(source: str, target: str, kind: str, **properties: Any) -> None:
        identifier = _identifier("edge", source, target, kind, properties)
        edges[identifier] = {
            "id": identifier,
            "source": source,
            "target": target,
            "type": kind,
            "properties": _json_value(properties),
        }

    def feature(identity: Any, key: str, declaration: dict[str, Any], **properties: Any) -> str:
        return node(
            "feature",
            identity,
            key,
            feature_key=key,
            shape=declaration.get("shape"),
            feature_type=declaration.get("type"),
            dtype=declaration.get("dtype"),
            **properties,
        )

    def read_json(path: Path) -> dict[str, Any]:
        path = _local_file(root, path.name)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path.name}")
        return value

    config: dict[str, Any] = {}
    if (root / "config.json").exists():
        config = read_json(root / "config.json")
        artifacts.add("config.json")
    inputs = config.get("input_features", {}) or {}
    outputs = config.get("output_features", {}) or {}
    # Historical configs persisted shapes rather than PolicyFeature objects.
    if not inputs and isinstance(config.get("input_shapes"), dict):
        inputs = {key: {"shape": shape} for key, shape in config["input_shapes"].items()}
    if not outputs and isinstance(config.get("output_shapes"), dict):
        outputs = {key: {"shape": shape} for key, shape in config["output_shapes"].items()}
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("Policy input_features and output_features must be objects")
    config_nodes: dict[str, str] = {}
    for side, declarations in (("input", inputs), ("output", outputs)):
        for key, declaration in sorted(declarations.items()):
            config_nodes[f"{side}:{key}"] = feature(
                ("config", side, key),
                key,
                declaration,
                origin=f"config.{side}_features",
                file_path="config.json",
            )

    pipelines = []
    for path in sorted(root.glob("*.json")):
        if path.name == "config.json":
            continue
        try:
            saved = read_json(path)
        except (OSError, ValueError) as exc:
            if "preprocessor" in path.name or "postprocessor" in path.name:
                errors.append({"file_path": path.name, "status": "UNKNOWN", "reason": str(exc)})
            continue
        if isinstance(saved.get("steps"), list):
            pipelines.append((path, saved))
            artifacts.add(path.name)
    pipelines.sort(key=lambda item: ({"pre": 0, "post": 1}.get(_role(*item) or "", 2), item[0].name))
    referenced_buffers: set[str] = set()

    def buffer(
        path: Path,
        state: dict[str, torch.Tensor],
        *,
        identity: Any = None,
        inline: bool = False,
        **properties: Any,
    ) -> str:
        relative = path.relative_to(root).as_posix()
        prior = nodes.get(_identifier("stat_buffer", identity or relative), {}).get("properties", {})
        properties["matched_keys"] = sorted(
            set(prior.get("matched_keys", [])) | set(properties.get("matched_keys", []))
        )
        return node(
            "stat_buffer",
            identity or relative,
            relative + (" inline stats" if inline else ""),
            file_path=relative,
            source_kind="inline" if inline else "safetensors",
            sha256=_sha(path),
            sha256_scope="processor_json" if inline else "whole_buffer_file",
            tensor_keys=sorted(state),
            tensors={
                key: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
                for key, value in sorted(state.items())
            },
            **properties,
        )

    for path, saved in pipelines:
        role = {"pre": "preprocessor", "post": "postprocessor"}.get(_role(path, saved) or "", "unclassified")
        declared = inputs if role == "preprocessor" else outputs
        current: dict[str, str] = {
            key: config_nodes[f"{'input' if role == 'preprocessor' else 'output'}:{key}"]
            for key in sorted(declared)
        }
        shapes: dict[str, dict[str, Any]] = {key: dict(value) for key, value in declared.items()}
        # Preprocessors may normalize training ACTION independently of their
        # observation declarations. This is an explicit output contract, not a
        # guess at a new observation feature.
        if role == "preprocessor" and "action" in outputs:
            current["action"] = config_nodes["output:action"]
            shapes["action"] = dict(outputs["action"])
        previous_step = None
        active_dtype: str | None = None
        uncertain_flow = False
        for index, entry in enumerate(saved["steps"]):
            try:
                _validate_step_entry(entry)
            except ValueError as exc:
                errors.append(
                    {"file_path": path.name, "status": "UNKNOWN", "reason": f"Invalid step {index}: {exc}"}
                )
                unknown_steps.append(
                    {"file_path": path.name, "step_index": index, "class_name": "invalid_step_declaration"}
                )
                uncertain_flow = True
                active_dtype = None
                shapes = {key: {**value, "shape": None, "dtype": None} for key, value in shapes.items()}
                continue
            entry_config = entry.get("config", {})
            cls = _step_class(entry)
            kind = entry.get("registry_name", entry.get("class", "undeclared"))
            named_class = cls
            if named_class is None and "registry_name" in entry:
                try:
                    named_class = ProcessorStepRegistry.get(kind)
                except KeyError:
                    pass
            class_name = named_class.__name__ if named_class is not None else str(kind).rsplit(".", 1)[-1]
            language_only = named_class in (TokenizerProcessorStep, NewLineTaskProcessorStep)
            if cls is None and not language_only:
                uncertain_flow = True
                active_dtype = None
                unknown_steps.append({"file_path": path.name, "step_index": index, "class_name": class_name})
                errors.append(
                    {
                        "file_path": path.name,
                        "status": "UNKNOWN",
                        "reason": f"Uninspected numeric feature flow at step {index}: {class_name}",
                    }
                )
                shapes = {key: {**value, "shape": None, "dtype": None} for key, value in shapes.items()}
            if cls is DeviceProcessorStep and entry_config.get("float_dtype") is not None:
                active_dtype = entry_config["float_dtype"]
            step_id = node(
                "processor_step",
                (path.name, index),
                f"{role}[{index}] {class_name}",
                file_path=path.name,
                pipeline_role=role,
                step_index=index,
                class_name=class_name,
                registry_name=entry.get("registry_name"),
                active_dtype=active_dtype,
                dtype_source="DeviceProcessorStep"
                if active_dtype is not None
                else "undeclared/input-dependent",
                sha256=_sha(path),
                step_config_sha256=hashlib.sha256(_canonical(entry).encode()).hexdigest(),
                inspected_native=cls is not None,
            )
            if previous_step is not None:
                edge(previous_step, step_id, "FLOWS_TO", pipeline_role=role, ordered=True)
            previous_step = step_id
            state: dict[str, torch.Tensor] = {}
            state_id = None
            state_read_failed = False
            if entry.get("state_file"):
                filename = entry["state_file"]
                referenced_buffers.add(filename)
                try:
                    state_path = _local_file(root, filename)
                    artifacts.add(filename)
                    if state_path.name == "model.safetensors" or state_path.name.startswith("model-"):
                        raise ValueError("Model weight bodies are outside processor-graph inspection scope")
                    if state_path.stat().st_size > _BUFFER_BUDGET:
                        raise ValueError(
                            "Referenced tensor artifact exceeds tiny-buffer budget; body not read"
                        )
                    state = _read_tensors(state_path)
                    state_id = buffer(
                        state_path, state, matched_keys=[], unbound_keys=sorted(state), orphan=False
                    )
                except (OSError, ValueError, RuntimeError, SafetensorError) as exc:
                    state_read_failed = True
                    errors.append({"file_path": str(filename), "status": "UNKNOWN", "reason": str(exc)})
                    state_id = node(
                        "stat_buffer",
                        filename,
                        str(filename),
                        file_path=filename,
                        status="UNKNOWN",
                        sha256=None,
                        matched_keys=[],
                        unbound_keys=[],
                        reason=str(exc),
                    )
            native = None
            base = _normalization_base(entry)
            if base is not None and base in (NormalizerProcessorStep, UnnormalizerProcessorStep):
                try:
                    native = base(**{**entry_config, "device": "cpu"})
                    native.load_state_dict(state)
                    state = native.state_dict()
                    if entry_config.get("stats"):
                        prior = state_id
                        state_id = buffer(
                            path,
                            state,
                            identity=(path.name, index, "inline"),
                            inline=True,
                            step_index=index,
                            matched_keys=[],
                            unbound_keys=sorted(state),
                            orphan=False,
                            inline_stats_sha256=hashlib.sha256(
                                _canonical(entry_config["stats"]).encode()
                            ).hexdigest(),
                        )
                        if prior is not None:
                            edge(
                                step_id,
                                prior,
                                "BINDS_STATS",
                                status="PASS",
                                active=False,
                                reason="Native explicit inline stats override saved tensor state",
                            )
                except (TypeError, ValueError, RuntimeError) as exc:
                    errors.append(
                        {"file_path": path.name, "status": "UNKNOWN", "reason": f"step {index}: {exc}"}
                    )
            contracts = (
                _normalizer_features(entry, state) if native is not None else entry_config.get("features", {})
            )
            if not current and not declared:
                for key, declaration in sorted(contracts.items()):
                    incoming = (
                        {**declaration, "shape": None, "dtype": None} if uncertain_flow else declaration
                    )
                    current[key] = feature(
                        (path.name, "declared_input", key),
                        key,
                        incoming,
                        origin="processor.feature_contract",
                        file_path=path.name,
                    )
                    shapes[key] = dict(incoming)
            matched: set[str] = set()
            for key, declaration in sorted(contracts.items()):
                if key not in current or shapes[key].get("shape") != declaration.get("shape"):
                    expected = feature(
                        (path.name, index, "expected", key),
                        key,
                        declaration,
                        origin="processor.feature_contract",
                        file_path=path.name,
                    )
                    actual = current.get(key, step_id)
                    edge(
                        actual,
                        expected,
                        "CONTRACT_MISMATCH",
                        status="UNKNOWN" if uncertain_flow else "CONFLICT",
                        step_index=index,
                        feature_key=key,
                        expected_shape=declaration.get("shape"),
                        actual_shape=shapes.get(key, {}).get("shape"),
                        reason="upstream_feature_flow_unverified"
                        if uncertain_flow
                        else ("feature_key_missing" if key not in current else "feature_shape_mismatch"),
                        available_feature_keys=sorted(current),
                        declared_feature_keys=sorted(contracts),
                        rename_map=entry_config.get("rename_map", {}),
                    )
                if native is None:
                    continue
                feature_type = FeatureType(declaration["type"])
                mode = native.norm_map.get(feature_type, NormalizationMode.IDENTITY)
                required = _MODE_STATS.get(mode, ())
                selected = native.normalize_observation_keys
                active = (feature_type is FeatureType.ACTION and key == "action") or (
                    feature_type is not FeatureType.ACTION and (selected is None or key in selected)
                )
                if not required or not active:
                    continue
                present = [f"{key}.{stat}" for stat in required if f"{key}.{stat}" in state]
                missing = [f"{key}.{stat}" for stat in required if f"{key}.{stat}" not in state]
                matched.update(present)
                if state_id is not None:
                    edge(
                        step_id,
                        state_id,
                        "BINDS_STATS",
                        status="PASS" if not missing else "FAIL",
                        active=True,
                        feature_key=key,
                        tensor_keys=present,
                        required_keys=[f"{key}.{s}" for s in required],
                        missing_keys=missing,
                        normalization_mode=mode.value,
                    )
                if missing:
                    # Native _apply_transform bypasses only if the entire
                    # feature namespace is absent; a partial namespace raises.
                    bypass = not state_read_failed and not any(name.startswith(key + ".") for name in state)
                    target = current.get(key) or feature(
                        (path.name, index, "absent", key),
                        key,
                        declaration,
                        origin="processor.feature_contract",
                    )
                    edge(
                        step_id,
                        target,
                        "UNBOUND_BYPASS" if bypass else "CONTRACT_MISMATCH",
                        status="UNKNOWN" if state_read_failed else "FAIL",
                        feature_key=key,
                        step_index=index,
                        normalization_mode=mode.value,
                        missing_keys=missing,
                        available_buffer_keys=sorted(state),
                        runtime_behavior="state file loading failed; runtime normalization unknown"
                        if state_read_failed
                        else (
                            "passes unchanged when present"
                            if bypass
                            else "native raises for incomplete stats"
                        ),
                        feature_declared_in_flow=key in current,
                    )
            if state_id is not None:
                properties = nodes[state_id]["properties"]
                properties["matched_keys"] = sorted(set(properties.get("matched_keys", [])) | matched)
                properties["unbound_keys"] = sorted(set(state) - set(properties["matched_keys"]))
            rename = entry_config.get("rename_map", {}) if cls is RenameObservationsProcessorStep else {}
            renamed = (
                cast(
                    dict[str, str],
                    RenameObservationsProcessorStep(rename_map=rename).observation(
                        {key: key for key in current if key != "action"}
                    ),
                )
                if rename
                else {}
            )
            new_current: dict[str, str] = {}
            new_shapes: dict[str, dict[str, Any]] = {}
            entering_state = current.get("observation.state")
            reverse_names = {old: new for new, old in renamed.items()}
            for key, source in sorted(current.items()):
                output_key = reverse_names.get(key, key)
                declaration = dict(shapes[key])
                declaration["dtype"] = _effective_dtype(active_dtype, declaration.get("dtype"))
                output = feature(
                    (path.name, index, "output", output_key),
                    output_key,
                    declaration,
                    origin="processor.output",
                    file_path=path.name,
                    step_index=index,
                    pipeline_role=role,
                )
                props: dict[str, Any] = {
                    "feature_key": key,
                    "output_feature_key": output_key,
                    "effective_dtype": declaration.get("dtype"),
                    "step_index": index,
                    "rename_map": rename,
                    "normalization_mode": "IDENTITY",
                    "contract_only": True,
                    "native_class_inspected": cls is not None or language_only,
                    "upstream_flow_verified": not uncertain_flow,
                    "shape_evolution_verified": not uncertain_flow,
                }
                if native is not None and key in contracts:
                    feature_type = FeatureType(contracts[key]["type"])
                    mode = native.norm_map.get(feature_type, NormalizationMode.IDENTITY)
                    selected = native.normalize_observation_keys
                    if (feature_type is FeatureType.ACTION and key != "action") or (
                        feature_type is not FeatureType.ACTION
                        and selected is not None
                        and key not in selected
                    ):
                        mode = NormalizationMode.IDENTITY
                    props.update(
                        normalization_mode=mode.value,
                        direction="inverse" if base is UnnormalizerProcessorStep else "forward",
                    )
                    if state_read_failed:
                        props.update(
                            gain_status="UNKNOWN",
                            max_forward_gain=None,
                            max_inverse_gain=None,
                            gain_uncertainty="Referenced state file loading failed",
                        )
                    elif kind in _MASKED_NORMALIZERS:
                        props.update(
                            gain_status="UNKNOWN",
                            max_forward_gain=None,
                            max_inverse_gain=None,
                            gain_uncertainty="Masked normalization is not a single unmasked affine map",
                        )
                    else:
                        try:
                            props.update(
                                _gain(
                                    state,
                                    key,
                                    mode,
                                    base is UnnormalizerProcessorStep,
                                    native.eps,
                                    props["effective_dtype"],
                                )
                            )
                        except (RuntimeError, ValueError, TypeError) as exc:
                            props.update(
                                gain_status="UNKNOWN",
                                max_forward_gain=None,
                                max_inverse_gain=None,
                                gain_uncertainty=str(exc),
                            )
                            errors.append(
                                {
                                    "file_path": path.name,
                                    "status": "UNKNOWN",
                                    "reason": f"step {index} {key} gain could not be evaluated: {exc}",
                                }
                            )
                edge(source, step_id, "FLOWS_TO", feature_key=key, effective_dtype=props["effective_dtype"])
                edge(step_id, output, "TRANSFORMS", **props)
                new_current[output_key], new_shapes[output_key] = output, declaration
            current, shapes = new_current, new_shapes
            if cls is RelativeActionsProcessorStep and entry_config.get("enabled", False):
                from .temporal import _resolve_metadata

                metadata, problem = _resolve_metadata(root, metadata_path)
                target = entering_state or feature(
                    (path.name, index, "undeclared_anchor_input"),
                    "observation.state",
                    {"type": "STATE", "shape": None},
                    origin="undeclared.anchor_input",
                )
                expected_anchor = (metadata or {}).get("expected_anchor_index")
                status = (
                    "UNKNOWN"
                    if metadata is None or uncertain_flow or entering_state is None
                    else "PASS"
                    if expected_anchor == 0
                    else "FAIL"
                )
                edge(
                    target,
                    step_id,
                    "TEMPORAL_ANCHOR",
                    status=status,
                    native_collapse_index=0,
                    native_expression="state[:, 0]",
                    applies_to="3D stacked observation.state only",
                    observation_delta_indices=(metadata or {}).get(
                        "effective_state_delta_indices", config.get("observation_delta_indices")
                    ),
                    generic_observation_delta_indices=config.get("observation_delta_indices"),
                    expected_anchor_index=expected_anchor,
                    chronology=metadata,
                    upstream_flow_verified=not uncertain_flow,
                    uncertainty=problem.to_dict() if problem is not None else None,
                )

    # Preserve orphan step artifacts as evidence without feeding them to any
    # active processor. Never scan full model weights for orphan statistics.
    for path in sorted(root.glob("*.safetensors")):
        if path.name in referenced_buffers or path.name in {
            "model.safetensors",
            "legacy_normalization.safetensors",
        }:
            continue
        if not (
            path.name.startswith(("policy_preprocessor_step_", "policy_postprocessor_step_"))
            or "processor_step_" in path.name
        ):
            continue  # An unreferenced neural shard is not a processor artifact.
        try:
            local = _local_file(root, path.name)
            if local.stat().st_size > _BUFFER_BUDGET:
                continue
            artifacts.add(path.name)
            orphan_state = _read_tensors(local)
            buffer(local, orphan_state, orphan=True, matched_keys=[], unbound_keys=sorted(orphan_state))
        except (OSError, ValueError, RuntimeError, SafetensorError) as exc:
            errors.append({"file_path": path.name, "status": "UNKNOWN", "reason": str(exc)})
    if not pipelines and (root / "model.safetensors.header.json").is_file():
        # Cached HTTP byte-range values are independently checked against the
        # original header and payload hashes before numerical graph annotation.
        try:
            state_path, legacy, metadata, header = _verified_legacy_values(root)
            artifacts.update((state_path.name, "legacy_normalization.json", "model.safetensors.header.json"))
            state_id = buffer(
                state_path,
                legacy,
                matched_keys=[],
                unbound_keys=sorted(legacy),
                legacy_embedded=True,
                original_header_sha256_prefix=header["header_sha256_prefix"],
            )
            legacy_matched_keys: list[str] = []
            covered_legacy_features: set[tuple[str, str]] = set()
            for group, key, declaration, mode in _legacy_declared_features(config):
                inverse = group == "unnormalize_outputs"
                covered_legacy_features.add(("output" if inverse else "input", key))
                aliases = {
                    f"{key}.{stat}": f"{group}.buffer_{key.replace('.', '_')}.{stat}"
                    for stat in _MODE_STATS.get(mode, ())
                }
                canonical = {new: legacy[old] for new, old in aliases.items() if old in legacy}
                legacy_matched_keys.extend(aliases.values())
                step_id = node(
                    "processor_step",
                    ("legacy", group, key),
                    f"legacy {group}: {key}",
                    file_path="model.safetensors",
                    pipeline_role="postprocessor" if inverse else "preprocessor",
                    step_index=None,
                    class_name="Unnormalize" if inverse else "Normalize",
                    active_dtype="float32",
                    dtype_source="legacy float32 buffer semantics; module casts unknown",
                    sha256=None,
                    legacy_embedded=True,
                    sha256_scope="model body not downloaded",
                    original_header_sha256_prefix=header["header_sha256_prefix"],
                )
                source = config_nodes.get(f"{'output' if inverse else 'input'}:{key}") or feature(
                    ("legacy", key),
                    key,
                    {"type": declaration.type.value, "shape": list(declaration.shape)},
                    origin="legacy.config",
                )
                edge(source, step_id, "FLOWS_TO", feature_key=key)
                edge(
                    step_id,
                    state_id,
                    "BINDS_STATS",
                    feature_key=key,
                    tensor_keys=[old for old in aliases.values() if old in legacy],
                    normalization_mode=mode.value,
                )
                target = feature(
                    ("legacy", group, "output", key),
                    key,
                    {"type": declaration.type.value, "shape": list(declaration.shape), "dtype": "float32"},
                    origin="legacy.normalization_output",
                )
                gain = _gain(canonical, key, mode, inverse, 1e-8, "float32", legacy=True)
                if inverse and mode not in (NormalizationMode.MEAN_STD, NormalizationMode.IDENTITY):
                    gain = {
                        "gain_status": "UNKNOWN",
                        "max_forward_gain": None,
                        "max_inverse_gain": None,
                        "gain_uncertainty": "Historical inverse range arithmetic is outside inspected native modern semantics",
                    }
                    errors.append(
                        {"file_path": "config.json", "status": "UNKNOWN", "reason": gain["gain_uncertainty"]}
                    )
                edge(
                    step_id,
                    target,
                    "TRANSFORMS",
                    feature_key=key,
                    normalization_mode=mode.value,
                    effective_dtype="float32",
                    legacy_execution_configuration_known=False,
                    direction="inverse" if inverse else "forward",
                    **gain,
                )
            nodes[state_id]["properties"]["matched_keys"] = sorted(set(legacy_matched_keys) & set(legacy))
            nodes[state_id]["properties"]["unbound_keys"] = sorted(set(legacy) - set(legacy_matched_keys))
            uncovered = sorted(
                f"{side}:{key}"
                for side, declarations in (("input", inputs), ("output", outputs))
                for key in declarations
                if (side, key) not in covered_legacy_features
            )
            if uncovered:
                errors.append(
                    {
                        "file_path": "config.json",
                        "status": "UNKNOWN",
                        "reason": "Legacy graph has declared features outside verified state/action normalization scope: "
                        + ", ".join(uncovered),
                    }
                )
        except (OSError, ValueError, KeyError, RuntimeError, SafetensorError) as exc:
            errors.append({"file_path": "legacy_normalization.json", "status": "UNKNOWN", "reason": str(exc)})
    if not pipelines and not any(item["type"] == "processor_step" for item in nodes.values()):
        errors.append(
            {
                "file_path": "config.json",
                "status": "UNKNOWN",
                "reason": "No inspectable modern processor or verified legacy normalization contract",
            }
        )
    return {
        "graph_version": 1,
        "format": _FORMAT,
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": sorted(edges.values(), key=lambda item: item["id"]),
        "summary": {
            "checkpoint_dir": str(root),
            "node_counts": dict(sorted(Counter(n["type"] for n in nodes.values()).items())),
            "edge_counts": dict(sorted(Counter(e["type"] for e in edges.values()).items())),
            "artifact_files": sorted(artifacts),
            "coverage_status": "UNKNOWN" if errors else "PASS",
            "errors": errors,
            "unknown_step_boundaries": unknown_steps,
            "has_contract_failures": any(
                e["properties"].get("status") in {"FAIL", "CONFLICT"} for e in edges.values()
            ),
            "diagnostic_status_counts": dict(
                sorted(
                    Counter(
                        e["properties"]["status"] for e in edges.values() if "status" in e["properties"]
                    ).items()
                )
            ),
            "execution_scope": "static artifact lineage; no policy/tokenizer execution",
        },
    }


def _label(value: Any) -> str:
    return html.escape(str(value), quote=True).replace("|", "&#124;").replace("`", "&#96;").replace("\n", " ")


def processor_graph_to_mermaid(graph: dict[str, Any]) -> str:
    """Render a dependency-free Mermaid graph; JSON preserves all properties."""
    lines = ["flowchart LR", _MARKER]
    for item in graph["nodes"]:
        props = item["properties"]
        label = item["label"]
        if item["type"] == "feature":
            label += f" {props.get('shape')} {props.get('dtype') or 'dtype undeclared'}"
        elif item["type"] == "processor_step":
            digest = props.get("sha256") or props.get("original_header_sha256_prefix")
            scope = "sha256" if props.get("sha256") else "header sha256"
            label += (
                f"; {props.get('active_dtype') or 'input dtype'}; {scope} {str(digest or 'unavailable')[:12]}"
            )
        else:
            label += f"; sha256 {str(props.get('sha256') or '')[:12]}"
            unbound = props.get("unbound_keys", [])
            if unbound:
                label += "; unbound: " + ", ".join(unbound)
        lines.append(f'  {item["id"]}["{_label(label)}"]')
    failure_edges = []
    for index, item in enumerate(graph["edges"]):
        props = item["properties"]
        label = item["type"]
        if props.get("feature_key"):
            label += ": " + props["feature_key"]
        if props.get("normalization_mode"):
            label += " " + props["normalization_mode"]
        if props.get("active_max_gain") is not None:
            value = props["active_max_gain"]
            label += f" gain={value:.8g}x" if isinstance(value, (int, float)) else f" gain={value}"
        if props.get("tensor_keys"):
            label += " " + ", ".join(props["tensor_keys"])
        if props.get("missing_keys"):
            label += "; missing " + ", ".join(props["missing_keys"])
        if item["type"] == "TEMPORAL_ANCHOR":
            label += f" state[:, 0]; expected={props.get('expected_anchor_index')}; deltas={props.get('observation_delta_indices')}"
        if props.get("status") in {"FAIL", "CONFLICT"}:
            failure_edges.append(str(index))
        arrow = "-.->" if item["type"] in {"UNBOUND_BYPASS", "CONTRACT_MISMATCH"} else "-->"
        lines.append(f'  {item["source"]} {arrow}|"{_label(label)}"| {item["target"]}')
    if failure_edges:
        lines.append("  linkStyle " + ",".join(failure_edges) + " stroke:#c62828,stroke-width:2px")
    return "\n".join(lines) + "\n"


def write_processor_graph(graph: dict[str, Any], output_path: Path) -> Path:
    """Export JSON, raw Mermaid (.mmd), or fenced Markdown without overwrites."""
    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix not in {".json", ".mmd", ".md"}:
        raise ValueError("Graph output must end in .json, .mmd or .md")
    if path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite a symlink: {path}")
    root = Path(graph["summary"]["checkpoint_dir"])
    if path.resolve() in {
        (root / artifact).resolve() for artifact in graph["summary"].get("artifact_files", [])
    }:
        raise FileExistsError(f"Refusing to overwrite an audited artifact: {path}")
    previously_exists = path.exists()
    if previously_exists:
        if not path.is_file():
            raise FileExistsError(f"Graph output exists and is not a file: {path}")
        previous = path.read_text(encoding="utf-8")
        try:
            recognized = (
                json.loads(previous).get("format") in {_FORMAT, *_LEGACY_FORMATS}
                if suffix == ".json"
                else (
                    any(
                        previous.startswith("flowchart LR\n%% " + marker)
                        for marker in (_FORMAT, *_LEGACY_FORMATS)
                    )
                    if suffix == ".mmd"
                    else any(
                        previous.startswith("<!-- " + marker + " -->")
                        for marker in (_FORMAT, *_LEGACY_FORMATS)
                    )
                )
            )
        except (ValueError, AttributeError):
            recognized = False
        if not recognized:
            raise FileExistsError(f"Refusing to overwrite an unrelated artifact: {path}")
    if suffix == ".json":
        content = json.dumps(_json_value(graph), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    else:
        content = processor_graph_to_mermaid(graph)
        if suffix == ".md":
            content = "<!-- " + _FORMAT + " -->\n\n```mermaid\n" + content + "```\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if previously_exists else "x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return path
