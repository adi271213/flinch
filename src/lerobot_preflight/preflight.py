"""Static, local-only validation of serialized LeRobot processor artifacts.

No recorded observations or neural policy are loaded. Bounded synthetic probes
exercise native normalization steps on CPU; no simulation or motor command runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    create_transition,
)
from safetensors import safe_open

from .models import DiagnosticFinding, DiagnosticStatus, PreflightReport, Provenance, Severity

_NORMALIZERS = (NormalizerProcessorStep, UnnormalizerProcessorStep)
_SAFE_CLASSES = (
    *_NORMALIZERS,
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
)
_CONFIG_NAMES = {
    "policy_preprocessor.json", "policy_postprocessor.json",
    "preprocessor_config.json", "postprocessor_config.json",
}
_SEMANTICS = ("joint_angle_units", "action_coordinate_frame", "temporal_observation_ordering")
_LEGACY_BUFFER_PREFIXES = (
    "normalize_inputs.",
    "unnormalize_outputs.",
)
_MASKED_NORMALIZERS = {
    "molmoact2_masked_normalizer": NormalizerProcessorStep,
    "molmoact2_masked_unnormalizer": UnnormalizerProcessorStep,
}
_MODE_STATS = {
    NormalizationMode.MEAN_STD: ("mean", "std"),
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}


def _normalizer_features(entry: dict[str, Any], state: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Return the native feature coverage, including its implicit ACTION path.

    Native action transforms consult ``norm_map[ACTION]`` independently of
    ``features``. Buffer-derived shapes below size diagnostic probes only; they
    do not establish a policy action width or any physical joint semantics.
    """
    config = entry.get("config", {})
    features = deepcopy(config.get("features", {}))
    mode = NormalizationMode(config.get("norm_map", {}).get(FeatureType.ACTION.value, "IDENTITY"))
    required = _MODE_STATS.get(mode, ())
    if required and "action" in features and features["action"].get("type") != FeatureType.ACTION.value:
        features["action"]["_declared_type"] = features["action"].get("type")
        features["action"]["type"] = FeatureType.ACTION.value
    if "action" not in features and required:
        tensors = [state[f"action.{name}"] for name in required if f"action.{name}" in state]
        if tensors:
            shape = torch.broadcast_shapes(*(tensor.shape for tensor in tensors))
            while len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            features["action"] = {"type": FeatureType.ACTION.value, "shape": list(shape),
                                  "_shape_source": "saved_stat_broadcast", "_declared": False}
    return features


def _native_normalization_probe(
    entries: list[dict[str, Any]], normalizers: dict[int, Any],
    feature_key: str, probes: torch.Tensor,
) -> list[dict[str, Any]]:
    """Trace a numeric normalization subpath through real native pipeline steps.

    No checkpoint-specific constructors execute: only CPU Device steps and
    isolated copies of already-inspected native normalizers are included. This
    is deliberately a normalization-subpath probe, not full pipeline coverage.
    Original serialized indices are retained in every snapshot.
    """
    steps, indices, feature_keys = [], [], []
    current_key = feature_key
    barrier: ValueError | None = None
    language_only = {"tokenizer_processor", "smolvla_new_line_processor", "pi0_new_line_processor",
                     "pi05_prepare_state_tokenizer_processor_step",
                     "lerobot.processor.tokenizer_processor.TokenizerProcessorStep",
                     "lerobot.processor.newline_task_processor.NewLineTaskProcessorStep",
                     "lerobot.policies.pi0.processor_pi0.Pi0NewLineProcessor",
                     "lerobot.policies.pi05.processor_pi05.Pi05PrepareStateTokenizerProcessorStep"}
    for index, entry in enumerate(entries):
        cls = _step_class(entry)
        kind = entry.get("registry_name", entry.get("class"))
        input_key = current_key
        if kind in _MASKED_NORMALIZERS:
            barrier = ValueError(f"Masked native normalization probe is unavailable at step {index}")
            break
        if cls is DeviceProcessorStep:
            step = DeviceProcessorStep(**{**entry.get("config", {}), "device": "cpu"})
        elif cls is RenameObservationsProcessorStep:
            if feature_key == "action":
                # Observation-only renames cannot change the independent
                # transition ACTION tensor. Its probe intentionally has no images/state.
                continue
            step = RenameObservationsProcessorStep(**entry.get("config", {}))
            if current_key != "action":
                current_key = next(iter(step.observation({current_key: None})))
        elif index in normalizers:
            candidate = normalizers[index]
            if current_key == "action":
                mode = candidate.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)
            else:
                feature = candidate.features.get(current_key)
                if feature is None or (candidate.normalize_observation_keys is not None
                                       and current_key not in candidate.normalize_observation_keys):
                    continue
                mode = candidate.norm_map.get(feature.type, NormalizationMode.IDENTITY)
            if mode is NormalizationMode.IDENTITY:
                continue
            if any(f"{current_key}.{name}" not in candidate.state_dict() for name in _MODE_STATS.get(mode, ())):
                barrier = ValueError(f"Active normalization probe buffers are missing at step {index}")
                break
            step = deepcopy(candidate)
            step.to(device="cpu")
        else:
            if cls in (AbsoluteActionsProcessorStep, RelativeActionsProcessorStep):
                if feature_key == "action" and entry.get("config", {}).get("enabled", True):
                    barrier = ValueError(f"State-dependent action conversion prevents normalization probe coverage at step {index}")
                    break
            elif cls in _NORMALIZERS:
                if any(key.startswith(current_key + ".") for key in _active_buffer_keys(entry)):
                    barrier = ValueError(f"Active native normalization is unavailable at step {index}")
                    break
            elif cls not in (IdentityProcessorStep, AddBatchDimensionProcessorStep) and kind not in language_only:
                barrier = ValueError(f"Uninspected numeric step prevents normalization probe coverage at step {index}: {kind}")
                break
            continue
        steps.append(step)
        indices.append(index)
        feature_keys.append((input_key, current_key))
    transition = (create_transition(action=probes.clone()) if feature_key == "action" else
                  create_transition(observation={feature_key: probes.clone()}))
    pipeline = PolicyProcessorPipeline(steps=steps, to_transition=lambda data: data, to_output=lambda data: data)
    snapshots = []
    previous = probes.detach().cpu().clone()
    iterator = iter(pipeline.step_through(transition))
    next(iterator)
    with torch.inference_mode():
        for index, step, (input_key, output_key) in zip(indices, steps, feature_keys, strict=True):
            current = next(iterator)
            output = (current["action"] if feature_key == "action" else current["observation"][output_key])
            copied = output.detach().cpu().clone()
            snapshots.append({"step_index": index, "step_class": type(step).__name__,
                              "feature_key": output_key, "input_feature_key": input_key,
                              "source_feature_key": feature_key, "input": previous, "output": copied,
                              "input_dtype": str(previous.dtype).removeprefix("torch."),
                              "output_dtype": str(copied.dtype).removeprefix("torch."),
                              "probe_scope": "native_normalization_subpath"})
            previous = copied
            if not torch.isfinite(copied).all():
                return snapshots  # Preserve proven prefix failure before later barriers/exceptions.
    if barrier is not None:
        raise barrier
    return snapshots


def _normalizer_boundary_probes(step: Any, key: str) -> torch.Tensor:
    """Use saved feature boundaries without assigning units or hardware limits."""
    feature = step.features.get(key)
    if feature is None:
        raise ValueError(f"The {key} feature has no declared probe shape")
    if key == "action" and isinstance(step, UnnormalizerProcessorStep):
        points = torch.tensor([-3., -1., 0., 1., 3.], dtype=torch.float32)
        return points.reshape((-1,) + (1,) * len(feature.shape)).expand((-1,) + feature.shape).clone()
    mode = step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
    state = step.state_dict()
    required = _MODE_STATS.get(mode, ())
    if not required or any(f"{key}.{name}" not in state for name in required):
        raise ValueError(f"The {key} feature has incomplete active normalization statistics")
    values = {}
    for name in required:
        tensor = state[f"{key}.{name}"]
        while tensor.ndim > len(feature.shape) and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        values[name] = torch.broadcast_to(tensor, feature.shape).float()
    if mode is NormalizationMode.MEAN_STD:
        points = [values["mean"] - values["std"], values["mean"], values["mean"] + values["std"]]
    else:
        lower, upper = (values[name] for name in required)
        points = [lower, (lower + upper) / 2, upper]
    return torch.stack(points)


def _screen_composed_forward_gain(
    root: Path, path: Path, config: dict[str, Any], normalizers: dict[int, Any], key: str,
    snapshots: list[dict[str, Any]], threshold: float, report: PreflightReport,
) -> None:
    """Compose native typed affine slopes after a complete finite subpath probe."""
    gain: torch.Tensor | None = None
    contributors: list[dict[str, Any]] = []
    forward_count = 0
    for snapshot in snapshots:
        index = snapshot["step_index"]
        step = normalizers.get(index)
        if step is None:
            continue
        stage_key = snapshot["feature_key"]
        kind = FeatureType.ACTION if stage_key == "action" else step.features[stage_key].type
        mode = step.norm_map.get(kind, NormalizationMode.IDENTITY)
        required = _MODE_STATS.get(mode, ())
        if not required:
            continue
        dtype = DeviceProcessorStep.DTYPE_MAPPING[snapshot["output_dtype"]]
        state = step.state_dict()
        values = {name: state[f"{stage_key}.{name}"].to(dtype) for name in required}
        inverse = isinstance(step, UnnormalizerProcessorStep)
        forward_count += not inverse
        if mode is NormalizationMode.MEAN_STD:
            factor = values["std"].double() if inverse else (values["std"] + step.eps).double().reciprocal()
            stat = "std"
        else:
            low, high = required
            span = values[high] - values[low]
            denom = torch.where(span == 0, torch.full_like(span, step.eps), span)
            factor = denom.double() / 2 if inverse else 2 / denom.double()
            stat = low
        gain = factor if gain is None else gain * factor
        entry = config["steps"][index]
        source = path if entry.get("config", {}).get("stats") or "state_file" not in entry else _local_file(root, entry["state_file"])
        contributors.append({"step_index": index, "dtype": snapshot["output_dtype"],
                             "feature_key": stage_key,
                             "mode": mode.value, "direction": "inverse" if inverse else "forward",
                             "provenance": _provenance(source, type(step).__name__, f"{stage_key}.{stat}").to_dict()})
    if forward_count < 2 or gain is None:
        return
    mask = gain.abs() > 1 / threshold
    if not bool(mask.any()):
        return
    final = contributors[-1]
    index = final["step_index"]
    step = normalizers[index]
    final_key = final["feature_key"]
    final_mode = step.norm_map.get(FeatureType.ACTION if final_key == "action" else step.features[final_key].type)
    stat = "std" if final_mode is NormalizationMode.MEAN_STD else _MODE_STATS[final_mode][0]
    saved = torch.broadcast_to(step.state_dict()[f"{final_key}.{stat}"], gain.shape)
    indices = mask.reshape(-1).nonzero().flatten().tolist()
    source = Path(final["provenance"]["file_path"])
    _add(report, "PATHOLOGICAL_STD_GAIN", DiagnosticStatus.FAIL,
         f"Composed forward normalization for {key} has gain up to {float(gain.abs().max()):.8g}x.",
         {"comparison_scope": "composed_forward_normalization", "step_index": index, "feature_key": final_key,
          "source_feature_key": key,
          "normalization_mode": "COMPOSED", "std_warn_threshold": threshold,
          "gain_threshold": 1 / threshold, "composed_gains": gain[mask].tolist(),
          "effective_gains_with_eps": gain[mask].tolist(), "contributing_stages": contributors,
          "contributing_step_indices": [part["step_index"] for part in contributors],
          "saved_values": saved[mask].tolist(), "tensor_slice": saved[mask].tolist(),
          "probe_scope": "native_normalization_subpath", "native_boundary_probes_finite": True,
          "gain_formula": "product of native stage slopes using each stage's effective runtime dtype",
          "indexing": "flattened row-major normalization-buffer coordinates"},
         _provenance(source, type(step).__name__, f"{final_key}.{stat}", indices))


def _screen_native_normalization_chain(
    root: Path, path: Path, config: dict[str, Any], report: PreflightReport, threshold: float = 1e-4,
) -> None:
    """Detect non-finite intermediate native normalization arithmetic."""
    normalizers: dict[int, Any] = {}
    for index, entry in enumerate(config["steps"]):
        cls = _normalization_base(entry)
        if cls not in _NORMALIZERS:
            continue
        try:
            step = cls(**{**entry.get("config", {}), "device": "cpu"})
            if "state_file" in entry:
                step.load_state_dict(_read_tensors(_local_file(root, entry["state_file"])))
            for key, declaration in _normalizer_features(entry, step.state_dict()).items():
                if key not in step.features or step.features[key].type != FeatureType(declaration["type"]):
                    step.features[key] = PolicyFeature(FeatureType(declaration["type"]), tuple(declaration["shape"]))
            normalizers[index] = step
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
            continue  # Existing static contract findings retain unavailable coverage.
    seeds: dict[str, tuple[int, Any]] = {}
    for index, step in normalizers.items():
        for key, feature in step.features.items():
            if feature.type not in (FeatureType.STATE, FeatureType.ENV, FeatureType.ACTION):
                continue
            if key != "action" and step.normalize_observation_keys is not None and key not in step.normalize_observation_keys:
                continue
            if step.norm_map.get(feature.type, NormalizationMode.IDENTITY) in _MODE_STATS:
                seeds.setdefault(key, (index, step))
    for key, (_, first_step) in seeds.items():
        try:
            probes = _normalizer_boundary_probes(first_step, key)
            snapshots = _native_normalization_probe(config["steps"], normalizers, key, probes)
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            _add(report, "NORMALIZATION_CHAIN_PROBE", DiagnosticStatus.UNKNOWN,
                 f"The complete native normalization subpath was not probed: {exc}",
                 {"feature_key": key, "required_coverage": True,
                  "probe_scope": "native_normalization_subpath", "constructor_scope": "inspected_native_only",
                  "reason": str(exc)},
                 _provenance(path))
            if (_role(path, config) == "pre" and sum(
                isinstance(step, NormalizerProcessorStep) for step in normalizers.values()
            ) > 1):
                _add(report, "PATHOLOGICAL_STD_GAIN", DiagnosticStatus.UNKNOWN,
                     "Composed forward gain could not be established across the complete numeric subpath.",
                     {"feature_key": key, "comparison_scope": "composed_forward_normalization",
                      "required_coverage": True, "reason": str(exc)}, _provenance(path))
            continue
        if _role(path, config) == "pre" and all(torch.isfinite(snapshot["output"]).all() for snapshot in snapshots):
            _screen_composed_forward_gain(root, path, config, normalizers, key, snapshots, threshold, report)
        for snapshot in snapshots:
            output = snapshot["output"]
            nonfinite = ~torch.isfinite(output)
            if not nonfinite.any():
                continue
            index = snapshot["step_index"]
            stage_key = snapshot["feature_key"]
            entry = config["steps"][index]
            source = path if entry.get("config", {}).get("stats") or "state_file" not in entry else _local_file(root, entry["state_file"])
            step = normalizers.get(index)
            feature = first_step.features[key]
            dimensions = nonfinite.reshape((-1, math.prod(feature.shape))).any(dim=0).nonzero().flatten().tolist()
            stage_type = (FeatureType.ACTION if stage_key == "action" else
                          step.features.get(stage_key, feature).type) if step is not None else None
            mode = step.norm_map.get(stage_type, NormalizationMode.IDENTITY) if step is not None else None
            stat_names = _MODE_STATS.get(mode, ())
            tensor_key = f"{stage_key}.{'std' if mode is NormalizationMode.MEAN_STD else stat_names[0]}" if stat_names else stage_key
            entered = torch.broadcast_to(snapshot["input"], output.shape)
            _add(report, "NORMALIZATION_CHAIN_NON_FINITE", DiagnosticStatus.FAIL,
                 f"Native normalization boundary probes produce NaN/Inf at step {index}, {key}.",
                 {"step_index": index, "step_class": snapshot["step_class"], "feature_key": stage_key,
                  "source_feature_key": key,
                  "pipeline_role": "preprocessor" if _role(path, config) == "pre" else "postprocessor",
                  "input_dtype": snapshot["input_dtype"], "output_dtype": snapshot["output_dtype"],
                  "probe_scope": "native_normalization_subpath", "observed_policy_execution": False,
                  "probe_input_values": probes.tolist(), "probe_coordinates": nonfinite.nonzero().tolist(),
                  "input_values": entered[nonfinite].tolist(), "output_values": output[nonfinite].tolist(),
                  "input_tensor_shape": list(snapshot["input"].shape), "output_tensor_shape": list(output.shape),
                  "tensor_slice": output[nonfinite].tolist(), "execution_device": "cpu",
                  "source_stat_values": {name: value.tolist() for name, value in step.state_dict().items()
                                         if name.startswith(stage_key + ".")} if step is not None else {}},
                 _provenance(source, snapshot["step_class"], tensor_key, dimensions))
            if output.dtype is torch.float16 and not any(
                finding.check_id == "FP16_NON_FINITE_RISK" and finding.status is DiagnosticStatus.FAIL
                and finding.details.get("step_index") == index and finding.details.get("feature_key") == stage_key
                and finding.provenance is not None and finding.provenance.file_path == str(source)
                for finding in report.findings
            ):
                finding = deepcopy(report.findings[-1])
                finding.check_id = "FP16_NON_FINITE_RISK"
                finding.details.update({"screened_dtype": "float16", "risk_reasons": ["native_intermediate_arithmetic_nonfinite"],
                                        "configured_runtime_nonfinite_observed": True,
                                        "normalization_implementation": "modern native processor probe"})
                report.findings.append(finding)
            break  # First non-finite stage retains causal attribution.


def _screen_paired_relative_chain(root: Path, report: PreflightReport, threshold: float) -> None:
    """Complete only action probes that need the native relative-state cache.

    The privilege gate's inspected native pair shares a Relative/Absolute cache;
    an isolated action-only projection cannot supply that state. Reuse that
    proof rather than treating missing projection coverage as a runtime fault.
    Other barriers, missing buffers, and observation coverage remain unchanged.
    """
    pending = [finding for finding in report.findings
               if finding.check_id == "NORMALIZATION_CHAIN_PROBE"
               and finding.status is DiagnosticStatus.UNKNOWN
               and finding.details.get("feature_key") == "action"
               and finding.details.get("reason", "").startswith("State-dependent action conversion")]
    if not pending:
        return
    # Late import: privilege_delta uses this module's artifact readers. Neither
    # path imports or constructs a neural policy or a tokenizer.
    from .privilege_delta import (
        _policy,
        _relative_action_mapping,
        _relative_native_probes,
        _static_pair,
        _step_provenance,
    )

    try:
        policy = _policy(root)
        pair = _static_pair(root, policy)
        mapping = _relative_action_mapping(root, policy, pair)
        evidence: list[DiagnosticFinding] = []
        state_points = mapping["state_probe_points"]
        if state_points is None:
            state_points = torch.tensor([-1., 0., 1.]).reshape(3, 1).expand(3, mapping["state_width"])
        outputs = _relative_native_probes(mapping, "checkpoint", evidence, state_points)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
        return
    for finding in evidence:
        finding.check_id = "NORMALIZATION_CHAIN_NON_FINITE" if finding.status is DiagnosticStatus.FAIL else "NORMALIZATION_CHAIN_PROBE"
        report.findings.append(finding)
    if outputs is None:
        return
    paired_paths = {str(path) for _, path, _ in pair}
    for finding in pending:
        if finding.provenance is None or finding.provenance.file_path not in paired_paths:
            continue
        finding.status = DiagnosticStatus.PASS
        finding.severity = Severity.INFO
        finding.summary = "Native paired relative/absolute action probes completed with finite intermediates."
        finding.details.update({"required_coverage": False, "probe_scope": "paired_native_relative_action_chain",
                                "resolved_temporal_anchor_index": mapping["metadata"]["expected_anchor_index"],
                                "relative_action_mask": mapping["mask"].bool().tolist(),
                                "contributing_stages": mapping["contributors"], "processor_execution_required": True})
    for finding in report.findings:
        if (finding.check_id != "PATHOLOGICAL_STD_GAIN" or finding.status is not DiagnosticStatus.UNKNOWN
                or finding.details.get("feature_key") != "action"
                or finding.details.get("comparison_scope") != "composed_forward_normalization"
                or not finding.details.get("reason", "").startswith("State-dependent action conversion")):
            continue
        gain = mapping["maps"]["preprocessor"]["action_slope"].abs()
        indices = (gain > 1 / threshold).nonzero().flatten().tolist()
        finding.status = DiagnosticStatus.FAIL if indices else DiagnosticStatus.PASS
        finding.severity = Severity.CRITICAL if indices else Severity.INFO
        finding.summary = ("Composed forward action gain exceeds the configured threshold." if indices else
                           "Composed forward action gain remains within the configured threshold.")
        finding.details.update({"required_coverage": False, "composed_gains": gain.tolist(),
                                "gain_threshold": 1 / threshold, "offending_dimensions": indices,
                                "probe_scope": "paired_native_relative_action_chain",
                                "contributing_stages": mapping["contributors"]})
        active = [(index, step) for index, step in pair[0][0].items()
                  if step.norm_map.get(FeatureType.ACTION) in _MODE_STATS]
        if active:
            index, step = active[-1]
            mode = step.norm_map[FeatureType.ACTION]
            stat = "std" if mode is NormalizationMode.MEAN_STD else _MODE_STATS[mode][0]
            key = f"action.{stat}"
            saved = torch.broadcast_to(step.state_dict()[key].squeeze(), gain.shape)
            finding.details.update({"step_index": index, "tensor_slice": saved[indices].tolist(),
                                    "saved_values": saved[indices].tolist(), "normalization_mode": "COMPOSED"})
            finding.provenance = _step_provenance(pair[0][1], pair[0][2], index, key, indices)


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(64).startswith(_LFS_POINTER_PREFIX)


def _find_lfs_pointers(root: Path, report: PreflightReport) -> set[Path]:
    pointers = set()
    for path in sorted(root.rglob("*.safetensors")):
        try:
            local = _local_file(root, str(path.relative_to(root)))
            if not _is_lfs_pointer(local):
                continue
        except (OSError, ValueError):
            continue
        pointers.add(local)
        # Pointer files are tiny; only extract standard pointer fields as evidence.
        with local.open("rb") as stream:
            pointer = stream.read(4096).decode(errors="replace")
        oid = re.search(r"^oid sha256:([0-9a-f]{64})\r?$", pointer, re.MULTILINE)
        size = re.search(r"^size (\d+)\r?$", pointer, re.MULTILINE)
        _add(report, "GIT_LFS_POINTER_ARTIFACT", DiagnosticStatus.FAIL,
             "A Git LFS pointer was saved instead of safetensors tensor data.",
             {"state_file": str(local), "pointer_size_bytes": local.stat().st_size,
              "lfs_oid_sha256": oid[1] if oid else None,
              "declared_object_size": int(size[1]) if size else None,
              "tensor_slice": [], "tensor_values_available": False},
             _provenance(local, "git_lfs_pointer", indices=[]))
    return pointers


def _provenance(
    path: Path, step_name: str | int | None = None, tensor_key: str | None = None,
    indices: list[int] | None = None,
) -> Provenance:
    digest = ""
    # Validated checkpoint artifacts and explicit metadata reach here as resolved
    # paths. Error provenance may retain a rejected link: never follow it to hash
    # an unrelated file merely to describe that validation failure.
    if not path.is_symlink() and path.is_file():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()[:12]
    return Provenance(str(path), step_name, tensor_key, digest, indices)


def _add(
    report: PreflightReport, check: str, status: DiagnosticStatus, summary: str,
    details: dict[str, Any] | None = None, provenance: Provenance | None = None,
    severity: Severity | None = None,
) -> None:
    report.findings.append(DiagnosticFinding(
        check, status, severity or (
            Severity.CRITICAL if status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
            else Severity.INFO
        ), summary, details or {}, provenance,
    ))


def _local_file(root: Path, filename: str, *, allow_directory: bool = False) -> Path:
    """Resolve contained local artifacts, keeping tensor state file-only by default."""
    root = root.resolve()
    if ".." in Path(filename).parts:
        raise ValueError(f"Checkpoint artifact path contains traversal: {filename}")
    path = (root / filename).resolve()
    valid_kind = path.is_file() or (allow_directory and path.is_dir())
    if not path.is_relative_to(root) or not valid_kind:
        raise ValueError(f"Required checkpoint artifact is missing or outside the checkpoint: {filename}")
    if allow_directory and path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink() and not child.resolve().is_relative_to(root):
                raise ValueError(f"Checkpoint artifact contains an outside symlink: {child.relative_to(root)}")
    return path


def _read_tensors(path: Path) -> dict[str, torch.Tensor]:
    if _is_lfs_pointer(path):
        raise ValueError("Git LFS pointer artifact contains no safetensors tensor data")
    with safe_open(path, framework="pt", device="cpu") as buffers:
        return {key: buffers.get_tensor(key) for key in buffers.keys()}


def _validate_step_entry(entry: Any) -> None:
    """Validate saved step structure before dispatching any native constructor.

    Registry names take precedence over import paths, as in LeRobot's loader.
    An unfamiliar string identifier is valid structure and remains uninspected.
    """
    if not isinstance(entry, dict):
        raise ValueError("Processor steps must be JSON objects")
    identifier = "registry_name" if "registry_name" in entry else "class"
    if not isinstance(entry.get(identifier), str) or not entry[identifier].strip():
        raise ValueError(f"Processor {identifier} must be a nonempty string")
    if not isinstance(entry.get("config", {}), dict):
        raise ValueError("Processor step config must be a JSON object")
    if "state_file" in entry and (not isinstance(entry["state_file"], str) or not entry["state_file"]):
        raise ValueError("Processor state_file must be a nonempty string")
    artifacts = entry.get("artifacts", {})
    if not isinstance(artifacts, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in artifacts.items()
    ):
        raise ValueError("Processor artifacts must map nonempty string keys to filenames")


def _step_class(entry: dict[str, Any]) -> type | None:
    """Resolve only inspected native classes; never import checkpoint-specified code."""
    if not isinstance(entry, dict):
        return None
    if "registry_name" in entry:
        if not isinstance(entry["registry_name"], str):
            return None
        try:
            cls = ProcessorStepRegistry.get(entry["registry_name"])
        except KeyError:
            return None
        return cls if cls in _SAFE_CLASSES else None
    return next((cls for cls in _SAFE_CLASSES if entry.get("class") ==
                 f"{cls.__module__}.{cls.__name__}"), None)


def _normalization_base(entry: dict[str, Any]) -> type | None:
    """Recognize two saved Molmo normalization subclasses without importing policy code.

    Their native implementation inherits the corresponding LeRobot normalization
    step and only masks selected coordinates after the base transform.
    """
    if not isinstance(entry, dict):
        return None
    name = entry.get("registry_name")
    return _MASKED_NORMALIZERS.get(name) if isinstance(name, str) and name in _MASKED_NORMALIZERS else _step_class(entry)


def _role(path: Path, config: dict[str, Any]) -> str | None:
    name = f"{path.stem} {config.get('name', '')}".lower()
    if "preprocessor" in name:
        return "pre"
    if "postprocessor" in name:
        return "post"
    return None


def _legacy_buffer_keys(keys: Any) -> list[str]:
    """Recognize embedded normalization names used by LeRobot's migration path."""
    return sorted(
        key for key in keys
        if isinstance(key, str) and key.startswith(_LEGACY_BUFFER_PREFIXES)
    )


def _metadata_provenance(root: Path, filename: str, tensor_key: str | None = None) -> Provenance:
    path = root / filename
    try:
        return _provenance(_local_file(root, filename), "legacy_embedded", tensor_key)
    except (OSError, ValueError):
        # An escaping symlink is not read or hashed as checkpoint evidence.
        return Provenance(str(path), "legacy_embedded", tensor_key, "")


def _legacy_header_metadata(root: Path) -> tuple[list[str], dict[str, Any]]:
    """Read a local sidecar containing safetensors header entries, never tensor values."""
    path = _local_file(root, "model.safetensors.header.json")
    header = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(header, dict) or header.get("format") != "safetensors-header-v1":
        raise ValueError("Expected safetensors-header-v1 metadata object")
    if header.get("source_file") != "model.safetensors":
        raise ValueError("Header metadata must identify model.safetensors as source_file")
    if "complete" in header and type(header["complete"]) is not bool:
        raise ValueError("Header complete marker must be boolean when present")
    tensors = header.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Header metadata must contain a nonempty tensors mapping")
    for key, descriptor in tensors.items():
        if not isinstance(key, str) or not key or not isinstance(descriptor, dict):
            raise ValueError("Header tensor entries must have string keys and object descriptors")
        dtype, shape, offsets = (
            descriptor.get("dtype"), descriptor.get("shape"), descriptor.get("data_offsets")
        )
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Header tensor {key!r} has no dtype")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"Header tensor {key!r} has an invalid shape")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"Header tensor {key!r} has invalid data_offsets")
    total_size = header.get("total_size")
    if total_size is not None and (type(total_size) is not int or total_size < 0):
        raise ValueError("Header total_size must be a nonnegative integer")
    keys = _legacy_buffer_keys(tensors)
    details = {
        "evidence_source": "safetensors_header_metadata",
        "digest_scope": "local_header_metadata_file",
        "source_file": header["source_file"],
        "legacy_buffer_keys": keys,
        "legacy_buffer_metadata": {
            key: {field: tensors[key][field] for field in ("dtype", "shape", "data_offsets")}
            for key in keys
        },
        "repo_id": header.get("repo_id"),
        "revision": header.get("revision"),
        "declared_header_sha256_prefix": header.get("header_sha256_prefix"),
        "declared_header_digest_scope": "safetensors_header_bytes_including_length_prefix",
        "header_complete": header.get("complete"),
        "declared_total_size": total_size,
        "tensor_values_available": False,
        "tensor_shape_dtype_available": True,
    }
    return keys, details


def _legacy_weight_index(root: Path) -> tuple[list[str], dict[str, Any]]:
    """Read only the key-to-local-shard map in a standard safetensors index."""
    path = _local_file(root, "model.safetensors.index.json")
    index = json.loads(path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Safetensors index must contain a nonempty weight_map")
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(filename, str) or not filename:
            raise ValueError("Safetensors weight_map must map tensor names to shard filenames")
        shard = Path(filename)
        if shard.is_absolute() or ".." in shard.parts or "://" in filename:
            raise ValueError(f"Safetensors index has an unsafe shard filename: {filename!r}")
    keys = _legacy_buffer_keys(weight_map)
    details = {
        "evidence_source": "safetensors_weight_index",
        "digest_scope": "local_weight_index_file",
        "legacy_buffer_keys": keys,
        "legacy_weight_map": {key: weight_map[key] for key in keys},
        "tensor_values_available": False,
        "tensor_shape_dtype_available": False,
    }
    metadata = index.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata:
        details["declared_total_size"] = metadata["total_size"]
    return keys, details


def _legacy_metadata_evidence(
    root: Path, report: PreflightReport,
) -> tuple[list[str], Path | None, dict[str, Any]]:
    """Prefer header metadata, then a local weight index when payload was skipped."""
    for filename, reader in (
        ("model.safetensors.header.json", _legacy_header_metadata),
        ("model.safetensors.index.json", _legacy_weight_index),
    ):
        path = root / filename
        if not path.exists() and not path.is_symlink():
            continue
        try:
            keys, details = reader(root)
            return keys, _local_file(root, filename), details
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _add(
                report, "SCHEMA_GENERATION", DiagnosticStatus.UNKNOWN,
                f"Local legacy metadata {filename} is unreadable or invalid: {exc}",
                {"evidence_source": filename, "digest_scope": "local_metadata_file",
                 "tensor_values_available": False},
                _metadata_provenance(root, filename),
            )
    return [], None, {}


def _model_weights_skipped(root: Path, report: PreflightReport) -> bool:
    """Read the sampler's explicit skipped-weight marker without using the network."""
    path = root / "download_manifest.json"
    if not path.exists() and not path.is_symlink():
        return False
    try:
        manifest_path = _local_file(root, path.name)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or type(manifest.get("model_weights_skipped")) is not bool:
            raise ValueError("Manifest must declare boolean model_weights_skipped")
        return manifest["model_weights_skipped"]
    except (OSError, ValueError, TypeError) as exc:
        _add(
            report, "SCHEMA_GENERATION", DiagnosticStatus.UNKNOWN,
            f"Cannot establish whether the model was skipped from download_manifest.json: {exc}",
            {"evidence_source": "download_manifest.json", "digest_scope": "local_manifest_file"},
            _metadata_provenance(root, path.name),
        )
        return False


def _discover(root: Path, report: PreflightReport) -> list[tuple[Path, dict[str, Any]]]:
    pipelines = []
    invalid_files: set[str] = set()
    for path in sorted(root.glob("*.json")):
        try:
            path = _local_file(root, path.name)
            config = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(config, dict) and isinstance(config.get("steps"), list):
                valid = True
                for index, entry in enumerate(config["steps"]):
                    try:
                        _validate_step_entry(entry)
                    except ValueError as exc:
                        valid = False
                        invalid_files.add(str(path))
                        _add(report, "SCHEMA_GENERATION", DiagnosticStatus.FAIL,
                             f"Invalid processor step {index}: {exc}",
                             {"step_index": index, "processor_file": path.name},
                             _provenance(path, index))
                if valid:
                    pipelines.append((path, config))
            elif path.name in _CONFIG_NAMES:
                raise ValueError("Expected native processor config with a steps list")
        except (OSError, ValueError) as exc:
            # An unrelated metadata file is not a processor schema failure.
            if path.name in _CONFIG_NAMES:
                invalid_files.add(str(path))
                try:
                    provenance = _provenance(_local_file(root, path.name))
                except (OSError, ValueError):
                    provenance = Provenance(str(path), None, None, "")
                _add(report, "SCHEMA_GENERATION", DiagnosticStatus.FAIL, str(exc),
                     provenance=provenance)

    legacy_path = root / "model.safetensors"
    model_skipped = _model_weights_skipped(root, report) if not legacy_path.is_file() else False
    legacy_keys: list[str] = []
    legacy_evidence_path: Path | None = None
    legacy_details: dict[str, Any] = {}
    try:
        model_pointer = legacy_path.is_file() and _is_lfs_pointer(_local_file(root, "model.safetensors"))
    except (OSError, ValueError):
        model_pointer = False
    if legacy_path.is_file() and not model_pointer:
        try:
            local_model = _local_file(root, "model.safetensors")
            with safe_open(local_model, framework="pt", device="cpu") as weights:
                legacy_keys = _legacy_buffer_keys(weights.keys())
            legacy_evidence_path = local_model
            legacy_details = {
                "evidence_source": "model_safetensors",
                "digest_scope": "whole_model_file",
                "legacy_buffer_keys": legacy_keys,
                "tensor_values_available": True,
            }
        except Exception as exc:
            _add(report, "SCHEMA_GENERATION", DiagnosticStatus.FAIL,
                 f"Cannot read model.safetensors: {exc}",
                 provenance=_metadata_provenance(root, "model.safetensors"))
    else:
        legacy_keys, legacy_evidence_path, legacy_details = _legacy_metadata_evidence(root, report)
        if model_skipped:
            legacy_details["model_weights_skipped"] = True
    if not pipelines:
        if legacy_keys:
            _add(report, "SCHEMA_GENERATION_MISMATCH", DiagnosticStatus.FAIL,
                 "Legacy embedded normalization buffers exist without modern processor JSONs.",
                 legacy_details,
                 _provenance(legacy_evidence_path, "legacy_embedded", legacy_keys[0])
                 if legacy_evidence_path is not None else None)
        else:
            metadata_present = any(
                (root / filename).exists() or (root / filename).is_symlink()
                for filename in ("model.safetensors.header.json", "model.safetensors.index.json")
            )
            unknown = model_pointer or (not legacy_path.is_file() and (model_skipped or metadata_present))
            _add(report, "SCHEMA_GENERATION",
                 DiagnosticStatus.UNKNOWN if unknown else DiagnosticStatus.FAIL,
                 "No modern processor JSONs or recognized legacy buffers were found." if legacy_path.is_file()
                 else "No modern processor JSONs or conclusive local model metadata were found." if unknown
                 else "No modern processor JSONs or recognized legacy buffers were found.",
                 {"evidence_source": legacy_details.get("evidence_source"),
                  "digest_scope": legacy_details.get("digest_scope"),
                  "tensor_values_available": legacy_details.get("tensor_values_available", False),
                  "model_weights_skipped": model_skipped},
                 _provenance(legacy_evidence_path) if legacy_evidence_path is not None else None)
    else:
        roles = {_role(path, config) for path, config in pipelines}
        missing = [role for role in ("pre", "post") if role not in roles] if roles - {None} else []
        conflicts = []
        for role in ("pre", "post"):
            entries = [(path, config) for path, config in pipelines if _role(path, config) == role]
            if entries and any(config != entries[0][1] for _, config in entries[1:]):
                conflicts.append({"pipeline_role": role, "processor_files": [str(path) for path, _ in entries],
                                  "sources": [_provenance(path).to_dict() for path, _ in entries]})
        status = (DiagnosticStatus.CONFLICT if conflicts else DiagnosticStatus.FAIL if missing or invalid_files
                  else DiagnosticStatus.PASS)
        _add(report, "SCHEMA_GENERATION", status,
             "Multiple processor JSONs for the same role disagree; no canonical pipeline was selected." if conflicts else
             "Some processor JSONs contain invalid declarations." if invalid_files else
             "Modern processor config pair is incomplete." if missing else "Native processor JSONs found.",
             {"processor_files": [str(path) for path, _ in pipelines], "missing_roles": missing,
              "invalid_processor_files": sorted(invalid_files),
              "conflicting_roles": conflicts,
              "legacy_buffer_keys": legacy_keys,
              "legacy_evidence_source": legacy_details.get("evidence_source"),
              "legacy_digest_scope": legacy_details.get("digest_scope")})
    return pipelines


def _buffer_finding(
    report: PreflightReport, check: str, path: Path, step_name: str, step_index: int | None,
    key: str, tensor: torch.Tensor, mask: torch.Tensor, extra: dict[str, Any],
    severity: Severity | None = None,
) -> None:
    indices = mask.reshape(-1).nonzero().flatten().tolist()
    if not indices:
        return
    values = tensor.reshape(-1)[indices].tolist()
    _add(report, check, DiagnosticStatus.FAIL,
         f"{key} has unsafe normalization buffer values at indices {indices}.",
         {"step_index": step_index, "feature_key": key.rsplit(".", 1)[0],
          "saved_values": values, "tensor_slice": values, "tensor_shape": list(tensor.shape),
          "indexing": "flattened row-major", **extra},
         _provenance(path, step_name, key, indices), severity=severity)


def _normalizer_contract(
    report: PreflightReport, path: Path, entry: dict[str, Any], index: int | None,
    saved_state: dict[str, torch.Tensor], *, config_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Check an inspected native step even if another step cannot be loaded."""
    cls = _normalization_base(entry)
    if cls not in _NORMALIZERS:
        return {}
    step = cls(**{**entry.get("config", {}), "device": "cpu"})
    step.load_state_dict(saved_state)
    state = step.state_dict()
    effective_path = config_path if entry.get("config", {}).get("stats") and config_path is not None else path
    name = entry.get("registry_name", cls.__name__)
    direction = "inverse" if cls is UnnormalizerProcessorStep else "forward"
    for tensor_key, active in state.items():
        saved = saved_state.get(tensor_key)
        if entry.get("config", {}).get("stats") and saved is not None and (saved.shape != active.shape or not torch.allclose(
                saved.double(), active.double(), rtol=0, atol=0, equal_nan=True)):
            _add(report, "INLINE_STAT_OVERRIDE_ADVISORY", DiagnosticStatus.PASS,
                 f"Explicit inline statistics override serialized tensor values for {tensor_key}; native loading ignores the state-file value.",
                 {"feature_key": tensor_key.rsplit(".", 1)[0], "step_index": index,
                  "ignored_state_values": saved.reshape(-1).tolist(), "ignored_state_shape": list(saved.shape),
                  "tensor_slice": active.reshape(-1).tolist(),
                  "active_json_values": active.reshape(-1).tolist(), "effective_source": "config.stats",
                  "state_file_ignored": True, "state_file": str(path), "runtime_precedence_verified": True,
                  "ignored_state_provenance": _provenance(path, name, tensor_key).to_dict()},
                 _provenance(effective_path, name, tensor_key, list(range(active.numel()))),
                 severity=Severity.WARNING)
    features = _normalizer_features(entry, state)
    for key, declaration in features.items():
        feature = PolicyFeature(FeatureType(declaration["type"]), tuple(declaration["shape"]))
        if "_declared_type" in declaration:
            _add(report, "FEATURE_CONTRACT", DiagnosticStatus.CONFLICT,
                 "The action declaration disagrees with the native ACTION normalization path.",
                 {"feature_key": key, "step_index": index,
                  "declared_feature_type": declaration["_declared_type"], "effective_feature_type": "ACTION",
                  "tensor_slice": []}, _provenance(effective_path, name, key, []))
        if (step.normalize_observation_keys is not None and feature.type != FeatureType.ACTION
                and key not in step.normalize_observation_keys):
            continue
        if feature.type == FeatureType.ACTION and key != "action":
            continue  # Native _normalize_action always looks up ACTION, not arbitrary names.
        mode = step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
        required = _MODE_STATS.get(mode, ())
        missing = [f"{key}.{stat}" for stat in required if f"{key}.{stat}" not in state]
        prefixed = sorted(candidate for candidate in state if any(
            candidate.endswith(f".{expected}") or candidate.endswith(f"/{expected}")
            for expected in missing
        ))
        if prefixed:
            _add(report, "STAT_KEY_NAMESPACE_MISMATCH", DiagnosticStatus.FAIL,
                 f"Required statistics for {key} are absent; saved keys use a different namespace.",
                 {"feature_key": key, "step_index": index, "required_buffer_keys": missing,
                  "expected_buffer_keys": [f"{key}.{name}" for name in required],
                  "available_buffer_keys": sorted(state), "matching_prefixed_keys": prefixed,
                  "declared_feature_keys": sorted(step.features),
                  "actual_feature_keys": sorted({name.rsplit(".", 1)[0] for name in state if "." in name}),
                  "tensor_slice": []}, _provenance(effective_path, name, missing[0], []))
        for stat in required:
            tensor_key = f"{key}.{stat}"
            tensor = state.get(tensor_key)
            if tensor is None:
                _add(report, "FEATURE_CONTRACT", DiagnosticStatus.FAIL,
                     f"Required normalization buffer {tensor_key} is missing; normalization may be bypassed.",
                     {"feature_key": key, "step_index": index, "tensor_slice": [],
                      "step_direction": direction, "normalization_mode": mode.value,
                      "available_buffer_keys": sorted(state)}, _provenance(effective_path, name, tensor_key, []))
                continue
            try:
                broadcast = torch.broadcast_shapes(tuple(tensor.shape), feature.shape)
                compatible = (broadcast[-len(feature.shape):] == feature.shape and all(
                    dim == 1 for dim in broadcast[:-len(feature.shape)])) if feature.shape else not broadcast
            except RuntimeError:
                compatible = False
            if not compatible:
                _add(report, "FEATURE_CONTRACT", DiagnosticStatus.FAIL,
                     f"{tensor_key} cannot broadcast to the declared feature shape.",
                     {"feature_key": key, "step_index": index, "buffer_shape": list(tensor.shape),
                      "feature_shape": list(feature.shape), "tensor_slice": tensor.reshape(-1).tolist()},
                     _provenance(effective_path, name, tensor_key, list(range(tensor.numel()))))
    return state


def _active_buffer_keys(entry: dict[str, Any]) -> list[str]:
    """Required keys for the native feature/mode/selection contract."""
    config = entry.get("config", {})
    keys = []
    for key, feature in config.get("features", {}).items():
        kind = feature.get("type")
        if kind == FeatureType.ACTION.value and key != "action":
            continue
        selected = config.get("normalize_observation_keys")
        if kind != FeatureType.ACTION.value and selected is not None and key not in selected:
            continue
        mode = NormalizationMode(config.get("norm_map", {}).get(kind, "IDENTITY"))
        keys.extend(f"{key}.{stat}" for stat in _MODE_STATS.get(mode, ()))
    action_mode = NormalizationMode(config.get("norm_map", {}).get(FeatureType.ACTION.value, "IDENTITY"))
    keys.extend(f"action.{stat}" for stat in _MODE_STATS.get(action_mode, ()))
    return list(dict.fromkeys(keys))


def _multi_dataset_action_prefixes(keys: Any) -> tuple[str, ...]:
    groups: dict[str, set[str]] = {}
    for key in keys:
        match = re.fullmatch(r"(.+)\.buffer\.action\.(mean|std)", key)
        if match:
            groups.setdefault(match[1], set()).add(match[2])
    complete = tuple(sorted(prefix for prefix, stats in groups.items() if stats == {"mean", "std"}))
    return complete if len(complete) >= 2 else ()


def _classify_unbound_base_stats(report: PreflightReport) -> None:
    """Distinguish a multi-dataset binding requirement from ordinary prefix drift.

    A postprocessor only declares action, so pair it with the forward state
    requirement using the same multi-dataset namespaces in this checkpoint.
    This does not establish whether neural weights have been fine-tuned.
    """
    missing_state: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for finding in report.findings:
        details = finding.details
        if (finding.check_id == "FEATURE_CONTRACT" and finding.status is DiagnosticStatus.FAIL
                and details.get("feature_key") == "observation.state"
                and details.get("step_direction") == "forward"
                and details.get("normalization_mode") == "MEAN_STD" and finding.provenance):
            prefixes = _multi_dataset_action_prefixes(details.get("available_buffer_keys", []))
            if prefixes:
                group = (finding.provenance.file_path, prefixes)
                missing_state.setdefault(group, set()).add(finding.provenance.tensor_key or "")
    unbound = {prefixes for (_, prefixes), keys in missing_state.items()
               if {"observation.state.mean", "observation.state.std"} <= keys}
    classified = set()
    for finding in report.findings:
        prefixes = _multi_dataset_action_prefixes(finding.details.get("available_buffer_keys", []))
        if finding.check_id != "STAT_KEY_NAMESPACE_MISMATCH" or prefixes not in unbound:
            continue
        finding.check_id = "UNBOUND_BASE_MODEL_STATS"
        classified.add(prefixes)
        finding.summary = (
            "Checkpoint contains multi-dataset pretraining buffers without bound target dataset statistics; "
            "zero-shot execution without dataset-stat binding or fine-tuning will silently bypass normalization."
        )
        finding.details.update({
            "subtype": "multi_dataset_pretraining_buffers", "dataset_namespaces": list(prefixes),
            "target_dataset_stats_bound": False, "missing_observation_state_stats": [
                "observation.state.mean", "observation.state.std"],
            "training_provenance_verified": False,
        })
    for prefixes in unbound - classified:
        # Action may already be bound while observation.state is still wholly
        # absent. That incomplete binding also bypasses state normalization.
        source = next(finding for finding in report.findings
                      if finding.check_id == "FEATURE_CONTRACT" and finding.provenance
                      and (finding.provenance.file_path, prefixes) in missing_state
                      and finding.details.get("feature_key") == "observation.state")
        _add(report, "UNBOUND_BASE_MODEL_STATS", DiagnosticStatus.FAIL,
             "Checkpoint contains multi-dataset pretraining buffers without bound target observation statistics; "
             "zero-shot execution without dataset-stat binding or fine-tuning will silently bypass state normalization.",
             {**source.details, "subtype": "multi_dataset_pretraining_buffers", "dataset_namespaces": list(prefixes),
              "target_dataset_stats_bound": False, "training_provenance_verified": False,
              "missing_observation_state_stats": ["observation.state.mean", "observation.state.std"]},
             source.provenance)


def _screen_buffers(
    report: PreflightReport, path: Path, entry: dict[str, Any], index: int | None,
    threshold: float, input_float_dtype: str | None = None, *, config_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    state = _read_tensors(path)
    effective = _normalizer_contract(report, path, entry, index, state, config_path=config_path)
    effective_path = config_path if entry.get("config", {}).get("stats") and config_path is not None else path
    return _screen_normalizer_state(report, effective_path, entry, index, threshold, effective, input_float_dtype,
                                    serialized_fp16_keys=({key for key, value in state.items() if value.dtype is torch.float16}
                                                          if not entry.get("config", {}).get("stats") else set()))


def _gain_probe_details(
    reference: torch.Tensor | None, denominator: torch.Tensor, mask: torch.Tensor, *, range_mode: bool,
) -> dict[str, Any]:
    """Evaluate a local float32 perturbation using the inspected forward formulas."""
    if reference is None:
        return {"probe_offset": 0.20, "probe_available": False}
    reference, denominator = torch.broadcast_tensors(reference.float(), denominator.float())
    selected = mask.expand_as(reference)
    probe = reference + 0.20
    baseline = (reference - reference) / denominator
    output = (probe - reference) / denominator
    if range_mode:
        baseline, output = 2 * baseline - 1, 2 * output - 1
    return {
        "probe_offset": 0.20, "probe_dtype": "float32", "probe_available": True,
        "probe_tensor_shape": list(reference.shape), "probe_is_synthetic": True,
        "baseline_output_values": baseline[selected].tolist(),
        "probe_reference_values": reference[selected].tolist(),
        "probe_input_values": probe[selected].tolist(),
        "probe_output_values": output[selected].tolist(),
    }


def _gain_finding(
    report: PreflightReport, path: Path, name: str, index: int | None, key: str,
    tensor: torch.Tensor, mask: torch.Tensor, extra: dict[str, Any],
) -> None:
    if not bool(mask.any()):
        return
    _buffer_finding(report, "PATHOLOGICAL_STD_GAIN", path, name, index, key, tensor, mask, extra)
    finding = report.findings[-1]
    maximum_gain = max(extra["effective_gains_with_eps"])
    finding.summary = (
        f"Forward {extra['normalization_mode']} normalization has gain up to {maximum_gain:.8g}x "
        f"at indices {finding.provenance.violating_indices}."
    )
    if extra.get("probe_available"):
        maximum_output = max(abs(value) for value in extra["probe_output_values"])
        finding.summary += f" A +0.20 float32 probe produces output magnitude up to {maximum_output:.8g}."


def _screen_normalizer_state(
    report: PreflightReport, path: Path, entry: dict[str, Any], index: int | None,
    threshold: float, state: dict[str, torch.Tensor], input_float_dtype: str | None = None,
    *, legacy: bool = False, serialized_fp16: bool = False,
    serialized_fp16_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Screen declared native normalization state already verified by its reader."""
    name = entry.get("registry_name", entry.get("class", path.stem))
    config = entry.get("config", {})
    eps = float(config.get("eps", 1e-8))
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("Normalization epsilon must be finite and nonnegative")
    direction = "inverse" if _normalization_base(entry) is UnnormalizerProcessorStep else "forward"
    target_dtype = DeviceProcessorStep.DTYPE_MAPPING.get(input_float_dtype, torch.float32)
    # Hypothetical half execution remains visible but cannot block a float32
    # checkpoint. Only an actual saved cast/buffer is evidence of FP16 use.
    features = _normalizer_features(entry, state)
    norm_map = config.get("norm_map", {})
    selected_observations = config.get("normalize_observation_keys")
    if not isinstance(features, dict) or not isinstance(norm_map, dict):
        raise ValueError("Normalizer features and norm_map must be mappings")
    for feature, declaration in features.items():
        if not isinstance(declaration, dict):
            continue
        feature_type = declaration.get("type")
        if feature_type == FeatureType.ACTION.value and feature != "action":
            continue  # Native action normalization always uses the ACTION constant.
        if (feature_type != FeatureType.ACTION.value and selected_observations is not None
                and feature not in selected_observations):
            continue
        mode_name = norm_map.get(feature_type, NormalizationMode.IDENTITY.value)
        try:
            mode = NormalizationMode(mode_name)
        except ValueError:
            raise ValueError(f"Unsupported normalization mode for {feature}: {mode_name}") from None
        required = _MODE_STATS.get(mode, ())
        if not required:
            continue
        feature_half_buffers = (serialized_fp16
                                or bool({f"{feature}.{stat}" for stat in required} & (serialized_fp16_keys or set()))
                                or any(state.get(f"{feature}.{stat}", torch.empty(0)).dtype is torch.float16
                                       for stat in required))
        configured_fp16 = (input_float_dtype == "float16" or feature_half_buffers) and not legacy
        # An explicit BF16 contract has a different exponent range. Do not
        # manufacture a hypothetical FP16 advisory that can only fail strict CI.
        explicit_bf16 = input_float_dtype == "bfloat16" and not feature_half_buffers and not legacy
        risk_dtype = torch.bfloat16 if explicit_bf16 else torch.float16
        risk_dtype_name = str(risk_dtype).removeprefix("torch.")
        mask = state.get(f"{feature}.mask") if entry.get("registry_name") in _MASKED_NORMALIZERS else None
        feature_shape = declaration.get("shape")
        feature_last_dim = feature_shape[-1] if isinstance(feature_shape, (list, tuple)) and feature_shape else None
        available = {stat: state[f"{feature}.{stat}"] for stat in required if f"{feature}.{stat}" in state}
        # Native arithmetic broadcasts paired statistics. Screen that same
        # coordinate space rather than requiring their stored shapes to match.
        broadcast_shape = torch.broadcast_shapes(*(value.shape for value in available.values()))
        broadcast_stats = {stat: value.expand(broadcast_shape) for stat, value in available.items()}
        for stat in required:
            key = f"{feature}.{stat}"
            tensor = broadcast_stats.get(stat)
            if tensor is None:
                continue  # The contract check above reports the missing active buffer.
            active = torch.ones_like(tensor, dtype=torch.bool)
            if mask is not None and mask.ndim == 1 and mask.shape[0] == feature_last_dim:
                if tensor.numel() == 1:
                    active.fill_(bool(mask.any()))
                elif tensor.ndim and mask.shape[0] == tensor.shape[-1]:
                    active = mask.bool().expand_as(tensor)
                elif mask.numel() == tensor.numel():
                    active = mask.reshape(tensor.shape).bool()
                # An unbroadcastable Molmo mask is ignored by its native mixin.
            if not bool(active.any()):
                continue
            saved = tensor.to(torch.float64)
            cast = tensor.to(risk_dtype)
            extra: dict[str, Any] = {
                "eps": eps, "screened_dtype": risk_dtype_name, "observed_execution": False,
                "buffer_screen_only": True, "normalization_mode": mode.value,
                "step_direction": direction, "input_float_dtype": input_float_dtype,
                "normalization_implementation": "legacy Normalize" if legacy else "modern NormalizerProcessorStep",
                "modern_stats_follow_input_dtype": True,
                "legacy_default_buffer_dtype": "float32",
                "fp16_execution_configured": configured_fp16,
                "serialized_fp16_buffers": feature_half_buffers,
                "feature_shape_source": declaration.get("_shape_source", "declared_feature"),
                "feature_shape_declared": declaration.get("_declared", True),
            }
            if available[stat].shape != tensor.shape:
                extra.update({"source_tensor_shape": list(available[stat].shape),
                              "source_saved_values": available[stat].reshape(-1).tolist(),
                              "broadcast_tensor_shape": list(tensor.shape),
                              "indexing": "flattened row-major broadcast normalization coordinates"})
            invalid = active & ~torch.isfinite(saved)
            reasons_invalid = ["nonfinite_saved_statistic"] if bool(invalid.any()) else []
            if stat == "std":
                negative = active & (saved < 0)
                invalid |= negative
                if bool(negative.any()):
                    reasons_invalid.append("negative_standard_deviation")
            elif stat == required[0] and mode != NormalizationMode.MEAN_STD:
                upper = broadcast_stats.get(required[1])
                if upper is not None:
                    reversed_bounds = (saved > upper.double()) & active
                    invalid |= reversed_bounds
                    if bool(reversed_bounds.any()):
                        reasons_invalid.append("reversed_interval")
                        extra["upper_buffer_key"] = f"{feature}.{required[1]}"
                        extra["upper_values"] = upper.reshape(-1).tolist()
            _buffer_finding(report, "INVALID_NORMALIZATION_STATS", path, name, index, key, tensor, invalid,
                            {**extra, "invalid_reasons": reasons_invalid})
            if stat == "std":
                if direction == "forward":
                    # Match native dtype rounding before estimating division gain.
                    effective_gain = (tensor.to(target_dtype) + eps).double().reciprocal()
                    small = active & (saved >= 0) & (saved < threshold) & (effective_gain > 1 / threshold)
                    probe = _gain_probe_details(state.get(f"{feature}.mean"), tensor.float() + eps,
                                                small, range_mode=False)
                    _gain_finding(report, path, name, index, key, tensor, small,
                                  {**extra, "std_warn_threshold": threshold,
                                   "amplification_factors": (1 / saved[small]).tolist(),
                                   "gain_dtype": str(target_dtype).removeprefix("torch."),
                                   "gain_formula": "1 / (std + eps)",
                                   "effective_gains_with_eps": effective_gain[small].tolist(), **probe})
                elif feature_type == FeatureType.ACTION.value:
                    dead = active & (saved == 0)
                    _buffer_finding(report, "DEAD_ACTION_DIMENSION", path, name, index, key, tensor, dead,
                                    {**extra, "action_response_scale": 0.0}, Severity.WARNING)
            risk = active & ~torch.isfinite(cast)
            reasons = [f"nonfinite_or_overflowing_{risk_dtype_name}_constant"] if bool(risk.any()) else []
            if direction == "forward" and stat == "std":
                denom = cast + eps
                denominator_risk = active & ((denom == 0) | ~torch.isfinite(denom))
                risk |= denominator_risk
                if bool(denominator_risk.any()):
                    reasons.append(f"unsafe_{risk_dtype_name}_denominator")
                    extra["denominator_values"] = denom.reshape(-1)[denominator_risk.reshape(-1)].tolist()
            elif direction == "forward" and stat == required[0] and mode != NormalizationMode.MEAN_STD:
                upper = broadcast_stats.get(required[1])
                if upper is not None:
                    span = upper.to(risk_dtype) - cast
                    # Historical Normalize adds epsilon to every MIN_MAX span;
                    # the modern step substitutes it only for an exactly zero span.
                    legacy_additive_eps = legacy and mode is NormalizationMode.MIN_MAX
                    denom = span + eps if legacy_additive_eps else torch.where(span == 0, torch.full_like(span, eps), span)
                    if mode in (NormalizationMode.MIN_MAX, NormalizationMode.QUANTILES, NormalizationMode.QUANTILE10):
                        typed_span = upper.to(target_dtype) - tensor.to(target_dtype)
                        typed_denom = (typed_span + eps if legacy_additive_eps else
                                       torch.where(typed_span == 0, torch.full_like(typed_span, eps), typed_span))
                        gain = 2 / typed_denom.double()
                        saved_span = upper.double() - saved
                        small = active & (saved_span >= 0) & (saved_span < threshold) & (gain > 1 / threshold)
                        span32 = upper.float() - tensor.float()
                        denom32 = (span32 + eps if legacy_additive_eps else
                                   torch.where(span32 == 0, torch.full_like(span32, eps), span32))
                        probe = _gain_probe_details(tensor, denom32, small, range_mode=True)
                        lower_name, upper_name = required
                        span_expression = f"{upper_name} - {lower_name}"
                        _gain_finding(report, path, name, index, key, tensor, small,
                                      {**extra, "std_warn_threshold": threshold,
                                       "upper_buffer_key": f"{feature}.{required[1]}",
                                       "upper_values": upper[small].tolist(),
                                       "saved_span_values": saved_span[small].tolist(),
                                       "amplification_factors": (2 / saved_span[small]).tolist(),
                                       "gain_dtype": str(target_dtype).removeprefix("torch."),
                                       "gain_formula": f"2 / ({span_expression} + eps)" if legacy_additive_eps else
                                       f"2 / where({span_expression} == 0, eps, {span_expression})",
                                       "effective_gains_with_eps": gain[small].tolist(), **probe})
                    denominator_risk = active & ((denom == 0) | ~torch.isfinite(denom))
                    risk |= denominator_risk
                    if bool(denominator_risk.any()):
                        reasons.append(f"unsafe_{risk_dtype_name}_range_denominator")
                        extra["upper_buffer_key"] = f"{feature}.{required[1]}"
                        extra["upper_values"] = upper.reshape(-1)[denominator_risk.reshape(-1)].tolist()
            precision_condition = (
                "Legacy Normalize keeps float32 buffers: FP16 input alone promotes arithmetic to float32. "
                "NaN/Inf requires casting the normalization module/buffers to .half(), or migration to modern "
                "NormalizerProcessorStep, which auto-casts stats to FP16 input dtype."
                if legacy else
                "Modern NormalizerProcessorStep auto-casts statistics to the input dtype; "
                "FP16 input can immediately produce NaN/Inf when the cast denominator is zero. "
                "Legacy Normalize instead retains float32 buffers unless explicitly cast to .half()."
            )
            before_risk = len(report.findings)
            # Keep explicit BF16 hazards in their own dtype scope; they are not
            # evidence of hypothetical FP16 execution.
            if explicit_bf16:
                _buffer_finding(report, "BF16_NON_FINITE_RISK", path, name, index, key, tensor, risk,
                                {**extra, "risk_reasons": reasons, "bfloat16_execution_configured": True},
                                Severity.CRITICAL)
                continue
            _buffer_finding(report, "FP16_NON_FINITE_RISK", path, name, index, key, tensor, risk,
                            {**extra, "risk_reasons": reasons, "precision_behavior": precision_condition,
                             "legacy_half_input_only_promotes_to_float32": legacy,
                             "requires_legacy_buffer_half_cast_or_modern_migration": legacy,
                             "configured_runtime_nonfinite_observed": False},
                            Severity.CRITICAL if configured_fp16 else Severity.WARNING)
            if len(report.findings) > before_risk:
                report.findings[-1].summary += " " + precision_condition
                if not configured_fp16:
                    report.findings[-1].status = DiagnosticStatus.PASS
                    report.findings[-1].summary = ("Conditional legacy precision advisory. " if legacy else
                                                   "Hypothetical FP16 execution advisory; FP16 is not configured. ") + report.findings[-1].summary
                    report.findings[-1].details.update({"advisory": True, "blocking": False,
                                                       "fp16_execution_conditional": True,
                                                       "default_buffer_arithmetic_dtype": "float32"})
    return state


def _feature_dict(value: Any) -> dict[str, PolicyFeature]:
    if not isinstance(value, dict):
        raise ValueError("Feature declarations must be dictionaries")
    result = {}
    for key, declaration in value.items():
        shape = tuple(declaration["shape"])
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"Invalid feature shape for {key}: {shape}")
        result[key] = PolicyFeature(FeatureType(declaration["type"]), shape)
    return result


def _feature_details(feature: PolicyFeature | None) -> dict[str, Any] | None:
    return {"type": feature.type.value, "shape": list(feature.shape)} if feature else None


def _visual_contract_without_constructors(
    root: Path, path: Path, config: dict[str, Any], report: PreflightReport,
    policy: dict[PipelineFeatureType, dict[str, PolicyFeature]] | None,
) -> None:
    """Compare declared camera keys even if a tokenizer constructor is unavailable.

    A normalizer's visual declarations pass through its native transform_features
    unchanged. Only a subsequent native rename can change their names here. If an
    unfamiliar later step could change the contract, leave it to native validation.
    """
    if policy is None or _role(path, config) != "pre":
        return
    expected = {key: feature for key, feature in policy[PipelineFeatureType.OBSERVATION].items()
                if feature.type == FeatureType.VISUAL}
    if not expected:
        return
    steps = config["steps"]
    passthrough = {
        "normalizer_processor", "unnormalizer_processor", "device_processor",
        "to_batch_processor", "smolvla_new_line_processor", "tokenizer_processor",
        "molmoact2_masked_normalizer", "molmoact2_masked_unnormalizer",
    }
    for index, entry in enumerate(steps):
        if not isinstance(entry, dict) or _normalization_base(entry) not in _NORMALIZERS:
            continue
        step_config = entry.get("config")
        declarations = step_config.get("features", {}) if isinstance(step_config, dict) else {}
        if not isinstance(declarations, dict):
            continue
        visual = {}
        for key, value in declarations.items():
            if isinstance(value, dict) and value.get("type") == FeatureType.VISUAL.value:
                visual[key] = value
        if not visual:
            continue
        for later in steps[index + 1:]:
            if not isinstance(later, dict):
                visual = {}
                break
            name = later.get("registry_name")
            if name == "rename_observations_processor":
                later_config = later.get("config")
                rename_map = later_config.get("rename_map", {}) if isinstance(later_config, dict) else {}
                if not isinstance(rename_map, dict):
                    visual = {}
                    break
                visual = {rename_map.get(key, key): value for key, value in visual.items()}
            elif name not in passthrough:
                visual = {}
                break
        if not visual:
            continue
        rename_maps = [later.get("config", {}).get("rename_map", {}) for later in steps
                       if isinstance(later, dict) and later.get("registry_name") == "rename_observations_processor"]
        # A preceding nonempty rename may already bind the raw camera keys. Do
        # not turn stale normalizer declarations into a routing claim here.
        if set(visual) != set(expected) and not any(rename_maps):
            visual_mode = step_config.get("norm_map", {}).get("VISUAL", "IDENTITY")
            identity_passthrough = visual_mode == "IDENTITY"
            _add(report, "CAMERA_KEY_CONTRACT_MISMATCH", DiagnosticStatus.UNKNOWN,
                 "Policy camera keys disagree with the processor visual declarations without a resolving rename map. "
                 + ("IDENTITY normalization passes camera keys through without a preprocessor KeyError; "
                    "the policy backbone input_features lookup is a separate downstream contract."
                    if identity_passthrough else "Downstream policy input features require separate validation."),
                 {"step_index": index, "processor_visual_keys": sorted(visual),
                  "policy_visual_keys": sorted(expected), "rename_maps": rename_maps,
                  "unmapped_processor_keys": sorted(set(visual) - set(expected)),
                  "missing_policy_keys": sorted(set(expected) - set(visual)),
                  "normalizer_visual_mode": visual_mode,
                  "identity_camera_keys_pass_through": identity_passthrough,
                  "preprocessor_camera_keyerror_expected": False if identity_passthrough else None,
                  "downstream_contract_scope": "policy backbone input_features lookup",
                  "downstream_failure_verified": False,
                  "downstream_failure_condition": "The actual batch lacks a camera key expected by the policy. "
                  "Supplying config.json camera keys can satisfy the policy despite stale normalizer declarations.",
                  "policy_config": str(root / "config.json")},
                 _provenance(path, entry.get("registry_name")), severity=Severity.WARNING)
        for key in sorted(set(visual) | set(expected)):
            declared = visual.get(key)
            actual = expected.get(key)
            if declared is None or actual is None or tuple(declared.get("shape", [])) != actual.shape:
                advisory = (step_config.get("norm_map", {}).get("VISUAL", "IDENTITY") == "IDENTITY"
                            and (declared is None or actual is None))
                _add(report, "FEATURE_CONTRACT", DiagnosticStatus.UNKNOWN if advisory else DiagnosticStatus.CONFLICT,
                     f"Processor and policy visual feature declarations disagree for {key}.",
                     {"feature_key": key, "step_index": index,
                      "processor_feature": declared, "policy_feature": _feature_details(actual),
                      "processor_visual_keys": sorted(visual), "policy_visual_keys": sorted(expected),
                      "policy_config": str(root / "config.json"),
                      "identity_camera_keys_pass_through": advisory,
                      "downstream_failure_verified": False},
                     _provenance(path, entry.get("registry_name"), key),
                     severity=Severity.WARNING if advisory else None)


def _input_float_dtype(steps: list[Any], index: int) -> str | None:
    """Use the latest preceding native Device step that declares a float cast."""
    dtype = None
    for entry in steps[:index]:
        if isinstance(entry, dict) and _step_class(entry) is DeviceProcessorStep:
            step_config = entry.get("config")
            candidate = step_config.get("float_dtype") if isinstance(step_config, dict) else None
            if candidate is not None:
                dtype = candidate
    return dtype


def _legacy_declared_features(config: dict[str, Any]) -> list[tuple[str, str, PolicyFeature, NormalizationMode]]:
    """Read persisted legacy contracts; do not derive modes from tensor names."""
    if not isinstance(config, dict):
        raise ValueError("Legacy policy configuration must be an object")
    declared = []
    for group, side, feature_type in (
        ("normalize_inputs", "input", FeatureType.STATE),
        ("unnormalize_outputs", "output", FeatureType.ACTION),
    ):
        features = config.get(f"{side}_features")
        if isinstance(features, dict):
            mapping = config.get("normalization_mapping", {})
            if not isinstance(mapping, dict):
                raise ValueError("Legacy normalization_mapping must be a mapping")
            for key, feature in _feature_dict(features).items():
                if feature.type != feature_type or (feature_type is FeatureType.ACTION and key != "action"):
                    continue
                if len(feature.shape) != 1:
                    continue
                mode = NormalizationMode(mapping.get(feature.type.value, "IDENTITY"))
                declared.append((group, key, feature, mode))
        else:
            shapes = config.get(f"{side}_shapes", {})
            modes = config.get(f"{side}_normalization_modes", {})
            if not isinstance(shapes, dict) or not isinstance(modes, dict):
                raise ValueError("Legacy shapes and normalization modes must be mappings")
            for key, shape in shapes.items():
                # Old contracts name their state fields explicitly. The STATE
                # enum here carries the declared per-feature normalization mode.
                if group == "normalize_inputs" and key not in {"observation.state", "observation.environment_state"}:
                    continue
                if group == "unnormalize_outputs" and key != "action":
                    continue
                if not isinstance(shape, (list, tuple)) or len(shape) != 1:
                    continue
                if any(type(dim) is not int or dim <= 0 for dim in shape):
                    raise ValueError(f"Invalid legacy feature shape for {key}")
                raw_mode = modes.get(key)
                if not isinstance(raw_mode, str):
                    raise ValueError(f"Legacy normalization mode is undeclared for {key}")
                mode = NormalizationMode(raw_mode.upper())
                declared.append((group, key, PolicyFeature(feature_type, tuple(shape)), mode))
    return declared


def _verified_legacy_values(root: Path) -> tuple[Path, dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Verify downloaded tiny tensor slices against their pinned header metadata."""
    header_path = _local_file(root, "model.safetensors.header.json")
    _legacy_header_metadata(root)  # Validate the independent header descriptors.
    header = json.loads(header_path.read_text(encoding="utf-8"))
    metadata = json.loads(_local_file(root, "legacy_normalization.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Legacy normalization range metadata must be an object")
    if metadata.get("format") != "legacy-normalization-ranges-v1":
        raise ValueError("Unsupported legacy normalization range metadata format")
    if metadata.get("tensor_values_available") is not True:
        raise ValueError("Legacy range metadata does not provide tensor values")
    if header.get("complete") is not True:
        raise ValueError("Legacy range screening requires a complete original model header")
    if any(not isinstance(header.get(field), str) or not header[field] for field in ("repo_id", "revision")):
        raise ValueError("Legacy range screening requires a pinned repository and revision")
    if not re.fullmatch(r"[0-9a-f]{12}", header.get("header_sha256_prefix", "")):
        raise ValueError("Legacy model header has no valid provenance digest prefix")
    for field in ("repo_id", "revision", "total_size", "header_len", "header_sha256_prefix"):
        if metadata.get(field) != header.get(field):
            raise ValueError(f"Legacy range metadata disagrees with the model header: {field}")
    if Path(metadata.get("source_file", "")).name != header["source_file"]:
        raise ValueError("Legacy range metadata identifies a different source file")
    manifest_path = root / "download_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(_local_file(root, manifest_path.name).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Download manifest must be an object")
        for field in ("repo_id", "revision"):
            if manifest.get(field) is not None and manifest[field] != metadata.get(field):
                raise ValueError(f"Legacy range origin disagrees with the download manifest: {field}")
        subdir = manifest.get("checkpoint_subdir")
        if subdir is not None and metadata["source_file"] != (Path(subdir) / "model.safetensors").as_posix():
            raise ValueError("Legacy source file disagrees with the selected checkpoint directory")
    if metadata.get("tensor_file") != "legacy_normalization.safetensors":
        raise ValueError("Unexpected legacy normalization tensor filename")
    state_path = _local_file(root, metadata["tensor_file"])
    if state_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Legacy normalization artifact exceeds the local tiny-buffer budget")
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    if metadata.get("tensor_file_sha256") != digest:
        raise ValueError("Legacy normalization tensor-file checksum does not match")
    state = _read_tensors(state_path)
    descriptors = metadata.get("tensors")
    if not isinstance(descriptors, dict) or set(state) != set(descriptors):
        raise ValueError("Legacy tensor metadata and tensor-file keys disagree")
    header_len, total_size = header.get("header_len"), header.get("total_size")
    if type(header_len) is not int or not 0 < header_len <= 4 * 1024 * 1024:
        raise ValueError("Invalid original model header length")
    if type(total_size) is not int or total_size < 8 + header_len:
        raise ValueError("Invalid original model size")
    dtype_names = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64"}
    for key, tensor in state.items():
        descriptor = descriptors[key]
        original = header["tensors"].get(key)
        if (not isinstance(descriptor, dict) or original is None or key not in _legacy_buffer_keys([key])
                or tensor.ndim > 1 or tensor.dtype not in dtype_names):
            raise ValueError(f"Unsupported legacy normalization tensor: {key}")
        for field in ("dtype", "shape", "data_offsets"):
            if descriptor.get(field) != original[field]:
                raise ValueError(f"Legacy range descriptor disagrees with header for {key}: {field}")
        if descriptor["shape"] != list(tensor.shape) or descriptor["dtype"] != dtype_names[tensor.dtype]:
            raise ValueError(f"Legacy saved tensor shape/dtype disagrees with header for {key}")
        start, end = descriptor["data_offsets"]
        expected_range = [8 + header_len + start, 8 + header_len + end]
        if (descriptor.get("absolute_range") != expected_range or expected_range[1] > total_size
                or end - start != tensor.numel() * tensor.element_size()):
            raise ValueError(f"Legacy tensor byte range is invalid for {key}")
        payload = tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        if descriptor.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"Legacy tensor payload checksum does not match for {key}")
    return state_path, state, metadata, header


def _screen_legacy_buffers(root: Path, report: PreflightReport, threshold: float) -> set[Path]:
    """Apply inspected legacy Normalize formulas to verified state/action values."""
    metadata_path = root / "legacy_normalization.json"
    header_path = root / "model.safetensors.header.json"
    if not metadata_path.exists() or not header_path.exists():
        return set()
    numeric_checks = ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK", "DEAD_ACTION_DIMENSION")

    def unavailable(summary: str, details: dict[str, Any]) -> None:
        for check in numeric_checks:
            _add(report, check, DiagnosticStatus.UNKNOWN, summary, details,
                 _metadata_provenance(root, "legacy_normalization.json"))

    try:
        state_path, state, metadata, header = _verified_legacy_values(root)
        config = json.loads(_local_file(root, "config.json").read_text(encoding="utf-8"))
        features = _legacy_declared_features(config)
    except Exception as exc:
        unavailable(f"Legacy normalization slices could not be verified: {exc}",
                    {"coverage_source": "legacy_tensor_ranges", "tensor_values_verified": False})
        return set()
    active = [item for item in features if item[3] in _MODE_STATS]
    if not active:
        unavailable("No declared active legacy state/action normalization contract is available.",
                    {"coverage_source": "legacy_tensor_ranges", "tensor_values_verified": True})
        return set()
    # Multi-dataset pretraining buffers cannot be silently selected as target stats.
    datasets: dict[str, set[str]] = {}
    for key in header["tensors"]:
        match = re.fullmatch(r"unnormalize_outputs\.(.+)_buffer_action\.(mean|std)", key)
        if match:
            datasets.setdefault(match[1], set()).add(match[2])
    multi = sorted(name for name, stats in datasets.items() if stats == {"mean", "std"})
    if len(multi) >= 2 and any(group == "normalize_inputs" and key == "observation.state"
                               and mode is NormalizationMode.MEAN_STD for group, key, _, mode in active):
        missing_state_keys = [f"normalize_inputs.buffer_observation_state.{stat}" for stat in ("mean", "std")
                              if f"normalize_inputs.buffer_observation_state.{stat}" not in state]
        if missing_state_keys:
            summary = (
                "Checkpoint contains multi-dataset pretraining buffers without bound target dataset statistics; "
                "zero-shot execution without dataset-stat binding or fine-tuning will silently bypass normalization."
                if len(missing_state_keys) == 2 else
                "Checkpoint contains multi-dataset pretraining buffers with incomplete target observation statistics; "
                "bind complete dataset statistics before zero-shot normalization."
            )
            _add(report, "UNBOUND_BASE_MODEL_STATS", DiagnosticStatus.FAIL,
                 summary,
                 {"dataset_namespaces": multi, "target_dataset_stats_bound": False,
                  "legacy_embedded": True, "training_provenance_verified": False,
                  "missing_observation_state_stats": missing_state_keys,
                  "available_buffer_keys": sorted(state), "tensor_slice": []},
                 _provenance(state_path, "legacy_embedded", sorted(state)[0] if state else None, []))
            unavailable("Multi-dataset legacy buffers have no explicit target-dataset binding.",
                        {"coverage_source": "legacy_tensor_ranges", "tensor_values_verified": True,
                         "binding_required": True})
            return set()
    screened_features = []
    missing = []
    for group, key, feature, mode in active:
        aliases = {f"{key}.{stat}": f"{group}.buffer_{key.replace('.', '_')}.{stat}" for stat in _MODE_STATS[mode]}
        absent = [original for original in aliases.values() if original not in state]
        if absent:
            missing.append({"feature_key": key, "group": group, "required_buffer_keys": absent})
            continue
        canonical = {alias: state[original] for alias, original in aliases.items()}
        if any(tensor.numel() not in (1, math.prod(feature.shape)) for tensor in canonical.values()):
            missing.append({"feature_key": key, "group": group, "reason": "buffer_shape_contract_mismatch"})
            continue
        cls = NormalizerProcessorStep if group == "normalize_inputs" else UnnormalizerProcessorStep
        # Use native serialization for the declared feature/mode. Historical
        # Normalize hard-codes 1e-8; optimizer epsilon is unrelated.
        native = cls(features={key: feature}, norm_map={feature.type: mode}, eps=1e-8)
        native.load_state_dict(canonical)
        entry = {"registry_name": "normalizer_processor" if cls is NormalizerProcessorStep else "unnormalizer_processor",
                 "config": native.get_config()}
        first = len(report.findings)
        _screen_normalizer_state(report, state_path, entry, None, threshold, native.state_dict(), legacy=True)
        for finding in report.findings[first:]:
            if finding.provenance is None:
                continue
            original = aliases.get(finding.provenance.tensor_key)
            if original is None:
                continue
            descriptor = metadata["tensors"][original]
            finding.provenance.tensor_key = original
            finding.provenance.step_name = f"legacy_{group}"
            if finding.details.get("upper_buffer_key") in aliases:
                finding.details["upper_buffer_key"] = aliases[finding.details["upper_buffer_key"]]
            finding.details.update({
                "legacy_group": group, "source_tensor_key": original, "source_file": metadata["source_file"],
                "data_offsets": descriptor["data_offsets"], "absolute_range": descriptor["absolute_range"],
                "payload_sha256": descriptor["payload_sha256"], "screening_semantics": "legacy_normalize_formulas",
                "eps_source": "inspected_legacy_Normalize_constant", "legacy_execution_configuration_known": False,
                "fp16_execution_conditional": True, "tensor_values_verified": True,
                "legacy_formula_reference": "huggingface/lerobot@d694ea1d38e53aad20ce36bc014a16e3da185f5c:"
                "lerobot/common/policies/normalize.py",
                "legacy_min_max_epsilon_behavior": "always add eps to max - min",
                "checkpoint_device": config.get("device"), "checkpoint_use_amp": config.get("use_amp"),
            })
        screened_features.append({"feature_key": key, "group": group, "normalization_mode": mode.value,
                                  "original_tensor_keys": list(aliases.values())})
    if missing:
        unavailable("Some declared legacy features lack verified required tensor slices; numerical coverage is incomplete.",
                    {"coverage_source": "legacy_tensor_ranges", "missing_features": missing,
                     "screened_features": screened_features, "tensor_values_verified": True})
    elif screened_features:
        for check in numeric_checks:
            if not any(finding.check_id == check for finding in report.findings):
                _add(report, check, DiagnosticStatus.PASS,
                     "No matching hazard in verified buffers under inspected legacy Normalize formulas "
                     "and conditional half-buffer screening.",
                     {"coverage_source": "legacy_tensor_ranges", "screened_features": screened_features,
                      "screening_semantics": "legacy_normalize_formulas", "eps_source": "inspected_legacy_Normalize_constant",
                      "legacy_execution_configuration_known": False, "fp16_execution_conditional": True,
                      "tensor_values_verified": True}, _provenance(state_path, "legacy_embedded"))
    return {state_path} if screened_features else set()


def _validate_native(
    root: Path, path: Path, config: dict[str, Any], report: PreflightReport,
    policy: dict[PipelineFeatureType, dict[str, PolicyFeature]] | None,
) -> None:
    overrides = {}
    for index, entry in enumerate(config["steps"]):
        cls = _step_class(entry)
        if cls is None:
            _add(report, "FEATURE_CONTRACT", DiagnosticStatus.UNKNOWN,
                 "Feature transformation requires an unsupported step constructor; it was not invoked.",
                 {"step_index": index, "step": entry.get("registry_name", entry.get("class"))},
                 _provenance(path, index))
            return
        artifacts = [(entry.get("state_file"), False)] + [
            (artifact, True) for artifact in entry.get("artifacts", {}).values()
        ]
        for artifact, allow_directory in artifacts:
            if artifact is not None:
                artifact_path = _local_file(root, artifact, allow_directory=allow_directory)
                if artifact_path.suffix == ".safetensors" and _is_lfs_pointer(artifact_path):
                    _add(report, "FEATURE_CONTRACT", DiagnosticStatus.UNKNOWN,
                         "Native feature validation cannot read a Git LFS pointer as tensor state.",
                         {"step_index": index, "state_file": artifact,
                          "blocked_by": "GIT_LFS_POINTER_ARTIFACT"}, _provenance(artifact_path, index))
                    return
        if cls in (*_NORMALIZERS, DeviceProcessorStep):
            overrides[entry.get("registry_name", cls.__name__)] = {"device": "cpu"}
    pipeline = PolicyProcessorPipeline.from_pretrained(
        path, config_filename=path.name, local_files_only=True, overrides=overrides,
    )
    # Use native state and feature transformations, never a second processor implementation.
    pipeline.state_dict()
    initial = deepcopy(policy) if policy is not None else {
        PipelineFeatureType.OBSERVATION: {}, PipelineFeatureType.ACTION: {},
    }
    if policy is None:
        for step in pipeline.steps:
            for key, feature in getattr(step, "features", {}).items():
                kind = PipelineFeatureType.ACTION if feature.type == FeatureType.ACTION else PipelineFeatureType.OBSERVATION
                initial[kind].setdefault(key, feature)
    elif _role(path, config) == "pre":
        # Policy input declarations describe the result of preprocessing. Recover
        # the raw names before asking the native pipeline to transform them.
        for step in reversed(pipeline.steps):
            if isinstance(step, RenameObservationsProcessorStep):
                reverse = {new: old for old, new in step.rename_map.items()}
                initial[PipelineFeatureType.OBSERVATION] = {
                    reverse.get(key, key): value for key, value in initial[PipelineFeatureType.OBSERVATION].items()
                }
    transformed = pipeline.transform_features(initial)
    current = deepcopy(initial)
    for index, (entry, step) in enumerate(zip(config["steps"], pipeline.steps, strict=True)):
        name = entry.get("registry_name", type(step).__name__)
        for key, feature in getattr(step, "features", {}).items():
            kind = PipelineFeatureType.ACTION if feature.type == FeatureType.ACTION else PipelineFeatureType.OBSERVATION
            declared = current.get(kind, {}).get(key)
            if (declared is None and feature.type == FeatureType.VISUAL
                    and isinstance(step, _NORMALIZERS)
                    and step.norm_map.get(FeatureType.VISUAL, NormalizationMode.IDENTITY) is NormalizationMode.IDENTITY):
                # A stale visual declaration does not route or reject batch keys
                # in the native identity normalizer. The static camera advisory
                # above records this uncertainty without inventing a crash.
                continue
            if declared != feature and (declared is not None or policy is not None):
                _add(report, "FEATURE_CONTRACT", DiagnosticStatus.CONFLICT,
                     f"Processor and feature declarations disagree for {key}.",
                     {"feature_key": key, "step_index": index,
                      "processor_feature": _feature_details(feature), "expected_feature": _feature_details(declared),
                      "policy_config": str(root / "config.json") if policy else None},
                     _provenance(path, name, key))
        current = step.transform_features(current)
    if policy is not None:
        kinds = (PipelineFeatureType.ACTION,) if _role(path, config) == "post" else tuple(policy)
        for kind in kinds:
            for key, expected in policy[kind].items():
                actual = transformed.get(kind, {}).get(key)
                if actual != expected:
                    _add(report, "FEATURE_CONTRACT", DiagnosticStatus.CONFLICT,
                         f"Native transform_features output disagrees with config.json for {key}.",
                         {"feature_key": key, "expected_feature": _feature_details(expected),
                          "transformed_feature": _feature_details(actual)}, _provenance(path, tensor_key=key))


def _physical_semantics(root: Path, report: PreflightReport, metadata_path: Path | None = None) -> None:
    declarations: dict[str, list[tuple[Path, Any]]] = {field: [] for field in _SEMANTICS}
    paths = [root / "config.json"]
    explicit = Path(metadata_path).resolve() if metadata_path is not None else None
    if explicit is not None and explicit not in paths:
        paths.append(explicit)
    for path in paths:
        if not path.is_file() and not path.is_symlink():
            if path == explicit:
                _add(report, "PHYSICAL_SEMANTICS", DiagnosticStatus.FAIL,
                     "The explicitly selected metadata file does not exist.",
                     {"metadata_path": str(path), "required_coverage": True}, _provenance(path))
            continue
        if path == root / "config.json":
            try:
                path = _local_file(root, "config.json")
            except (OSError, ValueError) as exc:
                _add(report, "PHYSICAL_SEMANTICS", DiagnosticStatus.FAIL,
                     f"Policy metadata is unavailable within the checkpoint: {exc}",
                     {"metadata_path": str(path), "required_coverage": True},
                     Provenance(str(path), None, None, ""))
                continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON object")
        except (OSError, ValueError) as exc:
            if path == explicit:
                _add(report, "PHYSICAL_SEMANTICS", DiagnosticStatus.FAIL,
                     f"The explicitly selected metadata file is unreadable or invalid: {exc}",
                     {"metadata_path": str(path), "required_coverage": True}, _provenance(path))
            continue
        sources = [metadata, metadata.get("physical_semantics", {}), metadata.get("metadata", {})]
        for source in sources:
            if not isinstance(source, dict):
                continue
            for field in _SEMANTICS:
                value = source.get(field)
                if field == "temporal_observation_ordering" and value is None:
                    value = source.get("observation_order")
                if value is not None and value != "":
                    declarations[field].append((path, value))
    for field, entries in declarations.items():
        if not entries:
            _add(report, "PHYSICAL_SEMANTICS", DiagnosticStatus.UNKNOWN,
                 f"{field} is not explicitly declared.", {"field": field, "strict_exempt": True,
                                                           "required_coverage": False, "native_schema_field": False})
        else:
            conflict = any(value != entries[0][1] for _, value in entries[1:])
            _add(report, "PHYSICAL_SEMANTICS", DiagnosticStatus.CONFLICT if conflict else DiagnosticStatus.PASS,
                 f"Explicit declarations for {field} disagree." if conflict else f"{field} is explicitly declared.",
                 {"field": field, "value": entries[0][1],
                  "declarations": [{"file_path": str(path), "value": value} for path, value in entries]},
                 _provenance(entries[0][0]))


def run_preflight(checkpoint_dir: Path, std_warn_threshold: float = 1e-4, *,
                  metadata_path: Path | None = None) -> PreflightReport:
    """Inspect artifacts and bounded native probes without recorded samples or network.

    Buffer indices use flattened row-major order. Synthetic normalization
    boundaries are explicit diagnostic probes, not a claim about real observations.
    """
    if not math.isfinite(std_warn_threshold) or std_warn_threshold <= 0:
        raise ValueError("std_warn_threshold must be positive and finite")
    root = Path(checkpoint_dir).resolve()
    report = PreflightReport(str(root))
    if not root.is_dir():
        _add(report, "SCHEMA_GENERATION", DiagnosticStatus.FAIL, "Checkpoint directory does not exist.")
        _physical_semantics(root, report, metadata_path)
        return report
    pointer_paths = _find_lfs_pointers(root, report)
    pipelines = _discover(root, report)
    for finding in report.findings:
        if finding.check_id != "GIT_LFS_POINTER_ARTIFACT" or finding.provenance is None:
            continue
        references = []
        for config_path, config in pipelines:
            for index, entry in enumerate(config["steps"]):
                if not isinstance(entry, dict) or not isinstance(entry.get("state_file"), str):
                    continue
                candidate = (root / entry["state_file"]).resolve()
                if str(candidate) == finding.provenance.file_path:
                    references.append({"processor_file": config_path.name, "step_index": index,
                                       "step_name": entry.get("registry_name", entry.get("class"))})
        finding.details["processor_references"] = references
        if references:
            finding.details["step_index"] = references[0]["step_index"]
            finding.provenance.step_name = references[0]["step_name"]
    screened: set[Path] = set()
    coverage_gaps: list[dict[str, Any]] = []
    inspected_numeric_contract = (
        {_role(path, config) for path, config in pipelines} >= {"pre", "post"}
        and all(isinstance(entry, dict) and _normalization_base(entry) is not None
                for _, config in pipelines for entry in config["steps"])
    )
    for path, config in pipelines:
        for index, entry in enumerate(config["steps"]):
            if not isinstance(entry, dict):
                continue
            cls = _normalization_base(entry)
            if cls not in _NORMALIZERS:
                step_config = entry.get("config", {})
                if (isinstance(step_config, dict) and isinstance(step_config.get("features"), dict)
                        and isinstance(step_config.get("norm_map"), dict)):
                    try:
                        required = _active_buffer_keys(entry)
                    except (AttributeError, TypeError, ValueError):
                        required = ["unverified_contract"] if any(
                            mode != NormalizationMode.IDENTITY.value
                            for mode in step_config["norm_map"].values()
                        ) else []
                    if required:
                        coverage_gaps.append({
                            "processor_file": path.name, "step_index": index,
                            "reason": "uninspected_active_normalization_contract",
                            "step": entry.get("registry_name", entry.get("class")),
                            "declared_required_buffer_keys": required,
                            "constructor_invoked": False,
                        })
                continue
            try:
                required_keys = _active_buffer_keys(entry)
                if "state_file" not in entry:
                    state_path = path
                    state = _normalizer_contract(report, path, entry, index, {})
                    _screen_normalizer_state(report, path, entry, index, std_warn_threshold, state,
                                             _input_float_dtype(config["steps"], index))
                else:
                    state_path = _local_file(root, entry["state_file"])
                    if state_path in pointer_paths:
                        if required_keys:
                            coverage_gaps.append({"processor_file": path.name, "step_index": index,
                                                  "reason": "GIT_LFS_POINTER_ARTIFACT"})
                        continue
                    state = _screen_buffers(report, state_path, entry, index, std_warn_threshold,
                                            _input_float_dtype(config["steps"], index), config_path=path)
                missing = [key for key in required_keys if key not in state]
                if missing:
                    coverage_gaps.append({"processor_file": path.name, "step_index": index,
                                          "missing_buffer_keys": missing})
                elif required_keys:
                    screened.add(state_path)
            except Exception as exc:
                coverage_gaps.append({"processor_file": path.name, "step_index": index,
                                      "reason": f"{type(exc).__name__}: {exc}"})
                _add(report, "FEATURE_CONTRACT", DiagnosticStatus.FAIL, f"Cannot screen step buffers: {exc}",
                     {"step_index": index}, _provenance(path, index))
    _classify_unbound_base_stats(report)
    for config_path, config in pipelines:
        _screen_native_normalization_chain(root, config_path, config, report, std_warn_threshold)
    _screen_paired_relative_chain(root, report, std_warn_threshold)
    screened.update(_screen_legacy_buffers(root, report, std_warn_threshold))
    # A filename alone cannot establish which feature, mode, or direction uses an orphan.
    for path in sorted(root.glob("*normalizer*.safetensors")):
        if path.resolve() not in screened and path.resolve() not in pointer_paths:
            try:
                provenance = _provenance(_local_file(root, path.name))
            except (OSError, ValueError):
                provenance = Provenance(str(path), None, None, "")
            _add(report, "FEATURE_CONTRACT", DiagnosticStatus.UNKNOWN,
                 "Unreferenced normalization state has no declared feature, mode, or direction.",
                 {"state_file": str(path)}, provenance)
    for check in ("PATHOLOGICAL_STD_GAIN", "FP16_NON_FINITE_RISK"):
        if coverage_gaps:
            report.findings[:] = [finding for finding in report.findings
                                  if not (finding.check_id == check and finding.status is DiagnosticStatus.PASS)]
            _add(report, check, DiagnosticStatus.UNKNOWN,
                 "Some active normalization statistics were unavailable; numeric screening coverage is incomplete.",
                 {"screened_files": [str(path) for path in sorted(screened)],
                  "coverage_gaps": coverage_gaps, "all_active_steps_screened": False})
        if not any(f.check_id == check for f in report.findings):
            unbound = any(f.check_id == "UNBOUND_BASE_MODEL_STATS" for f in report.findings)
            complete = bool(screened or inspected_numeric_contract) and not unbound
            _add(report, check, DiagnosticStatus.PASS if complete else DiagnosticStatus.UNKNOWN,
                 "Target dataset statistics are unbound; unused pretraining buffers do not establish target normalization safety."
                 if unbound else "No matching hazards found in serialized normalization buffers." if screened else
                 "The inspected modern processor contracts require no active normalization statistics."
                 if inspected_numeric_contract else
                 "No readable serialized normalization buffers are available to screen.",
                 {"screened_files": [str(path) for path in sorted(screened)],
                  "all_active_steps_screened": complete,
                  "normalization_statistics_required": True if screened else False if inspected_numeric_contract else None})
    policy = None
    policy_path = root / "config.json"
    if policy_path.is_file() or policy_path.is_symlink():
        provenance = Provenance(str(policy_path), None, None, "")
        try:
            policy_path = _local_file(root, "config.json")
            provenance = _provenance(policy_path)
            config = json.loads(policy_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Policy config must be a JSON object")
            if "input_features" in config and "output_features" in config:
                policy = {PipelineFeatureType.OBSERVATION: _feature_dict(config["input_features"]),
                          PipelineFeatureType.ACTION: _feature_dict(config["output_features"])}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            _add(report, "FEATURE_CONTRACT", DiagnosticStatus.FAIL, f"Invalid policy feature declarations: {exc}",
                 provenance=provenance)
    start = len(report.findings)
    for path, config in pipelines:
        try:
            _visual_contract_without_constructors(root, path, config, report, policy)
        except (TypeError, ValueError, KeyError) as exc:
            _add(report, "FEATURE_CONTRACT", DiagnosticStatus.UNKNOWN,
                 f"Cannot compare static visual feature declarations: {exc}", provenance=_provenance(path))
        try:
            _validate_native(root, path, config, report, policy)
        except Exception as exc:
            _add(report, "FEATURE_CONTRACT", DiagnosticStatus.FAIL,
                 f"Native processor feature validation failed: {exc}", provenance=_provenance(path))
    if pipelines and not any(f.check_id == "FEATURE_CONTRACT" for f in report.findings[start:]):
        _add(report, "FEATURE_CONTRACT", DiagnosticStatus.PASS if policy is not None else DiagnosticStatus.UNKNOWN,
             "Native features agree with config.json." if policy is not None else
             "Processor declarations checked; config.json input/output features are unavailable for comparison.")
    _physical_semantics(root, report, metadata_path)
    return report
