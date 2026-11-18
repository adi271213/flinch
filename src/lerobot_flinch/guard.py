"""An optional native runtime guard for selected floating observation tensors.

``add_input_guard`` selects complete, actively normalized observation features.
Use explicit ``keys`` to choose another envelope deliberately. A saved pipeline records
the step by its import path, so LeRobot loads it without any extra import.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import torch
from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)

_ERROR_SAMPLE_SIZE = 8
# The statistics each normalization mode needs. The one shared definition in Flinch.
MODE_STATS = {
    NormalizationMode.MEAN_STD: ("mean", "std"),
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}


class InputEnvelopeError(ValueError):
    """Bounded evidence identifying a rejected observation tensor."""

    def __init__(self, key: str, indices: list[int], values: list[float], total_count: int, max_abs: float):
        self.key = key
        self.indices = list(indices[:_ERROR_SAMPLE_SIZE])
        self.values = list(values[:_ERROR_SAMPLE_SIZE])
        self.total_count = total_count
        self.max_abs = max_abs
        super().__init__(
            f"{key} has {total_count} value(s) outside ±{max_abs:g} in the selected input envelope "
            f"(flattened indices {self.indices}, values {self.values}). "
            "Check the saved normalization statistics and the selected envelope before running this policy."
        )


# Deliberately not registered by name: LeRobot then saves the full import path and imports it
# on load, so a deployment can't fail just because it forgot to import this module first.
@dataclass
class InputEnvelopeGuardStep(ObservationProcessorStep):  # type: ignore[misc]  # LeRobot is untyped.
    """Raise or clamp selected observations; non-finite values always raise.

    ``keys=None`` explicitly checks every floating observation. The helper below
    chooses active normalized keys instead. It marks guards it manages so later
    calls can update them without replacing unrelated, manually installed guards.
    """

    max_abs: float = 10.0
    on_violation: str = "raise"
    keys: list[str] | None = None
    managed_by_helper: bool = False
    # How many values clamp mode has rewritten since this step was created; not saved.
    clamp_count: int = field(default=0, init=False, repr=False, compare=False)
    _limits: dict[tuple[Any, ...], tuple[torch.Tensor, bool]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.max_abs, bool) or not isinstance(self.max_abs, Real):
            raise ValueError("max_abs must be a positive, finite real number")
        try:
            limit = float(self.max_abs)
        except (ValueError, OverflowError):
            raise ValueError("max_abs must be a positive, finite real number") from None
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("max_abs must be a positive, finite real number")
        self.max_abs = limit
        if self.on_violation not in ("raise", "clamp"):
            raise ValueError("on_violation must be 'raise' or 'clamp'")
        if self.keys is not None:
            if not isinstance(self.keys, list) or not self.keys:
                raise ValueError("keys must be None or a nonempty list of observation key strings")
            if any(
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in key)
                for key in self.keys
            ):
                raise ValueError(
                    "keys must contain nonempty observation key strings without surrounding whitespace or control characters"
                )
            if len(set(self.keys)) != len(self.keys):
                raise ValueError("keys must not contain duplicate observation names")
            self.keys = list(self.keys)
        if not isinstance(self.managed_by_helper, bool):
            raise ValueError("managed_by_helper must be a boolean")

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        for key, value in observation.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            if not self.selects(key):
                continue
            limit, limit_is_finite = self._limit(value)
            magnitude = value.abs()
            # One pass for the common case: NaN compares False, so it falls through too.
            if limit_is_finite and bool((magnitude <= limit).all()):
                continue
            finite = torch.isfinite(value)
            all_finite = bool(finite.all())
            outside = finite & (magnitude > limit)
            if all_finite and not bool(outside.any()):
                continue
            if self.on_violation == "clamp" and all_finite:
                result[key] = value.clamp(-limit, limit)
                if self.clamp_count == 0:
                    logging.getLogger(__name__).warning(
                        "Input guard clamped %s to ±%g; later clamps are counted in clamp_count.",
                        key,
                        self.max_abs,
                    )
                self.clamp_count += int(outside.sum())
                continue
            rejected = (~finite | outside).reshape(-1)
            total_count = int(rejected.sum())
            indices = rejected.nonzero().flatten()[:_ERROR_SAMPLE_SIZE]  # capped before .tolist()
            values = value.reshape(-1)[indices].tolist()
            raise InputEnvelopeError(key, indices.tolist(), values, total_count, self.max_abs)
        return result

    def _limit(self, value: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """``max_abs`` in the tensor's dtype and device, built once per (dtype, device, max_abs)."""
        cache_key = (value.dtype, value.device, self.max_abs)
        if cache_key not in self._limits:
            if len(self._limits) >= 8:  # A few dtype/device pairs in practice; never grow without bound.
                self._limits.clear()
            limit = torch.tensor(self.max_abs, device=value.device, dtype=value.dtype)
            if bool(torch.isfinite(limit)) and float(limit) > self.max_abs:
                # A rounded-up half/bfloat limit would admit values above the
                # configured real envelope. Use the adjacent inward value.
                limit = torch.nextafter(limit, torch.zeros_like(limit))
            self._limits[cache_key] = (limit, bool(torch.isfinite(limit)))
        return self._limits[cache_key]

    def selects(self, key: str) -> bool:
        """Whether this guard checks ``key`` (every key when ``keys`` is None)."""
        return self.keys is None or key in self.keys

    def get_config(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "on_violation": self.on_violation,
            "keys": list(self.keys) if self.keys is not None else None,
            **({"managed_by_helper": True} if self.managed_by_helper else {}),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _normalized_observation_keys(steps: list[Any], stop: int) -> list[str]:
    normalized: set[str] = set()
    structural = (
        DeviceProcessorStep,
        IdentityProcessorStep,
        AddBatchDimensionProcessorStep,
        RelativeActionsProcessorStep,
        AbsoluteActionsProcessorStep,
        InputEnvelopeGuardStep,
    )
    for step in steps[: stop + 1]:
        if type(step) is RenameObservationsProcessorStep:
            normalized = set(step.observation({key: None for key in normalized}))
            continue
        if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
            masked = type(step).__module__ == "lerobot.policies.molmoact2.processor_molmoact2" and type(
                step
            ).__name__ in {
                "MolmoAct2MaskedNormalizerProcessorStep",
                "MolmoAct2MaskedUnnormalizerProcessorStep",
            }
            if type(step) not in (NormalizerProcessorStep, UnnormalizerProcessorStep) and not masked:
                raise ValueError(
                    "Cannot infer observation envelopes through an uninspected normalization subclass; supply explicit keys"
                )
            state = step.state_dict()
            for key, feature in step.features.items():
                if feature.type is FeatureType.ACTION or (
                    step.normalize_observation_keys is not None and key not in step.normalize_observation_keys
                ):
                    continue
                mode = step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
                required = MODE_STATS.get(mode, ())
                if not required or any(f"{key}.{name}" not in state for name in required):
                    continue
                mask = state.get(f"{key}.mask") if masked else None
                if (
                    mask is not None
                    and mask.ndim == 1
                    and feature.shape
                    and mask.numel() == feature.shape[-1]
                ):
                    if not bool(mask.bool().any()):
                        continue
                    if not bool(mask.bool().all()):
                        raise ValueError(
                            f"{key} is only partially normalized; supply explicit keys to choose an envelope for the whole tensor"
                        )
                if isinstance(step, UnnormalizerProcessorStep):
                    normalized.discard(key)
                else:
                    normalized.add(key)
        elif type(step) not in structural and normalized:
            raise ValueError(
                "Cannot infer normalized observation keys through an uninspected processor step; supply explicit keys"
            )
    if not normalized:
        raise ValueError(
            "The preprocessor has no complete active normalized observation features to guard; supply explicit keys"
        )
    return sorted(normalized)


def add_input_guard(
    preprocessor: PolicyProcessorPipeline,
    *,
    max_abs: float = 10.0,
    on_violation: str = "raise",
    keys: list[str] | None = None,
) -> PolicyProcessorPipeline:
    """Configure one managed guard after the last normalizer, preserving the pipeline.

    Default selection excludes IDENTITY, excluded and unbound observation stats.
    Explicit ``keys`` deliberately choose another envelope. Repeated calls update
    this helper's single guard; manual or multiple guards require explicit caller
    changes rather than silently dropping existing runtime constraints.
    """
    guard = InputEnvelopeGuardStep(
        max_abs=max_abs, on_violation=on_violation, keys=keys, managed_by_helper=True
    )
    original = list(preprocessor.steps)
    existing = [step for step in original if isinstance(step, InputEnvelopeGuardStep)]
    if len(existing) > 1 or (existing and not existing[0].managed_by_helper):
        raise ValueError(
            "Existing manual or multiple input guards are ambiguous; configure them explicitly before using add_input_guard"
        )
    steps = [step for step in original if not isinstance(step, InputEnvelopeGuardStep)]
    normalizers = [index for index, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)]
    if not normalizers:
        raise ValueError("The preprocessor has no NormalizerProcessorStep to guard")
    position = normalizers[-1] + 1
    if keys is None:
        guard.keys = _normalized_observation_keys(steps, normalizers[-1])
    if existing:
        current = existing[0]
        current.max_abs, current.on_violation, current.keys = guard.max_abs, guard.on_violation, guard.keys
        guard = current
    steps.insert(position, guard)
    # Native save/state_dict derive filenames from current indices. A loaded
    # pipeline reuses its preserved tuple solely by length, so even an equal-size
    # guard relocation must invalidate that old index-to-filename association.
    if hasattr(preprocessor, "_serialized_state_filenames"):
        preprocessor._serialized_state_filenames = None
    preprocessor.steps = steps
    return preprocessor
