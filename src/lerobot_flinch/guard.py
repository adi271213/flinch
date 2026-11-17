"""A runtime guard that stops exploded policy inputs before they reach the policy.

Static checks find stats that *could* blow up an input. This step catches it when it
actually happens: put it right after the normalizer in your preprocessor, and any
normalized observation value that is non-finite or larger than ``max_abs`` either
raises (the default) or is clamped.

    from lerobot_flinch.guard import add_input_guard
    preprocessor = add_input_guard(preprocessor, max_abs=10.0)

Importing ``lerobot_flinch.guard`` registers the step, so a pipeline saved with it can be
loaded again with ``PolicyProcessorPipeline.from_pretrained``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
)


class InputEnvelopeError(ValueError):
    """A normalized observation left the allowed envelope."""


@dataclass
@ProcessorStepRegistry.register(name="flinch_input_guard")
class InputEnvelopeGuardStep(ObservationProcessorStep):  # type: ignore[misc]  # LeRobot is untyped.
    """Raise or clamp when a normalized observation is non-finite or exceeds ``max_abs``.

    ``max_abs`` is in normalized units: with MEAN_STD, 10 means ten standard deviations
    from the training mean. Non-finite values always raise, because clamping would hide them.
    """

    max_abs: float = 10.0
    on_violation: str = "raise"  # "raise" or "clamp"
    keys: list[str] | None = None  # None checks every floating-point observation tensor.

    def __post_init__(self) -> None:
        if not (self.max_abs > 0 and self.max_abs != float("inf")):
            raise ValueError("max_abs must be a positive, finite number")
        if self.on_violation not in ("raise", "clamp"):
            raise ValueError("on_violation must be 'raise' or 'clamp'")

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        for key, value in observation.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            if self.keys is not None and key not in self.keys:
                continue
            finite = torch.isfinite(value)
            outside = finite & (value.abs() > self.max_abs)
            if bool(finite.all()) and not bool(outside.any()):
                continue
            if self.on_violation == "clamp" and bool(finite.all()):
                observation[key] = value.clamp(-self.max_abs, self.max_abs)
                continue
            bad = (~finite | outside).reshape(-1).nonzero().flatten()
            worst = value.reshape(-1)[bad[:8]].tolist()
            raise InputEnvelopeError(
                f"{key} has {bad.numel()} value(s) outside ±{self.max_abs:g} after normalization "
                f"(flattened indices {bad[:8].tolist()}, values {worst}). "
                "Check the saved normalization stats before running this policy."
            )
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"max_abs": self.max_abs, "on_violation": self.on_violation, "keys": self.keys}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def add_input_guard(
    preprocessor: PolicyProcessorPipeline, *, max_abs: float = 10.0, on_violation: str = "raise"
) -> PolicyProcessorPipeline:
    """Insert the guard right after the last normalizer of a preprocessor, in place."""
    steps = list(preprocessor.steps)
    normalizers = [i for i, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)]
    if not normalizers:
        raise ValueError("The preprocessor has no NormalizerProcessorStep to guard")
    steps.insert(normalizers[-1] + 1, InputEnvelopeGuardStep(max_abs=max_abs, on_violation=on_violation))
    preprocessor.steps = steps
    return preprocessor
