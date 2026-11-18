"""What a near-zero std does to a control loop, and how one line of Flinch stops it.

Run:  python examples/guard_demo.py

Everything here is simulated: the "robot" is a 4-joint state with 1e-3 of sensor noise,
and the "policy" is a stand-in that outputs a command proportional to its normalized
input. Joint 2's saved std is 1.5e-6: it barely moved in the training data.
"""

from __future__ import annotations

import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    batch_to_transition,
    transition_to_batch,
)

from lerobot_flinch.guard import InputEnvelopeError, add_input_guard

KEY = "observation.state"
MEAN = torch.tensor([0.10, -0.40, 0.25, 0.00])
STD = torch.tensor([0.30, 0.25, 1.5e-6, 0.20])  # joint 2 barely moved in the training data
STEPS = 5


def preprocessor() -> PolicyProcessorPipeline:
    normalizer = NormalizerProcessorStep(
        features={KEY: PolicyFeature(FeatureType.STATE, (4,))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={KEY: {"mean": MEAN, "std": STD}},
    )
    return PolicyProcessorPipeline(
        steps=[normalizer], to_transition=batch_to_transition, to_output=transition_to_batch
    )


def run(pipeline: PolicyProcessorPipeline, label: str) -> None:
    print(f"\n{label}")
    noise = torch.Generator().manual_seed(0)
    for step in range(STEPS):
        state = MEAN + torch.randn(4, generator=noise) * 1e-3  # the robot is holding still
        try:
            normalized = pipeline({KEY: state[None]})[KEY][0]
        except InputEnvelopeError as error:
            # The guard only raises; stopping here is this demo's choice. A real controller must
            # catch this and take its own safe action (hold, brake, hand over).
            print(
                f"  step {step}: guard raised before the policy saw the input; this demo stops sending commands\n    {error}"
            )
            return
        command = 0.05 * normalized  # stand-in policy
        print(f"  step {step}: largest joint command {command.abs().max().item():8.2f}")


if __name__ == "__main__":
    run(preprocessor(), "Without the guard (sensor noise of 1e-3 on a joint whose std is 1.5e-6):")
    run(add_input_guard(preprocessor()), "With the guard (one line: add_input_guard(preprocessor)):")
