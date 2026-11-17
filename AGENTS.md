# AGENTS.md — Engineering Rules for lerobot-preflight

## 1. Ground-Truth Source Code Inspection (MANDATORY)
- NEVER guess `lerobot` class names, method signatures, or JSON keys from training memory.
- Before writing any code, inspect the installed `lerobot` package in `src/lerobot/processor/` (specifically `PolicyProcessorPipeline`, `DataProcessorPipeline`, `ProcessorStep`, `EnvTransition`, and `NormalizerProcessorStep`).
- All diagnostics must be READ-ONLY: zero simulator rollouts, zero motor commands, zero network calls during trace execution.

## 2. Architecture Constraints
- Do NOT invent a parallel processor abstraction. Hook directly into `PolicyProcessorPipeline` using its native `steps`, `state_dict()`, `transform_features()`, and `before_step_hooks`.
- Every check must return one of four strict statuses: `PASS`, `FAIL` (with exact step index, key, and offending tensor slice), `CONFLICT` (when two artifact sources disagree), or `UNKNOWN` (when physical semantics like units/frames are undeclared).
- Never infer physical units (radians vs. degrees) from tensor shapes alone.

## 3. Verification Discipline
- Every module must have a self-contained `pytest` test reproducing one of the 4 target bug classes (#4415 silent bypass, #5289 low-std gain explosion, #4973 FP16 NaN, #5315 temporal anchor mismatch) before moving to the next phase.
