---
name: Feature request
about: Propose a processor check, compatibility improvement, or workflow change.
title: ""
labels: ""
assignees: ""
---

## Problem or workflow

Describe the checkpoint or deployment workflow that is difficult to audit today. Include the relevant policy family and LeRobot version when applicable.

## Proposed behavior

Explain what the tool should report or do. For a new diagnostic, identify the artifact or native processor behavior that supplies the evidence, and what should remain `UNKNOWN`.

## Example and alternatives

Provide a small example, public pinned artifact, or expected CLI output. Describe any workaround and why it is insufficient.

## Scope

Note compatibility requirements or tradeoffs. Requests should preserve read-only diagnostics: no motor commands, simulator rollouts, or trace-time network access.
