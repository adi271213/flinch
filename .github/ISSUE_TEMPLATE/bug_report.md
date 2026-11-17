---
name: Bug report
about: Report an incorrect finding, missing coverage, or execution failure.
title: ""
labels: ""
assignees: ""
---

## Problem

Describe the observed behavior and what you expected instead. Include the check ID, status, and CLI exit code when relevant.

## Reproduction

Provide the exact command and a minimal synthetic fixture or public checkpoint pinned to a revision. Include any explicit metadata and sample shapes/dtypes needed to reproduce the issue.

```text
flinch --policy ./checkpoint
```

## Environment

- flinch version (`flinch --version`):
- LeRobot version or commit:
- Python / PyTorch versions:
- Operating system:

## Evidence

Include the relevant finding or error output. Remove credentials, personal paths, and private tensor data before posting. Do not attach neural weight files. For a security vulnerability, follow [SECURITY.md](https://github.com/adi271213/flinch/blob/main/SECURITY.md).
