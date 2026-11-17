## Problem and change

Describe the verified problem, the resulting behavior, and any compatibility limit.

## Validation

List the commands you actually ran and their outcomes. For diagnostic changes, name the regression fixture and healthy control. Explain any checks that could not run.

## Review checklist

- [ ] Native LeRobot definitions and serialized behavior were inspected where relevant.
- [ ] Regression coverage verifies the finding, provenance, and CLI exit behavior.
- [ ] Diagnostics remain read-only and retain `UNKNOWN` for unsupported evidence or undeclared physical semantics.
- [ ] Documentation reflects the change; no credentials, cached checkpoints, model weights, or generated artifacts are included.

See the [contribution guide](https://github.com/adi271213/flinch/blob/main/CONTRIBUTING.md) for local checks and release validation.
