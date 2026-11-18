# Audit evidence and limits

This is a maintained summary of historical executions, not a fresh scan or a claim
that later source revisions passed the same checks. The underlying local records
are the 14-case head-to-head report and the CI matrix report. Raw caches, samples
and verbose research archives are not shipped with the package.

## Executed competitor comparison

The historical **14-case fixture comparison**, dated **2026-11-16**, used native
LeRobot processor contracts and clean controls, not runnable neural policies or
physical robot deployments.

| Tool | Executed scope and observed coverage |
| --- | --- |
| [robotruth 0.1.6](https://github.com/mulkakhileshmj/robotruth/tree/8eb1704cf446239e3f03687c2ebd707cbab68e98) | Public contract comparison APIs and eight unmodified CLI smoke runs. Explicit synthetic benchmark declarations avoided uniform missing-hardware-metadata blocks. Changed-stat and horizon baseline gates received credit. A changed scalar weight-file control was rejected by its identity hash; our processor gate allowed it. Its runtime monitors were not benchmarked. |
| [Tether 0.12.0](https://pypi.org/project/fastcrest-tether/0.12.0/) | Four released Doctor components and its separate embodiment schema, through disclosed adapters from LeRobot stats/shapes. No full Tether CLI, ONNX policy export or deployment was tested. Zero/NaN/wrong-length action std coverage overlaps. Relative anchoring and native epsilon arithmetic were not representable by the adapter. |
| Flinch (then lerobot-preflight) | Standalone native numerical diagnosis and chronology checking, plus baseline comparisons. `std[9]=1.48e-6` with a `+0.20` sample produced `134,228.1875`; the native trace pinpointed coordinate `9`. Namespace bypass and actual FP16 NaN were localized. The custom oldest-first anchor failed despite a passing native round-trip; the current-first control passed. |

robotruth's baseline blocked the tiny-spread statistics change; its self-check
passed. Tether's tested components and schema passed those tiny-spread values.
This supports a distinction between numerical validity and baseline equality,
not an assertion that competitors provide no protection. The comparison does not
establish overall superiority, market demand, aggregate accuracy or hardware safety.
No neural weight-identity gate is implemented here.

## Historical clean-install CI

On **2026-11-17**, [GitHub Actions run 44108621945](https://github.com/adi271213/flinch/actions/runs/44108621945)
passed at commit `aef9f1aac28569c6185b0d5a37156c9c7a0d1170`. The repository/run is
private and requires access. These results apply to that prior revision, not
automatically to subsequent changes.

| Python 3.12 runner | Wheel | Source distribution |
| --- | ---: | ---: |
| Linux x86-64 | 523 passed | 523 passed |
| macOS ARM64 | 523 passed | 523 passed |
| Windows x86-64 | 523 passed | 523 passed |

That is **3,138 test executions**, with zero failures, errors or skips. One isolated
build supplied the same wheel/sdist bytes to all jobs. Every job used a fresh venv,
resolved dependencies, verified installed runtime bytes against its artifact,
executed the real console script without the reference checkout or `PYTHONPATH`,
and ran full packaged pytest against the installed code. All jobs used LeRobot
`0.7.0`; Linux/Windows used CPU Torch `2.15.0+cpu`, macOS Torch `2.15.0`.

Two earlier runs exposed Windows cache-locator and displayed-path redaction defects
that were fixed before this successful run. An earlier Linux dependency download
also failed transiently. This record does not establish other Python versions,
GPU execution, future dependency releases or a PyPI upload. External Windows Hub
snapshot/report bundles must reside on the same drive when report-relative cache
locators are written; moving the complete bundle together remains supported.
