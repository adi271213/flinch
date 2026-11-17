# Audit evidence and limits

This is a maintained summary of historical executions, not a fresh scan or a claim
that later source revisions passed the same checks. The underlying local records
are `hub_100_report.json` / `.md`, the 14-case head-to-head report, and the CI matrix
report. Raw checkpoint caches, samples and verbose research archives are not shipped
with the package. The pinned public artifacts below identify the inputs independently
of those local paths.

## Public checkpoint audit

The cached 100-repository report was generated on **2026-11-16**. Its original live
selection used repositories tagged `lerobot`, with family searches for ACT,
Diffusion, SmolVLA, Pi0, VQ-BeT and TD-MPC, plus seed searches and tagged fallback.
This was a targeted convenience sample, **not a prevalence or accuracy estimate**.
All 100 revisions were pinned. The cached regeneration used no network.

| Classification | Repositories |
| --- | ---: |
| Modern policy | 42 |
| Legacy policy | 55 |
| Mixed legacy/modern policy | 1 |
| Non-policy repository | 1 |
| Gated or private | 0 |
| Unknown | 1 |

The downloader reconstructed **242 legacy normalization tensors from 6,984 bytes
of tensor payloads** across 56 legacy/mixed snapshots. That byte count excludes
configs, processor files, HTTP overhead and model headers; it is not the total
network cost. No full neural weight files were downloaded for the audit.

### Four defect or numerical-risk signatures

The following are artifact findings and formula/probe evidence. They do not
establish an observed robot failure or identify a coordinate's physical meaning.
Legacy embedded buffers have no modern processor step index. SHA-256 prefixes in
the table identify the locally reconstructed normalization safetensors container,
except for the Git LFS row, which identifies the pointer text itself.

| Pinned checkpoint | Exact evidence | Interpretation |
| --- | --- | --- |
| [jhseon-holiday/vqbet_xarm_holi_test](https://huggingface.co/jhseon-holiday/vqbet_xarm_holi_test/tree/7dba32e8f65251b494491d7cefecfa8a6d5c2da7) | Saved `MEAN_STD`; `normalize_inputs.buffer_observation_state.std[4] = 6.6630414039536845e-6`; container SHA `7d24b59de47a`. | `1 / (std + 1e-8) = 149,856.7078×`; synthetic `+0.20` produces `29,971.341796875`. The saved mode overrides the current VQ-BeT default. |
| [HaomingSong/lerobot-pi0-bridge](https://huggingface.co/HaomingSong/lerobot-pi0-bridge/tree/93c2840a2c68c337da0d472a2f52d1fd8d62132a) | Saved `MEAN_STD`; `normalize_inputs.buffer_observation_state.std[6] = 0`; container SHA `128ca7bfcb39`. | With `eps=1e-8`, float32 maps the mean to `0` and a `+0.20` perturbation to `20,000,000`: approximately `1e8×` gain. |
| [iantc104/Diffusion-RPL-UR5-PegInsert-v0](https://huggingface.co/iantc104/Diffusion-RPL-UR5-PegInsert-v0/tree/0703f194c88b43cc1b704c682f27fe554e1b830e) | Saved `MIN_MAX`; `normalize_inputs.buffer_observation_environment_state.min/max` are equal at indices `[11,12,13,14,15,16,17]`; container SHA `fb006f50ebd8`. | Legacy `2 / (max - min + 1e-8)` has approximately `2e8×` gain. The training constant maps to `-1`; `+0.20` probes produce `40,000,000` or `40,000,008` due to float32 rounding. |
| [y1y2y3/smolvla_base1](https://huggingface.co/y1y2y3/smolvla_base1/tree/3dd6295344c2585dc096ac1d862d1ec4017425a0) | Preprocessor step `5` and postprocessor step `0` reference 129-byte Git LFS pointer files; pointer SHA `809f43bfa90a`, declared object SHA starts `490ab239d96e`. | Pointer text is not tensor data; native safetensors loading cannot use these processor state files. |

The [xArm `175000` snapshot](https://huggingface.co/jhseon-holiday/vqbet_xarm_holi_test_175000/tree/9dd03107e5461e85afdf1849c17266b6a84443a4)
has the same statistics. It is a second affected repository, not an independent
numerical discovery. The xArm state-std tensor payload SHA starts `62c684b883e8`.

The equal UR5 min/max values at indices 11–17 are respectively
`[-0.01269999984651804, 1, 0, 0, 0, 1, 0]`. The reported legacy formulas use the
inspected historical LeRobot implementation; checkpoint artifacts do not pin every
deployment's code or precision.

**Conditional FP16 scope:** legacy `Normalize` keeps float32 statistics, so an FP16
observation alone is promoted by float32-buffer arithmetic. The zero-spread cases
above do not establish immediate NaN/Inf under that default. Half-casting the
legacy module/buffers, or migrating to modern `NormalizerProcessorStep`, which casts
statistics to the input dtype, can make the epsilon denominator zero in FP16.
The two legacy FP16 findings are therefore nonblocking `PASS/WARNING` advisories
under the recorded default arithmetic; `--strict` rejects those advisories. Pi0
Bridge explicitly saves CPU execution with `use_amp=false`.

### Base-model deployment prerequisites

These are separate from the four defect/numerical-risk signatures:

| Pinned checkpoint | Required binding |
| --- | --- |
| [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base/tree/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6) | Preprocessor step `5` and postprocessor step `0` expect `action.mean/std`, but the state files contain `so100{,-blue,-red}.buffer.action.*`. Required `observation.state.mean/std` is absent. Both state-file SHA prefixes are `490ab239d96e`. Bind target dataset statistics or fine-tune before execution. |
| [lerobot/pi0_old](https://huggingface.co/lerobot/pi0_old/tree/e4ed526af508e58f6008b29e9e48f1098278fdb5) | Preprocessor step `5` and postprocessor step `0` require state/action statistics, but the JSONs omit `state_file` and no processor statistics are supplied. JSON SHA prefixes: `b108f4c0e6ab` and `a681a9837016`. External statistics are required. |

The report groups these upstream bases and nine community SmolVLA copies/migrations
as one stat-binding prerequisite family: 11 repositories. `_INTENTIONAL_BASE_REPOS`
explicitly names `lerobot/smolvla_base` and `lerobot/pi0_old`; repository IDs and
shared stat payloads support that classification. They do not prove weight identity
or fine-tuning history. Unbound execution still blocks the deployment gate, but is
not evidence of defective pretrained weights.

Across overlapping categories, the report records **five artifact-finding families
in 15 repositories**: four defect/numerical-risk signatures plus that prerequisite
family. Fifty-five legacy schemas are migration/version-compatibility findings,
not 55 independent checkpoint defects.

Four camera-declaration advisories are `UNKNOWN/WARNING`. With `IDENTITY` visual
normalization, supplied camera keys pass through the preprocessor; stale declared
keys do not themselves prove a preprocessor `KeyError`. Missing policy camera names
can instead surface downstream when the policy selects its configured inputs.

### Coverage gaps

The historical report has **13/100 numeric-buffer coverage gaps**, **9/100 native
normalization-chain probe gaps**, and **76/100 temporal `UNKNOWN`s**. The remaining
temporal scope is **0 relative-action policies**, **23 non-relative policies**, and
one non-policy repository outside the check. No live relative-action policy was
exercised in this sample.

`TEMPORAL_ANCHOR_MISMATCH` is a cross-policy chronology contract check for custom or
migrated pipelines. Native `state[:, 0]` is intentional for current-first windows
whose delta indices start at `0`; it conflicts with oldest-first `[-k, ..., 0]`
history when the current state is last. These results do not reproduce an upstream
bug in the in-tree VLA-JEPA or LingBot-VA policies. Missing units, frames, chronology
or active statistics remain explicit coverage limits.

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
