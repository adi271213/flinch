<div align="center">
  <h1>lerobot-preflight</h1>
  <p>Audit LeRobot processor numerics and chronology before deploying a checkpoint—without a trusted baseline, ONNX export, or robot.</p>
  <p>
    <a href="docs/audit-evidence.md">Docs</a> ·
    <a href="https://github.com/adi271213/lerobot-preflight/issues/new?template=bug_report.md">Report Bug</a> ·
    <a href="https://github.com/adi271213/lerobot-preflight/issues/new?template=feature_request.md">Request Feature</a>
  </p>
  <p>
    <a href="https://github.com/adi271213/lerobot-preflight/actions/runs/44137080774"><img src="https://img.shields.io/badge/CI-6%2F6_at_5e9709d-brightgreen" alt="Python 3.12 matrix: six jobs passed at commit 5e9709d"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.12%2B-blue" alt="Requires Python 3.12 or later"></a>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0_prerelease-blue" alt="Source version 0.1.0 prerelease; not published to PyPI"></a>
  </p>
</div>

A native trace can pinpoint a finite but explosive coordinate:

```text
[EMPIRICAL_GAIN_EXPLOSION] EMPIRICAL_GAIN_EXPLOSION at preprocessor step 0, observation.state.
  step=0 | key=observation.state | dims=[9] | values=[134228.1875]
```

This terminal excerpt comes from the bundled `std[9] = 1.48e-6` fixture with a `+0.20` sample. The CI badge links to the exact previously verified commit; it does not certify later source changes. The repository is currently private, and no PyPI release or name reservation is claimed.

## Why lerobot-preflight?

A serialized checkpoint can load successfully while its processors skip statistics, amplify a small input perturbation, or use the wrong frame of an observation history. Comparing hashes alone cannot explain those numerical and contract failures. `lerobot-preflight` inspects the active buffers and runs bounded native processor paths to locate the failing step and coordinate.

- **Reference-free checks:** inspect active feature/mode/direction bindings, including missing dataset-stat namespaces and unused metadata.
- **Native execution evidence:** trace LeRobot processor boundaries for unchanged tensors, excessive growth, NaN and Inf; retain original step indices and dtype scope.
- **Chronology contracts:** compare saved observation delta indices with relative-action anchors, with explicit metadata overrides.
- **Baseline gates:** explain changes in composed observation/action mappings, offsets, envelopes, guards and execution horizons.
- **Bounded Hub acquisition:** fetch configurations, processor statistics, model headers and selected legacy tensor ranges without full neural weights.
- **Portable evidence:** JSON findings, SHA-256 provenance, Mermaid lineage and CI exit codes using the existing runtime dependencies.

Passing a processor audit does not certify hardware safety, neural weight identity, dataset quality, or robot behavior. No neural policy inference, simulator rollout, motor command, or trace-time network request runs.

## Architecture & System Design

```mermaid
flowchart LR
  ckpt[Local checkpoint] --> pre[Static preflight]
  ckpt --> temporal[Chronology audit]
  ckpt --> trace[Native numeric trace]
  sample[Optional NPZ sample] --> trace
  ckpt --> delta[Processor delta]
  baseline[Optional baseline] --> delta
  pre --> findings[Findings and provenance]
  temporal --> findings
  trace --> findings
  delta --> findings
  findings --> gate[Terminal report and exit gate]
  findings --> json[JSON report]
  ckpt --> graph[Feature and processor lineage]
```

The runtime uses LeRobot's `PolicyProcessorPipeline`, steps, state, feature transforms and hooks. Static and comparison checks also execute bounded native numeric probes on CPU. There is no parallel processor implementation, and uninspected checkpoint constructors are not imported to fill a coverage gap.

```text
src/lerobot_preflight/
  models.py              Findings, provenance and gate policy
  preflight.py           Artifact discovery, active stats and native probes
  trace.py               Offline sample execution and step evidence
  temporal.py            Observation-window and action-anchor contracts
  privilege_delta.py     Baseline mappings, guards and horizon comparison
  hub_scanner.py         Bounded Hub downloads and audit aggregation
  graph.py               Deterministic JSON/Mermaid lineage
  cli.py                 Report orchestration and console entrypoint
tests/fixtures/          Native checkpoint generators and healthy controls
scripts/                 Hub scanner and clean-distribution test harness
docs/                    Pinned audit evidence and scope limits
.github/                 Release matrix and contribution templates
```

## Quickstart

Use **Python 3.12** for the tested installation path, Git, and LeRobot **0.7.0 or later**. Package metadata allows Python 3.12+, while the hosted platform matrix tests 3.12. PyTorch and LeRobot are installed as dependencies; a GPU is not required. The first dependency download can take several minutes.

```bash
git clone https://github.com/adi271213/lerobot-preflight.git
cd lerobot-preflight
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
lerobot-preflight --version
```

Cloning the currently private repository requires access. In Windows PowerShell, create the environment with `py -3.12 -m venv .venv`, then use `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\lerobot-preflight.exe`; activation is optional.

Audit a public Hub checkpoint directly. This one ships a near-zero `observation.state` std:

```bash
lerobot-preflight --policy hf://jhseon-holiday/vqbet_xarm_holi_test@7dba32e8f65251b494491d7cefecfa8a6d5c2da7
```

```text
FAIL (2)
  [PATHOLOGICAL_STD_GAIN] Forward MEAN_STD normalization has gain up to 149856.71x at indices [4]. ...
    key=normalize_inputs.buffer_observation_state.std | dims=[4] | values=[6.6630414039536845e-06]
```

`hf://owner/name[@revision]` downloads only processor configs, processor statistics, the model header and bounded legacy normalization slices into a temporary directory, pins the resolved commit, and never downloads neural weights. Only public, ungated repositories are supported; for others, download the checkpoint and pass its local path. A bare `owner/name` that is not a local directory is rejected with a hint to add `hf://`.

Or run a local example from the checkout without network access:

```bash
python -c "from pathlib import Path; from tests.fixtures.make_bug_fixtures import make_bug_fixtures; make_bug_fixtures(Path('.preflight-demo'))"
lerobot-preflight --policy .preflight-demo/clean_act_modern \
  --sample .preflight-demo/clean_act_modern/sample_observation.npz
```

The generated clean processor fixture exits `0`; absent policy declarations and optional physical semantics remain `UNKNOWN`. The fixtures contain processor artifacts, not runnable neural policies. To reproduce the excerpt above, select `.preflight-demo/bug_5289_pathological_gain` and its sample instead: the expected rejecting exit code is `1`.

No `.env` file is required or loaded. See [.env.example](.env.example) for optional dependency offline switches.

## Usage & Real-World Examples

```bash
# Static artifacts, active normalization and chronology
lerobot-preflight --policy ./ckpt

# The same checks on a public Hub checkpoint, optionally pinned to a branch, tag or commit
lerobot-preflight --policy hf://lerobot/smolvla_base@main

# Trace a recorded numeric sample
lerobot-preflight --policy ./ckpt --sample ./obs.npz

# Compare candidate processor authority with production
lerobot-preflight --policy ./cand --baseline ./prod

# Export portable findings and feature lineage
lerobot-preflight --policy ./ckpt --report ./report.json --graph ./lineage.mmd

# Reject incomplete coverage and warning advisories
lerobot-preflight --policy ./ckpt --sample ./obs.npz --strict

# Explicitly select physical/chronology declarations
lerobot-preflight --policy ./ckpt --metadata ./preflight_metadata.json
```

Samples use exact feature keys and trailing shapes, with optional leading batch/time axes. A `(1, 1)` state cannot broadcast into a declared 14-coordinate feature. Include `action` in the NPZ or an adjacent `sample_action.npz` to exercise postprocessing; otherwise that trace coverage remains `UNKNOWN`.

Numeric VLA traces replace the inspected language tokenizer with a native identity step and mark language coverage `UNKNOWN`. No tokenizer assets are fetched. Token-dependent downstream behavior is not treated as verified. Checkpoints needing newer upstream processors may require a newer LeRobot installation; unsupported numeric paths remain `UNKNOWN`.

### Findings and exit policy

| Status | Meaning |
| --- | --- |
| `PASS` | The property holds within the inspected scope. |
| `FAIL` | Artifact or execution evidence establishes a violation. |
| `CONFLICT` | Artifact sources or declared contracts disagree. |
| `UNKNOWN` | Available evidence cannot complete the check. |

Normal mode exits **1** for `FAIL` or `CONFLICT` and **0** otherwise. `--baseline` also fails on required comparison `UNKNOWN`s. `--strict` additionally rejects coverage `UNKNOWN`s and `WARNING` advisories, including `PASS/WARNING`. Informational undeclared physical semantics are exempt from strict mode; shapes never establish radians, degrees, actuator identity or coordinate frames. Argument parsing errors exit **2**.

A metadata sidecar can declare `{"temporal_observation_ordering":"oldest_first","current_index":-1}`. Saved `state_observation_delta_indices` takes precedence over the generic observation window; explicitly selected metadata overrides inferred ordering. Native `state[:, 0]` is intentional for current-first windows such as `[0, 1]`. `TEMPORAL_ANCHOR_MISMATCH` is a cross-policy check for custom or migrated pipelines pairing that anchor with oldest-first `[-1, 0]` history—not a reproduced upstream bug in VLA-JEPA or LingBot-VA.

Numeric checks use active features, modes, direction and effective dtype. Unused metadata and `IDENTITY` statistics are skipped. Conditional legacy FP16 warnings do not claim immediate NaN under float32 buffers; actual active-dtype non-finite results block. Zero inverse action scale is reported as `DEAD_ACTION_DIMENSION`, without assuming whether the coordinate is an actuator or padding.

Missing statistics prevent a numeric `PASS`. An enabled ACTION normalization mode with neither declared action nor action stats has `UNKNOWN` coverage and fails strict mode. Local legacy numeric screening currently needs the scanner's verified header/tensor sidecars. Exports refuse to overwrite unrelated artifacts or symlinks; stable input files are assumed.

### Public checkpoint evidence and ecosystem

The [pinned evidence record](docs/audit-evidence.md) documents the 100-repository convenience sample: xArm's approximately **149,857×** gain, Pi0 Bridge's zero state spread, UR5 PegInsert's zero range, Git LFS pointers, and unbound base-model prerequisites. Legacy extraction reconstructed **242 tensors from 6,984 payload bytes**, excluding configs, headers and HTTP overhead. These are artifact findings, not observed robot failures, prevalence estimates, or general false-positive-rate measurements.

| Tool | Scope and overlap |
| --- | --- |
| [jashshah999/lerobot-doctor](https://github.com/jashshah999/lerobot-doctor) | Separate LeRobotDataset v2/v3 linter for metadata, episodes, videos, statistics and supplied/registered URDF limits. |
| [robotruth 0.1.6](https://github.com/mulkakhileshmj/robotruth/tree/8eb1704cf446239e3f03687c2ebd707cbab68e98) | Baseline/benchmark contract gate, including model weight-file identity checks that this tool does not implement. |
| [Tether 0.12.0](https://pypi.org/project/fastcrest-tether/0.12.0/) | ONNX runtime and embodiment validation; shares malformed action-stat coverage. Our comparison tested released components through disclosed adapters, not a full deployment. |
| **lerobot-preflight** | Native reference-free numerical diagnosis, active versus conditional precision risks, namespace bindings and chronology contracts. |

### Hub scanning

```bash
python scripts/scan_hub.py --targeted --limit 100 --output-dir ./hub-audit
python scripts/scan_hub.py --repo lerobot/smolvla_base --output-dir ./hub-audit
python scripts/scan_hub.py --cached-report ./hub-audit/hub_audit_report.json \
  --output-dir ./hub-audit
```

Downloading is the network-enabled stage; local diagnostics remain offline. Cached re-audits do not fetch missing tensor ranges unless `--fetch-legacy-stats` is requested. External Windows snapshots and their report must share a drive for relative cache locators; moving the complete bundle to another drive works. Ordinary `--policy` checks are unaffected.

## Configuration & Environment Reference

Checkpoint paths and audit decisions are explicit CLI arguments rather than hidden environment configuration. `--graph` supports `.json`, Mermaid `.mmd`, and fenced Markdown `.md`. JSON findings preserve non-finite values as explicit IEEE strings with typed annotations.

| Variable / flag | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--policy` | Path or `hf://` | None | Yes | Local checkpoint directory, or public Hub checkpoint `hf://owner/name[@revision]`. |
| `--sample` | Path | None | No | Numeric observation NPZ; optionally includes action. |
| `--baseline` | Path or `hf://` | None | No | Reference processor checkpoint for comparison; local or `hf://`. |
| `--report` | Path | None | No | New or recognized prior JSON report destination. |
| `--graph` | Path | None | No | JSON/Mermaid lineage destination. |
| `--metadata` | Path | None | No | Explicit semantics or chronology JSON sidecar. |
| `--std-threshold` | Float | `1e-4` | No | Positive finite small-spread threshold; not a physical safety limit. |
| `--strict` | Switch | `false` | No | Reject coverage gaps and warning advisories. |
| `--version` | Switch | `false` | No | Print source/package version and exit. |
| `-h`, `--help` | Switch | `false` | No | Print argument help and exit. |
| `HF_HUB_OFFLINE` | Dependency switch | Unset | No | Set to `1` for local-only Hugging Face access; blocks fresh Hub downloads. |
| `TRANSFORMERS_OFFLINE` | Dependency switch | Unset | No | Optional dependency offline mode; no dotenv loading is implemented. |

<details>
<summary>Hub scanner CLI options</summary>

| Flag | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--limit` | Integer | `30` | No | Maximum repositories, from 1 to 100. |
| `--targeted` | Switch | `false` | No | Balance tagged policy families and filter obvious asset listings. |
| `--repo` | Repeated string | None | No | Explicit public repository IDs; alternative to discovery/cache. |
| `--output-dir` | Path | Current directory | No | Hub report output directory. |
| `--report-name` | String | `hub_audit_report` | No | Plain filename stem for the JSON/Markdown pair. |
| `--snapshot-dir` | Path | `./.hub_audit_snapshots` | No | Storage root; each pinned snapshot gets a separate directory. |
| `--cached-report` | Repeated path | None | No | Re-audit previously pinned local snapshots. |
| `--fetch-legacy-stats` | Switch | `false` | No | Allow missing range reads during cached re-audits. |
| `--repo-file-evidence` | Path | None | No | Pinned file-list evidence for older cached reports. |
| `--std-threshold` | Float | `1e-4` | No | Active numeric screening threshold. |
| `-h`, `--help` | Switch | `false` | No | Print scanner argument help. |

</details>

## Development, Testing & Benchmarks

```bash
python -m pip install -e ".[test,dev]"
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m mypy
python -m pytest -q
python -m build
python -m twine check --strict dist/*.whl dist/*.tar.gz
```

Apply consistent formatting with `python -m ruff format src scripts tests`. Mypy checks all runtime modules in strict mode; third-party imports and dynamic JSON/tensor interfaces remain explicit boundaries, while native API and shape behavior are covered by pytest. No full third-party type-validation claim is made.

Tests generate native LeRobot checkpoints for silent bypass (#4415), low-spread gain (#5289), FP16 non-finite arithmetic (#4973), legacy buffers, and custom chronology mismatch (#5315), with healthy controls. The prior [six-job matrix at `5e9709d`](https://github.com/adi271213/lerobot-preflight/actions/runs/44137080774) passed **570 tests per job**, zero failures/errors/skips. That is historical validation of that commit; these later changes require their own run.

The [CI workflow](.github/workflows/ci.yml) checks lint, formatting, types, build and metadata, then installs the shared wheel/sdist pair in fresh Python 3.12 environments on Linux, macOS and Windows. Every cell runs the real CLI before installing pytest, verifies installed runtime bytes, and runs full packaged tests without a reference checkout or inherited `PYTHONPATH`. The [release harness](scripts/ci_release_check.py) can run those checks locally with separate empty work directories outside the checkout.

The 14-case competitor comparison and pinned Hub audit are correctness evidence, not throughput or adoption benchmarks. No universal runtime, installation-time, safety, or accuracy guarantee is claimed.

## Roadmap

- [x] Active-stat screening, native traces and exact finding provenance.
- [x] Composed baseline mappings and valid relative-action self-comparisons.
- [x] Bounded Hub header/tensor reads and deterministic lineage exports.
- [x] Cross-platform wheel/sdist release harness and project quality gates.
- [ ] Publish the first registry release after the release checklist is complete.
- [ ] Expand live relative-action checkpoint coverage beyond synthetic controls.
- [ ] Collect independent user reports and unseen-checkpoint regressions.

Unchecked items are planned work, not release-date commitments. The live 100-repository sample exercised no relative-action policy, so that contract coverage remains synthetic.

## Contributing & License

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branch/commit conventions, diagnostic evidence and PR checks. Use the issue templates for [bugs](https://github.com/adi271213/lerobot-preflight/issues/new?template=bug_report.md) and [feature requests](https://github.com/adi271213/lerobot-preflight/issues/new?template=feature_request.md). Report sensitive issues using [SECURITY.md](SECURITY.md).

Source is licensed under [MIT](LICENSE), copyright 2026 Aditya Jain. Third-party dependencies retain their own licenses. The source version is `0.1.0`; no registry release has been published.
