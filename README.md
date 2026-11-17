# lerobot-preflight

Reference-free numerical processor auditor and chronology/contract linter for LeRobot checkpoints.

Inspect serialized normalization, trace local processor tensors, and compare candidate processor mappings with a baseline before deployment. No trusted baseline or ONNX export is required for standalone checks. Diagnostics run on CPU without neural policy inference, simulator rollouts, motor commands, or trace-time network access.

This is a checkpoint processor audit, not a hardware-safety certification. It does not verify neural weight identity, dataset quality, or robot behavior.

## Install

Requires Python 3.12+ and LeRobot 0.7.0 or later. This is a source-installable prerelease; no PyPI publication is claimed.

From this checkout on Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
lerobot-preflight --version
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\lerobot-preflight.exe --version
```

PyTorch and LeRobot provide the runtime dependencies; a GPU is not required. Checkpoints with newer upstream processors may require a newer LeRobot installation. Uninspected custom processors remain `UNKNOWN`; the tool does not import arbitrary checkpoint code to complete an audit.

## Use

```bash
# Static artifacts, normalization buffers and temporal contracts
lerobot-preflight --policy ./ckpt

# Native numeric trace with a recorded observation
lerobot-preflight --policy ./ckpt --sample ./obs.npz

# Processor authority changes between production and candidate
lerobot-preflight --policy ./cand --baseline ./prod

# Machine-readable evidence and a Mermaid lineage diagram
lerobot-preflight --policy ./ckpt --report ./report.json --graph ./lineage.mmd

# Reject incomplete coverage and warning advisories
lerobot-preflight --policy ./ckpt --sample ./obs.npz --strict
```

`--graph` accepts `.json` for a property graph, `.mmd` for Mermaid, or `.md` for a fenced Mermaid diagram. Report and graph exports refuse to overwrite unrelated artifacts or symlinks. `--std-threshold` sets the small-spread screening threshold (default `1e-4`); it is not a physical safety limit.

Samples use exact checkpoint feature keys and trailing shapes, with optional leading batch/time axes. A sample `observation.state` shaped `(1, 1)` cannot broadcast into a declared 14-coordinate vector. Include `action` in the NPZ or an adjacent `sample_action.npz` to exercise postprocessing. Without it, postprocessor trace coverage is `UNKNOWN`.

Numeric VLA traces replace only the inspected language tokenizer with a native identity step and report the missing language coverage explicitly. No tokenizer assets are fetched. Token-consuming downstream steps are not treated as verified.

## Findings and exit codes

Every finding includes a check ID, status, severity, evidence, and artifact provenance where available. JSON uses explicit IEEE strings and typed annotations for non-finite values.

| Status | Meaning |
| --- | --- |
| `PASS` | The property holds within the inspected scope. |
| `FAIL` | Artifact or execution evidence establishes a violation. |
| `CONFLICT` | Artifact sources or declared contracts disagree. |
| `UNKNOWN` | The available evidence cannot complete the check. |

The CLI exits **1** for `FAIL` or `CONFLICT`, and **0** otherwise. With `--baseline`, required comparison coverage that is `UNKNOWN` also exits **1**. `--strict` additionally rejects coverage `UNKNOWN`s and `WARNING` advisories, including `PASS/WARNING`. Argument errors exit **2**.

Undeclared physical units and frames remain informational `UNKNOWN` notes and do not independently fail strict mode. Shapes never establish radians, degrees, actuator identity, or physical coordinate frames. To supply explicit semantics or override chronology, opt into a JSON sidecar:

```bash
lerobot-preflight --policy ./ckpt --metadata ./preflight_metadata.json
```

For example: `{"temporal_observation_ordering":"oldest_first","current_index":-1}`. Arbitrary metadata files are not automatically trusted.

## What is checked

| Area | Inspected behavior |
| --- | --- |
| Static normalization | Active features, declared modes, stat namespaces, missing/invalid buffers, small spreads, zero action scales, and intermediate dtype overflow. Unused metadata and `IDENTITY` statistics are excluded. |
| Offline trace | Missing inputs, exact trailing shapes, silent bypass, activation growth, and NaN/Inf at native processor boundaries. |
| Chronology contract | Saved observation delta indices versus the relative-action anchor, mask/shape compatibility, and scale-aware forward/inverse probes. |
| Baseline comparison | Composed observation/action mappings, offsets, envelopes, intermediate precision failures, guards, masks, and execution horizons. |
| Lineage graph | Feature paths, processor order, bound/orphan statistics, gains, dtypes, renames, contract drift, and temporal anchors. |

The native `RelativeActionsProcessorStep` intentionally uses `state[:, 0]`. It is valid for current-first windows such as `[0, 1]` used by in-tree VLA-JEPA and LingBot-VA. `TEMPORAL_ANCHOR_MISMATCH` (#5315) is a **cross-policy chronology contract check** for custom or migrated pipelines paired with oldest-first windows such as `[-1, 0]`, where the current frame is last. Saved `state_observation_delta_indices` takes precedence over the generic observation window; explicit metadata overrides inferred ordering.

Normalization checks respect transform direction and effective dtype. Forward transforms divide by spread; inverse action transforms with zero scale get `DEAD_ACTION_DIMENSION`. Legacy float32 buffers can yield finite but extreme perturbation gains. Hypothetical legacy half-precision risks are conditional warnings, not claims of immediate NaN under float32. Modern normalizers adapt stats to input dtype; active half-precision failures are blocking.

Missing active statistics prevent a numeric `PASS`. In particular, a preprocessor that enables `ACTION` in `norm_map` but declares no action feature or statistics has `UNKNOWN` numeric coverage, which fails `--strict`. Local legacy model files establish schema compatibility, but numeric screening currently requires the scanner's verified header/tensor sidecars. Unsupported state-dependent or non-affine comparisons also retain `UNKNOWN` coverage.

## Evidence and related tools

The [audit evidence record](docs/audit-evidence.md) summarizes pinned public checkpoint findings, the 14-case comparison, and the prior clean-install matrix. These are artifact and fixture results, not observed robot failures or a general false-positive-rate measurement.

| Tool | Scope and overlap |
| --- | --- |
| [jashshah999/lerobot-doctor](https://github.com/jashshah999/lerobot-doctor) | Separate pre-training LeRobotDataset v2/v3 linter: metadata, parquet episodes, videos, statistics, and supplied/registered URDF limits. |
| [robotruth 0.1.6](https://github.com/mulkakhileshmj/robotruth/tree/8eb1704cf446239e3f03687c2ebd707cbab68e98) | Baseline/benchmark contract gate. Also verifies model weight-file identity hashes, which this tool does not gate. |
| [Tether 0.12.0](https://pypi.org/project/fastcrest-tether/0.12.0/) | ONNX-based runtime and embodiment validation. Shares zero/NaN action-stat coverage. Our comparison exercised released components through documented adapters, not a complete deployment. |
| **lerobot-preflight** | Native, reference-free numerical diagnosis and chronology checking: finite-but-explosive gains, namespace bypasses, active versus conditional precision risks, and bounded Hub tensor-range auditing. |

## Hub auditing

The source scanner downloads bounded configuration files, processor statistics, model headers, and selected legacy state/action tensor slices. It never downloads full neural model weights. Network access occurs during downloading, before local diagnostics.

```bash
python scripts/scan_hub.py --targeted --limit 100 --output-dir ./hub-audit
python scripts/scan_hub.py --repo lerobot/smolvla_base --output-dir ./hub-audit
python scripts/scan_hub.py --cached-report ./hub-audit/hub_audit_report.json \
  --output-dir ./hub-audit
```

Cached re-audits use local snapshots unless `--fetch-legacy-stats` is explicitly requested. Reports distinguish policy generation, base-model prerequisites, numeric/file risks, advisories, and coverage gaps. The targeted selection is a convenience sample, not a prevalence estimate.

External snapshots use report-relative cache paths. On Windows, keep external snapshots and their Hub report on the same drive; moving the complete bundle to another drive works. This restriction does not apply to ordinary `--policy` CLI checks.

## Development and architecture

```bash
python -m pip install -e ".[test,dev]"
python -m ruff check src scripts tests
python -m pytest -q
python -m build
python -m twine check --strict dist/*.whl dist/*.tar.gz
```

| Module | Responsibility |
| --- | --- |
| `models.py` | Findings, provenance, strict JSON, and gate exit semantics. |
| `preflight.py` | Artifact discovery, active-stat screening, feature contracts, bounded native probes. |
| `trace.py`, `temporal.py` | Native sample traces and chronology audits. |
| `privilege_delta.py` | Serialized baseline comparison and native boundary probes. |
| `hub_scanner.py`, `graph.py` | Bounded Hub acquisition, aggregation, and deterministic lineage exports. |
| `cli.py` | Check orchestration and terminal/JSON output. |

The implementation uses LeRobot's native `PolicyProcessorPipeline`, steps, state, feature transforms, and hooks. It does not create a parallel processor runtime. Tests generate real processor checkpoints for silent bypass (#4415), low-spread gain (#5289), FP16 non-finite arithmetic (#4973), and custom chronology mismatches (#5315), with healthy controls.

The [release workflow](.github/workflows/ci.yml) builds one wheel/sdist pair, then installs each in a clean Python 3.12 environment on Linux, macOS, and Windows. The [release harness](scripts/ci_release_check.py) verifies exact installed source bytes, runs the real CLI before pytest is installed, and executes the full packaged suite. [Actions runs](https://github.com/adi271213/lerobot-preflight/actions) record the tested revision; a green historical run does not validate subsequent changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for engineering and verification requirements and [SECURITY.md](SECURITY.md) for the trust boundary and vulnerability reporting.

## License and release status

No project license has been selected. An owner-selected license and matching package metadata are required before an open-source release. Third-party dependencies retain their own licenses. Registry publication and package-name reservation have not been performed.
