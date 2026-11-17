# lerobot-preflight

**Reference-Free Numerical Processor Auditor & Chronology/Contract Linter for LeRobot Checkpoints.** Inspect serialized normalization and execute local processor traces without a trusted baseline or ONNX export. Optional baseline comparisons explain changes in processor mappings and execution horizons.

Findings cover checkpoint artifacts and inspected processor behavior. This tool does not certify hardware safety or provide a standalone ISO safety-compliance guarantee.

## Quickstart

Requires Python 3.12+ and the published LeRobot **0.7.0 or later**. Core processor support works with the PyPI release; git-main is optional for newer checkpoint-specific processors. Install from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

To inspect a checkpoint requiring newer upstream processor code, install that source explicitly:

```bash
python -m pip install --upgrade "lerobot @ git+https://github.com/huggingface/lerobot.git"
```

Diagnostics still leave uninspected custom steps `UNKNOWN`. Compatibility is tested against the actual PyPI `lerobot==0.7.0` wheel and the local upstream reference (`0.7.1`, commit `fe1c54594f6b`). Missing optional helpers use the installed release's native processor methods.

The [compatibility record](precision_hub_audit/compatibility_report.json) includes wheel hashes, both full test runs, and a built-wheel `--strict` CLI check against PyPI 0.7.0. Those checks reuse the existing dependency environment; they do not claim every platform was tested from an empty environment.

If this checkout already has its dependencies in `.venv`, install the real console script with:

```bash
.venv/bin/pip install -e . --no-deps
.venv/bin/lerobot-preflight --policy ./ckpt
```

Check a local checkpoint's processor artifacts and temporal anchor:

```bash
lerobot-preflight --policy ./ckpt
```

Trace native processor steps with a recorded observation:

```bash
lerobot-preflight --policy ./ckpt --sample ./obs.npz
```

Compare a candidate with the production checkpoint:

```bash
lerobot-preflight --policy ./cand --baseline ./prod
```

Export the processor and feature lineage with no additional graph dependencies:

```bash
lerobot-preflight --policy ./ckpt --graph ./processor-graph.json
lerobot-preflight --policy ./ckpt --graph ./processor-graph.mmd
lerobot-preflight --policy ./ckpt --graph ./processor-graph.md
```

JSON contains deterministic feature, processor-step, and stat-buffer nodes with typed edges. `.mmd` contains a Mermaid `flowchart LR`; `.md` wraps it in a Markdown code fence. Edges record normalization modes, effective dtypes, gains, renames, missing-stat bypasses, contract mismatches, and temporal anchors. This is a local artifact map, not a policy rollout. Unknown dtype or chronology stays unknown, and exports cannot overwrite checkpoint artifacts.

Add `--report ./preflight-report.json` to write strict JSON with check IDs, step indices, tensor keys, offending coordinates, and SHA-256 provenance prefixes. Non-finite values retain explicit IEEE strings (`NaN`, `Infinity`, `-Infinity`) plus typed `non_finite_values` annotations with JSON-pointer locations.

The CLI exits **1** for blocking `FAIL` or `CONFLICT` findings. With `--baseline`, an `UNKNOWN` on required comparison coverage also exits **1**, including missing action buffers, unverified processor compatibility, or incomplete guard comparison. Add `--strict` to fail on coverage `UNKNOWN`s and **WARNING advisories**, including `PASS/WARNING`. Undeclared units, frames, and other physical fields outside LeRobot's native schema remain informational `UNKNOWN` notes and do not independently fail strict mode. No units or frames are inferred.

```bash
lerobot-preflight --policy ./cand --baseline ./prod --report ./comparison.json
lerobot-preflight --policy ./ckpt --sample ./obs.npz --strict
```

Compatible relative/absolute-action pairs can pass a self-baseline comparison, including `--strict`. The gate compares effective masks, resolved temporal anchors and the surrounding action/state affine maps, then executes paired native boundary probes against shared inputs. Changed offsets, masks or anchors are reported with coordinate evidence; unsupported state-dependent or non-affine chains retain `UNKNOWN` coverage.

Optional physical semantics and chronology sidecars require explicit opt-in:

```bash
lerobot-preflight --policy ./ckpt --metadata ./preflight_metadata.json
```

Saved `config.json` delta indices can establish chronology without a sidecar: `observation_delta_indices: [-1, 0]` identifies an oldest-first history with the current frame last; `[0, 1]` identifies a current-first window. A non-null saved `state_observation_delta_indices` takes precedence for state, matching LeRobot's dataset factory. The delta sequence determines the probe's window length: native VLA configurations can keep `n_obs_steps=1` while requesting several delta frames. An explicitly selected chronology sidecar overrides the inferred ordering. For example, it can contain `{"temporal_observation_ordering":"oldest_first","current_index":-1}`. Physical declarations may be nested under `physical_semantics`. Only `config.json` and the selected file are read for this purpose; arbitrary metadata files in a checkpoint are not automatically trusted. A clean native checkpoint does not need custom physical fields to pass `--strict`.

`TEMPORAL_ANCHOR_MISMATCH` (#5315) is a **cross-policy chronology contract check** for custom or migrated pipelines. The inspected `RelativeActionsProcessorStep` and its inverse intentionally collapse a 3D state with `state[:, 0]`. This is valid for the in-tree VLA-JEPA and LingBot-VA policies, whose current-first `observation_delta_indices` starts at `0`. The contract check catches a custom or migrated pipeline that pairs this step with an oldest-first history window `[-k, ..., 0]`, where the current frame is `state[:, -1]`. Policy names and tensor shapes alone do not establish chronology. Some native configurations declare separate image and state delta windows; this audit uses the saved state-window declaration or explicit chronology.

The sample archive should contain numeric arrays under the checkpoint's exact keys, such as `observation.state`, with the exact declared trailing feature shape and optional leading batch/time dimensions. A `(1,1)` state cannot broadcast into a declared 14-coordinate feature: it fails `INVALID_SAMPLE_SHAPE` before step 0. Include `action` in the same archive, or in an adjacent `sample_action.npz`, to exercise postprocessing. Without a sample action, postprocessor trace coverage is `UNKNOWN`. For numeric VLA traces, the inspected language tokenizer is replaced in memory with LeRobot's native identity step; `OFFLINE_LANGUAGE_TOKENIZATION` is **UNKNOWN/INFO** because only numeric state/action subpaths are traced. Language tokenization and token-consuming downstream behavior remain unverified; declared downstream token consumers are blocked. Native numeric steps retain their original indices, and no tokenizer assets are fetched. Non-finite input samples are reported as `INVALID_SAMPLE_INPUT` before step 0.

## Cross-platform release checks

The [GitHub Actions workflow](.github/workflows/ci.yml) tests Python 3.12 on native Linux, macOS and Windows runners, with separate wheel and sdist jobs. One isolated build produces the exact artifacts used by all six jobs. Each job creates a new virtualenv, resolves the declared dependencies, checks the installed CLI before adding pytest, and runs the full test suite from the source archive against the installed package. No reference checkout or `PYTHONPATH` is used for the installed CLI checks.

Linux and Windows jobs request CPU PyTorch wheels from PyTorch's official index; no GPU is required. Jobs retain diagnostic JSON, exact commands, stdout/stderr, pip reports and JUnit results even when a check fails. The runner deliberately uses temporary paths containing spaces. See [Actions runs](https://github.com/adi271213/lerobot-preflight/actions) for actual platform results; adding the matrix alone is not evidence that every platform passes.

The same checks can be run with [scripts/ci_release_check.py](scripts/ci_release_check.py). Supply `--distribution wheel` or `sdist`, `--dist-dir`, an empty `--work-dir` outside this checkout, and `--results-dir`. Dependency installation may use the network; processor diagnostics run with Hub offline flags and use locally generated native processor fixtures.

## Ecosystem & Related Tools

The [executed 14-case comparison](precision_hub_audit/head_to_head/head_to_head_report.md) records the pinned competitor versions, input adaptations, shared coverage and limitations. Basic zero/NaN-stat checks and baseline drift gates already exist elsewhere; this tool adds native numerical diagnosis and chronology checking.

| Tool | Primary scope and overlap |
| --- | --- |
| [`jashshah999/lerobot-doctor`](https://github.com/jashshah999/lerobot-doctor) | Pre-training `LeRobotDataset` v2/v3 diagnostics: `info.json`, parquet episodes, videos, statistics and policy-window compatibility. Its documented kinematics check also uses supplied or registered URDF joint limits, with unit-mismatch caveats. This is the existing dataset linter on PyPI and the HF Dataset Visualizer; it is a separate project. |
| [`robotruth` 0.1.6](https://github.com/mulkakhileshmj/robotruth/tree/8eb1704cf446239e3f03687c2ebd707cbab68e98) | Evaluated-versus-deployed checkpoint contract gate. It detects normalization fingerprint and execution-horizon changes and also checks `model.safetensors` weight-file identity hashes. Our processor-only diff does not gate neural weight identity. |
| [`Tether` 0.12.0](https://pypi.org/project/fastcrest-tether/0.12.0/) | ONNX-based deployment/runtime tooling with [Doctor checks](https://docs.fastcrest.com/reference/doctor/) and an embodiment schema. Shared coverage includes zero/NaN or wrong-length action `std`; the separate schema also rejects zero state `std`. Our comparison exercised those components through explicit adapters, not a complete Tether deployment. |
| **`lerobot-preflight` — this tool** | Reference-free numerical diagnosis of serialized LeRobot checkpoints: finite-but-explosive gains (`std=1.48e-6`, a `0.20` sample → `134,228` with native epsilon), `so100-blue` multi-dataset namespace bypasses, conditional versus active FP16 overflow, HTTP byte-range auditing with **6,984 reconstructed tensor-payload bytes (~6.8 KiB)**, and saved `observation_delta_indices` versus `state[:, 0]` chronology contracts. No baseline or ONNX export is needed for standalone checks. |

No neural weight-identity gate or weight-hashing option is implemented here. Processor validation complements checkpoint identity, dataset quality and deployment-runtime checks; these fixture results do not establish broader product superiority or hardware safety.

## Findings in Public Checkpoint Artifacts

These are verified artifact findings from pinned public Hugging Face checkpoints. They identify deployment prerequisites and numerical risks; they do not establish an observed robot failure.

| Repository and pinned revision | Finding | Exact evidence |
| --- | --- | --- |
| [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base/tree/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6), `5e8d12a6e297` | `UNBOUND_BASE_MODEL_STATS` | Preprocessor step **5** and postprocessor step **0** expect `action.mean/std`, but contain multi-dataset `so100{,-blue,-red}.buffer.action.*` buffers with no target dataset bound. Required `observation.state` statistics are also absent. Both state files: SHA-256 `490ab239d96e`. This is a valid base checkpoint requiring dataset-stat binding or fine-tuning before execution. |
| [`lerobot/pi0_old`](https://huggingface.co/lerobot/pi0_old/tree/e4ed526af508e58f6008b29e9e48f1098278fdb5), `e4ed526af508` | `FEATURE_CONTRACT` | Preprocessor step **5** requires `observation.state.mean/std` and `action.mean/std`; postprocessor step **0** requires `action.mean/std`. Both JSONs omit `state_file`, and the repo supplies no processor statistics. Preprocessor JSON SHA-256: `b108f4c0e6ab`; postprocessor JSON: `a681a9837016`. The default native empty state bypasses normalization; external statistics are required. |
| [`jhseon-holiday/vqbet_xarm_holi_test`](https://huggingface.co/jhseon-holiday/vqbet_xarm_holi_test/tree/7dba32e8f65251b494491d7cefecfa8a6d5c2da7), `7dba32e8f652`, and [`…_175000`](https://huggingface.co/jhseon-holiday/vqbet_xarm_holi_test_175000/tree/9dd03107e5461e85afdf1849c17266b6a84443a4), `9dd03107e546` | `PATHOLOGICAL_STD_GAIN` | Embedded `normalize_inputs.buffer_observation_state.std[4] = 6.6630414039536845e-6`; native migration gain ≈ **149,857×** (rounded from 149,856.7078) with `eps=1e-8`. Extracted stat-file SHA-256: `7d24b59de47a`. |
| [`HaomingSong/lerobot-pi0-bridge`](https://huggingface.co/HaomingSong/lerobot-pi0-bridge/tree/93c2840a2c68c337da0d472a2f52d1fd8d62132a), `93c2840a2c68` | `PATHOLOGICAL_STD_GAIN` (`CRITICAL`), conditional `FP16_NON_FINITE_RISK` (`WARNING`) | Saved mode is `MEAN_STD`; `normalize_inputs.buffer_observation_state.std[6] = 0`. Float32 normalization maps the training mean to **0**, but a **+0.20** perturbation to **20,000,000**, a **100,000,000×** gain with `eps=1e-8`. NaN/Inf requires half normalization buffers or migration to a modern processor with FP16 input. Extracted stat-file SHA-256: `128ca7bfcb39`. |
| [`iantc104/Diffusion-RPL-UR5-PegInsert-v0`](https://huggingface.co/iantc104/Diffusion-RPL-UR5-PegInsert-v0/tree/0703f194c88b43cc1b704c682f27fe554e1b830e), `0703f194c88b` | `PATHOLOGICAL_STD_GAIN` (`CRITICAL`), conditional `FP16_NON_FINITE_RISK` (`WARNING`) | Saved mode is `MIN_MAX`; `normalize_inputs.buffer_observation_environment_state.min/max` have **`min == max` at dimensions 11–17**. Float32 normalization maps the training constant to **-1**, but a **+0.20** perturbation to approximately **40,000,000**, a **200,000,000×** gain with `eps=1e-8`. Extracted stat-file SHA-256: `fb006f50ebd8`. |

The xArm checkpoint explicitly saves `input_normalization_modes["observation.state"] = "mean_std"`; the current VQ-BeT default of `MIN_MAX` does not override that saved setting.

Legacy numeric values above were fetched through exact tensor byte ranges. The historical `Normalize` implementation keeps its statistics in **float32**: FP16 observations alone are promoted by float32 buffer arithmetic. Its NaN/Inf case requires casting the normalization module/buffers to `.half()`, or migrating to modern `NormalizerProcessorStep`, which **auto-casts statistics to the input dtype**. Conditional legacy FP16 risks are nonblocking **PASS/WARNING** advisories scoped to the declared/default float32 buffer arithmetic; they independently fail only `--strict`. Actual non-finite native probe results under an active dtype and modern FP16 denominator risks remain blocking **FAIL/CRITICAL** findings. The checkpoint files do not pin every deployment's implementation or precision; gain checks state the inspected `eps=1e-8` formulas rather than claiming an observed deployment failure. Embedded legacy buffers have no modern processor step index.

Pi0 Bridge explicitly saves `device: "cpu"` and `use_amp: false`. Its zero standard deviation does **not** by itself produce NaN at the training mean in that float32 configuration. Likewise, UR5's equal bounds produce finite **-1** at the training constant. The critical findings concern sensitivity to a nonzero perturbation; the FP16 findings describe a separate precision-dependent risk.

The earlier [26-checkpoint precision audit](precision_hub_audit/hub_audit_report.md) also verified this official legacy checkpoint, separately from the new live selection:

| Repository and pinned revision | Finding | Exact evidence |
| --- | --- | --- |
| [`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/ba73b2766f1371cdc133ca4efb97eb090d744625), `ba73b2766f13` | `SCHEMA_GENERATION_MISMATCH` | No modern processor JSONs; the model header declares legacy `normalize_inputs.buffer_observation_state.mean/std` and `unnormalize_outputs.buffer_action.mean/std`, each `F32[14]`. Header SHA-256 prefix: `9e3b17b4a3fd`. |

The SmolVLA checkpoint also has an advisory `CAMERA_KEY_CONTRACT_MISMATCH` (**UNKNOWN/WARNING**) between processor camera names (`observation.image`, `observation.image2`, `observation.image3`) and policy camera names (`observation.images.camera1`, `observation.images.camera2`, `observation.images.camera3`). `taekyukim/smolvla_base` instead declares policy cameras named `observation.images.front` and `observation.images.wrist`.

These visual features use `IDENTITY`. The native preprocessor passes supplied camera keys through and skips absent declared image entries; it does **not** throw a camera `KeyError`. Downstream, the policy backbone's `prepare_images` selects camera keys from the config's `input_features` and rejects a batch when **all** configured cameras are missing. Correctly named policy cameras work despite the stale normalizer declarations. This finding records declaration drift and downstream input-mapping risk, not a reproduced preprocessing crash. Base checkpoints may expect runtime dataset-statistics overrides; unbound statistics do not imply corrupt weights. The earlier official legacy example has header-only evidence establishing keys, shapes, and dtypes.

The [26-checkpoint JSON evidence](precision_hub_audit/hub_audit_report.json) records the earlier regression scan, which removed all eight previously identified FP16 false-positive repo flags by excluding unused metadata and identity features. It does not measure a general false-positive rate.

## Audit Public Hub Checkpoints

The byte-range audit of **100 tagged repositories** reconstructed **242 normalization tensors from just 6,984 bytes of tensor slices**, without downloading any full neural weight files. It found **149,857× gain on xArm**, **zero standard deviation on Pi0 Bridge**, and **equal min/max bounds on UR5 PegInsert**, as detailed above. Read the **[Markdown findings and provenance](precision_hub_audit/hub_100_report.md)** or **[full JSON report](precision_hub_audit/hub_100_report.json)** for the pinned snapshots.

| Classification | Repositories |
| --- | ---: |
| `MODERN_POLICY` | 42 |
| `LEGACY_POLICY` | 55 |
| `MIXED_POLICY` | 1 |
| `NON_POLICY_REPO` | 1 |
| `GATED_OR_PRIVATE` | 0 |
| `UNKNOWN` | 1 |

The headline is **5 independent artifact finding families across 15 repositories**: **4 defect/numeric-risk signatures** (xArm gain, Pi0 zero-std gain, UR5 zero-span gain, and Git LFS pointer data) plus **1 stat-binding prerequisite family**. Counting the intentional base-model prerequisite as a fifth independent defect would overstate the evidence. The two xArm repositories share identical statistics and contribute one signature.

The stat-binding prerequisite affects the official `lerobot/smolvla_base` plus **9 community forks/migrations** sharing its action-stat payloads, and the separate `lerobot/pi0_old` base checkpoint: **11 prerequisite-bearing repositories**. The canonical modern SmolVLA container is `490ab239d96e`; legacy containers and pointer text have different file hashes. Missing-stat `FEATURE_CONTRACT` symptoms on those bases belong to the same non-defect prerequisite. Numeric/file findings affect **5 repository snapshots**, including the repeated xArm checkpoint and one pointer-bearing base copy. These counts overlap.

The official-base classification is curated through `_INTENTIONAL_BASE_REPOS = {"lerobot/smolvla_base", "lerobot/pi0_old"}`. Repository IDs identify those two upstream bases; shared buffer hashes identify exact-stat copies, and unbound multi-dataset namespaces establish the binding prerequisite. These rules do not verify neural-weight identity or fine-tuning history.

Separately, **55 legacy checkpoints** appear in **Version-Compatibility / Legacy Schema (Requires PR #1452 Migration)**. They may be valid for their original LeRobot version; a migration requirement is not a standalone checkpoint defect. **2 conditional legacy FP16 PASS/WARNING advisories** have their own precision-risk table and are excluded from blocking counts. Camera drift has **4 UNKNOWN/WARNING advisories**. Unconfigured modern hypothetical half-precision risks follow the same advisory rule; only an active FP16 cast/buffer or proven native non-finite result supports a blocking precision finding. Deployment-gate statuses for unbound stats remain visible: a valid base still requires target dataset binding before execution.

The cached audit has **13/100** numeric-buffer coverage gaps, **9/100** native-chain probe gaps, and **76/100** temporal `UNKNOWN`s. The 23 non-UNKNOWN temporal policies comprise **0 relative-action policies and 23 non-relative policies**. One non-policy repository is outside that audit. **Zero multi-step relative policies were exercised** in this Hub sample; the cross-policy chronology contract checked by `TEMPORAL_ANCHOR_MISMATCH` was therefore unexercised in the live sample. All **100 revisions are pinned**, and cached regeneration fetched **zero new bytes**. Missing active statistics prevent a numeric `PASS`; no target dataset namespace is chosen automatically.

Shared-stat groups now use each **individual tensor-payload SHA-256**, independently of unrelated buffers or safetensors file layout. The two xArm revisions share `observation.state.std` payload `62c684b883e8…`; the modern base-model copies share, for example, `so100-blue.buffer.action.std` payload `a4762d32295a…`. Their file-level fingerprints (`7d24b59de47a` and `490ab239d96e`) remain separate provenance fields. Repeated payloads are not independent discoveries, and identical bytes under different normalization formulas or stat namespaces retain separate diagnostic evidence.

Selection hints covered 20 ACT, 20 Diffusion, 21 SmolVLA, 20 Pi0, and 19 VQ-BeT repositories; TD-MPC returned zero tagged candidates. The report lists every exact `HfApi.list_models` query, including seed searches for `smolvla_base` and `act_aloha_sim_transfer_cube_human` (limit 25 each), family searches (100 each), and tagged fallback (400), all with `filter="lerobot", full=True`. This targeted convenience sample is not a prevalence estimate. The [archived header-only audit](precision_hub_audit/archive/hub_100_report_header_only.md) downloaded **1,464,463 bytes** of configs/processor statistics and **7,965,944 bytes** of headers. Range reads reconstructed **242 tensors / 6,984 payload bytes** across **56 legacy/mixed checkpoints** without downloading full neural weights.

Re-audit the pinned selection entirely from cached snapshots:

```bash
.venv/bin/python scripts/scan_hub.py --limit 100 \
  --cached-report precision_hub_audit/hub_100_report.json \
  --output-dir precision_hub_audit --report-name hub_100_report
```

For a fresh selection, replace `--cached-report …` with `--targeted`; fresh scans automatically fetch legacy normalization ranges. Add `--fetch-legacy-stats` to a cached run only when missing ranges must be fetched. Targeted discovery prioritizes public repositories tagged `lerobot` across ACT, Diffusion, SmolVLA, Pi0, VQ-BeT, and TD-MPC. The downloader fetches bounded configs, processor statistics, model headers, and selected stat tensor ranges before local static and temporal checks. JSON and Markdown reports record selection, coverage gaps, pinned revisions, exact findings, and repo-relative provenance.

## What We Check (and What We Refuse to Guess)

| Check | Behavior |
| --- | --- |
| Static preflight | Active stats include native ACTION normalization even with `features={}`. Checks individual and composed forward gains, including native renames between normalizers, invalid stats, contracts, and bounded native dtype probes. Stacked individually modest gains cannot hide an excessive end-to-end gain. Uninspected gaps leave composed gain `UNKNOWN`. Explicit BF16 uses BF16 bounds rather than a hypothetical FP16 warning. |
| Offline trace | Checks required inputs and exact trailing shapes, then runs silent-bypass, activation-gain, and non-finite checks on **both preprocessing and postprocessing**. Gain requires output magnitude above 50 and more than 2× input magnitude by default. |
| Cross-policy chronology contract | Checks custom or migrated relative-action pipelines against their declared chronology and state/action/mask dimensions; missing action width stays `UNKNOWN`. Saved delta indices identify current-first versus oldest-first windows, with explicit sidecar override. Native `state[:, 0]` is valid for current-first windows. Round trips and anchor matching use a dtype/scale-aware rounding budget: `atol = max(1e-5, 4 * machine_epsilon * max_intermediate_scale)`, with `rtol=1e-5`. Native forward/inverse action values and active action-normalizer constants contribute to that scale, so large offsets do not become false inverse failures. Errors beyond the numerical budget still fail; ambiguous anchors remain `UNKNOWN`. |
| Privilege delta | Compares every active non-identity observation normalization, including STATE, ENV and **VISUAL** stats. Channel statistics use bounded native probes, without allocating full megapixel images. Compares composed policy-input mappings, action slopes/origins/intervals, intermediate dtype failures, ordered guards, masks, clamps and horizons. Implicit native ACTION stats are included even without a feature declaration. No tokenizer or policy inference runs. Probe domains are comparison inputs, not hardware limits. |

Numeric screening follows declared active features and normalization modes. Unused dataset metadata and `IDENTITY` features are skipped. Forward normalization is checked using its epsilon and declared input dtype. An inverse action coordinate with zero standard deviation gets `DEAD_ACTION_DIMENSION`, because its output stays at the saved mean. That coordinate may be padding; its physical actuator identity is undeclared.

If a preprocessor's `norm_map` enables `ACTION` while declaring neither an action feature nor action statistics, numeric coverage is `UNKNOWN`, which fails `--strict`. When action statistics exist, the native implicit ACTION path is screened even without a feature declaration. Actual BF16 overflow is reported in BF16 scope; a valid BF16 checkpoint does not acquire an unused FP16 advisory.

`Status` distinguishes blocking evidence from advisory scope. Hypothetical FP16 findings use `PASS/WARNING` when no half-precision cast or buffer is configured; camera drift uses `UNKNOWN/WARNING`. Standard mode permits both, while `--strict` rejects both. Undeclared optional physical semantics are `UNKNOWN/INFO` notes, exempt from strict mode. Explicitly blocking `FAIL` findings still block, including at `WARNING` severity.

| Status | Meaning |
| --- | --- |
| `PASS` | The check completed without finding a violation within its inspected scope. |
| `FAIL` | Artifact or trace evidence establishes the reported condition. |
| `CONFLICT` | Artifact sources or declared contracts disagree. |
| `UNKNOWN` | Available evidence cannot establish the property or complete the check. |

Joint-angle units and action coordinate frames are explicitly `UNKNOWN` unless metadata declares them. Temporal chronology also stays `UNKNOWN` unless saved delta indices or explicit chronology identify the current frame. Tensor shapes do not establish radians, degrees, physical frames, controller timing, or hardware safety limits. Numeric scale comparisons do not prove physical equivalence when those semantics are undeclared.

The local checks use LeRobot's native `PolicyProcessorPipeline`, processor state, feature transforms, and hooks. Static and baseline checks also evaluate bounded synthetic native numeric subpaths on CPU: action postprocessing uses `[-3,-1,0,1,3]`, while forward preprocessing uses saved means, mean ± std, or bound endpoints. Every intermediate dtype is retained. Unsupported value-changing steps make coverage `UNKNOWN`; an already-proven non-finite prefix still fails. No neural policy inference, simulator rollout, motor command, or trace-time network call runs. Numeric tracing bypasses only the inspected language tokenizer; other uninspected processors remain explicit coverage gaps.

## Verified Console-Script Gates

These archived gates used the former local console-script name `.venv/bin/lerobot-doctor`, before the `lerobot-preflight` rename. It was executed from `/private/tmp` with `PYTHONPATH` removed and Hub access disabled. These are real cached Hub snapshots, not generated checkpoint fixtures. See the unchanged [historical terminal output](precision_hub_audit/gates/terminal_output.txt) and [machine-readable gate results](precision_hub_audit/gates/gate_summary.json).

| Run | Exit | Verified result |
| --- | ---: | --- |
| Clean `axellin/act_cupfinal_2026-11-13_14-34-08` with synthetic observation/action NPZ | 0 | Native preprocessing and postprocessing traced; all three trace checks pass. Only undeclared physical semantics remain `UNKNOWN`. |
| Fine-tuned `r-oi/smolvla_base_so101_pickplace_cube_100ep_rand-pos_015000` with NPZ | 0 | All three numeric pre/postprocessing trace checks pass. Tokenizer step **3** is explicitly bypassed; language outputs are unverified. |
| Orellius base versus its `100k` candidate, with NPZ and `--baseline` (historical run before strict comparison coverage) | 0 | The archived run verified numeric trace and action-scale ratios of **1.0**; it predates affine-origin checks and fail-closed required coverage. It does not establish complete present-day comparison coverage. |
| `taekyukim/smolvla_base` | 1 | Unbound target statistics are blocking; camera declaration drift is now advisory, with IDENTITY passthrough scope made explicit. |
| `jhseon-holiday/vqbet_xarm_holi_test` | 1 | Legacy schema mismatch and active state gain at dimension `[4]` are reported. |
| Independent single-coordinate mutation of the clean ACT snapshot | 1 | Setting only `observation.state.std[2]` to float32 `3.1e-6` is caught as `PATHOLOGICAL_STD_GAIN` at normalizer step **3**, dimension `[2]`. |

Synthetic NPZs use the exact declared shapes, float32 state/action values at saved training-range midpoints, and RGB gradients in `[0,1]`. The mutation test preserves all other tensor coordinates and files and verifies the original snapshot is unchanged. The VLA rerun used the original NPZs and preserved all checkpoint artifacts; see its [actual terminal output](precision_hub_audit/gates/vla_terminal_output.txt). Remaining `UNKNOWN` findings concern separate preflight feature transformations, temporal audit coverage, or undeclared physical semantics. Numeric trace success does not verify language processing, neural policy behavior, or physical equivalence.

## Development

```bash
.venv/bin/pytest -q
```

The tests generate real LeRobot processor checkpoints reproducing silent bypass (#4415), pathological low-standard-deviation gain (#5289), and FP16 non-finite normalization (#4973), plus legacy embedded normalization buffers (PR #1452). Separate synthetic fixtures test the cross-policy chronology contract (#5315): they deliberately pair index-zero relative-action conversion with an oldest-first history, while current-first fixtures verify the intended native behavior.
