<div align="center">

# 🤖 Flinch

**Catch broken normalization in LeRobot checkpoints before your robot does.**

[![CI](https://github.com/adi271213/flinch/actions/workflows/ci.yml/badge.svg)](https://github.com/adi271213/flinch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)

<img src="docs/demo.gif" alt="Flinch auditing a public xArm checkpoint from the Hugging Face Hub and failing it for a 149,857x input gain" width="820">

</div>

A LeRobot checkpoint can load without a single warning and still be broken. If a saved standard deviation is nearly zero, LeRobot divides by it (plus a tiny `eps`), and a normal sensor reading turns into a huge number. The policy sees garbage, and the robot jerks. It flinches.

Flinch looks for that before anything moves. It reads the checkpoint's processor files, runs LeRobot's own processor code on a few test inputs on your CPU, and tells you exactly which step, key and coordinate is wrong.

## 🔍 What it found on the Hugging Face Hub

I pointed it at 100 public LeRobot checkpoints. Some of what turned up:

| Checkpoint | Problem | What it does to a +0.20 input |
| --- | --- | --- |
| `jhseon-holiday/vqbet_xarm_holi_test` | `observation.state` std is 6.7e-6 at index 4 | becomes ~29,971 (149,857× gain) |
| `HaomingSong/lerobot-pi0-bridge` | `observation.state` std is exactly 0 at index 6 | becomes ~20,000,000 |
| `iantc104/Diffusion-RPL-UR5-PegInsert-v0` | min equals max for 7 environment-state coordinates | becomes ~40,000,000 |
| `y1y2y3/smolvla_base1` | a 129-byte Git LFS pointer was saved instead of the stats tensor | the stats are simply missing |

Every row is pinned to an exact commit. The full write-up is in [docs/audit-evidence.md](docs/audit-evidence.md). These are problems in the files; I didn't run any of these on a real robot.

## 🚀 Quick start

You need Python 3.12. No GPU.

```bash
git clone https://github.com/adi271213/flinch.git
cd flinch
python3.12 -m venv .venv && source .venv/bin/activate
pip install .
```

The first install pulls in PyTorch and LeRobot, so give it a few minutes.

Now check a checkpoint straight from the Hub:

```bash
flinch --policy hf://jhseon-holiday/vqbet_xarm_holi_test
```

Or a checkpoint on your disk:

```bash
flinch --policy ./outputs/train/my_run/checkpoints/last/pretrained_model
```

With `hf://`, Flinch downloads only the small files it needs (processor configs, stats and the model header) into a temp folder that it deletes afterwards. It never downloads model weights. Add `@branch`, `@tag` or `@commit` to check a specific version. Private and gated repos aren't supported yet; download those and pass the local path.

No internet? Generate the test checkpoints that ship with the repo:

```bash
python -c "from pathlib import Path; from tests.fixtures.make_bug_fixtures import make_bug_fixtures; make_bug_fixtures(Path('.flinch-demo'))"
flinch --policy .flinch-demo/bug_5289_pathological_gain   # fails
flinch --policy .flinch-demo/clean_act_modern             # passes
```

## 🧰 More ways to use it

```bash
# Feed a real observation through the processors and watch each step
flinch --policy ./ckpt --sample ./obs.npz

# Compare a new checkpoint against the one you run today
flinch --policy ./candidate --baseline ./production

# Save the findings as JSON and draw the processor pipeline as a Mermaid diagram
flinch --policy ./ckpt --report ./report.json --graph ./pipeline.mmd

# Be strict: also fail when something couldn't be checked
flinch --policy ./ckpt --strict
```

For `--sample`, save an `.npz` whose keys match the checkpoint's feature names, like `observation.state`. Put `action` in it too (or in a `sample_action.npz` next to it) if you want the postprocessor checked as well.

## 🚦 Reading the output

Every finding gets one of four labels:

| Label | Meaning |
| --- | --- |
| `PASS` | Checked, and it's fine. |
| `FAIL` | Something is wrong, and Flinch shows you where. |
| `CONFLICT` | Two files in the checkpoint disagree with each other. |
| `UNKNOWN` | Flinch couldn't check this, so it won't pretend it did. |

Exit code is `1` if anything is `FAIL` or `CONFLICT`, and `0` otherwise, so it drops straight into CI. `--strict` also fails on gaps in what could be checked and on warnings. Bad arguments exit `2`.

Flinch never guesses physical units. A tensor's shape can't tell you whether a joint is in radians or degrees, so that stays `UNKNOWN` unless you declare it with `--metadata`.

## ✅ What it checks

- **Stats that blow up inputs:** near-zero std, min equal to max, NaN or Inf, and stats that are missing or saved under the wrong name so normalization is silently skipped.
- **Stacked normalizers:** two steps that are each fine but multiply into a huge gain.
- **Precision:** values that overflow when the checkpoint runs in float16.
- **Base models:** checkpoints like `lerobot/smolvla_base` that need your dataset's stats before they can run. You get one clear message telling you to fine-tune or bind stats, instead of a wall of errors.
- **Relative actions on stacked observations:** LeRobot's relative-action step treats the first frame of a stacked observation as "now". That's right for the built-in policies, but a custom pipeline that puts the oldest frame first would be anchored to the wrong frame. Flinch checks the saved frame order.
- **Changes between checkpoints:** with `--baseline`, any change to how inputs and actions are scaled, shifted or clipped, and to how many actions are executed per step.
- **Broken files:** Git LFS pointers saved in place of real data, and old-format checkpoints that need migrating.

## ⚙️ How it works

```mermaid
flowchart LR
  ckpt["Local or hf:// checkpoint"] --> pre[Stats and config checks]
  ckpt --> temporal[Frame order check]
  ckpt --> trace[Run a sample through each step]
  sample[Optional .npz sample] --> trace
  ckpt --> delta[Compare with baseline]
  baseline[Optional baseline] --> delta
  pre --> findings[Findings]
  temporal --> findings
  trace --> findings
  delta --> findings
  findings --> gate[Terminal output and exit code]
  findings --> json[JSON report]
  ckpt --> lineage[Pipeline diagram]
```

Flinch doesn't reimplement LeRobot. It loads the checkpoint with LeRobot's own `PolicyProcessorPipeline` and watches each step with LeRobot's hooks, so what it reports is what your robot would actually get. It never runs the neural network, a simulator or a motor, and it doesn't touch the network while checking. If a checkpoint uses a custom processor that Flinch can't inspect, it says `UNKNOWN` rather than importing unknown code.

<details>
<summary>Code layout</summary>

```text
src/lerobot_flinch/
  cli.py              the flinch command
  hub_source.py       hf:// downloads
  preflight.py        stats, config and file checks
  trace.py            runs a sample through each processor step
  temporal.py         frame order and relative-action checks
  privilege_delta.py  comparison against a baseline
  graph.py            pipeline diagram export
  hub_scanner.py      scanning many Hub repos at once
  models.py           findings and exit codes
tests/fixtures/       generates broken and healthy test checkpoints
scripts/              Hub scanner and release checks
```

</details>

<details>
<summary>All options</summary>

| Option | What it does |
| --- | --- |
| `--policy` | Checkpoint to check: a local folder or `hf://owner/name[@revision]`. Required. |
| `--sample` | `.npz` file with a real observation (and optionally an action) to trace. |
| `--baseline` | Checkpoint to compare against: local folder or `hf://`. |
| `--report` | Write findings as JSON. |
| `--graph` | Write the processor pipeline as `.json`, `.mmd` or `.md`. |
| `--metadata` | JSON file declaring units, frames or frame order, e.g. `{"temporal_observation_ordering": "oldest_first", "current_index": -1}`. |
| `--std-threshold` | How small a std counts as suspicious. Default `1e-4`. |
| `--strict` | Also fail on anything that couldn't be checked, and on warnings. |
| `--version` | Print the version. |

Set `HF_HUB_OFFLINE=1` to stop any Hub downloads.

</details>

## 🛰️ Scanning the Hub

The script behind the findings table above:

```bash
python scripts/scan_hub.py --targeted --limit 100 --output-dir ./hub-audit
python scripts/scan_hub.py --repo lerobot/smolvla_base --output-dir ./hub-audit
```

It downloads only small files and byte ranges, never full model weights, and pins every repo to an exact commit. Run `python scripts/scan_hub.py --help` for the rest.

## 🤝 Similar tools

| Tool | What it's for |
| --- | --- |
| [lerobot-doctor](https://github.com/jashshah999/lerobot-doctor) | Checks LeRobot **datasets** before training. Works well alongside Flinch. |
| [robotruth](https://github.com/mulkakhileshmj/robotruth) | Compares a checkpoint against a baseline or benchmark, including model weight hashes, which Flinch doesn't check. |
| [Tether](https://pypi.org/project/fastcrest-tether/) | Deploys policies to edge hardware through ONNX, with some overlap on zero or NaN action stats. |

Flinch is the one that runs LeRobot's own processors to find stats that are technically valid but explode at runtime.

## ⚠️ Limits

- A `PASS` means the processor files look right. It doesn't mean the policy is good or the robot is safe.
- It doesn't check model weights or dataset quality.
- The relative-action frame-order check is tested on generated checkpoints only; none of the 100 Hub checkpoints used relative actions.
- Not on PyPI yet, so install from source for now.

## 🛠️ Development

```bash
pip install -e ".[test,dev]"
python -m pytest -q
python -m ruff check src scripts tests && python -m ruff format --check src scripts tests
python -m mypy
```

The tests build real LeRobot checkpoints that reproduce four known bug types: silently skipped normalization (#4415), near-zero std (#5289), float16 NaN (#4973) and wrong relative-action frames (#5315). Each comes with a healthy twin that must pass. CI installs the built package fresh on Linux, macOS and Windows and runs the whole suite there.

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to report a security issue.

## 📄 License

MIT © 2026 Aditya Jain
