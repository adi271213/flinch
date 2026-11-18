# The 100-checkpoint Hub audit

This folder holds the full results behind the README's findings table, so anyone can check them.

- `hub_100_report.json.gz`: every finding for all 100 checkpoints, each pinned to an exact commit (`repo_id` and `revision` per entry).
- `hub_100_report.md`: the same results as a readable report.

The checkpoints were picked on 2026-11-16 from public repos tagged `lerobot`, spread across policy families (ACT, Diffusion, VQ-BeT, pi0, SmolVLA). It is a convenience sample, not a random one, and nobody labelled the checkpoints independently beforehand. So these results show **what Flinch found**, not how often it is right or wrong: there is no measured false-positive or false-negative rate.

The report was regenerated on 2026-11-18 with the current Flinch from the same pinned snapshots, offline. Every classification and every failing repo matched the original 2026-11-16 run.

## What each finding means

| Label | Check | Repos | Meaning |
| --- | --- | ---: | --- |
| **Confirmed defect** | `GIT_LFS_POINTER_ARTIFACT` | 1 | `y1y2y3/smolvla_base1` saved a 129-byte Git LFS pointer instead of its stats file. Anyone can confirm this by opening the file. |
| **Numerical risk** | `PATHOLOGICAL_STD_GAIN` | 4 | A saved spread at or near zero, so small input changes are hugely amplified: `jhseon-holiday/vqbet_xarm_holi_test` (and its `_175000` checkpoint), `HaomingSong/lerobot-pi0-bridge`, `iantc104/Diffusion-RPL-UR5-PegInsert-v0`. Whether this causes harm depends on how much those inputs move in deployment. UR5's flagged coordinates are `[-0.0127, 1, 0, 0, 0, 1, 0]`, which looks like a fixed pose in simulation, so it may well be harmless. None of these were run on a robot. |
| **Expected: base model** | `UNBOUND_BASE_MODEL_STATS` | 10 | Pretrained base models (mostly SmolVLA) that ship stats for several datasets and need your own dataset's stats before use. 8 of the 9 `FEATURE_CONTRACT` failures are these same repos, plus `lerobot/pi0_old`. Not defective weights. |
| **Expected: old format** | `SCHEMA_GENERATION_MISMATCH` | 55 | Checkpoints saved before LeRobot's processor files existed. They need LeRobot's migration before current versions can load them. |

## What wasn't checked

- **13 of 100**: some normalization stats couldn't be read, so the gain check is incomplete.
- **9 of 100**: a step Flinch can't inspect stopped the end-to-end normalization probe.
- **76 of 100**: the frame-order check had no saved frame order to compare, so it is `UNKNOWN`. None of the 23 decided cases used relative actions, so that check has only been exercised on generated test checkpoints, not on real ones.

## Checking a row yourself

Every row can be re-checked against the exact commit, with no model weights downloaded:

```bash
flinch --policy hf://jhseon-holiday/vqbet_xarm_holi_test@7dba32e8f65251b494491d7cefecfa8a6d5c2da7
```

To re-check all 100:

```bash
python - <<'EOF'
import gzip, json, subprocess
report = json.load(gzip.open("docs/hub-audit/hub_100_report.json.gz"))
for repo in report["repos"]:
    target = f"hf://{repo['repo_id']}@{repo['revision']}"
    code = subprocess.run(["flinch", "--policy", target], capture_output=True).returncode
    print(code, target)
EOF
```

If the Hub is busy, a run can end with `HUB_SNAPSHOT` under "Not verified" in its `COVERAGE` line; run it again.

A fresh scan of today's Hub (`python scripts/scan_hub.py --targeted --limit 100`) picks whatever checkpoints exist now, so it won't match this sample row for row. The raw snapshots behind this report (19 MB) are not in the repo.
