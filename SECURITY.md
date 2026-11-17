# Security

## Scope and trust boundary

`lerobot-preflight` inspects checkpoint artifacts and executes inspected native data processors on CPU. It does not run a neural policy, simulator, or motor controller. A passing report is not a hardware-safety certification or a guarantee about a checkpoint's neural weights.

Native processor construction is limited to inspected classes. Numeric VLA tracing does not construct the language tokenizer. NPZ samples are loaded with pickle disabled, and tensor buffers use safetensors. Missing support remains explicit `UNKNOWN` coverage.

Checkpoint paths referenced by processor JSON must remain within the checkpoint directory. Caller-selected samples and metadata sidecars may be external. Output exporters reject unrelated existing artifacts and symlink destinations. These protections are not a sandbox against concurrent filesystem changes: use a stable checkpoint copy when auditing untrusted or changing input. Large local inputs may still consume CPU or memory.

The Hub downloader is the network-enabled component, used by `scripts/scan_hub.py` and by `--policy`/`--baseline hf://owner/name`. It fetches public repositories at pinned revisions and enforces header, file, snapshot, and tensor-range byte limits. Local checks and traces do not require Hub credentials. Do not place access tokens in sample files, processor JSON, command-line arguments, or committed environment files.

Reports may contain tensor slices, dataset names, checkpoint paths, and caller-supplied metadata. Review them before sharing, particularly when using private checkpoints. Hub report path redaction does not guarantee that arbitrary input text contains no personal or confidential information.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/adi271213/lerobot-preflight/security/advisories/new) if the repository offers it. If that channel is unavailable, open an issue requesting a private contact channel without exploit details or sensitive artifacts. Private reporting availability is controlled by the repository owner; it has not been verified as enabled.

Include the package and LeRobot versions, platform, affected check or exporter, and a minimal synthetic reproduction when possible. Do not post tokens, private model data, or personal paths in a public issue. No response-time or security-support SLA is currently offered for this prerelease.

## Dependency checks

The release workflow resolves dependencies in clean environments and runs `pip check` plus regression tests. This verifies installation consistency, not the absence of published vulnerabilities. Dependency audit results apply to exact resolved versions and advisory data available at audit time; they are not a permanent guarantee about all versions allowed by package metadata.
