# Security

## What Flinch does and doesn't touch

Flinch reads checkpoint files and runs LeRobot's data processors on your CPU. It never runs the neural network, a simulator or a motor. A clean report means the processor files look right. It doesn't mean the robot is safe or that the model weights are what you expect.

Some protections are built in:

- Flinch only builds processor steps it knows and has inspected. It never imports code from a checkpoint.
- `.npz` samples are loaded with pickle turned off, and tensors are read with `safetensors`.
- File paths inside a checkpoint can't escape the checkpoint folder. `../` and symlinks pointing outside are rejected.
- `--report` and `--graph` won't overwrite a file that isn't an earlier Flinch output, and won't write through a symlink.
- With `hf://`, Flinch only downloads public, ungated repos, pinned to an exact commit. It enforces size limits on every file and never downloads model weights. It doesn't need or send a Hugging Face token.

Flinch isn't a sandbox, though. If you're checking a checkpoint you don't trust, or one that might change while you check it, copy it somewhere stable first. A very large local file can still use a lot of memory.

Reports can include file paths, dataset names and slices of the stats. Look them over before you share a report from a private checkpoint.

## Reporting a vulnerability

Please use [GitHub's private vulnerability reporting](https://github.com/adi271213/flinch/security/advisories/new). If that isn't available, open an issue asking for a private contact, without any exploit details.

It helps to include:

- your Flinch and LeRobot versions and OS;
- which check or option is affected;
- a small made-up checkpoint that reproduces the problem, if you can.

Don't post tokens, private model files or personal paths in a public issue.

This is a small project maintained in spare time, so I can't promise a response time, but security reports get looked at first.
