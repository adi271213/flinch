# Contributing to Flinch

Thanks for helping out! Bug reports with a checkpoint that Flinch gets wrong are the most useful thing you can send, so don't worry if you aren't planning to write code.

## Setting up

```bash
git clone https://github.com/adi271213/flinch.git
cd flinch
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[test,dev]"
```

Before you open a pull request, run:

```bash
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m mypy
```

`python -m ruff format src scripts tests` fixes formatting for you. The tests build small checkpoints on the fly, so you don't need a GPU, a robot, a simulator or any downloaded model.

## Ground rules

These are what keep Flinch's answers trustworthy. Please stick to them.

1. **Read LeRobot's code, don't guess it.** Before you touch a check, open the installed `lerobot/processor/` source and look at how the step really behaves and what it saves. Class names, JSON keys and formulas change between versions.
2. **Use LeRobot's own pipeline.** Load checkpoints with `PolicyProcessorPipeline` and watch them with its hooks. Don't copy a normalization formula into Flinch, because a copy can drift from the real thing.
3. **Look, never act.** Checks are read-only. No motor commands, no simulator, no network calls while checking. Only the Hub downloader talks to the network, and it runs before any check starts.
4. **Use the right label.** `FAIL` needs evidence and has to point at the exact step, key and coordinates. `CONFLICT` means two files disagree. `UNKNOWN` means Flinch couldn't tell. When in doubt, say `UNKNOWN`.
5. **Never guess units.** A tensor's shape can't tell you radians from degrees, or which frame an action is in. Leave it `UNKNOWN` unless the user declares it.
6. **Every check needs a broken example and a healthy one.** Add a test that builds a real checkpoint showing the bug, plus a healthy twin that must pass. Check the exit code too, not just the finding.

## Pull requests

- Keep each pull request to one problem or one change.
- Explain what was wrong, what happens now, and which commands you ran.
- Name branches like `fix/stat-namespace` or `docs/install`.
- Write commit messages like `fix: reject conflicting processor aliases` ([Conventional Commits](https://www.conventionalcommits.org/)).
- Don't force-push `main`, and don't rewrite commits that are already published.

## Release checks

CI builds the wheel and source package once, then installs each into a fresh Python 3.12 environment on Linux, macOS and Windows and runs everything there. To do the same locally:

```bash
python -m build
python -m twine check --strict dist/*
python scripts/ci_release_check.py --distribution wheel --dist-dir dist \
  --work-dir /tmp/flinch-wheel-check --results-dir ci-results
```

Run it again with `--distribution sdist` and a different empty `--work-dir` outside the repo. Add `--torch-cpu` on Linux or Windows to get the smaller CPU-only PyTorch.

## What not to commit

Virtual environments, downloaded checkpoints, model weights, build output, tokens, or private checkpoint data. Put personal ignore rules in `.git/info/exclude` instead of `.gitignore`.

## License

By contributing, you agree that your work is released under the [MIT License](LICENSE).
