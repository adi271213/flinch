# Contributing

Read [AGENTS.md](AGENTS.md) before changing processor diagnostics. Correctness against the installed LeRobot implementation takes priority over extending the set of supported steps.

## Local setup

Use Python 3.12 or later and install from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,dev]"
python -m ruff check src scripts tests
python -m pytest -q
```

The test suite generates small checkpoints with native LeRobot classes. It does not require downloaded neural weights, `.lerobot_ref`, a simulator, or hardware. Installed-release tests must not depend on an editable install or inherited `PYTHONPATH`. Test subprocesses that explicitly exercise source-tree imports use `tests/subprocess_env.py`.

Ruff checks syntax, undefined/unused names, import order, and basic Python errors. There is no repository-wide type-checking gate; type annotations are checked through review and relevant runtime tests. Avoid formatting unrelated code or adding broad lint suppressions.

## Diagnostic changes

1. Inspect the actual native processor source, including serialized config/state behavior. Record any upstream version boundary in tests or documentation.
2. Reproduce a verified defect with a regression test and include a healthy control where the distinction matters.
3. Preserve exact serialized step indices, feature keys, offending coordinates, dtype scope, and SHA provenance. Never select a dataset namespace or infer physical units automatically.
4. Keep diagnostics read-only and offline. Only the Hub downloader may perform network requests. Do not construct uninspected processors or fetch tokenizer assets during checks.
5. Return `UNKNOWN` for incomplete evidence, `CONFLICT` for disagreeing artifacts, and a blocking finding for a proven violation. Test the CLI exit behavior as well as the finding.

Use the existing native pipeline hooks and state interfaces. Shared static helpers currently live in `preflight.py`; avoid copying normalization formulas into a second runtime. A structural extraction should have a demonstrated need and preserve native ground-truth tests.

## Build and release checks

```bash
python -m build
python -m twine check --strict dist/*.whl dist/*.tar.gz
```

For a clean installation outside the checkout:

```bash
python scripts/ci_release_check.py --distribution wheel --dist-dir dist \
  --work-dir /path/to/new-empty-work-directory --results-dir ci-results
```

Repeat with `--distribution sdist` and a different empty work directory. On Windows, use an absolute path on the local filesystem; the harness chooses native Python and console-script paths. `--torch-cpu` requests PyTorch's official CPU wheel index on Linux/Windows. Installation requires network access; diagnostic probes remain offline. The work directory must be outside this checkout, and logs must not be committed as source.

The six-job GitHub Actions matrix checks both artifacts on Linux, macOS and Windows with Python 3.12. New source changes require a new run before claiming that revision passes all platforms. Review dependency versions and artifact hashes in the retained logs.

## Repository hygiene

Keep source under `src/lerobot_preflight`, regression tests under `tests`, source tools under `scripts`, and maintained prose under `docs`. Do not commit virtualenvs, cached checkpoints, model weights, bytecode, build products, tokens, or raw private diagnostic evidence. Historical local research archives are not required to install or test the package.

Use your legitimate Git identity and preserve authentic third-party attribution. Do not rewrite published history as part of cleanup. Keep changes focused and explain the verified problem, resulting behavior, and actual validation in the pull request.

The project has no selected license yet. Contributions and publication terms require the repository owner's decision before an open-source release.
