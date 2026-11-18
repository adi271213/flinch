"""Validate an actual wheel/sdist in an isolated, platform-native environment.

The parent imports only the standard library. Installed CLI smoke tests run
before pytest is installed; native LeRobot creates all diagnostic fixture data
inside the new environment, without importing this checkout or its tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import venv
import zipfile
from pathlib import Path, PurePosixPath

_ORIGINS = r"""
from pathlib import Path
import importlib.metadata as metadata
import hashlib, json, os, site, sys
import lerobot, lerobot_flinch, torch
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline
prefix, checkout = Path(sys.prefix).resolve(), Path(sys.argv[1]).resolve()
origins = {"lerobot": lerobot.__file__, "lerobot_flinch": lerobot_flinch.__file__, "torch": torch.__file__}
assert sys.prefix != sys.base_prefix
assert not site.ENABLE_USER_SITE
assert "PYTHONPATH" not in os.environ
assert all(Path(path).resolve().is_relative_to(prefix) for path in origins.values())
assert not any(".lerobot_ref" in path or Path(path).resolve().is_relative_to(checkout) for path in sys.path)
assert not any("__editable__" in repr(finder) for finder in sys.meta_path)
expected_runtime = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
package = Path(lerobot_flinch.__file__).resolve().parent
actual_runtime = {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in package.rglob("*.py") if path.is_file()}
assert expected_runtime and "__init__.py" in expected_runtime, "Artifact contains no Python runtime package"
mismatches = {name: {"expected": expected_runtime.get(name), "installed": actual_runtime.get(name)}
              for name in sorted(set(expected_runtime) | set(actual_runtime))
              if expected_runtime.get(name) != actual_runtime.get(name)}
assert not mismatches, "Installed runtime bytes differ from the exact input artifact: " + json.dumps(mismatches)
print(json.dumps({"prefix": sys.prefix, "python": sys.version, "origins": origins, "sys_path": sys.path,
                  "runtime_module_count": len(actual_runtime), "runtime_module_hashes": actual_runtime,
                  "installed_runtime_matches_input_artifact": True,
                  "versions": {name: metadata.version(name) for name in
                    ("lerobot-flinch", "lerobot", "torch", "numpy", "huggingface-hub", "safetensors")}}, indent=2))
"""


_FIXTURES = r"""
from pathlib import Path
import json, shutil, socket
import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (AbsoluteActionsProcessorStep, NormalizerProcessorStep,
    PolicyProcessorPipeline, RelativeActionsProcessorStep, UnnormalizerProcessorStep)
from safetensors.torch import load_file, save_file

def denied(*args, **kwargs):
    raise AssertionError("Native fixture generation must remain offline")
socket.socket = socket.create_connection = denied
root = Path(__import__("sys").argv[1])
root.mkdir(parents=True, exist_ok=True)
features = {"observation.state": PolicyFeature(FeatureType.STATE, (14,)),
            "action": PolicyFeature(FeatureType.ACTION, (14,))}
mapping = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
policy = {"type": "act", "n_obs_steps": 1, "n_action_steps": 50, "chunk_size": 100,
    "device": "cpu", "use_amp": False,
    "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
    "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    "normalization_mapping": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD", "VISUAL": "IDENTITY"}}
observed = {}
for name in ("clean", "silent_bypass", "pathological_gain", "fp16_nan"):
    path = root / name
    path.mkdir()
    std = torch.full((14,), .5)
    if name == "pathological_gain": std[9] = 1.48e-6
    if name == "fp16_nan": std[4] = 0.
    key = "wrong_dataset/observation.state" if name == "silent_bypass" else "observation.state"
    stats = {key: {"mean": torch.zeros(14), "std": std},
             "action": {"mean": torch.zeros(14), "std": torch.full((14,), .5)}}
    pre = PolicyProcessorPipeline(name="policy_preprocessor", steps=[NormalizerProcessorStep(
        features=features, norm_map=mapping, stats=stats)])
    post = PolicyProcessorPipeline(name="policy_postprocessor", steps=[UnnormalizerProcessorStep(
        features={"action": features["action"]}, norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": stats["action"]})])
    pre.save_pretrained(path, config_filename="policy_preprocessor.json")
    post.save_pretrained(path, config_filename="policy_postprocessor.json")
    (path / "config.json").write_text(json.dumps(policy), encoding="utf-8")
    sample = np.zeros((1, 14), dtype=np.float16) if name == "fp16_nan" else np.full((1, 14), .20, dtype=np.float32)
    np.savez(path / "sample.npz", **{"observation.state": sample})
    np.savez(path / "sample_action.npz", action=np.full((1, 14), .4, dtype=np.float32))
    result = pre({"observation.state": torch.from_numpy(sample.copy())})["observation.state"]
    if name == "silent_bypass": assert torch.equal(result, torch.from_numpy(sample))
    if name == "pathological_gain": assert result[0, 9] > 100_000
    if name == "fp16_nan": assert torch.isnan(result[0, 4])
    observed[name] = {"bit_identical": bool(torch.equal(result, torch.from_numpy(sample))),
                      "nonfinite_coordinates": (~torch.isfinite(result)).nonzero().tolist()}
    if name == "pathological_gain": observed[name]["output_coordinate_9"] = float(result[0, 9])
for name, deltas, normalized in (("oldest_first", [-2, -1, 0], False),
        ("current_first", [0, 1, 2], False), ("current_first_normalized", [0, 1, 2], True)):
    path = root / name
    path.mkdir()
    relative = RelativeActionsProcessorStep(enabled=True)
    pre_steps, post_steps = [], [AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)]
    if normalized:
        stats = {"observation.state": {"mean": [10.] * 14, "std": [2.] * 14},
                 "action": {"mean": [5.] * 14, "std": [2.] * 14}}
        pre_steps.append(NormalizerProcessorStep(features=features, norm_map=mapping, stats=stats))
        post_steps.append(UnnormalizerProcessorStep(features={"action": features["action"]},
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD}, stats={"action": stats["action"]}))
    pre_steps.append(relative)
    pre, post = (PolicyProcessorPipeline(name="policy_preprocessor", steps=pre_steps),
                 PolicyProcessorPipeline(name="policy_postprocessor", steps=post_steps))
    pre.save_pretrained(path, config_filename="policy_preprocessor.json")
    post.save_pretrained(path, config_filename="policy_postprocessor.json")
    config = {**policy, "n_obs_steps": 3, "observation_delta_indices": deltas}
    if not normalized: config["normalization_mapping"] = {"STATE": "IDENTITY", "ACTION": "IDENTITY", "VISUAL": "IDENTITY"}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    state = torch.tensor([.20, .45, .70]).reshape(1, 3, 1).expand(1, 3, 14).clone()
    action = torch.full((1, 1, 14), .90)
    np.savez(path / "sample.npz", **{"observation.state": state.numpy()})
    np.savez(path / "sample_action.npz", action=action.numpy())
    result = pre({"observation.state": state, "action": action})
    restored = post({"action": result["action"]})["action"]
    assert torch.allclose(restored, action, atol=1e-5)
    observed[name] = {"native_roundtrip_passed": True}
candidate = root / "expanded_candidate"
shutil.copytree(root / "clean", candidate)
post = json.loads((candidate / "policy_postprocessor.json").read_text())
state_file = candidate / post["steps"][0]["state_file"]
values = load_file(state_file)
values["action.std"][[2, 6]] *= 1.5
save_file(values, state_file)
(candidate / "config.json").write_text(json.dumps({**policy, "n_action_steps": 100}), encoding="utf-8")
conflict = root / "action_stats_conflict"
shutil.copytree(root / "clean", conflict)
post_config = json.loads((conflict / "policy_postprocessor.json").read_text())
state_file = conflict / post_config["steps"][0]["state_file"]
values = load_file(state_file)
values["action.std"][3] = 1e9
save_file(values, state_file)
pre = PolicyProcessorPipeline.from_pretrained(
    conflict, config_filename="policy_preprocessor.json", local_files_only=True)
post = PolicyProcessorPipeline.from_pretrained(
    conflict, config_filename="policy_postprocessor.json", local_files_only=True)
raw_action = torch.full((1, 14), .4)
restored = post(pre({"action": raw_action}))["action"]
assert restored[0, 3] > 1e8
observed["action_stats_conflict"] = {"native_output_coordinate_3": float(restored[0, 3]),
                                     "raw_input_coordinate_3": float(raw_action[0, 3])}
corrupt = root / "corrupt_stats"
shutil.copytree(root / "clean", corrupt)
pre_config = json.loads((corrupt / "policy_preprocessor.json").read_text())
corrupt_state = corrupt / pre_config["steps"][0]["state_file"]
corrupt_state.write_bytes(corrupt_state.read_bytes()[:12])
try:
    PolicyProcessorPipeline.from_pretrained(
        corrupt, config_filename="policy_preprocessor.json", local_files_only=True)
except Exception as exc:
    observed["corrupt_stats"] = {"native_load_error": type(exc).__name__}
else:
    raise AssertionError("Native loading must reject truncated normalization state")
legacy = root / "legacy_pathological"
legacy.mkdir()
legacy_std = torch.full((14,), .5)
legacy_std[4] = 6.6630414e-6
save_file({"model.weight": torch.zeros(1024),
           "normalize_inputs.buffer_observation_state.mean": torch.zeros(14),
           "normalize_inputs.buffer_observation_state.std": legacy_std,
           "unnormalize_outputs.buffer_action.mean": torch.zeros(14),
           "unnormalize_outputs.buffer_action.std": torch.full((14,), .5)}, legacy / "model.safetensors")
(legacy / "config.json").write_text(json.dumps(policy), encoding="utf-8")
np.savez(root / "invalid_shape.npz", **{"observation.state": np.zeros((1, 1), dtype=np.float32)})
(root / "native_ground_truth.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
assert not any(name == "lerobot_flinch" or name.startswith("lerobot_flinch.") for name in __import__("sys").modules)
print(json.dumps(observed, indent=2))
"""


_GUARD_FIXTURES = r"""
from pathlib import Path
import json, shutil, socket, sys
import numpy as np
import torch
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
import lerobot_flinch.guard as guard_module
from lerobot_flinch.guard import InputEnvelopeError, add_input_guard

assert Path(guard_module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
def denied(*args, **kwargs):
    raise AssertionError("Guard fixture generation and execution must remain offline")
socket.socket = socket.create_connection = denied
root = Path(sys.argv[1])
observed = {}
def load(path):
    return PolicyProcessorPipeline.from_pretrained(path, config_filename="policy_preprocessor.json",
        local_files_only=True, to_transition=batch_to_transition, to_output=transition_to_batch)
for name, source in (("guarded_clean", "clean"), ("guarded_gain", "pathological_gain"),
                     ("guarded_nan", "fp16_nan"), ("guarded_relative", "current_first_normalized")):
    path = root / name
    shutil.copytree(root / source, path)
    pre = add_input_guard(load(path), max_abs=10.0)
    pre.save_pretrained(path, config_filename="policy_preprocessor.json")
    pre = load(path)
    with np.load(path / "sample.npz", allow_pickle=False) as data:
        sample = {"observation.state": torch.from_numpy(data["observation.state"].copy())}
    try:
        output = pre(sample)["observation.state"]
    except InputEnvelopeError as exc:
        assert name in {"guarded_gain", "guarded_nan"}
        observed[name] = {"guard_stopped_input": True, "key": exc.key,
                          "indices": exc.indices, "total_count": exc.total_count}
    else:
        assert name in {"guarded_clean", "guarded_relative"}
        assert torch.isfinite(output).all() and output.abs().max() <= 10
        observed[name] = {"guard_stopped_input": False, "output_max_abs": float(output.abs().max())}
weakened = root / "guard_weakened"
shutil.copytree(root / "guarded_clean", weakened)
path = weakened / "policy_preprocessor.json"
config = json.loads(path.read_text())
next(step for step in config["steps"] if step.get("class", "").endswith(".InputEnvelopeGuardStep"))["config"]["max_abs"] = 100.0
path.write_text(json.dumps(config), encoding="utf-8")
sample = {"observation.state": torch.full((1, 14), 20.0)}
try:
    load(root / "guarded_clean")(sample)
except InputEnvelopeError:
    pass
else:
    raise AssertionError("The baseline envelope must reject this input")
output = load(weakened)(sample)["observation.state"]
assert output.abs().max() > 10 and output.abs().max() <= 100
observed["guard_weakened"] = {"baseline_rejected": True, "candidate_max_abs": float(output.abs().max())}
(root / "guard_ground_truth.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
print(json.dumps(observed, indent=2))
"""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_hashes(artifact: Path) -> dict[str, str]:
    """Read package source bytes from the chosen archive without importing it."""
    hashes: dict[str, str] = {}

    def add(relative: tuple[str, ...], data: bytes) -> None:
        if not relative or any(part in (".", "..") or "\\" in part for part in relative):
            raise ValueError("Unsafe Python module path in release artifact")
        name = "/".join(relative)
        if name in hashes:
            raise ValueError(f"Duplicate Python module in release artifact: {name}")
        hashes[name] = hashlib.sha256(data).hexdigest()

    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            for member in archive.infolist():
                parts = PurePosixPath(member.filename).parts
                if len(parts) >= 2 and parts[0] == "lerobot_flinch" and member.filename.endswith(".py"):
                    add(parts[1:], archive.read(member))
    else:
        with tarfile.open(artifact) as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if (
                    member.isfile()
                    and len(parts) >= 4
                    and parts[1:3] == ("src", "lerobot_flinch")
                    and member.name.endswith(".py")
                ):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Cannot read Python module from source artifact: {member.name}")
                    with stream:
                        add(parts[3:], stream.read())
    if "__init__.py" not in hashes:
        raise ValueError("Release artifact contains no lerobot_flinch Python package")
    return hashes


def _environment(work: Path, *, offline: bool) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONUSERBASE", "PYTHONUTF8"}
        and (not key.startswith("PIP_") or key == "PIP_CACHE_DIR")
    }
    # Keep the platform's filesystem/default-text encoding under test. Only
    # captured console streams use UTF-8, so Windows CP1252 regressions remain visible.
    environment.update(
        PYTHONNOUSERSITE="1",
        PYTHONIOENCODING="utf-8",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        HF_HOME=str(work / "empty-hf-cache"),
    )
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if offline:
            environment[key] = "1"
        else:
            environment.pop(key, None)
    return environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=("wheel", "sdist"), required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "install", "smoke", "pytest"), default="all")
    parser.add_argument(
        "--torch-cpu", action="store_true", help="Resolve CPU PyTorch from its official extra index"
    )
    args = parser.parse_args(argv)
    work, results = args.work_dir.resolve(), args.results_dir.resolve()
    dist = args.dist_dir.resolve()
    checkout = Path(__file__).resolve().parents[1]
    if work.is_relative_to(checkout):
        parser.error("--work-dir must be outside the checkout")
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    summary_path = results / "summary.json"
    if args.phase != "all" and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "distribution": args.distribution,
            "platform": platform.platform(),
            "python": sys.version,
            "work_dir": str(work),
            "phase": args.phase,
            "commands": [],
            "validations": [],
            "pythonpath_present_during_diagnostics": False,
            "system_site_packages": False,
            "torch_cpu_extra_index": args.torch_cpu,
            "neural_policy_execution": False,
        }
    environment = _environment(work, offline=False)
    runtime_environment = _environment(work, offline=True)
    environment_dir = work / "installed environment"
    scripts = environment_dir / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    console = scripts / ("flinch.exe" if os.name == "nt" else "flinch")

    def save() -> None:
        summary["passed"] = (
            not summary.get("error")
            and all(item["passed"] for item in summary["commands"])
            and all(item["passed"] for item in summary["validations"])
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def validate(name: str, condition: bool, details: object = None) -> bool:
        summary["validations"].append({"name": name, "passed": condition, "details": details})
        save()
        if not condition:
            print(f"FAIL: {name}: {details}", flush=True)
        return condition

    def run(
        name: str,
        command: list[str | Path],
        expected: int = 0,
        *,
        offline: bool = True,
        cwd: Path = work,
        timeout: int = 300,
    ) -> bool:
        command = [str(part) for part in command]
        stdout, stderr = results / f"{name}.stdout.txt", results / f"{name}.stderr.txt"
        print(f"Running {name}", flush=True)
        started = time.monotonic()
        timed_out = False
        with stdout.open("w", encoding="utf-8") as output, stderr.open("w", encoding="utf-8") as error:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=runtime_environment if offline else environment,
                    stdout=output,
                    stderr=error,
                )
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    code, timed_out = -1, True
                    error.write(f"\nTimed out after {timeout} seconds.\n")
            except OSError as exc:
                error.write(f"{type(exc).__name__}: {exc}\n")
                code = -1
        passed = code == expected and not timed_out
        summary["commands"].append(
            {
                "name": name,
                "command": command,
                "cwd": str(cwd),
                "exit_code": code,
                "expected_exit_code": expected,
                "passed": passed,
                "timed_out": timed_out,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "offline_flags": offline,
                "stdout": stdout.name,
                "stderr": stderr.name,
            }
        )
        save()
        print(f"{name}: {'PASS' if passed else 'FAIL'} (exit {code}, expected {expected})", flush=True)
        if not passed:
            print(stdout.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
            print(stderr.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
        return passed

    try:
        wheels, sdists = (
            sorted(dist.glob("lerobot_flinch-*.whl")),
            sorted(dist.glob("lerobot_flinch-*.tar.gz")),
        )
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError("dist directory must contain exactly one wheel and one sdist")
        artifact, source_archive = (wheels[0] if args.distribution == "wheel" else sdists[0]), sdists[0]
        summary["artifacts"] = {path.name: _hash(path) for path in (wheels[0], source_archive)}
        expected_runtime = _runtime_hashes(artifact)
        runtime_manifest = results / "expected-runtime-hashes.json"
        runtime_manifest.write_text(json.dumps(expected_runtime, indent=2) + "\n", encoding="utf-8")
        summary["expected_runtime_module_hashes"] = expected_runtime
        manifest_path = dist / "artifact_hashes.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not validate(
                "shared_artifact_hashes",
                all(manifest.get(name) == digest for name, digest in summary["artifacts"].items()),
                summary["artifacts"],
            ):
                return 1
        if args.phase in ("all", "install"):
            if environment_dir.exists():
                raise FileExistsError(
                    f"Refusing to reuse a pre-existing install environment: {environment_dir}"
                )
            venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment_dir)
            pip_options = [
                "--index-url",
                "https://pypi.org/simple",
                "--disable-pip-version-check",
                "--progress-bar",
                "off",
                "--timeout",
                "45",
                "--retries",
                "2",
            ]
            if args.torch_cpu:
                pip_options += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            if not run(
                "install",
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    *pip_options,
                    "--report",
                    results / "pip-install.json",
                    artifact,
                ],
                offline=False,
                timeout=1500,
            ):
                return 1
            summary["installed_artifact_sha256"] = _hash(artifact)
        elif not python.is_file() or summary.get("installed_artifact_sha256") != _hash(artifact):
            raise ValueError("A successful install phase of this exact artifact is required first")
        if args.phase in ("all", "smoke"):
            origins_ok = run("import_origins", [python, "-I", "-c", _ORIGINS, checkout, runtime_manifest])
            run("pip_check", [python, "-m", "pip", "check"])
            run("version", [console, "--version"])
            run("help", [console, "--help"])
            fixtures = work / "native fixtures with spaces"
            generated = origins_ok and run(
                "generate_native_fixtures", [python, "-I", "-c", _FIXTURES, fixtures]
            )
            if generated:
                generated = run("generate_guard_fixtures", [python, "-I", "-c", _GUARD_FIXTURES, fixtures])
            if generated:
                hashes = {
                    str(path.relative_to(fixtures)): _hash(path)
                    for path in fixtures.rglob("*")
                    if path.is_file()
                }

                def case(
                    name: str,
                    checkpoint: str,
                    expected: int,
                    checks: tuple[str, ...] = (),
                    *,
                    sample: bool = True,
                    conflicts: tuple[str, ...] = (),
                    tool_errors: bool = False,
                    extra: tuple[str | Path, ...] = (),
                ) -> None:
                    report = results / f"{name}.report.json"
                    command: list[str | Path] = [
                        console,
                        "--policy",
                        fixtures / checkpoint,
                        "--report",
                        report,
                    ]
                    if sample:
                        command += ["--sample", fixtures / checkpoint / "sample.npz"]
                    run(name, [*command, *extra], expected)
                    try:
                        document = json.loads(report.read_text(encoding="utf-8"))
                        findings = document["findings"]
                        validate(name + "_report_exit_code", document.get("exit_code") == expected)
                        validate(
                            name + "_check_ids",
                            all(
                                any(
                                    finding["check_id"] == check and finding["status"] == "FAIL"
                                    for finding in findings
                                )
                                for check in checks
                            ),
                            list(checks),
                        )
                        if conflicts:
                            validate(
                                name + "_conflict_check_ids",
                                all(
                                    any(
                                        finding["check_id"] == check and finding["status"] == "CONFLICT"
                                        for finding in findings
                                    )
                                    for check in conflicts
                                ),
                                list(conflicts),
                            )
                        if tool_errors:
                            validate(
                                name + "_tool_error_retained",
                                any(finding.get("details", {}).get("tool_error") for finding in findings),
                            )
                    except (OSError, KeyError, ValueError) as exc:
                        validate(name + "_json_report", False, str(exc))

                case("clean", "clean", 0)
                case("clean_strict", "clean", 0, extra=("--strict",))
                case("silent_bypass", "silent_bypass", 1, ("SILENT_STEP_BYPASS",))
                case("pathological_gain", "pathological_gain", 1, ("EMPIRICAL_GAIN_EXPLOSION",))
                case("fp16_nan", "fp16_nan", 1, ("RUNTIME_NON_FINITE",))
                case("corrupt_stats", "corrupt_stats", 1, ("FEATURE_CONTRACT",), sample=False)
                case(
                    "local_legacy",
                    "legacy_pathological",
                    1,
                    ("SCHEMA_GENERATION_MISMATCH", "PATHOLOGICAL_STD_GAIN"),
                    sample=False,
                )
                validate(
                    "local_legacy_header_digest_scope",
                    "digest_scope=safetensors_header_bytes_including_length_prefix"
                    in (results / "local_legacy.stdout.txt").read_text(encoding="utf-8"),
                )
                case(
                    "action_stats_conflict",
                    "action_stats_conflict",
                    1,
                    sample=False,
                    conflicts=("ACTION_STATS_CONFLICT",),
                )
                case("oldest_first", "oldest_first", 1, ("TEMPORAL_ANCHOR_MISMATCH",))
                case("guarded_clean", "guarded_clean", 0)
                case(
                    "guarded_gain",
                    "guarded_gain",
                    1,
                    ("EMPIRICAL_GAIN_EXPLOSION", "INPUT_ENVELOPE_VIOLATION"),
                )
                case("guarded_nan", "guarded_nan", 1, ("RUNTIME_NON_FINITE",))
                for name in ("guarded_clean", "guarded_relative"):
                    case(
                        name + "_self_strict",
                        name,
                        0,
                        sample=False,
                        extra=("--baseline", fixtures / name, "--strict"),
                    )
                case(
                    "guard_weakened",
                    "guard_weakened",
                    1,
                    ("PRIVILEGE_GUARD_REMOVAL",),
                    sample=False,
                    extra=("--baseline", fixtures / "guarded_clean"),
                )
                for name in ("current_first", "current_first_normalized"):
                    case(name, name, 0)
                    case(name + "_self", name, 0, sample=False, extra=("--baseline", fixtures / name))
                    case(
                        name + "_self_strict",
                        name,
                        0,
                        sample=False,
                        extra=("--baseline", fixtures / name, "--strict"),
                    )
                case(
                    "privilege_expansion",
                    "expanded_candidate",
                    1,
                    ("PRIVILEGE_ACTION_GAIN_ESCALATION", "PRIVILEGE_HORIZON_ESCALATION"),
                    extra=("--baseline", fixtures / "clean"),
                )
                case(
                    "invalid_sample_shape",
                    "clean",
                    1,
                    ("INVALID_SAMPLE_SHAPE",),
                    sample=False,
                    extra=("--sample", fixtures / "invalid_shape.npz"),
                )
                for extension in ("json", "mmd"):
                    graph = results / f"processor_graph.{extension}"
                    case("graph_" + extension, "clean", 0, extra=("--graph", graph))
                    validate("graph_" + extension + "_created", graph.is_file())
                (fixtures / "empty").mkdir()
                case("empty_checkpoint", "empty", 1, ("SCHEMA_GENERATION",), sample=False)
                missing_report = results / "missing_checkpoint.report.json"
                run(
                    "missing_checkpoint",
                    [console, "--policy", fixtures / "missing", "--report", missing_report],
                    2,
                )
                missing_error = (results / "missing_checkpoint.stderr.txt").read_text(encoding="utf-8")
                validate(
                    "missing_checkpoint_usage_error",
                    "--policy" in missing_error
                    and "does not exist" in missing_error
                    and "Traceback" not in missing_error
                    and not missing_report.exists(),
                )
                for name, option, value in (
                    ("invalid_threshold", "--std-threshold", "nan"),
                    ("invalid_graph_suffix", "--graph", results / "invalid.svg"),
                ):
                    run(name, [console, "--policy", fixtures / "clean", option, value], 2)
                    error = (results / f"{name}.stderr.txt").read_text(encoding="utf-8")
                    validate(name + "_usage_error", option in error and "Traceback" not in error)
                hub_report = results / "hub_offline_tool_error.report.json"
                run(
                    "hub_offline_tool_error",
                    [
                        console,
                        "--policy",
                        "hf://flinch-test/offline-checkpoint",
                        "--report",
                        hub_report,
                    ],
                    3,
                )
                try:
                    hub_document = json.loads(hub_report.read_text(encoding="utf-8"))
                    validate(
                        "hub_offline_structured_tool_error",
                        hub_document.get("exit_code") == 3
                        and any(
                            finding.get("details", {}).get("tool_error")
                            for finding in hub_document["findings"]
                        )
                        and not any(
                            finding["status"] in {"FAIL", "CONFLICT"} for finding in hub_document["findings"]
                        ),
                    )
                except (OSError, KeyError, ValueError) as exc:
                    validate("hub_offline_structured_tool_error", False, str(exc))
                foreign_report = results / "unrelated-report.json"
                foreign_report.write_text('{"user_data": true}\n', encoding="utf-8")
                foreign_report_hash = _hash(foreign_report)
                for checkpoint, expected in (("clean", 3), ("pathological_gain", 1)):
                    name = checkpoint + "_report_tool_error"
                    run(
                        name,
                        [console, "--policy", fixtures / checkpoint, "--report", foreign_report],
                        expected,
                    )
                    validate(name + "_preserves_foreign_output", _hash(foreign_report) == foreign_report_hash)
                foreign_graph = results / "unrelated-graph.mmd"
                foreign_graph.write_text("flowchart LR\n  user --> data\n", encoding="utf-8")
                foreign_graph_hash = _hash(foreign_graph)
                for checkpoint, expected in (("clean", 3), ("pathological_gain", 1)):
                    name = checkpoint + "_graph_tool_error"
                    case(
                        name,
                        checkpoint,
                        expected,
                        ("PATHOLOGICAL_STD_GAIN",) if expected == 1 else (),
                        sample=False,
                        tool_errors=True,
                        extra=("--graph", foreign_graph),
                    )
                    validate(name + "_preserves_foreign_output", _hash(foreign_graph) == foreign_graph_hash)
                validate(
                    "checkpoint_inputs_unchanged",
                    hashes
                    == {
                        str(path.relative_to(fixtures)): _hash(path)
                        for path in fixtures.rglob("*")
                        if path.is_file()
                    },
                )
        if args.phase in ("all", "pytest"):
            # Test-only dependencies are added only after installed CLI proof.
            if run(
                "install_pytest",
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    "https://pypi.org/simple",
                    "--disable-pip-version-check",
                    "--progress-bar",
                    "off",
                    "--timeout",
                    "45",
                    "--retries",
                    "2",
                    "pytest>=8",
                ],
                offline=False,
                timeout=300,
            ):
                source = work / "extracted source archive"
                with tarfile.open(source_archive) as archive:
                    archive.extractall(source, filter="data")
                projects = list(source.glob("*/pyproject.toml"))
                if len(projects) != 1:
                    raise ValueError("Source archive must contain one project root")
                # Keep the installed package under test; sources are present for
                # the explicit source-portability tests and package metadata.
                run(
                    "pytest",
                    [
                        python,
                        "-m",
                        "pytest",
                        "-q",
                        "-o",
                        "pythonpath=",
                        "--junitxml",
                        results / "pytest-junit.xml",
                    ],
                    cwd=projects[0].parent,
                    timeout=900,
                )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(summary["error"], flush=True)
    finally:
        save()
        lines = [
            f"### {platform.system()} / Python {sys.version_info.major}.{sys.version_info.minor} / {args.distribution}",
            "",
            f"**{'PASS' if summary['passed'] else 'FAIL'}**",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
        lines += [
            f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} |" for item in summary["commands"]
        ]
        if summary.get("error"):
            lines += ["", "Error: " + summary["error"]]
        markdown = "\n".join(lines) + "\n"
        (results / "summary.md").write_text(markdown, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
                output.write(markdown)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
