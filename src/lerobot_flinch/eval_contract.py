"""Does a checkpoint match the environment it will be evaluated in and the tasks it learned?

Every check here comes from a mismatch LeRobot users reported, measured and could not see in
the logs. All of them read files only: the checkpoint's config and processor JSON, LeRobot's
own env config and processor factory source, and the dataset's metadata. Nothing is run.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from lerobot.configs import FeatureType

from .models import DiagnosticFinding, DiagnosticStatus, Severity

ISSUES = "https://github.com/huggingface/lerobot/issues/"


def _finding(
    check: str, status: DiagnosticStatus, summary: str, details: dict[str, Any]
) -> DiagnosticFinding:
    severity = (
        Severity.CRITICAL
        if status in (DiagnosticStatus.FAIL, DiagnosticStatus.CONFLICT)
        else Severity.WARNING
        if details.get("advisory")
        else Severity.INFO
    )
    return DiagnosticFinding(check, status, severity, summary, details)


def _visual_inputs(config: dict[str, Any]) -> dict[str, list[int]]:
    features = config.get("input_features") or {}
    return {
        key: list(spec.get("shape") or [])
        for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("type") == FeatureType.VISUAL.value
    }


# --- Checkpoint only -------------------------------------------------------------------------


# Steps whose absence changes nothing on LeRobot's own loading paths.
_HARMLESS_WHEN_ABSENT = {
    # The model keeps the same image prep as idempotent guards (ImagePrepProcessorStep docstring).
    "vla_jepa_image_prep",
    # A no-op on images already in [0, 1], which LeRobot's env and robot paths deliver.
    "xvla_image_to_float",
    # The step's default domain_id is 0, the model's own fallback without it (_get_domain_id).
    "xvla_add_domain_id",
}


def _factory_steps(policy_type: str) -> tuple[set[str], set[str]] | None:
    """Registry names of the steps LeRobot's own factory builds, read from its source.

    Returns (always built, built only under a condition), or None when there is no factory
    or LeRobot rebuilds this policy's processors at load time instead of using the saved files.
    """
    try:
        module = importlib.import_module(f"lerobot.policies.{policy_type}.processor_{policy_type}")
        if hasattr(module, f"make_{policy_type}_pre_post_processors_from_pretrained"):
            return None
        factory = getattr(module, f"make_{policy_type}_pre_post_processors")
        tree = ast.parse(inspect.getsource(factory))
    except (ImportError, AttributeError, OSError, SyntaxError):
        return None
    always: set[str] = set()
    conditional: set[str] = set()

    def visit(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = guarded or isinstance(child, (ast.If, ast.IfExp))
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                cls = getattr(module, child.func.id, None)
                name = getattr(cls, "_registry_name", None)
                # A step built with enabled=<config flag> is switched off by default, so a saved
                # pipeline without it is fine (e.g. relative actions with use_relative_actions=False).
                switched = any(keyword.arg == "enabled" for keyword in child.keywords)
                if isinstance(name, str):
                    (conditional if inner or switched else always).add(name)
            visit(child, inner)

    visit(tree, False)
    return always, conditional - always


def _env_steps(env: Any, policy_type: str) -> set[str]:
    """Steps lerobot-eval's env processor adds for this env and policy type (e.g. X-VLA on LIBERO)."""
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env_pre_post_processors

        importlib.import_module(f"lerobot.policies.{policy_type}.configuration_{policy_type}")
        policy_class = PreTrainedConfig.get_choice_class(policy_type)
        # The factory only checks the config's class, so an unparsed instance is enough.
        preprocessor, _ = make_env_pre_post_processors(env, policy_class.__new__(policy_class))
        return {
            name
            for step in preprocessor.steps
            if isinstance(name := getattr(step, "_registry_name", None), str)
        }
    except Exception:  # noqa: BLE001 - LeRobot internals vary by version; fall back to the saved files.
        return set()


def check_processor_steps(
    checkpoint: Path, config: dict[str, Any], env: Any | None = None
) -> list[DiagnosticFinding]:
    """#3674, #2954: saved processors missing steps the factory adds; rollouts silently fail."""
    policy_type = config.get("type")
    saved: set[str] = set()
    found_files = False
    for path in sorted(checkpoint.glob("*processor*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and isinstance(document.get("steps"), list):
            found_files = True
            for step in document["steps"]:
                if isinstance(step, dict) and isinstance(step.get("registry_name"), str):
                    saved.add(step["registry_name"])
    if not found_files or not isinstance(policy_type, str):
        return []  # Old-format checkpoints are already reported by the preflight checks.
    expected = _factory_steps(policy_type)
    if expected is None:
        return []  # Not a built-in LeRobot policy type: there is no reference pipeline to compare.
    always, conditional = expected
    supplied = saved | (_env_steps(env, policy_type) if env is not None else set())
    missing, maybe = sorted(always - supplied - _HARMLESS_WHEN_ABSENT), sorted(conditional - supplied)
    if missing:
        where = (
            f"and the {env.type} env's processor doesn't add them either"
            if env is not None
            else "unless your environment's processor adds them"
        )
        return [
            _finding(
                "PROCESSOR_STEPS_MISSING",
                DiagnosticStatus.CONFLICT,
                f"The saved processors lack steps LeRobot's {policy_type} factory always adds: "
                f"{', '.join(missing)}, {where}. Inputs will not match training. See {ISSUES}3674.",
                {"policy_type": policy_type, "missing_steps": missing, "conditional_missing": maybe},
            )
        ]
    return [
        _finding(
            "PROCESSOR_STEPS_MISSING",
            DiagnosticStatus.PASS,
            f"The saved processors contain every step LeRobot's {policy_type} factory always adds."
            + (f" Steps added only in some configurations: {', '.join(maybe)}." if maybe else ""),
            {"policy_type": policy_type, "conditional_missing": maybe},
        )
    ]


# --- Against an environment -----------------------------------------------------------------


def load_env_config(env_type: str, options: dict[str, str]) -> Any:
    """LeRobot's own env config, with overrides. Importing it starts no simulator."""
    from lerobot.envs.configs import EnvConfig

    cls = EnvConfig.get_choice_class(env_type)
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(f"Unknown option(s) for env {env_type!r}: {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for key, raw in options.items():
        try:
            values[key] = json.loads(raw)
        except ValueError:
            values[key] = raw
    return cls(**values)


def _env_cameras(env: Any) -> list[str] | None:
    for name in ("camera_name", "camera_names"):
        value = getattr(env, name, None)
        if isinstance(value, str):
            return [camera.strip() for camera in value.split(",") if camera.strip()]
        if isinstance(value, (list, tuple)):
            return [str(camera) for camera in value]
    return None


def check_env(config: dict[str, Any], env: Any) -> list[DiagnosticFinding]:
    """#5185 (image size) and #5118, #4245 (camera count)."""
    findings: list[DiagnosticFinding] = []
    visual = _visual_inputs(config)
    height, width = getattr(env, "observation_height", None), getattr(env, "observation_width", None)
    if visual and isinstance(height, int) and isinstance(width, int):
        wrong = {
            key: shape for key, shape in visual.items() if len(shape) == 3 and shape[1:] != [height, width]
        }
        findings.append(
            _finding(
                "ENV_IMAGE_SIZE_MISMATCH",
                DiagnosticStatus.CONFLICT if wrong else DiagnosticStatus.PASS,
                (
                    f"The {env.type} env renders {height}x{width} images, but the checkpoint was trained on "
                    + ", ".join(f"{key} {shape[1]}x{shape[2]}" for key, shape in wrong.items())
                    + f". Set --env-option observation_height/observation_width to match. See {ISSUES}5185."
                )
                if wrong
                else f"The {env.type} env renders {height}x{width}, matching the checkpoint's images.",
                {"env_size": [height, width], "checkpoint_sizes": visual},
            )
        )
    cameras = _env_cameras(env)
    if visual and cameras is not None:
        empty = config.get("empty_cameras")
        placeholders = empty if isinstance(empty, int) else 0
        expected = len(visual) - placeholders
        status = DiagnosticStatus.CONFLICT if len(cameras) < expected else DiagnosticStatus.PASS
        findings.append(
            _finding(
                "ENV_CAMERA_COUNT_MISMATCH",
                status,
                (
                    f"The {env.type} env provides {len(cameras)} camera(s) ({', '.join(cameras)}), but the "
                    f"checkpoint expects {expected}. Some policies then shift the remaining images into the "
                    f"wrong slots without warning. See {ISSUES}5118."
                )
                if status is DiagnosticStatus.CONFLICT
                else f"The {env.type} env provides {len(cameras)} camera(s) for {expected} expected.",
                {"env_cameras": cameras, "checkpoint_cameras": sorted(visual), "empty_cameras": placeholders},
            )
        )
    return findings


# --- Against a dataset ----------------------------------------------------------------------


def read_dataset_tasks(root: Path) -> list[str] | None:
    """The dataset's task strings, or None when they can't be read."""
    tasks: list[str] | None = None
    jsonl = root / "meta" / "tasks.jsonl"
    parquet = root / "meta" / "tasks.parquet"
    if jsonl.is_file():
        tasks = [
            str(json.loads(line).get("task"))
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line
        ]
    elif parquet.is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            tasks = None
        else:
            table = pq.read_table(parquet)
            column = "task" if "task" in table.column_names else "__index_level_0__"
            if column in table.column_names:
                tasks = [str(task) for task in table.column(column).to_pylist()]
    return tasks


def check_task_text(tasks: list[str] | None) -> list[DiagnosticFinding]:
    """#4045, #4548: a policy trained on bare task IDs, evaluated with sentences."""
    findings: list[DiagnosticFinding] = []
    if tasks is None:
        findings.append(
            _finding(
                "TASK_TEXT_FORMAT",
                DiagnosticStatus.UNKNOWN,
                "The dataset's task list could not be read (install pyarrow for meta/tasks.parquet).",
                {},
            )
        )
    elif tasks and all(re.fullmatch(r"[A-Za-z0-9_]+", task) for task in tasks):
        findings.append(
            _finding(
                "TASK_TEXT_FORMAT",
                DiagnosticStatus.PASS,
                f"All {len(tasks)} training tasks are bare identifiers (e.g. {tasks[0]!r}), not sentences. "
                "An evaluation that sends natural-language instructions is out of distribution for this "
                f"policy. See {ISSUES}4548.",
                {"examples": tasks[:5], "task_count": len(tasks), "advisory": True},
            )
        )
    return findings
