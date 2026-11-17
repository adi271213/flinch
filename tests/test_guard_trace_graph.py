"""A serialized runtime envelope must retain its real trace and lineage semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch

from lerobot_flinch.graph import build_processor_graph, processor_graph_to_mermaid, write_processor_graph
from lerobot_flinch.guard import InputEnvelopeGuardStep, add_input_guard
from lerobot_flinch.models import DiagnosticStatus
from lerobot_flinch.trace import run_offline_trace
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, Path]:
    return make_bug_fixtures(tmp_path / "fixtures")


def _preprocessor(checkpoint: Path) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="preprocessor_config.json",
        local_files_only=True,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )


def _save_guard(checkpoint: Path, **kwargs: object) -> None:
    add_input_guard(_preprocessor(checkpoint), **kwargs).save_pretrained(
        checkpoint, config_filename="preprocessor_config.json"
    )


def _digests(checkpoint: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.iterdir()
        if path.is_file()
    }


def test_healthy_offline_trace_executes_real_guard_and_keeps_files_unchanged(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _save_guard(checkpoint)
    before = _digests(checkpoint)
    native_observation = InputEnvelopeGuardStep.observation
    calls = []

    def observed(self: InputEnvelopeGuardStep, observation: dict) -> dict:
        calls.append(observation["observation.state"].clone())
        return native_observation(self, observation)

    monkeypatch.setattr(InputEnvelopeGuardStep, "observation", observed)
    findings = run_offline_trace(checkpoint, checkpoint / "sample_observation.npz")
    assert len(calls) == 1
    assert not any(f.status is DiagnosticStatus.FAIL for f in findings)
    (guard,) = [f for f in findings if f.check_id == "INPUT_ENVELOPE_GUARD"]
    assert guard.status is DiagnosticStatus.PASS
    assert guard.details["step_index"] == 1
    assert guard.details["checked_tensor_keys"] == ["observation.state"]
    assert guard.details["clamped_tensor_keys"] == []
    assert _digests(checkpoint) == before


def test_guard_rejection_has_exact_evidence_and_preserves_normalizer_explosion(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    _save_guard(checkpoint)
    before = _digests(checkpoint)
    findings = run_offline_trace(checkpoint, checkpoint / "sample_observation.npz")
    (guard,) = [f for f in findings if f.check_id == "INPUT_ENVELOPE_VIOLATION"]
    assert guard.status is DiagnosticStatus.FAIL
    assert guard.details["step_index"] == 1
    assert guard.details["step_class"] == "InputEnvelopeGuardStep"
    assert guard.details["tensor_key"] == "observation.state"
    assert guard.provenance.violating_indices == [9]
    assert guard.details["violating_index_count"] == 1
    assert guard.details["tensor_slice"][0] > 1e5
    assert guard.details["processor_output_available"] is False
    assert len(guard.provenance.sha256_prefix) == 12
    assert any(f.check_id == "EMPIRICAL_GAIN_EXPLOSION" and f.details["step_index"] == 0 for f in findings)
    assert not any(f.check_id == "TRACE_EXECUTION" for f in findings)
    assert _digests(checkpoint) == before


def test_guard_reports_rejected_key_rather_than_first_transition_tensor(
    checkpoints: dict[str, Path],
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _save_guard(checkpoint, keys=["observation.extra"])
    sample = checkpoint / "sample_observation.npz"
    with np.load(sample, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["observation.extra"] = np.array([[0.0, 20.0, 0.0]], dtype=np.float32)
    np.savez(sample, **values)
    findings = run_offline_trace(checkpoint, sample)
    (guard,) = [f for f in findings if f.check_id == "INPUT_ENVELOPE_VIOLATION"]
    assert guard.details["tensor_key"] == "observation.extra"
    assert guard.provenance.tensor_key == "observation.extra"
    assert guard.provenance.violating_indices == [1]
    assert guard.details["input_values"] == [20.0]


def test_guard_trace_retains_exact_total_count_with_bounded_evidence(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _save_guard(checkpoint, keys=["observation.extra"])
    sample = checkpoint / "sample_observation.npz"
    with np.load(sample, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["observation.extra"] = np.full((1, 70), 20.0, dtype=np.float32)
    np.savez(sample, **values)
    findings = run_offline_trace(checkpoint, sample)
    (guard,) = [finding for finding in findings if finding.check_id == "INPUT_ENVELOPE_VIOLATION"]
    assert guard.details["violating_index_count"] == 70
    assert guard.provenance.violating_indices == list(range(8))
    assert guard.details["tensor_slice"] == [20.0] * 8
    assert guard.to_dict()["provenance"]["violating_index_count"] == 70


def test_clamp_executes_real_guard_without_erasing_earlier_gain_evidence(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    _save_guard(checkpoint, max_abs=5.0, on_violation="clamp")
    native_observation = InputEnvelopeGuardStep.observation
    outputs = []

    def observed(self: InputEnvelopeGuardStep, observation: dict) -> dict:
        result = native_observation(self, observation)
        outputs.append(result["observation.state"].clone())
        return result

    monkeypatch.setattr(InputEnvelopeGuardStep, "observation", observed)
    findings = run_offline_trace(checkpoint, checkpoint / "sample_observation.npz")
    assert len(outputs) == 1 and bool((outputs[0].abs() <= 5).all())
    (guard,) = [f for f in findings if f.check_id == "INPUT_ENVELOPE_GUARD"]
    assert guard.details["clamped_tensor_keys"] == ["observation.state"]
    assert any(f.check_id == "EMPIRICAL_GAIN_EXPLOSION" for f in findings)
    assert not any(f.check_id == "INPUT_ENVELOPE_VIOLATION" for f in findings)


def test_nonfinite_origin_remains_the_normalizer_before_guard(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["bug_4973_fp16_nan"]
    preprocessor = _preprocessor(checkpoint)
    normalizer_index = next(
        index
        for index, step in enumerate(preprocessor.steps)
        if type(step).__name__ == "NormalizerProcessorStep"
    )
    _save_guard(checkpoint)
    findings = run_offline_trace(checkpoint, checkpoint / "sample_observation.npz")
    (nonfinite,) = [f for f in findings if f.check_id == "RUNTIME_NON_FINITE"]
    assert nonfinite.details["step_index"] == normalizer_index
    assert nonfinite.details["step_class"] == "NormalizerProcessorStep"
    assert not any(f.check_id.startswith("INPUT_ENVELOPE") for f in findings)


@pytest.mark.parametrize("mode", ["raise", "clamp"])
def test_graph_describes_envelope_instead_of_identity_and_excludes_action(
    checkpoints: dict[str, Path], tmp_path: Path, mode: str
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _save_guard(checkpoint, max_abs=7.5, on_violation=mode)
    before = _digests(checkpoint)
    graph = build_processor_graph(checkpoint)
    (guard,) = [
        node for node in graph["nodes"] if node["properties"].get("class_name") == "InputEnvelopeGuardStep"
    ]
    assert guard["properties"]["max_abs"] == 7.5
    assert guard["properties"]["on_violation"] == mode
    assert guard["properties"]["keys"] == ["observation.state"]
    assert guard["properties"]["action_tensor_guarded"] is False
    outgoing = [edge for edge in graph["edges"] if edge["source"] == guard["id"]]
    state = next(edge for edge in outgoing if edge["properties"].get("feature_key") == "observation.state")
    assert state["type"] == ("CLAMPS" if mode == "clamp" else "GUARDS")
    assert state["properties"]["guard_applies"] is True
    assert state["properties"]["guard_dtype_conditional"] is True
    assert "normalization_mode" not in state["properties"]
    action = next(edge for edge in outgoing if edge["properties"].get("feature_key") == "action")
    assert action["type"] == "TRANSFORMS" and action["properties"]["guard_applies"] is False
    assert "normalization_mode" not in action["properties"]
    text = processor_graph_to_mermaid(graph)
    assert "abs &lt;= 7.5" in text and mode in text
    assert "insertion point only" in text and "outside selected observation guard scope" in text
    for suffix in ("json", "mmd"):
        output = write_processor_graph(graph, tmp_path / f"guard.{suffix}")
        assert output.is_file()
        if suffix == "json":
            assert json.loads(output.read_text())["edges"] == graph["edges"]
    assert _digests(checkpoint) == before


def test_graph_never_calls_guard_runtime_observation(
    checkpoints: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _save_guard(checkpoint)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("A static graph executed the guard's runtime check")

    monkeypatch.setattr(InputEnvelopeGuardStep, "observation", forbidden)
    assert build_processor_graph(checkpoint)["summary"]["coverage_status"] == "PASS"


def test_graph_invalid_guard_config_remains_unknown(checkpoints: dict[str, Path]) -> None:
    checkpoint = checkpoints["clean_act_modern"]
    _save_guard(checkpoint)
    path = checkpoint / "preprocessor_config.json"
    saved = json.loads(path.read_text())
    saved["steps"][1]["config"]["max_abs"] = True
    path.write_text(json.dumps(saved))
    graph = build_processor_graph(checkpoint)
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert any("positive, finite" in error["reason"] for error in graph["summary"]["errors"])
