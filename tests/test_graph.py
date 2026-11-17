"""Static lineage must retain exact contracts, including Issue #4415 bypasses."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from safetensors.torch import load_file, save_file

from lerobot_preflight.graph import build_processor_graph, processor_graph_to_mermaid, write_processor_graph
from tests.fixtures.make_bug_fixtures import make_bug_fixtures


@pytest.fixture
def checkpoints(tmp_path, monkeypatch):
    checkpoints = make_bug_fixtures(tmp_path / "fixtures")

    def no_network(*args, **kwargs):
        pytest.fail("A lineage export attempted a network request")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    return checkpoints


def _edges(graph, kind):
    return [edge for edge in graph["edges"] if edge["type"] == kind]


def _nodes(graph, kind):
    return [node for node in graph["nodes"] if node["type"] == kind]


@pytest.mark.parametrize(
    "entry",
    [
        {"registry_name": ["normalizer_processor"], "config": {}},
        {"registry_name": {"name": "normalizer_processor"}, "config": {}},
        {"class": ["lerobot.processor.normalize_processor.NormalizerProcessorStep"], "config": {}},
        {"registry_name": "normalizer_processor", "config": []},
        {"registry_name": "normalizer_processor", "state_file": ["buffers.safetensors"]},
    ],
)
def test_graph_retains_malformed_steps_as_unknown_before_native_construction(
    checkpoints,
    monkeypatch,
    entry,
):
    checkpoint = checkpoints["bug_4415_silent_bypass"]
    for filename, steps in (("preprocessor_config.json", [entry]), ("postprocessor_config.json", [])):
        path = checkpoint / filename
        saved = json.loads(path.read_text())
        saved["steps"] = steps
        path.write_text(json.dumps(saved))

    def forbidden_constructor(*args, **kwargs):
        pytest.fail("Malformed processor declarations must not reach native constructors")

    monkeypatch.setattr(NormalizerProcessorStep, "__init__", forbidden_constructor)
    graph = build_processor_graph(checkpoint)
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert graph["summary"]["unknown_step_boundaries"] == [
        {
            "file_path": "preprocessor_config.json",
            "step_index": 0,
            "class_name": "invalid_step_declaration",
        }
    ]
    assert any(
        error["status"] == "UNKNOWN" and "Invalid step 0" in error["reason"]
        for error in graph["summary"]["errors"]
    )


def test_graph_4415_records_missing_namespace_and_exact_prefixed_keys(checkpoints):
    checkpoint = checkpoints["bug_4415_silent_bypass"]
    graph = build_processor_graph(checkpoint)
    (bypass,) = _edges(graph, "UNBOUND_BYPASS")
    assert bypass["properties"]["missing_keys"] == ["observation.state.mean", "observation.state.std"]
    assert bypass["properties"]["step_index"] == 0
    assert bypass["properties"]["normalization_mode"] == "MEAN_STD"
    assert "wrong_dataset/observation.state.std" in bypass["properties"]["available_buffer_keys"]
    buffer = next(node for node in _nodes(graph, "stat_buffer") if "preprocessor" in node["label"])
    assert buffer["properties"]["matched_keys"] == ["action.mean", "action.std"]
    assert buffer["properties"]["unbound_keys"] == [
        "wrong_dataset/observation.state.mean",
        "wrong_dataset/observation.state.std",
    ]
    assert len(buffer["properties"]["sha256"]) == 64
    text = processor_graph_to_mermaid(graph)
    assert "UNBOUND_BYPASS" in text and "wrong_dataset/observation.state.std" in text
    assert "stroke:#c62828" in text


def test_graph_gain_uses_active_mode_epsilon_and_explicit_dtype(checkpoints):
    checkpoint = checkpoints["bug_5289_pathological_gain"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(
        0, {"registry_name": "device_processor", "config": {"device": "cuda", "float_dtype": "float32"}}
    )
    path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    (transform,) = [
        edge
        for edge in _edges(graph, "TRANSFORMS")
        if edge["properties"].get("feature_key") == "observation.state"
        and edge["properties"].get("normalization_mode") == "MEAN_STD"
    ]
    props = transform["properties"]
    assert props["max_forward_gain"] == pytest.approx(float(1 / (torch.tensor(1.48e-6) + 1e-8)))
    assert props["max_inverse_gain"] == 0.5
    assert props["effective_dtype"] == "float32"
    assert props["gain_dtype_conditional"] is False
    assert props["step_index"] == 1


def test_undeclared_runtime_dtype_is_not_invented(checkpoints):
    graph = build_processor_graph(checkpoints["clean_act_modern"])
    assert all(node["properties"]["active_dtype"] is None for node in _nodes(graph, "processor_step"))
    transform = next(
        edge["properties"]
        for edge in _edges(graph, "TRANSFORMS")
        if edge["properties"].get("direction") == "forward"
    )
    assert transform["effective_dtype"] is None
    assert transform["gain_dtype_conditional"] is True
    assert transform["gain_evaluation_dtype"] == "float32"


def test_partial_stats_are_contract_failure_not_silent_bypass(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    state_path = checkpoint / config["steps"][0]["state_file"]
    tensors = load_file(state_path)
    del tensors["observation.state.std"]
    save_file(tensors, state_path)
    graph = build_processor_graph(checkpoint)
    assert not _edges(graph, "UNBOUND_BYPASS")
    (mismatch,) = _edges(graph, "CONTRACT_MISMATCH")
    assert mismatch["properties"]["runtime_behavior"] == "native raises for incomplete stats"


def test_inline_stats_override_and_orphan_files_remain_distinct(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["stats"] = {
        "observation.state": {"mean": [0.0] * 14, "std": [0.25] * 14},
        "action": {"mean": [0.0] * 14, "std": [0.25] * 14},
    }
    path.write_text(json.dumps(config))
    save_file(
        {"dataset.index.std": torch.tensor([0.0])},
        checkpoint / "policy_preprocessor_step_99_unused.safetensors",
    )
    graph = build_processor_graph(checkpoint)
    (inline,) = [
        node for node in _nodes(graph, "stat_buffer") if node["properties"].get("source_kind") == "inline"
    ]
    assert len(inline["properties"]["inline_stats_sha256"]) == 64
    assert inline["properties"]["matched_keys"] == [
        "action.mean",
        "action.std",
        "observation.state.mean",
        "observation.state.std",
    ]
    (orphan,) = [node for node in _nodes(graph, "stat_buffer") if node["properties"].get("orphan")]
    assert orphan["properties"]["unbound_keys"] == ["dataset.index.std"]
    assert not [edge for edge in graph["edges"] if edge["target"] == orphan["id"]]
    transform = next(
        edge["properties"]
        for edge in _edges(graph, "TRANSFORMS")
        if edge["properties"].get("direction") == "forward"
    )
    assert transform["max_forward_gain"] == pytest.approx(4.0)


def test_graph_preserves_camera_and_shape_contract_drift(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [7], "dtype": "float32"},
                    "observation.images.front": {"type": "VISUAL", "shape": [3, 16, 16]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [14]}},
            }
        )
    )
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["features"]["observation.image"] = {"type": "VISUAL", "shape": [3, 16, 16]}
    config["steps"][0]["config"]["norm_map"]["VISUAL"] = "IDENTITY"
    path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    mismatches = [edge["properties"] for edge in _edges(graph, "CONTRACT_MISMATCH")]
    assert any(
        item["feature_key"] == "observation.state"
        and item["actual_shape"] == [7]
        and item["expected_shape"] == [14]
        for item in mismatches
    )
    camera = next(item for item in mismatches if item["feature_key"] == "observation.image")
    assert "observation.images.front" in camera["available_feature_keys"]
    assert camera["rename_map"] == {}


@pytest.mark.parametrize("deltas,status,expected", [([-1, 0], "FAIL", 1), ([0, 1], "PASS", 0)])
def test_graph_temporal_anchor_uses_declared_delta_chronology(checkpoints, deltas, status, expected):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["observation_delta_indices"] = deltas
    path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    (anchor,) = _edges(graph, "TEMPORAL_ANCHOR")
    assert anchor["properties"]["native_collapse_index"] == 0
    assert anchor["properties"]["expected_anchor_index"] == expected
    assert anchor["properties"]["status"] == status


def test_native_rename_is_visible_between_feature_versions(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    features = {"observation.raw": PolicyFeature(FeatureType.STATE, (14,))}
    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"observation.raw": {"mean": torch.zeros(14), "std": torch.ones(14)}},
    )
    PolicyProcessorPipeline(
        name="policy_preprocessor",
        steps=[
            RenameObservationsProcessorStep(rename_map={"observation.state": "observation.raw"}),
            normalizer,
        ],
    ).save_pretrained(checkpoint, config_filename="preprocessor_config.json")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"type": "STATE", "shape": [14]}},
                "output_features": {"action": {"type": "ACTION", "shape": [14]}},
            }
        )
    )
    graph = build_processor_graph(checkpoint)
    rename = next(
        edge["properties"]
        for edge in _edges(graph, "TRANSFORMS")
        if edge["properties"].get("rename_map")
        and edge["properties"].get("feature_key") == "observation.state"
    )
    assert rename["output_feature_key"] == "observation.raw"
    assert not _edges(graph, "CONTRACT_MISMATCH")


@pytest.mark.parametrize("suffix", [".json", ".mmd", ".md"])
def test_exports_are_deterministic_and_can_replace_only_own_output(checkpoints, tmp_path, suffix):
    checkpoint = checkpoints["bug_4415_silent_bypass"]
    original = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    graph = build_processor_graph(checkpoint)
    assert build_processor_graph(checkpoint) == graph
    target = tmp_path / ("graph" + suffix)
    assert write_processor_graph(graph, target) == target
    before = target.read_bytes()
    write_processor_graph(graph, target)
    assert target.read_bytes() == before
    if suffix == ".json":
        assert json.loads(before) == graph
    elif suffix == ".mmd":
        assert before.startswith(b"flowchart LR\n")
    else:
        assert b"```mermaid\nflowchart LR\n" in before
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == original


@pytest.mark.parametrize("suffix", [".json", ".mmd", ".md"])
def test_graph_export_uses_utf8_and_platform_independent_newlines(checkpoints, tmp_path, monkeypatch, suffix):
    graph = build_processor_graph(checkpoints["bug_4415_silent_bypass"])
    graph["nodes"][0]["label"] = "joint → état ≈ 状態"
    target = tmp_path / ("graph" + suffix)
    original = Path.open
    writes = []

    def emulate_windows_open(path, mode="r", *args, **kwargs):
        if path == target and mode in {"x", "w"}:
            writes.append((kwargs.get("encoding"), kwargs.get("newline")))
            # A Windows text stream translates LF unless explicitly disabled.
            kwargs.setdefault("encoding", "cp1252")
            kwargs.setdefault("newline", "\r\n")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", emulate_windows_open)
    write_processor_graph(graph, target)
    write_processor_graph(graph, target)
    assert writes == [("utf-8", "\n"), ("utf-8", "\n")]
    content = target.read_bytes()
    assert b"\r\n" not in content
    assert "joint → état ≈ 状態" in content.decode("utf-8")


def test_graph_output_inside_checkpoint_does_not_become_an_input_artifact(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    graph = build_processor_graph(checkpoint)
    target = checkpoint / "lineage.json"
    write_processor_graph(graph, target)
    assert build_processor_graph(checkpoint) == graph
    write_processor_graph(build_processor_graph(checkpoint), target)


def test_exports_reject_artifacts_unrelated_files_and_symlinks(checkpoints, tmp_path):
    checkpoint = checkpoints["clean_act_modern"]
    graph = build_processor_graph(checkpoint)
    for path in (checkpoint / "preprocessor_config.json", tmp_path / "notes.md"):
        if not path.exists():
            path.write_text("unrelated user data")
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            write_processor_graph(graph, path)
        assert path.read_bytes() == before
    link = tmp_path / "linked.json"
    link.symlink_to(checkpoint / "preprocessor_config.json")
    with pytest.raises(FileExistsError):
        write_processor_graph(graph, link)


def test_mermaid_escapes_edge_delimiters_and_html(checkpoints):
    graph = build_processor_graph(checkpoints["clean_act_modern"])
    graph["nodes"][0]["label"] = '<script>" | bracket[]'
    graph["edges"][0]["properties"]["feature_key"] = '<b>"|'
    text = processor_graph_to_mermaid(graph)
    assert "<script>" not in text and "<b>" not in text
    assert "&lt;script&gt;&quot; &#124;" in text


def test_legacy_without_verified_tiny_slices_is_explicitly_unknown(checkpoints):
    graph = build_processor_graph(checkpoints["legacy_act_embedded"])
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert "verified legacy" in graph["summary"]["errors"][0]["reason"]


def test_missing_state_file_is_unknown_loading_failure_not_silent_bypass(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    (checkpoint / config["steps"][0]["state_file"]).unlink()
    graph = build_processor_graph(checkpoint)
    assert not _edges(graph, "UNBOUND_BYPASS")
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert any(
        edge["properties"].get("runtime_behavior", "").startswith("state file loading failed")
        for edge in _edges(graph, "CONTRACT_MISMATCH")
    )


def test_uninspected_step_breaks_shape_and_dtype_continuity(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(0, {"registry_name": "uninspected_numeric_transform", "config": {}})
    path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert graph["summary"]["unknown_step_boundaries"][0]["step_index"] == 0
    transform = next(
        edge["properties"]
        for edge in _edges(graph, "TRANSFORMS")
        if edge["properties"].get("normalization_mode") == "MEAN_STD"
        and edge["properties"].get("feature_key") == "observation.state"
    )
    assert transform["upstream_flow_verified"] is False
    assert transform["effective_dtype"] is None
    assert transform["gain_dtype_conditional"] is True
    assert all(edge["properties"]["status"] == "UNKNOWN" for edge in _edges(graph, "CONTRACT_MISMATCH"))


def test_invalid_gain_buffers_are_unknown_without_aborting_graph(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["norm_map"]["STATE"] = "MIN_MAX"
    state_path = checkpoint / config["steps"][0]["state_file"]
    state = load_file(state_path)
    state["observation.state.min"] = torch.zeros(3)
    state["observation.state.max"] = torch.ones(5)
    save_file(state, state_path)
    path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert any(edge["properties"].get("gain_status") == "UNKNOWN" for edge in _edges(graph, "TRANSFORMS"))


def test_device_cast_preserves_explicit_integer_feature_dtype(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [14], "dtype": "float16"},
                    "observation.images.front": {"type": "VISUAL", "shape": [3, 16, 16], "dtype": "uint8"},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [14]}},
            }
        )
    )
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"].insert(
        0, {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": "float16"}}
    )
    path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    transforms = [edge["properties"] for edge in _edges(graph, "TRANSFORMS")]
    assert all(
        edge["effective_dtype"] == "uint8"
        for edge in transforms
        if edge["feature_key"] == "observation.images.front"
    )
    state = next(
        edge
        for edge in transforms
        if edge["feature_key"] == "observation.state" and edge["normalization_mode"] == "MEAN_STD"
    )
    assert state["gain_evaluation_dtype"] == "float16"


def test_noncanonical_action_feature_has_no_phantom_normalization(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    path = checkpoint / "preprocessor_config.json"
    config = json.loads(path.read_text())
    config["steps"][0]["config"]["features"]["action.alias"] = {"type": "ACTION", "shape": [14]}
    path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    alias = next(
        edge["properties"]
        for edge in _edges(graph, "TRANSFORMS")
        if edge["properties"].get("feature_key") == "action.alias"
    )
    assert alias["normalization_mode"] == "IDENTITY"


def test_unreferenced_neural_weight_shards_are_never_read(checkpoints, monkeypatch):
    from lerobot_preflight import graph as module

    checkpoint = checkpoints["clean_act_modern"]
    shard = checkpoint / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"not a processor; must not read")
    original = module._read_tensors

    def read(path):
        assert path != shard
        return original(path)

    monkeypatch.setattr(module, "_read_tensors", read)
    graph = build_processor_graph(checkpoint)
    assert shard.name not in graph["summary"]["artifact_files"]


def test_nested_identically_named_buffers_do_not_collapse_into_one_node(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    for folder, config_name in (("pre", "preprocessor_config.json"), ("post", "postprocessor_config.json")):
        config_path = checkpoint / config_name
        config = json.loads(config_path.read_text())
        original = checkpoint / config["steps"][0]["state_file"]
        target = checkpoint / folder / "stats.safetensors"
        target.parent.mkdir()
        original.rename(target)
        config["steps"][0]["state_file"] = f"{folder}/stats.safetensors"
        config_path.write_text(json.dumps(config))
    graph = build_processor_graph(checkpoint)
    buffers = _nodes(graph, "stat_buffer")
    assert {node["properties"]["file_path"] for node in buffers} == {
        "pre/stats.safetensors",
        "post/stats.safetensors",
    }
    assert len({node["id"] for node in buffers}) == 2
    assert len({node["properties"]["sha256"] for node in buffers}) == 2


def test_corrupt_tensor_file_is_unknown_instead_of_aborting_graph(checkpoints):
    checkpoint = checkpoints["clean_act_modern"]
    config = json.loads((checkpoint / "preprocessor_config.json").read_text())
    (checkpoint / config["steps"][0]["state_file"]).write_bytes(b"not safetensors")
    graph = build_processor_graph(checkpoint)
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert not _edges(graph, "UNBOUND_BYPASS")


def test_temporal_anchor_edge_uses_entering_state_and_no_false_output_cycle(checkpoints):
    checkpoint = checkpoints["bug_5315_temporal_anchor"]
    graph = build_processor_graph(checkpoint, metadata_path=checkpoint / "metadata.json")
    (anchor,) = _edges(graph, "TEMPORAL_ANCHOR")
    assert anchor["source"] != anchor["target"]
    assert not any(
        edge["source"] == anchor["target"] and edge["target"] == anchor["source"]
        for edge in _edges(graph, "TRANSFORMS")
    )
    assert graph["summary"]["has_contract_failures"] is True


def test_legacy_visual_paths_remain_unknown_and_header_digest_scope_is_visible(tmp_path, monkeypatch):
    from lerobot_preflight import graph as module

    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [2]},
                    "observation.images.front": {"type": "VISUAL", "shape": [3, 16, 16]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [2]}},
                "normalization_mapping": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD", "VISUAL": "MEAN_STD"},
            }
        )
    )
    state = {
        f"{group}.buffer_{key}.{stat}": torch.full((2,), value)
        for group, key in (("normalize_inputs", "observation_state"), ("unnormalize_outputs", "action"))
        for stat, value in (("mean", 0.0), ("std", 0.5))
    }
    path = checkpoint / "legacy_normalization.safetensors"
    save_file(state, path)
    (checkpoint / "model.safetensors.header.json").write_text("{}")
    monkeypatch.setattr(
        module,
        "_verified_legacy_values",
        lambda root: (path, state, {}, {"header_sha256_prefix": "0123456789ab"}),
    )
    graph = build_processor_graph(checkpoint)
    assert graph["summary"]["coverage_status"] == "UNKNOWN"
    assert any("input:observation.images.front" in error["reason"] for error in graph["summary"]["errors"])
    assert "header sha256 0123456789ab" in processor_graph_to_mermaid(graph)
