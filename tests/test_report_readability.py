"""Keep the maintainer narrative ahead of the complete, expandable evidence."""

from __future__ import annotations

import json
import re
from copy import deepcopy

import pytest

from lerobot_flinch import hub_scanner


def _finding(
    check,
    *,
    key="observation.state.std",
    digest="a" * 12,
    indices=None,
    details=None,
    status="FAIL",
    severity="CRITICAL",
    category=None,
):
    finding = {
        "check_id": check,
        "status": status,
        "severity": severity,
        "summary": "Verified saved artifact condition.",
        "details": {"feature_key": key.rsplit(".", 1)[0], **(details or {})},
        "provenance": {
            "file_path": "stats.safetensors",
            "step_name": None,
            "tensor_key": key,
            "sha256_prefix": digest,
            "violating_indices": indices or [],
        },
    }
    if category:
        finding["report_category"] = category
    return finding


def _repo(repo_id, findings, *, classification="LEGACY_POLICY", payloads=None):
    return {
        "repo_id": repo_id,
        "revision": "1" * 40,
        "classification": classification,
        "checkpoint_dir": "local-artifacts/" + repo_id.replace("/", "_"),
        "errors": [],
        "findings": findings,
        "stat_tensor_payloads": payloads or [],
    }


@pytest.fixture
def readability_report():
    state_key = "normalize_inputs.buffer_observation_state.std"
    xarm = _finding(
        "PATHOLOGICAL_STD_GAIN",
        key=state_key,
        digest="7d24b59de47a",
        indices=[4],
        details={
            "feature_key": "observation.state",
            "normalization_mode": "MEAN_STD",
            "source_tensor_key": state_key,
            "saved_values": [6.6630414039536845e-6],
            "effective_gains_with_eps": [149856.70780574824],
            "eps": 1e-8,
            "probe_output_values": [29971.341796875],
            "step_direction": "forward",
        },
    )
    bridge = _finding(
        "PATHOLOGICAL_STD_GAIN",
        key=state_key,
        digest="128ca7bfcb39",
        indices=[6],
        details={
            "feature_key": "observation.state",
            "normalization_mode": "MEAN_STD",
            "saved_values": [0.0],
            "effective_gains_with_eps": [1e8],
            "eps": 1e-8,
            "probe_output_values": [20_000_000.0],
            "step_direction": "forward",
        },
    )
    lower = "normalize_inputs.buffer_observation_environment_state.min"
    upper = "normalize_inputs.buffer_observation_environment_state.max"
    ur5 = _finding(
        "PATHOLOGICAL_STD_GAIN",
        key=lower,
        digest="fb006f50ebd8",
        indices=list(range(11, 18)),
        details={
            "feature_key": "observation.environment_state",
            "normalization_mode": "MIN_MAX",
            "saved_values": [0.0] * 7,
            "upper_values": [0.0] * 7,
            "upper_buffer_key": upper,
            "saved_span_values": [0.0] * 7,
            "effective_gains_with_eps": [2e8] * 7,
            "eps": 1e-8,
            "probe_output_values": [40_000_000.0] * 7,
            "step_direction": "forward",
        },
    )
    pointer = _finding(
        "GIT_LFS_POINTER_ARTIFACT",
        digest="f" * 12,
        details={
            "pointer_size_bytes": 129,
            "lfs_oid_sha256": "490ab239d96e" + "0" * 52,
            "declared_object_size": 640,
            "step_index": 5,
            "tensor_values_available": False,
        },
    )
    namespaced = [
        f"so100-{color}.buffer.action.{stat}" for color in ("blue", "red") for stat in ("mean", "std")
    ]
    base = _finding(
        "UNBOUND_BASE_MODEL_STATS",
        key="action.mean",
        digest="490ab239d96e",
        details={
            "step_index": 5,
            "required_buffer_keys": ["action.mean", "action.std"],
            "declared_feature_keys": ["observation.state", "action"],
            "available_buffer_keys": namespaced,
            "matching_prefixed_keys": namespaced,
            "missing_observation_state_stats": ["observation.state.mean", "observation.state.std"],
            "target_dataset_stats_bound": False,
        },
        category="base_model_prerequisite",
    )
    pi0 = _finding(
        "FEATURE_CONTRACT",
        key="action.mean",
        details={
            "step_index": 0,
            "available_buffer_keys": [],
            "normalization_mode": "MEAN_STD",
        },
        category="base_model_prerequisite",
    )
    camera = _finding(
        "CAMERA_KEY_CONTRACT_MISMATCH",
        status="UNKNOWN",
        severity="WARNING",
        details={
            "step_index": 5,
            "processor_visual_keys": ["observation.image", "observation.image2", "observation.image3"],
            "policy_visual_keys": ["observation.images.front", "observation.images.wrist"],
            "rename_maps": [{}],
            "normalizer_visual_mode": "IDENTITY",
            "identity_camera_keys_pass_through": True,
            "preprocessor_camera_keyerror_expected": False,
        },
    )
    # A large evidence appendix must not push the main story below the fold.
    repeated = [
        {
            "file_path": "stats.safetensors",
            "tensor_key": f"dataset.metadata_{index}.std",
            "payload_sha256": f"{index:064x}",
            "dtype": "F32",
            "shape": [1],
            "digest_scope": "individual_tensor_payload",
        }
        for index in range(100)
    ]
    repos = [
        _repo("jhseon-holiday/vqbet_xarm_holi_test", [xarm], payloads=repeated),
        _repo("HaomingSong/lerobot-pi0-bridge", [bridge]),
        _repo("iantc104/Diffusion-RPL-UR5-PegInsert-v0", [ur5]),
        _repo("y1y2y3/smolvla_base1", [pointer]),
        _repo("lerobot/smolvla_base", [base], classification="MODERN_POLICY"),
        _repo("lerobot/pi0_old", [pi0], classification="MODERN_POLICY"),
        _repo("taekyukim/smolvla_base", [base, camera], classification="MODERN_POLICY"),
        _repo("org/shared-payload-copy", [], payloads=repeated),
    ]
    repos += [_repo(f"org/clean-{index}", [], classification="MODERN_POLICY") for index in range(92)]
    for repo in repos:
        repo["findings"].append(
            {
                "check_id": "TEMPORAL_ANCHOR_MISMATCH",
                "status": "PASS",
                "severity": "INFO",
                "summary": "No enabled relative-action conversion requires a temporal anchor.",
                "details": {},
            }
        )
    return hub_scanner._report(repos, {"mode": "cached"})


def test_five_finding_families_and_both_lineage_diagrams_fit_first_sixty_lines(readability_report, tmp_path):
    _, path = hub_scanner.write_hub_reports(readability_report, tmp_path)
    first_lines = path.read_text().splitlines()[:60]
    first_sixty = "\n".join(first_lines)
    for repo_id in (
        "jhseon-holiday/vqbet_xarm_holi_test",
        "HaomingSong/lerobot-pi0-bridge",
        "iantc104/Diffusion-RPL-UR5-PegInsert-v0",
        "y1y2y3/smolvla_base1",
        "lerobot/smolvla_base",
        "lerobot/pi0_old",
        "taekyukim/smolvla_base",
    ):
        assert repo_id in first_sixty
    assert "149,857" in first_sixty
    assert "129" in first_sixty
    assert first_sixty.count("```mermaid") >= 2
    assert first_sixty.count("flowchart LR") >= 2
    assert "UNBOUND" in first_sixty
    assert "observation.images.front" in first_sixty
    assert "observation.image" in first_sixty
    assert "xarm_processor_lineage.mmd" in first_sixty
    depth = 0
    for line in first_lines:
        depth += line.count("<details>")
        if "```mermaid" in line:
            assert depth == 0, "Headline lineage diagrams must be visible without expansion"
        depth -= line.count("</details>")


def test_dense_payload_and_repository_appendix_tables_are_collapsed_without_losing_evidence(
    readability_report, tmp_path
):
    json_path, path = hub_scanner.write_hub_reports(readability_report, tmp_path)
    text = path.read_text()
    depth = 0
    payload_rows = coverage_rows = 0
    for line in text.splitlines():
        depth += line.count("<details>")
        if re.match(r"\| `[0-9a-f]{64}` \|", line):
            payload_rows += 1
            assert depth > 0, "Raw tensor SHA rows must be collapsed"
        if line.startswith("| Repository | Revision | Generation") or line.startswith(
            "| Repository / check |"
        ):
            coverage_rows += 1
            assert depth > 0, "Dense per-repository evidence must be collapsed"
        depth -= line.count("</details>")
        assert depth >= 0
    assert depth == 0
    assert payload_rows == 100 and coverage_rows >= 2
    assert text.count("<details>") == text.count("</details>") >= 3
    assert re.search(r"</summary>\n\n", text)
    assert all(block.startswith("\n\n") for block in text.split("</summary>")[1:])
    assert "7d24b59de47a" in text and "normalize_inputs.buffer_observation_state.std" in text
    assert "fb006f50ebd8" in text and "11, 12, 13, 14, 15, 16, 17" in text
    assert "1" * 40 in text  # Exact pinned Hub tree link survives the layout change.
    assert json.loads(json_path.read_text()) == hub_scanner._portable_report(readability_report)


def test_temporal_report_text_is_a_cross_policy_contract_check(readability_report, tmp_path):
    _, path = hub_scanner.write_hub_reports(readability_report, tmp_path)
    text = path.read_text()
    assert "cross-policy chronology contract check" in text
    assert "custom or migrated" in text
    assert "VLA-JEPA" in text and "LingBot-VA" in text
    assert "state[:, 0]" in text and "[-1, 0]" in text
    assert "intentional" in text and "valid" in text
    assert "not a reproduced upstream bug" in text


def test_smaller_unrelated_report_never_invents_wild_checkpoint_findings(tmp_path):
    report = hub_scanner._report([_repo("org/clean", [])], {"mode": "cached"})
    _, path = hub_scanner.write_hub_reports(report, tmp_path)
    text = path.read_text()
    assert "jhseon-holiday/vqbet_xarm_holi_test" not in text
    assert "HaomingSong/lerobot-pi0-bridge" not in text
    assert "iantc104/Diffusion-RPL-UR5-PegInsert-v0" not in text
    assert "y1y2y3/smolvla_base1" not in text
    assert "149,857" not in text


def test_known_repository_name_without_matching_finding_cannot_create_a_headline(tmp_path):
    report = hub_scanner._report(
        [
            _repo(repo_id, [])
            for repo_id in (
                "jhseon-holiday/vqbet_xarm_holi_test",
                "HaomingSong/lerobot-pi0-bridge",
                "iantc104/Diffusion-RPL-UR5-PegInsert-v0",
                "y1y2y3/smolvla_base1",
            )
        ],
        {"mode": "cached"},
    )
    _, path = hub_scanner.write_hub_reports(report, tmp_path)
    first_sixty = "\n".join(path.read_text().splitlines()[:60])
    assert "149,857" not in first_sixty
    assert "129-byte" not in first_sixty
    assert "20,000,000" not in first_sixty
    assert "40,000,000" not in first_sixty


def test_taek_overview_uses_preprocessor_index_when_postprocessor_is_listed_first(
    readability_report, tmp_path
):
    report = deepcopy(readability_report)
    repo = next(item for item in report["repos"] if item["repo_id"] == "taekyukim/smolvla_base")
    pre = next(item for item in repo["findings"] if item["check_id"] == "UNBOUND_BASE_MODEL_STATS")
    pre["provenance"]["file_path"] = "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post = deepcopy(pre)
    post["details"].update(step_index=0, declared_feature_keys=["action"], missing_observation_state_stats=[])
    post["provenance"]["file_path"] = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    repo["findings"].insert(0, post)
    _, path = hub_scanner.write_hub_reports(report, tmp_path)
    overview = "\n".join(path.read_text().splitlines()[:60])
    assert "taekyukim/smolvla_base preprocessor[5]" in overview
    assert "preprocessor[0]" not in overview
