"""Tests for the issue #1615 LiDAR learned-policy launch packet."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.feature_extractors.config import FeatureExtractorPresets
from scripts.validation.check_learned_policy_eligibility import (
    load_candidate_spec,
    validate_learned_policy_eligibility,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKET_PATH = (
    _REPO_ROOT / "configs/training/lidar/lidar_learned_policy_launch_packet_issue_1615.yaml"
)
_ELIGIBILITY_SPECS = (
    _REPO_ROOT / "configs/training/lidar/lidar_ppo_mlp_eligibility_issue_1615.yaml",
    _REPO_ROOT / "configs/training/lidar/lidar_perception_adapter_eligibility_issue_1615.yaml",
)


def _load_packet() -> dict[str, object]:
    payload = yaml.safe_load(_PACKET_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_issue_1615_launch_packet_keeps_lidar_runtime_contract() -> None:
    """Runtime inputs should stay at LiDAR rays plus drive-state metadata."""
    packet = _load_packet()
    contract = packet["observation_contract"]

    assert contract["observation_mode"] == "DEFAULT_GYM"
    assert contract["benchmark_observation_level"] == "lidar_2d"
    assert contract["deployment_observable"] == ["drive_state", "rays"]
    assert set(contract["required_keys"]) == {"drive_state", "rays"}
    assert "occupancy_grid" in contract["forbidden_runtime_fields"]
    assert "socnav_struct_pedestrian_positions" in contract["forbidden_runtime_fields"]


def test_issue_1615_candidates_reference_existing_training_entrypoints() -> None:
    """Candidate baselines should point at checked-in config and extractor entrypoints."""
    packet = _load_packet()
    candidate_ids = [candidate["candidate_id"] for candidate in packet["candidate_baselines"]]

    assert len(candidate_ids) == len(set(candidate_ids))
    for candidate in packet["candidate_baselines"]:
        base_config = candidate.get("base_config")
        if base_config is not None:
            assert (_REPO_ROOT / str(base_config)).exists()
        preset = candidate.get("feature_extractor_preset")
        if preset is not None:
            assert hasattr(FeatureExtractorPresets, str(preset))


def test_issue_1615_eligibility_specs_pass_preflight() -> None:
    """The checked-in learned-policy checklist specs should be complete."""
    for spec_path in _ELIGIBILITY_SPECS:
        assert validate_learned_policy_eligibility(load_candidate_spec(spec_path)) == []


def test_issue_1615_artifact_policy_defers_checkpoint_promotion() -> None:
    """The launch packet should not imply that a checkpoint belongs in git."""
    packet = _load_packet()
    artifact_policy = packet["artifact_policy"]
    follow_up = packet["follow_up_boundaries"]

    assert artifact_policy["checkpoints_in_git"] is False
    assert artifact_policy["durable_checkpoint_source_required"] is True
    assert follow_up["full_training_in_this_issue"] is False
    assert follow_up["benchmark_promotion_in_this_issue"] is False
