"""Benchmark contract tests for the LiDAR-derived SocialForce adapter."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    planner_contract_for_algorithm,
    resolve_observation_mode,
)
from robot_sf.benchmark.algorithm_readiness import require_algorithm_allowed
from robot_sf.benchmark.map_runner import _build_policy


def test_lidar_social_force_metadata_accepts_lidar_level_without_privileged_inputs() -> None:
    """The explicit adapter should be compatible with the lidar_2d contract."""
    assert (
        resolve_observation_mode("lidar_social_force", observation_level="lidar_2d")
        == "sensor_fusion_state"
    )
    contract = planner_contract_for_algorithm(
        "lidar_social_force",
        observation_level="lidar_2d",
        robot_kinematics="differential_drive",
    ).to_metadata()

    observation = contract["observation_contract"]
    action = contract["action_contract"]
    assert observation["observation_level"] == "lidar_2d"
    assert observation["active_mode"] == "sensor_fusion_state"
    assert observation["required_inputs"] == ["robot_state", "goal", "lidar_rays"]
    assert action["command_space"] == "unicycle_vw"


def test_existing_social_force_still_rejects_lidar_observation_level() -> None:
    """The original SocialForce planner should keep failing closed for LiDAR-only runs."""
    with pytest.raises(ValueError, match="Observation level 'lidar_2d' is not supported"):
        resolve_observation_mode("social_force", observation_level="lidar_2d")


def test_lidar_social_force_remains_explicit_opt_in() -> None:
    """Testing-only adapter should not become baseline-safe by registration."""
    with pytest.raises(ValueError, match="blocked by profile 'baseline-safe'"):
        require_algorithm_allowed(
            algo="lidar_social_force",
            benchmark_profile="baseline-safe",
            ppo_paper_ready=False,
        )

    readiness = require_algorithm_allowed(
        algo="lidar_tracked_social_force",
        benchmark_profile="experimental",
        ppo_paper_ready=False,
        allow_testing_algorithms=True,
    )
    assert readiness.canonical_name == "lidar_social_force"


def test_lidar_social_force_map_runner_policy_exposes_adapter_runtime() -> None:
    """Map-runner construction should expose adapter mode and LiDAR tracking diagnostics."""
    policy, meta = _build_policy(
        "lidar_social_force",
        {
            "lidar_tracking": {
                "max_scan_dist": 10.0,
                "normalized_observation": True,
                "max_tracks": 4,
            },
            "social_force": {
                "max_linear_speed": 0.6,
                "max_angular_speed": 0.8,
            },
        },
        robot_kinematics="differential_drive",
    )
    observation = {
        "drive_state": np.array([[0.0, 0.0, 3.0 / 50.0, 0.0, 0.0]], dtype=np.float32),
        "rays": np.ones((1, 16), dtype=np.float32),
    }
    observation["rays"][0, 8] = 0.25
    command = policy(observation)

    assert 0.0 <= command[0] <= 0.6
    assert abs(command[1]) <= 0.8
    assert meta["canonical_algorithm"] == "lidar_social_force"
    assert meta["observation_level"]["key"] == "lidar_2d"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"
    assert meta["planner_kinematics"]["adapter_name"] == "LidarTrackedSocialForceAdapter"

    stats = policy._planner_stats()
    assert stats["status"] == "ok"
    assert stats["derived_payload"] == "tracked_agents"
    assert stats["tracking_assumption"] == "single_frame_lidar_endpoint_clusters"
    assert stats["track_count"] == 1


def test_lidar_social_force_enriched_metadata_marks_testing_only_adapter() -> None:
    """Enriched metadata should distinguish the LiDAR adapter from SocialForce itself."""
    meta = enrich_algorithm_metadata(
        algo="lidar_tracked_social_force",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )

    assert meta["canonical_algorithm"] == "lidar_social_force"
    assert meta["policy_semantics"] == "lidar_endpoint_tracked_social_force_adapter"
    assert meta["observation_spec"]["active_mode"] == "sensor_fusion_state"
    assert meta["observation_level"]["key"] == "lidar_2d"
    assert meta["planner_kinematics"]["testing_only_adapter"] is True
    assert (
        meta["planner_kinematics"]["perception_tracking_mode"]
        == "single_frame_endpoint_clusters_zero_velocity"
    )
