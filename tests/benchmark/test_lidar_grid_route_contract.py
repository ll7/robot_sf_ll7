"""Benchmark contract tests for the LiDAR-derived grid-route adapter."""

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


def test_lidar_grid_route_metadata_accepts_lidar_level_without_privileged_inputs() -> None:
    """The explicit adapter should be compatible with the lidar_2d contract."""
    assert (
        resolve_observation_mode("lidar_grid_route", observation_level="lidar_2d")
        == "sensor_fusion_state"
    )
    contract = planner_contract_for_algorithm(
        "lidar_grid_route",
        observation_level="lidar_2d",
        robot_kinematics="differential_drive",
    ).to_metadata()

    observation = contract["observation_contract"]
    action = contract["action_contract"]
    assert observation["observation_level"] == "lidar_2d"
    assert observation["active_mode"] == "sensor_fusion_state"
    assert observation["required_inputs"] == ["robot_state", "goal", "lidar_rays"]
    assert action["command_space"] == "unicycle_vw"


def test_existing_grid_route_still_rejects_lidar_observation_level() -> None:
    """The original occupancy-grid route planner should keep failing closed for LiDAR-only runs."""
    with pytest.raises(ValueError, match="Observation level 'lidar_2d' is not supported"):
        resolve_observation_mode("grid_route", observation_level="lidar_2d")


def test_lidar_grid_route_remains_explicit_opt_in() -> None:
    """Testing-only adapter should not become baseline-safe by registration."""
    with pytest.raises(ValueError, match="blocked by profile 'baseline-safe'"):
        require_algorithm_allowed(
            algo="lidar_grid_route",
            benchmark_profile="baseline-safe",
            ppo_paper_ready=False,
        )

    readiness = require_algorithm_allowed(
        algo="lidar_occupancy_grid_route",
        benchmark_profile="experimental",
        ppo_paper_ready=False,
        allow_testing_algorithms=True,
    )
    assert readiness.canonical_name == "lidar_grid_route"


def test_lidar_grid_route_map_runner_policy_exposes_adapter_runtime() -> None:
    """Map-runner construction should expose adapter mode and LiDAR diagnostics."""
    policy, meta = _build_policy(
        "lidar_grid_route",
        {
            "lidar_occupancy": {
                "width": 6.0,
                "height": 6.0,
                "max_scan_dist": 10.0,
                "normalized_observation": True,
            },
            "grid_route": {"max_linear_speed": 0.5, "max_angular_speed": 0.7},
        },
        robot_kinematics="differential_drive",
    )
    command = policy(
        {
            "drive_state": np.array([[0.0, 0.0, 3.0 / 50.0, 0.0, 0.0]], dtype=np.float32),
            "rays": np.ones((1, 16), dtype=np.float32),
        }
    )

    assert 0.0 <= command[0] <= 0.5
    assert abs(command[1]) <= 0.7
    assert meta["canonical_algorithm"] == "lidar_grid_route"
    assert meta["observation_level"]["key"] == "lidar_2d"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"
    assert meta["planner_kinematics"]["adapter_name"] == "LidarOccupancyGridRouteAdapter"

    stats = policy._planner_stats()
    assert stats["status"] == "ok"
    assert stats["derived_payload"] == "ego_occupancy_grid"
    assert stats["runtime_inputs"] == ["drive_state", "rays"]


def test_lidar_grid_route_enriched_metadata_marks_testing_only_adapter() -> None:
    """Enriched metadata should distinguish the LiDAR adapter from grid_route itself."""
    meta = enrich_algorithm_metadata(
        algo="lidar_occupancy_grid_route",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )

    assert meta["canonical_algorithm"] == "lidar_grid_route"
    assert meta["policy_semantics"] == "lidar_ego_occupancy_grid_route_tracking"
    assert meta["observation_spec"]["active_mode"] == "sensor_fusion_state"
    assert meta["observation_level"]["key"] == "lidar_2d"
    assert meta["planner_kinematics"]["testing_only_adapter"] is True
