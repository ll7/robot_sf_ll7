"""Tests for LiDAR-derived ego occupancy planner adapters."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.lidar_occupancy import (
    LidarOccupancyGridConfig,
    LidarOccupancyPlannerAdapter,
    lidar_rays_to_occupancy_observation,
)
from robot_sf.planner.safety_barrier import SafetyBarrierPlannerAdapter, SafetyBarrierPlannerConfig


def _raw_lidar_config() -> LidarOccupancyGridConfig:
    """Return a compact single-ray config for deterministic adapter tests."""
    return LidarOccupancyGridConfig(
        resolution=0.5,
        width=4.0,
        height=4.0,
        max_range=4.0,
        angle_min=0.0,
        angle_max=0.0,
        obstacle_inflation_cells=0,
        normalized_rays=False,
        normalized_drive_state=False,
    )


def test_lidar_rays_to_occupancy_marks_endpoint_without_pedestrian_state() -> None:
    """A finite ray hit should create ego occupancy without adding tracked pedestrians."""
    planner_obs = lidar_rays_to_occupancy_observation(
        {
            "rays": np.asarray([1.0], dtype=np.float32),
            "drive_state": np.asarray([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32),
        },
        _raw_lidar_config(),
    )

    grid = planner_obs["occupancy_grid"]

    assert grid.shape == (3, 8, 8)
    assert grid[0, 4, 6] == 1.0
    assert grid[2, 4, 6] == 1.0
    assert planner_obs["pedestrians"]["count"][0] == 0
    assert planner_obs["goal"]["current"].tolist() == [2.0, 0.0]
    assert planner_obs["occupancy_grid_meta_use_ego_frame"][0] == 1.0
    assert planner_obs["occupancy_grid_meta_channel_indices"].tolist() == [0, 1, -1, 2]


def test_lidar_occupancy_wrapper_drives_safety_barrier_from_rays_only() -> None:
    """The safety-barrier wrapper should slow down when the LiDAR grid blocks forward motion."""
    wrapper = LidarOccupancyPlannerAdapter(
        SafetyBarrierPlannerAdapter(
            SafetyBarrierPlannerConfig(stop_distance=0.6, safe_distance=1.0)
        ),
        _raw_lidar_config(),
    )
    clear_obs = {
        "rays": np.asarray([4.0], dtype=np.float32),
        "drive_state": np.asarray([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32),
    }
    blocked_obs = {
        "rays": np.asarray([0.4], dtype=np.float32),
        "drive_state": np.asarray([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32),
    }

    clear_linear, _clear_angular = wrapper.plan(clear_obs)
    blocked_linear, blocked_angular = wrapper.plan(blocked_obs)

    assert clear_linear > 0.0
    assert blocked_linear == 0.0
    assert abs(blocked_angular) > 0.0
    assert wrapper.diagnostics()["lidar_occupancy_adapter"]["converted_observations"] == 2


def test_lidar_occupancy_wrapper_fails_closed_without_rays() -> None:
    """Missing LiDAR rays should stop the adapter instead of falling back to privileged state."""
    wrapper = LidarOccupancyPlannerAdapter(
        SafetyBarrierPlannerAdapter(),
        _raw_lidar_config(),
    )

    command = wrapper.plan({"drive_state": np.asarray([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32)})

    stats = wrapper.diagnostics()["lidar_occupancy_adapter"]
    assert command == (0.0, 0.0)
    assert stats["unavailable_observations"] == 1
    assert "rays" in stats["last_error"]
