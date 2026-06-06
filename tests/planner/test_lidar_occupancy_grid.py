"""Tests for LiDAR-derived ego occupancy grid adapter."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.lidar_occupancy_grid import (
    LidarOccupancyGridConfig,
    LidarOccupancyGridRouteAdapter,
    build_lidar_grid_route_config,
    lidar_ray_angles,
    lidar_rays_to_ego_occupancy_grid,
    sensor_fusion_to_grid_route_observation,
)


def test_lidar_ray_angles_match_range_sensor_endpoint_convention() -> None:
    """Ray angles should exclude the duplicated upper endpoint."""
    angles = lidar_ray_angles(4, visual_angle_portion=1.0)
    np.testing.assert_allclose(angles, [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0])


def test_lidar_ray_angles_reuses_immutable_cached_array() -> None:
    """Stable LiDAR grid configurations should not rebuild angle arrays every step."""
    first = lidar_ray_angles(16, visual_angle_portion=1.0 / 3.0)
    second = lidar_ray_angles(16, visual_angle_portion=1.0 / 3.0)

    assert first is second
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0] = 0.0


def test_lidar_rays_to_ego_occupancy_grid_marks_hit_endpoint_only() -> None:
    """Finite non-max range returns should rasterize into the ego obstacle channel."""
    cfg = LidarOccupancyGridConfig(
        resolution=0.5,
        width=4.0,
        height=4.0,
        max_scan_dist=10.0,
        obstacle_thickness_cells=0,
        normalized_observation=True,
    )
    rays = np.ones((2, 4), dtype=np.float32)
    rays[-1, 2] = 0.1  # +x ray at 1m after unnormalization.

    grid, meta = lidar_rays_to_ego_occupancy_grid(rays, cfg)

    assert grid.shape == (3, 8, 8)
    assert meta["use_ego_frame"] == [1.0]
    assert meta["channel_indices"] == [0, 1, -1, 2]
    assert grid[0, 4, 6] == 1.0
    assert grid[2, 4, 6] == 1.0
    assert np.count_nonzero(grid[1]) == 0


def test_sensor_fusion_adapter_uses_only_drive_state_and_rays() -> None:
    """Sensor-fusion observations should become grid-route observations without SocNav state."""
    cfg = LidarOccupancyGridConfig(
        resolution=0.5,
        width=4.0,
        height=4.0,
        max_scan_dist=10.0,
        obstacle_thickness_cells=0,
        normalized_observation=True,
    )
    observation = {
        "drive_state": np.array([[0.0, 0.0, 2.0 / 50.0, 0.0, 0.0]], dtype=np.float32),
        "rays": np.ones((1, 8), dtype=np.float32),
        "robot": {"position": [99.0, 99.0]},
        "pedestrians": {"count": [99]},
    }

    adapted = sensor_fusion_to_grid_route_observation(observation, cfg)

    assert adapted["robot"]["position"] == [0.0, 0.0]
    np.testing.assert_allclose(adapted["goal"]["current"], [2.0, 0.0])
    assert adapted["pedestrians"]["count"] == [0]
    assert adapted["occupancy_grid"].shape == (3, 8, 8)


def test_lidar_grid_route_adapter_returns_bounded_command_and_diagnostics() -> None:
    """Adapter should plan from normalized LiDAR inputs and report its testing-only contract."""
    adapter = LidarOccupancyGridRouteAdapter(
        config=LidarOccupancyGridConfig(
            width=6.0,
            height=6.0,
            max_scan_dist=10.0,
            normalized_observation=True,
        )
    )
    observation = {
        "drive_state": np.array([[0.0, 0.0, 3.0 / 50.0, 0.0, 0.0]], dtype=np.float32),
        "rays": np.ones((1, 16), dtype=np.float32),
    }

    linear, angular = adapter.plan(observation)
    diagnostics = adapter.diagnostics()

    assert 0.0 <= linear <= 0.9
    assert abs(angular) <= 1.2
    assert diagnostics["status"] == "ok"
    assert diagnostics["observation_level"] == "lidar_2d"
    assert diagnostics["runtime_inputs"] == ["drive_state", "rays"]
    assert "map" in diagnostics["forbidden_runtime_inputs_not_read"]


def test_lidar_grid_route_adapter_fails_closed_without_rays() -> None:
    """Missing LiDAR inputs should produce an explicit not-available stop."""
    adapter = LidarOccupancyGridRouteAdapter()

    assert adapter.plan({"drive_state": np.zeros((1, 5), dtype=np.float32)}) == (0.0, 0.0)
    diagnostics = adapter.diagnostics()
    assert diagnostics["status"] == "not_available"
    assert "drive_state and rays" in str(diagnostics["error"])


def test_lidar_grid_route_adapter_fails_closed_on_unexpected_planner_error() -> None:
    """Unexpected wrapped-planner errors should not escape the LiDAR adapter."""

    class _ExplodingGridRoute:
        def plan(self, observation):
            del observation
            raise RuntimeError("planner exploded")

    adapter = LidarOccupancyGridRouteAdapter(grid_route=_ExplodingGridRoute())
    observation = {
        "drive_state": np.array([[0.0, 0.0, 3.0 / 50.0, 0.0, 0.0]], dtype=np.float32),
        "rays": np.ones((1, 16), dtype=np.float32),
    }

    assert adapter.plan(observation) == (0.0, 0.0)
    diagnostics = adapter.diagnostics()
    assert diagnostics["status"] == "not_available"
    assert diagnostics["error"] == "planner exploded"


def test_lidar_grid_route_config_treats_yaml_nulls_as_defaults() -> None:
    """Explicit null config values should not crash numeric parsing."""
    parsed = build_lidar_grid_route_config(
        {
            "lidar_occupancy": {
                "resolution": None,
                "width": None,
                "height": None,
                "max_scan_dist": None,
                "visual_angle_portion": None,
                "obstacle_thickness_cells": None,
                "normalized_observation": None,
                "target_distance_scale": None,
                "linear_speed_scale": None,
                "angular_speed_scale": None,
                "robot_radius": None,
            }
        }
    )

    assert parsed.lidar_occupancy == LidarOccupancyGridConfig()
