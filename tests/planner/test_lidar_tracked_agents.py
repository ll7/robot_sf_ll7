"""Tests for LiDAR-derived tracked-agent SocialForce adapter."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.lidar_tracked_agents import (
    LidarTrackedAgentsConfig,
    LidarTrackedSocialForceAdapter,
    lidar_ray_angles,
    lidar_rays_to_tracked_agents,
    sensor_fusion_to_social_force_observation,
)


def test_lidar_ray_angles_match_range_sensor_endpoint_convention() -> None:
    """Ray angles should exclude the duplicated upper endpoint."""
    angles = lidar_ray_angles(4, visual_angle_portion=1.0)
    np.testing.assert_allclose(angles, [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0])


def test_lidar_ray_angles_cache_reuses_array_for_same_args() -> None:
    """Identical (num_rays, visual_angle_portion) should return the cached object."""
    a = lidar_ray_angles(4, visual_angle_portion=1.0)
    b = lidar_ray_angles(4, visual_angle_portion=1.0)
    assert a is b


def test_lidar_ray_angles_cache_distinguishes_different_args() -> None:
    """Different (num_rays, visual_angle_portion) should produce distinct arrays."""
    a = lidar_ray_angles(4, visual_angle_portion=1.0)
    b = lidar_ray_angles(4, visual_angle_portion=0.5)
    assert a is not b


def test_lidar_ray_angles_returned_array_is_read_only() -> None:
    """Returned angle arrays should not be writeable."""
    angles = lidar_ray_angles(4, visual_angle_portion=1.0)
    assert not angles.flags.writeable
    with pytest.raises(ValueError):
        angles[0] = 0.0


def test_lidar_rays_to_tracked_agents_clusters_adjacent_hits() -> None:
    """Adjacent finite returns should become one visible endpoint track."""
    cfg = LidarTrackedAgentsConfig(
        max_scan_dist=10.0,
        normalized_observation=True,
        cluster_gap_rays=1,
        max_tracks=4,
    )
    rays = np.ones((2, 8), dtype=np.float32)
    rays[-1, 4] = 0.2
    rays[-1, 5] = 0.2

    positions, velocities = lidar_rays_to_tracked_agents(rays, cfg)

    assert positions.shape == (1, 2)
    assert velocities.shape == (1, 2)
    assert positions[0, 0] > 1.0
    assert positions[0, 1] > 0.0
    np.testing.assert_allclose(velocities, np.zeros_like(velocities))


def test_lidar_rays_to_tracked_agents_keeps_nearest_tracks_first() -> None:
    """Track extraction should cap and sort tracks by range."""
    cfg = LidarTrackedAgentsConfig(
        max_scan_dist=10.0,
        normalized_observation=True,
        cluster_gap_rays=0,
        max_tracks=1,
    )
    rays = np.ones((1, 8), dtype=np.float32)
    rays[0, 4] = 0.5
    rays[0, 6] = 0.2

    positions, _velocities = lidar_rays_to_tracked_agents(rays, cfg)

    assert positions.shape == (1, 2)
    assert np.linalg.norm(positions[0]) == np.float32(2.0)


def test_lidar_tracked_agents_config_rejects_min_range_at_scan_cap() -> None:
    """The minimum track range must leave detectable scan space below the cap."""
    with pytest.raises(ValueError, match="min_track_range must be less than max_scan_dist"):
        LidarTrackedAgentsConfig(max_scan_dist=10.0, min_track_range=10.0)


def test_sensor_fusion_adapter_ignores_privileged_socnav_decoys() -> None:
    """Adapter should derive SocialForce input from drive_state and rays only."""
    cfg = LidarTrackedAgentsConfig(max_scan_dist=10.0, normalized_observation=True)
    observation = {
        "drive_state": np.array([[0.0, 0.0, 2.0 / 50.0, 0.0, 0.0]], dtype=np.float32),
        "rays": np.ones((1, 8), dtype=np.float32),
        "robot": {"position": [99.0, 99.0]},
        "pedestrians": {
            "positions": [[99.0, 99.0]],
            "velocities": [[1.0, 1.0]],
            "count": [1],
        },
    }
    observation["rays"][0, 4] = 0.2

    adapted = sensor_fusion_to_social_force_observation(observation, cfg)

    assert adapted["robot"]["position"] == [0.0, 0.0]
    np.testing.assert_allclose(adapted["goal"]["current"], [2.0, 0.0])
    assert adapted["pedestrians"]["count"] == [1]
    assert adapted["pedestrians"]["positions"][0, 0] != 99.0
    np.testing.assert_allclose(adapted["pedestrians"]["velocities"], [[0.0, 0.0]])


def test_lidar_tracked_social_force_adapter_returns_bounded_command_and_diagnostics() -> None:
    """Adapter should plan from normalized LiDAR inputs and expose caveated tracking metadata."""
    adapter = LidarTrackedSocialForceAdapter(
        config=LidarTrackedAgentsConfig(max_scan_dist=10.0, normalized_observation=True)
    )
    observation = {
        "drive_state": np.array([[0.0, 0.0, 3.0 / 50.0, 0.0, 0.0]], dtype=np.float32),
        "rays": np.ones((1, 16), dtype=np.float32),
    }
    observation["rays"][0, 8] = 0.25

    linear, angular = adapter.plan(observation)
    diagnostics = adapter.diagnostics()

    assert 0.0 <= linear <= 1.0
    assert abs(angular) <= 1.0
    assert diagnostics["status"] == "ok"
    assert diagnostics["derived_payload"] == "tracked_agents"
    assert diagnostics["track_count"] == 1
    assert diagnostics["velocity_assumption"] == "zero_velocity_no_identity_persistence"
    assert diagnostics["fallback_or_degraded_success"] is False
    assert "pedestrians" in diagnostics["forbidden_runtime_inputs_not_read"]


def test_lidar_tracked_social_force_adapter_fails_closed_without_rays() -> None:
    """Missing LiDAR inputs should produce an explicit not-available stop."""
    adapter = LidarTrackedSocialForceAdapter()

    assert adapter.plan({"drive_state": np.zeros((1, 5), dtype=np.float32)}) == (0.0, 0.0)
    diagnostics = adapter.diagnostics()
    assert diagnostics["status"] == "not_available"
    assert "drive_state and rays" in str(diagnostics["error"])
