"""Tests for goal sensor bounds and observation clipping."""

from __future__ import annotations

import pytest

from robot_sf.sensor.goal_sensor import (
    TARGET_DISTANCE_CAP_M,
    target_sensor_obs,
    target_sensor_space,
)


def test_target_sensor_space_uses_global_cap() -> None:
    """Target sensor space should use a fixed cap for cross-scenario compatibility."""
    space = target_sensor_space(999.0)
    assert space.high[0] == pytest.approx(TARGET_DISTANCE_CAP_M)


def test_target_sensor_obs_clips_distance() -> None:
    """Target sensor observations should clip distance to keep values in-bounds."""
    robot_pose = ((0.0, 0.0), 0.0)
    goal_pos = (TARGET_DISTANCE_CAP_M * 3.0, 0.0)
    dist, _angle, _next_angle = target_sensor_obs(robot_pose, goal_pos, None)
    assert dist == pytest.approx(TARGET_DISTANCE_CAP_M)
