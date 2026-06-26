"""Regression tests for rollover proxy yaw-rate plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings


class _CurrentSpeedRobot:
    """Robot double exposing the legacy ``(linear_velocity, yaw_rate)`` speed tuple."""

    current_speed = (0.5, 0.5)


def _make_env(robot: object) -> RobotEnv:
    env = RobotEnv.__new__(RobotEnv)
    env.env_config = EnvSettings(rollover_proxy_enabled=True)
    env.simulator = SimpleNamespace(robots=[robot])
    return env


def test_rollover_proxy_uses_bicycle_current_yaw_rate() -> None:
    """Bicycle heading from ``current_speed`` must not be treated as yaw rate."""
    robot = BicycleDriveRobot(
        BicycleDriveSettings(wheelbase=0.5, max_accel=8.0, max_velocity=5.0, max_steer=1.0)
    )
    robot.apply_action((2.0, 0.8), 0.25)

    record = _make_env(robot)._rollover_proxy_record()

    assert record is not None
    assert robot.current_speed[1] != pytest.approx(robot.current_yaw_rate)
    assert record["linear_velocity"] == pytest.approx(robot.current_speed[0])
    assert record["yaw_rate"] == pytest.approx(robot.current_yaw_rate)


def test_rollover_proxy_keeps_current_speed_yaw_rate_fallback() -> None:
    """Drive models without explicit yaw rate keep the legacy speed tuple contract."""
    record = _make_env(_CurrentSpeedRobot())._rollover_proxy_record()

    assert record is not None
    assert record["linear_velocity"] == pytest.approx(0.5)
    assert record["yaw_rate"] == pytest.approx(0.5)
