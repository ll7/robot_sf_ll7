"""Unit tests for holonomic robot model behavior."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.robot.holonomic_drive import (
    HolonomicDriveRobot,
    HolonomicDriveSettings,
)


def test_holonomic_drive_vx_vy_updates_pose() -> None:
    """Holonomic ``vx/vy`` commands should integrate position in Cartesian space."""
    cfg = HolonomicDriveSettings(max_speed=2.0, command_mode="vx_vy")
    robot = HolonomicDriveRobot(cfg)
    robot.apply_action((1.0, 0.0), d_t=0.5)
    assert robot.pos[0] == pytest.approx(0.5)
    assert robot.pos[1] == pytest.approx(0.0)
    assert robot.current_speed[0] == pytest.approx(1.0)
    assert robot.current_speed[1] == pytest.approx(0.0)


def test_holonomic_drive_vx_vy_is_world_frame_and_aligns_heading() -> None:
    """vx/vy commands are world-frame translations; moving changes heading to velocity direction."""
    cfg = HolonomicDriveSettings(max_speed=2.0, command_mode="vx_vy")
    robot = HolonomicDriveRobot(cfg)
    robot.reset_state(((0.0, 0.0), np.pi / 2))

    robot.apply_action((1.0, 0.0), d_t=1.0)

    assert robot.pos == pytest.approx((1.0, 0.0))
    assert robot.pose[1] == pytest.approx(0.0)
    assert robot.state.velocity_xy == pytest.approx((1.0, 0.0))


def test_holonomic_drive_zero_vx_vy_preserves_heading() -> None:
    """A zero world-velocity command should not erase the last heading state."""
    cfg = HolonomicDriveSettings(max_speed=2.0, command_mode="vx_vy")
    robot = HolonomicDriveRobot(cfg)
    robot.reset_state(((0.0, 0.0), 1.2))

    robot.apply_action((0.0, 0.0), d_t=0.5)

    assert robot.pose[1] == pytest.approx(1.2)
    assert robot.state.velocity_xy == pytest.approx((0.0, 0.0))


def test_holonomic_drive_unicycle_mode_updates_heading() -> None:
    """Holonomic ``v/omega`` mode should integrate heading with bounded speed."""
    cfg = HolonomicDriveSettings(max_speed=2.0, max_angular_speed=2.0, command_mode="unicycle_vw")
    robot = HolonomicDriveRobot(cfg)
    robot.apply_action((1.0, 1.0), d_t=1.0)
    assert robot.pose[1] == pytest.approx(1.0)
    assert np.hypot(*robot.state.velocity_xy) == pytest.approx(1.0)
    assert robot.current_speed[1] == pytest.approx(1.0)


def test_holonomic_drive_invalid_command_mode_rejected() -> None:
    """Invalid command_mode values should fail during settings construction."""
    with pytest.raises(ValueError, match="command_mode"):
        HolonomicDriveSettings(command_mode="invalid")
