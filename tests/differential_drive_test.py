"""Tests for differential-drive kinematics and wheel-speed conventions."""

from math import pi

import pytest

from robot_sf.robot.differential_drive import (
    DifferentialDriveMotion,
    DifferentialDriveRobot,
    DifferentialDriveSettings,
    DifferentialDriveState,
)


def norm_angle(angle: float) -> float:
    """Normalize an angle to the ``[0, 2*pi)`` range.

    Args:
        angle: Angle in radians.

    Returns:
        Equivalent non-negative angle in radians.
    """
    while angle < 0:
        angle += 2 * pi
    while angle >= 2 * pi:
        angle -= 2 * pi
    return angle


def test_can_drive_right_curve():
    """A right-turn action should move and orient the robot into the fourth quadrant."""
    motion = DifferentialDriveMotion(DifferentialDriveSettings(1, 1, 1, 1))
    pose_before, vel_before, wheel_speeds = ((0, 0), 0), (1, 0), (1, 1)
    state = DifferentialDriveState(pose_before, vel_before, wheel_speeds, wheel_speeds)
    right_curve_action = (1, -0.5)
    motion.move(state, right_curve_action, 1.0)
    pos_after, orient_after = state.pose
    assert pos_after[0] > 0 and pos_after[1] < 0  # position in 4th quadrant
    assert 1.5 * pi < norm_angle(orient_after) < 2 * pi  # orientation in 4th quadrant


def test_can_drive_left_curve():
    """A left-turn action should move and orient the robot into the first quadrant."""
    motion = DifferentialDriveMotion(DifferentialDriveSettings(1, 1, 1, 1))
    pose_before, vel_before, wheel_speeds = ((0, 0), 0), (1, 0), (1, 1)
    state = DifferentialDriveState(pose_before, vel_before, wheel_speeds, wheel_speeds)
    left_curve_action = (1, 0.5)
    motion.move(state, left_curve_action, 1.0)
    pos_after, orient_after = state.pose
    assert pos_after[0] > 0 and pos_after[1] > 0  # position in 1st quadrant
    assert 0 < norm_angle(orient_after) < 0.5 * pi  # orientation in 1st quadrant


def test_straight_line_distance_matches_kinematics() -> None:
    """Verify kinematics math so straight motion stays accurate for planners."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=5.0,
            max_angular_speed=5.0,
            wheel_radius=1.0,
            interaxis_length=1.0,
        ),
    )
    pose_before = ((0.0, 0.0), 0.0)
    state = DifferentialDriveState(pose_before, (1.0, 0.0), (1.0, 1.0), (1.0, 1.0))
    motion.move(state, (0.0, 0.0), 1.0)
    pos_after, orient_after = state.pose
    assert pos_after == (1.0, 0.0)
    assert orient_after == 0.0


def test_resulting_wheel_speeds_are_angular_rates() -> None:
    """WheelSpeedState stores left/right wheel angular velocity in rad/s."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=5.0,
            max_angular_speed=5.0,
            wheel_radius=0.5,
            interaxis_length=1.0,
        ),
    )

    straight_left_rad_s, straight_right_rad_s = motion._resulting_wheel_speeds((1.0, 0.0))
    left_wheel_rad_s, right_wheel_rad_s = motion._resulting_wheel_speeds((1.0, 2.0))

    assert straight_left_rad_s == 2.0
    assert straight_right_rad_s == 2.0
    assert left_wheel_rad_s == 0.0
    assert right_wheel_rad_s == 4.0


def test_action_space_uses_delta_velocity_bounds() -> None:
    """Action bounds represent velocity deltas, not absolute speed limits."""
    robot = DifferentialDriveRobot(
        DifferentialDriveSettings(
            max_linear_speed=3.0,
            max_angular_speed=2.0,
            max_linear_accel=0.4,
            max_angular_accel=0.3,
        )
    )

    assert tuple(robot.action_space.low) == pytest.approx((-0.4, -0.3))
    assert tuple(robot.action_space.high) == pytest.approx((0.4, 0.3))


def test_observation_space_still_uses_speed_bounds() -> None:
    """Observation velocity bounds remain absolute robot speed limits."""
    robot = DifferentialDriveRobot(
        DifferentialDriveSettings(
            max_linear_speed=3.0,
            max_angular_speed=2.0,
            max_linear_accel=0.4,
            max_angular_accel=0.3,
        )
    )

    assert tuple(robot.observation_space.low) == pytest.approx((0.0, -2.0))
    assert tuple(robot.observation_space.high) == pytest.approx((3.0, 2.0))


def test_over_limit_linear_delta_clipped_to_action_bound() -> None:
    """Linear velocity deltas are clipped before updating speed."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(max_linear_speed=5.0, max_linear_accel=0.4)
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    motion.move(state, (100.0, 0.0), 1.0)

    dot_x, _ = state.velocity
    assert dot_x == 0.4


def test_over_limit_angular_delta_clipped_to_action_bound() -> None:
    """Angular velocity deltas are clipped before updating angular speed."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(max_angular_speed=5.0, max_angular_accel=0.25)
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    motion.move(state, (0.0, 100.0), 1.0)

    _, dot_orient = state.velocity
    assert dot_orient == 0.25


def test_over_limit_linear_speed_clipped() -> None:
    """Over-limit linear delta is clipped to max_linear_speed."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=2.0,
            max_angular_speed=5.0,
            max_linear_accel=100.0,
        )
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    # Apply a huge positive linear delta
    motion.move(state, (100.0, 0.0), 1.0)
    dot_x, _ = state.velocity
    assert dot_x == 2.0


def test_over_limit_angular_speed_clipped() -> None:
    """Over-limit angular delta is clipped to max_angular_speed."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=5.0,
            max_angular_speed=1.0,
            max_angular_accel=100.0,
        )
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    motion.move(state, (0.0, 100.0), 1.0)
    _, dot_orient = state.velocity
    assert dot_orient == 1.0


def test_over_limit_negative_linear_speed_clipped() -> None:
    """Over-limit negative linear delta is clipped to min_linear_speed."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(max_linear_speed=2.0, max_angular_speed=5.0)
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    motion.move(state, (-100.0, 0.0), 1.0)
    dot_x, _ = state.velocity
    assert dot_x == 0.0  # allow_backwards=False


def test_over_limit_negative_angular_speed_clipped() -> None:
    """Negative over-limit angular delta is clipped."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=5.0,
            max_angular_speed=1.0,
            max_angular_accel=100.0,
        )
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    motion.move(state, (0.0, -100.0), 1.0)
    _, dot_orient = state.velocity
    assert dot_orient == -1.0


def test_velocity_delta_scales_with_timestep() -> None:
    """Per-step velocity delta scales linearly with d_t (issue #3711).

    The action is a commanded acceleration; the realized velocity change over one
    step must be ``accel * d_t``. Halving the timestep must halve the velocity
    delta for the same command.
    """
    settings = DifferentialDriveSettings(max_linear_speed=10.0, max_linear_accel=5.0)

    state_half = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    DifferentialDriveMotion(settings).move(state_half, (2.0, 0.0), 0.5)

    state_tenth = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    DifferentialDriveMotion(settings).move(state_tenth, (2.0, 0.0), 0.1)

    # delta = accel (2.0) * d_t
    assert state_half.velocity[0] == pytest.approx(1.0)
    assert state_tenth.velocity[0] == pytest.approx(0.2)
    # Halving d_t (0.5 -> 0.1, factor 5) scales the delta by the same factor.
    assert state_half.velocity[0] == pytest.approx(5.0 * state_tenth.velocity[0])


def test_acceleration_clip_then_scaled_by_dt() -> None:
    """Over-limit accel commands are clipped, then integrated over d_t (issue #3711).

    With ``max_linear_accel=0.4`` and ``d_t=0.1`` the maximum reachable per-step
    delta is ``0.4 * 0.1 = 0.04`` (contrast the d_t=1.0 case which yields 0.4).
    """
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(max_linear_speed=10.0, max_linear_accel=0.4)
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    motion.move(state, (100.0, 0.0), 0.1)

    assert state.velocity[0] == pytest.approx(0.04)


def test_velocity_update_is_timestep_invariant_over_equal_real_time() -> None:
    """Equal real elapsed time yields the same velocity regardless of step size.

    Integrating a constant acceleration command for 1.0s in a single d_t=1.0 step
    must match ten d_t=0.1 substeps. Before issue #3711 the finer schedule
    inflated the velocity tenfold because the delta ignored d_t.
    """
    settings = DifferentialDriveSettings(
        max_linear_speed=10.0,
        max_angular_speed=10.0,
        max_linear_accel=5.0,
        max_angular_accel=5.0,
    )
    accel = (1.0, 0.5)

    coarse = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    DifferentialDriveMotion(settings).move(coarse, accel, 1.0)

    fine = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    fine_motion = DifferentialDriveMotion(settings)
    for _ in range(10):
        fine_motion.move(fine, accel, 0.1)

    assert fine.velocity[0] == pytest.approx(coarse.velocity[0])
    assert fine.velocity[1] == pytest.approx(coarse.velocity[1])
    # Sanity: the constant command reaches accel * total_time after 1.0s.
    assert coarse.velocity[0] == pytest.approx(1.0)
    assert coarse.velocity[1] == pytest.approx(0.5)


def test_in_place_rotation_matches_kinematics() -> None:
    """Verify in-place rotation math to prevent drift in heading updates."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=5.0,
            max_angular_speed=5.0,
            wheel_radius=1.0,
            interaxis_length=1.0,
        ),
    )
    pose_before = ((0.0, 0.0), 0.0)
    state = DifferentialDriveState(pose_before, (0.0, 2.0), (-1.0, 1.0), (-1.0, 1.0))
    motion.move(state, (0.0, 0.0), 1.0)
    pos_after, orient_after = state.pose
    assert pos_after == (0.0, 0.0)
    assert orient_after == 2.0
