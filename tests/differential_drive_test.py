"""TODO docstring. Document this module."""

from math import pi

from robot_sf.robot.differential_drive import (
    DifferentialDriveMotion,
    DifferentialDriveSettings,
    DifferentialDriveState,
)


def norm_angle(angle: float) -> float:
    """TODO docstring. Document this function.

    Args:
        angle: TODO docstring.

    Returns:
        TODO docstring.
    """
    while angle < 0:
        angle += 2 * pi
    while angle >= 2 * pi:
        angle -= 2 * pi
    return angle


def test_can_drive_right_curve():
    """TODO docstring. Document this function."""
    motion = DifferentialDriveMotion(DifferentialDriveSettings(1, 1, 1, 1))
    pose_before, vel_before, wheel_speeds = ((0, 0), 0), (1, 0), (1, 1)
    state = DifferentialDriveState(pose_before, vel_before, wheel_speeds, wheel_speeds)
    right_curve_action = (1, -0.5)
    motion.move(state, right_curve_action, 1.0)
    pos_after, orient_after = state.pose
    assert pos_after[0] > 0 and pos_after[1] < 0  # position in 4th quadrant
    assert 1.5 * pi < norm_angle(orient_after) < 2 * pi  # orientation in 4th quadrant


def test_can_drive_left_curve():
    """TODO docstring. Document this function."""
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
