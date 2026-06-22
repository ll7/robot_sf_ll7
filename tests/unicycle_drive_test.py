"""TODO docstring. Document this module."""

from math import pi

from robot_sf.ped_ego.unicycle_drive import (
    UnicycleAction,
    UnicycleDriveSettings,
    UnicycleDriveState,
    UnicycleMotion,
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


def test_unicycle_can_drive_right_curve():
    """TODO docstring. Document this function."""
    motion = UnicycleMotion(UnicycleDriveSettings())
    pose_before, vel_before = ((0, 0), 0), 1
    state = UnicycleDriveState(pose_before, vel_before)

    right_curve_action: UnicycleAction = (0, -0.78)
    motion.move(state, right_curve_action, 1.0)

    steer_straight_action: UnicycleAction = (0, 0)
    motion.move(state, steer_straight_action, 1.0)
    pos_after, orient_after = state.pose

    assert pos_after[0] > 0 and pos_after[1] < 0  # position in 4th quadrant
    assert 1.5 * pi < norm_angle(orient_after) < 2 * pi  # orientation in 4th quadrant


def test_unicycle_can_drive_left_curve():
    """TODO docstring. Document this function."""
    motion = UnicycleMotion(UnicycleDriveSettings())
    pose_before, vel_before = ((0, 0), 0), 1
    state = UnicycleDriveState(pose_before, vel_before)

    left_curve_action: UnicycleAction = (0, 0.78)
    motion.move(state, left_curve_action, 1.0)

    steer_straight_action: UnicycleAction = (0, 0)
    motion.move(state, steer_straight_action, 1.0)
    pos_after, orient_after = state.pose

    assert pos_after[0] > 0 and pos_after[1] > 0  # position in 1st quadrant
    assert 0 < norm_angle(orient_after) < 0.5 * pi  # orientation in 1st quadrant


def test_unicycle_acceleration():
    """TODO docstring. Document this function."""
    motion = UnicycleMotion(UnicycleDriveSettings(allow_backwards=True))
    pose_before, vel_before = ((0, 0), 0), 0
    state = UnicycleDriveState(pose_before, vel_before)

    acceleration_action: UnicycleAction = (1, 0)
    motion.move(state, acceleration_action, 1.0)
    speed = state.velocity
    assert speed == 1  # assert gained speed

    motion.move(state, acceleration_action, 1.0)
    motion.move(state, acceleration_action, 1.0)
    pos_after, orient_after = state.pose
    speed = state.velocity

    assert pos_after[1] == 0 and orient_after == 0  # assert steering straight
    assert pos_after[0] > 0 and speed == 3  # assert moving forward and max speed

    deceleration_action: UnicycleAction = (-1, 0)
    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == 2  # assert lost speed

    motion.move(state, deceleration_action, 1.0)
    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == 0  # assert stopped

    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == -1  # assert moving backwards

    motion = UnicycleMotion(UnicycleDriveSettings(allow_backwards=False))
    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == 0  # assert not moving backwards


def test_unicycle_default_state_moves_with_scalar_current_speed():
    """Default state uses scalar velocity compatible with motion integration."""
    motion = UnicycleMotion(UnicycleDriveSettings())
    state = UnicycleDriveState()

    motion.move(state, (1.0, 0.0), 1.0)

    current_speed = state.current_speed
    assert isinstance(state.velocity, float)
    assert isinstance(current_speed[0], float)
    assert current_speed == (1.0, state.orient)


def test_unicycle_over_limit_acceleration_clipped():
    """Acceleration beyond max_accel is clipped."""
    motion = UnicycleMotion(UnicycleDriveSettings(max_accel=1.0))
    state = UnicycleDriveState(((0.0, 0.0), 0.0), 0.0)
    motion.move(state, (100.0, 0.0), 1.0)
    assert state.velocity == 1.0


def test_unicycle_over_limit_deceleration_clipped():
    """Deceleration beyond max_accel in negative direction is clipped."""
    motion = UnicycleMotion(UnicycleDriveSettings(max_accel=1.0, allow_backwards=True))
    state = UnicycleDriveState(((0.0, 0.0), 0.0), 0.0)
    motion.move(state, (-100.0, 0.0), 1.0)
    assert state.velocity == -1.0


def test_unicycle_over_limit_steering_clipped():
    """Steering angle beyond max_steer is clipped."""
    motion = UnicycleMotion(UnicycleDriveSettings(max_steer=0.78))
    state = UnicycleDriveState(((0.0, 0.0), 0.0), 1.0)
    motion.move(state, (0.0, 100.0), 1.0)
    # state should have moved with clipped steering; check position changed
    pos_after, _ = state.pose
    assert pos_after[0] > 0.0


def test_unicycle_over_limit_negative_steering_clipped():
    """Negative steering angle beyond max_steer is clipped."""
    motion = UnicycleMotion(UnicycleDriveSettings(max_steer=0.78))
    state = UnicycleDriveState(((0.0, 0.0), 0.0), 1.0)
    motion.move(state, (0.0, -100.0), 1.0)
    pos_after, _ = state.pose
    assert pos_after[0] > 0.0
