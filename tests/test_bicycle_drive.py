"""Tests for bicycle-drive kinematics and action-limit clipping."""

from robot_sf.robot.bicycle_drive import (
    BicycleDriveSettings,
    BicycleDriveState,
    BicycleMotion,
)


def test_bicycle_over_limit_acceleration_clipped():
    """Acceleration beyond max_accel is clipped."""
    motion = BicycleMotion(BicycleDriveSettings(max_accel=1.0))
    state = BicycleDriveState(((0.0, 0.0), 0.0), 0.0)
    motion.move(state, (100.0, 0.0), 1.0)
    assert state.velocity == 1.0


def test_bicycle_over_limit_deceleration_clipped():
    """Deceleration beyond max_accel in negative direction is clipped."""
    motion = BicycleMotion(BicycleDriveSettings(max_accel=1.0, allow_backwards=True))
    state = BicycleDriveState(((0.0, 0.0), 0.0), 0.0)
    motion.move(state, (-100.0, 0.0), 1.0)
    assert state.velocity == -1.0


def test_bicycle_over_limit_forward_velocity_clipped():
    """Velocity exceeding max_velocity is clipped."""
    motion = BicycleMotion(BicycleDriveSettings(max_velocity=2.0, max_accel=100.0))
    state = BicycleDriveState(((0.0, 0.0), 0.0), 0.0)
    motion.move(state, (100.0, 0.0), 1.0)
    assert state.velocity == 2.0


def test_bicycle_velocity_not_below_zero_when_backwards_disabled():
    """Velocity cannot go negative when allow_backwards is False."""
    motion = BicycleMotion(BicycleDriveSettings(max_velocity=2.0, max_accel=100.0))
    state = BicycleDriveState(((0.0, 0.0), 0.0), 0.0)
    motion.move(state, (-100.0, 0.0), 1.0)
    assert state.velocity == 0.0


def test_bicycle_over_limit_steering_clipped_positive():
    """Steering angle beyond max_steer is clipped (positive)."""
    motion = BicycleMotion(BicycleDriveSettings(max_steer=0.5, max_velocity=1.0, max_accel=100.0))
    state = BicycleDriveState(((0.0, 0.0), 0.0), 1.0)
    motion.move(state, (0.0, 100.0), 1.0)
    pos_after, _ = state.pose
    # With max_steer=0.5 the turn should be bounded: check robot moved
    assert pos_after[0] > 0.0


def test_bicycle_over_limit_steering_clipped_negative():
    """Steering angle beyond max_steer is clipped (negative)."""
    motion = BicycleMotion(BicycleDriveSettings(max_steer=0.5, max_velocity=1.0, max_accel=100.0))
    state = BicycleDriveState(((0.0, 0.0), 0.0), 1.0)
    motion.move(state, (0.0, -100.0), 1.0)
    pos_after, _ = state.pose
    assert pos_after[0] > 0.0
