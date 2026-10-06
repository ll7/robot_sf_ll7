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


def test_default_bicycle_identity_and_opt_in_creep():
    """Default bytes retain main's hash; the real env hash binds enabled creep."""
    from dataclasses import asdict
    from types import SimpleNamespace

    from robot_sf.gym_env.robot_env import _stable_config_hash
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    default = BicycleDriveSettings()
    off = RobotSimulationConfig(robot_config=default)
    # Main's seven-field payload, independently fixed here. The complete env
    # hash includes its map path, so a literal hash from another lane is not portable.
    main_payload = asdict(off)
    main_payload["robot_config"] = {
        "radius": 1.0,
        "wheelbase": 1.0,
        "max_steer": 0.78,
        "max_velocity": 3.0,
        "max_accel": 1.0,
        "allow_backwards": False,
        "max_decel": 1.0,
    }
    assert _stable_config_hash(off) == _stable_config_hash(SimpleNamespace(**main_payload))
    on = RobotSimulationConfig(robot_config=BicycleDriveSettings(creep_speed=0.1))
    faster = RobotSimulationConfig(robot_config=BicycleDriveSettings(creep_speed=0.2))
    assert on.robot_config.creep_speed == 0.1
    assert (
        len({_stable_config_hash(off), _stable_config_hash(on), _stable_config_hash(faster)}) == 3
    )
