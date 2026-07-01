"""Tests for selectable issue-3976 kinematic robot dynamics."""

from __future__ import annotations

from math import inf, nan, pi

import pytest

from robot_sf.robot.dynamics import RobotDynamicsState, build_robot_dynamics


@pytest.mark.parametrize(
    ("name", "control", "expected_pose"),
    [
        ("holonomic_disc", (1.0, 0.5), (1.0, 0.5, pytest.approx(0.463647609))),
        ("differential_drive", (20.0, 20.0), (1.0, 0.0, 0.0)),
        ("unicycle", (1.0, 0.0), (1.0, 0.0, 0.0)),
        ("kinematic_bicycle", (1.0, 0.0), (1.0, 0.0, 0.0)),
    ],
)
def test_kinematic_dynamics_step_forward_known_control(
    name: str,
    control: tuple[float, float],
    expected_pose: tuple[float, float, float],
) -> None:
    """Each named dynamics model advances a known one-second command."""

    model = build_robot_dynamics(name)
    next_state = model.step(RobotDynamicsState(), control, dt=1.0)

    assert next_state.x == pytest.approx(expected_pose[0])
    assert next_state.y == pytest.approx(expected_pose[1])
    assert next_state.heading == expected_pose[2]


def test_unicycle_integrates_arc_with_midpoint_heading() -> None:
    """Unicycle motion follows an arc for simultaneous translation and rotation."""

    model = build_robot_dynamics("unicycle", max_linear_speed=2.0, max_angular_speed=2.0)

    next_state = model.step(RobotDynamicsState(), (1.0, pi / 2.0), dt=1.0)

    assert next_state.x == pytest.approx(2**0.5 / 2.0)
    assert next_state.y == pytest.approx(2**0.5 / 2.0)
    assert next_state.heading == pytest.approx(pi / 2.0)
    assert next_state.angular_speed == pytest.approx(pi / 2.0)


def test_differential_drive_turns_from_wheel_speed_difference() -> None:
    """Differential drive derives linear and angular speed from wheel speeds."""

    model = build_robot_dynamics(
        "differential_drive",
        wheel_radius=0.1,
        track_width=0.5,
        max_wheel_angular_speed=20.0,
    )

    next_state = model.step(RobotDynamicsState(), (5.0, 10.0), dt=1.0)

    assert next_state.linear_speed == pytest.approx(0.75)
    assert next_state.angular_speed == pytest.approx(1.0)
    assert next_state.heading == pytest.approx(1.0)


def test_kinematic_bicycle_heading_depends_on_steering_angle() -> None:
    """Kinematic bicycle heading changes with steering angle and wheelbase."""

    model = build_robot_dynamics(
        "kinematic_bicycle",
        wheelbase=2.0,
        max_linear_speed=5.0,
        max_steering_angle=0.8,
    )

    next_state = model.step(RobotDynamicsState(), (2.0, 0.5), dt=1.0)

    assert next_state.x == pytest.approx(1.925851153)
    assert next_state.y == pytest.approx(0.539534371)
    assert next_state.heading == pytest.approx(0.5463024898)
    assert next_state.steering_angle == pytest.approx(0.5)


def test_unknown_dynamics_name_fails_closed() -> None:
    """Unknown dynamics names fail closed instead of silently falling back."""

    with pytest.raises(ValueError, match="Unknown robot dynamics model"):
        build_robot_dynamics("tire_slip")


def test_state_pose_matches_repository_pose_shape() -> None:
    """Shared dynamics state exposes the existing nested robot-pose tuple."""

    state = RobotDynamicsState(x=1.0, y=2.0, heading=0.25)

    assert state.pose == ((1.0, 2.0), 0.25)


@pytest.mark.parametrize(
    ("name", "dt"),
    [
        ("holonomic_disc", 0.0),
        ("differential_drive", 0.0),
        ("unicycle", 0.0),
        ("kinematic_bicycle", 0.0),
        ("holonomic_disc", nan),
        ("unicycle", inf),
    ],
)
def test_dynamics_reject_invalid_dt(name: str, dt: float) -> None:
    """Dynamics models reject non-positive and non-finite timesteps."""

    model = build_robot_dynamics(name)

    with pytest.raises(ValueError, match="dt must be finite and positive"):
        model.step(RobotDynamicsState(), (0.0, 0.0), dt=dt)


@pytest.mark.parametrize(
    "name",
    ["holonomic_disc", "differential_drive", "unicycle", "kinematic_bicycle"],
)
def test_dynamics_reject_wrong_control_dimension(name: str) -> None:
    """All dynamics models require a two-value control input."""

    model = build_robot_dynamics(name)

    with pytest.raises(ValueError, match="exactly two values"):
        model.step(RobotDynamicsState(), (1.0,), dt=1.0)


def test_build_robot_dynamics_accepts_normalized_names_and_kwargs() -> None:
    """The registry normalizes names and forwards model-specific parameters."""

    model = build_robot_dynamics(" Unicycle ", max_linear_speed=4.0)

    next_state = model.step(RobotDynamicsState(), (10.0, 0.0), dt=1.0)
    assert next_state.linear_speed == pytest.approx(4.0)


def test_holonomic_disc_clips_velocity_norm_and_preserves_stopped_heading() -> None:
    """Holonomic disc clips vector speed and keeps heading when stopped."""

    model = build_robot_dynamics("holonomic_disc", max_speed=1.0)
    clipped = model.step(RobotDynamicsState(), (3.0, 4.0), dt=1.0)
    stopped = model.step(RobotDynamicsState(heading=0.75), (0.0, 0.0), dt=1.0)

    assert clipped.x == pytest.approx(0.6)
    assert clipped.y == pytest.approx(0.8)
    assert clipped.linear_speed == pytest.approx(1.0)
    assert stopped.heading == pytest.approx(0.75)


def test_unicycle_clips_speed_and_angular_velocity() -> None:
    """Unicycle commands are clipped to configured velocity bounds."""

    model = build_robot_dynamics("unicycle", max_linear_speed=1.0, max_angular_speed=0.5)

    next_state = model.step(RobotDynamicsState(), (5.0, 2.0), dt=1.0)

    assert next_state.linear_speed == pytest.approx(1.0)
    assert next_state.angular_speed == pytest.approx(0.5)


def test_differential_drive_clips_wheel_speeds() -> None:
    """Differential-drive wheel speeds are clipped before twist derivation."""

    model = build_robot_dynamics(
        "differential_drive",
        wheel_radius=0.1,
        track_width=1.0,
        max_wheel_angular_speed=2.0,
    )

    next_state = model.step(RobotDynamicsState(), (100.0, 100.0), dt=1.0)

    assert next_state.linear_speed == pytest.approx(0.2)
    assert next_state.angular_speed == pytest.approx(0.0)


def test_kinematic_bicycle_clips_speed_and_steering() -> None:
    """Kinematic bicycle clips speed and steering before integration."""

    model = build_robot_dynamics(
        "kinematic_bicycle",
        wheelbase=2.0,
        max_linear_speed=1.0,
        max_steering_angle=0.25,
    )

    next_state = model.step(RobotDynamicsState(), (5.0, 1.0), dt=1.0)

    assert next_state.x == pytest.approx(0.997963)
    assert next_state.steering_angle == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("name", "kwargs", "match"),
    [
        ("holonomic_disc", {"max_speed": 0.0}, "max_speed"),
        ("holonomic_disc", {"max_speed": nan}, "max_speed"),
        ("differential_drive", {"wheel_radius": 0.0}, "wheel_radius"),
        ("differential_drive", {"wheel_radius": inf}, "wheel_radius"),
        ("differential_drive", {"track_width": 0.0}, "track_width"),
        ("differential_drive", {"track_width": nan}, "track_width"),
        ("differential_drive", {"max_wheel_angular_speed": 0.0}, "max_wheel"),
        ("differential_drive", {"max_wheel_angular_speed": inf}, "max_wheel"),
        ("unicycle", {"max_linear_speed": 0.0}, "max_linear_speed"),
        ("unicycle", {"max_linear_speed": nan}, "max_linear_speed"),
        ("unicycle", {"max_angular_speed": 0.0}, "max_angular_speed"),
        ("unicycle", {"max_angular_speed": inf}, "max_angular_speed"),
        ("kinematic_bicycle", {"wheelbase": 0.0}, "wheelbase"),
        ("kinematic_bicycle", {"wheelbase": nan}, "wheelbase"),
        ("kinematic_bicycle", {"max_linear_speed": 0.0}, "max_linear_speed"),
        ("kinematic_bicycle", {"max_linear_speed": inf}, "max_linear_speed"),
        ("kinematic_bicycle", {"max_steering_angle": 0.0}, "max_steering_angle"),
        ("kinematic_bicycle", {"max_steering_angle": 1.6}, "max_steering_angle"),
        ("kinematic_bicycle", {"max_steering_angle": nan}, "max_steering_angle"),
        ("kinematic_bicycle", {"max_steering_angle": inf}, "max_steering_angle"),
    ],
)
def test_dynamics_constructor_validation(
    name: str,
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Dynamics constructors fail closed for invalid physical limits."""

    with pytest.raises(ValueError, match=match):
        build_robot_dynamics(name, **kwargs)
