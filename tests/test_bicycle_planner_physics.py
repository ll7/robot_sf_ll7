"""Diagnostic regression witnesses, with independent physical-law assertions."""

import math

import pytest

from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings


def adapter():
    robot = BicycleDriveRobot(
        BicycleDriveSettings(radius=1.0, wheelbase=0.85, max_steer=0.6, max_velocity=2.0)
    )
    return robot, PlannerActionAdapter(robot, robot.action_space, 0.1)


def test_counter_rejects_low_speed_excess_yaw():
    _robot, a = adapter()
    model = a._default_kinematics_model()
    assert not model.is_feasible((0.1, 0.5)), ".5 rad/s exceeds .1*tan(.6)/.85 = .0804867 rad/s"
    v, w = model.project((0.1, 0.5))
    assert abs(w) <= abs(v) * math.tan(0.6) / 0.85 + 1e-12


def test_steering_uses_achievable_speed():
    robot, a = adapter()
    robot.apply_action(tuple(a.from_velocity_command((2.0, 0.08))), 0.1)
    assert robot.state.velocity == pytest.approx(0.1)
    assert robot.current_yaw_rate == pytest.approx(0.08, abs=1e-6), (
        "reachable yaw .08 must survive acceleration clipping"
    )


def test_zero_speed_turn_creeps():
    robot, a = adapter()
    robot.apply_action(tuple(a.from_velocity_command((0.0, 0.5))), 0.1)
    assert robot.state.velocity > 0.0, "turn request lost at standstill"
    assert robot.current_yaw_rate > 0.0


def test_stop_stays_stopped():
    robot, a = adapter()
    robot.apply_action(tuple(a.from_velocity_command((0.0, 0.0))), 0.1)
    assert robot.state.velocity == 0.0 and robot.current_yaw_rate == 0.0


def test_map_runner_world_velocity_not_silently_zeroed():
    from types import SimpleNamespace

    from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
        policy_command_to_env_action,
    )

    robot, _a = adapter()
    env = SimpleNamespace(
        simulator=SimpleNamespace(robots=[robot]), action_space=robot.action_space
    )
    cfg = SimpleNamespace(
        robot_config=robot.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    action = policy_command_to_env_action(
        env=env, config=cfg, command={"command_kind": "holonomic_vxy_world", "vx": 1.0, "vy": 1.0}
    )
    robot.apply_action(tuple(action), 0.1)
    assert robot.state.velocity > 0.0 and robot.current_yaw_rate > 0.0, (
        "world-velocity caps must come from bicycle settings"
    )


def test_braking_uses_achievable_speed_and_deceleration():
    robot = BicycleDriveRobot(
        BicycleDriveSettings(
            wheelbase=0.85, max_steer=0.6, max_velocity=2.0, max_accel=1.0, max_decel=3.0
        )
    )
    robot.state.velocity = 1.0
    a = PlannerActionAdapter(robot, robot.action_space, 0.1)
    robot.apply_action(tuple(a.from_velocity_command((0.5, 0.2))), 0.1)
    assert robot.state.velocity == pytest.approx(0.7)
    assert robot.current_yaw_rate == pytest.approx(0.2, abs=1e-6)


def test_reverse_turn_yaw_has_requested_sign():
    robot = BicycleDriveRobot(
        BicycleDriveSettings(wheelbase=0.85, max_steer=0.6, max_velocity=2.0, allow_backwards=True)
    )
    a = PlannerActionAdapter(robot, robot.action_space, 0.1)
    robot.apply_action(tuple(a.from_velocity_command((-0.5, 0.05))), 0.1)
    assert robot.state.velocity == pytest.approx(-0.1)
    assert robot.current_yaw_rate == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize(
    "command,expected",
    [
        ((0.25, 0.3), (0.25, 0.05)),
        ((0.25, -0.3), (0.25, -0.05)),
        ((0.0, 0.3), (0.1, 0.02)),
        ((0.0, -0.3), (0.1, -0.02)),
        ((0.0, 0.0), (0.0, 0.0)),
        ((-0.5, 0.3), (0.0, 0.0)),
        ((3.0, 0.6), (2.0, 0.4)),
    ],
)
def test_speed_priority_coupled_projection(command, expected):
    """Preserve speed and enforce curvature .2, including turn-only creep."""
    from robot_sf.planner.kinematics_model import BicycleDriveKinematicsModel

    model = BicycleDriveKinematicsModel(max_velocity=2.0, max_angular_speed=0.4)
    projected = model.project(command)
    assert projected == pytest.approx(expected)
    assert model.is_feasible(projected)


def test_creep_respects_low_speed_cap():
    """A .04m/s platform cannot be advanced by the .1m/s creep default."""
    from robot_sf.planner.kinematics_model import BicycleDriveKinematicsModel

    model = BicycleDriveKinematicsModel(max_velocity=0.04, max_angular_speed=0.008)
    assert model.project((0.0, 0.1)) == pytest.approx((0.04, 0.008))


def test_runner_model_uses_actual_bicycle_caps():
    """Planner projection must use the plant caps ahead of legacy planner limits."""
    from robot_sf.planner.kinematics_model import resolve_benchmark_kinematics_model

    model = resolve_benchmark_kinematics_model(
        robot_kinematics="bicycle_drive",
        command_limits={
            "v_max": 2.0,
            "omega_max": 1.0,
            "bicycle_max_velocity": 1.34,
            "bicycle_max_angular_speed": 0.4,
            "bicycle_max_curvature": 0.3,
        },
    )
    assert model.project((2.0, 1.0)) == pytest.approx((1.34, 0.4))
    assert model.project((0.2, 1.0)) == pytest.approx((0.2, 0.06))


@pytest.mark.parametrize("variant,steer", [("30deg", 0.52), ("45deg", 0.79)])
def test_opt_in_t60_config_reaches_real_robot(variant, steer):
    """Load tracked bytes through the scenario builder into physical settings."""
    from pathlib import Path

    import yaml

    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    root = Path(__file__).resolve().parents[1]
    selected = yaml.safe_load((root / f"configs/robots/t60_bicycle_{variant}_v1.yaml").read_text())
    config = build_robot_config_from_scenario(
        selected, scenario_path=root / f"configs/robots/t60_bicycle_{variant}_v1.yaml"
    )
    robot = config.robot_factory()
    assert isinstance(robot, BicycleDriveRobot)
    assert robot.config.radius == 0.64
    assert robot.config.wheelbase == 0.90
    assert robot.config.max_steer == steer
    assert robot.config.max_velocity == 1.34
    assert robot.config.max_accel == robot.config.max_decel == 1.0
    assert robot.config.min_velocity == 0.0
