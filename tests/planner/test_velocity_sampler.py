"""Focused behavior checks for the experimental velocity sampler."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.algorithm_contract import CONTRACT_RECORDS_BY_NAME
from robot_sf.benchmark.algorithm_readiness import require_algorithm_allowed
from robot_sf.benchmark.map_runner.map_runner import _build_policy
from robot_sf.planner.clearance_geometry import swept_pedestrian_clearance
from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.velocity_sampler import (
    VelocitySamplerAdapter,
    VelocitySamplerWeights,
    _forecast,
    reachable_commands,
)
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


def test_window_commands_match_drive_one_step() -> None:
    """Every sampled target is reachable by the physical drive in one step."""
    drive = DifferentialDriveSettings(
        max_linear_accel=0.8, max_linear_decel=1.2, max_angular_accel=0.7
    )
    speeds, turns = reachable_commands(
        1.0,
        0.3,
        0.1,
        drive,
        speed_cap=2.0,
        turn_cap=1.0,
        speed_samples=3,
        turn_samples=9,
    )
    assert (min(speeds), max(speeds)) == pytest.approx((0.88, 1.08))
    assert (min(turns), max(turns)) == pytest.approx((0.23, 0.37))
    for speed, turn in zip(speeds, turns, strict=True):
        robot = DifferentialDriveRobot(drive)
        robot.state.velocity = (1.0, 0.3)
        robot.apply_action(((speed - 1.0) / 0.1, (turn - 0.3) / 0.1), 0.1)
        assert robot.current_speed == pytest.approx((speed, turn))


def test_first_forecast_step_matches_drive_odometry() -> None:
    """The predicted first step agrees with the plant's wheel odometry."""
    drive = DifferentialDriveSettings()
    robot = DifferentialDriveRobot(drive)
    robot.state.pose = ((2.0, 3.0), 0.4)
    robot.state.velocity = (1.0, 0.2)
    robot.state.wheel_speeds = robot.movement._resulting_wheel_speeds(robot.current_speed)
    robot.apply_action((1.0, 1.0), 0.1)
    positions, headings, times = _forecast(
        np.array([2.0, 3.0]),
        0.4,
        1.0,
        0.2,
        np.array([1.1]),
        np.array([0.3]),
        0.1,
        2.4,
        0.2,
    )
    assert times[1] == pytest.approx(0.1)
    assert positions[0, 1] == pytest.approx(robot.pos)
    assert headings[0, 1] == pytest.approx(robot.pose[1])


def test_swept_moving_pedestrian_crosses_between_samples() -> None:
    """The relative sweep catches a crossing missed by both endpoint distances."""
    points = np.array([[[0.0, 0.0], [2.0, 0.0]]])
    actual = swept_pedestrian_clearance(
        points, np.array([0.0, 1.0]), np.array([[1.0, 1.0]]), np.array([[0.0, -2.0]]), 0.5
    )
    stationary = swept_pedestrian_clearance(
        points, np.array([0.0, 1.0]), np.array([[1.0, 1.0]]), np.zeros((1, 2)), 0.5
    )
    assert actual[0, 0] == pytest.approx(-0.5)
    assert stationary[0, 0] == pytest.approx(0.5)


def test_blocked_candidates_still_return_reachable_command() -> None:
    """A pedestrian overlap is scored softly and cannot force an unreachable stop."""
    config = SocNavPlannerConfig(
        socnav_sampling_version="bounded_v2", sampling_pedestrian_prediction=True
    )
    adapter = VelocitySamplerAdapter(config=config, weights=VelocitySamplerWeights())
    drive = DifferentialDriveSettings()
    adapter.bind_env(
        SimpleNamespace(env_config=SimpleNamespace(robot_config=drive), simulator=SimpleNamespace())
    )
    obs = {
        "robot": {"position": [2.0, 2.0], "heading": [0.0], "speed": [1.0, 0.0], "radius": [1.0]},
        "goal": {"current": [9.0, 2.0]},
        "pedestrians": {
            "positions": [[2.0, 2.0]],
            "velocities": [[0.0, 0.0]],
            "count": [1],
            "radius": [0.4],
        },
        "sim": {"timestep": [0.1]},
    }
    speed, turn = adapter.plan(obs)
    assert 0.9 - 1e-8 <= speed <= 1.1 + 1e-8
    assert -0.1 - 1e-8 <= turn <= 0.1 + 1e-8
    assert adapter.diagnostics()["fallback_triggered"] is False
    del obs["pedestrians"]["velocities"]
    with pytest.raises(ValueError, match="velocity observations are required"):
        adapter.plan(obs)


def test_pedestrian_velocity_changes_the_selected_command() -> None:
    """A moving pedestrian changes the decision at identical observed positions."""
    config = SocNavPlannerConfig(
        socnav_sampling_version="bounded_v2", sampling_pedestrian_prediction=True
    )
    weights = VelocitySamplerWeights(
        horizon_s=1.8, pedestrian_weight=4.0, obstacle_weight=2.0, clearance_scale=0.3
    )
    adapter = VelocitySamplerAdapter(config=config, weights=weights)
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(),
        )
    )
    obs = {
        "robot": {"position": [2.0, 2.0], "heading": [0.0], "speed": [1.0, 0.0], "radius": [1.0]},
        "goal": {"current": [10.0, 2.0]},
        "pedestrians": {
            "positions": [[4.0, 3.0]],
            "velocities": [[0.0, 0.0]],
            "count": [1],
            "radius": [0.4],
        },
        "sim": {"timestep": [0.1]},
    }
    stationary = adapter.plan(obs)
    adapter.reset()
    obs["pedestrians"]["velocities"] = [[0.0, -1.0]]
    moving = adapter.plan(obs)
    assert stationary[1] < 0.0 < moving[1]

    committed = VelocitySamplerAdapter(
        config=config,
        weights=VelocitySamplerWeights(
            horizon_s=1.8,
            pedestrian_weight=4.0,
            obstacle_weight=2.0,
            clearance_scale=0.3,
            switch_weight=1.0,
        ),
    )
    committed.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(),
        )
    )
    obs["pedestrians"]["velocities"] = [[0.0, 0.0]]
    assert committed.plan(obs)[1] < 0.0
    obs["pedestrians"]["velocities"] = [[0.0, -1.0]]
    assert committed.plan(obs)[1] <= 0.0
    committed.reset()
    assert committed.plan(obs)[1] > 0.0


def test_registered_planner_has_its_own_experimental_identity() -> None:
    """Dispatch selects the new adapter under an explicit experimental profile."""
    from robot_sf.planner.velocity_sampler_r9 import ContactVelocitySamplerAdapter

    record = CONTRACT_RECORDS_BY_NAME["velocity_sampler"]
    assert record.requires_explicit_opt_in and not record.paper_baseline_eligible
    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="velocity_sampler", benchmark_profile="experimental", ppo_paper_ready=False
        )
    with pytest.raises(ValueError, match="blocked by profile 'baseline-safe'"):
        require_algorithm_allowed(
            algo="velocity_sampler",
            benchmark_profile="baseline-safe",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )
    readiness = require_algorithm_allowed(
        algo="velocity_sampler",
        benchmark_profile="experimental",
        ppo_paper_ready=False,
        allow_testing_algorithms=True,
    )
    assert readiness is not None and readiness.canonical_name == "velocity_sampler"
    policy, _ = _build_policy(
        "velocity_sampler",
        {
            "allow_testing_algorithms": True,
            "socnav_sampling_version": "bounded_v2",
            "sampling_pedestrian_prediction": True,
        },
        robot_kinematics="differential_drive",
    )
    assert type(vars(policy)["_planner_adapter"]) is ContactVelocitySamplerAdapter
    baseline, _ = _build_policy(
        "socnav_sampling",
        {"socnav_sampling_version": "bounded_v2"},
        robot_kinematics="differential_drive",
    )
    assert type(vars(baseline)["_planner_adapter"]) is not VelocitySamplerAdapter
