"""Seed-free regression cases for static clearance in the revised sampler."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.velocity_sampler import VelocitySamplerWeights
from robot_sf.planner.velocity_sampler_r2 import (
    StaticSafeVelocitySamplerAdapter,
    swept_wall_clearance,
)
from robot_sf.robot.differential_drive import DifferentialDriveSettings


def test_wall_between_rollout_samples_is_detected() -> None:
    """A short wall crossed between trajectory samples cannot disappear."""
    points = np.array([[[0.0, 0.0], [4.0, 0.0]]])
    wall = np.array([[1.0, -0.1, 1.0, 0.1]])
    assert swept_wall_clearance(points, wall, 0.2)[0, 0] == pytest.approx(-0.2)
    assert swept_wall_clearance(points, np.empty((0, 4)), 0.2)[0, 0] == float("inf")


def test_wall_in_braking_range_requires_braking() -> None:
    """Reject acceleration when it leaves no wall-safe braking continuation."""
    config = SocNavPlannerConfig(
        socnav_sampling_version="bounded_v2", sampling_pedestrian_prediction=True
    )
    adapter = StaticSafeVelocitySamplerAdapter(
        config=config, weights=VelocitySamplerWeights(horizon_s=1.5, obstacle_weight=2.0)
    )
    wall = [((1.6, -5.0), (1.6, 5.0))]
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: wall),
        )
    )
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [1.0, 0.0], "radius": [1.0]},
        "goal": {"current": [10.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.4]},
        "sim": {"timestep": [0.1]},
    }
    speed, _ = adapter.plan(obs)
    assert speed == pytest.approx(0.9)


def _multistage_adapter():
    from robot_sf.planner.velocity_sampler_r2 import (
        MultiStageVelocitySamplerAdapter,
        MultiStageWeights,
    )

    config = SocNavPlannerConfig(
        socnav_sampling_version="bounded_v2", sampling_pedestrian_prediction=True
    )
    adapter = MultiStageVelocitySamplerAdapter(
        config=config, weights=MultiStageWeights(horizon_s=2.0, stage_s=0.3)
    )
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: []),
        )
    )
    return adapter


def test_yield_continuation_counts_down_and_departure_is_not_restarted() -> None:
    """A previously selected three-step yield transitions to its go stage."""
    adapter = _multistage_adapter()
    # A previously chosen profile: stop for three control steps, then drive.
    adapter._active_profile = ((0.0, 0.0, 2.0, 0.0, 3, 0, 2), 6)
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0, 0.0], "radius": [1.0]},
        "goal": {"current": [10.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.4]},
        "sim": {"timestep": [0.1]},
    }
    outputs = [adapter.plan(obs)[0] for _ in range(4)]
    assert outputs[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert outputs[3] == pytest.approx(0.1)
    assert adapter.diagnostics()["continued_steps"] == 4


def test_passing_side_is_held_until_safety_rules_it_out() -> None:
    """A cheaper opposite side cannot displace the committed safe side."""
    adapter = _multistage_adapter()
    adapter._locked_side = 1
    costs = np.array([1.0, 0.0])
    points = np.array([[[0.0, 0.0], [2.0, 1.0]], [[0.0, 0.0], [2.0, -1.0]]])
    speeds = np.array([0.1, 0.1])
    brake_clear = np.ones((2, 1))
    ped_clear = np.ones((2, 1))
    times = np.array([0.0, 1.0])
    peds = np.array([[2.0, 0.0]])
    velocities = np.zeros((1, 2))
    direction = np.array([1.0, 0.0])
    adapter._continuation_index = None
    args = (speeds, brake_clear, ped_clear, points, times, peds, velocities, direction)
    assert adapter._select_candidate(costs, np.array([True, True]), *args) == 0
    assert adapter._select_candidate(costs, np.array([False, True]), *args) == 1


def test_gap_with_grid_keeps_a_safe_braking_path() -> None:
    """The footprint does not commit to contact in a diameter-width gap."""
    from robot_sf.planner.velocity_sampler_r2 import _braking_forecast

    config = SocNavPlannerConfig(
        socnav_sampling_version="bounded_v2",
        sampling_pedestrian_prediction=True,
        sampling_path_distance=False,
    )
    weights = VelocitySamplerWeights(horizon_s=1.5, obstacle_weight=2.0, clearance_scale=0.3)
    drive = DifferentialDriveSettings()
    walls = [((1.3, -6.0), (1.3, -1.0)), ((1.3, 1.0), (1.3, 6.0))]
    adapter = StaticSafeVelocitySamplerAdapter(config=config, weights=weights)
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=drive),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: walls),
        )
    )
    grid = np.zeros((4, 120, 120))
    grid[0, :50, 53] = 1.0
    grid[0, 70:, 53] = 1.0
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [1.5, 0.0], "radius": [1.0]},
        "goal": {"current": [9.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.4]},
        "sim": {"timestep": [0.1]},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": [-4.0, -6.0],
        "occupancy_grid_meta_resolution": [0.1],
        "occupancy_grid_meta_size": [12.0, 12.0],
        "occupancy_grid_meta_use_ego_frame": [0.0],
        "occupancy_grid_meta_channel_indices": [0, 1, 2, 3],
    }
    speed, turn = adapter.plan(obs)
    path = _braking_forecast(
        np.array([0.0, 0.0]), 0.0, 1.5, 0.0, np.array([speed]), np.array([turn]), 0.1, drive
    )
    assert adapter.diagnostics()["static_grid_steps"] == 1
    assert swept_wall_clearance(path, np.asarray(walls).reshape(-1, 4), 1.0).min() > 1e-5


def test_multistage_clear_route_keeps_accelerating() -> None:
    """A straight, open route has a first stage at the drive's full speed."""
    adapter = _multistage_adapter()
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [1.2, 0.0], "radius": [1.0]},
        "goal": {"current": [30.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.4]},
        "sim": {"timestep": [0.1]},
    }
    speed, turn = adapter.plan(obs)
    assert speed == pytest.approx(1.3)
    assert turn == pytest.approx(0.0)


def test_multistage_rollout_matches_drive_during_brake_turn_and_go() -> None:
    """Both stages obey the plant's acceleration and wheel odometry."""
    from robot_sf.robot.differential_drive import DifferentialDriveRobot

    adapter = _multistage_adapter()
    drive = DifferentialDriveSettings()
    position = np.array([2.0, 3.0])
    speeds = np.array([1.2])
    turns = np.array([0.2])
    _, _, points, headings, times = adapter._candidate_forecasts(
        position,
        0.4,
        1.2,
        0.2,
        speeds,
        turns,
        0.1,
        drive,
        np.array([1.0, 0.0]),
        np.empty((0, 2)),
        np.empty((0, 2)),
    )
    index = next(
        i
        for i, profile in enumerate(adapter._profiles)
        if profile[:4] == (0.0, 1.0, 2.0, 0.0) and profile[5:] == (0, 1)
    )
    stage = adapter._profiles[index][4]
    robot = DifferentialDriveRobot(drive)
    robot.state.pose = ((2.0, 3.0), 0.4)
    robot.state.velocity = (1.2, 0.2)
    robot.state.wheel_speeds = robot.movement._resulting_wheel_speeds(robot.current_speed)
    for i in range(len(times) - 1):
        desired = np.array([0.0, 1.0]) if i < stage else np.array([2.0, 0.0])
        robot.apply_action(tuple((desired - robot.current_speed) / 0.1), 0.1)
        assert points[index, i + 1] == pytest.approx(robot.pos, abs=1e-10)
        assert headings[index, i + 1] == pytest.approx(robot.pose[1], abs=1e-10)
