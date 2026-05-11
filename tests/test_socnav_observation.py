"""Tests for SocNav observation bounds and clipping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import ObservationVisibilitySettings, RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sensor.socnav_observation import SocNavObservationFusion, socnav_observation_space


def _build_map_def(width: float, height: float) -> MapDefinition:
    """Build a map definition sized for SocNav observation bound tests."""
    obstacles = [Obstacle([(0, 0), (width, 0), (width, 1), (0, 1)])]
    robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
    ped_spawn_zones = [((3, 3), (4, 3), (4, 4))]
    robot_goal_zones = [((width - 2, height - 2), (width - 1, height - 2), (width - 1, height - 1))]
    bounds = [
        (0, width, 0, 0),
        (0, width, height, height),
        (0, 0, 0, height),
        (width, width, 0, height),
    ]
    ped_goal_zones = [((6, 6), (7, 6), (7, 7))]
    ped_crowded_zones: list = []
    robot_routes: list = []
    ped_routes: list = []
    single_pedestrians: list = []
    return MapDefinition(
        width,
        height,
        obstacles,
        robot_spawn_zones,
        ped_spawn_zones,
        robot_goal_zones,
        bounds,
        robot_routes,
        ped_goal_zones,
        ped_crowded_zones,
        ped_routes,
        single_pedestrians,
    )


def test_socnav_observation_space_uses_map_aware_cap() -> None:
    """SocNav observation space should preserve coordinates on maps larger than 50 m."""
    map_def = _build_map_def(120.0, 80.0)
    env_config = RobotSimulationConfig()
    space = socnav_observation_space(map_def, env_config, max_pedestrians=4)

    robot_pos_high = space["robot"]["position"].high
    map_size_high = space["map"]["size"].high

    assert robot_pos_high[0] == pytest.approx(120.0)
    assert robot_pos_high[1] == pytest.approx(80.0)
    assert map_size_high[0] == pytest.approx(120.0)
    assert map_size_high[1] == pytest.approx(80.0)


def test_socnav_observation_clips_positions_to_map_aware_cap() -> None:
    """SocNav observations should not clip valid large-map coordinates at 50 m."""
    env_config = RobotSimulationConfig()
    simulator = SimpleNamespace(
        ped_pos=np.array([[100.0, 100.0]], dtype=np.float32),
        ped_vel=np.array([[1.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((100.0, 100.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([120.0, 80.0], dtype=np.float32)],
        next_goal_pos=[np.array([70.0, 70.0], dtype=np.float32)],
        map_def=SimpleNamespace(width=120.0, height=80.0),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    fusion = SocNavObservationFusion(simulator=simulator, env_config=env_config, max_pedestrians=4)
    obs = fusion.next_obs()

    assert obs["robot"]["position"].tolist() == pytest.approx([100.0, 80.0])
    assert obs["goal"]["current"].tolist() == pytest.approx([120.0, 80.0])
    assert obs["goal"]["next"].tolist() == pytest.approx([70.0, 70.0])
    assert obs["pedestrians"]["positions"][0].tolist() == pytest.approx([100.0, 80.0])
    assert obs["map"]["size"].tolist() == pytest.approx([120.0, 80.0])


def test_socnav_observation_visibility_filters_fov_without_mutating_ground_truth() -> None:
    """Planner-facing pedestrian observations should honor opt-in FOV settings only."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    obs = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    ).next_obs()

    assert obs["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert obs["pedestrians"]["positions"][0].tolist() == pytest.approx([3.0, 0.0])
    assert simulator.ped_pos.shape == (2, 2)


def test_socnav_observation_visibility_filters_static_occluded_pedestrian() -> None:
    """Static obstacle geometry should hide pedestrians behind an occluding polygon."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        static_occlusion=True,
    )
    obstacle = Obstacle([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[obstacle]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    obs = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    ).next_obs()

    assert obs["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert obs["pedestrians"]["positions"][0].tolist() == pytest.approx([0.0, 3.0])
    np.testing.assert_allclose(
        simulator.ped_pos,
        np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    )
