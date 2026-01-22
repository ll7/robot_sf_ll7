"""Tests for SocNav observation bounds and clipping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sensor.socnav_observation import (
    SOCNAV_POSITION_CAP_M,
    SocNavObservationFusion,
    socnav_observation_space,
)


def _build_map_def(width: float, height: float) -> MapDefinition:
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


def test_socnav_observation_space_uses_global_cap() -> None:
    """SocNav observation space should be capped for mixed-map training."""
    map_def = _build_map_def(120.0, 80.0)
    env_config = RobotSimulationConfig()
    space = socnav_observation_space(map_def, env_config, max_pedestrians=4)

    robot_pos_high = space["robot"]["position"].high
    map_size_high = space["map"]["size"].high

    assert robot_pos_high[0] == pytest.approx(SOCNAV_POSITION_CAP_M)
    assert robot_pos_high[1] == pytest.approx(SOCNAV_POSITION_CAP_M)
    assert map_size_high[0] == pytest.approx(SOCNAV_POSITION_CAP_M)
    assert map_size_high[1] == pytest.approx(SOCNAV_POSITION_CAP_M)


def test_socnav_observation_clips_positions() -> None:
    """SocNav observations should clip positions to the global cap."""
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

    assert np.all(obs["robot"]["position"] <= SOCNAV_POSITION_CAP_M)
    assert np.all(obs["goal"]["current"] <= SOCNAV_POSITION_CAP_M)
    assert np.all(obs["goal"]["next"] <= SOCNAV_POSITION_CAP_M)
    assert np.all(obs["pedestrians"]["positions"] <= SOCNAV_POSITION_CAP_M)
    assert np.all(obs["map"]["size"] <= SOCNAV_POSITION_CAP_M)
