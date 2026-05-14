"""Regression tests for the dummy simulator backend contract.

These tests verify the minimal simulator surface that `RobotEnv` now relies on
for the backend-selection example. They matter because the example is CI-enabled
through the manifest smoke harness, so the dummy backend must keep exposing
route, pose, and obstacle metadata during reset and step.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.sim.backends import dummy_backend
from robot_sf.sim.backends.dummy_backend import DummySimulator, dummy_factory


def test_dummy_factory_exposes_robot_env_contract(test_map: MapDefinition) -> None:
    """Verify the factory returns the simulator fields RobotEnv accesses during smoke runs."""
    config = RobotSimulationConfig()
    config.backend = "dummy"
    config.seed = 123

    simulator = dummy_factory(config, test_map, False)

    assert simulator.map_def is test_map
    assert simulator.timestep == 0
    assert len(simulator.robots) == 1
    assert len(simulator.robot_navs) == 1
    assert simulator.goal_pos[0] == simulator.robot_navs[0].current_waypoint
    assert simulator.next_goal_pos[0] == simulator.robot_navs[0].next_waypoint
    assert simulator.robot_poses[0] == simulator.robots[0].pose
    assert simulator.robot_pos[0] == simulator.robots[0].pose[0]
    assert simulator.ped_pos.shape == (0, 2)
    assert simulator.ped_radii.shape == (0,)
    assert simulator.get_obstacle_lines().shape[1] == 4


def test_dummy_simulator_step_updates_pose_and_navigation(test_map: MapDefinition) -> None:
    """Verify stepping the dummy backend keeps pose, speed, and navigator state in sync."""
    simulator = DummySimulator(map_def=test_map, seed=7, step_dt=0.2, goal_proximity_threshold=1.0)

    initial_pose = simulator.robot_poses[0]
    parsed = simulator.robots[0].parse_action(np.array([0.4, -0.3], dtype=np.float32))

    simulator.step_once([parsed])

    assert parsed == pytest.approx((0.4, -0.3))
    assert simulator.timestep == 1
    assert simulator.robots[0].current_speed == pytest.approx(parsed)
    assert simulator.robot_poses[0] != initial_pose
    assert simulator.robot_navs[0].pos == simulator.robot_poses[0][0]


def test_dummy_simulator_reset_restores_spawn_and_clears_speed(test_map: MapDefinition) -> None:
    """Verify reset reinitializes the dummy robot state instead of keeping stale commands."""
    simulator = DummySimulator(map_def=test_map, seed=99, step_dt=0.1, goal_proximity_threshold=1.0)

    simulator.step_once([(0.8, 0.2)])
    stepped_pose = simulator.robot_poses[0]

    simulator.reset_state()

    assert simulator.timestep == 0
    assert simulator.robots[0].current_speed == (0.0, 0.0)
    assert simulator.robot_poses[0] != stepped_pose
    assert simulator.robot_navs[0].pos == simulator.robot_poses[0][0]


def test_dummy_simulator_uses_default_spawn_selection_when_resetting(
    test_map: MapDefinition, monkeypatch
) -> None:
    """Verify the dummy backend lets route sampling pick a valid spawn automatically."""
    seen_spawn_ids: list[int | None] = []

    def fake_sample_route(
        _map_def: MapDefinition, spawn_id: int | None
    ) -> list[tuple[float, float]]:
        seen_spawn_ids.append(spawn_id)
        return [(0.0, 0.0), (1.0, 0.0)]

    monkeypatch.setattr(dummy_backend, "sample_route", fake_sample_route)

    DummySimulator(map_def=test_map, seed=1, step_dt=0.1, goal_proximity_threshold=1.0)

    assert seen_spawn_ids == [None]
