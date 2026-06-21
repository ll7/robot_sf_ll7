"""Regression tests for simulator robot/action count validation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.common.types import Line2D, Rect
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig, RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import PedSimulator, init_ped_simulators, init_simulators


def _minimal_map() -> MapDefinition:
    """Build a compact map with one deterministic robot route."""
    width = 10.0
    height = 10.0
    spawn_zone: Rect = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    goal_zone: Rect = ((8.0, 8.0), (9.0, 8.0), (8.0, 9.0))
    bounds: list[Line2D] = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.2, 1.2), (8.8, 8.8)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


@pytest.mark.parametrize("actions", [[], [(0.0, 0.0), (0.1, 0.0)]])
def test_simulator_step_once_rejects_mismatched_robot_action_count(actions):
    """Simulator should fail loudly when action count does not match robots."""
    map_def = _minimal_map()
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0]),
    )
    simulator = init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    with pytest.raises(ValueError, match="expected 1 robot action"):
        simulator.step_once(actions)


@pytest.mark.parametrize("actions", [[], [(0.0, 0.0), (0.1, 0.0)]])
def test_ped_simulator_step_once_rejects_mismatched_robot_action_count(actions):
    """PedSimulator should share the same robot/action count contract."""
    map_def = _minimal_map()
    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0]),
    )
    simulator = init_ped_simulators(
        config,
        map_def,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    with pytest.raises(ValueError, match="expected 1 robot action"):
        simulator.step_once(actions, ego_ped_actions=[(0.0, 0.0)])


@pytest.mark.parametrize("ego_ped_actions", [[], [(0.0, 0.0), (0.1, 0.0)]])
def test_ped_simulator_step_once_rejects_mismatched_ego_ped_action_count(ego_ped_actions):
    """PedSimulator should fail loudly when ego-ped action count is not exactly one."""
    map_def = _minimal_map()
    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0]),
    )
    simulator = init_ped_simulators(
        config,
        map_def,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    with pytest.raises(ValueError, match="expected 1 ego pedestrian action"):
        simulator.step_once([(0.0, 0.0)], ego_ped_actions=ego_ped_actions)


def test_simulator_step_once_forwards_cached_group_lists(monkeypatch) -> None:
    """Simulator.step_once should pass the cached group-list snapshot to PySF stepping."""
    map_def = _minimal_map()
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0]),
    )
    simulator = init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    captured: dict[str, list[list[int]] | None] = {"groups": None}
    initial_group_lists = simulator.groups.groups_as_lists

    def _capture_group_lists(_ped_forces, groups: list[list[int]]) -> None:
        captured["groups"] = groups

    monkeypatch.setattr(simulator.pysf_sim.peds, "step", _capture_group_lists)
    simulator.step_once([(0.0, 0.0)])

    assert captured["groups"] is initial_group_lists


def test_ped_simulator_step_once_forwards_cached_group_lists(monkeypatch) -> None:
    """PedSimulator.step_once should pass the cached group-list snapshot to PySF stepping."""
    map_def = _minimal_map()
    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0]),
    )
    simulator = init_ped_simulators(
        config,
        map_def,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    captured: dict[str, list[list[int]] | None] = {"groups": None}
    initial_group_lists = simulator.groups.groups_as_lists

    def _capture_group_lists(_ped_forces, groups: list[list[int]]) -> None:
        captured["groups"] = groups

    monkeypatch.setattr(simulator.pysf_sim.peds, "step", _capture_group_lists)
    simulator.step_once([(0.0, 0.0)], ego_ped_actions=[(0.0, 0.0)])

    assert captured["groups"] is initial_group_lists


def test_ped_simulator_ped_and_ego_pos_returns_pysf_position_view() -> None:
    """Combined pedestrian positions should reuse the PySF state position view."""
    positions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    simulator = PedSimulator.__new__(PedSimulator)
    simulator.pysf_state = SimpleNamespace(ped_positions=positions)

    assert simulator.ped_and_ego_pos is positions


def test_ped_simulator_ego_ped_next_goal_tracks_robot_current_waypoint() -> None:
    """Ego-pedestrian next-goal target should follow robot route progress."""
    simulator = PedSimulator.__new__(PedSimulator)
    navigator = RouteNavigator(waypoints=[(2.0, 1.0), (4.0, 3.0)])
    simulator.robot_navs = [navigator]

    assert simulator.ego_ped_next_goal_pos == (2.0, 1.0)

    navigator.waypoint_id = 1

    assert simulator.ego_ped_next_goal_pos == (4.0, 3.0)
