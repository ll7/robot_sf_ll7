"""Versioned robot goal sampling guards for issue #9859."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.nav.map_config import (
    ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
    ROBOT_GOAL_SAMPLING_LEGACY_V1,
    normalize_robot_goal_sampling_policy,
)
from robot_sf.nav.navigation import sample_route
from robot_sf.nav.spawn_clearance import SPAWN_CLEARANCE_MARGIN_M, robot_obstacle_clearance
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.backends.dummy_backend import DummySimulator
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[2]
T_INTERSECTION = (
    ROOT / "maps/successor_svg_maps/issue_9762_classic_t_intersection_goal_zone_entry_v2.svg"
)
RADIUS = 1.0


@pytest.fixture(scope="module")
def t_intersection_map():
    """Load a versioned map with a partially unsafe robot goal zone."""
    result = convert_map(str(T_INTERSECTION))
    assert result is not None
    return result


def test_policy_names_are_versioned_and_unknown_values_fail_closed() -> None:
    """Absent policy retains historical sampling; unknown names cannot silently fall back."""
    assert normalize_robot_goal_sampling_policy(None) == ROBOT_GOAL_SAMPLING_LEGACY_V1
    assert (
        normalize_robot_goal_sampling_policy(ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1)
        == ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1
    )
    with pytest.raises(ValueError, match="robot_goal_sampling_policy"):
        normalize_robot_goal_sampling_policy("footprint_clearance_v9")


def test_opt_in_replaces_only_unsafe_final_goal(t_intersection_map) -> None:
    """The recorded defect draw is rejected while spawn and route prefix stay identical."""
    np.random.seed(1009)
    legacy = sample_route(t_intersection_map, 0, robot_radius=RADIUS)
    assert robot_obstacle_clearance(t_intersection_map, legacy[-1], RADIUS) < 0.1

    np.random.seed(1009)
    corrected = sample_route(
        t_intersection_map,
        0,
        robot_radius=RADIUS,
        robot_goal_sampling_policy=ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
    )
    assert corrected[:-1] == legacy[:-1]
    assert corrected[-1] != legacy[-1]
    assert robot_obstacle_clearance(t_intersection_map, corrected[-1], RADIUS) >= (
        SPAWN_CLEARANCE_MARGIN_M - 1e-9
    )
    assert corrected.goal_zone == legacy.goal_zone


def test_opt_in_preserves_already_safe_goal_draw(t_intersection_map) -> None:
    """A candidate valid under both policies stays bit-identical."""
    np.random.seed(1001)
    legacy = sample_route(t_intersection_map, 0, robot_radius=RADIUS)
    np.random.seed(1001)
    corrected = sample_route(
        t_intersection_map,
        0,
        robot_radius=RADIUS,
        robot_goal_sampling_policy=ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
    )
    assert corrected == legacy


def test_opt_in_guards_planner_route_goal(t_intersection_map) -> None:
    """Planner-generated routes receive the same safe final target as fallback routes."""

    class StraightPlanner:
        def plan(self, start, goal):
            return [start, goal]

    map_with_planner = deepcopy(t_intersection_map)
    map_with_planner._use_planner = True
    map_with_planner._global_planner = StraightPlanner()
    np.random.seed(1009)
    route = sample_route(
        map_with_planner,
        0,
        robot_radius=RADIUS,
        robot_goal_sampling_policy=ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
    )
    assert robot_obstacle_clearance(map_with_planner, route[-1], RADIUS) >= (
        SPAWN_CLEARANCE_MARGIN_M - 1e-9
    )


def test_dummy_backend_propagates_opt_in_goal_policy(t_intersection_map) -> None:
    """The smoke backend uses the selected policy when a footprint radius is supplied."""
    np.random.seed(1009)
    simulator = DummySimulator(
        map_def=t_intersection_map,
        robot_goal_sampling_policy=ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
        robot_radius=RADIUS,
    )
    target = simulator.robot_navs[0].waypoints[-1]
    assert robot_obstacle_clearance(t_intersection_map, target, RADIUS) >= (
        SPAWN_CLEARANCE_MARGIN_M - 1e-9
    )


def test_opt_in_requires_radius_and_rejects_goal_zone_without_safe_point(
    t_intersection_map,
) -> None:
    """An impossible corrected goal fails rather than reverting to a centre-only draw."""
    with pytest.raises(ValueError, match="robot_radius"):
        sample_route(
            t_intersection_map,
            0,
            robot_goal_sampling_policy=ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
        )

    map_without_safe_goal = deepcopy(t_intersection_map)
    map_without_safe_goal.robot_routes[0].goal_zone = (
        (1.1, 10.0),
        (1.4, 10.0),
        (1.4, 10.3),
    )
    np.random.seed(1009)
    with pytest.raises(RuntimeError, match="No robot goal with wall clearance"):
        sample_route(
            map_without_safe_goal,
            0,
            robot_radius=RADIUS,
            robot_goal_sampling_policy=ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
        )


def test_scenario_override_selects_versioned_goal_sampling_policy(tmp_path: Path) -> None:
    """The scenario config carries opt-in semantics into the simulation config."""
    config = build_robot_config_from_scenario(
        {
            "name": "goal-clearance-runtime-smoke",
            "simulation_config": {
                "robot_goal_sampling_policy": ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1,
            },
        },
        scenario_path=tmp_path / "scenario.yaml",
    )
    assert (
        config.sim_config.robot_goal_sampling_policy == ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1
    )
    with pytest.raises(ValueError, match="robot_goal_sampling_policy"):
        build_robot_config_from_scenario(
            {
                "name": "unknown-goal-sampling-runtime-smoke",
                "simulation_config": {"robot_goal_sampling_policy": "footprint_clearance_v9"},
            },
            scenario_path=tmp_path / "scenario.yaml",
        )


def test_map_runner_reset_uses_explicit_goal_sampling_policy() -> None:
    """The benchmark config path passes the opt-in policy to the real simulator reset."""
    matrix = ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3.yaml"
    scenario = next(
        row for row in load_scenarios(matrix) if row["name"] == "classic_t_intersection_low"
    )
    scenario = dict(scenario)
    simulation_config = dict(scenario.get("simulation_config", {}))
    simulation_config["robot_goal_sampling_policy"] = ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1
    scenario["simulation_config"] = simulation_config
    seed = 1001
    scenario = _scenario_with_episode_seed_defaults(scenario, seed=seed)
    env = make_robot_env(config=build_env_config(scenario, scenario_path=matrix), seed=seed)
    try:
        env.reset(seed=seed)
        simulator = env.simulator
        assert simulator.robot_goal_sampling_policy == ROBOT_GOAL_SAMPLING_FOOTPRINT_CLEARANCE_V1
        target = simulator.robot_navs[0].waypoints[-1]
        radius = float(simulator.robots[0].config.radius)
        assert robot_obstacle_clearance(simulator.map_def, target, radius) >= (
            SPAWN_CLEARANCE_MARGIN_M - 1e-9
        )
    finally:
        env.close()
