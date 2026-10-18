"""Actual-start reaction clearance and route fallback rectangle regressions."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.ped_npc.ped_behavior import CrowdedZoneBehavior
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation
from robot_sf.training.scenario_loader import _route_zone_from_map, load_scenarios

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")


def test_zoned_crowd_spawns_and_later_goals_respect_reserved_robot_zones() -> None:
    """Zoned crowds retain the synthesized path's robot start and goal exclusions."""
    np.random.seed(1001)
    reserved = [((3.0, 3.0), (4.0, 3.0), (4.0, 4.0)), ((7.0, 7.0), (8.0, 7.0), (8.0, 8.0))]
    states, _, behaviors = populate_simulation(
        0.5,
        PedSpawnConfig(peds_per_area_m2=0.5, max_group_members=1),
        [],
        [((0.0, 0.0), (12.0, 0.0), (12.0, 12.0))],
        reserved_zones=reserved,
        ped_radius=0.3,
        reserved_zone_radius=1.15,
    )
    crowd = next(b for b in behaviors if isinstance(b, CrowdedZoneBehavior))
    points = [
        *states.ped_positions,
        *(crowd._sample_goal(crowd.crowded_zones[0]) for _ in range(200)),
    ]
    points = np.asarray(points)
    # Hand-derived distance to each axis-aligned reserved square, plus 0.3 + 1.15 m.
    for low in (3.0, 7.0):
        distance = np.linalg.norm(
            np.maximum(np.maximum(low - points, np.asarray(points) - low - 1), 0), axis=1
        )
        assert distance.min() >= 1.45 - 0.002  # Shapely buffer's polygon approximation.


@pytest.mark.parametrize("name", ["francis2023_robot_crowding", "francis2023_circular_crossing"])
@pytest.mark.parametrize("seed", range(1001, 1031))
def test_actual_robot_start_has_reaction_clearance(name: str, seed: int) -> None:
    """Real release maps start clear and survive ten stationary steps on dev seeds."""
    scenario = next(row for row in load_scenarios(MATRIX) if row["name"] == name)
    config = build_env_config(
        _scenario_with_episode_seed_defaults(scenario, seed=seed), scenario_path=MATRIX
    )
    env = make_robot_env(config=config, seed=seed, debug=False)
    try:
        env.reset(seed=seed)
        sim = env.simulator
        # One second at the population speed cap, plus the existing 0.1 m margin.
        buffer = 0.1 + max(
            np.max(sim.pysf_sim.peds.max_speeds), np.linalg.norm(sim.ped_vel, axis=1).max()
        )
        distances = np.linalg.norm(sim.ped_pos - np.asarray(sim.robot_pos[0]), axis=1)
        radii = float(sim.robots[0].config.radius) + float(sim.config.ped_radius)
        assert distances.min() >= radii + buffer - 1e-6, (name, seed, distances.min(), buffer)
        for step in range(10):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            meta = info.get("meta", info)
            assert not meta.get("is_pedestrian_collision"), (name, seed, step + 1)
            assert not terminated and not truncated, (name, seed, step + 1)
    finally:
        env.close()


@pytest.mark.parametrize("is_robot", [True, False])
def test_fallback_route_zones_have_the_right_angle_at_b(is_robot: bool) -> None:
    """Missing zone IDs produce the intended axis-aligned 0.1 m square."""
    empty_map = SimpleNamespace(
        robot_spawn_zones=[],
        robot_goal_zones=[],
        robot_routes=[],
        ped_spawn_zones=[],
        ped_goal_zones=[],
        ped_routes=[],
    )
    spawn, goal = _route_zone_from_map(
        empty_map, is_robot=is_robot, spawn_id=7, goal_id=8, waypoints=[(2.0, 3.0), (5.0, 6.0)]
    )
    assert spawn == ((2.0, 3.0), (2.1, 3.0), (2.1, 3.1))
    assert goal == ((5.0, 6.0), (5.1, 6.0), (5.1, 6.1))
