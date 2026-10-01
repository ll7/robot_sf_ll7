"""Goal-area crowd density and simulator route-respawn reaction regressions."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.ped_npc.ped_behavior import CrowdedZoneBehavior, FollowRouteBehavior
from robot_sf.training.scenario_loader import load_scenarios

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")


def test_robot_crowding_spawns_and_goals_can_occupy_robot_goal_zone() -> None:
    """The real crowding map reserves the robot start, while keeping its goal populated."""
    scenario = next(
        row for row in load_scenarios(MATRIX) if row["name"] == "francis2023_robot_crowding"
    )
    hits = {"spawn": 0, "initial_goal": 0, "later_goal": 0}
    for seed in range(1001, 1031):
        config = build_env_config(
            _scenario_with_episode_seed_defaults(scenario, seed=seed), scenario_path=MATRIX
        )
        env = make_robot_env(config=config, seed=seed, debug=False)
        try:
            env.reset(seed=seed)
            sim = env.simulator
            # Zone's authored B corner joins two perpendicular rectangle edges.
            a, b, c = np.asarray(sim.map_def.robot_goal_zones[0])
            goal_zone = Polygon([a, b, c, a + c - b])
            crowd = next(b for b in sim.peds_behaviors if isinstance(b, CrowdedZoneBehavior))
            samples = {
                "spawn": sim.pysf_state.ped_positions,
                "initial_goal": sim.pysf_state.pysf_states()[:, 4:6],
                "later_goal": [crowd._sample_goal(crowd.crowded_zones[0]) for _ in range(100)],
            }
            for kind, points in samples.items():
                hits[kind] += sum(goal_zone.contains(Point(point)) for point in points)
        finally:
            env.close()
    # Many deterministic draws distinguish a permitted goal rectangle from an excluded island.
    assert all(count > 0 for count in hits.values()), hits


@pytest.mark.parametrize("desired_speed,buffer", [(None, 0.75), (1.1, 1.2)])
def test_route_respawns_keep_the_actual_reset_reaction_buffer(
    desired_speed: float | None, buffer: float
) -> None:
    """Simulator wiring gives route-end respawns a full second at the walking cap."""
    scenario = next(
        row for row in load_scenarios(MATRIX) if row["name"] == "francis2023_circular_crossing"
    )
    config = build_env_config(
        _scenario_with_episode_seed_defaults(scenario, seed=1001), scenario_path=MATRIX
    )
    config.sim_config.desired_speed_mean = desired_speed
    config.sim_config.desired_speed_std = 0.0
    config.sim_config.desired_speed_seed = 1001
    env = make_robot_env(config=config, seed=1001, debug=False)
    try:
        env.reset(seed=1001)
        sim = env.simulator
        routes = [b for b in sim.peds_behaviors if isinstance(b, FollowRouteBehavior)]
        assert routes
        # Release-map walkers start at 0.5 m/s with a 1.3 speed cap multiplier:
        # 0.1 m margin + 1 s * 0.65 m/s = 0.75 m surface clearance.
        # An explicit 1.1 m/s walking cap instead requires 1.2 m after reset.
        expected_radius = float(sim.robots[0].config.radius) + float(sim.config.ped_radius) + buffer
        for behavior in routes:
            assert np.isclose(behavior.robot_exclusion_radius, expected_radius)
            # The guard must follow the live robot, including movement after reset.
            sim.robots[0].reset_state(((10.0, 10.0), 0.0))
            assert behavior.robot_pose_provider() == sim.robot_poses
    finally:
        env.close()
