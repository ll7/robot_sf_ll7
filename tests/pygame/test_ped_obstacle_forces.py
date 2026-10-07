"""
Visually test the Pedestrian and Obstacle forces.
"""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger
from pysocialforce.forces import ObstacleForce

from robot_sf.gym_env import robot_env_with_pedestrian_obstacle_forces as obstacle_env
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import global_reset_seed
from robot_sf.gym_env.robot_env_with_pedestrian_obstacle_forces import (
    RobotEnvWithPedestrianObstacleForces,
)
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.sim_config import SimulationSettings


def test_pedestrian_obstacle_avoidance(monkeypatch: pytest.MonkeyPatch):
    """Run deterministic rendering/stepping with active pedestrian obstacle repulsion."""
    seed = 1001
    # Population sampling starts in the constructor, before reset can seed it.
    monkeypatch.setattr(
        obstacle_env,
        "EnvSettings",
        partial(EnvSettings, sim_config=SimulationSettings(pedestrian_seed=seed)),
    )
    logger.info("Testing Pedestrian and Obstacle forces")
    map_def = convert_map("maps/svg_maps/example_map_with_obstacles.svg")
    logger.debug(f"type map_def: {type(map_def)}")
    with global_reset_seed(seed):
        env = RobotEnvWithPedestrianObstacleForces(map_def=map_def, debug=True)
        try:
            logger.info("created environment")
            # Keep real rendering, but don't pace a CI smoke test at display FPS.
            env.sim_ui.clock = SimpleNamespace(tick=lambda _fps: None)
            env.action_space.seed(seed)
            env.reset(seed=seed)
            saw_obstacle_repulsion = False
            for _ in range(1000):
                forces = [
                    force
                    for force in env.simulator.pysf_sim.forces
                    if isinstance(force, ObstacleForce)
                ]
                assert forces, "Pedestrian obstacle forces must be enabled"
                for force in forces:
                    repulsion = force()
                    assert np.isfinite(repulsion).all()
                    saw_obstacle_repulsion |= bool(np.any(repulsion != 0.0))
                rand_action = env.action_space.sample()
                _, _, done, _, _ = env.step(rand_action)
                env.render()
                if done:
                    env.reset(seed=seed)
            assert saw_obstacle_repulsion, "Obstacles must exert nonzero pedestrian repulsion"
        finally:
            env.close()


if __name__ == "__main__":
    logger.info("Testing Pedestrian and Obstacle forces")
    with pytest.MonkeyPatch.context() as patch:
        test_pedestrian_obstacle_avoidance(patch)
    logger.info("All tests passed")
