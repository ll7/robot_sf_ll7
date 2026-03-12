"""Tests for ego pedestrian social force emission and state correctness.

This module verifies that:
- NPC pedestrians properly respond to ego pedestrian presence via social forces
- Ego pedestrian position is correctly tracked in simulator state
"""

import numpy as np
import pytest
from stable_baselines3 import PPO

from robot_sf.common.types import Line2D
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.nav_types import SvgRectangle
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings


@pytest.fixture
def dummy_map():
    """Create a minimal map definition for deterministic pedestrian routing.

    This keeps environment setup stable across tests to isolate behavior under test.
    """
    width = 40.0
    height = 40.0
    bounds: list[Line2D] = [
        (0, width, 0, 0),  # bottom
        (0, width, height, height),  # top
        (0, 0, 0, height),  # left
        (width, width, 0, height),  # right
    ]
    r_spawn = SvgRectangle(
        x=20, y=30, width=1, height=1, label="robot_spawn", id_="robot_spawn"
    ).get_zone()

    r_goal = SvgRectangle(
        x=30, y=30, width=1, height=1, label="robot_goal", id_="robot_goal"
    ).get_zone()

    r_path = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[[22, 30], [29, 30]],
        spawn_zone=r_spawn,
        goal_zone=r_goal,
    )

    single_pedestrians = [
        SinglePedestrianDefinition(
            id="npc_0",
            start=(1.0, 10.0),
            goal=(34.0, 10.0),
        ),
    ]

    map_def = MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[r_spawn],
        ped_spawn_zones=[],
        robot_goal_zones=[r_goal],
        bounds=bounds,
        robot_routes=[r_path],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=single_pedestrians,
    )
    return map_def


@pytest.fixture
def env(dummy_map):
    """Build a pedestrian environment with a fixed PPO model.

    This ensures all tests run against the same policy and simulator configuration.
    """
    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": dummy_map}),
        sim_config=SimulationSettings(
            debug_without_robot_movement=True,
            peds_reset_follow_route_at_start=True,
            difficulty=0,
            ped_density_by_difficulty=[0.0],
        ),  # Enable debug mode to isolate social force effects
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )

    robot_model = "./model/run_043"

    robot_model = PPO.load(robot_model, env=None)
    env = make_pedestrian_env(
        config=config,
        robot_model=robot_model,
        debug=False,
        recording_enabled=False,
        peds_have_obstacle_forces=False,
    )
    return env


def test_npc_pedestrian_avoids_ego_pedestrian_social_force(env):
    """
    Test that the NPC pedestrian senses the ego pedestrian using social forces.
    """
    _, _ = env.reset()
    assert len(env.simulator.ped_pos) == 1, (
        f"Expected exactly one NPC pedestrian in this test, not {len(env.simulator.ped_pos)}"
    )
    # Case 1: Ego in the way of NPC
    # Place ego at (12, 10), which is along the NPC's route from (7, 10) to (34, 10)
    env.simulator.ego_ped.state.pose = ((12.0, 10.0), 0)
    ped_positions = env.simulator.pysf_state.ped_positions
    print("Pedestrian positions over time:", ped_positions)
    social_force_threshold = 0.5
    threshold_reached = False
    for i in range(200):
        action = np.array([0, 0])  # No action for the ego pedestrian
        _, _, done, _, _ = env.step(action)
        # Get the social force for the NPC (index 0)
        sf = env.simulator.pysf_sim.forces[1]()  # index 1 is social force
        social_force_norm = float(np.linalg.norm(sf[0]))
        if social_force_norm > social_force_threshold:
            assert social_force_norm > social_force_threshold
            threshold_reached = True
            break
        if done:
            break

    assert threshold_reached, "NPC social force never exceeded 0.5 when ego was in the path"

    _, _ = env.reset()
    # Case 2: Ego not in the way of NPC
    # Place ego at (30, 30), which is not along the NPC's route from (1.0, 10.0) to (34.0, 10.0)
    env.simulator.ego_ped.state.pose = ((30.0, 30.0), 0)
    social_forces = []
    for i in range(200):
        action = np.array([0, 0])  # No action for the ego pedestrian
        _, _, done, _, _ = env.step(action)
        # Get the social force for the NPC (index 0)
        sf = env.simulator.pysf_sim.forces[1]()  # index 1 is social force
        social_forces.append(sf[0].copy())

        if done:
            break

    # Check that the social force doesn't change at some point
    diffs = [
        np.linalg.norm(social_forces[i + 1] - social_forces[i])
        for i in range(len(social_forces) - 1)
    ]
    assert not any(d > 1e-6 for d in diffs), (
        "Social force on NPC changed during the episode without ego ped"
    )

    env.exit()


def test_ego_position_correct_in_states(env):
    """
    Test that the ego pedestrian's position is correctly reflected in the simulator state.
    """
    _, _ = env.reset()

    for i in range(5):
        action = np.array([0, 0])  # No action for the ego pedestrian
        _, _, _, _, _ = env.step(action)

        assert np.allclose(
            env.simulator.pysf_state.pysf_states()[-1, 0:2], env.simulator.ego_ped_pos
        ), "Ego pedestrian position does not match the simulator state"

    env.simulator.ego_ped.state.pose = ((12.0, 5.0), 0)
    for i in range(5):
        action = np.array([0, 0])  # No action for the ego pedestrian
        _, _, _, _, _ = env.step(action)

        assert np.allclose(
            env.simulator.pysf_state.pysf_states()[-1, 0:2], env.simulator.ego_ped_pos
        ), "Ego pedestrian position does not match the simulator state"

    env.simulator.ego_ped.state.pose = ((20.0, 7.0), 0)
    for i in range(5):
        action = np.array([0, 0])  # No action for the ego pedestrian
        _, _, _, _, _ = env.step(action)

        assert np.allclose(
            env.simulator.pysf_state.pysf_states()[-1, 0:2], env.simulator.ego_ped_pos
        ), "Ego pedestrian position does not match the simulator state"

    env.exit()


if __name__ == "__main__":
    pytest.main([__file__])
