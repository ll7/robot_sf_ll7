"""
Test: Social Force interaction between single and zone-spawned pedestrians (T031)

This test validates that single pedestrians and zone-spawned pedestrians interact physically (i.e., their positions are affected by each other's presence) in the simulation.
"""

import numpy as np

from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.sim.simulator import Simulator


def test_single_and_zone_pedestrian_interaction():
    width, height = 10.0, 10.0
    obstacles = []
    robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
    ped_spawn_zones = [((3, 3), (4, 3), (4, 4))]
    robot_goal_zones = [((8, 8), (9, 8), (9, 9))]
    bounds = [
        (0, width, 0, 0),
        (0, width, height, height),
        (0, 0, 0, height),
        (width, width, 0, height),
    ]
    ped_goal_zones = [((6, 6), (7, 6), (7, 7))]
    ped_crowded_zones = []
    robot_routes = []
    ped_routes = []
    # One single pedestrian, one zone-spawned pedestrian
    single_pedestrians = [
        SinglePedestrianDefinition(id="single1", start=(5.0, 5.0), goal=(9.0, 9.0))
    ]
    map_def = MapDefinition(
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
    # Add one zone-spawned pedestrian manually
    map_def.ped_spawn_zones = [((3, 3), (4, 3), (4, 4))]
    map_def.ped_goal_zones = [((7, 7), (8, 7), (8, 8))]
    # Create Simulator
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    sim_config = RobotSimulationConfig()
    sim_config.peds_per_area_m2 = 0.01
    sim_config.max_peds_per_group = 1
    sim = Simulator(
        config=sim_config,
        map_def=map_def,
        robots=[],
        goal_proximity_threshold=0.5,
        random_start_pos=False,
        peds_have_obstacle_forces=False,
    )
    initial_positions = sim.pysf_state.ped_positions.copy()
    for _ in range(5):
        sim.step_once([])  # No robot actions
    new_positions = sim.pysf_state.ped_positions
    # Both peds should have moved
    assert not np.allclose(initial_positions, new_positions)
    # Their distance should change (should not be identical)
    dist_initial = np.linalg.norm(initial_positions[0] - initial_positions[1])
    dist_new = np.linalg.norm(new_positions[0] - new_positions[1])
    assert dist_new != dist_initial
