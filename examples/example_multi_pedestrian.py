"""
Example: Multi-Pedestrian Scenario (T020)

Demonstrates spawning and simulation of multiple single pedestrians (goal-based, trajectory-based, static) in a single map.

Usage:
    python examples/example_multi_pedestrian.py
"""

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle


def create_multi_pedestrian_map() -> MapDefinition:
    width, height = 20.0, 20.0
    obstacles = [
        Obstacle([(5, 0), (6, 0), (6, 8), (5, 8)]),
        Obstacle([(14, 12), (15, 12), (15, 20), (14, 20)]),
    ]
    robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
    robot_goal_zones = [((18, 18), (19, 18), (19, 19))]
    ped_spawn_zones = []
    ped_goal_zones = []
    ped_crowded_zones = []
    bounds = [
        (0, width, 0, 0),  # bottom
        (0, width, height, height),  # top
        (0, 0, 0, height),  # left
        (width, width, 0, height),  # right
    ]
    robot_routes = [
        GlobalRoute(
            spawn_id=0,
            goal_id=0,
            waypoints=[(1.5, 1.5), (10, 10), (18.5, 18.5)],
            spawn_zone=robot_spawn_zones[0],
            goal_zone=robot_goal_zones[0],
        ),
    ]
    ped_routes = []
    single_pedestrians = [
        SinglePedestrianDefinition(
            id="ped_goal_1",
            start=(3.0, 10.0),
            goal=(17.0, 10.0),
        ),
        SinglePedestrianDefinition(
            id="ped_goal_2",
            start=(8.0, 4.0),
            goal=(12.0, 16.0),
        ),
        SinglePedestrianDefinition(
            id="ped_static",
            start=(10.0, 10.0),
        ),
        SinglePedestrianDefinition(
            id="ped_traj_1",
            start=(2.0, 15.0),
            trajectory=[(8.0, 15.0), (8.0, 5.0), (18.0, 5.0)],
        ),
    ]
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


def run_multi_pedestrian_simulation():
    map_def = create_multi_pedestrian_map()
    pool = MapDefinitionPool(map_defs={"multi_ped": map_def})
    config = RobotSimulationConfig()
    config.map_pool = pool
    env = make_robot_env(config=config, debug=True)
    env.reset()
    print("\nMulti-pedestrian scenario (T033): 4 single pedestrians spawned.")
    for ped in map_def.single_pedestrians:
        if ped.trajectory:
            print(f"  - {ped.id}: start={ped.start}, trajectory={len(ped.trajectory)} waypoints")
        elif ped.goal:
            print(f"  - {ped.id}: start={ped.start}, goal={ped.goal}")
        else:
            print(f"  - {ped.id}: start={ped.start} (static)")
    print("\nObserve their interaction and movement during simulation.")
    done = False
    step = 0
    while not done and step < 200:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        step += 1
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    run_multi_pedestrian_simulation()
