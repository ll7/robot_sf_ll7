"""Spawn and simulate single pedestrians with custom maps.

Usage:
    uv run python examples/advanced/07_single_pedestrian.py

Prerequisites:
    - None

Expected Output:
    - Console overview of configured pedestrians and an interactive pygame window.

Limitations:
    - Requires display or headless setup to render the simulation.

References:
    - docs/dev_guide.md#pedestrian-environments
"""

from typing import TYPE_CHECKING

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle

if TYPE_CHECKING:
    from robot_sf.common.types import Vec2D


def create_simple_map_with_single_pedestrians() -> MapDefinition:
    """
    Create a simple map with single pedestrians for demonstration.

    Returns:
        MapDefinition: A map with obstacles and single pedestrians
    """
    # Map dimensions
    width, height = 20.0, 20.0

    # Create a simple corridor with obstacles
    obstacles = [
        Obstacle([(5, 0), (6, 0), (6, 8), (5, 8)]),  # Left wall
        Obstacle([(14, 12), (15, 12), (15, 20), (14, 20)]),  # Right wall
    ]

    # Define spawn and goal zones for the robot
    robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
    robot_goal_zones = [((18, 18), (19, 18), (19, 19))]

    # Pedestrian zones (for crowd pedestrians - not used in this example)
    ped_spawn_zones = []
    ped_goal_zones = []
    ped_crowded_zones = []

    # Map bounds (edges)
    bounds = [
        (0, width, 0, 0),  # bottom
        (0, width, height, height),  # top
        (0, 0, 0, height),  # left
        (width, width, 0, height),  # right
    ]

    # Robot routes (at least one required)
    from robot_sf.nav.global_route import GlobalRoute

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

    # Define single pedestrians with different behaviors
    single_pedestrians = [
        # Pedestrian 1: Moving from left to right with a fixed goal
        SinglePedestrianDefinition(
            id="ped_goal_1",
            start=(3.0, 10.0),
            goal=(17.0, 10.0),
        ),
        # Pedestrian 2: Moving diagonally
        SinglePedestrianDefinition(
            id="ped_goal_2",
            start=(8.0, 4.0),
            goal=(12.0, 16.0),
        ),
        # Pedestrian 3: Static pedestrian (no goal, no trajectory)
        SinglePedestrianDefinition(
            id="ped_static",
            start=(10.0, 10.0),
        ),
        # Pedestrian 4: Following a trajectory with waypoints
        SinglePedestrianDefinition(
            id="ped_traj",
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


def run_simulation_with_single_pedestrians(num_steps: int = 500):
    """
    Run a simulation with single pedestrians and visualize the results.

    Args:
        num_steps: Number of simulation steps to run
    """
    print("=" * 60)
    print("Single Pedestrian Spawning Example")
    print("=" * 60)

    # Create custom map with single pedestrians
    custom_map = create_simple_map_with_single_pedestrians()

    print(f"\nMap created with {len(custom_map.single_pedestrians)} single pedestrians:")
    for ped in custom_map.single_pedestrians:
        if ped.goal:
            print(f"  - {ped.id}: start={ped.start}, goal={ped.goal}")
        elif ped.trajectory:
            print(f"  - {ped.id}: start={ped.start}, trajectory={len(ped.trajectory)} waypoints")
        else:
            print(f"  - {ped.id}: start={ped.start} (static)")

    # Configure simulation with custom map
    # Create a map pool with just our custom map
    map_pool = MapDefinitionPool(map_defs={"single_ped_demo": custom_map})
    config = RobotSimulationConfig(map_pool=map_pool)

    # Create environment
    # The map doesn't have ped_crowded_zones, so no crowd pedestrians will spawn
    env = make_robot_env(config=config, debug=True)

    print(f"\nRunning simulation for {num_steps} steps...")
    print("Press Ctrl+C to stop early\n")

    try:
        obs, _ = env.reset()
        print(f"Initial observation type: {type(obs)}")
        print("Simulation initialized with single pedestrians")

        # Run simulation
        for step in range(num_steps):
            # Random robot action (for demonstration)
            action = env.action_space.sample()

            obs, _, terminated, truncated, _ = env.step(action)
            env.render()  # Active pygame visualization

            if step % 50 == 0:
                print(f"Step {step:3d}: Simulation running...")

            if terminated or truncated:
                print(f"\nEpisode finished at step {step}")
                break

        print("\n" + "=" * 60)
        print("Simulation completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

    finally:
        env.close()


def demonstrate_programmatic_pedestrian_creation():
    """
    Demonstrate how to create single pedestrians programmatically.
    """
    print("\n" + "=" * 60)
    print("Programmatic Pedestrian Creation Example")
    print("=" * 60 + "\n")

    # Example 1: Goal-based pedestrian
    ped1 = SinglePedestrianDefinition(
        id="commuter_1",
        start=(2.0, 5.0),
        goal=(18.0, 5.0),
    )
    print(f"Created goal-based pedestrian: {ped1}")

    # Example 2: Trajectory-based pedestrian
    trajectory: list[Vec2D] = [(5.0, 10.0), (10.0, 15.0), (15.0, 10.0)]
    ped2 = SinglePedestrianDefinition(
        id="tourist_1",
        start=(1.0, 10.0),
        trajectory=trajectory,
    )
    print(f"Created trajectory-based pedestrian: {ped2}")

    # Example 3: Static pedestrian
    ped3 = SinglePedestrianDefinition(
        id="vendor_1",
        start=(10.0, 10.0),
    )
    print(f"Created static pedestrian: {ped3}")

    # Demonstrate validation
    print("\n" + "-" * 60)
    print("Validation Examples:")
    print("-" * 60)

    try:
        # This will raise an error (goal and trajectory are mutually exclusive)
        SinglePedestrianDefinition(
            id="invalid",
            start=(5.0, 5.0),
            goal=(10.0, 10.0),
            trajectory=[(7.5, 7.5)],
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")

    print("\nAll validation examples passed!")


def main():
    """Main entry point for the example."""
    # Demonstrate programmatic creation
    demonstrate_programmatic_pedestrian_creation()

    # Run full simulation
    print("\n")
    run_simulation_with_single_pedestrians(num_steps=200)


if __name__ == "__main__":
    main()
