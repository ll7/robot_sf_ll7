"""Global planner demo: show integration with environment factory.

The global planner provides deterministic waypoint generation for robot navigation.
This demo shows how to enable and use the planner in a simulation environment.

Note: Detailed planner API usage can be found in the test suite
(tests/test_planner/) and in the robot_sf.planner module docstrings.
"""

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig


def main() -> None:
    """Demonstrate global planner integration with environment.

    The planner is automatically used by the navigation module when enabled.
    It provides deterministic collision-free paths for robot waypoint generation.
    """
    print("Global Planner Demo")
    print("=" * 60)
    print("\nCreating environment with planner enabled...")

    # Create config with global planner enabled
    config = RobotSimulationConfig()
    config.use_planner = True

    # Create environment - planner is integrated internally
    try:
        env = make_robot_env(config=config, debug=False)
        _, _ = env.reset()

        print("✓ Environment created successfully")
        print(f"  Map size: {env.map_definition.width} x {env.map_definition.height}")
        print(f"  Obstacles: {len(env.map_definition.obstacles)}")
        print(f"  Robot spawn zones: {len(env.map_definition.robot_spawn_zones)}")
        print(f"  POI positions: {len(env.map_definition.poi_positions)}")

        # Run simulation with planner
        print("\nRunning 10 simulation steps with planner...")
        for step in range(10):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            if (step + 1) % 2 == 0:
                print(f"  Step {step + 1:2d}: reward={reward:7.3f}")
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break

        print("\n✓ Demo completed successfully!")
        print("\nThe planner automatically handles:")
        print("  - Collision-free path generation")
        print("  - Obstacle avoidance via visibility graphs")
        print("  - Multi-goal routing with caching")
        print("  - Path smoothing for robot navigation")

    except Exception as e:
        print(f"\n✗ Error during demo: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
