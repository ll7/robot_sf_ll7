#!/usr/bin/env python3
"""
Demonstration of the new environment factory pattern.

This script shows how the refactored environment system provides
a consistent, clean interface for creating different types of
simulation environments.
"""

import os
import sys

# Add the robot_sf package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_sf.gym_env.environment_factory import (
    EnvironmentFactory,
    make_image_robot_env,
    make_robot_env,
)
from robot_sf.gym_env.unified_config import (
    ImageRobotConfig,
    MultiRobotConfig,
    PedestrianSimulationConfig,
    RobotSimulationConfig,
)


def demo_factory_pattern():
    """Demonstrate the new factory pattern for environment creation."""
    print("🤖 Environment Factory Pattern Demo")
    print("=" * 50)

    # 1. Create a basic robot environment
    print("\n1. Creating basic robot environment...")
    try:
        robot_env = make_robot_env(debug=False)
        print(f"✅ Created {type(robot_env).__name__}")
        print(f"   Action space: {robot_env.action_space}")
        print(f"   Observation space keys: {robot_env.observation_space.spaces.keys()}")
        robot_env.exit()
    except Exception as e:
        print(f"❌ Failed to create robot environment: {e}")

    # 2. Create robot environment with image observations
    print("\n2. Creating robot environment with image observations...")
    try:
        image_robot_env = make_image_robot_env(debug=False)
        print(f"✅ Created {type(image_robot_env).__name__}")
        print(f"   Has image observations: {'image' in image_robot_env.observation_space.spaces}")
        image_robot_env.exit()
    except Exception as e:
        print(f"❌ Failed to create image robot environment: {e}")

    # 3. Create environment using factory with custom config
    print("\n3. Creating environment with custom configuration...")
    try:
        custom_config = RobotSimulationConfig()
        custom_config.peds_have_obstacle_forces = True

        custom_env = EnvironmentFactory.create_robot_env(
            config=custom_config,
            debug=False,
            recording_enabled=True,
        )
        print(f"✅ Created {type(custom_env).__name__} with custom config")
        print(f"   Pedestrian obstacle forces: {custom_env.config.peds_have_obstacle_forces}")
        print(f"   Recording enabled: {custom_env.recording_enabled}")
        custom_env.exit()
    except Exception as e:
        print(f"❌ Failed to create custom environment: {e}")

    # 4. Show configuration hierarchy
    print("\n4. Configuration hierarchy demonstration...")
    try:
        base_config = RobotSimulationConfig()
        image_config = ImageRobotConfig()
        ped_config = PedestrianSimulationConfig()
        multi_config = MultiRobotConfig(num_robots=3)

        print(f"✅ RobotSimulationConfig - use_image_obs: {base_config.use_image_obs}")
        print(f"✅ ImageRobotConfig - use_image_obs: {image_config.use_image_obs}")
        print(
            f"✅ PedestrianSimulationConfig has ego_ped_config: {hasattr(ped_config, 'ego_ped_config')}",
        )
        print(f"✅ MultiRobotConfig - num_robots: {multi_config.num_robots}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")


def demo_consistency():
    """Demonstrate consistent interface across all environment types."""
    print("\n\n🔄 Interface Consistency Demo")
    print("=" * 50)

    environments = []

    try:
        # Create different types of environments
        robot_env = make_robot_env(debug=False)
        environments.append(("Robot Environment", robot_env))

        # Note: Pedestrian env needs a robot model, so we'll skip it in this demo
        # ped_env = make_pedestrian_env(robot_model=None, debug=False)
        # environments.append(("Pedestrian Environment", ped_env))

    except Exception as e:
        print(f"❌ Error creating environments: {e}")
        return

    # Test consistent interface
    print("\nTesting consistent interface across environments:")
    for name, env in environments:
        print(f"\n{name}:")
        try:
            # All environments should have these methods
            methods = ["step", "reset", "render", "exit"]
            for method in methods:
                has_method = hasattr(env, method)
                print(f"  {method}(): {'✅' if has_method else '❌'}")

            # All should have these attributes
            attributes = ["action_space", "observation_space", "config"]
            for attr in attributes:
                has_attr = hasattr(env, attr)
                print(f"  {attr}: {'✅' if has_attr else '❌'}")

        except Exception as e:
            print(f"  ❌ Error testing {name}: {e}")
        finally:
            env.exit()


def demo_backward_compatibility():
    """Demonstrate backward compatibility with old interface."""
    print("\n\n🔙 Backward Compatibility Demo")
    print("=" * 50)

    try:
        # Show that old config classes still work (with deprecation warnings)
        from robot_sf.gym_env.env_config import EnvSettings

        print("✅ Old configuration classes can still be imported")
        print("   (Though they may show deprecation warnings)")

        # Show factory can work with old-style configurations
        old_config = EnvSettings()
        print(f"✅ Created old-style EnvSettings: {type(old_config).__name__}")

    except Exception as e:
        print(f"❌ Backward compatibility issue: {e}")


if __name__ == "__main__":
    print("🚀 Robot SF Environment Refactoring Demo")
    print("This demo shows the new consistent environment interface")
    print()

    # Run demonstrations
    demo_factory_pattern()
    demo_consistency()
    demo_backward_compatibility()

    print("\n\n✨ Demo complete!")
    print("\nKey benefits of the new system:")
    print("• Consistent interface across all environment types")
    print("• Reduced code duplication")
    print("• Clear configuration hierarchy")
    print("• Easy factory-based creation")
    print("• Backward compatibility with existing code")
    print("• Better extensibility for new environment types")
