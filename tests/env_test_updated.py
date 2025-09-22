"""
Updated test file demonstrating both old and new patterns.

This shows backward compatibility while encouraging migration to the new factory pattern.
"""

from typing import cast

from gymnasium import spaces
from stable_baselines3 import PPO

# New factory pattern imports
from robot_sf.gym_env.environment_factory import (
    EnvironmentFactory,
    make_image_robot_env,
    make_robot_env,
)

# Legacy imports (still work)
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.unified_config import (
    ImageRobotConfig,
    PedestrianSimulationConfig,
    RobotSimulationConfig,
)
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


def test_can_create_env_legacy():
    """Test legacy environment creation - backward compatibility."""
    env = RobotEnv()
    assert env is not None
    # Check that config attribute was added for consistency
    assert hasattr(env, "config")


def test_can_create_env_new():
    """Test new factory pattern environment creation."""
    env = make_robot_env()
    assert env is not None
    assert hasattr(env, "config")
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")


def test_can_return_valid_observation_legacy():
    """Test observations with legacy pattern."""
    env = RobotEnv()
    obs_dict = cast(spaces.Dict, env.observation_space)
    drive_state_spec = cast(spaces.Box, obs_dict[OBS_DRIVE_STATE])
    lidar_state_spec = cast(spaces.Box, obs_dict[OBS_RAYS])

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert OBS_DRIVE_STATE in obs and OBS_RAYS in obs
    assert drive_state_spec.shape == obs[OBS_DRIVE_STATE].shape
    assert lidar_state_spec.shape == obs[OBS_RAYS].shape


def test_can_return_valid_observation_new():
    """Test observations with new factory pattern."""
    env = make_robot_env()
    obs_dict = cast(spaces.Dict, env.observation_space)
    drive_state_spec = cast(spaces.Box, obs_dict[OBS_DRIVE_STATE])
    lidar_state_spec = cast(spaces.Box, obs_dict[OBS_RAYS])

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert OBS_DRIVE_STATE in obs and OBS_RAYS in obs
    assert drive_state_spec.shape == obs[OBS_DRIVE_STATE].shape
    assert lidar_state_spec.shape == obs[OBS_RAYS].shape


def test_can_simulate_with_pedestrians_legacy():
    """Test simulation with legacy pattern."""
    total_steps = 100  # Reduced for faster testing
    env = RobotEnv()
    env.reset()
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(rand_action)
        done = terminated or truncated
        if done:
            env.reset()


def test_can_simulate_with_pedestrians_new():
    """Test simulation with new factory pattern."""
    total_steps = 100  # Reduced for faster testing
    env = make_robot_env()
    env.reset()
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(rand_action)
        done = terminated or truncated
        if done:
            env.reset()


def test_image_robot_env():
    """Test image robot environment creation."""
    env = make_image_robot_env(debug=False)
    assert env is not None
    assert hasattr(env, "config")
    assert env.config.use_image_obs

    # Should have image observations in observation space
    obs_space = cast(spaces.Dict, env.observation_space)
    obs_space_keys = set(obs_space.spaces.keys())
    expected_keys = {OBS_DRIVE_STATE, OBS_RAYS, "image"}
    assert expected_keys.issubset(obs_space_keys)

    env.exit()


def test_config_hierarchy():
    """Test the new configuration hierarchy."""
    # Test basic robot config
    robot_config = RobotSimulationConfig()
    assert not robot_config.use_image_obs

    # Test image robot config
    image_config = ImageRobotConfig()
    assert image_config.use_image_obs
    assert hasattr(image_config, "image_config")

    # Test pedestrian config
    ped_config = PedestrianSimulationConfig()
    assert hasattr(ped_config, "ego_ped_config")
    assert hasattr(ped_config, "robot_config")


def test_factory_consistency():
    """Test that all factory-created environments have consistent interfaces."""
    environments = [
        ("Robot", make_robot_env(debug=False)),
        ("Image Robot", make_image_robot_env(debug=False)),
    ]

    required_methods = ["step", "reset", "render", "exit"]
    required_attributes = ["action_space", "observation_space", "config"]

    for name, env in environments:
        try:
            # Test methods
            for method in required_methods:
                assert hasattr(env, method), f"{name} missing method {method}"

            # Test attributes
            for attr in required_attributes:
                assert hasattr(env, attr), f"{name} missing attribute {attr}"

            # Test that config is properly typed
            assert hasattr(env.config, "sim_config")
            assert hasattr(env.config, "map_pool")

        finally:
            env.exit()


def test_ego_ped_env_legacy():
    """Test legacy pedestrian environment."""
    try:
        total_steps = 10  # Very short test
        robot_model = PPO.load("./model/run_043", env=None)
        env = PedestrianEnv(robot_model=robot_model)
        assert env is not None
        env.reset()
        for _ in range(total_steps):
            rand_action = env.action_space.sample()
            _, _, done, _, _ = env.step(rand_action)
            if done:
                env.reset()
    except FileNotFoundError:
        # Skip test if model file doesn't exist
        pass


def test_environment_factory_methods():
    """Test the EnvironmentFactory class directly."""
    # Test robot environment creation
    robot_env = EnvironmentFactory.create_robot_env(debug=False)
    assert robot_env is not None
    assert hasattr(robot_env, "config")
    robot_env.exit()

    # Test image robot environment creation
    image_env = EnvironmentFactory.create_robot_env(use_image_obs=True, debug=False)
    assert image_env is not None
    assert hasattr(image_env, "config")
    image_env.exit()


if __name__ == "__main__":
    # Run a subset of tests
    print("Testing legacy environment creation...")
    test_can_create_env_legacy()

    print("Testing new factory pattern...")
    test_can_create_env_new()

    print("Testing configuration hierarchy...")
    test_config_hierarchy()

    print("Testing factory consistency...")
    test_factory_consistency()

    print("Testing image environment...")
    test_image_robot_env()

    print("All tests passed! ✅")
