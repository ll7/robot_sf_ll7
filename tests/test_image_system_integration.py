#!/usr/bin/env python3
"""
Integration test for the image-based observation system.
This is a simple validation script to ensure the image system works end-to-end.
"""

from robot_sf.gym_env.env_config import RobotEnvSettings
from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage
from robot_sf.sensor.image_sensor import ImageSensorSettings


def test_image_system_integration():
    """Test that the image observation system works end-to-end."""
    print("Testing image observation system integration...")

    # Test 1: Environment with image observations disabled
    print("1. Testing environment with image observations disabled...")
    settings_disabled = RobotEnvSettings(use_image_obs=False)
    env_disabled = RobotEnvWithImage(env_config=settings_disabled, debug=False)

    # Basic functionality test
    obs = env_disabled.reset()
    print(f"   Observation space keys: {list(obs[0].keys())}")
    assert "image" not in obs[0], "Image should not be in observation space when disabled"

    # Test exit (this was the failing case)
    env_disabled.exit()
    print("   ✓ Environment with disabled images works correctly")

    # Test 2: Environment with image observations enabled
    print("2. Testing environment with image observations enabled...")
    image_config = ImageSensorSettings(
        width=64, height=64, channels=3, normalize=True, grayscale=False
    )
    settings_enabled = RobotEnvSettings(use_image_obs=True, image_config=image_config)
    env_enabled = RobotEnvWithImage(env_config=settings_enabled, debug=True)

    # Basic functionality test
    obs = env_enabled.reset()
    print(f"   Observation space keys: {list(obs[0].keys())}")
    assert "image" in obs[0], "Image should be in observation space when enabled"

    # Check image shape
    image_obs = obs[0]["image"]
    print(f"   Image observation shape: {image_obs.shape}")
    assert image_obs.shape == (64, 64, 3), f"Expected (64, 64, 3), got {image_obs.shape}"

    # Test step
    action = env_enabled.action_space.sample()
    obs, reward, done, truncated, info = env_enabled.step(action)
    assert "image" in obs, "Image should be in observation after step"

    env_enabled.exit()
    print("   ✓ Environment with enabled images works correctly")

    print("✓ All image system integration tests passed!")


if __name__ == "__main__":
    test_image_system_integration()
