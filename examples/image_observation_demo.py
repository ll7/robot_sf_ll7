#!/usr/bin/env python3
"""
Example demonstrating image-based observation space for the robot environment.

This script shows how to use the extended robot environment with visual observations
captured from the pygame rendering system.
"""

from robot_sf.gym_env.env_config import RobotEnvSettings
from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage
from robot_sf.sensor.image_sensor import ImageSensorSettings


def main():
    """
    Runs a demonstration of image-based observations in a robot environment.
    
    Configures an image sensor for RGB observations, sets up the robot environment to use image-based observations, and interacts with the environment for several steps. Prints details about the action and observation spaces, including image observation properties, and handles environment resets and closure.
    """

    # Configure image sensor settings
    image_config = ImageSensorSettings(
        width=84,  # Standard size for RL vision tasks
        height=84,  # Standard size for RL vision tasks
        channels=3,  # RGB channels
        normalize=True,  # Normalize pixel values to [0, 1]
        grayscale=False,  # Keep color information
    )

    # Configure environment with image observations enabled
    env_config = RobotEnvSettings(
        image_config=image_config,
        use_image_obs=True,  # Enable image observations
    )

    # Create the robot environment with image observations
    env = RobotEnvWithImage(
        env_config=env_config,
        debug=True,  # Enable visualization
        recording_enabled=False,
        record_video=False,
    )

    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")

    if "image" in env.observation_space.spaces:
        image_space = env.observation_space.spaces["image"]
        print(f"Image observation space: {image_space}")
        print(f"Image shape: {image_space.shape}")
        print(f"Image dtype: {image_space.dtype}")

    try:
        # Reset the environment to get initial observation
        obs, info = env.reset()
        print(f"Initial observation keys: {list(obs.keys())}")

        # Check if image observation is present
        if "image" in obs:
            image_obs = obs["image"]
            print(f"Image observation shape: {image_obs.shape}")
            print(f"Image observation dtype: {image_obs.dtype}")
            print(f"Image value range: [{image_obs.min():.3f}, {image_obs.max():.3f}]")

        # Take a few random actions to see if the environment works
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")

            if "image" in obs:
                image_obs = obs["image"]
                print(
                    f"  Image shape: {image_obs.shape}, range: [{image_obs.min():.3f}, {image_obs.max():.3f}]"
                )

            if terminated or truncated:
                print("Episode finished, resetting...")
                obs, info = env.reset()
                break

    except Exception as e:
        print(f"Error during environment execution: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
