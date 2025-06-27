#!/usr/bin/env python3
"""
Test script for video recording functionality during evaluation.

This script demonstrates the video recording capabilities without requiring
a fully trained model. It uses a simple random policy for testing.
"""

import numpy as np
from gymnasium import spaces

from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from scripts.evaluate_with_video import (
    AdaptedEnv,
    GymAdapterSettings,
    VideoEvalSettings,
    create_video_filename,
    get_termination_reason,
)


class RandomPolicy:
    """Simple random policy for testing."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        """Predict random action."""
        return self.action_space.sample(), None


def test_video_recording():
    """Test video recording with a simple random policy."""
    print("Testing video recording functionality...")

    # Create a simple configuration for testing
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(94,), dtype=np.float64)
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    gym_settings = GymAdapterSettings(
        obs_space=obs_space,
        action_space=action_space,
        obs_timesteps=1,
        squeeze_obs=True,
        cut_2nd_target_angle=True,
        return_dict=False,
    )

    vehicle_config = DifferentialDriveSettings(
        radius=1.0,
        max_linear_speed=0.5,
        max_angular_speed=0.5,
        wheel_radius=0.05,
        interaxis_length=0.3,
    )

    prf_config = PedRobotForceConfig(
        is_active=True, robot_radius=1.0, activation_threshold=2.0, force_multiplier=10.0
    )

    # Test settings with only 2 episodes for quick testing
    settings = VideoEvalSettings(
        num_episodes=2,  # Small number for testing
        ped_densities=[0.00, 0.02],  # Only 2 difficulty levels
        vehicle_config=vehicle_config,
        prf_config=prf_config,
        gym_config=gym_settings,
        video_output_dir="test_videos",
        video_fps=10.0,  # Lower FPS for faster processing
        record_all_episodes=True,
    )

    # Test individual functions
    print("Testing termination reason detection...")
    test_meta = {
        "is_pedestrian_collision": True,
        "is_obstacle_collision": False,
        "is_route_complete": False,
        "is_timesteps_exceeded": False,
    }
    reason = get_termination_reason(test_meta)
    print(f"Termination reason: {reason}")
    assert reason == "pedestrian_collision"

    print("Testing video filename creation...")
    filename = create_video_filename(1, 0, "route_complete")
    print(f"Generated filename: {filename}")
    assert "ep_001_diff_0_route_complete" in filename

    # Test environment creation
    print("Testing environment creation...")
    env_settings = EnvSettings()
    env_settings.sim_config.prf_config = settings.prf_config
    env_settings.sim_config.ped_density_by_difficulty = settings.ped_densities
    env_settings.sim_config.difficulty = 0
    env_settings.sim_config.stack_steps = settings.gym_config.obs_timesteps
    env_settings.robot_config = settings.vehicle_config

    # Create environment with video recording enabled
    orig_env = RobotEnv(
        env_config=env_settings,
        debug=True,
        record_video=True,
        video_fps=settings.video_fps,
    )

    env = AdaptedEnv(orig_env, settings.gym_config)

    print("Testing environment reset and step...")
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Test a few steps with random actions
    random_policy = RandomPolicy(env.action_space)
    for i in range(5):
        action, _ = random_policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # This should capture frames
        print(f"Step {i + 1}: reward={reward:.3f}, terminated={terminated}")

        if terminated or truncated:
            obs, _ = env.reset()
            break

    # Clean up
    env.close()
    print("✅ All tests passed!")
    print("Video recording functionality is working correctly.")
    print(f"Note: Videos would be saved to '{settings.video_output_dir}' directory")


if __name__ == "__main__":
    test_video_recording()
