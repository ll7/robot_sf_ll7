"""
Enhanced evaluation script with video recording for trained RL models.

This script extends the standard evaluation functionality by recording videos
of each episode and labeling them according to the termination reason:
- Route completion (successful navigation)
- Pedestrian collision
- Obstacle collision
- Robot collision
- Timeout (max steps exceeded)

Videos are saved in organized directories for easy analysis of policy behavior.

Usage:
    python scripts/evaluate_with_video.py

Requirements:
    - Trained RL model (PPO or A2C)
    - MoviePy for video recording
    - Debug mode enabled in environment

The script will:
1. Load the trained model
2. Evaluate on multiple difficulty levels
3. Record videos of simulation episodes
4. Organize videos by termination reason
5. Save evaluation metrics to JSON file
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Union

import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, PPO
from tqdm import tqdm

from robot_sf.eval import EnvMetrics
from robot_sf.gym_env.robot_env import EnvSettings, RobotEnv
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

# Type aliases for better readability
DriveModel = Union[PPO, A2C]
VehicleConfig = Union[DifferentialDriveSettings, BicycleDriveSettings]


@dataclass
class GymAdapterSettings:
    """
    Configuration for adapting the environment observation space to model requirements.
    """

    obs_space: spaces.Space
    action_space: spaces.Space
    obs_timesteps: int
    squeeze_obs: bool
    cut_2nd_target_angle: bool
    return_dict: bool

    def obs_adapter(self, obs):
        """Adapt environment observations to the format expected by the model."""
        if self.return_dict:
            return obs
        else:
            drive_state = obs[OBS_DRIVE_STATE]
            ray_state = obs[OBS_RAYS]

            if self.cut_2nd_target_angle:
                drive_state = drive_state[:, :-1]

            if self.squeeze_obs:
                drive_state = np.squeeze(drive_state)
                ray_state = np.squeeze(ray_state)

            axis = 0 if self.obs_timesteps == 1 else 1
            return np.concatenate((ray_state, drive_state), axis=axis)


@dataclass
class VideoEvalSettings:
    """Settings for video-enabled evaluation."""

    num_episodes: int
    ped_densities: List[float]
    vehicle_config: VehicleConfig
    prf_config: PedRobotForceConfig
    gym_config: GymAdapterSettings
    video_output_dir: str = "evaluation_videos"
    video_fps: float = 30.0
    record_all_episodes: bool = True  # Record all episodes or only failures


@dataclass
class AdaptedEnv(gymnasium.Env):
    """Environment wrapper that adapts the original environment to the expected format."""

    orig_env: RobotEnv
    config: GymAdapterSettings

    @property
    def observation_space(self) -> spaces.Space:
        return self.config.obs_space

    @property
    def action_space(self) -> spaces.Space:
        return self.config.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.orig_env.step(action)
        obs = self.config.obs_adapter(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.orig_env.reset(**kwargs)
        return self.config.obs_adapter(obs), info

    def render(self):
        """Forward render call to original environment."""
        return self.orig_env.render()

    def close(self):
        """Forward close call to original environment."""
        return self.orig_env.close()


def get_termination_reason(meta: dict) -> str:
    """
    Determine the termination reason from episode metadata.

    Args:
        meta: Episode metadata dictionary containing termination flags

    Returns:
        String describing termination reason for video file labeling
    """
    # Priority order: collisions first, then completion, then timeout
    if meta.get("is_pedestrian_collision", False):
        return "pedestrian_collision"
    elif meta.get("is_obstacle_collision", False):
        return "obstacle_collision"
    elif meta.get("is_robot_collision", False):
        return "robot_collision"
    elif meta.get("is_route_complete", False):
        return "route_complete"
    elif meta.get("is_timesteps_exceeded", False):
        return "timeout"
    else:
        return "unknown"


def create_video_filename(episode_num: int, difficulty: int, termination_reason: str) -> str:
    """
    Create a descriptive filename for the recorded video.

    Args:
        episode_num: Episode number
        difficulty: Difficulty level
        termination_reason: How the episode ended

    Returns:
        Formatted filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ep_{episode_num:03d}_diff_{difficulty}_{termination_reason}_{timestamp}.mp4"


def _setup_video_recording(env, episode_num: int, output_dir: Path) -> str:
    """Setup video recording for an episode."""
    if hasattr(env, "orig_env") and hasattr(env.orig_env, "sim_ui") and env.orig_env.sim_ui:
        # Clear any existing frames from previous episodes
        env.orig_env.sim_ui.frames = []
        env.orig_env.sim_ui.record_video = True
        return f"episode_{episode_num}.mp4"
    elif hasattr(env, "sim_ui") and env.sim_ui:
        # Direct environment access
        env.sim_ui.frames = []
        env.sim_ui.record_video = True
        return f"episode_{episode_num}.mp4"
    return f"episode_{episode_num}.mp4"  # Return filename even if no UI


def _finalize_video_recording(
    env, episode_meta, episode_num: int, difficulty: int, subdirs: dict, record_all: bool
) -> None:
    """Finalize and save video recording if needed."""
    if not episode_meta:
        return

    # Get simulation UI reference
    sim_ui = None
    if hasattr(env, "orig_env") and hasattr(env.orig_env, "sim_ui"):
        sim_ui = env.orig_env.sim_ui
    elif hasattr(env, "sim_ui"):
        sim_ui = env.sim_ui

    if not sim_ui or not sim_ui.record_video:
        return

    termination_reason = get_termination_reason(episode_meta)
    should_save_video = record_all or termination_reason != "route_complete"

    if should_save_video and sim_ui.frames:
        final_filename = create_video_filename(episode_num, difficulty, termination_reason)
        final_path = subdirs[termination_reason] / final_filename

        try:
            # Set the final path and trigger video creation
            sim_ui.video_path = str(final_path)

            # Manually create video from frames using moviepy if available
            try:
                from moviepy import ImageSequenceClip

                if sim_ui.frames:
                    clip = ImageSequenceClip(sim_ui.frames, fps=sim_ui.video_fps)
                    clip.write_videofile(str(final_path), verbose=False, logger=None)
                    print(f"Saved video: {final_path}")
            except ImportError:
                print(f"MoviePy not available. Cannot save video for episode {episode_num}")
            except Exception as e:
                print(f"Error saving video for episode {episode_num}: {e}")
        except Exception as e:
            print(f"Error finalizing video for episode {episode_num}: {e}")

    # Reset for next episode
    sim_ui.record_video = True  # Keep recording enabled
    sim_ui.frames = []  # Clear frames for next episode


def evaluate_with_video_recording(
    env: gymnasium.Env,
    model: DriveModel,
    num_episodes: int,
    video_output_dir: str,
    difficulty: int,
    record_all: bool = True,
) -> EnvMetrics:
    """
    Evaluate a model with video recording enabled.

    Args:
        env: Environment to evaluate on
        model: Model to evaluate
        num_episodes: Number of episodes to evaluate
        video_output_dir: Directory to save videos
        difficulty: Current difficulty level
        record_all: Whether to record all episodes or only failures

    Returns:
        Metrics collected during evaluation
    """
    # Initialize metrics collector
    eval_metrics = EnvMetrics(cache_size=num_episodes)

    # Create output directories
    output_dir = Path(video_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different termination reasons
    subdirs = {
        "route_complete": output_dir / "route_complete",
        "pedestrian_collision": output_dir / "pedestrian_collision",
        "obstacle_collision": output_dir / "obstacle_collision",
        "robot_collision": output_dir / "robot_collision",
        "timeout": output_dir / "timeout",
        "unknown": output_dir / "unknown",
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    # Run evaluation episodes
    for episode_num in tqdm(range(num_episodes), desc=f"Evaluating Difficulty {difficulty}"):
        obs, _ = env.reset()  # Handle tuple return from gym environment reset

        # Set up video recording for this episode
        _setup_video_recording(env, episode_num, output_dir)

        episode_meta = None
        is_end_of_route = False

        # Run episode
        while not is_end_of_route:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render frame for video recording
            env.render()

            # Extract metadata from info dict
            episode_meta = info.get("meta", {})
            eval_metrics.update(episode_meta)

            # Check if episode is done
            if done:
                obs, _ = env.reset()  # Handle tuple return from gym environment reset
                is_end_of_route = (
                    episode_meta.get("is_pedestrian_collision", False)
                    or episode_meta.get("is_obstacle_collision", False)
                    or episode_meta.get("is_route_complete", False)
                    or episode_meta.get("is_timesteps_exceeded", False)
                )

        # Finalize video recording
        if episode_meta:
            _finalize_video_recording(
                env, episode_meta, episode_num, difficulty, subdirs, record_all
            )

    return eval_metrics


def prepare_env_with_video(settings: VideoEvalSettings, difficulty: int) -> gymnasium.Env:
    """
    Prepare the environment with video recording enabled.

    Args:
        settings: Video evaluation settings
        difficulty: Difficulty level index

    Returns:
        Configured environment with video recording
    """
    # Create environment settings
    env_settings = EnvSettings()
    env_settings.sim_config.prf_config = settings.prf_config
    env_settings.sim_config.ped_density_by_difficulty = settings.ped_densities
    env_settings.sim_config.difficulty = difficulty
    env_settings.sim_config.stack_steps = settings.gym_config.obs_timesteps
    env_settings.robot_config = settings.vehicle_config

    # Create original environment with video recording enabled
    orig_env = RobotEnv(
        env_config=env_settings,
        debug=True,  # Enable debug mode for rendering
        record_video=True,
        video_fps=settings.video_fps,
    )

    # Return wrapped environment
    return AdaptedEnv(orig_env, settings.gym_config)


def prepare_model(model_path: str, env: gymnasium.Env) -> DriveModel:
    """Load a trained model from the given path."""
    try:
        # Try loading as PPO first (most common)
        return PPO.load(model_path, env=env)
    except Exception:
        try:
            # Fall back to A2C
            return A2C.load(model_path, env=env)
        except Exception as e:
            raise ValueError(f"Could not load model from {model_path}. Error: {e}")


def video_evaluation_series(model_path: str, settings: VideoEvalSettings):
    """
    Run a series of evaluations with video recording for different difficulty levels.

    Args:
        model_path: Path to the saved model
        settings: Video evaluation settings
    """
    # Dictionary to store metrics for each difficulty level
    all_metrics = dict()

    # Create base output directory
    base_output_dir = Path(settings.video_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate for each difficulty level
    for difficulty in range(len(settings.ped_densities)):
        print(f"\n=== Evaluating Difficulty Level {difficulty} ===")
        print(f"Pedestrian density: {settings.ped_densities[difficulty]}")

        # Prepare environment and model
        env = prepare_env_with_video(settings, difficulty)
        model = prepare_model(model_path, env)

        # Create difficulty-specific output directory
        difficulty_output_dir = base_output_dir / f"difficulty_{difficulty}"

        # Run evaluation with video recording
        eval_metrics = evaluate_with_video_recording(
            env,
            model,
            settings.num_episodes,
            str(difficulty_output_dir),
            difficulty,
            settings.record_all_episodes,
        )

        # Extract key metrics
        metrics = {
            "route_completion_rate": eval_metrics.route_completion_rate,
            "obstacle_collision_rate": eval_metrics.obstacle_collision_rate,
            "pedestrian_collision_rate": eval_metrics.pedestrian_collision_rate,
            "timeout_rate": eval_metrics.timeout_rate,
        }
        print(f"Metrics for difficulty {difficulty}:", metrics)

        # Store metrics and save to file
        all_metrics[difficulty] = metrics

        # Save results
        results_file = base_output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

        # Clean up environment
        env.close()

    print("\n=== Evaluation Complete ===")
    print(f"Results saved to: {base_output_dir}")
    print("Videos saved in subdirectories organized by termination reason")


def main():
    """Main function to run the video evaluation."""
    # Path to the saved model - Update this to your actual model path
    model_path = "./model/ppo_model_retrained_10m_2025-02-01.zip"

    # Prepare observation and action spaces for robot environment
    # The model was trained with a dictionary observation space
    from robot_sf.nav.map_config import MapDefinitionPool
    from robot_sf.sensor.goal_sensor import target_sensor_space
    from robot_sf.sensor.range_sensor import lidar_sensor_space
    from robot_sf.sensor.sensor_fusion import fused_sensor_space

    # Get a sample map to determine target sensor space
    map_pool = MapDefinitionPool()
    sample_map = map_pool.choose_random_map()

    # Create environment settings to get proper observation space
    env_settings = EnvSettings(
        robot_config=DifferentialDriveSettings(
            radius=1.0,
            max_linear_speed=0.5,
            max_angular_speed=0.5,
            wheel_radius=0.05,
            interaxis_length=0.3,
        )
    )

    # Create the robot to get its observation space
    robot = env_settings.robot_factory()

    # Create the dictionary observation space as expected by the trained model
    obs_space, orig_obs_space = fused_sensor_space(
        timesteps=3,  # default timesteps
        robot_obs=robot.observation_space,
        target_obs=target_sensor_space(sample_map.max_target_dist),
        lidar_obs=lidar_sensor_space(num_rays=16, max_scan_dist=10.0),  # typical values
    )

    action_space = robot.action_space

    # Configure gym adapter settings for dictionary observation space
    gym_settings = GymAdapterSettings(
        obs_space=obs_space,
        action_space=action_space,
        obs_timesteps=3,
        squeeze_obs=False,  # Keep dictionary structure
        cut_2nd_target_angle=True,
        return_dict=True,  # Return dictionary observations
    )

    # Configure vehicle settings
    vehicle_config = env_settings.robot_config

    # Configure vehicle settings
    vehicle_config = DifferentialDriveSettings(
        radius=1.0,
        max_linear_speed=0.5,
        max_angular_speed=0.5,
        wheel_radius=0.05,
        interaxis_length=0.3,
    )

    # Configure pedestrian-robot force settings
    prf_config = PedRobotForceConfig(
        is_active=True, robot_radius=1.0, activation_threshold=2.0, force_multiplier=10.0
    )

    # Create video evaluation settings
    settings = VideoEvalSettings(
        num_episodes=2,  # Number of episodes per difficulty level (reduced for testing)
        ped_densities=[0.00, 0.02],  # Different difficulty levels (reduced for testing)
        vehicle_config=vehicle_config,
        prf_config=prf_config,
        gym_config=gym_settings,
        video_output_dir="evaluation_videos",
        video_fps=30.0,
        record_all_episodes=True,  # Record all episodes, not just failures
    )

    # Run video evaluation series
    video_evaluation_series(model_path, settings)


if __name__ == "__main__":
    main()


# Example usage:
#
# To run evaluation with video recording:
# 1. Ensure you have a trained model (PPO or A2C)
# 2. Update the model_path in main() to point to your model
# 3. Run: python scripts/evaluate_with_video.py
#
# The script will:
# - Evaluate the model across different difficulty levels
# - Record videos of each episode
# - Organize videos by termination reason:
#   * route_complete/ - Successfully completed episodes
#   * pedestrian_collision/ - Episodes ending in pedestrian collision
#   * obstacle_collision/ - Episodes ending in obstacle collision
#   * robot_collision/ - Episodes ending in robot collision
#   * timeout/ - Episodes that exceeded maximum steps
#   * unknown/ - Episodes with unidentified termination
# - Save evaluation metrics to evaluation_results.json
#
# Videos are named: ep_XXX_diff_Y_TERMINATION_TIMESTAMP.mp4
# where XXX = episode number, Y = difficulty level
