"""
Environment Factory for creating standardized simulation environments.

This factory provides a consistent interface for creating different types
of environments while hiding the complexity of configuration and setup.
"""

from typing import Callable, Optional

from robot_sf.gym_env.abstract_envs import MultiAgentEnv, SingleAgentEnv
from robot_sf.gym_env.unified_config import (
    ImageRobotConfig,
    MultiRobotConfig,
    PedestrianSimulationConfig,
    RobotSimulationConfig,
)


class EnvironmentFactory:
    """
    Factory class for creating simulation environments with consistent interfaces.

    This factory encapsulates the complexity of choosing the right environment
    class and configuration, providing a simple interface for users.
    """

    @staticmethod
    def create_robot_env(
        config: Optional[RobotSimulationConfig] = None,
        *,
        use_image_obs: bool = False,
        peds_have_obstacle_forces: bool = False,
        reward_func: Optional[Callable] = None,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None,
        video_fps: Optional[float] = None,
        **kwargs,
    ) -> SingleAgentEnv:
        """
        Create a robot environment with appropriate configuration.

        Args:
            config: Environment configuration. If None, creates default config.
            use_image_obs: Whether to enable image observations.
            peds_have_obstacle_forces: Whether pedestrians exert obstacle forces.
            reward_func: Custom reward function.
            debug: Enable debug mode.
            recording_enabled: Enable state recording.
            record_video: Enable video recording.
            video_path: Path for video output.
            video_fps: Video frame rate.
            **kwargs: Additional environment parameters.

        Returns:
            Configured robot environment.
        """
        # Create appropriate config if none provided
        if config is None:
            if use_image_obs:
                config = ImageRobotConfig()
            else:
                config = RobotSimulationConfig()

        # Update config flags
        config.use_image_obs = use_image_obs
        config.peds_have_obstacle_forces = peds_have_obstacle_forces

        # Import here to avoid circular imports
        if use_image_obs:
            from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage

            env_class = RobotEnvWithImage
        else:
            from robot_sf.gym_env.robot_env import RobotEnv

            env_class = RobotEnv

        return env_class(
            env_config=config,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
            **kwargs,
        )

    @staticmethod
    def create_pedestrian_env(
        config: Optional[PedestrianSimulationConfig] = None,
        *,
        robot_model=None,
        reward_func: Optional[Callable] = None,
        debug: bool = False,
        recording_enabled: bool = False,
        peds_have_obstacle_forces: bool = False,
        **kwargs,
    ) -> SingleAgentEnv:
        """
        Create a pedestrian environment for adversarial training.

        Args:
            config: Pedestrian environment configuration.
            robot_model: Pre-trained robot model for adversarial interaction.
            reward_func: Custom reward function for pedestrian.
            debug: Enable debug mode.
            recording_enabled: Enable state recording.
            peds_have_obstacle_forces: Whether pedestrians exert obstacle forces.
            **kwargs: Additional environment parameters.

        Returns:
            Configured pedestrian environment.
        """
        if config is None:
            config = PedestrianSimulationConfig()

        if reward_func is None:
            from robot_sf.gym_env.reward import simple_ped_reward

            reward_func = simple_ped_reward

        config.peds_have_obstacle_forces = peds_have_obstacle_forces

        from robot_sf.gym_env.pedestrian_env_refactored import RefactoredPedestrianEnv

        return RefactoredPedestrianEnv(
            config=config,
            robot_model=robot_model,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
            **kwargs,
        )

    @staticmethod
    def create_multi_robot_env(
        config: Optional[MultiRobotConfig] = None,
        *,
        num_robots: int = 2,
        reward_func: Optional[Callable] = None,
        debug: bool = False,
        **kwargs,
    ) -> MultiAgentEnv:
        """
        Create a multi-robot environment.

        Args:
            config: Multi-robot environment configuration.
            num_robots: Number of robots in the environment.
            reward_func: Custom reward function.
            debug: Enable debug mode.
            **kwargs: Additional environment parameters.

        Returns:
            Configured multi-robot environment.
        """
        if config is None:
            config = MultiRobotConfig(num_robots=num_robots)
        else:
            config.num_robots = num_robots

        from robot_sf.gym_env.multi_robot_env import MultiRobotEnv

        return MultiRobotEnv(
            env_config=config, reward_func=reward_func, debug=debug, num_robots=num_robots, **kwargs
        )

    @staticmethod
    def create_simple_env(**kwargs):
        """
        Create a simple robot environment for basic use cases.

        Returns:
            Simple robot environment with minimal configuration.
        """
        from robot_sf.gym_env.simple_robot_env import SimpleRobotEnv

        return SimpleRobotEnv(**kwargs)


# Convenience functions for common use cases
def make_robot_env(**kwargs) -> SingleAgentEnv:
    """Convenience function to create a standard robot environment."""
    return EnvironmentFactory.create_robot_env(**kwargs)


def make_image_robot_env(**kwargs) -> SingleAgentEnv:
    """Convenience function to create a robot environment with image observations."""
    return EnvironmentFactory.create_robot_env(use_image_obs=True, **kwargs)


def make_pedestrian_env(**kwargs) -> SingleAgentEnv:
    """Convenience function to create a pedestrian environment."""
    return EnvironmentFactory.create_pedestrian_env(**kwargs)


def make_multi_robot_env(num_robots: int = 2, **kwargs) -> MultiAgentEnv:
    """Convenience function to create a multi-robot environment."""
    return EnvironmentFactory.create_multi_robot_env(num_robots=num_robots, **kwargs)
