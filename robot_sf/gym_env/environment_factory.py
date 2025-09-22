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


# Convenience functions for common use cases (explicit signatures — Option A)
def make_robot_env(
    config: Optional[RobotSimulationConfig] = None,
    *,
    peds_have_obstacle_forces: bool = False,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
) -> SingleAgentEnv:
    """Create a standard robot environment (non-image observations).

    Parameters
    ----------
    config: RobotSimulationConfig | None
        Optional pre-configured robot simulation config (will be created if None).
    peds_have_obstacle_forces: bool
        Enable pedestrian obstacle force interactions.
    reward_func: Callable | None
        Optional custom reward function (defaults to simple_reward internally if None).
    debug: bool
        Enable debug visualization (required for live rendering & frame capture).
    recording_enabled: bool
        Enable state recording list for later inspection.
    record_video: bool
        Enable video recording (requires debug=True for real frames).
    video_path: str | None
        Output path for recorded video.
    video_fps: float | None
        Frames per second for recorded video.
    """
    return EnvironmentFactory.create_robot_env(
        config=config,
        use_image_obs=False,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        record_video=record_video,
        video_path=video_path,
        video_fps=video_fps,
    )


def make_image_robot_env(
    config: Optional[ImageRobotConfig] = None,
    *,
    peds_have_obstacle_forces: bool = False,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
) -> SingleAgentEnv:
    """Create a robot environment with image observations.

    Mirrors `make_robot_env` but sets image observation mode.
    """
    return EnvironmentFactory.create_robot_env(
        config=config,  # type: ignore[arg-type]
        use_image_obs=True,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        record_video=record_video,
        video_path=video_path,
        video_fps=video_fps,
    )


def make_pedestrian_env(
    config: Optional[PedestrianSimulationConfig] = None,
    *,
    robot_model=None,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    peds_have_obstacle_forces: bool = False,
) -> SingleAgentEnv:
    """Create a pedestrian (adversarial) environment.

    Parameters mirror `EnvironmentFactory.create_pedestrian_env` explicitly for discoverability.
    """
    return EnvironmentFactory.create_pedestrian_env(
        config=config,
        robot_model=robot_model,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
    )


def make_multi_robot_env(
    num_robots: int = 2,
    *,
    config: Optional[MultiRobotConfig] = None,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
) -> MultiAgentEnv:
    """Create a multi-robot environment.

    Parameters
    ----------
    num_robots: int
        Number of robots to include.
    config: MultiRobotConfig | None
        Optional pre-configured multi-robot config.
    reward_func: Callable | None
        Optional custom reward.
    debug: bool
        Enable debug visualization.
    """
    return EnvironmentFactory.create_multi_robot_env(
        config=config,
        num_robots=num_robots,
        reward_func=reward_func,
        debug=debug,
    )
