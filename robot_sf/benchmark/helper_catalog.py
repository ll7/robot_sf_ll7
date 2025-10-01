"""Benchmark helper catalog for reusable environment setup and episode execution.

This module provides helper functions extracted from examples and scripts for
setting up environments, loading policies, and running benchmark episodes.
"""

from pathlib import Path
from typing import Any, cast

from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig


def prepare_classic_env(
    config_override: RobotSimulationConfig | None = None,
) -> tuple[Any, list[int]]:
    """Initialize a classic interaction environment using factory functions.

    Args:
        config_override: Optional configuration override for environment setup

    Returns:
        Tuple of (configured environment, deterministic seed list)

    Raises:
        RuntimeError: If environment creation fails
    """
    try:
        # Use provided config or create default
        config = config_override or RobotSimulationConfig()

        # Apply classic interaction defaults
        config.render_scaling = 20  # Common scaling for classic interactions

        env = make_robot_env(
            config=config,
            debug=True,
        )

        # Generate deterministic seed list for reproducibility
        seeds = [42, 123, 456, 789, 999]  # Default seeds from classic examples

        logger.info("Classic environment prepared successfully")
        return env, seeds

    except Exception as e:
        logger.error(f"Failed to prepare classic environment: {e}")
        raise RuntimeError(f"Environment preparation failed: {e}") from e


def load_trained_policy(path: str):
    """Load and cache a trained PPO policy from the specified path.

    This function implements caching based on absolute path to ensure
    correct behavior when paths change during testing.

    Args:
        path: Path to the trained model file

    Returns:
        Loaded PPO policy model

    Raises:
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If stable_baselines3 is not available
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise RuntimeError(
            "stable_baselines3 PPO import failed. Install with 'uv add stable-baselines3' to use this helper."
        ) from e

    abs_path = str(Path(path).resolve())
    cache_map = getattr(load_trained_policy, "_cache_map", None)
    if cache_map is None:
        cache_map = {}
    # Store cache on the function __dict__ to avoid protected-member access lint
    cast(Any, load_trained_policy).__dict__["_cache_map"] = cache_map

    if abs_path in cache_map:
        logger.debug(f"Loading cached policy from {abs_path}")
        return cache_map[abs_path]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Download or place the pre-trained PPO model at this path. "
            "See docs/dev/issues/classic-interactions-ppo/ for guidance."
        )

    logger.info(f"Loading policy from {path}")
    model = PPO.load(path)
    cache_map[abs_path] = model
    return model


def run_episodes_with_recording(
    env,
    policy,
    seeds: list[int],
    record: bool,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Execute episodes with optional recording and return structured summaries.

    Args:
        env: Configured environment instance
        policy: Trained policy model
        seeds: List of seeds for episode execution
        record: Whether to enable recording
        output_dir: Directory for recording outputs

    Returns:
        List of episode summaries with metadata

    Raises:
        RuntimeError: If episode execution fails
    """
    results = []

    try:
        if record:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Recording enabled, output directory: {output_dir}")

        for i, seed in enumerate(seeds):
            logger.debug(f"Running episode {i + 1}/{len(seeds)} with seed {seed}")

            # Reset environment with seed
            obs, _ = env.reset(seed=seed)
            episode_reward = 0.0
            steps = 0
            done = False

            while not done:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            # Create episode summary
            episode_summary = {
                "seed": seed,
                "episode_reward": float(episode_reward),
                "steps": steps,
                "recording_enabled": record,
                "output_dir": str(output_dir) if record else None,
            }

            results.append(episode_summary)
            logger.debug(f"Episode {i + 1} completed: reward={episode_reward:.2f}, steps={steps}")

        if record and not results:
            logger.warning("Recording was enabled but no episodes were executed")

        logger.info(f"Completed {len(results)} episodes")
        return results

    except Exception as e:
        logger.error(f"Episode execution failed: {e}")
        raise RuntimeError(f"Failed to run episodes: {e}") from e
