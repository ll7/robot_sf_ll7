"""TODO docstring. Document this module."""

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import psutil
from loguru import logger

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark runs"""

    steps_per_second: float
    avg_step_time_ms: float
    total_episodes: int
    system_info: dict
    config_hash: str
    observation_space_info: dict
    used_random_actions: bool = False
    env_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return {
            "steps_per_second": self.steps_per_second,
            "avg_step_time_ms": self.avg_step_time_ms,
            "total_episodes": self.total_episodes,
            "system_info": self.system_info,
            "config_hash": self.config_hash,
            "observation_space_info": self.observation_space_info,
            "used_random_actions": self.used_random_actions,
            "env_info": self.env_info,
        }


def run_standardized_benchmark(
    num_steps: int = 2_000,
    model_path: str | None = "./model/run_043",
) -> BenchmarkMetrics:
    """Run a standardized simulation benchmark.

    Args:
        num_steps: Number of simulation steps to run
        model_path: Path to the model file. If None, uses random actions
    """
    # Fixed configuration
    env_config = EnvSettings()
    env_config.sim_config.difficulty = 2
    env_config.sim_config.ped_density_by_difficulty = [0.02, 0.04, 0.08]

    # Initialize environment
    env = RobotEnv(env_config)

    # Record observation space info
    from gymnasium import spaces as _spaces  # local import to avoid global dependency for types

    obs_dict = cast("_spaces.Dict", env.observation_space)
    drive_space = cast("_spaces.Box", obs_dict["drive_state"])  # type: ignore[index]
    rays_space = cast("_spaces.Box", obs_dict["rays"])  # type: ignore[index]

    obs_space_info = {
        "drive_state_shape": drive_space.shape,
        "rays_shape": rays_space.shape,
        "drive_state_bounds": {
            "low": drive_space.low.tolist(),
            "high": drive_space.high.tolist(),
        },
    }

    # Try to load model, fall back to random actions if fails
    used_random_actions = False
    if model_path:
        try:
            model = load_trained_policy(model_path)
            logger.info("Successfully loaded model")
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to load model: {e}")
            logger.info("Falling back to random actions")
            model = None
            used_random_actions = True
    else:
        model = None
        used_random_actions = True

    # Track timing
    step_times = []
    episodes = 0
    obs, _ = env.reset()

    logger.info("Starting benchmark run...")
    for i in range(num_steps):
        start = time.perf_counter()

        # Get action from model or random
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, _, done, _, _ = env.step(action)

        step_time = time.perf_counter() - start
        step_times.append(step_time)

        if done:
            episodes += 1
            obs, _ = env.reset()

        if i % 1000 == 0:
            logger.debug(f"Completed {i}/{num_steps} steps")

    # Calculate metrics
    avg_step_time = float(np.mean(step_times))
    steps_per_sec = float(1.0 / avg_step_time) if avg_step_time > 0 else 0.0

    env.close()
    logger.info("Benchmark run complete. env closed. return metrics")

    # System info
    # Some psutil installations/platforms may not expose cpu_freq; guard the attribute
    if hasattr(psutil, "cpu_freq"):
        cpu_freq_obj = psutil.cpu_freq()
    else:
        cpu_freq_obj = None
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_freq": cpu_freq_obj._asdict() if cpu_freq_obj is not None else None,
    }

    # Environment info
    env_info = {
        "difficulty": env_config.sim_config.difficulty,
        "ped_density_by_difficulty": env_config.sim_config.ped_density_by_difficulty,
        "map_name": list(env_config.map_pool.map_defs.keys()),
    }

    # Generate config hash
    config_str = str(env_config.sim_config.__dict__)
    config_hash = str(hash(config_str))

    return BenchmarkMetrics(
        steps_per_second=steps_per_sec,
        avg_step_time_ms=avg_step_time * 1000,
        total_episodes=episodes,
        system_info=system_info,
        config_hash=config_hash,
        observation_space_info=obs_space_info,
        used_random_actions=used_random_actions,
        env_info=env_info,
    )


def _default_benchmark_results_path() -> Path:
    """Return the canonical benchmark results path under the artifact tree."""

    ensure_canonical_tree(categories=("benchmarks",))
    return get_artifact_category_path("benchmarks") / "benchmark_results.json"


def save_benchmark_results(
    results: BenchmarkMetrics,
    json_file: str | Path | None = None,
    append: bool = True,
):
    """
    Save benchmark results to a JSON file.

    Parameters:
    results (BenchmarkMetrics): The benchmark metrics to save.
    json_file (str | Path | None): Optional override path for the JSON output. When
        omitted the canonical artifact location under ``output/benchmarks`` is used.
    append (bool): If True, append the results to the existing file. If False,
        overwrite the file. Defaults to True.

    Raises:
    FileNotFoundError: If the file does not exist and append is True, a new file will be created.
    """
    target_path = Path(json_file) if json_file is not None else _default_benchmark_results_path()

    if append:
        try:
            with target_path.open("r+", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                data.append(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "metrics": results.to_dict(),
                    },
                )
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            logger.info(f"Appended results to {target_path}")
        except FileNotFoundError:
            with target_path.open("w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "metrics": results.to_dict(),
                        },
                    ],
                    f,
                    indent=2,
                )
            logger.warning(f"Appending failed. Created new file {target_path}")
    else:
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "metrics": results.to_dict(),
                    },
                ],
                f,
                indent=2,
            )
        logger.info(f"Saved results to {target_path}")


if __name__ == "__main__":
    logger.info("Running standardized benchmark...")

    # Run benchmark
    metrics = run_standardized_benchmark(model_path="model/ppo_model_retrained_10m_2025-02-01.zip")

    logger.info(f"Steps per second: {metrics.steps_per_second:.2f}")
    logger.info(f"Average step time: {metrics.avg_step_time_ms:.2f} ms")
    logger.info(f"Total episodes: {metrics.total_episodes}")
    logger.info(f"Used random actions: {metrics.used_random_actions}")

    # Save results
    save_benchmark_results(metrics)
