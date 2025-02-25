from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import json
import platform
import psutil
import numpy as np
from loguru import logger
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings
from stable_baselines3 import PPO


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark runs"""

    steps_per_second: float
    avg_step_time_ms: float
    total_episodes: int
    system_info: Dict
    config_hash: str
    observation_space_info: Dict
    used_random_actions: bool = False
    env_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
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
    num_steps: int = 2_000, model_path: Optional[str] = "./model/run_043"
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
    obs_space_info = {
        "drive_state_shape": env.observation_space["drive_state"].shape,
        "rays_shape": env.observation_space["rays"].shape,
        "drive_state_bounds": {
            "low": env.observation_space["drive_state"].low.tolist(),
            "high": env.observation_space["drive_state"].high.tolist(),
        },
    }

    # Try to load model, fall back to random actions if fails
    used_random_actions = False
    if model_path:
        try:
            model = PPO.load(model_path, env=env)
            logger.info("Successfully loaded model")
        except ValueError as e:
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
    avg_step_time = np.mean(step_times)
    steps_per_sec = 1.0 / avg_step_time

    env.close()
    logger.info("Benchmark run complete. env closed. return metrics")

    # System info
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
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


def save_benchmark_results(
    results: BenchmarkMetrics,
    json_file: str = "benchmark_results.json",
    append: bool = True,
):
    """
    Save benchmark results to a JSON file.

    Parameters:
    results (BenchmarkMetrics): The benchmark metrics to save.
    json_file (str): The path to the JSON file where results will be saved.
        Defaults to "benchmark_results.json".
    append (bool): If True, append the results to the existing file.
        If False, overwrite the file. Defaults to True.

    Raises:
    FileNotFoundError: If the file does not exist and append is True, a new file will be created.
    """
    if append:
        try:
            with open(json_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                data.append(
                    {
                        "timestamp": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime()
                        ),
                        "metrics": results.to_dict(),
                    }
                )
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            logger.info(f"Appended results to {json_file}")
        except FileNotFoundError:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "timestamp": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime()
                            ),
                            "metrics": results.to_dict(),
                        }
                    ],
                    f,
                    indent=2,
                )
            logger.warning(f"Appending failed. Created new file {json_file}")
    else:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "timestamp": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime()
                        ),
                        "metrics": results.to_dict(),
                    }
                ],
                f,
                indent=2,
            )
        logger.info(f"Saved results to {json_file}")


if __name__ == "__main__":
    logger.info("Running standardized benchmark...")

    # Run benchmark
    metrics = run_standardized_benchmark(
        model_path="model/ppo_model_retrained_10m_2025-02-01.zip"
    )

    logger.info(f"Steps per second: {metrics.steps_per_second:.2f}")
    logger.info(f"Average step time: {metrics.avg_step_time_ms:.2f} ms")
    logger.info(f"Total episodes: {metrics.total_episodes}")
    logger.info(f"Used random actions: {metrics.used_random_actions}")

    # Save results
    save_benchmark_results(metrics)
