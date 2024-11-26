from dataclasses import dataclass
from typing import Dict
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
    steps_per_second: float
    avg_step_time_ms: float
    total_episodes: int
    system_info: Dict
    config_hash: str

    def to_dict(self) -> Dict:
        return {
            "steps_per_second": self.steps_per_second,
            "avg_step_time_ms": self.avg_step_time_ms,
            "total_episodes": self.total_episodes,
            "system_info": self.system_info,
            "config_hash": self.config_hash,
        }


def run_standardized_benchmark(num_steps: int = 10000) -> BenchmarkMetrics:
    """Run a standardized simulation benchmark."""
    # Fixed configuration
    env_config = EnvSettings()
    env_config.sim_config.difficulty = 2
    env_config.sim_config.ped_density_by_difficulty = [0.02, 0.04, 0.08]

    # Initialize environment
    env = RobotEnv(env_config)
    model = PPO.load("./model/run_043", env=env)

    # Track timing
    step_times = []
    episodes = 0
    obs = env.reset()

    for _ in range(num_steps):
        start = time.perf_counter()

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        step_times.append(time.perf_counter() - start)

        if done:
            episodes += 1
            obs = env.reset()

    # Calculate metrics
    avg_step_time = np.mean(step_times)
    steps_per_sec = 1.0 / avg_step_time

    # System info
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
    }

    # Generate config hash
    config_str = str(env_config.sim_config.__dict__)
    config_hash = hash(config_str)

    return BenchmarkMetrics(
        steps_per_second=steps_per_sec,
        avg_step_time_ms=avg_step_time * 1000,
        total_episodes=episodes,
        system_info=system_info,
        config_hash=str(config_hash),
    )


def save_benchmark_results(
    benchmark_metrics: BenchmarkMetrics, baseline_file: str = "benchmark_baseline.json"
):
    """Save benchmark results and compare to baseline."""

    # Load baseline if exists
    try:
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline = json.load(f)
    except FileNotFoundError:
        baseline = None

    # Current results
    results = {"timestamp": time.time(), "metrics": benchmark_metrics.to_dict()}

    # Calculate relative performance
    if baseline and baseline["metrics"]["config_hash"] == benchmark_metrics.config_hash:
        relative_perf = (
            benchmark_metrics.steps_per_second / baseline["metrics"]["steps_per_second"]
        )
        results["relative_performance"] = relative_perf

    # Save results
    with open(
        f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)


def create_baseline():
    """Create a baseline benchmark for future comparisons."""
    baseline_metrics = run_standardized_benchmark()

    with open("benchmark_baseline.json", "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": time.time(), "metrics": baseline_metrics.to_dict()},
            f,
            indent=2,
        )


if __name__ == "__main__":
    logger.info("Running standardized benchmark...")
    # Run benchmark
    metrics = run_standardized_benchmark()

    logger.info(f"Steps per second: {metrics.steps_per_second:.2f}")
    logger.info(f"Average step time: {metrics.avg_step_time_ms:.2f} ms")
    logger.info(f"Total episodes: {metrics.total_episodes}")

    logger.info("Saving benchmark results...")
    # Save and compare to baseline
    save_benchmark_results(metrics)
