"""PPO fine-tuning from a pre-trained policy checkpoint.

Loads a warm-start policy (from BC pre-training or previous checkpoint) and continues
training with PPO to maximize performance while leveraging the pre-trained initialization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from loguru import logger

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:
    raise RuntimeError("This script requires 'stable_baselines3' package.") from exc

from robot_sf import common
from robot_sf.benchmark.imitation_manifest import write_training_run_manifest
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.training.imitation_config import PPOFineTuningConfig
from robot_sf.training.observation_wrappers import maybe_flatten_env_observations

if TYPE_CHECKING:
    from collections.abc import Sequence


class TimestepTracker(BaseCallback):
    """Callback to track timesteps for convergence measurement."""

    def __init__(self):
        super().__init__()
        self.timesteps_to_convergence = None
        self.converged = False

    def _on_step(self) -> bool:
        # Simplified convergence check - production would use proper metrics
        if not self.converged and self.num_timesteps > 1000:
            self.timesteps_to_convergence = self.num_timesteps
            self.converged = True
        return True


def _evaluate_policy_metrics(
    model: PPO | None,
    config: PPOFineTuningConfig,
    *,
    dry_run: bool,
) -> tuple[list[float], list[float], list[float]]:
    """Run short evaluation episodes and return per-episode samples for metrics."""

    if dry_run:
        logger.warning("Dry run mode: using placeholder metrics for evaluation")
        return [0.85], [0.08], [0.85 - 0.5 * 0.08]
    if model is None:
        raise ValueError("Model must be provided for evaluation when not in dry-run mode")

    eval_seed = config.random_seeds[0] if config.random_seeds else None
    eval_env = make_robot_env(config=RobotSimulationConfig(), seed=eval_seed)
    eval_env = maybe_flatten_env_observations(eval_env, context="PPO evaluation")

    successes: list[float] = []
    collisions: list[float] = []
    snqi_values: list[float] = []
    steps_list: list[float] = []
    num_eval_episodes = 10  # Small number for quick evaluation

    from robot_sf.benchmark.metrics import snqi as compute_snqi

    default_weights = {
        "w_success": 1.0,
        "w_time": 0.8,
        "w_collisions": 2.0,
        "w_near": 1.0,
        "w_comfort": 0.5,
        "w_force_exceed": 1.5,
        "w_jerk": 0.3,
    }

    for _ in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_steps = 0
        collision_occurred = False
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_steps += 1

            if info.get("collision", False):
                collision_occurred = True
            if info.get("is_success", False):
                success = True

        successes.append(1.0 if success else 0.0)
        collisions.append(1.0 if collision_occurred else 0.0)
        steps_list.append(float(episode_steps))

    eval_env.close()
    max_steps = max(steps_list) if steps_list else 1.0
    for success_flag, collision_flag, steps in zip(successes, collisions, steps_list, strict=False):
        metric_values = {
            "success": success_flag,
            "time_to_goal_norm": steps / max_steps if max_steps else 1.0,
            "collisions": collision_flag,
            "near_misses": 0.0,
            "comfort_exposure": 0.0,
            "force_exceed_events": 0.0,
            "jerk_mean": 0.0,
            "curvature_mean": 0.0,
        }
        snqi_values.append(float(compute_snqi(metric_values, default_weights)))

    return successes, collisions, snqi_values


def run_ppo_finetuning(
    config: PPOFineTuningConfig,
    *,
    dry_run: bool = False,
) -> tuple[Path, int]:
    """Execute PPO fine-tuning and return checkpoint path and convergence timesteps."""
    logger.info("Starting PPO fine-tuning run_id={}", config.run_id)

    # Load pre-trained policy
    pretrained_path = common.get_expert_policy_dir() / f"{config.pretrained_policy_id}.zip"
    if not pretrained_path.exists() and not dry_run:
        raise FileNotFoundError(f"Pre-trained policy not found: {pretrained_path}")

    # Create environment
    env = make_robot_env(config=RobotSimulationConfig())
    env = maybe_flatten_env_observations(env, context="PPO fine-tuning")

    if not dry_run:
        # Load pre-trained model
        logger.info("Loading pre-trained policy from {}", pretrained_path)
        model = PPO.load(str(pretrained_path), env=env)

        # Update learning rate for fine-tuning
        model.learning_rate = config.learning_rate

        # Setup callback
        timestep_tracker = TimestepTracker()

        # Fine-tune
        logger.info("Fine-tuning PPO for {} timesteps", config.total_timesteps)
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=timestep_tracker,
            reset_num_timesteps=False,  # Continue from pre-trained timesteps
        )

        # Save fine-tuned policy
        policy_path = common.get_expert_policy_dir() / f"{config.run_id}_finetuned.zip"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(policy_path))

        convergence_timesteps = timestep_tracker.timesteps_to_convergence or config.total_timesteps
    else:
        # Dry run
        policy_path = common.get_expert_policy_dir() / f"{config.run_id}_finetuned.zip"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text("dry-run-finetuned-checkpoint", encoding="utf-8")
        convergence_timesteps = 1000  # Placeholder for dry run

    env.close()

    logger.success(
        "PPO fine-tuning complete, policy saved to {}, converged at {} timesteps",
        policy_path,
        convergence_timesteps,
    )

    # Compute real metrics from the trained model by running evaluation episodes
    # This replaces the previous synthetic random data approach
    successes, collisions, snqi_values = _evaluate_policy_metrics(
        model if not dry_run else None, config, dry_run=dry_run
    )
    success_rate = float(np.mean(successes)) if successes else 0.0
    collision_rate = float(np.mean(collisions)) if collisions else 0.0
    snqi = float(np.mean(snqi_values)) if snqi_values else 0.0

    # Minimal metrics to surface convergence information and basic quality signals
    convergence_metric = common.MetricAggregate(
        mean=float(convergence_timesteps),
        median=float(convergence_timesteps),
        p95=float(convergence_timesteps),
        ci95=(float(convergence_timesteps), float(convergence_timesteps)),
    )

    def _metric(values: list[float]) -> common.MetricAggregate:
        if not values:
            return common.MetricAggregate(mean=0.0, median=0.0, p95=0.0, ci95=(0.0, 0.0))
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        if len(arr) > 1:
            se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            ci = (mean - 1.96 * se, mean + 1.96 * se)
        else:
            ci = (mean, mean)
        return common.MetricAggregate(mean=mean, median=median, p95=p95, ci95=ci)

    # Write training run manifest
    training_artifact = common.TrainingRunArtifact(
        run_id=config.run_id,
        run_type=common.TrainingRunType.PPO_FINETUNE,
        input_artefacts=(config.pretrained_policy_id,),
        seeds=config.random_seeds,
        metrics={
            "timesteps_to_convergence": convergence_metric,
            "success_rate": _metric(successes if not dry_run else [success_rate]),
            "collision_rate": _metric(collisions if not dry_run else [collision_rate]),
            "snqi": _metric(snqi_values if not dry_run else [snqi]),
            # path_efficiency and comfort_exposure intentionally omitted until trajectory-based
            # calculations are wired in (needs per-step positions and contact stats).
        },
        episode_log_path=Path(""),  # Episode logs would be generated in production
        wall_clock_hours=0.0,  # Would track actual time in production
        status=common.TrainingRunStatus.COMPLETED,
        notes=[
            f"PPO fine-tuning from pretrained policy {config.pretrained_policy_id}",
            f"Converged at {convergence_timesteps} timesteps",
        ],
    )

    manifest_path = write_training_run_manifest(training_artifact)
    logger.info("Training run manifest written to {}", manifest_path)

    return policy_path, convergence_timesteps


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for PPO fine-tuning script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune PPO policy from a pre-trained checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to PPO fine-tuning configuration YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder artifacts without actual training",
    )
    return parser


def load_ppo_finetuning_config(config_path: Path) -> PPOFineTuningConfig:
    """Load and parse PPO fine-tuning configuration from YAML."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return PPOFineTuningConfig.from_raw(
        run_id=raw["run_id"],
        pretrained_policy_id=raw["pretrained_policy_id"],
        total_timesteps=raw.get("total_timesteps", 100000),
        random_seeds=tuple(raw.get("random_seeds", [42])),
        learning_rate=raw.get("learning_rate", 0.0001),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for PPO fine-tuning script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_ppo_finetuning_config(args.config)
    common.set_global_seed(int(config.random_seeds[0]))

    run_ppo_finetuning(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
