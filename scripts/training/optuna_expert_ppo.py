"""Optuna study runner for expert PPO training.

This script sweeps PPO hyperparameters for the expert imitation pipeline using the
same configuration format as ``train_expert_ppo.py``. Trials run shorter training
loops by default so the study can iterate quickly; adjust the CLI flags for full
length sweeps.
"""

from __future__ import annotations

import argparse
import copy
from datetime import UTC, datetime
from pathlib import Path

import optuna
from loguru import logger

from robot_sf.common import ensure_seed_tuple
from robot_sf.training.imitation_config import EvaluationSchedule
from train_expert_ppo import _resolve_num_envs, load_expert_training_config, run_expert_training


def _suggest_ppo_hyperparams(trial: optuna.Trial, *, max_batch_size: int) -> dict[str, object]:
    """Sample PPO hyperparameters for the expert training pipeline."""
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    batch_size = min(batch_size, max_batch_size)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_categorical("n_epochs", [2, 4, 8]),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.02, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "target_kl": trial.suggest_float("target_kl", 0.01, 0.03),
        "gamma": trial.suggest_categorical("gamma", [0.98, 0.99, 0.995]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
    }


def _resolve_metric_direction(metric: str) -> tuple[str, str]:
    """Return (metric_name, direction) for Optuna."""
    normalized = metric.strip().lower()
    if normalized in {"collision_rate", "comfort_exposure"}:
        return normalized, "minimize"
    return normalized, "maximize"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Optuna sweep for expert PPO training.")
    parser.add_argument("--config", required=True, help="Base expert PPO config YAML.")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument(
        "--metric",
        default="snqi",
        help="Metric to optimize (e.g., snqi, success_rate, collision_rate).",
    )
    parser.add_argument(
        "--trial-timesteps",
        type=int,
        default=1_000_000,
        help="Timesteps per trial (default: 1M).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=250_000,
        help="Evaluation cadence in steps during trials.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optional study name (defaults to policy_id + timestamp).",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (e.g., sqlite:///path/to/db).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Optuna sampler seed (default: 123).",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B for all trials (recommended).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic seeds for trial evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Optuna sweep."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    base_config = load_expert_training_config(config_path)
    metric_name, direction = _resolve_metric_direction(args.metric)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    study_name = args.study_name or f"{base_config.policy_id}_optuna_{timestamp}"
    storage = args.storage
    if storage is None:
        storage_path = (
            Path("output/benchmarks/ppo_imitation/hparam_opt") / f"{study_name}.db"
        )
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path}"

    logger.info(
        "Starting Optuna study '{}' direction={} metric={} storage={}",
        study_name,
        direction,
        metric_name,
        storage,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        """Single Optuna objective run."""
        config = copy.deepcopy(base_config)
        config.policy_id = f"{base_config.policy_id}_optuna_{trial.number:03d}"
        config.best_checkpoint_metric = metric_name
        config.total_timesteps = int(args.trial_timesteps)
        config.evaluation = EvaluationSchedule(
            frequency_episodes=config.evaluation.frequency_episodes,
            evaluation_episodes=int(args.eval_episodes),
            hold_out_scenarios=config.evaluation.hold_out_scenarios,
            step_schedule=((None, int(args.eval_every)),),
        )
        if args.deterministic:
            config.randomize_seeds = False
            if not config.seeds:
                config.seeds = ensure_seed_tuple([0, 1, 2])
        if args.disable_wandb:
            wandb_cfg = config.tracking.get("wandb", {})
            if isinstance(wandb_cfg, dict):
                wandb_cfg["enabled"] = False
                config.tracking["wandb"] = wandb_cfg

        num_envs = _resolve_num_envs(config)
        max_batch = max(1, num_envs * 512)
        config.ppo_hyperparams = _suggest_ppo_hyperparams(trial, max_batch_size=max_batch)

        result = run_expert_training(config, config_path=config_path, dry_run=False)
        best = result.best_checkpoint
        metric_value = best.metrics.get(metric_name) if best is not None else None
        if metric_value is None:
            aggregate = result.metrics.get(metric_name)
            metric_value = aggregate.mean if aggregate is not None else None
        if metric_value is None:
            raise ValueError(f"Metric '{metric_name}' not found in training output.")
        trial.set_user_attr("policy_id", config.policy_id)
        best_path = (
            best.checkpoint_path if best is not None else result.expert_artifact.checkpoint_path
        )
        trial.set_user_attr("best_checkpoint", str(best_path))
        return float(metric_value)

    study.optimize(objective, n_trials=int(args.trials))
    best = study.best_trial
    logger.success(
        "Optuna study complete: best_value={} best_trial={} params={}",
        best.value,
        best.number,
        best.params,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
