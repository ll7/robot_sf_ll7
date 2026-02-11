"""Optuna study runner for expert PPO training.

This script sweeps PPO hyperparameters for the expert imitation pipeline using the
same configuration format as ``train_expert_ppo.py``. Trials run shorter training
loops by default so the study can iterate quickly; adjust the CLI flags for full
length sweeps.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
from loguru import logger
from optuna.trial import TrialState
from sqlalchemy.engine import make_url

_OBJECTIVE_MODES = ("best_checkpoint", "final_eval", "last_n_mean", "auc")
_LOG_LEVEL_CHOICES = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
_DEFAULT_STORAGE_ROOT = Path("output/benchmarks/ppo_imitation/hparam_opt").resolve()
_ALLOWED_SQLITE_ROOT = Path("output").resolve()


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


def _sanitize_storage_filename(raw: str, *, fallback: str = "optuna_study") -> str:
    """Return a filename-safe study token for sqlite database paths."""
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw.strip())
    sanitized = sanitized.strip("._")
    return sanitized or fallback


def _resolve_sqlite_database_path(database: str, *, allowed_root: Path | None = None) -> Path:
    """Resolve sqlite database path and optionally enforce an allowed root."""
    resolved = Path(database).expanduser()
    resolved = (
        (Path.cwd() / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()
    )
    if allowed_root is not None and not resolved.is_relative_to(allowed_root):
        raise ValueError(
            f"Refusing sqlite storage path outside allowed root '{allowed_root}': {resolved}",
        )
    return resolved


def _ensure_sqlite_storage_parent(storage: str, *, allowed_root: Path | None = None) -> None:
    """Create sqlite parent directory while rejecting paths outside an allowed root."""
    try:
        url = make_url(storage)
    except Exception:
        return
    if url.get_backend_name() != "sqlite":
        return
    database = url.database
    if not database or database == ":memory:":
        return
    resolved_database = _resolve_sqlite_database_path(database, allowed_root=allowed_root)
    resolved_database.parent.mkdir(parents=True, exist_ok=True)


def _configure_optuna_verbosity(log_level: str) -> None:
    """Align Optuna's logger verbosity with the selected console log level."""
    normalized = log_level.upper()
    if normalized in {"TRACE", "DEBUG"}:
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
    elif normalized in {"INFO", "SUCCESS"}:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    elif normalized == "WARNING":
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.ERROR)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Optuna sweep for expert PPO training.")
    parser.add_argument("--config", required=True, help="Base expert PPO config YAML.")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument(
        "--metric",
        default="eval_episode_return",
        help="Metric to optimize (e.g., eval_episode_return, snqi, success_rate, collision_rate).",
    )
    parser.add_argument(
        "--objective-mode",
        default="last_n_mean",
        choices=_OBJECTIVE_MODES,
        help=(
            "How to reduce evaluation history to a scalar objective: "
            "best_checkpoint|final_eval|last_n_mean|auc."
        ),
    )
    parser.add_argument(
        "--objective-window",
        type=int,
        default=3,
        help="Window size for --objective-mode=last_n_mean.",
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
        help="Disable W&B for all trials (W&B is enabled by default).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic seeds for trial evaluation.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=_LOG_LEVEL_CHOICES,
        help="Console log level (default: WARNING).",
    )
    return parser


def _build_trial_config(
    trial: optuna.Trial,
    *,
    base_config,
    metric_name: str,
    objective_mode: str,
    study_name: str,
    args: argparse.Namespace,
    resolve_num_envs_fn: Any,
    evaluation_schedule_cls: Any,
):
    """Create one trial-specific training configuration."""
    config = copy.deepcopy(base_config)
    config.policy_id = f"{base_config.policy_id}_optuna_{trial.number:03d}"
    config.best_checkpoint_metric = metric_name
    config.total_timesteps = int(args.trial_timesteps)
    config.evaluation = evaluation_schedule_cls(
        frequency_episodes=config.evaluation.frequency_episodes,
        evaluation_episodes=int(args.eval_episodes),
        hold_out_scenarios=config.evaluation.hold_out_scenarios,
        step_schedule=((None, int(args.eval_every)),),
    )
    if args.deterministic:
        from robot_sf.common import ensure_seed_tuple

        config.randomize_seeds = False
        if not config.seeds:
            config.seeds = ensure_seed_tuple([0, 1, 2])
    tracking = dict(config.tracking or {})
    wandb_cfg_raw = tracking.get("wandb")
    wandb_cfg = dict(wandb_cfg_raw) if isinstance(wandb_cfg_raw, dict) else {}
    if args.disable_wandb:
        wandb_cfg["enabled"] = False
    else:
        wandb_cfg.setdefault("enabled", True)
    if bool(wandb_cfg.get("enabled", False)):
        wandb_cfg.setdefault("project", "robot_sf")
        wandb_cfg.setdefault("group", study_name)
        wandb_cfg.setdefault("job_type", f"optuna-trial-{objective_mode}")
        wandb_cfg.setdefault("name", config.policy_id)
        wandb_cfg.setdefault(
            "tags",
            [
                "optuna",
                f"study:{study_name}",
                f"objective_mode:{objective_mode}",
                f"metric:{metric_name}",
            ],
        )
        wandb_cfg.setdefault(
            "notes",
            (
                f"Optuna trial for study '{study_name}' "
                f"(objective_mode={objective_mode}, metric={metric_name})"
            ),
        )
    tracking["wandb"] = wandb_cfg
    config.tracking = tracking
    num_envs = resolve_num_envs_fn(config)
    max_batch = max(1, num_envs * 512)
    config.ppo_hyperparams = _suggest_ppo_hyperparams(trial, max_batch_size=max_batch)
    return config


def _resolve_trial_metric(
    *,
    result,
    metric_name: str,
    objective_mode: str,
    objective_window: int,
    load_episode_records_fn: Any,
    eval_metric_series_fn: Any,
    objective_from_series_fn: Any,
) -> tuple[float | None, list[tuple[int, float]]]:
    """Resolve the optimization scalar from training outputs."""
    best = result.best_checkpoint
    metric_value: float | None = None
    if objective_mode == "best_checkpoint":
        metric_value = best.metrics.get(metric_name) if best is not None else None

    episode_records = load_episode_records_fn(result.training_run_artifact.episode_log_path)
    series = eval_metric_series_fn(episode_records, metric_name=metric_name)
    derived_metric = objective_from_series_fn(
        series,
        mode=objective_mode,
        window=objective_window,
    )
    if derived_metric is not None:
        metric_value = float(derived_metric)

    if metric_value is None:
        aggregate = result.metrics.get(metric_name)
        metric_value = aggregate.mean if aggregate is not None else None
    return metric_value, series


def _register_trial_metadata(
    trial: optuna.Trial,
    *,
    config,
    result,
    metric_name: str,
    objective_mode: str,
    objective_window: int,
    series: list[tuple[int, float]],
) -> None:
    """Persist useful metadata on the trial for downstream analysis."""
    best = result.best_checkpoint
    trial.set_user_attr("policy_id", config.policy_id)
    trial.set_user_attr("objective_mode", objective_mode)
    trial.set_user_attr("objective_window", objective_window)
    trial.set_user_attr("objective_metric", metric_name)
    if series:
        trial.set_user_attr("objective_eval_points", len(series))
        trial.set_user_attr("objective_eval_last_step", int(series[-1][0]))
    best_path = best.checkpoint_path if best is not None else result.expert_artifact.checkpoint_path
    trial.set_user_attr("best_checkpoint", str(best_path))


def main(argv: list[str] | None = None) -> int:
    """Run the Optuna sweep."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    previous_loguru_level = os.environ.get("LOGURU_LEVEL")

    try:
        log_level = str(args.log_level).upper()
        os.environ["LOGURU_LEVEL"] = log_level
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        _configure_optuna_verbosity(log_level)

        from train_expert_ppo import (
            _resolve_num_envs,
            load_expert_training_config,
            run_expert_training,
        )

        from robot_sf.training.imitation_config import EvaluationSchedule
        from robot_sf.training.optuna_objective import (
            eval_metric_series,
            load_episode_records,
            objective_from_series,
        )

        config_path = Path(args.config).resolve()
        base_config = load_expert_training_config(config_path)
        metric_name, direction = _resolve_metric_direction(args.metric)
        objective_mode = str(args.objective_mode)
        objective_window = max(1, int(args.objective_window))
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        study_name = args.study_name or f"{base_config.policy_id}_optuna_{timestamp}"
        storage = args.storage
        if storage is None:
            safe_study_name = _sanitize_storage_filename(study_name)
            storage_path = _DEFAULT_STORAGE_ROOT / f"{safe_study_name}.db"
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{storage_path}"
        _ensure_sqlite_storage_parent(storage, allowed_root=_ALLOWED_SQLITE_ROOT)

        logger.info(
            "Starting Optuna study '{}' direction={} metric={} objective_mode={} storage={}",
            study_name,
            direction,
            metric_name,
            objective_mode,
            make_url(storage).render_as_string(hide_password=True),
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
            config = _build_trial_config(
                trial,
                base_config=base_config,
                metric_name=metric_name,
                objective_mode=objective_mode,
                study_name=study_name,
                args=args,
                resolve_num_envs_fn=_resolve_num_envs,
                evaluation_schedule_cls=EvaluationSchedule,
            )

            result = run_expert_training(config, config_path=config_path, dry_run=False)
            metric_value, series = _resolve_trial_metric(
                result=result,
                metric_name=metric_name,
                objective_mode=objective_mode,
                objective_window=objective_window,
                load_episode_records_fn=load_episode_records,
                eval_metric_series_fn=eval_metric_series,
                objective_from_series_fn=objective_from_series,
            )
            if metric_value is None:
                raise ValueError(f"Metric '{metric_name}' not found in training output.")
            _register_trial_metadata(
                trial,
                config=config,
                result=result,
                metric_name=metric_name,
                objective_mode=objective_mode,
                objective_window=objective_window,
                series=series,
            )
            trial.set_user_attr("log_level", log_level)
            return float(metric_value)

        study.optimize(objective, n_trials=int(args.trials))
        completed_trials = study.get_trials(states=[TrialState.COMPLETE])
        if not completed_trials:
            logger.warning("Optuna study completed without any successful trials.")
            return 0
        best = study.best_trial
        logger.success(
            "Optuna study complete: best_value={} best_trial={} params={}",
            best.value,
            best.number,
            best.params,
        )
        return 0
    finally:
        if previous_loguru_level is None:
            os.environ.pop("LOGURU_LEVEL", None)
        else:
            os.environ["LOGURU_LEVEL"] = previous_loguru_level


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
