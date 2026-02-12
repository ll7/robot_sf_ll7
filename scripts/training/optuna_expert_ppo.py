"""Optuna study runner for expert PPO training.

This script sweeps PPO hyperparameters for the expert imitation pipeline using the
same configuration format as ``train_expert_ppo.py``. Trials run shorter training
loops by default so the study can iterate quickly; adjust the CLI flags for full
length sweeps.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
from loguru import logger
from optuna.trial import TrialState
from sqlalchemy.engine import make_url

_OBJECTIVE_MODES = ("best_checkpoint", "final_eval", "last_n_mean", "auc")
_CONSTRAINT_HANDLING_CHOICES = ("penalize", "prune")
_LOG_LEVEL_CHOICES = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
_DEFAULT_STORAGE_ROOT = Path("output/benchmarks/ppo_imitation/hparam_opt").resolve()
_ALLOWED_SQLITE_ROOT = Path("output").resolve()
_CONSTRAINT_PENALTY_OFFSET = 1_000_000.0
_CONSTRAINT_PENALTY_SCALE = 1_000.0


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
        "--constraint-collision-rate-max",
        type=float,
        default=None,
        help=(
            "Optional safety gate: require collision_rate <= threshold. "
            "Infeasible trials are penalized or pruned (see --constraint-handling)."
        ),
    )
    parser.add_argument(
        "--constraint-comfort-exposure-max",
        type=float,
        default=None,
        help=(
            "Optional safety gate: require comfort_exposure <= threshold. "
            "Infeasible trials are penalized or pruned (see --constraint-handling)."
        ),
    )
    parser.add_argument(
        "--constraint-handling",
        choices=_CONSTRAINT_HANDLING_CHOICES,
        default="penalize",
        help=(
            "How to handle infeasible trials when constraints are configured: "
            "penalize (default) or prune."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=_LOG_LEVEL_CHOICES,
        help="Console log level (default: WARNING).",
    )
    return parser


def _build_safety_constraints(args: argparse.Namespace) -> dict[str, float]:
    """Extract safety constraints from CLI arguments.

    Returns:
        Mapping of metric name to max-allowed threshold.
    """
    constraints: dict[str, float] = {}
    if args.constraint_collision_rate_max is not None:
        constraints["collision_rate"] = float(args.constraint_collision_rate_max)
    if args.constraint_comfort_exposure_max is not None:
        constraints["comfort_exposure"] = float(args.constraint_comfort_exposure_max)
    for metric_name, threshold in constraints.items():
        if not math.isfinite(threshold):
            raise ValueError(f"Constraint threshold for '{metric_name}' must be finite.")
        if threshold < 0.0:
            raise ValueError(
                f"Constraint threshold for '{metric_name}' must be >= 0.0, got {threshold}."
            )
    return constraints


def _resolve_metric_for_mode(
    *,
    result,
    records: list[dict[str, object]],
    metric_name: str,
    objective_mode: str,
    objective_window: int,
    eval_metric_series_fn: Any,
    objective_from_series_fn: Any,
    prefer_best_checkpoint: bool,
) -> tuple[float | None, list[tuple[int, float]]]:
    """Resolve a scalar metric using objective-mode reducers and training fallbacks.

    Returns:
        Tuple of scalar metric value (or ``None``) and the reduced eval series.
    """
    metric_value: float | None = None
    if prefer_best_checkpoint:
        best = result.best_checkpoint
        raw = best.metrics.get(metric_name) if best is not None else None
        if raw is not None:
            metric_value = float(raw)

    series = eval_metric_series_fn(records, metric_name=metric_name)
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


def _evaluate_safety_constraints(
    *,
    result,
    records: list[dict[str, object]],
    constraints: dict[str, float],
    objective_mode: str,
    objective_window: int,
    eval_metric_series_fn: Any,
    objective_from_series_fn: Any,
) -> tuple[bool, dict[str, float], dict[str, float], list[str]]:
    """Evaluate configured safety constraints for one trial.

    Returns:
        A tuple ``(feasible, values, violations, missing_metrics)`` where:
        - ``values`` maps constraint metric -> resolved scalar value.
        - ``violations`` maps metric -> positive margin above threshold.
        - ``missing_metrics`` lists constraints that could not be resolved.
    """
    if not constraints:
        return True, {}, {}, []

    prefer_best_checkpoint = objective_mode == "best_checkpoint"
    values: dict[str, float] = {}
    violations: dict[str, float] = {}
    missing_metrics: list[str] = []
    for metric_name, threshold in constraints.items():
        metric_value, _ = _resolve_metric_for_mode(
            result=result,
            records=records,
            metric_name=metric_name,
            objective_mode=objective_mode,
            objective_window=objective_window,
            eval_metric_series_fn=eval_metric_series_fn,
            objective_from_series_fn=objective_from_series_fn,
            prefer_best_checkpoint=prefer_best_checkpoint,
        )
        if metric_value is None:
            missing_metrics.append(metric_name)
            continue
        values[metric_name] = float(metric_value)
        if metric_value > threshold:
            violations[metric_name] = float(metric_value - threshold)
    feasible = not violations and not missing_metrics
    return feasible, values, violations, missing_metrics


def _apply_constraint_handling(
    *,
    objective_value: float,
    direction: str,
    handling: str,
    violations: dict[str, float],
    missing_metrics: list[str],
) -> float:
    """Apply infeasible-trial handling and return an adjusted objective.

    Raises:
        optuna.TrialPruned: When ``handling == "prune"``.
    """
    if handling == "prune":
        raise optuna.TrialPruned(
            "Trial pruned by safety constraints "
            f"(violations={violations}, missing_metrics={missing_metrics})."
        )

    if handling != "penalize":
        raise ValueError(
            f"Unknown constraint handling '{handling}'. "
            f"Expected one of: {', '.join(_CONSTRAINT_HANDLING_CHOICES)}"
        )

    violation_budget = sum(max(0.0, float(v)) for v in violations.values()) + float(
        len(missing_metrics)
    )
    penalty = _CONSTRAINT_PENALTY_OFFSET + (_CONSTRAINT_PENALTY_SCALE * violation_budget)
    if direction == "maximize":
        return float(objective_value - penalty)
    if direction == "minimize":
        return float(objective_value + penalty)
    raise ValueError(f"Unknown study direction '{direction}'.")


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
    records: list[dict[str, object]],
    metric_name: str,
    objective_mode: str,
    objective_window: int,
    eval_metric_series_fn: Any,
    objective_from_series_fn: Any,
) -> tuple[float | None, list[tuple[int, float]]]:
    """Resolve the optimization scalar from training outputs."""
    return _resolve_metric_for_mode(
        result=result,
        records=records,
        metric_name=metric_name,
        objective_mode=objective_mode,
        objective_window=objective_window,
        eval_metric_series_fn=eval_metric_series_fn,
        objective_from_series_fn=objective_from_series_fn,
        prefer_best_checkpoint=(objective_mode == "best_checkpoint"),
    )


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


def _resolve_storage_url(study_name: str, storage: str | None) -> str:
    """Resolve study storage URL and ensure sqlite parent directories exist."""
    resolved_storage = storage
    if resolved_storage is None:
        safe_study_name = _sanitize_storage_filename(study_name)
        storage_path = _DEFAULT_STORAGE_ROOT / f"{safe_study_name}.db"
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_storage = f"sqlite:///{storage_path}"
    _ensure_sqlite_storage_parent(resolved_storage, allowed_root=_ALLOWED_SQLITE_ROOT)
    return resolved_storage


@dataclass(slots=True)
class _ObjectiveContext:
    """Captured runtime context for one Optuna objective function."""

    args: argparse.Namespace
    base_config: Any
    config_path: Path
    metric_name: str
    direction: str
    objective_mode: str
    objective_window: int
    study_name: str
    safety_constraints: dict[str, float]
    constraint_handling: str
    log_level: str
    resolve_num_envs_fn: Any
    evaluation_schedule_cls: Any
    run_expert_training_fn: Any
    load_episode_records_fn: Any
    eval_metric_series_fn: Any
    objective_from_series_fn: Any


def _make_objective(ctx: _ObjectiveContext) -> Any:
    """Build the Optuna objective function with configured safety gating."""

    def objective(trial: optuna.Trial) -> float:
        """Single Optuna objective run."""
        config = _build_trial_config(
            trial,
            base_config=ctx.base_config,
            metric_name=ctx.metric_name,
            objective_mode=ctx.objective_mode,
            study_name=ctx.study_name,
            args=ctx.args,
            resolve_num_envs_fn=ctx.resolve_num_envs_fn,
            evaluation_schedule_cls=ctx.evaluation_schedule_cls,
        )

        result = ctx.run_expert_training_fn(config, config_path=ctx.config_path, dry_run=False)
        episode_records = ctx.load_episode_records_fn(result.training_run_artifact.episode_log_path)
        metric_value, series = _resolve_trial_metric(
            result=result,
            records=episode_records,
            metric_name=ctx.metric_name,
            objective_mode=ctx.objective_mode,
            objective_window=ctx.objective_window,
            eval_metric_series_fn=ctx.eval_metric_series_fn,
            objective_from_series_fn=ctx.objective_from_series_fn,
        )
        if metric_value is None:
            raise ValueError(f"Metric '{ctx.metric_name}' not found in training output.")

        feasible, constraint_values, violations, missing_metrics = _evaluate_safety_constraints(
            result=result,
            records=episode_records,
            constraints=ctx.safety_constraints,
            objective_mode=ctx.objective_mode,
            objective_window=ctx.objective_window,
            eval_metric_series_fn=ctx.eval_metric_series_fn,
            objective_from_series_fn=ctx.objective_from_series_fn,
        )
        _register_trial_metadata(
            trial,
            config=config,
            result=result,
            metric_name=ctx.metric_name,
            objective_mode=ctx.objective_mode,
            objective_window=ctx.objective_window,
            series=series,
        )
        trial.set_user_attr("safety_constraints_enabled", bool(ctx.safety_constraints))
        if ctx.safety_constraints:
            trial.set_user_attr("safety_constraints", dict(ctx.safety_constraints))
            trial.set_user_attr("safety_constraint_values", constraint_values)
            trial.set_user_attr("safety_constraint_violations", violations)
            trial.set_user_attr("safety_constraint_missing_metrics", missing_metrics)
            trial.set_user_attr("safety_constraints_feasible", feasible)
            trial.set_user_attr("safety_constraint_handling", ctx.constraint_handling)
            if not feasible:
                logger.warning(
                    "Trial {} infeasible under safety constraints: values={} violations={} missing={}",
                    trial.number,
                    constraint_values,
                    violations,
                    missing_metrics,
                )
                metric_value = _apply_constraint_handling(
                    objective_value=float(metric_value),
                    direction=ctx.direction,
                    handling=ctx.constraint_handling,
                    violations=violations,
                    missing_metrics=missing_metrics,
                )
        else:
            trial.set_user_attr("safety_constraints_feasible", True)
        trial.set_user_attr("log_level", ctx.log_level)
        return float(metric_value)

    return objective


def _log_safety_constraint_summary(study: optuna.Study, *, enabled: bool) -> None:
    """Log feasible/infeasible trial counts when safety constraints are enabled."""
    if not enabled:
        return
    all_trials = study.get_trials(deepcopy=False)
    feasible_trials = sum(1 for t in all_trials if t.user_attrs.get("safety_constraints_feasible"))
    infeasible_trials = sum(
        1 for t in all_trials if t.user_attrs.get("safety_constraints_feasible") is False
    )
    logger.info(
        "Safety constraints summary: feasible_trials={} infeasible_trials={} total_trials={}",
        feasible_trials,
        infeasible_trials,
        len(all_trials),
    )


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
        safety_constraints = _build_safety_constraints(args)
        constraint_handling = str(args.constraint_handling)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        study_name = args.study_name or f"{base_config.policy_id}_optuna_{timestamp}"
        storage = _resolve_storage_url(study_name, args.storage)

        logger.info(
            "Starting Optuna study '{}' direction={} metric={} objective_mode={} constraints={} handling={} storage={}",
            study_name,
            direction,
            metric_name,
            objective_mode,
            safety_constraints or None,
            constraint_handling,
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

        objective = _make_objective(
            _ObjectiveContext(
                args=args,
                base_config=base_config,
                config_path=config_path,
                metric_name=metric_name,
                direction=direction,
                objective_mode=objective_mode,
                objective_window=objective_window,
                study_name=study_name,
                safety_constraints=safety_constraints,
                constraint_handling=constraint_handling,
                log_level=log_level,
                resolve_num_envs_fn=_resolve_num_envs,
                evaluation_schedule_cls=EvaluationSchedule,
                run_expert_training_fn=run_expert_training,
                load_episode_records_fn=load_episode_records,
                eval_metric_series_fn=eval_metric_series,
                objective_from_series_fn=objective_from_series,
            )
        )

        study.optimize(objective, n_trials=int(args.trials))
        completed_trials = study.get_trials(states=[TrialState.COMPLETE])
        if not completed_trials:
            logger.warning("Optuna study completed without any successful trials.")
            return 0
        best = study.best_trial
        _log_safety_constraint_summary(study, enabled=bool(safety_constraints))
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
