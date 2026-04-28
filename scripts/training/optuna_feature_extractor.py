"""Optuna study for feature extractor architecture search.

Each trial picks a feature extractor type and a matching architecture size,
trains for ``--trial-timesteps`` steps, then reports the chosen metric back
to the study.  Wall-clock throughput (env steps / second) is measured and
stored per trial; candidates below ``--fps-warn-threshold`` are flagged as
slow at the end of the run.

Two execution modes
-------------------
Local (default):
    Runs trials in-process sequentially.  Suitable for quick smoke tests
    with ``--trial-timesteps 32000``.

    uv run python scripts/training/optuna_feature_extractor.py \\
        --config configs/training/ppo/feature_extractor_sweep_base.yaml \\
        --trials 10 --trial-timesteps 32000 --disable-wandb

SLURM distributed:
    Creates the study DB, then submits N SLURM array-job elements that each
    pick up one trial independently from the shared DB.  Use
    ``--trial-timesteps 4000000`` for the full benchmark run.

    uv run python scripts/training/optuna_feature_extractor.py \\
        --config configs/training/ppo/feature_extractor_sweep_base.yaml \\
        --trials 20 --trial-timesteps 4000000 \\
        --slurm --slurm-time 08:00:00 --slurm-gpus 1 --slurm-cpus 8 \\
        --study-name feat_sweep_4m --storage sqlite:///output/optuna/feat_extractor/feat_sweep_4m.db

Search space
------------
- extractor_type : categorical (dynamics, mlp, attention, lightweight_cnn, lstm)
- arch_size      : categorical (small, medium, large)
- policy_arch    : categorical (small=[64,64], medium=[128,128], large=[256,256])
- dropout_rate   : float [0.0, 0.3]

Per-extractor architecture mapping (size → kwargs):
    dynamics      : num_filters / dropout_rates
    mlp           : ray_hidden_dims / drive_hidden_dims
    attention     : embed_dim / num_heads / num_layers
    lightweight_cnn : num_filters / kernel_sizes
    lstm          : hidden_size / num_layers / bidirectional
"""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
from loguru import logger
from optuna.trial import TrialState
from sqlalchemy.engine import make_url

_LOG_LEVEL_CHOICES = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
_DEFAULT_STORAGE_ROOT = Path("output/optuna/feat_extractor").resolve()
_ALLOWED_SQLITE_ROOT = Path("output").resolve()
_DEFAULT_FPS_WARN_THRESHOLD = 100.0
_KNOWN_DETERMINISTIC_CUDA_MARKER = (
    "adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation"
)

# ---------------------------------------------------------------------------
# Architecture search space
# ---------------------------------------------------------------------------

_EXTRACTOR_TYPES = ["dynamics", "mlp", "attention", "lightweight_cnn", "lstm"]
_ARCH_SIZES = ["small", "medium", "large"]
_POLICY_ARCH_MAP = {
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
}


def _extractor_kwargs(extractor_type: str, arch_size: str, dropout_rate: float) -> dict[str, Any]:
    """Return feature_extractor_kwargs for a given extractor type and size.

    Args:
        extractor_type: One of the extractor names in ``_EXTRACTOR_TYPES``.
        arch_size: One of 'small', 'medium', 'large'.
        dropout_rate: Dropout probability (0.0–0.3).

    Returns:
        Dict suitable for ``feature_extractor_kwargs`` in the YAML config.
    """
    if extractor_type == "dynamics":
        filters = {"small": [32, 16, 16, 8], "medium": [64, 16, 16, 16], "large": [128, 32, 32, 16]}
        return {
            "num_filters": filters[arch_size],
            "dropout_rates": [dropout_rate] * 4,
        }
    if extractor_type == "mlp":
        ray_dims = {"small": [64, 32], "medium": [128, 64], "large": [256, 128, 64]}
        drive_dims = {"small": [16, 8], "medium": [32, 16], "large": [64, 32, 16]}
        return {
            "ray_hidden_dims": ray_dims[arch_size],
            "drive_hidden_dims": drive_dims[arch_size],
            "dropout_rate": dropout_rate,
        }
    if extractor_type == "attention":
        cfg = {
            "small": {"embed_dim": 32, "num_heads": 2, "num_layers": 1},
            "medium": {"embed_dim": 64, "num_heads": 4, "num_layers": 2},
            "large": {"embed_dim": 128, "num_heads": 8, "num_layers": 3},
        }
        return {**cfg[arch_size], "dropout_rate": dropout_rate}
    if extractor_type == "lightweight_cnn":
        filters = {"small": [16, 8], "medium": [32, 16], "large": [64, 32]}
        kernels = {"small": [5, 3], "medium": [5, 3], "large": [7, 5]}
        return {
            "num_filters": filters[arch_size],
            "kernel_sizes": kernels[arch_size],
            "dropout_rate": dropout_rate,
        }
    if extractor_type == "lstm":
        cfg = {
            "small": {"hidden_size": 32, "num_layers": 1, "bidirectional": False},
            "medium": {"hidden_size": 64, "num_layers": 1, "bidirectional": False},
            "large": {
                "hidden_size": 128,
                "num_layers": 2,
                "lstm_dropout": 0.1,
                "bidirectional": True,
            },
        }
        drive = {"small": [16, 8], "medium": [32, 16], "large": [64, 32]}
        return {**cfg[arch_size], "drive_hidden_dims": drive[arch_size]}
    raise ValueError(f"Unknown extractor_type: {extractor_type!r}")


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------


def _sanitize_name(raw: str, fallback: str = "optuna_feat") -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw.strip())
    return sanitized.strip("._") or fallback


def _ensure_sqlite_parent(storage: str, *, allowed_root: Path | None = None) -> None:
    try:
        url = make_url(storage)
    except Exception:
        return
    if url.get_backend_name() != "sqlite":
        return
    database = url.database
    if not database or database == ":memory:":
        return
    resolved = Path(database).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if allowed_root is not None and not resolved.is_relative_to(allowed_root):
        raise ValueError(f"Refusing sqlite path outside allowed root {allowed_root!r}: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)


def _configure_optuna_verbosity(log_level: str) -> None:
    normalized = log_level.upper()
    if normalized in {"TRACE", "DEBUG"}:
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
    elif normalized in {"INFO", "SUCCESS"}:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    elif normalized == "WARNING":
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.ERROR)


def _sampler_seed(base_seed: int, worker_index: int | None) -> int:
    """Return the deterministic Optuna sampler seed for one worker process."""
    if worker_index is None:
        return base_seed
    return (base_seed + worker_index * 1_000_003) % (2**32 - 1)


def _classify_trial_failure(exc: BaseException) -> tuple[str, str]:
    """Return a stable failure type plus a concise message for Optuna attrs."""
    message = str(exc)
    if _KNOWN_DETERMINISTIC_CUDA_MARKER in message:
        return "deterministic_cuda_kernel", message
    return type(exc).__name__, message


# ---------------------------------------------------------------------------
# Trial config builder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TrialSpec:
    """Resolved hyperparameters for one trial."""

    extractor_type: str
    arch_size: str
    dropout_rate: float
    policy_arch: list[int]
    extractor_kwargs: dict[str, Any]


def _suggest_trial(trial: optuna.Trial) -> _TrialSpec:
    """Sample all feature-extractor hyperparameters for one trial.

    Args:
        trial: Active Optuna trial.

    Returns:
        Fully resolved ``_TrialSpec``.
    """
    extractor_type = trial.suggest_categorical("extractor_type", _EXTRACTOR_TYPES)
    arch_size = trial.suggest_categorical("arch_size", _ARCH_SIZES)
    policy_arch_size = trial.suggest_categorical("policy_arch_size", list(_POLICY_ARCH_MAP))
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    kwargs = _extractor_kwargs(extractor_type, arch_size, dropout_rate)
    return _TrialSpec(
        extractor_type=extractor_type,
        arch_size=arch_size,
        dropout_rate=dropout_rate,
        policy_arch=_POLICY_ARCH_MAP[policy_arch_size],
        extractor_kwargs=kwargs,
    )


def _apply_trial_spec(
    base_config: Any, spec: _TrialSpec, trial_num: int, args: argparse.Namespace, study_name: str
) -> Any:
    """Patch a deep-copied base config with trial-specific values.

    Args:
        base_config: Parsed ``ExpertTrainingConfig``.
        spec: Trial-level hyperparameter spec.
        trial_num: Optuna trial index (used for unique policy_id).
        args: CLI namespace (timesteps, eval cadence, W&B flags, etc.).
        study_name: Study name for W&B grouping.

    Returns:
        Modified config ready for ``run_expert_training``.
    """
    config = copy.deepcopy(base_config)
    config.policy_id = (
        f"{base_config.policy_id}_fe_{trial_num:03d}_{spec.extractor_type}_{spec.arch_size}"
    )
    config.feature_extractor = spec.extractor_type
    config.feature_extractor_kwargs = spec.extractor_kwargs
    config.policy_net_arch = spec.policy_arch
    config.total_timesteps = int(args.trial_timesteps)

    # Evaluation schedule
    from robot_sf.training.imitation_config import EvaluationSchedule

    eval_every = int(args.eval_every)
    config.evaluation = EvaluationSchedule(
        frequency_episodes=config.evaluation.frequency_episodes,
        evaluation_episodes=int(args.eval_episodes),
        hold_out_scenarios=config.evaluation.hold_out_scenarios,
        step_schedule=((None, eval_every),),
    )

    # W&B / tracking
    tracking = dict(config.tracking or {})
    wandb_cfg = dict(tracking.get("wandb") or {})
    if args.disable_wandb:
        wandb_cfg["enabled"] = False
    else:
        wandb_cfg.setdefault("enabled", True)
    if wandb_cfg.get("enabled"):
        wandb_cfg.setdefault("project", "robot_sf")
        wandb_cfg.setdefault("group", study_name)
        wandb_cfg.setdefault("job_type", "feat-extractor-sweep")
        wandb_cfg.setdefault("name", config.policy_id)
        wandb_cfg.setdefault(
            "tags",
            ["optuna", "feat-sweep", f"extractor:{spec.extractor_type}", f"arch:{spec.arch_size}"],
        )
    tracking["wandb"] = wandb_cfg
    config.tracking = tracking
    return config


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ObjectiveContext:
    args: argparse.Namespace
    base_config: Any
    config_path: Path
    metric_name: str
    study_name: str
    fps_warn_threshold: float


def _make_objective(ctx: _ObjectiveContext):
    """Build the Optuna objective closure.

    Returns:
        Callable[[optuna.Trial], float] that trains one trial and returns
        the optimisation metric.
    """

    def objective(trial: optuna.Trial) -> float:
        spec = _suggest_trial(trial)
        config = _apply_trial_spec(
            ctx.base_config,
            spec,
            trial_num=trial.number,
            args=ctx.args,
            study_name=ctx.study_name,
        )

        # Record hyperparams as trial user-attrs for easy inspection.
        trial.set_user_attr("extractor_type", spec.extractor_type)
        trial.set_user_attr("arch_size", spec.arch_size)
        trial.set_user_attr("policy_arch", spec.policy_arch)
        trial.set_user_attr("extractor_kwargs", spec.extractor_kwargs)

        # ---- train --------------------------------------------------------
        from train_ppo import (  # noqa: F401
            _resolve_num_envs,
            load_expert_training_config,
            run_expert_training,
        )

        t0 = time.monotonic()
        try:
            result = run_expert_training(config, config_path=ctx.config_path, dry_run=False)
        except Exception as exc:
            failure_type, failure_message = _classify_trial_failure(exc)
            trial.set_user_attr("failure_type", failure_type)
            trial.set_user_attr("failure_message", failure_message[:1000])
            logger.warning(
                "Trial {} failed during training: {} ({})",
                trial.number,
                failure_type,
                failure_message,
            )
            raise
        elapsed = max(time.monotonic() - t0, 1e-6)

        # FPS measured as total env steps / wall-clock seconds.
        fps = config.total_timesteps / elapsed
        trial.set_user_attr("fps", round(fps, 1))
        trial.set_user_attr("elapsed_seconds", round(elapsed, 1))
        if fps < ctx.fps_warn_threshold:
            logger.warning(
                "Trial {} ({} {}) is slow: {:.0f} fps < threshold {:.0f}. "
                "Consider excluding this extractor from the full SLURM run.",
                trial.number,
                spec.extractor_type,
                spec.arch_size,
                fps,
                ctx.fps_warn_threshold,
            )

        # ---- metric -------------------------------------------------------
        from robot_sf.training.optuna_objective import (
            eval_metric_series,
            load_episode_records,
            objective_from_series,
        )

        records = load_episode_records(result.training_run_artifact.episode_log_path)
        series = eval_metric_series(records, metric_name=ctx.metric_name)
        metric_value = objective_from_series(series, mode="last_n_mean", window=3)
        if metric_value is None:
            agg = result.metrics.get(ctx.metric_name)
            metric_value = float(agg.mean) if agg is not None else 0.0

        trial.set_user_attr("policy_id", config.policy_id)
        trial.set_user_attr("metric", ctx.metric_name)
        trial.set_user_attr("metric_value", float(metric_value))
        return float(metric_value)

    return objective


# ---------------------------------------------------------------------------
# SLURM launcher
# ---------------------------------------------------------------------------


def _submit_slurm_jobs(  # noqa: PLR0913
    *,
    n_trials: int,
    config: str,
    storage: str,
    study_name: str,
    trial_timesteps: int,
    eval_every: int,
    eval_episodes: int,
    metric: str,
    slurm_time: str,
    slurm_gpus: int,
    slurm_cpus: int,
    slurm_mem: str,
    slurm_partition: str | None,
    disable_wandb: bool,
    log_level: str,
    fps_warn_threshold: float,
    seed: int,
    extractor_exclude: list[str],
) -> None:
    """Submit N independent SLURM jobs that each pick up one Optuna trial.

    The standard distributed-Optuna pattern: every job calls this same
    script with ``--n-trials 1``.  Optuna's TPE sampler handles
    deduplication via the shared SQLite storage.

    Args:
        n_trials: Number of SLURM jobs (= target number of completed trials).
        config: Path to base training config YAML.
        storage: Optuna SQLite storage URL.
        study_name: Optuna study name shared across all jobs.
        trial_timesteps: Timesteps per trial.
        eval_every: Evaluation cadence in steps.
        eval_episodes: Episodes per evaluation checkpoint.
        metric: Metric name to optimise.
        slurm_time: SLURM time limit (e.g. '08:00:00').
        slurm_gpus: GPUs per node.
        slurm_cpus: CPUs per task.
        slurm_mem: Memory per job (e.g. '32G').
        slurm_partition: Optional SLURM partition name.
        disable_wandb: Whether to disable W&B in SLURM jobs.
        log_level: Loguru log level for worker jobs.
        fps_warn_threshold: FPS below which a trial is flagged as slow.
        seed: Base Optuna sampler seed.
        extractor_exclude: Extractor types to exclude in worker jobs.
    """
    script = Path(__file__).resolve()
    python = sys.executable

    base_cmd = [
        python,
        str(script),
        "--config",
        config,
        "--trials",
        "1",
        "--storage",
        storage,
        "--study-name",
        study_name,
        "--trial-timesteps",
        str(trial_timesteps),
        "--eval-every",
        str(eval_every),
        "--eval-episodes",
        str(eval_episodes),
        "--metric",
        metric,
        "--fps-warn-threshold",
        str(fps_warn_threshold),
        "--log-level",
        log_level,
        "--seed",
        str(seed),
    ]
    if disable_wandb:
        base_cmd.append("--disable-wandb")
    if extractor_exclude:
        base_cmd.append("--extractor-exclude")
        base_cmd.extend(extractor_exclude)

    sbatch_flags = [
        f"--time={slurm_time}",
        f"--cpus-per-task={slurm_cpus}",
        f"--mem={slurm_mem}",
        f"--gres=gpu:{slurm_gpus}",
        "--output=output/slurm/feat_sweep_%j.out",
        "--error=output/slurm/feat_sweep_%j.err",
        "--job-name=feat_sweep",
    ]
    if slurm_partition:
        sbatch_flags.append(f"--partition={slurm_partition}")

    Path("output/slurm").mkdir(parents=True, exist_ok=True)

    submitted = 0
    for i in range(n_trials):
        job_name = f"feat_sweep_t{i:03d}"
        worker_cmd = base_cmd + ["--worker-index", str(i)]
        sbatch_cmd = (
            ["sbatch", f"--job-name={job_name}"] + sbatch_flags + ["--wrap", shlex.join(worker_cmd)]
        )
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            submitted += 1
            logger.info("Submitted SLURM job {}/{}: {}", i + 1, n_trials, result.stdout.strip())
        else:
            logger.error(
                "Failed to submit SLURM job {}/{}: {}",
                i + 1,
                n_trials,
                result.stderr.strip(),
            )

    logger.success(
        "Submitted {}/{} SLURM jobs for study '{}'. "
        "Monitor progress with: uv run python scripts/tools/inspect_optuna_db.py --db {}",
        submitted,
        n_trials,
        study_name,
        make_url(storage).database or storage,
    )


# ---------------------------------------------------------------------------
# FPS summary
# ---------------------------------------------------------------------------


def _log_fps_summary(study: optuna.Study, *, threshold: float) -> None:
    """Print per-extractor FPS stats and flag slow candidates.

    Args:
        study: Completed (or partially completed) Optuna study.
        threshold: FPS below which a candidate is considered slow.
    """
    completed = study.get_trials(states=[TrialState.COMPLETE])
    if not completed:
        return

    from collections import defaultdict

    fps_by_extractor: dict[str, list[float]] = defaultdict(list)
    for t in completed:
        ext = t.user_attrs.get("extractor_type", "unknown")
        fps = t.user_attrs.get("fps")
        if fps is not None:
            fps_by_extractor[ext].append(float(fps))

    logger.info("--- FPS summary by extractor type ---")
    slow: list[str] = []
    for ext, fps_list in sorted(fps_by_extractor.items()):
        mean_fps = sum(fps_list) / len(fps_list)
        min_fps = min(fps_list)
        flag = " [SLOW]" if min_fps < threshold else ""
        logger.info("  {:<20} mean={:>7.0f}  min={:>7.0f}{}", ext, mean_fps, min_fps, flag)
        if min_fps < threshold:
            slow.append(ext)

    if slow:
        logger.warning(
            "Slow extractors (min fps < {:.0f}): {}. "
            "Exclude them from the full SLURM run with --extractor-exclude.",
            threshold,
            ", ".join(slow),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the feature extractor sweep."""
    p = argparse.ArgumentParser(description="Optuna feature extractor architecture sweep.")
    p.add_argument("--config", required=True, help="Base training config YAML.")
    p.add_argument("--trials", type=int, default=10, help="Number of Optuna trials.")
    p.add_argument(
        "--metric",
        default="eval_episode_return",
        help="Metric to maximise (default: eval_episode_return).",
    )
    p.add_argument(
        "--trial-timesteps",
        type=int,
        default=32_000,
        help="Training steps per trial (default: 32 000 for local smoke tests).",
    )
    p.add_argument("--eval-every", type=int, default=16_000, help="Evaluation cadence in steps.")
    p.add_argument(
        "--eval-episodes", type=int, default=5, help="Episodes per evaluation checkpoint."
    )
    p.add_argument(
        "--study-name", default=None, help="Optuna study name (defaults to policy_id + timestamp)."
    )
    p.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL.  Defaults to sqlite under output/optuna/.",
    )
    p.add_argument("--seed", type=int, default=42, help="Optuna sampler seed.")
    p.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--fps-warn-threshold",
        type=float,
        default=_DEFAULT_FPS_WARN_THRESHOLD,
        help=f"Flag trials below this FPS as slow (default: {_DEFAULT_FPS_WARN_THRESHOLD}).",
    )
    p.add_argument(
        "--extractor-exclude",
        nargs="*",
        default=[],
        help="Extractor types to exclude from the search space.",
    )
    p.add_argument("--disable-wandb", action="store_true", help="Disable W&B for all trials.")
    p.add_argument("--log-level", default="WARNING", choices=_LOG_LEVEL_CHOICES)
    # SLURM mode
    slurm_grp = p.add_argument_group("SLURM mode")
    slurm_grp.add_argument(
        "--slurm",
        action="store_true",
        help="Submit trials as SLURM jobs instead of running in-process.",
    )
    slurm_grp.add_argument(
        "--slurm-time", default="04:00:00", help="SLURM time limit per job (default: 04:00:00)."
    )
    slurm_grp.add_argument(
        "--slurm-gpus", type=int, default=1, help="GPUs per SLURM job (default: 1)."
    )
    slurm_grp.add_argument(
        "--slurm-cpus", type=int, default=8, help="CPUs per SLURM task (default: 8)."
    )
    slurm_grp.add_argument(
        "--slurm-mem", default="32G", help="Memory per SLURM job (default: 32G)."
    )
    slurm_grp.add_argument("--slurm-partition", default=None, help="Optional SLURM partition name.")
    return p


def _resolve_storage(study_name: str, storage: str | None) -> str:
    if storage is not None:
        _ensure_sqlite_parent(storage, allowed_root=_ALLOWED_SQLITE_ROOT)
        return storage
    safe = _sanitize_name(study_name)
    path = _DEFAULT_STORAGE_ROOT / f"{safe}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the feature extractor Optuna sweep."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    prev_level = os.environ.get("LOGURU_LEVEL")
    try:
        log_level = str(args.log_level).upper()
        os.environ["LOGURU_LEVEL"] = log_level
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        _configure_optuna_verbosity(log_level)

        # Patch extractor list based on exclusions.
        active_extractors = [e for e in _EXTRACTOR_TYPES if e not in (args.extractor_exclude or [])]
        if not active_extractors:
            logger.error("All extractor types excluded — nothing to sweep.")
            return 1

        sys.path.insert(0, str(Path(__file__).parent))
        from train_ppo import load_expert_training_config

        config_path = Path(args.config).resolve()
        base_config = load_expert_training_config(config_path)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        study_name = args.study_name or f"{base_config.policy_id}_feat_{timestamp}"
        storage = _resolve_storage(study_name, args.storage)

        logger.info(
            "Feature extractor sweep: study='{}' extractors={} trials={} timesteps={} storage={}",
            study_name,
            active_extractors,
            args.trials,
            args.trial_timesteps,
            make_url(storage).render_as_string(hide_password=True),
        )

        sampler = optuna.samplers.TPESampler(seed=_sampler_seed(int(args.seed), args.worker_index))
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            load_if_exists=True,
        )

        if args.slurm:
            _submit_slurm_jobs(
                n_trials=args.trials,
                config=args.config,
                storage=storage,
                study_name=study_name,
                trial_timesteps=args.trial_timesteps,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                metric=args.metric,
                slurm_time=args.slurm_time,
                slurm_gpus=args.slurm_gpus,
                slurm_cpus=args.slurm_cpus,
                slurm_mem=args.slurm_mem,
                slurm_partition=args.slurm_partition,
                disable_wandb=args.disable_wandb,
                log_level=log_level,
                fps_warn_threshold=args.fps_warn_threshold,
                seed=int(args.seed),
                extractor_exclude=list(args.extractor_exclude or []),
            )
            return 0

        # Local in-process run.
        # Monkey-patch the active extractor list into the module-level list so
        # _suggest_trial respects --extractor-exclude.
        _EXTRACTOR_TYPES[:] = active_extractors

        objective = _make_objective(
            _ObjectiveContext(
                args=args,
                base_config=base_config,
                config_path=config_path,
                metric_name=args.metric,
                study_name=study_name,
                fps_warn_threshold=args.fps_warn_threshold,
            )
        )
        # Catch all exceptions so a single OOM or crash records a FAILED trial
        # and the study continues rather than terminating the whole process.
        study.optimize(objective, n_trials=args.trials, catch=(Exception,))

        completed = study.get_trials(states=[TrialState.COMPLETE])
        if not completed:
            logger.warning("No trials completed.")
            return 0

        _log_fps_summary(study, threshold=args.fps_warn_threshold)

        best = study.best_trial
        logger.success(
            "Best trial: #{} value={:.4f} extractor={} arch={} policy_arch={}",
            best.number,
            best.value,
            best.user_attrs.get("extractor_type"),
            best.user_attrs.get("arch_size"),
            best.user_attrs.get("policy_arch"),
        )
        return 0
    finally:
        if prev_level is None:
            os.environ.pop("LOGURU_LEVEL", None)
        else:
            os.environ["LOGURU_LEVEL"] = prev_level


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
