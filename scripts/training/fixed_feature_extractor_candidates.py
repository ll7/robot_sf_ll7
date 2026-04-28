"""Run a fixed feature-extractor candidate matrix and store results in Optuna.

This script is intentionally not an optimizer.  It maps one array index to one
predeclared candidate row so follow-up validation remains auditable while still
using the existing Optuna DB inspection workflow.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import yaml
from loguru import logger
from optuna.trial import TrialState
from sqlalchemy.engine import make_url

_DEFAULT_STORAGE_ROOT = Path("output/optuna/feat_extractor").resolve()
_ALLOWED_SQLITE_ROOT = Path("output").resolve()
_LOG_LEVEL_CHOICES = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


@dataclass(slots=True)
class FixedCandidate:
    """One fixed validation job from the candidate matrix."""

    candidate_id: str
    extractor_type: str
    arch_size: str
    policy_arch_size: str
    policy_arch: list[int]
    dropout_rate: float
    seed: int
    extractor_kwargs: dict[str, Any]


def _sanitize_name(raw: str, fallback: str = "feat_fixed") -> str:
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


def _resolve_storage(study_name: str, storage: str | None) -> str:
    if storage is not None:
        _ensure_sqlite_parent(storage, allowed_root=_ALLOWED_SQLITE_ROOT)
        return storage
    safe = _sanitize_name(study_name)
    path = _DEFAULT_STORAGE_ROOT / f"{safe}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Candidate file must contain a mapping: {path}")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate file must contain a non-empty 'candidates' list: {path}")
    return raw


def _candidate_from_raw(raw: dict[str, Any], *, index: int) -> FixedCandidate:
    required = {
        "id",
        "extractor_type",
        "arch_size",
        "policy_arch_size",
        "policy_arch",
        "dropout_rate",
        "seed",
        "extractor_kwargs",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Candidate #{index} missing required fields: {', '.join(missing)}")
    policy_arch = raw["policy_arch"]
    if not isinstance(policy_arch, list) or not all(isinstance(v, int) for v in policy_arch):
        raise ValueError(f"Candidate #{index} policy_arch must be a list of integers.")
    extractor_kwargs = raw["extractor_kwargs"]
    if not isinstance(extractor_kwargs, dict):
        raise ValueError(f"Candidate #{index} extractor_kwargs must be a mapping.")
    return FixedCandidate(
        candidate_id=str(raw["id"]),
        extractor_type=str(raw["extractor_type"]),
        arch_size=str(raw["arch_size"]),
        policy_arch_size=str(raw["policy_arch_size"]),
        policy_arch=[int(v) for v in policy_arch],
        dropout_rate=float(raw["dropout_rate"]),
        seed=int(raw["seed"]),
        extractor_kwargs=dict(extractor_kwargs),
    )


def _configure_candidate(
    base_config: Any,
    *,
    candidate: FixedCandidate,
    matrix: dict[str, Any],
    study_name: str,
    disable_wandb: bool,
) -> Any:
    config = copy.deepcopy(base_config)
    config.policy_id = f"{base_config.policy_id}_12m_{candidate.candidate_id}"
    config.total_timesteps = int(matrix["total_timesteps"])
    config.seeds = (int(candidate.seed),)
    config.randomize_seeds = False
    config.feature_extractor = candidate.extractor_type
    config.feature_extractor_kwargs = dict(candidate.extractor_kwargs)
    config.policy_net_arch = tuple(candidate.policy_arch)

    eval_every = int(matrix.get("eval_every", 48_000))
    eval_episodes = int(matrix.get("eval_episodes", config.evaluation.evaluation_episodes))
    from robot_sf.training.imitation_config import EvaluationSchedule

    config.evaluation = EvaluationSchedule(
        frequency_episodes=config.evaluation.frequency_episodes,
        evaluation_episodes=eval_episodes,
        hold_out_scenarios=config.evaluation.hold_out_scenarios,
        step_schedule=((None, eval_every),),
        randomize_seeds=False,
        scenario_config=config.evaluation.scenario_config,
    )

    tracking = dict(config.tracking or {})
    wandb_cfg = dict(tracking.get("wandb") or {})
    if disable_wandb:
        wandb_cfg["enabled"] = False
    else:
        wandb_cfg.setdefault("enabled", True)
        wandb_cfg.setdefault("project", "robot_sf")
        wandb_cfg.setdefault("group", study_name)
        wandb_cfg.setdefault("job_type", "feat-extractor-fixed-12m")
        wandb_cfg.setdefault("name", config.policy_id)
        wandb_cfg.setdefault(
            "tags",
            [
                "fixed-candidate",
                "issue-193",
                f"extractor:{candidate.extractor_type}",
                f"arch:{candidate.arch_size}",
                f"seed:{candidate.seed}",
            ],
        )
    tracking["wandb"] = wandb_cfg
    config.tracking = tracking
    return config


def _record_params(trial: optuna.Trial, candidate: FixedCandidate) -> None:
    """Record fixed candidate metadata without pretending Optuna sampled it.

    Optuna requires a stable distribution for each param name within a study.
    These jobs are a fixed validation matrix, so user attrs are the right place
    to store candidate identity and avoid dynamic categorical distributions.
    """
    trial.set_user_attr("candidate_id", candidate.candidate_id)
    trial.set_user_attr("extractor_type", candidate.extractor_type)
    trial.set_user_attr("arch_size", candidate.arch_size)
    trial.set_user_attr("policy_arch", candidate.policy_arch)
    trial.set_user_attr("policy_arch_size", candidate.policy_arch_size)
    trial.set_user_attr("dropout_rate", candidate.dropout_rate)
    trial.set_user_attr("training_seed", candidate.seed)
    trial.set_user_attr("extractor_kwargs", candidate.extractor_kwargs)


def _metric_from_result(result: Any, *, metric_name: str) -> float:
    from robot_sf.training.optuna_objective import (
        eval_metric_series,
        load_episode_records,
        objective_from_series,
    )

    records = load_episode_records(result.training_run_artifact.episode_log_path)
    series = eval_metric_series(records, metric_name=metric_name)
    metric_value = objective_from_series(series, mode="last_n_mean", window=3)
    if metric_value is None:
        agg = result.metrics.get(metric_name)
        metric_value = float(agg.mean) if agg is not None else 0.0
    return float(metric_value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Run one fixed feature-extractor candidate.")
    parser.add_argument("--candidate-file", required=True, help="Fixed candidate YAML matrix.")
    parser.add_argument("--candidate-index", type=int, required=True, help="Candidate row index.")
    parser.add_argument("--config", default=None, help="Override base training config path.")
    parser.add_argument("--study-name", required=True, help="Optuna study name.")
    parser.add_argument("--storage", default=None, help="Optuna storage URL.")
    parser.add_argument("--metric", default=None, help="Metric to maximize.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B tracking.")
    parser.add_argument("--log-level", default="WARNING", choices=_LOG_LEVEL_CHOICES)
    parser.add_argument(
        "--print-count",
        action="store_true",
        help="Print candidate count and exit; used by SLURM launchers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one candidate and record it in Optuna."""
    args = build_arg_parser().parse_args(argv)
    os.environ["LOGURU_LEVEL"] = str(args.log_level).upper()
    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    matrix_path = Path(args.candidate_file).resolve()
    matrix = _load_matrix(matrix_path)
    raw_candidates = matrix["candidates"]
    if args.print_count:
        print(len(raw_candidates))
        return 0
    if args.candidate_index < 0 or args.candidate_index >= len(raw_candidates):
        raise IndexError(
            f"candidate-index {args.candidate_index} outside 0..{len(raw_candidates) - 1}"
        )

    candidate = _candidate_from_raw(
        raw_candidates[args.candidate_index], index=args.candidate_index
    )
    metric_name = str(args.metric or matrix.get("metric", "eval_episode_return"))
    storage = _resolve_storage(args.study_name, args.storage)

    sys.path.insert(0, str(Path(__file__).parent))
    from train_ppo import load_expert_training_config, run_expert_training

    base_config_path = Path(args.config or matrix["base_config"]).resolve()
    base_config = load_expert_training_config(base_config_path)
    config = _configure_candidate(
        base_config,
        candidate=candidate,
        matrix=matrix,
        study_name=args.study_name,
        disable_wandb=bool(args.disable_wandb),
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )
    trial = study.ask()
    _record_params(trial, candidate)
    trial.set_user_attr("candidate_index", args.candidate_index)
    trial.set_user_attr("metric", metric_name)
    trial.set_user_attr("total_timesteps", int(config.total_timesteps))
    trial.set_user_attr("base_config", str(base_config_path))
    trial.set_user_attr("candidate_file", str(matrix_path))
    trial.set_user_attr("policy_id", config.policy_id)

    start = time.monotonic()
    try:
        result = run_expert_training(config, config_path=base_config_path, dry_run=False)
        elapsed = max(time.monotonic() - start, 1e-6)
        fps = config.total_timesteps / elapsed
        metric_value = _metric_from_result(result, metric_name=metric_name)
        trial.set_user_attr("fps", round(fps, 1))
        trial.set_user_attr("elapsed_seconds", round(elapsed, 1))
        trial.set_user_attr("metric_value", float(metric_value))
        study.tell(trial, float(metric_value))
        logger.success(
            "Candidate {} complete: metric={} value={:.4f} fps={:.1f}",
            candidate.candidate_id,
            metric_name,
            metric_value,
            fps,
        )
    except Exception as exc:
        trial.set_user_attr("failure_type", type(exc).__name__)
        trial.set_user_attr("failure_message", str(exc)[:1000])
        study.tell(trial, state=TrialState.FAIL)
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
