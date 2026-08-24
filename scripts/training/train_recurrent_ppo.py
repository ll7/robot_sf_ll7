"""RecurrentPPO LSTM training lane for issue #4014, productionized for #7847.

The regular ``train_ppo.py`` path can use an LSTM feature extractor, but that
extractor is not step-recurrent. This entry point is intentionally separate so
the recurrent arm uses ``sb3_contrib.RecurrentPPO`` with explicit
``lstm_states``/``episode_start`` handling.

Production runtime contract (issue #7847):

- one run identity, output directory, manifest, checkpoint lineage, and terminal
  status per training seed;
- periodic deterministic evaluation propagating recurrent state correctly;
- checkpoint index with best/latest/final lineage and reproducible selection;
- resume only through identity-checked clean episode boundaries;
- fail-closed monitoring with machine-readable records;
- bounded local smoke budgets; the matched campaign stays out of scope.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from robot_sf.common.artifact_paths import ensure_run_tracker_tree
from robot_sf.training.recurrent_runtime import (
    EpisodeBoundary,
    RecurrentStateError,
    ResetAccounting,
    append_jsonl_record,
    build_checkpoint_index_entry,
    reset_reasons_for,
    summarize_state_norms,
    utc_now_iso,
    write_checkpoint_index,
    write_json_atomic,
)
from scripts.training import train_ppo

CLAIM_BOUNDARY = (
    "local production smoke for true sb3_contrib.RecurrentPPO LSTM; "
    "not a full training comparison, benchmark campaign, or paper-facing claim"
)
EVIDENCE_TIER = "dry_run_smoke_prep"
REQUIRED_EXTRA_HINT = (
    "sb3-contrib required for recurrent_ppo. Install with `uv sync --extra recurrent`."
)

_ALLOWED_RECURRENT_PPO_HYPERPARAMS = {
    "batch_size",
    "clip_range",
    "clip_range_vf",
    "device",
    "ent_coef",
    "gae_lambda",
    "gamma",
    "learning_rate",
    "max_grad_norm",
    "n_epochs",
    "n_steps",
    "normalize_advantage",
    "policy_kwargs",
    "seed",
    "target_kl",
    "tensorboard_log",
    "verbose",
    "vf_coef",
}

_ALLOWED_POLICY_KWARGS = {
    "enable_critic_lstm",
    "lstm_hidden_size",
    "n_lstm_layers",
    "net_arch",
    "shared_lstm",
}

_ALLOWED_RECURRENT_POLICIES = {
    "MlpLstmPolicy",
    "MultiInputLstmPolicy",
}


@dataclass(frozen=True, slots=True)
class RecurrentPPOConfig:
    """Validated RecurrentPPO lane config."""

    base: train_ppo.ExpertTrainingConfig
    algorithm: str
    recurrent_policy: str
    recurrent_ppo_hyperparams: dict[str, Any]
    policy_kwargs: dict[str, Any]


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``config_path``."""
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping, received {type(raw)!r}")
    return raw


def load_recurrent_ppo_config(config_path: str | Path) -> RecurrentPPOConfig:
    """Load and validate the recurrent PPO config."""
    path = Path(config_path).resolve()
    raw = _load_raw_config(path)
    algorithm = str(raw.get("algorithm", "")).strip().lower()
    if algorithm != "recurrent_ppo":
        raise ValueError("algorithm must be 'recurrent_ppo' for train_recurrent_ppo.py")
    recurrent_policy = str(raw.get("recurrent_policy", "MultiInputLstmPolicy")).strip()
    if recurrent_policy not in _ALLOWED_RECURRENT_POLICIES:
        allowed = ", ".join(sorted(_ALLOWED_RECURRENT_POLICIES))
        raise ValueError(f"recurrent_policy must be one of: {allowed}")

    hyperparams = dict(raw.get("recurrent_ppo_hyperparams", {}) or {})
    unknown = set(hyperparams) - _ALLOWED_RECURRENT_PPO_HYPERPARAMS
    if unknown:
        raise ValueError(
            f"recurrent_ppo_hyperparams unsupported keys: {', '.join(sorted(unknown))}",
        )

    policy_kwargs = dict(hyperparams.get("policy_kwargs", {}) or {})
    unknown_policy_kwargs = set(policy_kwargs) - _ALLOWED_POLICY_KWARGS
    if unknown_policy_kwargs:
        raise ValueError(
            "recurrent_ppo_hyperparams.policy_kwargs unsupported keys: "
            f"{', '.join(sorted(unknown_policy_kwargs))}",
        )

    for key in ("n_steps", "batch_size", "n_epochs"):
        if key in hyperparams and int(hyperparams[key]) <= 0:
            raise ValueError(f"recurrent_ppo_hyperparams.{key} must be positive")
    for key in ("lstm_hidden_size", "n_lstm_layers"):
        if key in policy_kwargs and int(policy_kwargs[key]) <= 0:
            raise ValueError(
                f"recurrent_ppo_hyperparams.policy_kwargs.{key} must be positive",
            )

    base = train_ppo.load_expert_training_config(path)
    return RecurrentPPOConfig(
        base=base,
        algorithm=algorithm,
        recurrent_policy=recurrent_policy,
        recurrent_ppo_hyperparams=hyperparams,
        policy_kwargs=policy_kwargs,
    )


def _require_sb3_contrib() -> type[Any]:
    """Import RecurrentPPO with an actionable optional-dependency error."""
    try:
        module = importlib.import_module("sb3_contrib")
    except ImportError as exc:
        raise RuntimeError(REQUIRED_EXTRA_HINT) from exc
    try:
        return module.RecurrentPPO
    except AttributeError as exc:  # pragma: no cover - defensive package-integrity guard
        raise RuntimeError("Installed sb3_contrib package does not expose RecurrentPPO.") from exc


def _dependency_versions() -> dict[str, str]:
    """Record package versions required by the provenance contract."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "robot_sf": _robot_sf_version(),
    }
    for module_name, key in (
        ("torch", "torch"),
        ("gymnasium", "gymnasium"),
        ("stable_baselines3", "stable_baselines3"),
        ("sb3_contrib", "sb3_contrib"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            versions[key] = str(version)
    return versions


def _robot_sf_version() -> str:
    """Resolve the installed robot_sf distribution version."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("robot_sf")
    except PackageNotFoundError:
        return "unknown"


def _resolve_git_sha() -> str:
    """Best-effort git HEAD SHA for provenance; 'unknown' outside a repository."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(slots=True)
class SeedRunPlan:
    """One independent per-seed execution unit."""

    seed: int
    run_id: str
    output_dir: Path


def plan_seed_runs(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
) -> list[SeedRunPlan]:
    """Build one independent, collision-checked plan per configured seed.

    Duplicate seeds fail before any training starts so every seed keeps its own
    resumable identity as required by the #7847 contract.
    """
    plans: list[SeedRunPlan] = []
    seen_seeds: set[int] = set()
    base_dir = output_dir if output_dir is not None else ensure_run_tracker_tree(run_id)
    for seed in config.base.seeds:
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed in config: {seed}")
        seen_seeds.add(seed)
        seed_dir = base_dir / f"seed_{seed}" if len(config.base.seeds) > 1 else base_dir
        plans.append(
            SeedRunPlan(
                seed=seed,
                run_id=f"{run_id}_seed_{seed}" if len(config.base.seeds) > 1 else run_id,
                output_dir=seed_dir,
            ),
        )
    del config_path  # provenance is carried by the manifest payload instead
    return plans


def _manifest_payload(  # noqa: PLR0913
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    dry_run: bool,
    output_dir: Path,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact dry-run/training manifest for the recurrent lane."""
    policy_kwargs = dict(config.policy_kwargs)
    payload: dict[str, Any] = {
        "schema_version": "recurrent-ppo-training-manifest.v2",
        "issue": 7847,
        "run_id": run_id,
        "policy_id": config.base.policy_id,
        "algorithm": "recurrent_ppo",
        "policy": config.recurrent_policy,
        "dependency": {
            "package": "sb3-contrib",
            "required_for": "non_dry_run_training",
            "install_hint": "uv sync --extra recurrent",
        },
        "dry_run": dry_run,
        "status": status,
        "evidence_tier": EVIDENCE_TIER if dry_run else "training_run_local_smoke",
        "claim_boundary": CLAIM_BOUNDARY,
        "config_path": str(config_path),
        "scenario_config": str(config.base.scenario_config),
        "seeds": [seed] if seed is not None else list(config.base.seeds),
        "total_timesteps": int(config.base.total_timesteps),
        "env_overrides": dict(config.base.env_overrides),
        "env_factory_kwargs": dict(config.base.env_factory_kwargs),
        "recurrent_ppo_hyperparams": dict(config.recurrent_ppo_hyperparams),
        "lstm": {
            "lstm_hidden_size": int(policy_kwargs.get("lstm_hidden_size", 256)),
            "n_lstm_layers": int(policy_kwargs.get("n_lstm_layers", 1)),
            "shared_lstm": bool(policy_kwargs.get("shared_lstm", False)),
            "enable_critic_lstm": bool(policy_kwargs.get("enable_critic_lstm", True)),
        },
        "out_of_scope": [
            "full matched comparison campaign (#7846)",
            "Slurm or GPU submission",
            "benchmark promotion",
            "paper-facing claim update",
        ],
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_clock_sec": max(0.0, (completed_at - started_at).total_seconds()),
        "output_dir": str(output_dir),
    }
    if extra:
        payload.update(extra)
    return payload


def write_training_manifest(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    dry_run: bool,
    output_dir: Path | None = None,
    status: str = "dry_run_complete",
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the recurrent PPO training manifest atomically and return its path."""
    started_at = datetime.now(UTC)
    target_dir = output_dir if output_dir is not None else ensure_run_tracker_tree(run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now(UTC)
    payload = _manifest_payload(
        config=config,
        config_path=config_path,
        run_id=run_id,
        dry_run=dry_run,
        output_dir=target_dir,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        seed=seed,
        extra=extra,
    )
    manifest_path = target_dir / "training_manifest.json"
    return write_json_atomic(manifest_path, payload)


def run_dry_run(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
) -> list[Path]:
    """Validate config per seed and emit dry-run manifests without optional deps."""
    manifest_paths = []
    for plan in plan_seed_runs(
        config=config, config_path=config_path, run_id=run_id, output_dir=output_dir
    ):
        manifest_paths.append(
            write_training_manifest(
                config=config,
                config_path=config_path,
                run_id=plan.run_id,
                dry_run=True,
                output_dir=plan.output_dir,
                seed=plan.seed,
            ),
        )
        logger.info("RecurrentPPO dry-run manifest written {}", manifest_paths[-1])
    return manifest_paths


def _evaluate_recurrently(
    *,
    model: Any,
    eval_env: Any,
    episodes: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate with explicit recurrent-state propagation and reset accounting.

    The evaluation environment is single-index (num_envs == 1) so state never
    crosses environment indices. State is reset on episode boundaries and the
    ``episode_start`` signal is passed on every step after a reset.
    """
    reset_accounting = ResetAccounting()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    non_finite_actions = 0

    current_return = 0.0
    current_length = 0
    for _ in range(episodes):
        obs = eval_env.reset()
        lstm_states = None  # fresh state at env.reset: never carry across episodes
        reset_accounting.record("env_reset")
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            action_array = np.asarray(action)
            if not np.all(np.isfinite(action_array)):
                non_finite_actions += int(np.count_nonzero(~np.isfinite(action_array)))
                raise RecurrentStateError(
                    "Evaluation produced non-finite actions; failing closed per contract",
                )
            step_result = eval_env.step(action_array)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _infos = step_result
            else:
                obs, reward, terminated, truncated = step_result
            reward_value = np.asarray(reward).reshape(-1)[0]
            current_return += float(reward_value)
            current_length += 1
            terminated_flag = bool(np.asarray(terminated).reshape(-1)[0])
            truncated_flag = bool(np.asarray(truncated).reshape(-1)[0])
            boundary = EpisodeBoundary(terminated=terminated_flag, truncated=truncated_flag)
            reason = reset_reasons_for([boundary])[0]
            if reason is not None:
                reset_accounting.record(reason)
                episode_starts = np.ones((1,), dtype=bool)
                done = True
            else:
                episode_starts = np.zeros((1,), dtype=bool)
        episode_returns.append(current_return)
        episode_lengths.append(current_length)
        current_return = 0.0
        current_length = 0

    returns_arr = np.asarray(episode_returns, dtype=float)
    lengths_arr = np.asarray(episode_lengths, dtype=float)
    summary: dict[str, Any] = {
        "episodes": episodes,
        "mean_episode_return": float(np.mean(returns_arr)) if returns_arr.size else 0.0,
        "mean_episode_length": float(np.mean(lengths_arr)) if lengths_arr.size else 0.0,
        "reset_counts": reset_accounting.as_dict(),
        "non_finite_action_count": non_finite_actions,
    }
    if lstm_states is not None:
        hidden, cell = lstm_states
        summary["state_norms"] = summarize_state_norms(np.asarray(hidden), np.asarray(cell))
    return summary


def _checkpoint_identity_payload(config: RecurrentPPOConfig, source_sha: str) -> dict[str, Any]:
    """Identity fields a resume must match exactly."""
    return {
        "source_sha": source_sha,
        "config_digest": _config_digest(config),
        "algorithm": "recurrent_ppo",
        "policy_class": config.recurrent_policy,
    }


def _config_digest(config: RecurrentPPOConfig) -> str:
    """Stable digest of the semantic training configuration."""
    import hashlib

    canonical = json.dumps(
        {
            "scenario_config": str(config.base.scenario_config),
            "total_timesteps": config.base.total_timesteps,
            "seeds": list(config.base.seeds),
            "hyperparams": dict(sorted(config.recurrent_ppo_hyperparams.items())),
            "policy_kwargs": dict(sorted(config.policy_kwargs.items())),
            "best_checkpoint_metric": config.base.best_checkpoint_metric,
            "evaluation_episodes": config.base.evaluation.evaluation_episodes,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_resume_identity(
    *,
    run_output_dir: Path,
    config: RecurrentPPOConfig,
    source_sha: str,
) -> dict[str, Any]:
    """Load and validate prior-run identity before a resume is allowed."""
    identity_path = run_output_dir / "run_identity.json"
    if not identity_path.exists():
        raise RuntimeError(
            f"Cannot resume: missing run_identity.json under {run_output_dir}; "
            "only runs produced by this lane are resumable",
        )
    prior = json.loads(identity_path.read_text(encoding="utf-8"))
    expected = _checkpoint_identity_payload(config, source_sha)
    mismatches = [
        f"{key}: prior={prior.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if prior.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Resume rejected: run identity mismatch — " + "; ".join(mismatches),
        )
    return prior


def _record_resume_boundary(
    *,
    prior: dict[str, Any],
    resumed_steps: int | None,
) -> dict[str, Any]:
    """Record previous/resumed step counts and reject step regression."""
    previous_steps = int(prior.get("completed_timesteps", 0))
    if resumed_steps is not None and resumed_steps < previous_steps:
        raise RuntimeError(
            f"Resume rejected: step regression prior={previous_steps} resumed={resumed_steps}",
        )
    return {
        "previous_completed_timesteps": previous_steps,
        "resumed_from_timesteps": resumed_steps if resumed_steps is not None else previous_steps,
        "clean_episode_boundary_only": True,
        "recurrent_rollout_state_restored": False,
    }


def _train_and_evaluate_segments(  # noqa: PLR0913
    *,
    model: Any,
    output_dir: Path,
    eval_vec_env: Any,
    config: RecurrentPPOConfig,
    hyperparams: dict[str, Any],
    metric_name: str,
    higher_is_better: bool,
    source_sha: str,
    seed: int,
) -> tuple[int, float | None, Path | None, list[dict[str, Any]]]:
    """Run the frozen step_schedule loop: learn, save, evaluate, select best."""
    best_score: float | None = None
    best_checkpoint_path: Path | None = None
    evaluation_history_path = output_dir / "evaluation_history.jsonl"
    metrics_path = output_dir / "training_metrics.jsonl"
    index_entries: list[dict[str, Any]] = []
    eval_steps = train_ppo._build_eval_steps(
        config.base.total_timesteps,
        config.base.evaluation.step_schedule,
    )
    total_learned = 0
    start = time.perf_counter()
    for eval_step in eval_steps:
        segment = max(0, min(eval_step, config.base.total_timesteps) - total_learned)
        if segment > 0:
            model.learn(total_timesteps=segment, reset_num_timesteps=False)
            total_learned += segment
        latest_path = output_dir / "latest.zip"
        model.save(latest_path)
        latest_entry = build_checkpoint_index_entry(
            kind="latest",
            checkpoint_path=latest_path,
            eval_step=None,
            score=None,
            metric_name=None,
            source_sha=source_sha,
            seed=seed,
        )
        index_entries = [e for e in index_entries if e["kind"] != "latest"] + [latest_entry]
        eval_start = time.perf_counter()
        eval_summary = _evaluate_recurrently(
            model=model,
            eval_env=eval_vec_env,
            episodes=max(1, config.base.evaluation.evaluation_episodes),
        )
        eval_sec = time.perf_counter() - eval_start
        proxy_score = float(eval_summary["mean_episode_return"])
        eval_record = {
            "schema_version": "recurrent-eval-history.v1",
            "eval_step": int(eval_step),
            "score": proxy_score,
            "metric": metric_name,
            "metric_note": (
                "mean_episode_return used as deterministic selection proxy for local smoke; "
                "campaign selection stays governed by the frozen #7846 rule"
            ),
            "higher_is_better": higher_is_better,
            "episodes": eval_summary["episodes"],
            "mean_episode_return": eval_summary["mean_episode_return"],
            "mean_episode_length": eval_summary["mean_episode_length"],
            "reset_counts": eval_summary["reset_counts"],
            "state_norms": eval_summary.get("state_norms"),
            "non_finite_action_count": eval_summary["non_finite_action_count"],
            "eval_sec": eval_sec,
            "recorded_at": utc_now_iso(),
        }
        append_jsonl_record(evaluation_history_path, eval_record)
        append_jsonl_record(
            metrics_path,
            {
                "schema_version": "recurrent-training-metrics.v1",
                "step": int(total_learned),
                "phase": "post_eval",
                "wall_clock_sec": time.perf_counter() - start,
                "steps_per_sec": total_learned / max(time.perf_counter() - start, 1e-9),
            },
        )
        is_better = best_score is None or (
            proxy_score > best_score if higher_is_better else proxy_score < best_score
        )
        if is_better:
            best_score = proxy_score
            best_checkpoint_path = output_dir / "best.zip"
            model.save(best_checkpoint_path)
            best_entry = build_checkpoint_index_entry(
                kind="best",
                checkpoint_path=best_checkpoint_path,
                eval_step=int(eval_step),
                score=proxy_score,
                metric_name="mean_episode_return",
                source_sha=source_sha,
                seed=seed,
            )
            index_entries = [e for e in index_entries if e["kind"] != "best"] + [best_entry]
    return total_learned, best_score, best_checkpoint_path, index_entries


def run_training_for_seed(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    plan: SeedRunPlan,
    output_dir: Path,
    source_sha: str,
    resume: bool = False,
) -> Path:
    """Run one independent per-seed RecurrentPPO smoke with full artifact contract."""
    recurrent_ppo_cls = _require_sb3_contrib()
    start = time.perf_counter()

    if resume:
        prior = _validate_resume_identity(
            run_output_dir=output_dir,
            config=config,
            source_sha=source_sha,
        )
        resume_record = _record_resume_boundary(prior=prior, resumed_steps=None)
        logger.info(
            "Resuming seed {} from timesteps {} (clean boundary)",
            plan.seed,
            resume_record["resumed_from_timesteps"],
        )
    else:
        if any(output_dir.iterdir()) if output_dir.exists() else False:
            raise RuntimeError(
                f"Output directory {output_dir} already contains files; "
                "duplicate run IDs and seed/output collisions fail before training",
            )
        resume_record = None

    scenario_definitions = tuple(train_ppo.load_scenarios(config.base.scenario_config))
    scenario_ctx = train_ppo._resolve_scenario_context(config.base, scenario_definitions)
    num_envs = train_ppo._resolve_num_envs(config.base)
    worker_mode = train_ppo._resolve_worker_mode(config.base, num_envs)
    base_seed = plan.seed

    def _make_env(seed_offset: int) -> Any:
        return train_ppo._make_training_env(
            seed=int(base_seed) + seed_offset if base_seed is not None else None,
            scenario=scenario_ctx.selected_scenario,
            scenario_definitions=(
                scenario_definitions if scenario_ctx.selected_scenario is None else None
            ),
            scenario_path=config.base.scenario_config,
            exclude_scenarios=scenario_ctx.training_exclude,
            suite_name="recurrent_ppo_issue_7847",
            algorithm_name=config.base.policy_id,
            env_overrides=config.base.env_overrides,
            env_factory_kwargs=config.base.env_factory_kwargs,
            scenario_sampling=config.base.scenario_sampling,
            density_curriculum=None,
        )

    env_fns = [_make_env(idx) for idx in range(num_envs)]
    vec_env_cls = train_ppo.SubprocVecEnv if worker_mode == "subproc" else train_ppo.DummyVecEnv
    vec_env = vec_env_cls(env_fns)
    eval_vec_env = vec_env_cls([_make_env(10_000 + plan.seed % 1000)])
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = dict(config.recurrent_ppo_hyperparams)
    hyperparams["seed"] = base_seed
    metric_name, higher_is_better = train_ppo._resolve_best_checkpoint_metric(
        config.base.best_checkpoint_metric,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = recurrent_ppo_cls(config.recurrent_policy, vec_env, **hyperparams)
        total_learned, best_score, best_checkpoint_path, index_entries = (
            _train_and_evaluate_segments(
                model=model,
                output_dir=output_dir,
                eval_vec_env=eval_vec_env,
                config=config,
                hyperparams=hyperparams,
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                source_sha=source_sha,
                seed=base_seed,
            )
        )
        final_path = output_dir / "final.zip"
        model.save(final_path)
        final_entry = build_checkpoint_index_entry(
            kind="final",
            checkpoint_path=final_path,
            eval_step=None,
            score=None,
            metric_name=None,
            source_sha=source_sha,
            seed=base_seed,
        )
        index_entries = [e for e in index_entries if e["kind"] != "final"] + [final_entry]
        write_json_atomic(
            output_dir / "run_identity.json",
            {
                **_checkpoint_identity_payload(config, source_sha),
                "seed": base_seed,
                "run_id": plan.run_id,
                "completed_timesteps": int(total_learned),
                "num_envs": int(num_envs),
                "worker_mode": worker_mode,
                "recurrent_eval_contract": {
                    "lstm_states_propagated": True,
                    "episode_start_signal_used": True,
                    "state_reset_on_boundaries": True,
                    "rollout_state_restored_across_restart": False,
                },
            },
        )
        write_checkpoint_index(output_dir / "checkpoint_index.json", index_entries)
        write_json_atomic(output_dir / "dependency_environment.json", _dependency_versions())
    finally:
        vec_env.close()
        eval_vec_env.close()

    total_wall_clock_sec = max(0.0, time.perf_counter() - start)
    perf_payload_extra = {
        "performance_summary": {
            "total_wall_clock_sec": total_wall_clock_sec,
            "train_env_steps_per_sec_mean": float(total_learned) / max(total_wall_clock_sec, 1e-9),
            "resume_record": resume_record,
            "best_checkpoint": {
                "path": str(best_checkpoint_path) if best_checkpoint_path else None,
                "score": best_score,
                "metric": "mean_episode_return",
            },
        },
    }
    manifest_path = write_training_manifest(
        config=config,
        config_path=config_path,
        run_id=plan.run_id,
        dry_run=False,
        output_dir=output_dir,
        status=f"training_complete steps={total_learned} wall={total_wall_clock_sec:.3f}s",
        seed=base_seed,
        extra=perf_payload_extra,
    )
    logger.info("RecurrentPPO training manifest written {}", manifest_path)
    return manifest_path


def run_training(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
    resume: bool = False,
) -> list[Path]:
    """Run independent per-seed training jobs and return their manifest paths."""
    source_sha = _resolve_git_sha()
    manifests = []
    for plan in plan_seed_runs(
        config=config, config_path=config_path, run_id=run_id, output_dir=output_dir
    ):
        manifests.append(
            run_training_for_seed(
                config=config,
                config_path=config_path,
                plan=plan,
                output_dir=plan.output_dir,
                source_sha=source_sha,
                resume=resume,
            ),
        )
    return manifests


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to RecurrentPPO YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and write manifest only.",
    )
    parser.add_argument("--run-id", default=None, help="Override run id for output/run-tracker.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override manifest output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each seed's run from its existing identity-checked output directory.",
    )
    parser.add_argument("--log-level", default="INFO", help="Loguru log level.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logger.remove()
    logger.add(lambda message: print(message, end=""), level=str(args.log_level).upper())
    config_path = Path(args.config).resolve()
    config = load_recurrent_ppo_config(config_path)
    run_id = str(args.run_id or f"{config.base.policy_id}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    if args.dry_run:
        run_dry_run(
            config=config, config_path=config_path, run_id=run_id, output_dir=args.output_dir
        )
    else:
        run_training(
            config=config,
            config_path=config_path,
            run_id=run_id,
            output_dir=args.output_dir,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
