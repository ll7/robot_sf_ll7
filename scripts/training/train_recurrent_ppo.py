"""RecurrentPPO LSTM training lane for issue #4014.

The regular ``train_ppo.py`` path can use an LSTM feature extractor, but that
extractor is not step-recurrent. This entry point is intentionally separate so
the issue #4014 LSTM row can use ``sb3_contrib.RecurrentPPO`` and keep its claim
boundary explicit. ``--dry-run`` validates the config and writes the manifest
without importing the optional dependency.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.common.artifact_paths import ensure_run_tracker_tree
from scripts.training import train_ppo

CLAIM_BOUNDARY = (
    "dry-run/smoke-prep for true sb3_contrib.RecurrentPPO LSTM; "
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


@dataclass(frozen=True, slots=True)
class RecurrentPPOConfig:
    """Validated RecurrentPPO lane config."""

    base: train_ppo.ExpertTrainingConfig
    algorithm: str
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
    """Load and validate the issue #4014 recurrent PPO config."""
    path = Path(config_path).resolve()
    raw = _load_raw_config(path)
    algorithm = str(raw.get("algorithm", "")).strip().lower()
    if algorithm != "recurrent_ppo":
        raise ValueError("algorithm must be 'recurrent_ppo' for train_recurrent_ppo.py")

    hyperparams = dict(raw.get("recurrent_ppo_hyperparams", {}) or {})
    unknown = set(hyperparams) - _ALLOWED_RECURRENT_PPO_HYPERPARAMS
    if unknown:
        raise ValueError(
            f"recurrent_ppo_hyperparams unsupported keys: {', '.join(sorted(unknown))}"
        )

    policy_kwargs = dict(hyperparams.get("policy_kwargs", {}) or {})
    unknown_policy_kwargs = set(policy_kwargs) - _ALLOWED_POLICY_KWARGS
    if unknown_policy_kwargs:
        raise ValueError(
            "recurrent_ppo_hyperparams.policy_kwargs unsupported keys: "
            f"{', '.join(sorted(unknown_policy_kwargs))}"
        )

    for key in ("n_steps", "batch_size", "n_epochs"):
        if key in hyperparams and int(hyperparams[key]) <= 0:
            raise ValueError(f"recurrent_ppo_hyperparams.{key} must be positive")
    for key in ("lstm_hidden_size", "n_lstm_layers"):
        if key in policy_kwargs and int(policy_kwargs[key]) <= 0:
            raise ValueError(f"recurrent_ppo_hyperparams.policy_kwargs.{key} must be positive")

    base = train_ppo.load_expert_training_config(path)
    return RecurrentPPOConfig(
        base=base,
        algorithm=algorithm,
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


def _manifest_payload(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    dry_run: bool,
    output_dir: Path,
    status: str,
    started_at: datetime,
    completed_at: datetime,
) -> dict[str, Any]:
    """Build the compact dry-run/training manifest for the recurrent lane."""
    policy_kwargs = dict(config.policy_kwargs)
    return {
        "schema_version": "recurrent-ppo-training-manifest.v1",
        "issue": 4014,
        "run_id": run_id,
        "policy_id": config.base.policy_id,
        "algorithm": "recurrent_ppo",
        "policy": "MultiInputLstmPolicy",
        "dependency": {
            "package": "sb3-contrib",
            "required_for": "non_dry_run_training",
            "install_hint": "uv sync --extra recurrent",
        },
        "dry_run": dry_run,
        "status": status,
        "evidence_tier": EVIDENCE_TIER if dry_run else "training_run",
        "claim_boundary": CLAIM_BOUNDARY,
        "config_path": str(config_path),
        "scenario_config": str(config.base.scenario_config),
        "seeds": list(config.base.seeds),
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
            "Mamba extractor rework",
            "full three-way training comparison",
            "Slurm or GPU submission",
            "benchmark campaign",
            "paper-facing claim update",
        ],
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_clock_sec": max(0.0, (completed_at - started_at).total_seconds()),
        "output_dir": str(output_dir),
    }


def write_training_manifest(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    dry_run: bool,
    output_dir: Path | None = None,
    status: str = "dry_run_complete",
) -> Path:
    """Write the recurrent PPO training manifest and return its path."""
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
    )
    manifest_path = target_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def run_dry_run(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
) -> Path:
    """Validate config and emit a dry-run manifest without optional dependency imports."""
    manifest_path = write_training_manifest(
        config=config,
        config_path=config_path,
        run_id=run_id,
        dry_run=True,
        output_dir=output_dir,
    )
    logger.info("RecurrentPPO dry-run manifest written {}", manifest_path)
    return manifest_path


def run_training(
    *,
    config: RecurrentPPOConfig,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
) -> Path:
    """Run a minimal true RecurrentPPO training job and write a manifest.

    This path is intentionally small; the issue #4014 slice is dry-run/smoke-prep,
    while the full matched comparison remains out of scope.
    """
    recurrent_ppo_cls = _require_sb3_contrib()
    start = time.perf_counter()
    scenario_definitions = tuple(train_ppo.load_scenarios(config.base.scenario_config))
    scenario_ctx = train_ppo._resolve_scenario_context(config.base, scenario_definitions)
    num_envs = train_ppo._resolve_num_envs(config.base)
    worker_mode = train_ppo._resolve_worker_mode(config.base, num_envs)
    env_fns = [
        train_ppo._make_training_env(
            seed=int(config.base.seeds[0]) + idx if config.base.seeds else None,
            scenario=scenario_ctx.selected_scenario,
            scenario_definitions=scenario_definitions
            if scenario_ctx.selected_scenario is None
            else None,
            scenario_path=config.base.scenario_config,
            exclude_scenarios=scenario_ctx.training_exclude,
            suite_name="recurrent_ppo_issue_4014",
            algorithm_name=config.base.policy_id,
            env_overrides=config.base.env_overrides,
            env_factory_kwargs=config.base.env_factory_kwargs,
            scenario_sampling=config.base.scenario_sampling,
            density_curriculum=None,
        )
        for idx in range(num_envs)
    ]
    vec_env_cls = train_ppo.SubprocVecEnv if worker_mode == "subproc" else train_ppo.DummyVecEnv
    vec_env = vec_env_cls(env_fns)
    try:
        hyperparams = dict(config.recurrent_ppo_hyperparams)
        model = recurrent_ppo_cls("MultiInputLstmPolicy", vec_env, **hyperparams)
        model.learn(total_timesteps=config.base.total_timesteps)
        target_dir = output_dir if output_dir is not None else ensure_run_tracker_tree(run_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        model.save(target_dir / "model.zip")
    finally:
        vec_env.close()

    manifest_path = write_training_manifest(
        config=config,
        config_path=config_path,
        run_id=run_id,
        dry_run=False,
        output_dir=target_dir,
        status=f"training_complete in {time.perf_counter() - start:.3f}s",
    )
    logger.info("RecurrentPPO training manifest written {}", manifest_path)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to RecurrentPPO YAML config.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and write manifest only."
    )
    parser.add_argument("--run-id", default=None, help="Override run id for output/run-tracker.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override manifest output directory."
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
            config=config, config_path=config_path, run_id=run_id, output_dir=args.output_dir
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
