"""SAC (Soft Actor-Critic) training script for Robot SF.

Train an off-policy SAC policy via Stable-Baselines3 on Robot SF navigation
scenarios.  The script follows the same config-first pattern as
``scripts/training/train_ppo.py``:

1. Load a YAML training config.
2. Build the Gymnasium environment via the existing scenario/env factory.
3. Instantiate SB3 SAC and train.
4. Save a checkpoint and (optionally) log to W&B.

Example usage::

    uv run python scripts/training/train_sac_sb3.py \\
        --config configs/training/sac/gate.yaml

Dry-run (no actual training, fast smoke test)::

    uv run python scripts/training/train_sac_sb3.py \\
        --config configs/training/sac/gate.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from gymnasium import ObservationWrapper
from gymnasium import spaces as gym_spaces
from loguru import logger

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:
    raise RuntimeError("Stable-Baselines3 must be installed to run SAC training.") from exc

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)

# ---------------------------------------------------------------------------
# Supported SAC hyperparameter keys (mirrors SB3 SAC constructor arguments).
# ---------------------------------------------------------------------------
_ALLOWED_SAC_HYPERPARAMS: frozenset[str] = frozenset(
    {
        "learning_rate",
        "buffer_size",
        "batch_size",
        "tau",
        "gamma",
        "train_freq",
        "gradient_steps",
        "ent_coef",
        "target_entropy",
        "learning_starts",
        "optimize_memory_usage",
    }
)

_DEFAULT_SAC_HYPERPARAMS: dict[str, object] = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "learning_starts": 1_000,
}

_DRY_RUN_TIMESTEPS = 64

_ALLOWED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "policy_id",
        "scenario_config",
        "total_timesteps",
        "sac_hyperparams",
        "env_overrides",
        "env_factory_kwargs",
        "tracking",
        "output_dir",
        "seed",
    }
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SACTrainingConfig:
    """Typed container for SAC training configuration."""

    policy_id: str
    scenario_config: Path
    total_timesteps: int
    sac_hyperparams: dict[str, object] = field(default_factory=dict)
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)
    tracking: dict[str, object] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("output/models/sac"))
    seed: int | None = None


def load_sac_training_config(config_path: str | Path) -> SACTrainingConfig:
    """Load :class:`SACTrainingConfig` from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        SACTrainingConfig: Populated configuration instance.
    """
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a YAML mapping; got {type(data)!r}")

    unknown_keys = set(data) - _ALLOWED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown root config keys: {sorted(unknown_keys)}. "
            f"Allowed: {sorted(_ALLOWED_CONFIG_KEYS)}"
        )

    # Resolve scenario_config relative to the config file location.
    scenario_raw = Path(data["scenario_config"])
    scenario_config = (
        (path.parent / scenario_raw).resolve()
        if not scenario_raw.is_absolute()
        else scenario_raw.resolve()
    )

    raw_hyperparams = dict(data.get("sac_hyperparams", {}) or {})
    unknown = set(raw_hyperparams) - _ALLOWED_SAC_HYPERPARAMS
    if unknown:
        raise ValueError(
            f"Unknown SAC hyperparameter keys: {sorted(unknown)}.  "
            f"Allowed: {sorted(_ALLOWED_SAC_HYPERPARAMS)}"
        )

    output_dir_raw = data.get("output_dir", "output/models/sac")
    output_dir = _resolve_output_dir(output_dir_raw)

    seed_raw = data.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None

    return SACTrainingConfig(
        policy_id=str(data["policy_id"]),
        scenario_config=scenario_config,
        total_timesteps=int(data["total_timesteps"]),
        sac_hyperparams=raw_hyperparams,
        env_overrides=dict(data.get("env_overrides", {}) or {}),
        env_factory_kwargs=dict(data.get("env_factory_kwargs", {}) or {}),
        tracking=dict(data.get("tracking", {}) or {}),
        output_dir=output_dir,
        seed=seed,
    )


def _resolve_output_dir(raw_value: object) -> Path:
    """Resolve output directories against the repository working tree.

    Returns:
        Path: Absolute output directory path.
    """
    path_value = Path(str(raw_value).strip())
    if not path_value.is_absolute():
        path_value = (Path.cwd() / path_value).resolve()
    return path_value


# ---------------------------------------------------------------------------
# Environment factory helper
# ---------------------------------------------------------------------------


def _build_env(
    config: SACTrainingConfig,
    *,
    scenario_definitions: list[Mapping[str, Any]],
) -> DummyVecEnv:
    """Build a vectorised single-env wrapper for SAC training.

    Args:
        config: SAC training configuration.
        scenario_definitions: List of scenario definitions loaded from YAML.

    Returns:
        DummyVecEnv: A single-environment vectorised wrapper.
    """
    scenario = select_scenario(scenario_definitions, scenario_id=None)
    robot_config = build_robot_config_from_scenario(
        scenario,
        scenario_path=config.scenario_config,
    )
    _apply_env_overrides(robot_config, config.env_overrides)

    env_kwargs: dict[str, Any] = {
        "config": robot_config,
        **config.env_factory_kwargs,
    }
    if config.seed is not None:
        env_kwargs["seed"] = config.seed

    def _make() -> Any:
        env = make_robot_env(**env_kwargs)
        return _maybe_flatten_nested_dict_env(env)

    return DummyVecEnv([_make])


def _apply_env_overrides(robot_config: Any, overrides: Mapping[str, object]) -> None:
    """Apply a minimal set of config-driven environment overrides.

    SAC currently needs this primarily to switch between `default_gym` and
    benchmark-compatible `socnav_struct` observations without forking the script.
    """
    if not overrides:
        return

    for key, value in overrides.items():
        if key == "observation_mode":
            robot_config.observation_mode = ObservationMode(str(value))
            continue
        if not hasattr(robot_config, key):
            raise ValueError(f"Unknown env_overrides key: {key}")
        setattr(robot_config, key, value)


def _flatten_nested_dict_spaces(obs_space: gym_spaces.Dict) -> gym_spaces.Dict:
    """Flatten nested dict spaces to a single top-level Dict for SB3 compatibility."""
    flattened: dict[str, gym_spaces.Space] = {}

    def _walk(space_dict: dict[str, gym_spaces.Space], prefix: str = "") -> None:
        for key, subspace in space_dict.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
            if isinstance(subspace, gym_spaces.Dict):
                _walk(subspace.spaces, full_key)
            else:
                flattened[full_key] = subspace

    _walk(obs_space.spaces)
    return gym_spaces.Dict(flattened)


def _flatten_nested_dict_obs(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested dict observations to match `_flatten_nested_dict_spaces`."""
    flattened: dict[str, Any] = {}

    def _walk(obs_dict: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in obs_dict.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
            if isinstance(value, Mapping):
                _walk(value, full_key)
            else:
                flattened[full_key] = value

    _walk(obs)
    return flattened


class _FlattenNestedDictObservation(ObservationWrapper):
    """Observation wrapper that flattens nested Dict observations into a flat Dict."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.observation_space = _flatten_nested_dict_spaces(env.observation_space)

    def observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return _flatten_nested_dict_obs(observation)


def _has_nested_dict_space(obs_space: gym_spaces.Space) -> bool:
    """Return whether the observation space contains Dict-valued children."""
    return isinstance(obs_space, gym_spaces.Dict) and any(
        isinstance(subspace, gym_spaces.Dict) for subspace in obs_space.spaces.values()
    )


def _maybe_flatten_nested_dict_env(env: Any) -> Any:
    """Flatten nested Dict observations without collapsing them to a Box."""
    obs_space = getattr(env, "observation_space", None)
    if _has_nested_dict_space(obs_space):
        return _FlattenNestedDictObservation(env)
    return env


def _resolve_policy_name(observation_space: gym_spaces.Space) -> str:
    """Return the SB3 policy class name matching the env observation contract.

    Returns:
        str: ``MultiInputPolicy`` for dict observations, otherwise
            ``MlpPolicy``.
    """
    if isinstance(observation_space, gym_spaces.Dict):
        return "MultiInputPolicy"
    return "MlpPolicy"


# ---------------------------------------------------------------------------
# W&B integration (optional)
# ---------------------------------------------------------------------------


def _maybe_init_wandb(
    config: SACTrainingConfig,
    *,
    dry_run: bool,
) -> object | None:
    """Initialise W&B run when tracking.enabled=true in the config.

    Args:
        config: SAC training configuration.
        dry_run: When True, skip W&B initialisation.

    Returns:
        object | None: W&B run object or None.
    """
    tracking = config.tracking
    if not tracking.get("enabled", False) or dry_run:
        return None
    try:
        import wandb  # type: ignore[import-untyped]

        return wandb.init(
            project=str(tracking.get("project", "robot_sf_sac")),
            name=str(tracking.get("run_name", config.policy_id)),
            tags=list(tracking.get("tags", [])),
            config={
                "policy_id": config.policy_id,
                "total_timesteps": config.total_timesteps,
                "seed": config.seed,
                **config.sac_hyperparams,
            },
        )
    except ImportError:
        logger.warning("wandb not installed; skipping W&B tracking.")
        return None


# ---------------------------------------------------------------------------
# Core training entry point
# ---------------------------------------------------------------------------


def run_sac_training(
    config: SACTrainingConfig,
    *,
    dry_run: bool = False,
) -> Path:
    """Execute SAC training and persist a checkpoint.

    Args:
        config: SAC training configuration.
        dry_run: When True, skip actual training (useful for CI/smoke tests).

    Returns:
        Path: Path to the saved checkpoint file.
    """
    logger.info(
        "Starting SAC training: policy_id={}, total_timesteps={}, dry_run={}",
        config.policy_id,
        config.total_timesteps,
        dry_run,
    )

    scenario_definitions = load_scenarios(config.scenario_config)
    vec_env = _build_env(config, scenario_definitions=scenario_definitions)
    try:
        # Merge defaults with config overrides.
        hyperparams = {**_DEFAULT_SAC_HYPERPARAMS, **config.sac_hyperparams}
        policy_kwargs: dict[str, Any] = {"net_arch": [256, 256]}
        policy_name = _resolve_policy_name(vec_env.observation_space)

        model = SAC(
            policy_name,
            vec_env,
            verbose=1,
            seed=config.seed,
            policy_kwargs=policy_kwargs,
            **hyperparams,  # type: ignore[arg-type]
        )

        wandb_run = _maybe_init_wandb(config, dry_run=dry_run)
        timesteps = _DRY_RUN_TIMESTEPS if dry_run else config.total_timesteps
        try:
            model.learn(total_timesteps=timesteps, log_interval=100)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.output_dir / f"{config.policy_id}.zip"
        model.save(str(checkpoint_path))
        logger.info("Checkpoint saved to {}", checkpoint_path)
        return checkpoint_path
    finally:
        vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the SAC training script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train a SAC policy on Robot SF navigation scenarios."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML training configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Run only a short 64-step smoke test of the pipeline (no real training)."),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ),
        help="Console log level (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for SAC training.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        int: Exit code (0 on success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_level = str(args.log_level).upper()
    os.environ["LOGURU_LEVEL"] = log_level
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    config = load_sac_training_config(args.config)
    run_sac_training(config, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
