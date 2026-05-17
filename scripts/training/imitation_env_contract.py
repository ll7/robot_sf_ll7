"""Shared environment-contract helpers for imitation warm-start workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from gymnasium import spaces
from gymnasium.wrappers import FilterObservation
from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)
from scripts.training.train_ppo import _apply_env_overrides

if TYPE_CHECKING:
    from collections.abc import Sequence


def resolve_config_path(raw_value: object, *, base_dir: Path) -> Path | None:
    """Resolve an optional path value relative to a config file directory."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _load_training_config_mapping(training_config_path: Path | None) -> dict[str, Any] | None:
    """Load a training config file and require a mapping payload when provided."""
    if training_config_path is None:
        return None
    if not training_config_path.is_file():
        raise FileNotFoundError(f"Training config is not a file: {training_config_path}")
    raw = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("training config must be a mapping")
    return raw


def load_training_env_overrides(training_config_path: Path | None) -> dict[str, object]:
    """Load ``env_overrides`` from an expert training config."""
    raw = _load_training_config_mapping(training_config_path)
    if raw is None:
        return {}
    overrides = raw.get("env_overrides", {})
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("training config env_overrides must be a mapping")
    return dict(overrides)


def load_training_env_factory_kwargs(training_config_path: Path | None) -> dict[str, object]:
    """Load ``env_factory_kwargs`` from an expert training config."""
    raw = _load_training_config_mapping(training_config_path)
    if raw is None:
        return {}
    kwargs = raw.get("env_factory_kwargs", {})
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise ValueError("training config env_factory_kwargs must be a mapping")
    return dict(kwargs)


def resolve_scenario_config_path(
    *,
    scenario_config_path: Path | None,
    training_config_path: Path | None,
) -> Path | None:
    """Return the scenario config path from explicit config or training config."""
    if scenario_config_path is not None:
        return scenario_config_path
    raw = _load_training_config_mapping(training_config_path)
    if raw is None:
        return None
    assert training_config_path is not None
    return resolve_config_path(raw.get("scenario_config"), base_dir=training_config_path.parent)


def make_training_contract_env(
    *,
    training_config_path: Path | None,
    scenario_config_path: Path | None,
    scenario_id: str | None = None,
    seed: int | None = None,
    observation_keys: Sequence[str] | None = None,
    env_overrides: dict[str, object] | None = None,
    env_factory_kwargs: dict[str, object] | None = None,
) -> Any:
    """Create an env using training-config overrides and optional observation-key filtering."""
    resolved_scenario_path = resolve_scenario_config_path(
        scenario_config_path=scenario_config_path,
        training_config_path=training_config_path,
    )
    if resolved_scenario_path is None:
        env_config = RobotSimulationConfig()
    else:
        scenarios = load_scenarios(resolved_scenario_path)
        selected_scenario = select_scenario(scenarios, scenario_id)
        env_config = build_robot_config_from_scenario(
            selected_scenario,
            scenario_path=resolved_scenario_path,
        )

    resolved_overrides = load_training_env_overrides(training_config_path)
    resolved_overrides.update(env_overrides or {})
    resolved_factory_kwargs = load_training_env_factory_kwargs(training_config_path)
    resolved_factory_kwargs.update(env_factory_kwargs or {})

    _apply_env_overrides(env_config, resolved_overrides)
    env = make_robot_env(config=env_config, seed=seed, **resolved_factory_kwargs)
    if observation_keys:
        obs_space = getattr(env, "observation_space", None)
        if not isinstance(obs_space, spaces.Dict):
            raise ValueError(
                "Cannot filter env observation space; expected Dict observation space, got "
                f"{type(obs_space).__name__}"
            )
        available = set(obs_space.spaces)
        missing = [key for key in observation_keys if key not in available]
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(f"Cannot filter env observation space; missing keys: {preview}")
        logger.info(
            "Filtering env observation keys to imitation contract: {}", list(observation_keys)
        )
        env = FilterObservation(env, list(observation_keys))
    return env


def observation_contract_from_space(observation_space: Any) -> dict[str, object]:
    """Serialize the subset of observation-space metadata needed by imitation stages."""
    if not isinstance(observation_space, spaces.Dict):
        return {
            "type": type(observation_space).__name__,
            "shape": list(getattr(observation_space, "shape", ()) or ()),
            "dtype": str(getattr(observation_space, "dtype", "")),
        }
    keys = [str(key) for key in observation_space.spaces]
    return {
        "type": "Dict",
        "keys": keys,
        "spaces": {
            str(key): {
                "shape": list(getattr(subspace, "shape", ()) or ()),
                "dtype": str(getattr(subspace, "dtype", "")),
            }
            for key, subspace in observation_space.spaces.items()
        },
    }
