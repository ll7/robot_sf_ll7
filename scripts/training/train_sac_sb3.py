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

import gymnasium
import numpy as np
import yaml
from gymnasium import ObservationWrapper
from gymnasium import spaces as gym_spaces
from loguru import logger

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
except ImportError as exc:
    raise RuntimeError("Stable-Baselines3 must be installed to run SAC training.") from exc

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.sensor.socnav_observation import SOCNAV_POSITION_CAP_M
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
)
from robot_sf.training.scenario_sampling import ScenarioSampler, ScenarioSwitchingEnv
from scripts.validation import evaluate_sac as sac_eval

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
        "scenario_sampling",
        "evaluation",
        "tracking",
        "output_dir",
        "seed",
        "num_envs",
        "device",
        "action_semantics",
        "relative_obs",
        "obs_transform",
    }
)

# Action semantics options:
# "delta"    — SAC outputs delta velocities. This is now the canonical setting
#              because map_runner can pass SAC commands directly to env.step()
#              through the _planner_native_env_action hook.
# "absolute" — Experimental path that wraps target velocities back into deltas
#              for the DifferentialDrive env. Kept for comparison/debugging.
_DEFAULT_ACTION_SEMANTICS = "delta"
_DEFAULT_OBS_TRANSFORM = "none"
_DEFAULT_EVAL_SCENARIO_MATRIX = Path("configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml")
_DEFAULT_EVAL_ALGO_CONFIG = Path("configs/baselines/sac_gate_socnav_struct.yaml")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SACEvaluationConfig:
    """Periodic benchmark-evaluation settings for SAC training."""

    enabled: bool = False
    frequency_steps: int = 0
    scenario_matrix: Path | None = None
    algo_config: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("output/tmp/sac_eval"))
    tag_prefix: str = "sac_eval"
    horizon: int = 120
    dt: float = 0.1
    workers: int = 1
    min_success_rate: float = 0.3
    device: str | None = None


@dataclass
class SACScenarioSamplingConfig:
    """Scenario sampling settings for SAC training."""

    include_scenarios: tuple[str, ...] = field(default_factory=tuple)
    exclude_scenarios: tuple[str, ...] = field(default_factory=tuple)
    weights: dict[str, float] = field(default_factory=dict)
    strategy: str = "random"


@dataclass
class SACTrainingConfig:
    """Typed container for SAC training configuration."""

    policy_id: str
    scenario_config: Path
    total_timesteps: int
    sac_hyperparams: dict[str, object] = field(default_factory=dict)
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)
    scenario_sampling: SACScenarioSamplingConfig = field(default_factory=SACScenarioSamplingConfig)
    tracking: dict[str, object] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("output/models/sac"))
    seed: int | None = None
    num_envs: int = 1
    device: str = "auto"
    action_semantics: str = _DEFAULT_ACTION_SEMANTICS
    relative_obs: bool = True
    obs_transform: str = _DEFAULT_OBS_TRANSFORM
    evaluation: SACEvaluationConfig = field(default_factory=SACEvaluationConfig)


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
    output_dir = _resolve_output_dir(output_dir_raw, base_dir=path.parent)

    seed_raw = data.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None
    num_envs_raw = data.get("num_envs", 1)
    num_envs = int(num_envs_raw)
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1; got {num_envs}")
    device = str(data.get("device", "auto")).strip() or "auto"
    action_semantics = (
        str(data.get("action_semantics", _DEFAULT_ACTION_SEMANTICS)).strip().lower()
        or _DEFAULT_ACTION_SEMANTICS
    )
    relative_obs_raw = data.get("relative_obs", True)
    relative_obs = (
        bool(relative_obs_raw) if not isinstance(relative_obs_raw, bool) else relative_obs_raw
    )
    obs_transform = (
        str(data.get("obs_transform", _DEFAULT_OBS_TRANSFORM)).strip().lower()
        or _DEFAULT_OBS_TRANSFORM
    )
    if obs_transform == _DEFAULT_OBS_TRANSFORM and relative_obs:
        obs_transform = "relative"
    evaluation = _load_eval_config(data.get("evaluation", {}) or {}, config_dir=path.parent)

    return SACTrainingConfig(
        policy_id=str(data["policy_id"]),
        scenario_config=scenario_config,
        total_timesteps=int(data["total_timesteps"]),
        sac_hyperparams=raw_hyperparams,
        env_overrides=dict(data.get("env_overrides", {}) or {}),
        env_factory_kwargs=dict(data.get("env_factory_kwargs", {}) or {}),
        scenario_sampling=_load_scenario_sampling_config(
            data.get("scenario_sampling"),
            config_dir=path.parent,
        ),
        tracking=dict(data.get("tracking", {}) or {}),
        output_dir=output_dir,
        seed=seed,
        num_envs=num_envs,
        device=device,
        action_semantics=action_semantics,
        relative_obs=relative_obs,
        obs_transform=obs_transform,
        evaluation=evaluation,
    )


def _resolve_output_dir(raw_value: object, *, base_dir: Path) -> Path:
    """Resolve output directories against the training config location.

    Returns:
        Path: Absolute output directory path.
    """
    path_value = Path(str(raw_value).strip())
    if not path_value.is_absolute():
        path_value = (base_dir / path_value).resolve()
    return path_value


def _resolve_config_path(raw_value: object | None, *, config_dir: Path) -> Path | None:
    """Resolve an optional config path relative to the training config file."""
    if raw_value in (None, ""):
        return None
    raw_path = Path(str(raw_value))
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (config_dir / raw_path).resolve()


def _load_eval_config(raw_value: object, *, config_dir: Path) -> SACEvaluationConfig:
    """Load periodic evaluation settings from the training config."""
    if not isinstance(raw_value, Mapping):
        raise ValueError(f"evaluation block must be a mapping; got {type(raw_value)!r}")

    unknown = set(raw_value) - {
        "enabled",
        "frequency_steps",
        "scenario_matrix",
        "algo_config",
        "output_dir",
        "tag_prefix",
        "horizon",
        "dt",
        "workers",
        "min_success_rate",
        "device",
    }
    if unknown:
        raise ValueError(f"Unknown evaluation config keys: {sorted(unknown)}")

    return SACEvaluationConfig(
        enabled=bool(raw_value.get("enabled", False)),
        frequency_steps=int(raw_value.get("frequency_steps", 0) or 0),
        scenario_matrix=_resolve_config_path(
            raw_value.get("scenario_matrix"), config_dir=config_dir
        )
        or None,
        algo_config=_resolve_config_path(raw_value.get("algo_config"), config_dir=config_dir),
        output_dir=_resolve_output_dir(
            raw_value.get("output_dir", "output/tmp/sac_eval"),
            base_dir=config_dir,
        ),
        tag_prefix=str(raw_value.get("tag_prefix", "sac_eval")).strip() or "sac_eval",
        horizon=int(raw_value.get("horizon", 120)),
        dt=float(raw_value.get("dt", 0.1)),
        workers=int(raw_value.get("workers", 1)),
        min_success_rate=float(raw_value.get("min_success_rate", 0.3)),
        device=(
            str(raw_value.get("device")).strip()
            if raw_value.get("device") not in (None, "")
            else None
        ),
    )


def _load_scenario_sampling_config(
    raw_value: object | None,
    *,
    config_dir: Path,
) -> SACScenarioSamplingConfig:
    """Load scenario sampling settings from the training config."""
    if raw_value in (None, {}):
        return SACScenarioSamplingConfig()
    if not isinstance(raw_value, Mapping):
        raise ValueError(f"scenario_sampling block must be a mapping; got {type(raw_value)!r}")

    unknown = set(raw_value) - {
        "include_scenarios",
        "exclude_scenarios",
        "weights",
        "strategy",
    }
    if unknown:
        raise ValueError(f"Unknown scenario_sampling config keys: {sorted(unknown)}")

    include_scenarios = tuple(str(x).strip() for x in raw_value.get("include_scenarios", ()) or [])
    exclude_scenarios = tuple(str(x).strip() for x in raw_value.get("exclude_scenarios", ()) or [])
    weights = {str(k): float(v) for k, v in dict(raw_value.get("weights", {}) or {}).items()}
    strategy = str(raw_value.get("strategy", "random")).strip().lower() or "random"
    if strategy not in {"random", "cycle"}:
        raise ValueError(
            f"Unknown scenario_sampling.strategy: {strategy}; expected 'random' or 'cycle'"
        )

    return SACScenarioSamplingConfig(
        include_scenarios=include_scenarios,
        exclude_scenarios=exclude_scenarios,
        weights=weights,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Environment factory helper
# ---------------------------------------------------------------------------


def _build_env(
    config: SACTrainingConfig,
    *,
    scenario_definitions: list[Mapping[str, Any]],
) -> VecEnv:
    """Build a vectorised environment for SAC training.

    Args:
        config: SAC training configuration.
        scenario_definitions: List of scenario definitions loaded from YAML.

    Returns:
        VecEnv: Vectorized environment wrapper (dummy for one env, subprocess for many envs).
    """
    use_abs = config.action_semantics.strip().lower() == "absolute"
    obs_transform = config.obs_transform.strip().lower()

    def _build_robot_config_for_sampling(scenario: Mapping[str, Any]) -> Any:
        robot_config = build_robot_config_from_scenario(
            scenario,
            scenario_path=config.scenario_config,
        )
        _apply_env_overrides(robot_config, config.env_overrides)
        return robot_config

    def _make(env_index: int) -> Any:
        base_seed = config.seed
        env_seed = None if base_seed is None else int(base_seed) + int(env_index)
        env = ScenarioSwitchingEnv(
            scenario_sampler=ScenarioSampler(
                scenarios=scenario_definitions,
                include_scenarios=config.scenario_sampling.include_scenarios,
                exclude_scenarios=config.scenario_sampling.exclude_scenarios,
                weights=(config.scenario_sampling.weights or None),
                seed=env_seed,
                strategy=config.scenario_sampling.strategy,
            ),
            scenario_path=config.scenario_config,
            env_factory=make_robot_env,
            config_builder=_build_robot_config_for_sampling,
            env_factory_kwargs=config.env_factory_kwargs,
            suite_name="sac_training",
            algorithm_name="sac",
            switch_per_reset=True,
            seed=env_seed,
        )
        env = _maybe_flatten_nested_dict_env(env)
        if obs_transform == "relative":
            env = _RelativeSocNavObservation(env)
        elif obs_transform == "ego":
            env = _EgoSocNavObservation(env)
        if use_abs:
            env = _VelocityTargetActionWrapper(env)
        return env

    env_fns = [(lambda idx=i: _make(idx)) for i in range(config.num_envs)]
    if config.num_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


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


def _relative_socnav_obs(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert SocNav structured observations to a robot-relative frame.

    The SAC checkpoint otherwise overfits to absolute map coordinates because
    training scenarios occupy a different world-coordinate range than benchmark
    maps. Keeping the same keys but translating position-like fields by the
    robot position makes the policy invariant to map origin offsets.
    """
    converted = {str(key): value for key, value in observation.items()}
    if "robot_position" not in converted:
        return converted

    robot_position = np.asarray(converted["robot_position"], dtype=np.float32).reshape(-1)
    if robot_position.size < 2:
        return converted
    robot_xy = robot_position[:2]

    def _shift_xy(key: str) -> None:
        if key not in converted:
            return
        arr = np.asarray(converted[key], dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 2:
            rel = arr.copy()
            rel[:2] = np.clip(rel[:2] - robot_xy, -SOCNAV_POSITION_CAP_M, SOCNAV_POSITION_CAP_M)
            converted[key] = rel
            return
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            rel = arr.copy()
            mask = np.any(np.abs(rel[..., :2]) > 1e-8, axis=-1)
            rel[..., :2][mask] = np.clip(
                rel[..., :2][mask] - robot_xy,
                -SOCNAV_POSITION_CAP_M,
                SOCNAV_POSITION_CAP_M,
            )
            converted[key] = rel

    converted["robot_position"] = np.zeros_like(
        np.asarray(converted["robot_position"], dtype=np.float32)
    )
    _shift_xy("goal_current")
    _shift_xy("goal_next")
    _shift_xy("pedestrians_positions")
    return converted


def _rotate_xy_to_ego(xy_values: np.ndarray, heading: float) -> np.ndarray:
    """Rotate XY vectors from world frame into the robot ego frame."""
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    x_ego = cos_h * xy_values[..., 0] + sin_h * xy_values[..., 1]
    y_ego = -sin_h * xy_values[..., 0] + cos_h * xy_values[..., 1]
    rotated = np.array(xy_values, copy=True, dtype=np.float32)
    rotated[..., 0] = x_ego
    rotated[..., 1] = y_ego
    return rotated


def _ego_socnav_obs(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert flat SocNav observations to robot ego-frame position-like features."""
    converted = _relative_socnav_obs(observation)
    if "robot_heading" not in converted:
        return converted

    heading_arr = np.asarray(converted["robot_heading"], dtype=np.float32).reshape(-1)
    if heading_arr.size == 0:
        return converted
    heading = float(heading_arr[0])

    def _rotate_key(key: str) -> None:
        if key not in converted:
            return
        arr = np.asarray(converted[key], dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 2:
            rotated = _rotate_xy_to_ego(arr[:2].reshape(1, 2), heading).reshape(2)
            rel = arr.copy()
            rel[:2] = np.clip(rotated, -SOCNAV_POSITION_CAP_M, SOCNAV_POSITION_CAP_M)
            converted[key] = rel
            return
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            rel = arr.copy()
            mask = np.any(np.abs(rel[..., :2]) > 1e-8, axis=-1)
            if np.any(mask):
                rotated = _rotate_xy_to_ego(rel[..., :2][mask], heading)
                rel[..., :2][mask] = np.clip(
                    rotated,
                    -SOCNAV_POSITION_CAP_M,
                    SOCNAV_POSITION_CAP_M,
                )
            converted[key] = rel

    _rotate_key("goal_current")
    _rotate_key("goal_next")
    _rotate_key("pedestrians_positions")
    return converted


class _RelativeSocNavObservation(ObservationWrapper):
    """Translate flat SocNav observations into a robot-relative coordinate frame."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.observation_space = self._relative_observation_space(env.observation_space)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute lookups to the wrapped environment."""
        return getattr(self.env, name)

    def observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return _relative_socnav_obs(observation)

    @staticmethod
    def _relative_observation_space(obs_space: gym_spaces.Space) -> gym_spaces.Space:
        if not isinstance(obs_space, gym_spaces.Dict):
            return obs_space

        spaces = dict(obs_space.spaces)
        rel_cap = np.array([SOCNAV_POSITION_CAP_M, SOCNAV_POSITION_CAP_M], dtype=np.float32)
        rel_low = -rel_cap
        rel_high = rel_cap

        for key in ("robot_position", "goal_current", "goal_next"):
            space = spaces.get(key)
            if isinstance(space, gym_spaces.Box) and space.shape == (2,):
                spaces[key] = gym_spaces.Box(low=rel_low, high=rel_high, dtype=np.float32)

        ped_space = spaces.get("pedestrians_positions")
        if (
            isinstance(ped_space, gym_spaces.Box)
            and len(ped_space.shape) == 2
            and ped_space.shape[1] == 2
        ):
            ped_low = np.broadcast_to(rel_low, ped_space.shape).astype(np.float32)
            ped_high = np.broadcast_to(rel_high, ped_space.shape).astype(np.float32)
            spaces["pedestrians_positions"] = gym_spaces.Box(
                low=ped_low,
                high=ped_high,
                dtype=np.float32,
            )

        return gym_spaces.Dict(spaces)


class _EgoSocNavObservation(_RelativeSocNavObservation):
    """Rotate relative SocNav observations into the robot ego frame."""

    def observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return _ego_socnav_obs(observation)


class _VelocityTargetActionWrapper(gymnasium.ActionWrapper):
    """Convert absolute velocity target actions to delta actions for DifferentialDrive envs.

    The benchmark runner (``map_runner._policy_command_to_env_action``) converts
    SAC policy commands to absolute velocity targets before calling ``env.step()``.
    Specifically it computes ``delta = target_v - current_v`` then passes this
    delta to the underlying env whose dynamics are ``new_v = current_v + delta``,
    so the robot ultimately moves at ``target_v``.

    Without this wrapper the SAC model trains with raw-delta semantics
    (``env.step(delta)`` → ``new_v = old_v + delta``).  With the wrapper,
    ``env.step(target)`` → ``delta = target - current_v`` → underlying env
    applies ``new_v = current_v + delta = target``.  This makes training
    semantics identical to benchmark runner semantics.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute lookups to the wrapped environment."""
        return getattr(self.env, name)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Convert absolute velocity target to delta action.

        Args:
            action: Target velocity ``[v_target, omega_target]``.

        Returns:
            np.ndarray: Delta velocity ``[v_target - v_current, omega_target - omega_current]``.
        """
        try:
            robots = self.env.simulator.robots  # type: ignore[attr-defined]
            if robots:
                current_v, current_omega = robots[0].current_speed
                return np.array(
                    [float(action[0]) - float(current_v), float(action[1]) - float(current_omega)],
                    dtype=np.float32,
                )
        except (AttributeError, IndexError):
            pass
        return np.asarray(action, dtype=np.float32)


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


def _default_eval_algo_config(config: SACTrainingConfig) -> Path:
    """Return the benchmark algo config template used for periodic evaluation."""
    obs_mode = str(config.env_overrides.get("observation_mode", "")).strip().lower()
    if config.obs_transform == "ego":
        return Path("configs/baselines/sac_gate_socnav_struct_ego.yaml")
    if obs_mode == "socnav_struct":
        return Path("configs/baselines/sac_gate_socnav_struct.yaml")
    return _DEFAULT_EVAL_ALGO_CONFIG


def _save_sac_checkpoint_with_gym_shim(model: SAC, checkpoint_path: Path) -> None:
    """Save an SB3 SAC checkpoint while preserving gym compatibility shims."""
    gym_module = sys.modules.get("gym")
    restore_gym_module = gym_module is None or not hasattr(gym_module, "__version__")
    if restore_gym_module:
        sys.modules["gym"] = gymnasium
    try:
        model.save(str(checkpoint_path))
    finally:
        if restore_gym_module:
            if gym_module is None:
                sys.modules.pop("gym", None)
            else:
                sys.modules["gym"] = gym_module


class _PeriodicSACEvaluationCallback(BaseCallback):
    """Periodically evaluate SAC checkpoints against a real scenario matrix."""

    def __init__(
        self,
        *,
        training_config: SACTrainingConfig,
        evaluation_config: SACEvaluationConfig,
        wandb_run: object | None = None,
    ) -> None:
        super().__init__()
        self._training_config = training_config
        self._evaluation_config = evaluation_config
        self._wandb_run = wandb_run
        self._last_eval_step = 0
        self._eval_index = 0

    def _on_step(self) -> bool:
        if not self._evaluation_config.enabled:
            return True
        frequency = int(self._evaluation_config.frequency_steps)
        if frequency <= 0:
            return True
        if self.num_timesteps - self._last_eval_step < frequency:
            return True
        self._run_periodic_evaluation()
        self._last_eval_step = self.num_timesteps
        return True

    def _run_periodic_evaluation(self) -> None:
        scenario_matrix = (
            self._evaluation_config.scenario_matrix or self._training_config.scenario_config
        )
        algo_config = self._evaluation_config.algo_config or _default_eval_algo_config(
            self._training_config
        )
        eval_tag = f"{self._evaluation_config.tag_prefix}_{self.num_timesteps:08d}"
        eval_dir = self._evaluation_config.output_dir / eval_tag
        checkpoint_path = (
            eval_dir / f"{self._training_config.policy_id}_{self.num_timesteps:08d}.zip"
        )
        eval_dir.mkdir(parents=True, exist_ok=True)
        try:
            _save_sac_checkpoint_with_gym_shim(self.model, checkpoint_path)
            report = sac_eval.run_sac_evaluation(
                checkpoint=checkpoint_path,
                scenario_matrix=scenario_matrix,
                algo_config=algo_config,
                output_dir=eval_dir,
                tag=eval_tag,
                device=self._evaluation_config.device or self._training_config.device,
                horizon=self._evaluation_config.horizon,
                dt=self._evaluation_config.dt,
                workers=self._evaluation_config.workers,
                min_success_rate=self._evaluation_config.min_success_rate,
            )
        except Exception as exc:  # pragma: no cover - defensive logging for long runs
            logger.warning(
                "Periodic SAC evaluation failed at step {}: {}",
                self.num_timesteps,
                exc,
            )
            return

        logger.info(
            "Periodic SAC evaluation step={} success_rate={:.1%} gate_pass={}",
            self.num_timesteps,
            float(report.get("success_rate", 0.0)),
            bool(report.get("gate_pass", False)),
        )
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(
                    {
                        "sac/eval_success_rate": float(report.get("success_rate", 0.0)),
                        "sac/eval_mean_min_distance": float(report.get("mean_min_distance") or 0.0),
                        "sac/eval_mean_avg_speed": float(report.get("mean_avg_speed", 0.0)),
                        "sac/eval_gate_pass": 1.0 if report.get("gate_pass", False) else 0.0,
                        "sac/eval_step": float(self.num_timesteps),
                    },
                    step=self.num_timesteps,
                )
            except Exception:  # pragma: no cover - wandb is optional
                logger.warning("Failed to log periodic SAC eval metrics to W&B.")


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
        "Starting SAC training: policy_id={}, total_timesteps={}, num_envs={}, dry_run={}",
        config.policy_id,
        config.total_timesteps,
        config.num_envs,
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
            device=config.device,
            policy_kwargs=policy_kwargs,
            **hyperparams,  # type: ignore[arg-type]
        )

        wandb_run = _maybe_init_wandb(config, dry_run=dry_run)
        eval_callback = (
            _PeriodicSACEvaluationCallback(
                training_config=config,
                evaluation_config=config.evaluation,
                wandb_run=wandb_run,
            )
            if config.evaluation.enabled and not dry_run
            else None
        )
        timesteps = _DRY_RUN_TIMESTEPS if dry_run else config.total_timesteps
        try:
            model.learn(
                total_timesteps=timesteps,
                log_interval=100,
                callback=eval_callback,
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.output_dir / f"{config.policy_id}.zip"
        _save_sac_checkpoint_with_gym_shim(model, checkpoint_path)
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
