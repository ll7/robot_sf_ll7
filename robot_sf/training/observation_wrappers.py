"""Observation helpers shared by training and playback scripts.

Provides utilities to align observation shapes with Stable-Baselines3 policies.

Quick reference:
- maybe_flatten_env_observations: wrap dict observations with FlattenObservation.
- resolve_policy_obs_adapter: select an adapter for PPO checkpoints.
- resolve_policy_stack_steps: infer stack_steps from a PPO observation space.
- sync_policy_spaces: align env observation/action spaces with a loaded policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from gymnasium import spaces as gym_spaces
from gymnasium.wrappers import FlattenObservation
from loguru import logger

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

__all__ = [
    "LegacyRun023ObsAdapter",
    "adapt_dict_observation_to_policy_space",
    "maybe_flatten_env_observations",
    "resolve_policy_obs_adapter",
    "resolve_policy_stack_steps",
    "sync_policy_spaces",
]

_DICT_OBS_COMPAT_ALIASES: dict[str, tuple[str, ...]] = {
    "robot_speed": ("robot_velocity_xy",),
    "robot_velocity_xy": ("robot_speed",),
}


class LegacyRun023ObsAdapter:
    """Wrap a PPO model so ``run_023`` receives its legacy flattened observation format."""

    def __init__(self, model: Any):
        """Store the wrapped model and expose action-space compatibility hooks."""
        self._model = model
        self.action_space = getattr(model, "action_space", None)

    def set_action_space(self, action_space: Any) -> None:
        """Allow env-side action-space synchronization."""
        self.action_space = action_space
        if hasattr(self._model, "set_action_space"):
            self._model.set_action_space(action_space)

    def predict(self, obs: Any, deterministic: bool = True) -> Any:
        """Adapt dict observations to the ``run_023`` flattened format before inference.

        Returns:
            Any: The wrapped model prediction tuple using the adapted observation payload.
        """
        adapted_obs = obs
        if isinstance(obs, Mapping):
            drive_state = np.asarray(obs[OBS_DRIVE_STATE])[:, :-1].copy()
            ray_state = np.asarray(obs[OBS_RAYS])
            drive_state[:, 2] *= 10
            drive_state = np.squeeze(drive_state).reshape(-1)
            ray_state = np.squeeze(ray_state).reshape(-1)
            adapted_obs = np.concatenate((ray_state, drive_state), axis=0)
        return self._model.predict(adapted_obs, deterministic=deterministic)


def maybe_flatten_env_observations(env: Any, *, context: str = "training") -> Any:
    """Wrap ``env`` in ``FlattenObservation`` when it exposes a dict space.

    Stable-Baselines3 persists the observation space in its checkpoints. If BC or
    PPO fine-tuning runs with a flattened space, every subsequent consumer must
    recreate the same shape before calling ``PPO.load``. This helper provides a
    single place to apply the wrapper and emit consistent logging.

    Returns:
        Any: The original environment or a ``FlattenObservation``-wrapped instance.
    """

    if isinstance(getattr(env, "observation_space", None), gym_spaces.Dict):
        logger.info(
            "Applying FlattenObservation wrapper for dict observation space during {}.",
            context,
        )
        return FlattenObservation(env)

    return env


def _reshape_box_obs(obs: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Reshape or repeat observations to match the expected Box shape.

    Returns:
        Reshaped or repeated observation array matching expected_shape.
    """
    if obs.shape == expected_shape:
        return obs
    if len(expected_shape) == 1:
        if obs.ndim == 2 and obs.shape[0] == 1 and obs.shape[1:] == expected_shape:
            return np.squeeze(obs, axis=0)
        return obs
    if len(expected_shape) == 2:
        stack, features = expected_shape
        if obs.ndim == 1 and obs.shape == (features,):
            return np.repeat(obs[np.newaxis, :], stack, axis=0)
        if obs.ndim == 2 and obs.shape[1:] == (features,):
            if obs.shape[0] == 1:
                return np.repeat(obs, stack, axis=0)
    return obs


def _make_drive_state_adapter(
    expected_shape: tuple[int, ...],
) -> Callable[[Mapping[str, Any]], np.ndarray]:
    """Return an adapter that extracts drive_state and matches the expected shape.

    Returns:
        Callable adapter that extracts drive_state observations and reshapes them.
    """

    def _adapter(orig_obs: Mapping[str, Any]) -> np.ndarray:
        drive_state = np.asarray(orig_obs[OBS_DRIVE_STATE])
        return _reshape_box_obs(drive_state, expected_shape)

    return _adapter


def _make_ray_obs_adapter(
    expected_shape: tuple[int, ...],
) -> Callable[[Mapping[str, Any]], np.ndarray]:
    """Return an adapter that extracts ray observations and matches the expected shape.

    Returns:
        Callable adapter that extracts ray observations and reshapes them.
    """

    def _adapter(orig_obs: Mapping[str, Any]) -> np.ndarray:
        ray_state = np.asarray(orig_obs[OBS_RAYS])
        return _reshape_box_obs(ray_state, expected_shape)

    return _adapter


def _reshape_to_target_shape(
    arr: np.ndarray,
    *,
    key: str,
    target_shape: tuple[int, ...] | None,
) -> np.ndarray:
    """Return ``arr`` reshaped to the policy-declared target shape when compatible."""
    if target_shape is None or tuple(arr.shape) == tuple(target_shape):
        return arr
    target_size = int(np.prod(target_shape))
    if int(arr.size) != target_size:
        raise ValueError(
            f"Dict observation key '{key}' shape mismatch: got {tuple(arr.shape)}, "
            f"expected {tuple(target_shape)}."
        )
    return np.asarray(arr).reshape(target_shape)


def adapt_dict_observation_to_policy_space(
    obs: Mapping[str, Any],
    policy_model: Any | None,
) -> Mapping[str, Any]:
    """Filter and reshape dict observations to match a loaded PPO policy space.

    Stable-Baselines3 multi-input policies reject unexpected dict keys. This helper
    keeps only the checkpoint-declared keys, reshapes values to the declared subspace
    shapes, and backfills a small alias set for renamed robot kinematics fields.

    Returns:
        Mapping[str, Any]: Observation payload aligned to the model-declared Dict space.
    """
    if policy_model is None:
        return obs
    obs_space = getattr(policy_model, "observation_space", None)
    if not isinstance(obs_space, gym_spaces.Dict):
        return obs

    aligned: dict[str, np.ndarray] = {}
    missing: list[str] = []
    source_obs = dict(obs)
    for key, subspace in obs_space.spaces.items():
        value = source_obs.get(key)
        if value is None:
            for alias in _DICT_OBS_COMPAT_ALIASES.get(key, ()):
                if alias in source_obs:
                    value = source_obs[alias]
                    break
        if value is None:
            missing.append(str(key))
            continue
        arr = np.asarray(value, dtype=getattr(subspace, "dtype", None))
        aligned[str(key)] = _reshape_to_target_shape(
            arr,
            key=str(key),
            target_shape=getattr(subspace, "shape", None),
        )
    if missing:
        missing_preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Missing required dict observation keys: {missing_preview}")
    return aligned


def resolve_policy_obs_adapter(
    policy_model: Any | None,
    *,
    fallback_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
) -> Callable[[Mapping[str, Any]], np.ndarray] | None:
    """Select an observation adapter based on the PPO policy observation space.

    Returns:
        Observation adapter callable for the policy, or None if no adapter needed.
    """
    if policy_model is None:
        return None
    obs_space = getattr(policy_model, "observation_space", None)
    if obs_space is None:
        if fallback_adapter is not None:
            logger.warning("PPO policy missing observation_space; using fallback adapter.")
        return fallback_adapter
    if isinstance(obs_space, gym_spaces.Dict):
        logger.info(
            "PPO policy expects dict observations; aligning runtime dict keys to model space."
        )
        return lambda orig_obs: adapt_dict_observation_to_policy_space(orig_obs, policy_model)
    if isinstance(obs_space, gym_spaces.Box):
        shape = tuple(obs_space.shape)
        if shape and shape[-1] == 5:
            logger.info("PPO policy expects drive_state observations; using drive-state adapter.")
            return _make_drive_state_adapter(shape)
        if shape and shape[-1] == 272:
            logger.info("PPO policy expects ray observations; using ray adapter.")
            return _make_ray_obs_adapter(shape)
    if fallback_adapter is not None:
        logger.info("PPO policy expects flat observations; using fallback adapter.")
    return fallback_adapter


def _extract_stack_from_shape(shape: tuple[int, ...] | None) -> int | None:
    """Extract stack dimension from observation shape.

    Returns:
        Stack dimension if shape is valid, None otherwise.
    """
    if shape and len(shape) >= 1:
        return int(shape[0])
    return None


def _extract_stack_from_dict_space(obs_space: gym_spaces.Dict) -> int | None:
    """Extract stack dimension from a Dict observation space.

    Returns:
        Stack dimension if found, None otherwise.
    """
    spaces = getattr(obs_space, "spaces", {})
    # Check priority keys first
    for key in (OBS_DRIVE_STATE, OBS_RAYS):
        subspace = spaces.get(key)
        if subspace is not None:
            shape = getattr(subspace, "shape", None)
            result = _extract_stack_from_shape(shape)
            if result is not None:
                return result
    # Fallback to any subspace
    for subspace in spaces.values():
        shape = getattr(subspace, "shape", None)
        result = _extract_stack_from_shape(shape)
        if result is not None:
            return result
    return None


def resolve_policy_stack_steps(policy_model: Any | None) -> int | None:
    """Infer stack_steps from the PPO policy observation space when possible.

    Returns:
        Inferred stack_steps from policy observation space, or None if unavailable.
    """
    if policy_model is None:
        return None
    obs_space = getattr(policy_model, "observation_space", None)
    if obs_space is None:
        return None

    # Handle Dict spaces
    if isinstance(obs_space, gym_spaces.Dict):
        return _extract_stack_from_dict_space(obs_space)

    # Handle Box spaces
    if isinstance(obs_space, gym_spaces.Box):
        shape = getattr(obs_space, "shape", None)
        if shape and len(shape) > 1:
            return int(shape[0])
    return None


def sync_policy_spaces(env: Any, policy_model: Any | None) -> None:
    """Align env observation/action spaces with a loaded policy when available."""
    if policy_model is None:
        return
    obs_space = getattr(policy_model, "observation_space", None)
    if obs_space is not None:
        env.observation_space = obs_space
    action_space = getattr(policy_model, "action_space", None)
    if action_space is not None:
        env.action_space = action_space
