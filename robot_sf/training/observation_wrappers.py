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
    "maybe_flatten_env_observations",
    "resolve_policy_obs_adapter",
    "resolve_policy_stack_steps",
    "sync_policy_spaces",
]


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
    """Reshape or repeat observations to match the expected Box shape."""
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
    """Return an adapter that extracts drive_state and matches the expected shape."""

    def _adapter(orig_obs: Mapping[str, Any]) -> np.ndarray:
        drive_state = np.asarray(orig_obs[OBS_DRIVE_STATE])
        return _reshape_box_obs(drive_state, expected_shape)

    return _adapter


def _make_ray_obs_adapter(
    expected_shape: tuple[int, ...],
) -> Callable[[Mapping[str, Any]], np.ndarray]:
    """Return an adapter that extracts ray observations and matches the expected shape."""

    def _adapter(orig_obs: Mapping[str, Any]) -> np.ndarray:
        ray_state = np.asarray(orig_obs[OBS_RAYS])
        return _reshape_box_obs(ray_state, expected_shape)

    return _adapter


def resolve_policy_obs_adapter(
    policy_model: Any | None,
    *,
    fallback_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
) -> Callable[[Mapping[str, Any]], np.ndarray] | None:
    """Select an observation adapter based on the PPO policy observation space."""
    if policy_model is None:
        return None
    obs_space = getattr(policy_model, "observation_space", None)
    if obs_space is None:
        if fallback_adapter is not None:
            logger.warning("PPO policy missing observation_space; using fallback adapter.")
        return fallback_adapter
    if isinstance(obs_space, gym_spaces.Dict):
        logger.info("PPO policy expects dict observations; skipping adapter.")
        return None
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


def resolve_policy_stack_steps(policy_model: Any | None) -> int | None:
    """Infer stack_steps from the PPO policy observation space when possible."""
    if policy_model is None:
        return None
    obs_space = getattr(policy_model, "observation_space", None)
    if obs_space is None:
        return None
    if isinstance(obs_space, gym_spaces.Dict):
        spaces = getattr(obs_space, "spaces", {})
        for key in (OBS_DRIVE_STATE, OBS_RAYS):
            subspace = spaces.get(key)
            if subspace is None:
                continue
            shape = getattr(subspace, "shape", None)
            if shape and len(shape) >= 1:
                return int(shape[0])
        for subspace in spaces.values():
            shape = getattr(subspace, "shape", None)
            if shape and len(shape) >= 1:
                return int(shape[0])
        return None
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
