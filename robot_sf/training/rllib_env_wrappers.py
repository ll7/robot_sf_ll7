"""Environment wrappers for RLlib DreamerV3 training on Robot SF observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, spaces

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_FLATTEN_KEYS: tuple[str, ...] = (OBS_DRIVE_STATE, OBS_RAYS)


class DriveStateRaysFlattenWrapper(ObservationWrapper):
    """Flatten selected Dict observation keys into a single float32 vector.

    The wrapper enforces a deterministic key order so training pipelines get stable
    feature layout independent of the backing ``spaces.Dict`` insertion ordering.
    """

    def __init__(self, env: Any, *, keys: Sequence[str] = DEFAULT_FLATTEN_KEYS) -> None:
        """Initialize flattening wrapper for the given observation keys.

        Args:
            env: Wrapped gymnasium environment.
            keys: Ordered observation keys to concatenate into one vector.

        Raises:
            TypeError: If env observation space is not a Dict or subspaces are not Box.
            KeyError: If one of the requested keys is missing.
            ValueError: If no keys are provided.
        """
        super().__init__(env)
        if not keys:
            raise ValueError("DriveStateRaysFlattenWrapper requires at least one key.")
        self._keys = tuple(str(key) for key in keys)

        obs_space = getattr(env, "observation_space", None)
        if not isinstance(obs_space, spaces.Dict):
            raise TypeError("DriveStateRaysFlattenWrapper requires a Dict observation space.")

        missing_keys = [key for key in self._keys if key not in obs_space.spaces]
        if missing_keys:
            raise KeyError(f"Missing observation keys for flattening: {missing_keys}")

        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for key in self._keys:
            subspace = obs_space.spaces[key]
            if not isinstance(subspace, spaces.Box):
                raise TypeError(f"Observation key '{key}' must map to gymnasium.spaces.Box.")
            lows.append(np.asarray(subspace.low, dtype=np.float32).reshape(-1))
            highs.append(np.asarray(subspace.high, dtype=np.float32).reshape(-1))

        flat_low = np.concatenate(lows).astype(np.float32, copy=False)
        flat_high = np.concatenate(highs).astype(np.float32, copy=False)
        self.observation_space = spaces.Box(
            low=flat_low,
            high=flat_high,
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        """Convert Dict observation into a concatenated float32 vector.

        Returns:
            Flattened float32 observation vector.
        """
        parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in self._keys]
        return np.concatenate(parts).astype(np.float32, copy=False)


class SymmetricActionRescaleWrapper(ActionWrapper):
    """Expose a ``[-1, 1]`` action space while mapping actions to environment bounds."""

    def __init__(self, env: Any) -> None:
        """Initialize action rescaling wrapper.

        Args:
            env: Wrapped gymnasium environment with Box action space.

        Raises:
            TypeError: If action space is not gymnasium.spaces.Box.
            ValueError: If action bounds are degenerate.
        """
        super().__init__(env)
        action_space = getattr(env, "action_space", None)
        if not isinstance(action_space, spaces.Box):
            raise TypeError("SymmetricActionRescaleWrapper requires a Box action space.")

        self._low = np.asarray(action_space.low, dtype=np.float32)
        self._high = np.asarray(action_space.high, dtype=np.float32)
        if np.any(self._high <= self._low):
            raise ValueError("Action bounds must satisfy high > low for all dimensions.")

        self.action_space = spaces.Box(
            low=np.full(action_space.shape, -1.0, dtype=np.float32),
            high=np.full(action_space.shape, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Rescale incoming ``[-1, 1]`` actions to original environment bounds.

        Returns:
            Action vector mapped into the wrapped environment action space.
        """
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scaled = self._low + (0.5 * (clipped + 1.0) * (self._high - self._low))
        return scaled.astype(np.float32, copy=False)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Map environment-space actions back to ``[-1, 1]``.

        Returns:
            Action vector in normalized ``[-1, 1]`` coordinates.
        """
        raw = np.asarray(action, dtype=np.float32)
        normalized = (2.0 * (raw - self._low) / (self._high - self._low)) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def wrap_for_dreamerv3(
    env: Any,
    *,
    flatten_observation: bool,
    flatten_keys: Sequence[str] = DEFAULT_FLATTEN_KEYS,
    normalize_actions: bool,
) -> Any:
    """Apply the standard Robot SF wrappers for DreamerV3 training.

    Returns:
        Environment wrapped according to the selected flattening/action settings.
    """
    wrapped = env
    if flatten_observation:
        wrapped = DriveStateRaysFlattenWrapper(wrapped, keys=flatten_keys)
    if normalize_actions:
        wrapped = SymmetricActionRescaleWrapper(wrapped)
    return wrapped


__all__ = [
    "DEFAULT_FLATTEN_KEYS",
    "DriveStateRaysFlattenWrapper",
    "SymmetricActionRescaleWrapper",
    "wrap_for_dreamerv3",
]
