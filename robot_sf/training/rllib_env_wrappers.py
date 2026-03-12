"""Environment wrappers for RLlib DreamerV3 training on Robot SF observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, Wrapper, spaces

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

DEFAULT_FLATTEN_KEYS: tuple[str, ...] = (OBS_DRIVE_STATE, OBS_RAYS)


def _iter_leaf_spaces(space: spaces.Space[Any]) -> list[spaces.Box]:
    """Return leaf Box spaces in deterministic order.

    Raises:
        TypeError: If the observation tree contains non-Box leaves.
    """
    if isinstance(space, spaces.Box):
        return [space]
    if isinstance(space, spaces.Dict):
        leaves: list[spaces.Box] = []
        for subspace in space.spaces.values():
            leaves.extend(_iter_leaf_spaces(subspace))
        return leaves
    raise TypeError(
        "FlattenDictObservationWrapper only supports Dict observations with Box leaves. "
        f"Received {type(space).__name__}."
    )


def _flatten_observation_leaves(space: spaces.Space[Any], observation: Any) -> list[np.ndarray]:
    """Flatten one observation payload using the declared space traversal order."""
    if isinstance(space, spaces.Box):
        return [np.asarray(observation, dtype=np.float32).reshape(-1)]
    if isinstance(space, spaces.Dict):
        if not isinstance(observation, Mapping):
            raise TypeError("FlattenDictObservationWrapper requires mapping observations.")
        parts: list[np.ndarray] = []
        for key, subspace in space.spaces.items():
            if key not in observation:
                raise KeyError(f"Missing observation key for flattening: {key}")
            parts.extend(_flatten_observation_leaves(subspace, observation[key]))
        return parts
    raise TypeError(
        "FlattenDictObservationWrapper only supports Dict observations with Box leaves. "
        f"Received {type(space).__name__}."
    )


def _normalized_box_bound(
    bound: np.ndarray, *, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    """Broadcast Box bounds to the declared dtype to avoid float64 precision warnings."""
    array = np.asarray(bound, dtype=dtype)
    if array.shape == shape:
        return array
    return np.broadcast_to(array, shape).astype(dtype, copy=False)


def _normalize_space_dtypes(space: spaces.Space[Any]) -> spaces.Space[Any]:
    """Clone Box/Dict spaces so their bounds use the declared leaf dtypes exactly."""
    if isinstance(space, spaces.Box):
        dtype = np.dtype(space.dtype)
        low = _normalized_box_bound(space.low, shape=space.shape, dtype=dtype)
        high = _normalized_box_bound(space.high, shape=space.shape, dtype=dtype)
        return spaces.Box(low=low, high=high, dtype=dtype)
    if isinstance(space, spaces.Dict):
        return spaces.Dict(
            {key: _normalize_space_dtypes(subspace) for key, subspace in space.spaces.items()}
        )
    return space


def _coerce_observation_to_space(space: spaces.Space[Any], observation: Any) -> Any:
    """Cast observation payloads to the dtypes declared by the observation space."""
    if isinstance(space, spaces.Box):
        return np.asarray(observation, dtype=space.dtype)
    if isinstance(space, spaces.Dict):
        if not isinstance(observation, Mapping):
            raise TypeError("ObservationSpaceDtypeWrapper requires mapping observations.")
        return {
            key: _coerce_observation_to_space(subspace, observation[key])
            for key, subspace in space.spaces.items()
        }
    return observation


class FlattenDictObservationWrapper(ObservationWrapper):
    """Flatten Dict observations into a single float32 vector.

    The wrapper enforces a deterministic traversal order so training pipelines get
    stable feature layout independent of runtime dict ordering. Nested ``spaces.Dict``
    subtrees are traversed recursively in declaration order.
    """

    def __init__(self, env: Any, *, keys: Sequence[str] | None = None) -> None:
        """Initialize flattening wrapper for the given observation keys.

        Args:
            env: Wrapped gymnasium environment.
            keys: Optional ordered top-level observation keys to concatenate into one
                vector. When ``None``, all top-level keys are flattened.

        Raises:
            TypeError: If env observation space is not a Dict or leaves are not Box.
            KeyError: If one of the requested keys is missing.
            ValueError: If the observation space has no keys to flatten.
        """
        super().__init__(env)

        obs_space = getattr(env, "observation_space", None)
        if not isinstance(obs_space, spaces.Dict):
            raise TypeError("FlattenDictObservationWrapper requires a Dict observation space.")

        resolved_keys = (
            tuple(obs_space.spaces.keys()) if keys is None else tuple(str(key) for key in keys)
        )
        if not resolved_keys:
            raise ValueError("FlattenDictObservationWrapper requires at least one key.")
        self._keys = resolved_keys

        missing_keys = [key for key in self._keys if key not in obs_space.spaces]
        if missing_keys:
            raise KeyError(f"Missing observation keys for flattening: {missing_keys}")

        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for key in self._keys:
            for leaf in _iter_leaf_spaces(obs_space.spaces[key]):
                lows.append(np.asarray(leaf.low, dtype=np.float32).reshape(-1))
                highs.append(np.asarray(leaf.high, dtype=np.float32).reshape(-1))

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
        parts: list[np.ndarray] = []
        for key in self._keys:
            parts.extend(
                _flatten_observation_leaves(
                    self.env.observation_space.spaces[key], observation[key]
                )
            )
        return np.concatenate(parts).astype(np.float32, copy=False)


class DriveStateRaysFlattenWrapper(FlattenDictObservationWrapper):
    """Backwards-compatible alias for the legacy drive-state/rays flattening wrapper."""

    def __init__(self, env: Any, *, keys: Sequence[str] = DEFAULT_FLATTEN_KEYS) -> None:
        """Initialize the legacy drive-state/rays flattening wrapper."""
        super().__init__(env, keys=keys)


class ObservationSpaceDtypeWrapper(ObservationWrapper):
    """Force observations to match the declared observation-space dtypes exactly."""

    def __init__(self, env: Wrapper) -> None:
        super().__init__(env)
        self.observation_space = _normalize_space_dtypes(env.observation_space)

    def observation(self, observation: Any) -> Any:
        return _coerce_observation_to_space(self.observation_space, observation)


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
    flatten_keys: Sequence[str] | None = DEFAULT_FLATTEN_KEYS,
    normalize_actions: bool,
) -> Any:
    """Apply the standard Robot SF wrappers for DreamerV3 training.

    Returns:
        Environment wrapped according to the selected flattening/action settings.
    """
    wrapped = env
    if flatten_observation:
        wrapped = FlattenDictObservationWrapper(wrapped, keys=flatten_keys)
    wrapped = ObservationSpaceDtypeWrapper(wrapped)
    if normalize_actions:
        wrapped = SymmetricActionRescaleWrapper(wrapped)
    return wrapped


__all__ = [
    "DEFAULT_FLATTEN_KEYS",
    "DriveStateRaysFlattenWrapper",
    "FlattenDictObservationWrapper",
    "ObservationSpaceDtypeWrapper",
    "SymmetricActionRescaleWrapper",
    "wrap_for_dreamerv3",
]
