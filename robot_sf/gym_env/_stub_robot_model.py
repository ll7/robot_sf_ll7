"""Shared lightweight stub robot model used as a fallback.

Centralizing the stub avoids duplication across factories and environments.
The stub intentionally performs a local NumPy import inside ``predict`` to keep
module import cost negligible when the stub is unused.

Rationale
---------
Historically both ``environment_factory`` and ``pedestrian_env`` defined an
inline ``_StubRobotModel`` class when a concrete robot policy/model was not
provided. Some tests rely on permissive behavior (auto-injection) to reduce
boilerplate for pedestrian environment smoke tests. This module provides a
single source of truth to ensure consistent behavior and simplify future
extension (e.g., adding action space validation or logging).
"""

from __future__ import annotations

import importlib
from typing import Any


def _get_numpy():
    """Lazy-import numpy to avoid import cost for non-test usage.

    Returns:
        module: The numpy module.
    """
    return importlib.import_module("numpy")


class StubRobotModel:  # pragma: no cover - trivial
    """Fallback model returning a zero action.

    The action dimensionality (2,) matches the historical expectation in tests.
    Returning ``(action, None)`` mirrors common RL model predict signatures
    (e.g., Stable Baselines returning (action, state)).
    """

    def __init__(
        self, action_space: Any | None = None, action_shape: tuple[int, ...] | None = None
    ):
        self._action_shape: tuple[int, ...] | None = None
        if action_space is not None and hasattr(action_space, "shape"):
            try:
                self._action_shape = tuple(action_space.shape)
            except (TypeError, ValueError):
                self._action_shape = None
        elif action_shape is not None:
            self._action_shape = tuple(action_shape)

    def set_action_space(self, action_space: Any) -> None:
        """Set the action space used for zero-action sizing."""
        if hasattr(action_space, "shape"):
            try:
                self._action_shape = tuple(action_space.shape)
            except (TypeError, ValueError):
                self._action_shape = None

    def predict(self, _obs: Any, **_ignored: Any) -> tuple[object, None]:
        """Return a zero action mimicking RL model predict signatures.

        Args:
            _obs: Observation (ignored by stub).
            _ignored: Additional kwargs (ignored by stub).

        Returns:
            tuple[object, None]: Zero action array and None state (matches SB3 signature).
        """
        np = _get_numpy()
        shape = self._action_shape or (2,)
        return np.zeros(shape, dtype=float), None
