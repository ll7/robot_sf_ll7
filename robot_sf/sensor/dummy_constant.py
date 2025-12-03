"""Dummy constant sensor implementation.

A simple Sensor that always returns a constant value. Useful for testing
and demonstrating the sensor registry and config-driven wiring.

Config schema
-------------
- type: "dummy_constant"
- name: str  # observation key base name
- value: float | int | list[float] | list[int]
- shape: list[int] (optional)  # if provided, value is broadcast/reshaped
- dtype: str (optional)  # "float32" (default) or "int32"
- space: dict (optional)
    - low: float | int | list[...] (broadcastable to shape)
    - high: float | int | list[...]
    - shape: list[int]  # required if low/high provided as scalars

Note: The observation space must be declared by the caller (env_util) based on
config.space to integrate into gym spaces. This class focuses on producing
observations only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_sf.sensor.base import Sensor


class DummyConstantSensor(Sensor):
    """A sensor that returns a constant numpy array observation."""

    def __init__(self, config: dict[str, Any]):
        """TODO docstring. Document this function.

        Args:
            config: TODO docstring.
        """
        self._config = config
        self._obs = self._build_value(config)

    def _build_value(self, cfg: dict[str, Any]) -> np.ndarray:
        """TODO docstring. Document this function.

        Args:
            cfg: TODO docstring.

        Returns:
            TODO docstring.
        """
        dtype = np.float32 if cfg.get("dtype", "float32") == "float32" else np.int32
        value = cfg.get("value", 0.0)
        arr = np.array(value, dtype=dtype)
        shape = cfg.get("shape")
        if shape is not None:
            arr = np.broadcast_to(arr, shape).astype(dtype, copy=False)
        return arr

    def reset(self) -> None:  # no state
        """TODO docstring. Document this function."""
        return None

    def step(self, state: Any) -> None:  # no dependence on state
        """TODO docstring. Document this function.

        Args:
            state: TODO docstring.
        """
        return None

    def get_observation(self) -> np.ndarray:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self._obs


def factory(config: dict[str, Any]) -> DummyConstantSensor:
    """Factory function for registry."""
    return DummyConstantSensor(config)
