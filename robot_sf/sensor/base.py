"""Base sensor interface for the sensor registry.

This module defines the abstract Sensor interface that all sensor implementations
should conform to. Sensors are registered in the sensor registry and can be
dynamically instantiated from configuration.

Purpose
-------
Provide a unified interface for sensor implementations to enable:
- Dynamic sensor instantiation from configuration
- Pluggable sensor architecture without hardcoded types
- Consistent sensor lifecycle (reset, step, get_observation)

Interface
---------
All sensor implementations should provide:
- reset() -> None: Initialize or reset sensor state
- step(state) -> None: Update sensor with new simulation state
- get_observation() -> Any: Return current sensor observation

Example
-------
```python
from robot_sf.sensor.base import Sensor
from robot_sf.sensor.registry import register_sensor


class MySensor(Sensor):
    def __init__(self, config):
        self.config = config
        self.observation = None

    def reset(self) -> None:
        self.observation = None

    def step(self, state) -> None:
        # Update observation based on state
        self.observation = self._compute(state)

    def get_observation(self):
        return self.observation


# Register the sensor
register_sensor("my_sensor", lambda cfg: MySensor(cfg))
```
"""

from __future__ import annotations

from typing import Any, Protocol


class Sensor(Protocol):
    """Abstract sensor interface for all sensor implementations.

    All sensors should implement these methods to be compatible with
    the sensor registry and fusion system.
    """

    def reset(self) -> None:
        """Initialize or reset the sensor state.

        Called when the environment is reset. Should initialize any
        internal state and prepare for new episode.
        """

    def step(self, state: Any) -> None:
        """Update sensor with new simulation state.

        Parameters
        ----------
        state : Any
            Current simulation state containing information needed
            to update the sensor observation.
        """

    def get_observation(self) -> Any:
        """Return current sensor observation.

        Returns
        -------
        Any
            The current sensor observation in the format expected
            by the environment observation space.
        """
