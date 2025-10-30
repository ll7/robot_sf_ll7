"""Sensor registry for dynamic sensor instantiation.

This module provides a centralized registry for sensor implementations,
allowing sensors to be registered and retrieved by name. This enables
configuration-driven sensor selection without hardcoding sensor types.

Purpose
-------
- Centralized sensor registration and lookup
- Dynamic sensor instantiation from configuration
- Extensible architecture for adding new sensors

Usage
-----
Register a sensor factory:
```python
from robot_sf.sensor.registry import register_sensor


def my_sensor_factory(config):
    return MySensor(config)


register_sensor("my_sensor", my_sensor_factory)
```

Retrieve and instantiate a sensor:
```python
from robot_sf.sensor.registry import get_sensor

factory = get_sensor("my_sensor")
sensor = factory(config)
```

List all registered sensors:
```python
from robot_sf.sensor.registry import list_sensors

sensors = list_sensors()
print(f"Available sensors: {list(sensors.keys())}")
```

Error Handling
--------------
- Unknown sensor name raises KeyError with suggestions of known sensors
- Duplicate registration raises ValueError unless override=True
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

# Type alias for sensor factory functions
SensorFactory = Callable[[Any], Any]

# Global sensor registry
_SENSOR_REGISTRY: dict[str, SensorFactory] = {}


def register_sensor(name: str, factory: SensorFactory, *, override: bool = False) -> None:
    """Register a sensor factory in the global registry.

    Parameters
    ----------
    name : str
        Unique name for the sensor (case-sensitive).
    factory : SensorFactory
        Factory function that takes a config and returns a Sensor instance.
    override : bool, optional
        If True, allow overriding existing registration. Default is False.

    Raises
    ------
    ValueError
        If sensor name already registered and override is False.

    Examples
    --------
    >>> def lidar_factory(config):
    ...     return LidarSensor(config)
    >>> register_sensor("lidar", lidar_factory)
    """
    if name in _SENSOR_REGISTRY and not override:
        msg = f"Sensor '{name}' is already registered. Use override=True to replace."
        raise ValueError(msg)

    _SENSOR_REGISTRY[name] = factory
    logger.debug("Registered sensor '{}' in registry", name)


def get_sensor(name: str) -> SensorFactory:
    """Retrieve a sensor factory from the registry.

    Parameters
    ----------
    name : str
        Name of the sensor to retrieve.

    Returns
    -------
    SensorFactory
        The factory function for the requested sensor.

    Raises
    ------
    KeyError
        If sensor name is not registered. Error message includes
        list of available sensors.

    Examples
    --------
    >>> factory = get_sensor("lidar")
    >>> sensor = factory(config)
    """
    if name not in _SENSOR_REGISTRY:
        known = ", ".join(sorted(_SENSOR_REGISTRY.keys()))
        msg = f"Unknown sensor '{name}'. Available sensors: {known}"
        raise KeyError(msg)

    return _SENSOR_REGISTRY[name]


def list_sensors() -> dict[str, SensorFactory]:
    """List all registered sensors.

    Returns
    -------
    dict[str, SensorFactory]
        Dictionary mapping sensor names to their factory functions.

    Examples
    --------
    >>> sensors = list_sensors()
    >>> print(f"Available: {list(sensors.keys())}")
    """
    return _SENSOR_REGISTRY.copy()


def unregister_sensor(name: str) -> None:
    """Remove a sensor from the registry.

    Parameters
    ----------
    name : str
        Name of the sensor to remove.

    Raises
    ------
    KeyError
        If sensor name is not registered.

    Note
    ----
    This is primarily useful for testing. In production, sensors should
    remain registered for the lifetime of the process.
    """
    if name not in _SENSOR_REGISTRY:
        msg = f"Cannot unregister unknown sensor '{name}'"
        raise KeyError(msg)

    del _SENSOR_REGISTRY[name]
    logger.debug("Unregistered sensor '{}'", name)


# Register built-in sensors
# Note: Actual sensor implementations to be registered in subsequent tasks
