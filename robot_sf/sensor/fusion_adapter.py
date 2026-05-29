"""Sensor fusion adapter for registry-based sensor management.

This module provides an adapter layer between the sensor registry and the
existing SensorFusion class. It enables configuration-driven sensor selection
without requiring changes to the core fusion logic.

Purpose
-------
- Bridge sensor registry with existing SensorFusion infrastructure
- Enable dynamic sensor instantiation from config
- Maintain backward compatibility with existing code

Design
------
The adapter takes a list of sensor configurations and uses the sensor registry
to instantiate the appropriate sensor implementations. It then provides a
consistent interface for the environment to interact with sensors.

Usage
-----
```python
from robot_sf.sensor.fusion_adapter import create_sensors_from_config
from robot_sf.sensor.registry import register_sensor

# Register sensors
register_sensor("lidar", lambda cfg: LidarSensor(cfg))
register_sensor("camera", lambda cfg: CameraSensor(cfg))

# Create sensors from config
sensor_configs = [
    {"type": "lidar", "range": 10.0},
    {"type": "camera", "resolution": (640, 480)},
]
sensors = create_sensors_from_config(sensor_configs)

# Use sensors
for sensor in sensors:
    sensor.reset()
    sensor.step(state)
    obs = sensor.get_observation()
```

Error Handling
--------------
- Missing "type" key in config raises ValueError with config details
- Unknown sensor type raises KeyError with available sensor list
- Sensor construction errors are propagated with context
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.sensor.registry import get_sensor

if TYPE_CHECKING:
    from robot_sf.sensor.base import Sensor


def create_sensors_from_config(sensor_configs: list[dict[str, Any]]) -> list[Sensor]:
    """Create registry-backed sensor instances from configuration dictionaries.

    The input list is read in order and is not mutated. Each configuration is
    passed unchanged to its registered factory, so sensor-specific keys and
    value types are owned by that sensor implementation.

    Parameters
    ----------
    sensor_configs : list[dict[str, Any]]
        List of sensor configurations. Each dict must contain a "type"
        key specifying the sensor name in the registry, plus any
        sensor-specific configuration parameters.

    Returns
    -------
    list[Sensor]
        List of instantiated sensor objects conforming to Sensor interface.

    Raises
    ------
    ValueError
        If a sensor config is missing the "type" key.
    KeyError
        If a sensor type is not registered. Includes list of available types.
    RuntimeError
        If the registered factory raises during sensor construction; the
        original exception is chained as ``__cause__``.

    Examples
    --------
    >>> configs = [
    ...     {"type": "lidar", "range": 10.0, "num_rays": 64},
    ...     {"type": "camera", "resolution": (640, 480)},
    ... ]
    >>> sensors = create_sensors_from_config(configs)
    >>> len(sensors)
    2
    """
    sensors = []

    for idx, config in enumerate(sensor_configs):
        # Validate config has "type" key
        if "type" not in config:
            msg = f"Sensor config at index {idx} missing 'type' key. Config: {config}"
            raise ValueError(msg)

        sensor_type = config["type"]

        # Get factory from registry (raises KeyError with helpful message if not found)
        try:
            factory = get_sensor(sensor_type)
        except KeyError as e:
            logger.error(
                "Failed to create sensor type '{}' at index {}: {}",
                sensor_type,
                idx,
                e,
            )
            raise

        # Instantiate sensor with config
        try:
            sensor = factory(config)
            sensors.append(sensor)
            logger.debug(
                "Created sensor type '{}' at index {} with config: {}",
                sensor_type,
                idx,
                config,
            )
        except Exception as e:
            msg = (
                f"Failed to instantiate sensor '{sensor_type}' at index {idx} "
                f"with config {config}: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

    return sensors


def validate_sensor_configs(sensor_configs: list[dict[str, Any]]) -> list[str]:
    """Validate sensor configuration dictionaries without instantiating sensors.

    This only checks adapter-owned fields: every config must contain ``"type"``
    and the referenced type must be registered. Sensor-specific schemas are not
    evaluated here because factories own those contracts.

    Parameters
    ----------
    sensor_configs : list[dict[str, Any]]
        List of sensor configurations to validate.

    Returns
    -------
    list[str]
        List of human-readable validation errors. Empty if all adapter-owned
        checks pass. The input list and config dictionaries are not mutated.

    Examples
    --------
    >>> configs = [{"type": "lidar"}, {"missing": "type"}]
    >>> errors = validate_sensor_configs(configs)
    >>> len(errors)
    1
    """
    errors = []

    for idx, config in enumerate(sensor_configs):
        # Check for "type" key
        if "type" not in config:
            errors.append(f"Config at index {idx} missing 'type' key: {config}")
            continue

        sensor_type = config["type"]

        # Check if sensor type is registered
        try:
            get_sensor(sensor_type)
        except KeyError as e:
            errors.append(
                f"Config at index {idx} references unknown sensor type '{sensor_type}': {e}"
            )

    return errors


class MergedObservationFusion:
    """Wrapper that merges base SensorFusion observations with registry sensors.

    This wrapper does not modify SensorFusion internals. It calls the base
    fusion's ``next_obs()`` and then augments the returned dictionary with
    additional sensor observations under keys of the form ``"custom.<name>"``.
    The returned dictionary is mutated in place; callers that need a pristine
    base observation should copy it before wrapping.

    Sensors are stepped with a lightweight state dict containing the simulator
    and robot_id for potential future use by sensor implementations. The state
    shape is ``{"sim": sim, "robot_id": robot_id}``; values may be ``None``.
    """

    def __init__(
        self,
        base_fusion: Any,
        sensors: list[Sensor],
        sensor_names: list[str],
        *,
        sim: Any | None = None,
        robot_id: int | None = None,
    ) -> None:
        """Store the base fusion object and registry-backed sensor instances.

        Parameters
        ----------
        base_fusion : Any
            Existing fusion object that provides ``next_obs`` and optionally
            ``reset_cache``.
        sensors : list[Sensor]
            Additional registry-created sensors to step after the base
            observation is produced.
        sensor_names : list[str]
            Names used to place sensor observations under ``custom.<name>``
            keys. Must align with ``sensors`` order.
        sim : Any | None
            Optional simulator handle passed to custom sensors in their
            lightweight state dict.
        robot_id : int | None
            Optional robot identifier passed to custom sensors in their
            lightweight state dict.

        Raises
        ------
        ValueError
            If ``sensors`` and ``sensor_names`` have different lengths, or if
            custom sensor names collide case-insensitively.
        """
        if len(sensors) != len(sensor_names):
            msg = (
                "sensors and sensor_names must have the same length "
                f"(got {len(sensors)} sensors and {len(sensor_names)} names)"
            )
            raise ValueError(msg)
        seen_names: dict[str, str] = {}
        for name in sensor_names:
            normalized_name = name.casefold()
            if normalized_name in seen_names:
                msg = (
                    "MergedObservationFusion requires unique sensor names "
                    f"(duplicate custom observation name '{name}' conflicts with "
                    f"'{seen_names[normalized_name]}')."
                )
                raise ValueError(msg)
            seen_names[normalized_name] = name

        self._base = base_fusion
        self._sensors = sensors
        self._names = sensor_names
        self._sim = sim
        self._robot_id = robot_id

    def next_obs(self) -> dict[str, Any]:
        """Return base observations augmented with registry sensor outputs.

        Each custom sensor receives a state dict containing ``sim`` and
        ``robot_id`` before its observation is read. Sensor failures are logged
        with the configured sensor name and then re-raised. Custom observations
        are stored without copying, preserving the object returned by
        ``sensor.get_observation()``.

        Returns
        -------
        dict[str, Any]
            Observation dictionary returned by the base fusion plus
            ``custom.<name>`` entries for every configured registry sensor.

        Raises
        ------
        AttributeError
            If the base object does not provide ``next_obs``.
        Exception
            Any exception raised by ``base_fusion.next_obs()``, ``sensor.step()``,
            or ``sensor.get_observation()`` is propagated.
        """
        obs = self._base.next_obs()
        state = {"sim": self._sim, "robot_id": self._robot_id}
        for name, sensor in zip(self._names, self._sensors, strict=True):
            try:
                sensor.step(state)
                obs[f"custom.{name}"] = sensor.get_observation()
            except Exception as e:  # pragma: no cover - defensive logging only
                logger.error("Sensor '{}' failed: {}", name, e)
                raise
        return obs

    # Pass-through API used by RobotState
    def reset_cache(self) -> None:
        """Reset cached base-fusion state and custom sensors when supported.

        The base object's ``reset_cache`` method is optional and called first
        when present. Custom sensor ``reset`` methods are also optional at
        runtime for compatibility with older duck-typed test doubles, even
        though registry sensors are expected to follow the ``Sensor`` protocol.
        Exceptions raised by either reset path are propagated.
        """
        if hasattr(self._base, "reset_cache"):
            self._base.reset_cache()
        for sensor in self._sensors:
            if hasattr(sensor, "reset"):
                sensor.reset()
