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
    """Create sensor instances from configuration list.

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
    """Validate sensor configurations without instantiation.

    Parameters
    ----------
    sensor_configs : list[dict[str, Any]]
        List of sensor configurations to validate.

    Returns
    -------
    list[str]
        List of validation errors. Empty if all configs are valid.

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
    fusion's next_obs() and then augments the dict with additional sensor
    observations under keys of the form "custom.<name>".

    Sensors are stepped with a lightweight state dict containing the simulator
    and robot_id for potential future use by sensor implementations.
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
        """Init.

        Args:
            base_fusion: Auto-generated placeholder description.
            sensors: Auto-generated placeholder description.
            sensor_names: Auto-generated placeholder description.
            sim: Auto-generated placeholder description.
            robot_id: Auto-generated placeholder description.

        Returns:
            None: Auto-generated placeholder description.
        """
        self._base = base_fusion
        self._sensors = sensors
        self._names = sensor_names
        self._sim = sim
        self._robot_id = robot_id

    def next_obs(self) -> dict[str, Any]:
        """Next obs.

        Returns:
            dict[str, Any]: Auto-generated placeholder description.
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
        """Reset cache.

        Returns:
            None: Auto-generated placeholder description.
        """
        if hasattr(self._base, "reset_cache"):
            self._base.reset_cache()
        for sensor in self._sensors:
            if hasattr(sensor, "reset"):
                sensor.reset()
