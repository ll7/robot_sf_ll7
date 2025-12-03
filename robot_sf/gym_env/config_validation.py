"""Configuration validation utilities.

Validates unified configuration for unknown keys, conflicts, and schema violations.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from robot_sf.gym_env.unified_config import BaseSimulationConfig

from robot_sf.sensor.registry import list_sensors
from robot_sf.sim.registry import list_backends


def _get_valid_field_names(config: BaseSimulationConfig) -> set[str]:
    """Extract valid field names from a dataclass config.

    Parameters
    ----------
    config : BaseSimulationConfig
        Configuration instance

    Returns
    -------
    set[str]
        Set of valid field names for this config type
    """
    if not is_dataclass(config):
        # Fallback: all non-private attributes
        return {k for k in dir(config) if not k.startswith("_")}

    # Get all fields from the dataclass and its bases
    valid_names = {f.name for f in fields(config)}
    return valid_names


def _check_unknown_keys(config: BaseSimulationConfig, *, strict: bool) -> None:
    """Check for unknown keys in config dictionary.

    Parameters
    ----------
    config : BaseSimulationConfig
        Configuration to check
    strict : bool
        If True, raise ValueError on unknown keys; if False, log warning

    Raises
    ------
    ValueError
        If strict and unknown keys found
    """
    valid_keys = _get_valid_field_names(config)
    actual_keys = {k for k in config.__dict__ if not k.startswith("_")}
    unknown = actual_keys - valid_keys

    if unknown:
        msg = (
            f"Unknown config keys: {sorted(unknown)}. "
            f"Valid keys for {type(config).__name__}: {sorted(valid_keys)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)


def _check_backend_valid(config: BaseSimulationConfig) -> None:
    """Check backend name against registry.

    Parameters
    ----------
    config : BaseSimulationConfig
        Configuration to check

    Raises
    ------
    KeyError
        If backend name not registered
    """
    backend = getattr(config, "backend", "fast-pysf")
    available = list_backends()

    if backend not in available:
        known = ", ".join(available)
        raise KeyError(f"Unknown backend '{backend}'. Available backends: {known}")


def _check_sensor_names_valid(config: BaseSimulationConfig) -> None:
    """Check sensor names against registry.

    Parameters
    ----------
    config : BaseSimulationConfig
        Configuration to check

    Raises
    ------
    KeyError
        If sensor type not registered
    """
    sensors = getattr(config, "sensors", [])
    if not sensors:
        return

    available = list_sensors()

    for idx, sensor_cfg in enumerate(sensors):
        if not isinstance(sensor_cfg, dict):
            raise ValueError(f"Sensor config at index {idx} must be dict, got {type(sensor_cfg)}")

        sensor_type = sensor_cfg.get("type")
        if not sensor_type:
            raise ValueError(f"Sensor config at index {idx} missing required 'type' field")

        if sensor_type not in available:
            known = ", ".join(sorted(available.keys()))
            raise KeyError(
                f"Unknown sensor type '{sensor_type}' at index {idx}. Available sensors: {known}"
            )


def validate_config(config: BaseSimulationConfig, *, strict: bool = True) -> None:
    """Validate configuration for unknown keys and conflicts.

    Parameters
    ----------
    config : BaseSimulationConfig
        Configuration to validate
    strict : bool
        If True, raise on unknown keys; if False, only warn

    Raises
    ------
    ValueError
        If strict and unknown keys found, or if conflicts detected
    KeyError
        If backend or sensor name not found in registry
    """
    logger.debug("Validating config type={} strict={}", type(config).__name__, strict)

    # Check for unknown keys (T028)
    _check_unknown_keys(config, strict=strict)

    # Validate backend name against registry
    _check_backend_valid(config)

    # Validate sensor names against registry
    _check_sensor_names_valid(config)

    # Check for mutual exclusion conflicts (T029)
    use_image = getattr(config, "use_image_obs", False)
    if use_image and not hasattr(config, "image_config"):
        raise ValueError(
            "Config conflict: use_image_obs=True but image_config missing. "
            "Use ImageRobotConfig instead of RobotSimulationConfig."
        )

    # Check sensors list type
    sensors = getattr(config, "sensors", [])
    if sensors and not isinstance(sensors, list):
        raise ValueError(f"Config error: sensors must be a list, got {type(sensors)}")


def get_resolved_config_dict(config: BaseSimulationConfig) -> dict:
    """Serialize configuration to a resolved dictionary for logging.

    Parameters
    ----------
    config : BaseSimulationConfig
        Configuration instance

    Returns
    -------
    dict
        Dictionary representation with all defaults resolved
    """
    if is_dataclass(config):
        return asdict(config)
    # Fallback for non-dataclass configs
    return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
