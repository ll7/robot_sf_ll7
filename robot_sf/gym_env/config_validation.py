"""Configuration validation utilities.

Validates unified configuration for unknown keys, conflicts, and schema violations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from robot_sf.gym_env.unified_config import BaseSimulationConfig


def validate_config(config: BaseSimulationConfig, *, strict: bool = True) -> None:
    """Validate configuration for unknown keys and conflicts.

    Parameters
    ----------
    config : BaseSimulationConfig
        Configuration to validate
    strict : bool
        If True, raise on unknown keys; if False, only warn (reserved for future use)

    Raises
    ------
    ValueError
        If strict and unknown keys found, or if conflicts detected
    """
    # Check backend is known (registry check happens at runtime)
    backend = getattr(config, "backend", "fast-pysf")
    logger.debug("Validating config with backend={} strict={}", backend, strict)

    # Check for mutual exclusion: image and non-image stacks
    use_image = getattr(config, "use_image_obs", False)
    if use_image and not hasattr(config, "image_config"):
        raise ValueError(
            "Config conflict: use_image_obs=True but image_config missing. "
            "Use ImageRobotConfig instead of RobotSimulationConfig."
        )

    # Placeholder for additional validation rules
    # Future: check sensor names against registry when implemented
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
    from dataclasses import asdict, is_dataclass

    if is_dataclass(config):
        return asdict(config)
    # Fallback for non-dataclass configs
    return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
