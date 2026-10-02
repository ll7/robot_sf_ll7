"""Opt-in limited reverse contract; defaults retain historical identity bytes."""

from math import isfinite
from numbers import Real
from typing import Any


def validate_reverse_settings(limited_reverse: bool, max_reverse_speed: float) -> None:
    """Reject invalid reverse settings before constructing a plant or adapter."""
    if not isinstance(limited_reverse, bool):
        raise ValueError("limited_reverse must be a boolean")
    if (
        isinstance(max_reverse_speed, bool)
        or not isinstance(max_reverse_speed, Real)
        or not isfinite(max_reverse_speed)
        or max_reverse_speed <= 0.0
    ):
        raise ValueError("max_reverse_speed must be finite and positive")


def reverse_identity(settings: Any) -> dict[str, Any]:
    """Return versioned opt-in identity; omit the entire block for legacy settings."""
    if not getattr(settings, "limited_reverse", False):
        return {}
    return {
        "reverse_drive": {
            "schema_version": "limited_reverse.v1",
            "max_reverse_speed_m_s": float(settings.max_reverse_speed),
        }
    }


def bound_drive_settings(env: Any) -> Any:
    """Read the live plant settings without consulting future simulator state.

    Returns:
        The live drive settings, or None when no robot is bound.
    """
    config = getattr(env, "env_config", None) or getattr(env, "config", None)
    drive = getattr(config, "robot_config", None)
    if drive is not None:
        return drive
    robots = getattr(getattr(env, "simulator", None), "robots", None)
    return getattr(robots[0], "config", None) if robots else None
