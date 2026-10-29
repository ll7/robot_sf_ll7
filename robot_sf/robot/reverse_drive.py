"""Opt-in limited reverse contract; defaults retain historical identity bytes."""

from math import isfinite
from numbers import Real
from typing import Any

from loguru import logger


def warn_unsupported_reverse(adapter: Any, drive: Any) -> None:
    """Warn once per adapter when the live plant enables unsupported reverse.

    ORCA subclasses and hybrid v4 implement guarded reverse commands. Other
    adapters retain their existing samples; the signed plant can still accept
    negative commands from external or learned policies. Unbound policy
    callables are not adapters and must not be described as such.
    """
    if callable(adapter):
        adapter = getattr(adapter, "_planner_adapter", None)
    if adapter is None:
        return
    if not getattr(drive, "limited_reverse", False):
        return
    families = {cls.__name__ for cls in type(adapter).__mro__}
    aware = "ORCAPlannerAdapter" in families or (
        "HybridRuleLocalPlannerAdapter" in families
        and getattr(adapter, "_v4_clearance_braking", False)
    )
    if aware or getattr(adapter, "_limited_reverse_warning_emitted", False):
        return
    logger.warning(
        "limited_reverse enabled for {}; this adapter is not reverse-aware. "
        "Its existing sampling is unchanged, while the plant accepts negative commands.",
        type(adapter).__name__,
    )
    adapter._limited_reverse_warning_emitted = True


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


def bound_drive_settings(env: Any, *, adapter: Any = None) -> Any:
    """Read the live plant settings without consulting future simulator state.

    Returns:
        The live drive settings, or None when no robot is bound.
    """
    config = getattr(env, "env_config", None) or getattr(env, "config", None)
    drive = getattr(config, "robot_config", None)
    if drive is None:
        robots = getattr(getattr(env, "simulator", None), "robots", None)
        drive = getattr(robots[0], "config", None) if robots else None
    if adapter is not None:
        warn_unsupported_reverse(adapter, drive)
    return drive
