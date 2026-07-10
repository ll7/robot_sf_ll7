"""Stopping-distance envelope helpers for actuation-realism provenance (issue #4976).

A near-miss or time-to-collision (TTC) reading is only interpretable against the
*braking capability* the robot actually has. A policy that can decelerate at
``5 m/s^2`` and one capped at ``1 m/s^2`` produce very different
stopping-distance envelopes for the same approach speed, so reporting a near-miss
count without the braking envelope hides a safety-critical degree of freedom.

This module owns the small, pure, JSON-safe helpers that:

* compute the worst-case stopping distance ``d = v^2 / (2a)`` from a peak speed
  ``v`` and a braking-deceleration magnitude ``a``, and
* extract the per-drive-model envelope (forward acceleration, braking
  authority, peak speed, and the resulting stopping distance) from a
  drive-settings object via duck typing, so benchmark run metadata can record
  the envelope alongside sensor-noise and collision-regime provenance.

These helpers are deliberately standard-library only so they can be imported
without simulator dependencies, and they degrade to explicit ``not_available``
sentinels when a drive model exposes no deceleration authority (e.g. a
velocity-level model), instead of silently emitting a meaningless envelope.
"""

from __future__ import annotations

import math
from typing import Any

#: Human-readable note explaining how the envelope block should be interpreted.
ACTUATION_ENVELOPE_INTERPRETATION = (
    "actuation_envelope: enforced acceleration/braking authority and the resulting "
    "worst-case stopping distance v^2/(2a); near-miss / TTC readings must be "
    "interpreted against this braking capability, not in isolation"
)

#: Schema tag for the emitted envelope block, so downstream readers can version it.
ACTUATION_ENVELOPE_SCHEMA = "actuation_envelope.v1"

#: Sentinel for envelope values that cannot be computed from a drive model.
NOT_AVAILABLE = "not_available"


def stopping_distance(speed: float, deceleration: float) -> float:
    """Return the worst-case stopping distance ``v^2 / (2a)`` from constant braking.

    The envelope is the distance required to bring the robot to rest from
    ``speed`` when braking at the constant deceleration magnitude
    ``deceleration``. It is the kinematic lower bound on the braking-feasibility
    envelope central to safety interpretations of near-miss / TTC metrics.

    Args:
        speed: Forward speed magnitude in m/s (must be finite and non-negative).
        deceleration: Braking deceleration magnitude in m/s^2 (must be finite and
            strictly positive).

    Returns:
        The stopping distance in meters.

    Raises:
        ValueError: If ``speed`` is negative or non-finite, or ``deceleration``
            is not finite and strictly positive.
    """
    speed_value = float(speed)
    decel_value = float(deceleration)
    if not math.isfinite(speed_value) or speed_value < 0.0:
        raise ValueError(f"speed must be finite and non-negative (got {speed!r})")
    if not math.isfinite(decel_value) or decel_value <= 0.0:
        raise ValueError(
            f"deceleration must be finite and strictly positive (got {deceleration!r})"
        )
    return speed_value * speed_value / (2.0 * decel_value)


def _as_finite_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it cannot be coerced."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _resolve_diff_drive_envelope(robot_config: Any) -> dict[str, Any] | None:
    """Extract the envelope from a differential-drive settings object.

    Returns:
        The differential-drive envelope block, or ``None`` when the object does
        not expose the differential-drive braking-authority field
        (``max_linear_decel``).
    """
    max_accel = _as_finite_float(getattr(robot_config, "max_linear_accel", None))
    max_decel = _as_finite_float(getattr(robot_config, "max_linear_decel", None))
    # ``max_linear_speed`` is the differential-drive peak forward speed.
    max_speed = _as_finite_float(getattr(robot_config, "max_linear_speed", None))
    if max_accel is None or max_decel is None:
        return None
    return _build_envelope(
        drive_model="differential_drive",
        max_accel=max_accel,
        max_decel=max_decel,
        max_speed=max_speed,
    )


def _resolve_bicycle_envelope(robot_config: Any) -> dict[str, Any] | None:
    """Extract the envelope from a bicycle-drive settings object.

    Returns:
        The bicycle-drive envelope block, or ``None`` when the object does not
        expose the bicycle braking-authority field (``max_decel``).
    """
    max_accel = _as_finite_float(getattr(robot_config, "max_accel", None))
    max_decel = _as_finite_float(getattr(robot_config, "max_decel", None))
    # ``max_velocity`` is the bicycle-drive peak forward speed.
    max_speed = _as_finite_float(getattr(robot_config, "max_velocity", None))
    if max_accel is None or max_decel is None:
        return None
    return _build_envelope(
        drive_model="bicycle_drive",
        max_accel=max_accel,
        max_decel=max_decel,
        max_speed=max_speed,
    )


def _build_envelope(
    *,
    drive_model: str,
    max_accel: float,
    max_decel: float,
    max_speed: float | None,
) -> dict[str, Any]:
    """Assemble a JSON-safe envelope block from resolved scalar limits.

    The worst-case stopping distance is computed at the drive model's peak
    forward speed when that speed is available and finite; otherwise it is
    reported as ``not_available`` so a reader is never given a distance that was
    not actually bounded by a peak speed.

    Returns:
        A JSON-safe actuation-envelope dictionary.
    """
    stopping_dist: float | str
    if max_speed is not None and max_speed > 0.0:
        stopping_dist = stopping_distance(max_speed, max_decel)
    else:
        stopping_dist = NOT_AVAILABLE
    payload: dict[str, Any] = {
        "drive_model": drive_model,
        "max_forward_accel_m_s2": max_accel,
        "max_braking_decel_m_s2": max_decel,
        "braking_distinct_from_accel": max_decel != max_accel,
        "stopping_distance_envelope_m": stopping_dist,
        "envelope_formula": "v^2 / (2a) at peak forward speed and max braking deceleration",
    }
    if max_speed is not None:
        payload["peak_forward_speed_m_s"] = max_speed
    return payload


def actuation_envelope_from_drive_config(robot_config: Any) -> dict[str, Any] | None:
    """Extract a JSON-safe actuation-envelope block from a drive-settings object.

    Reads the drive model's enforced acceleration / braking authority and peak
    forward speed via duck typing. The bicycle drive exposes ``max_accel``,
    ``max_decel`` and ``max_velocity``; the differential drive exposes
    ``max_linear_accel``, ``max_linear_decel`` and ``max_linear_speed``.

    Returns ``None`` when ``robot_config`` exposes neither braking-authority field
    (e.g. a holonomic / velocity-level model with no first-class deceleration
    cap), so callers can omit the block rather than emit a meaningless envelope.

    Args:
        robot_config: A drive-settings-like object (e.g.
            ``DifferentialDriveSettings`` or ``BicycleDriveSettings``), or any
            object exposing the relevant attributes.

    Returns:
        A JSON-safe envelope dictionary, or ``None`` when the drive model exposes
        no braking authority.
    """
    if robot_config is None:
        return None
    # The differential-drive braking-authority field is the discriminator: only
    # the differential drive exposes ``max_linear_decel``.
    if hasattr(robot_config, "max_linear_decel"):
        return _resolve_diff_drive_envelope(robot_config)
    # The bicycle-drive braking-authority field is the discriminator.
    if hasattr(robot_config, "max_decel") and hasattr(robot_config, "max_velocity"):
        return _resolve_bicycle_envelope(robot_config)
    return None


def actuation_envelope_for_run_config(config: Any) -> dict[str, Any] | None:
    """Extract the envelope from an environment config-like object.

    Reads ``config.robot_config`` via duck typing so the helper works on any
    environment config exposing that attribute (e.g. ``EnvSettings`` /
    ``RobotSimulationConfig``).

    Returns:
        The envelope block, or ``None`` when ``config`` has no ``robot_config``
        or the drive model exposes no braking authority.
    """
    robot_config = getattr(config, "robot_config", None)
    return actuation_envelope_from_drive_config(robot_config)


__all__ = [
    "ACTUATION_ENVELOPE_INTERPRETATION",
    "ACTUATION_ENVELOPE_SCHEMA",
    "NOT_AVAILABLE",
    "actuation_envelope_for_run_config",
    "actuation_envelope_from_drive_config",
    "stopping_distance",
]
