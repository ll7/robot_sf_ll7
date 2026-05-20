"""Shared scalar math helpers used across Robot SF modules."""

from __future__ import annotations

from math import atan2, cos, pi, sin


def wrap_angle_pi(angle: float) -> float:
    """Wrap an angle to the ``[-pi, pi)`` interval.

    This preserves the modulo-based behavior historically used by most
    planner helpers: positive odd multiples of ``pi`` wrap to ``-pi``.

    Returns:
        float: Wrapped angle in radians.
    """
    return float((float(angle) + pi) % (2.0 * pi) - pi)


def wrap_angle_pi_closed(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]`` while keeping positive ``pi``.

    This matches helpers that previously returned ``pi`` for positive odd
    multiples of ``pi`` and ``-pi`` for negative odd multiples.

    Returns:
        float: Wrapped angle in radians.
    """
    value = float(angle)
    wrapped = wrap_angle_pi(value)
    if wrapped == -pi and value > 0.0:
        return pi
    return wrapped


def normalize_angle_atan2(angle: float) -> float:
    """Normalize an angle using ``atan2(sin(angle), cos(angle))`` semantics.

    Returns:
        float: Wrapped angle in radians.
    """
    value = float(angle)
    return float(atan2(sin(value), cos(value)))
