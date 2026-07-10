"""Route-corridor geometry helpers for the hybrid-rule local planner family.

Extracted from ``HybridRuleLocalPlannerAdapter`` (issue #4987, part of #4770) as
the lowest-risk first decomposition move: these are pure, ``self``-independent
functions that take a ``route_corridor`` diagnostics dict (and arrays) and
return scalars / points / headings, with no ``self`` mutation. The adapter now
delegates to these module-level functions via thin wrappers so its observable
``plan()`` / ``diagnostics()`` outputs stay byte-identical.

The functions intentionally use their own module-level ``_EPS`` constant that
mirrors the adapter's ``_EPS = 1e-9``; the two must be kept in sync if either is
ever changed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Mirrors ``_EPS`` in ``hybrid_rule_local_planner.py``; kept in sync so extracted
# behavior is byte-identical.
_EPS = 1e-9


def route_point(route_corridor: dict[str, Any] | None, key: str) -> np.ndarray | None:
    """Read one finite ``(x, y)`` point from route-corridor diagnostics.

    Returns:
        np.ndarray | None: Point array, or ``None`` when unavailable.
    """
    if not isinstance(route_corridor, dict):
        return None
    raw = route_corridor.get(key)
    try:
        point = np.asarray(raw, dtype=float).reshape(-1)[:2]
    except (TypeError, ValueError):
        return None
    if point.shape[0] != 2 or not np.all(np.isfinite(point)):
        return None
    return point


def route_float(route_corridor: dict[str, Any] | None, key: str) -> float | None:
    """Read one finite scalar from route-corridor diagnostics.

    Returns:
        float | None: Finite scalar, or ``None`` when unavailable.
    """
    if not isinstance(route_corridor, dict):
        return None
    value = route_corridor.get(key)
    if isinstance(value, int | float | np.integer | np.floating) and np.isfinite(value):
        return float(value)
    return None


def route_progress_pair(
    route_corridor: dict[str, Any] | None,
) -> tuple[float, float] | None:
    """Read finite 1s and 3s route-arc progress diagnostics.

    Returns:
        tuple[float, float] | None: Progress over 1s and 3s, or ``None`` when unavailable.
    """
    if not isinstance(route_corridor, dict):
        return None
    route_progress = route_corridor.get("route_arc_progress_windows")
    if not isinstance(route_progress, dict):
        return None
    try:
        route_progress_1s = float(route_progress["1s"])
        route_progress_3s = float(route_progress["3s"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(route_progress_1s) or not np.isfinite(route_progress_3s):
        return None
    return route_progress_1s, route_progress_3s


def route_tangent_heading(route_corridor: dict[str, Any] | None) -> float | None:
    """Return a finite route tangent heading, deriving it from route points if needed."""
    heading = route_float(route_corridor, "route_tangent_heading")
    if heading is not None:
        return heading
    start = route_point(route_corridor, "route_start_world")
    stop = route_point(route_corridor, "route_next_world")
    if stop is None:
        stop = route_point(route_corridor, "route_waypoint_world")
    if start is None or stop is None:
        return None
    delta = stop - start
    if float(np.linalg.norm(delta)) <= _EPS:
        return None
    return float(np.arctan2(delta[1], delta[0]))


def lateral_offset_to_segment(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_stop: np.ndarray,
) -> float | None:
    """Return lateral distance from ``point`` to a world-space segment."""
    segment = segment_stop - segment_start
    length = float(np.linalg.norm(segment))
    if length <= _EPS:
        return None
    relative = point - segment_start
    return float(abs(segment[0] * relative[1] - segment[1] * relative[0]) / length)
