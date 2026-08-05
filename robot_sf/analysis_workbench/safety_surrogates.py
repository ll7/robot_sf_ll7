"""Diagnostic safety-surrogate helpers for worked-example process traces."""

from __future__ import annotations

import math
from typing import Any

SAFETY_SURROGATE_PROFILE_VERSION = "worked_example_safety_surrogates.v1"


def proxy_surface_clearance_m(
    center_distance_m: float,
    *,
    robot_radius_m: float | None,
    actor_radius_m: float | None,
) -> dict[str, Any]:
    """Return proxy-envelope clearance or an explicit unavailable record.

    Returns:
        JSON-safe clearance diagnostic record.
    """

    if robot_radius_m is None or actor_radius_m is None:
        return {
            "status": "unavailable",
            "reason": "missing_proxy_radius",
            "value_m": None,
        }
    return {
        "status": "available",
        "reason": "declared_proxy_radii",
        "value_m": center_distance_m - robot_radius_m - actor_radius_m,
    }


def constant_velocity_closest_approach(
    rel_pos: tuple[float, float],
    rel_vel: tuple[float, float],
    *,
    radius_sum_m: float | None,
) -> dict[str, Any]:
    """Project local constant-velocity closest approach, failing closed.

    Returns:
        JSON-safe closest-approach diagnostic record.
    """

    speed_sq = _dot(rel_vel, rel_vel)
    if speed_sq <= 1e-18:
        return {"status": "unavailable", "reason": "degenerate_relative_velocity"}
    t_star = -_dot(rel_pos, rel_vel) / speed_sq
    if t_star < 0:
        return {"status": "unavailable", "reason": "closest_approach_in_past_or_diverging"}
    closest = (rel_pos[0] + rel_vel[0] * t_star, rel_pos[1] + rel_vel[1] * t_star)
    center_distance = math.hypot(closest[0], closest[1])
    proxy_clearance = None if radius_sum_m is None else center_distance - radius_sum_m
    return {
        "status": "available",
        "time_to_closest_approach_s": t_star,
        "center_distance_at_closest_approach_m": center_distance,
        "proxy_surface_clearance_at_closest_approach_m": proxy_clearance,
        "proxy_surface_clearance_status": (
            "available" if radius_sum_m is not None else "unavailable"
        ),
        "proxy_surface_clearance_reason": (
            "declared_proxy_radii" if radius_sum_m is not None else "missing_proxy_radius"
        ),
        "model": "local_constant_velocity",
        "profile_version": SAFETY_SURROGATE_PROFILE_VERSION,
    }


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]
