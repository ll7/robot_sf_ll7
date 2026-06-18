"""Metric normalization helpers for map-based benchmark runs."""

from __future__ import annotations

import math
from typing import Any


def normalize_pedestrian_impact_controls(
    *,
    experimental_ped_impact: bool,
    ped_impact_radius_m: float,
    ped_impact_window_steps: int,
) -> tuple[float, int]:
    """Normalize pedestrian-impact controls and fail fast for invalid opt-in values.

    Returns:
        Normalized radius/window pair for downstream metric computation.
    """

    radius = float(ped_impact_radius_m)
    window_value = float(ped_impact_window_steps)
    window_steps = int(window_value)
    if experimental_ped_impact:
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("ped_impact_radius_m must be a finite value > 0.")
        if (
            not math.isfinite(window_value)
            or float(window_steps) != window_value
            or window_steps < 1
        ):
            raise ValueError("ped_impact_window_steps must be an integer >= 1.")
    return radius, window_steps


def collision_metric_value(metrics: dict[str, Any], key: str) -> float:
    """Return a finite collision metric value, treating missing/non-finite values as zero."""
    value = metrics.get(key)
    if value is None:
        value = 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def floor_collision_metrics_from_flags(
    metrics: dict[str, Any],
    *,
    collision_seen: bool,
    ped_collision_seen: bool,
    obstacle_collision_seen: bool,
    robot_collision_seen: bool,
) -> None:
    """Preserve exact environment collision flags when sampled metrics miss the contact.

    Obstacle metrics are computed from sampled wall points, while the environment detects obstacle
    collisions against exact geometry. When the exact detector reports a collision between sampled
    points, keep the episode usable by flooring the corresponding count to one instead of failing
    the outcome/metric integrity check.
    """

    collision_keys = {
        "ped_collision_count": ped_collision_seen,
        "obstacle_collision_count": obstacle_collision_seen,
        "agent_collision_count": robot_collision_seen,
    }
    for key, typed_collision_seen in collision_keys.items():
        if typed_collision_seen and collision_metric_value(metrics, key) <= 0.0:
            metrics[key] = 1.0

    typed_collision_count = sum(collision_metric_value(metrics, key) for key in collision_keys)
    sampled_collision_count = max(
        collision_metric_value(metrics, "total_collision_count"),
        collision_metric_value(metrics, "collisions"),
        collision_metric_value(metrics, "wall_collisions"),
    )
    if typed_collision_count > 0.0:
        aggregate_collision_count = max(sampled_collision_count, typed_collision_count)
        metrics["total_collision_count"] = aggregate_collision_count
        metrics["collisions"] = aggregate_collision_count
        if obstacle_collision_seen and collision_metric_value(metrics, "wall_collisions") <= 0.0:
            metrics["wall_collisions"] = collision_metric_value(metrics, "obstacle_collision_count")
    elif collision_seen:
        aggregate_collision_count = max(sampled_collision_count, 1.0)
        metrics["total_collision_count"] = aggregate_collision_count
        metrics["collisions"] = aggregate_collision_count
