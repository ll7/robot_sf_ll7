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


def _finite_float(value: Any) -> float | None:
    """Return a finite float value, or ``None`` when the input is unavailable."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _exact_collision_event(record: dict[str, Any]) -> bool | None:
    """Return the exact environment collision flag for one episode record.

    Returns:
        ``True``/``False`` when ``outcome.collision_event`` is present and not null,
        otherwise ``None`` when the exact flag is unavailable.
    """
    outcome = record.get("outcome")
    if not isinstance(outcome, dict) or "collision_event" not in outcome:
        return None
    event = outcome.get("collision_event")
    if event is None:
        return None
    return bool(event)


def _episode_collision_value(record: dict[str, Any]) -> tuple[float | None, str | None]:
    """Return one episode's collision value and source path when available.

    Fails closed against silent collision undercounting: when the exact environment
    collision flag (``outcome.collision_event``) fired, the returned value is floored to at
    least 1.0 even if the sampled collision metrics missed the contact and report zero. The
    no-collision case is never inflated -- an exact flag of ``False`` (or an unavailable flag)
    leaves the sampled metric untouched.
    """
    exact_event = _exact_collision_event(record)

    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key in ("collisions", "total_collision_count", "collision_count"):
            if key in metrics:
                value = _finite_float(metrics.get(key))
                if value is not None:
                    if exact_event and value <= 0.0:
                        return 1.0, "episode.outcome.collision_event"
                    return value, f"episode.metrics.{key}"

    if exact_event is not None:
        return (1.0 if exact_event else 0.0), "episode.outcome.collision_event"
    return None, None


def summarize_collision_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a batch-level collision summary from successful episode records.

    Returns:
        Backward-compatible scalar aliases plus explicit availability metadata.
    """
    denominator = len(records)
    if denominator <= 0:
        return {
            "collision": "not_available",
            "collision_count": "not_available",
            "collision_rate": "not_available",
            "collision_status": {
                "status": "not_available",
                "reason": "no successful episode records were available for aggregation",
                "denominator": 0,
                "source": None,
            },
        }

    collision_count = 0.0
    collided_episodes = 0
    available = 0
    sources: set[str] = set()
    for record in records:
        value, source = _episode_collision_value(record)
        if value is None:
            continue
        if source is not None:
            sources.add(source)
        available += 1
        collision_count += value
        if value > 0.0:
            collided_episodes += 1

    if available <= 0:
        return {
            "collision": "not_available",
            "collision_count": "not_available",
            "collision_rate": "not_available",
            "collision_status": {
                "status": "not_available",
                "reason": "successful episode records did not emit collision metrics",
                "denominator": denominator,
                "source": None,
            },
        }

    status = "available" if available == denominator else "partial"
    reason = None if status == "available" else "some successful records lacked collision metrics"
    source = ",".join(sorted(sources))
    return {
        "collision": float(collision_count),
        "collision_count": float(collision_count),
        "collision_rate": float(collided_episodes / available),
        "collision_status": {
            "status": status,
            "reason": reason,
            "denominator": denominator,
            "source": source,
        },
    }


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
