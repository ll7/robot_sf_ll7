"""Canonical termination reason helpers for evaluation and reporting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

TERMINATION_REASONS: tuple[str, ...] = (
    "success",
    "collision",
    "terminated",
    "truncated",
    "max_steps",
    "error",
)

_COLLISION_COMPONENT_KEYS: tuple[str, ...] = (
    "ped_collision_count",
    "obstacle_collision_count",
    "agent_collision_count",
)


def route_complete_success(info: Mapping[str, Any] | None) -> bool:
    """Return success based strictly on route completion."""
    if not isinstance(info, Mapping):
        return False
    meta = info.get("meta")
    if not isinstance(meta, Mapping):
        return False
    return bool(meta.get("is_route_complete"))


def collision_event(info: Mapping[str, Any] | None) -> bool:
    """Return whether a collision event is present in the step payload."""
    if not isinstance(info, Mapping):
        return False
    if bool(info.get("collision")):
        return True
    meta = info.get("meta")
    if not isinstance(meta, Mapping):
        return False
    return bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_robot_collision")
        or meta.get("is_obstacle_collision")
    )


def build_outcome_payload(
    *,
    route_complete: bool,
    collision: bool,
    timeout: bool,
) -> dict[str, bool]:
    """Build canonical outcome payload for episode artifacts.

    Returns:
        dict[str, bool]: Canonical route/collision/timeout flags.
    """
    return {
        "route_complete": bool(route_complete),
        "collision_event": bool(collision),
        "timeout_event": bool(timeout),
    }


def _resolved_total_collision_count(metrics: Mapping[str, Any]) -> float:
    """Return the best available collision count from an episode metrics payload."""
    explicit_total = metric_scalar(metrics, "total_collision_count", default=float("nan"))
    if math.isfinite(explicit_total):
        return max(0.0, explicit_total)

    component_total = 0.0
    saw_component = False
    for key in _COLLISION_COMPONENT_KEYS:
        if key not in metrics:
            continue
        saw_component = True
        component_value = metric_scalar(metrics, key, default=0.0)
        if not math.isfinite(component_value):
            continue
        component_total += max(0.0, component_value)
    if saw_component:
        return component_total if math.isfinite(component_total) else 0.0

    fallback = metric_scalar(metrics, "collisions", default=0.0)
    return max(0.0, fallback) if math.isfinite(fallback) else 0.0


def canonicalize_collision_metrics(
    metrics: Mapping[str, Any] | None,
    *,
    collision: bool,
) -> dict[str, Any]:
    """Return metrics with canonical episode-level collision encoding.

    New episode outputs use ``metrics.collisions`` as the authoritative 0/1
    event flag that mirrors ``outcome.collision_event``. Any count-style signal
    remains available under ``total_collision_count`` and the component count
    fields.
    """
    normalized = dict(metrics.items()) if isinstance(metrics, Mapping) else {}
    if isinstance(metrics, Mapping) and (
        "total_collision_count" not in normalized
        and (
            "collisions" in normalized
            or any(key in normalized for key in _COLLISION_COMPONENT_KEYS)
        )
    ):
        normalized["total_collision_count"] = int(_resolved_total_collision_count(metrics))
    normalized["collisions"] = int(bool(collision))
    return normalized


def _metric_outcome_contradictions(
    *,
    route_complete: bool,
    collision: bool,
    metrics: Mapping[str, Any],
) -> list[str]:
    """Return contradiction messages involving metric aliases and outcome flags."""
    contradictions: list[str] = []
    success_metric = _metric_scalar(metrics, "success", "success_rate")
    collision_metric = _metric_scalar(metrics, "collisions", "collision_rate")
    if collision and collision_metric <= 0.0:
        contradictions.append("outcome.collision_event=true but metrics.collisions <= 0")
    if (not collision) and collision_metric > 0.0:
        contradictions.append("outcome.collision_event=false but metrics.collisions > 0")
    if collision and success_metric > 0.0:
        contradictions.append("collision outcome but metrics.success > 0")
    if route_complete and collision_metric > 0.0:
        contradictions.append("route_complete outcome but metrics.collisions > 0")
    if route_complete and success_metric <= 0.0:
        contradictions.append("outcome.route_complete=true but metrics.success <= 0")
    if (not route_complete) and success_metric > 0.0:
        contradictions.append("outcome.route_complete=false but metrics.success > 0")
    return contradictions


def metric_scalar(
    metrics: Mapping[str, Any] | None,
    *keys: str,
    default: float = 0.0,
) -> float:
    """Return a numeric metric from the first available key.

    Returns:
        float: Parsed metric value or ``default`` when missing/invalid.
    """
    if metrics is None:
        return float(default)
    for key in keys:
        if key not in metrics:
            continue
        try:
            return float(metrics.get(key) or default)
        except (TypeError, ValueError):
            return float(default)
    return float(default)


def _metric_scalar(metrics: Mapping[str, Any], *keys: str) -> float:
    """Backward-compatible internal alias for metric parsing.

    Returns:
        float: Parsed metric value or ``0.0`` when missing/invalid.
    """
    return metric_scalar(metrics, *keys, default=0.0)


def outcome_contradictions(
    *,
    termination_reason: str,
    outcome: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return semantic contradictions for one episode payload."""
    contradictions: list[str] = []
    route_complete = bool(outcome.get("route_complete"))
    collision = bool(outcome.get("collision_event"))

    if collision and route_complete:
        contradictions.append("outcome has both collision_event=true and route_complete=true")

    term = str(termination_reason).strip()
    if term == "collision" and route_complete:
        contradictions.append("termination_reason=collision but outcome.route_complete=true")
    if term == "success" and collision:
        contradictions.append("termination_reason=success but outcome.collision_event=true")

    if metrics is not None:
        contradictions.extend(
            _metric_outcome_contradictions(
                route_complete=route_complete,
                collision=collision,
                metrics=metrics,
            )
        )
    return contradictions


def resolve_termination_reason(
    *,
    terminated: bool,
    truncated: bool,
    success: bool,
    collision: bool,
    reached_max_steps: bool = False,
    had_error: bool = False,
) -> str:
    """Resolve a normalized termination reason from step outcomes.

    Precedence is: ``error`` > terminal/truncation signals > info flags.
    When both ``success`` and ``collision`` are true, ``collision`` wins to
    match collision-aware success semantics in benchmark metrics.
    If no signal is present at all, the resolver defaults to ``"max_steps"``.

    Returns:
        str: One of ``TERMINATION_REASONS``.
    """
    if had_error:
        return "error"
    if terminated:
        if collision:
            return "collision"
        if success:
            return "success"
        return "terminated"
    if truncated:
        return "truncated"
    if reached_max_steps:
        return "max_steps"
    # Defensive fallback for callers that only provide info flags.
    if collision:
        return "collision"
    if success:
        return "success"
    return "max_steps"


def status_from_termination_reason(reason: str) -> str:
    """Map termination reason to the high-level status field used in reports.

    Returns:
        str: ``"success"``, ``"collision"``, or ``"failure"``.
    """
    if reason == "success":
        return "success"
    if reason == "collision":
        return "collision"
    return "failure"
