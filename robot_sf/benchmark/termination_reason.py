"""Canonical termination reason helpers for evaluation and reporting."""

from __future__ import annotations

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


def _metric_scalar(metrics: Mapping[str, Any], *keys: str) -> float:
    """Return a numeric metric from the first available key.

    Returns:
        float: Parsed metric value or ``0.0`` when missing/invalid.
    """
    for key in keys:
        if key not in metrics:
            continue
        try:
            return float(metrics.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


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
        success_metric = _metric_scalar(metrics, "success", "success_rate")
        collision_metric = _metric_scalar(metrics, "collisions", "collision_rate")
        if collision and success_metric > 0.0:
            contradictions.append("collision outcome but metrics.success > 0")
        if route_complete and collision_metric > 0.0:
            contradictions.append("route_complete outcome but metrics.collisions > 0")
        if route_complete and success_metric <= 0.0:
            contradictions.append("outcome.route_complete=true but metrics.success <= 0")
        if (not route_complete) and success_metric > 0.0:
            contradictions.append("outcome.route_complete=false but metrics.success > 0")
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
