"""Canonical termination reason helpers for evaluation and reporting."""

from __future__ import annotations

TERMINATION_REASONS: tuple[str, ...] = (
    "success",
    "collision",
    "terminated",
    "truncated",
    "max_steps",
    "error",
)


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

    Returns:
        str: One of ``TERMINATION_REASONS``.
    """
    if had_error:
        return "error"
    if terminated:
        if success:
            return "success"
        if collision:
            return "collision"
        return "terminated"
    if truncated:
        return "truncated"
    if reached_max_steps:
        return "max_steps"
    # Defensive fallback for callers that only provide info flags.
    if success:
        return "success"
    if collision:
        return "collision"
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
