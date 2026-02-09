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
