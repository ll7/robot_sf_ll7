#!/usr/bin/env python3
"""Runtime guards for the issue #5592 paired cross-matrix campaign.

The campaign treats collision and timeout as benchmark outcomes, not process
failures.  This module keeps that distinction explicit while continuing to
reject execution errors and fallback/degraded planner rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COMPLETED_TIMEOUT_REASONS = frozenset({"max_steps", "terminated", "timeout", "truncated"})
FORBIDDEN_EXECUTION_STATUSES = frozenset(
    {"blocked", "degraded", "error", "failed", "fallback", "not_available", "skipped"}
)


class CrossMatrixRowError(RuntimeError):
    """Raised when an episode row is not eligible for cross-matrix aggregation."""


def _normalized(value: object) -> str:
    """Return a lowercase status token, or an empty string when absent."""
    return str(value or "").strip().lower()


def require_completed_execution_row(
    row: Mapping[str, Any],
    *,
    matrix_name: str,
    planner_key: str,
) -> None:
    """Require a completed native/adapter episode with eligible execution metadata.

    Collision and timeout are completed benchmark-negative outcomes. Execution
    errors and fallback/degraded/unavailable modes remain fatal so they cannot
    enter the structural ranking.

    Args:
        row: Parsed benchmark episode record.
        matrix_name: Matrix label used in failure diagnostics.
        planner_key: Planner roster key used in failure diagnostics.

    Raises:
        CrossMatrixRowError: If the row is malformed, represents an execution
            error, or carries fallback/degraded execution metadata.
    """
    label = f"{matrix_name}/{planner_key}"
    status = _normalized(row.get("status"))
    termination = _normalized(row.get("termination_reason"))
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        raise CrossMatrixRowError(f"missing outcome metadata {label}")

    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise CrossMatrixRowError(f"missing algorithm metadata {label}")
    kinematics = metadata.get("planner_kinematics")
    kinematics = kinematics if isinstance(kinematics, Mapping) else {}
    observed_execution_statuses = {
        _normalized(metadata.get("status")),
        _normalized(kinematics.get("execution_mode")),
        _normalized(metadata.get("availability_status")),
    }
    forbidden = observed_execution_statuses & FORBIDDEN_EXECUTION_STATUSES
    if forbidden:
        raise CrossMatrixRowError(f"ineligible execution {label}: {sorted(forbidden)}")

    if status in {"error", "failed"} or termination == "error":
        raise CrossMatrixRowError(
            f"execution error {label}: status={status!r}, termination_reason={termination!r}"
        )

    collision_completed = (
        status == "collision"
        and termination == "collision"
        and outcome.get("collision_event") is True
    )
    success_completed = (
        status == "success" and termination == "success" and outcome.get("route_complete") is True
    )
    timeout_completed = (
        status in {"failure", "timeout"}
        and termination in COMPLETED_TIMEOUT_REASONS
        and outcome.get("timeout_event") is True
    )
    if collision_completed or success_completed or timeout_completed:
        return

    raise CrossMatrixRowError(
        f"non-completed execution row {label}: "
        f"status={status!r}, termination_reason={termination!r}"
    )
