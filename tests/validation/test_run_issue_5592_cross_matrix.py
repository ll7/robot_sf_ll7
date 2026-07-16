"""Regression tests for the issue #5592 cross-matrix runtime row gate."""

from __future__ import annotations

import pytest

from scripts.validation.run_issue_5592_cross_matrix import (
    CrossMatrixRowError,
    require_completed_execution_row,
)


def _row(*, status: str, termination_reason: str) -> dict[str, object]:
    """Build the smallest row carrying valid execution and outcome metadata."""
    return {
        "status": status,
        "termination_reason": termination_reason,
        "outcome": {
            "collision_event": termination_reason == "collision",
            "route_complete": termination_reason == "success",
            "timeout_event": termination_reason in {"max_steps", "terminated", "timeout"},
        },
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "adapter"},
        },
        "result_provenance": {"repo_commit": "0" * 40},
    }


def test_collision_terminated_reference_row_is_completed() -> None:
    """A collision is benchmark-negative data, not an execution failure."""
    require_completed_execution_row(
        _row(status="collision", termination_reason="collision"),
        matrix_name="reference",
        planner_key="scenario_adaptive_hybrid_orca_v1",
    )


def test_timeout_terminated_reference_row_is_completed() -> None:
    """A horizon timeout remains a completed benchmark-negative episode."""
    require_completed_execution_row(
        _row(status="failure", termination_reason="max_steps"),
        matrix_name="reference",
        planner_key="scenario_adaptive_hybrid_orca_v1",
    )


def test_genuine_execution_error_still_fails_closed() -> None:
    """An error termination cannot enter the structural ranking."""
    with pytest.raises(CrossMatrixRowError, match="execution error reference/goal"):
        require_completed_execution_row(
            _row(status="failure", termination_reason="error"),
            matrix_name="reference",
            planner_key="goal",
        )


def test_fallback_execution_still_fails_closed() -> None:
    """Adapter fallback remains ineligible even when an episode row exists."""
    row = _row(status="success", termination_reason="success")
    metadata = row["algorithm_metadata"]
    assert isinstance(metadata, dict)
    metadata["planner_kinematics"] = {"execution_mode": "fallback"}

    with pytest.raises(CrossMatrixRowError, match="ineligible execution reference/goal"):
        require_completed_execution_row(
            row,
            matrix_name="reference",
            planner_key="goal",
        )
