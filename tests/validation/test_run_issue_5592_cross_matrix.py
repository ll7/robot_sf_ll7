"""Regression tests for the issue #5592 cross-matrix runtime row gate."""

from __future__ import annotations

import pytest

from scripts.validation.run_issue_5592_cross_matrix import (
    COMPLETED_TIMEOUT_REASONS,
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
            "timeout_event": termination_reason in COMPLETED_TIMEOUT_REASONS,
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


def test_successful_reference_row_is_completed() -> None:
    """A native successful episode is eligible for aggregation."""
    require_completed_execution_row(
        _row(status="success", termination_reason="success"),
        matrix_name="reference",
        planner_key="goal",
    )


@pytest.mark.parametrize("termination_reason", sorted(COMPLETED_TIMEOUT_REASONS))
def test_timeout_terminated_reference_row_is_completed(termination_reason: str) -> None:
    """A horizon timeout remains a completed benchmark-negative episode."""
    require_completed_execution_row(
        _row(status="failure", termination_reason=termination_reason),
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


@pytest.mark.parametrize("metadata_status", [None, "error", "fallback", "placeholder"])
def test_non_ok_algorithm_metadata_still_fails_closed(metadata_status: str | None) -> None:
    """Missing or non-OK planner metadata cannot enter the structural ranking."""
    row = _row(status="success", termination_reason="success")
    metadata = row["algorithm_metadata"]
    assert isinstance(metadata, dict)
    if metadata_status is None:
        metadata.pop("status")
    else:
        metadata["status"] = metadata_status

    with pytest.raises(CrossMatrixRowError, match="algorithm_metadata.status"):
        require_completed_execution_row(row, matrix_name="reference", planner_key="goal")


@pytest.mark.parametrize("execution_mode", [None, "degraded", "fallback", "unknown"])
def test_non_benchmark_execution_modes_still_fail_closed(execution_mode: str | None) -> None:
    """Only declared native, adapter, or mixed execution is aggregation-eligible."""
    row = _row(status="success", termination_reason="success")
    metadata = row["algorithm_metadata"]
    assert isinstance(metadata, dict)
    kinematics = metadata["planner_kinematics"]
    assert isinstance(kinematics, dict)
    if execution_mode is None:
        kinematics.pop("execution_mode")
    else:
        kinematics["execution_mode"] = execution_mode

    with pytest.raises(CrossMatrixRowError, match="execution_mode"):
        require_completed_execution_row(row, matrix_name="reference", planner_key="goal")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("availability_status", "not_available"),
        ("availability_status", "partial-failure"),
        ("readiness_status", "degraded"),
        ("readiness_status", "fallback"),
    ],
)
def test_non_benchmark_availability_metadata_still_fails_closed(field: str, value: str) -> None:
    """Optional campaign availability markers are enforced when present on a row."""
    row = _row(status="success", termination_reason="success")
    metadata = row["algorithm_metadata"]
    assert isinstance(metadata, dict)
    metadata[field] = value

    with pytest.raises(CrossMatrixRowError, match=field):
        require_completed_execution_row(row, matrix_name="reference", planner_key="goal")


@pytest.mark.parametrize("missing_key", ["outcome", "algorithm_metadata"])
def test_missing_required_metadata_still_fails_closed(missing_key: str) -> None:
    """Rows lacking required outcome or algorithm metadata are malformed."""
    row = _row(status="success", termination_reason="success")
    row.pop(missing_key)

    with pytest.raises(CrossMatrixRowError, match="missing"):
        require_completed_execution_row(row, matrix_name="reference", planner_key="goal")


def test_non_completed_outcome_still_fails_closed() -> None:
    """A failure without a matching timeout outcome is not a completed row."""
    row = _row(status="failure", termination_reason="max_steps")
    outcome = row["outcome"]
    assert isinstance(outcome, dict)
    outcome["timeout_event"] = False

    with pytest.raises(CrossMatrixRowError, match="non-completed execution row"):
        require_completed_execution_row(row, matrix_name="reference", planner_key="goal")
