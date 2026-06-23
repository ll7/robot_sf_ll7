"""Tests for issue #2601 adversarial manifest quality metrics."""

from __future__ import annotations

from robot_sf.adversarial.manifest_quality import (
    PlannerOutcome,
    _has_low_progress,
    _row_is_failure,
)


def test_has_low_progress_true_on_timeout_with_low_displacement() -> None:
    """A timeout row with displacement below 0.25m should be low-progress."""
    row = {"termination_reason": "timeout", "robot_displacement_m": 0.1}
    assert _has_low_progress(row) is True


def test_has_low_progress_false_on_success() -> None:
    """A successful row should not be low-progress."""
    row = {"termination_reason": "success", "robot_displacement_m": 0.1}
    assert _has_low_progress(row) is False


def test_has_low_progress_false_on_high_displacement() -> None:
    """A timeout row with displacement above 0.25m should not be low-progress."""
    row = {"termination_reason": "timeout", "robot_displacement_m": 1.0}
    assert _has_low_progress(row) is False


def test_has_low_progress_false_on_collision_termination() -> None:
    """Collision-term rows should not be classified as low-progress."""
    row = {"termination_reason": "collision", "robot_displacement_m": 0.1}
    assert _has_low_progress(row) is False


def test_has_low_progress_handles_missing_displacement() -> None:
    """Rows without displacement should not be flagged as low-progress."""
    row = {"termination_reason": "timeout"}
    assert _has_low_progress(row) is False


def test_planner_outcome_includes_low_progress_fields() -> None:
    """PlannerOutcome should carry low_progress_count and low_progress_yield."""
    outcome = PlannerOutcome(
        planner="test",
        episodes=10,
        failure_count=2,
        near_miss_count=1,
        failure_yield=0.2,
        near_miss_yield=0.1,
        low_progress_count=3,
        low_progress_yield=0.3,
        source="test",
    )
    assert outcome.low_progress_count == 3
    assert outcome.low_progress_yield == 0.3


def test_planner_outcome_to_dict_includes_low_progress() -> None:
    """PlannerOutcome.to_dict should include low_progress fields."""
    outcome = PlannerOutcome(
        planner="test",
        episodes=10,
        failure_count=2,
        near_miss_count=1,
        failure_yield=0.2,
        near_miss_yield=0.1,
        low_progress_count=3,
        low_progress_yield=0.3,
        source="test",
    )
    d = outcome.to_dict()
    assert d["low_progress_count"] == 3
    assert d["low_progress_yield"] == 0.3


def test_has_low_progress_accepts_truncated_and_max_steps() -> None:
    """Truncated and max_steps terminations should also be low-progress candidates."""
    for term in ("truncated", "max_steps"):
        row = {"termination_reason": term, "robot_displacement_m": 0.1}
        assert _has_low_progress(row) is True, f"Failed for {term}"


def test_row_is_failure_vs_low_progress_collision() -> None:
    """A collision row is both a failure AND NOT low-progress."""
    row = {"termination_reason": "collision", "robot_displacement_m": 0.1}
    assert _row_is_failure(row) is True
    assert _has_low_progress(row) is False


def test_row_is_failure_vs_low_progress_timeout_low_disp() -> None:
    """A timeout with low displacement is both a failure AND low-progress."""
    row = {"termination_reason": "timeout", "robot_displacement_m": 0.1}
    assert _row_is_failure(row) is True
    assert _has_low_progress(row) is True
