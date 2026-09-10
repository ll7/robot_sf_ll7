"""Tests for fail-closed continuation experiment scaffolding."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.continuation_experiment_plan import (
    BLOCKED_ADMISSION,
    CONTINUATION_MODES,
    ContinuationExperimentPlan,
    estimate_continuation_cost,
    evaluate_admission,
)


def test_preparation_plan_requires_all_comparison_modes() -> None:
    """The plan keeps full, exact, pose-only, and late-crop controls together."""
    plan = ContinuationExperimentPlan("rw06-rw07-preparation")
    assert plan.modes == CONTINUATION_MODES
    receipt = evaluate_admission(plan)
    assert receipt.status == BLOCKED_ADMISSION
    assert "execution_not_enabled_in_preparation_plan" in receipt.blockers
    assert "missing_live_authority:#7381" in receipt.blockers


def test_plan_rejects_dropped_negative_control() -> None:
    """A plan cannot silently remove the pose-only or late-crop negative control."""
    with pytest.raises(ValueError, match="modes must be exactly"):
        ContinuationExperimentPlan("bad", modes=("full_parent", "exact_restart"))


def test_execution_enabled_plan_requires_nonempty_authority_ids() -> None:
    """An execution-enabled plan cannot reach READY_NOT_LAUNCHED without authority IDs."""
    with pytest.raises(ValueError, match="non-empty authority_issue_ids"):
        ContinuationExperimentPlan(
            "missing-authority",
            authority_issue_ids=(),
            execution_allowed=True,
        )


def test_cost_table_reports_no_positive_break_even() -> None:
    """When loading plus the window costs more than a full branch, report no break-even."""
    estimate = estimate_continuation_cost(
        parent_steps=100,
        window_steps=90,
        branches=2,
        parent_cost_s=10.0,
        selection_cost_s=1.0,
        validation_cost_s=1.0,
        load_cost_s=10.0,
        step_cost_s=0.1,
    )
    assert estimate["break_even_status"] == "no_positive_break_even"
    assert estimate["crossover_reuse_count"] is None
    assert estimate["plain_total_s"] == pytest.approx(20.0)


def test_cost_table_reports_positive_crossover() -> None:
    """Cheap repeated windows expose a transparent positive crossover count."""
    estimate = estimate_continuation_cost(
        parent_steps=100,
        window_steps=10,
        branches=3,
        parent_cost_s=10.0,
        selection_cost_s=1.0,
        validation_cost_s=1.0,
        load_cost_s=0.1,
        step_cost_s=0.1,
    )
    assert estimate["break_even_status"] == "positive"
    assert estimate["crossover_reuse_count"] == 2
    assert estimate["marginal_savings_per_branch_s"] > 0.0
