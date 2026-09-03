"""Focused tests for the independent continuous goal-force metric slice."""

import math

import pytest

from robot_sf.benchmark.goal_force_metrics import (
    GOAL_FORCE_METRICS_CLAIM_BOUNDARY,
    GoalForceMetricRow,
    evaluate_goal_force_rows,
)


def test_hand_calculated_vector_and_direction_metrics() -> None:
    summary = evaluate_goal_force_rows(
        [
            GoalForceMetricRow((3.0, 4.0), (0.0, 4.0)),
            GoalForceMetricRow((1.0, 0.0), (1.0, 0.0)),
        ]
    )

    assert summary.exact_vector_count == 2
    assert summary.vector_mae == pytest.approx(1.5)
    assert summary.vector_rmse == pytest.approx(math.sqrt(4.5))
    assert summary.component_bias_xy == pytest.approx((1.5, 0.0))
    assert summary.angular_error_rad == pytest.approx(math.acos(0.8) / 2)
    assert summary.cosine_similarity == pytest.approx((0.8 + 1.0) / 2)
    assert summary.magnitude_mae == pytest.approx(0.5)
    assert summary.claim_boundary == GOAL_FORCE_METRICS_CLAIM_BOUNDARY


def test_censored_rows_are_direction_only_and_missing_truth_is_unavailable() -> None:
    summary = evaluate_goal_force_rows(
        [
            GoalForceMetricRow((2.0, 0.0), (1.0, 0.0), censored=True),
            GoalForceMetricRow((1.0, 0.0), None),
        ]
    )

    assert summary.censored_count == 1
    assert summary.unavailable_count == 1
    assert summary.exact_vector_count == 0
    assert summary.direction_count == 1
    assert summary.magnitude_count == 0
    assert summary.vector_mae is None


def test_zero_vectors_are_not_fabricated_as_direction_accuracy() -> None:
    summary = evaluate_goal_force_rows([GoalForceMetricRow((0.0, 0.0), (0.0, 0.0))])

    assert summary.exact_vector_count == 1
    assert summary.direction_count == 0
    assert summary.angular_error_rad is None
    assert summary.cosine_similarity is None


def test_rows_and_vectors_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        GoalForceMetricRow((1.0, 2.0, 3.0), (1.0, 2.0))
    with pytest.raises(TypeError, match="GoalForceMetricRow"):
        evaluate_goal_force_rows([{"predicted_force_xy": (1.0, 0.0)}])  # type: ignore[list-item]
