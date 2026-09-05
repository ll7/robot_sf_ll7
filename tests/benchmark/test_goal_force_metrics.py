"""Focused tests for the independent continuous goal-force metric slice."""

import json
import math
from decimal import Decimal

import numpy as np
import pytest

from robot_sf.benchmark.goal_force_metrics import (
    GOAL_FORCE_METRICS_CLAIM_BOUNDARY,
    GOAL_FORCE_METRICS_NORM_EPSILON,
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
    assert summary.norm_epsilon == GOAL_FORCE_METRICS_NORM_EPSILON
    assert summary.vector_l2_mae == pytest.approx(1.5)
    assert summary.vector_l2_rmse == pytest.approx(math.sqrt(4.5))
    assert summary.component_bias_xy == pytest.approx((1.5, 0.0))
    assert summary.angular_error_rad == pytest.approx(math.acos(0.8) / 2)
    assert summary.cosine_similarity == pytest.approx((0.8 + 1.0) / 2)
    assert summary.magnitude_mae == pytest.approx(0.5)
    assert summary.direction_count == 2
    assert summary.direction_excluded_count == 0
    assert summary.magnitude_count == 2
    assert summary.relative_magnitude_count == 2
    assert summary.relative_magnitude_excluded_count == 0
    assert summary.claim_boundary == GOAL_FORCE_METRICS_CLAIM_BOUNDARY
    assert summary.to_dict()["vector_l2_mae"] == pytest.approx(1.5)
    assert summary.to_dict()["norm_epsilon"] == GOAL_FORCE_METRICS_NORM_EPSILON
    assert "vector_mae" not in summary.to_dict()


def test_censored_rows_are_direction_only_and_missing_truth_is_unavailable() -> None:
    summary = evaluate_goal_force_rows(
        [
            GoalForceMetricRow((2.0, 0.0), (1.0, 0.0), censored=True),
            GoalForceMetricRow((1.0, 0.0), None, censored=True),
        ]
    )

    assert summary.censored_count == 2
    assert summary.unavailable_count == 1
    assert summary.exact_vector_count == 0
    assert summary.direction_count == 1
    assert summary.direction_excluded_count == 0
    assert summary.magnitude_count == 0
    assert summary.relative_magnitude_count == 0
    assert summary.relative_magnitude_excluded_count == 0
    assert summary.vector_l2_mae is None


def test_zero_vectors_are_not_fabricated_as_direction_accuracy() -> None:
    summary = evaluate_goal_force_rows(
        [
            GoalForceMetricRow((1.0, 0.0), (0.0, 0.0)),
            GoalForceMetricRow((0.0, 0.0), (0.0, 0.0)),
        ]
    )

    assert summary.exact_vector_count == 2
    assert summary.direction_count == 0
    assert summary.direction_excluded_count == 2
    assert summary.angular_error_rad is None
    assert summary.cosine_similarity is None
    assert summary.magnitude_count == 2
    assert summary.relative_magnitude_count == 0
    assert summary.relative_magnitude_excluded_count == 2


def test_norm_epsilon_boundary_is_counted_and_serialized() -> None:
    summary = evaluate_goal_force_rows(
        [
            GoalForceMetricRow(
                (0.5 * GOAL_FORCE_METRICS_NORM_EPSILON, 0.0),
                (0.5 * GOAL_FORCE_METRICS_NORM_EPSILON, 0.0),
            ),
            GoalForceMetricRow(
                (2.0 * GOAL_FORCE_METRICS_NORM_EPSILON, 0.0),
                (2.0 * GOAL_FORCE_METRICS_NORM_EPSILON, 0.0),
            ),
        ]
    )

    assert summary.direction_count == 1
    assert summary.direction_excluded_count == 1
    assert summary.relative_magnitude_count == 1
    assert summary.relative_magnitude_excluded_count == 1
    assert summary.to_dict()["direction_excluded_count"] == 1
    assert summary.to_dict()["relative_magnitude_excluded_count"] == 1


def test_extreme_finite_opposite_vectors_are_stable_and_strict_json_safe() -> None:
    component = 1e200
    summary = evaluate_goal_force_rows(
        [GoalForceMetricRow((component, component), (-component, -component))]
    )

    expected_error = math.hypot(2.0 * component, 2.0 * component)
    assert summary.vector_l2_mae == pytest.approx(expected_error)
    assert summary.vector_l2_rmse == pytest.approx(expected_error)
    assert summary.angular_error_rad == pytest.approx(math.pi)
    assert summary.cosine_similarity == pytest.approx(-1.0)
    json.dumps(summary.to_dict(), allow_nan=False)


def test_unrepresentable_derived_metric_fails_closed_as_unavailable() -> None:
    with pytest.raises(ValueError, match="result unavailable"):
        evaluate_goal_force_rows([GoalForceMetricRow((1e308, 0.0), (-1e308, 0.0))])


def test_rows_and_vectors_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        GoalForceMetricRow((1.0, 2.0, 3.0), (1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GoalForceMetricRow"):
        evaluate_goal_force_rows([{"predicted_force_xy": (1.0, 0.0)}])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        GoalForceMetricRow((math.nan, 0.0), (0.0, 0.0))


@pytest.mark.parametrize(
    "component",
    [
        pytest.param(False, id="builtin-false"),
        pytest.param(True, id="builtin-true"),
        pytest.param(np.bool_(False), id="numpy-false"),
        pytest.param(np.bool_(True), id="numpy-true"),
    ],
)
def test_boolean_vector_components_are_not_numeric(component: object) -> None:
    """Builtin and NumPy booleans must not silently become force components."""
    with pytest.raises(ValueError, match="finite numeric"):
        GoalForceMetricRow((component, 0.0), (0.0, 0.0))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "component",
    [
        pytest.param("1.0", id="numeric-string"),
        pytest.param(1 + 2j, id="builtin-complex"),
        pytest.param(np.complex64(1 + 2j), id="numpy-complex"),
        pytest.param(Decimal("1.0"), id="decimal"),
        pytest.param(np.array(1.0), id="zero-dimensional-array"),
    ],
)
def test_non_real_vector_components_are_not_coerced(component: object) -> None:
    """Vector components must be real scalar values, not merely float-coercible."""
    with pytest.raises(ValueError, match="finite numeric"):
        GoalForceMetricRow((component, 0.0), (0.0, 0.0))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "component",
    [
        pytest.param(1, id="builtin-int"),
        pytest.param(1.25, id="builtin-float"),
        pytest.param(np.int64(2), id="numpy-int"),
        pytest.param(np.float32(2.5), id="numpy-float"),
    ],
)
def test_real_scalar_vector_components_are_accepted(component: object) -> None:
    """Builtin and NumPy real scalar components remain accepted."""
    row = GoalForceMetricRow((component, 0.0), (0.0, 0.0))  # type: ignore[arg-type]
    assert row.predicted_force_xy == pytest.approx((float(component), 0.0))
