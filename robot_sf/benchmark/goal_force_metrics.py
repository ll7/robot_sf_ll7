"""Pure continuous goal-force metrics for evaluator-side rows.

This module intentionally scores only paired vector rows.  It does not depend
on a goal posterior, candidate hierarchy, trace join, or simulator state.

Metric semantics are fixed as follows: ``vector_l2_mae`` is the mean, over
uncensored rows with available truth, of the Euclidean error ``||prediction -
truth||_2``; ``vector_l2_rmse`` is the root mean square of those same per-row
Euclidean errors.  ``GOAL_FORCE_METRICS_NORM_EPSILON`` is the force-norm
threshold for direction and relative-magnitude eligibility.  The former
``vector_mae``/``vector_rmse`` names were draft-only and are intentionally not
serialized as compatibility aliases: this contract is versioned as v2 before
any downstream consumer is admitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

Vector2 = tuple[float, float]

GOAL_FORCE_METRICS_SCHEMA_VERSION = "goal_force_metrics.v2"
GOAL_FORCE_METRICS_CLAIM_BOUNDARY = "pure_continuous_metric_contract"
GOAL_FORCE_METRICS_NORM_EPSILON = 1e-12
"""Minimum vector norm required for direction and relative-magnitude scores."""


def _vector(value: Sequence[float], field_name: str) -> Vector2:
    """Validate and normalize one finite two-dimensional vector.

    Returns:
        The validated vector as a tuple of floats.
    """

    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{field_name} must contain finite values")
    return result


def _norm(value: Vector2) -> float:
    return math.hypot(value[0], value[1])


@dataclass(frozen=True, slots=True)
class GoalForceMetricRow:
    """One paired prediction/truth row supplied by an evaluator or fixture."""

    predicted_force_xy: Vector2
    oracle_force_xy: Vector2 | None
    censored: bool = False

    def __post_init__(self) -> None:
        """Validate vector shape, finiteness, and censoring type."""

        object.__setattr__(
            self, "predicted_force_xy", _vector(self.predicted_force_xy, "predicted_force_xy")
        )
        if self.oracle_force_xy is not None:
            object.__setattr__(
                self, "oracle_force_xy", _vector(self.oracle_force_xy, "oracle_force_xy")
            )
        if type(self.censored) is not bool:
            raise TypeError("censored must be bool")


@dataclass(frozen=True, slots=True)
class GoalForceMetricSummary:
    """Deterministic aggregate with explicit per-metric denominators."""

    schema_version: str
    claim_boundary: str
    norm_epsilon: float
    row_count: int
    unavailable_count: int
    censored_count: int
    exact_vector_count: int
    direction_count: int
    direction_excluded_count: int
    magnitude_count: int
    relative_magnitude_count: int
    relative_magnitude_excluded_count: int
    vector_l2_mae: float | None
    vector_l2_rmse: float | None
    component_bias_xy: Vector2 | None
    angular_error_rad: float | None
    cosine_similarity: float | None
    magnitude_mae: float | None
    relative_magnitude_mae: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible summary."""

        return {
            "schema_version": self.schema_version,
            "claim_boundary": self.claim_boundary,
            "norm_epsilon": self.norm_epsilon,
            "row_count": self.row_count,
            "unavailable_count": self.unavailable_count,
            "censored_count": self.censored_count,
            "exact_vector_count": self.exact_vector_count,
            "direction_count": self.direction_count,
            "direction_excluded_count": self.direction_excluded_count,
            "magnitude_count": self.magnitude_count,
            "relative_magnitude_count": self.relative_magnitude_count,
            "relative_magnitude_excluded_count": self.relative_magnitude_excluded_count,
            "vector_l2_mae": self.vector_l2_mae,
            "vector_l2_rmse": self.vector_l2_rmse,
            "component_bias_xy": list(self.component_bias_xy)
            if self.component_bias_xy is not None
            else None,
            "angular_error_rad": self.angular_error_rad,
            "cosine_similarity": self.cosine_similarity,
            "magnitude_mae": self.magnitude_mae,
            "relative_magnitude_mae": self.relative_magnitude_mae,
        }


def evaluate_goal_force_rows(rows: Sequence[GoalForceMetricRow]) -> GoalForceMetricSummary:
    """Score paired force rows without mixing censored or unavailable truth.

    ``vector_l2_mae`` is the mean per-row Euclidean error and
    ``vector_l2_rmse`` is the root mean square of that same error.  Both use
    the uncensored, available-truth denominator.

    Censored rows contribute to direction metrics when both vectors have a
    norm strictly greater than :data:`GOAL_FORCE_METRICS_NORM_EPSILON`, but
    never to force-magnitude or vector-error metrics.  Rows with missing oracle
    truth contribute only to availability counts.  Available rows excluded
    from direction scoring by the norm threshold are counted separately, as
    are uncensored rows excluded from relative-magnitude scoring because their
    truth norm is too small.

    Returns:
        A summary with separate denominators and exclusion counts for exact,
        direction, magnitude, and relative-magnitude metrics.
    """

    normalized_rows = tuple(rows)
    if any(type(row) is not GoalForceMetricRow for row in normalized_rows):
        raise TypeError("rows must contain GoalForceMetricRow values")

    exact_errors: list[float] = []
    exact_squared_errors: list[float] = []
    biases: list[Vector2] = []
    angles: list[float] = []
    cosines: list[float] = []
    magnitudes: list[float] = []
    relative_magnitudes: list[float] = []
    unavailable_count = 0
    censored_count = 0
    direction_excluded_count = 0
    relative_magnitude_excluded_count = 0

    for row in normalized_rows:
        truth = row.oracle_force_xy
        if row.censored:
            censored_count += 1
        if truth is None:
            unavailable_count += 1
            continue
        dx = row.predicted_force_xy[0] - truth[0]
        dy = row.predicted_force_xy[1] - truth[1]
        predicted_norm = _norm(row.predicted_force_xy)
        truth_norm = _norm(truth)
        if not row.censored:
            error = math.hypot(dx, dy)
            exact_errors.append(error)
            exact_squared_errors.append(error * error)
            biases.append((dx, dy))
            magnitudes.append(abs(predicted_norm - truth_norm))
            if truth_norm > GOAL_FORCE_METRICS_NORM_EPSILON:
                relative_magnitudes.append(abs(predicted_norm - truth_norm) / truth_norm)
            else:
                relative_magnitude_excluded_count += 1
        if (
            predicted_norm > GOAL_FORCE_METRICS_NORM_EPSILON
            and truth_norm > GOAL_FORCE_METRICS_NORM_EPSILON
        ):
            cosine = row.predicted_force_xy[0] * truth[0] + row.predicted_force_xy[1] * truth[1]
            cosine /= predicted_norm * truth_norm
            cosines.append(max(-1.0, min(1.0, cosine)))
            angles.append(math.acos(max(-1.0, min(1.0, cosine))))
        else:
            direction_excluded_count += 1

    def mean(values: list[float]) -> float | None:
        return math.fsum(values) / len(values) if values else None

    component_bias = (
        (mean([value[0] for value in biases]), mean([value[1] for value in biases]))
        if biases
        else None
    )
    return GoalForceMetricSummary(
        schema_version=GOAL_FORCE_METRICS_SCHEMA_VERSION,
        claim_boundary=GOAL_FORCE_METRICS_CLAIM_BOUNDARY,
        norm_epsilon=GOAL_FORCE_METRICS_NORM_EPSILON,
        row_count=len(normalized_rows),
        unavailable_count=unavailable_count,
        censored_count=censored_count,
        exact_vector_count=len(exact_errors),
        direction_count=len(angles),
        direction_excluded_count=direction_excluded_count,
        magnitude_count=len(magnitudes),
        relative_magnitude_count=len(relative_magnitudes),
        relative_magnitude_excluded_count=relative_magnitude_excluded_count,
        vector_l2_mae=mean(exact_errors),
        vector_l2_rmse=math.sqrt(mean(exact_squared_errors)) if exact_squared_errors else None,
        component_bias_xy=component_bias,
        angular_error_rad=mean(angles),
        cosine_similarity=mean(cosines),
        magnitude_mae=mean(magnitudes),
        relative_magnitude_mae=mean(relative_magnitudes),
    )


__all__ = [
    "GOAL_FORCE_METRICS_CLAIM_BOUNDARY",
    "GOAL_FORCE_METRICS_NORM_EPSILON",
    "GOAL_FORCE_METRICS_SCHEMA_VERSION",
    "GoalForceMetricRow",
    "GoalForceMetricSummary",
    "evaluate_goal_force_rows",
]
