"""Pure continuous goal-force metrics for evaluator-side rows.

This module intentionally scores only paired vector rows.  It does not depend
on a goal posterior, candidate hierarchy, trace join, or simulator state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

Vector2 = tuple[float, float]

GOAL_FORCE_METRICS_SCHEMA_VERSION = "goal_force_metrics.v1"
GOAL_FORCE_METRICS_CLAIM_BOUNDARY = "pure_continuous_metric_contract"
_EPSILON = 1e-12


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
    """Deterministic aggregate with separate exact and direction denominators."""

    schema_version: str
    claim_boundary: str
    row_count: int
    unavailable_count: int
    censored_count: int
    exact_vector_count: int
    direction_count: int
    magnitude_count: int
    vector_mae: float | None
    vector_rmse: float | None
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
            "row_count": self.row_count,
            "unavailable_count": self.unavailable_count,
            "censored_count": self.censored_count,
            "exact_vector_count": self.exact_vector_count,
            "direction_count": self.direction_count,
            "magnitude_count": self.magnitude_count,
            "vector_mae": self.vector_mae,
            "vector_rmse": self.vector_rmse,
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

    Censored rows contribute to direction metrics when both vectors have a
    non-zero direction, but never to force-magnitude or vector-error metrics.
    Rows with missing oracle truth contribute only to availability counts.

    Returns:
        A summary with separate denominators for exact, direction, and
        magnitude metrics.
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

    for row in normalized_rows:
        truth = row.oracle_force_xy
        if truth is None:
            unavailable_count += 1
            continue
        if row.censored:
            censored_count += 1
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
            if truth_norm > _EPSILON:
                relative_magnitudes.append(abs(predicted_norm - truth_norm) / truth_norm)
        if predicted_norm > _EPSILON and truth_norm > _EPSILON:
            cosine = row.predicted_force_xy[0] * truth[0] + row.predicted_force_xy[1] * truth[1]
            cosine /= predicted_norm * truth_norm
            cosines.append(max(-1.0, min(1.0, cosine)))
            angles.append(math.acos(max(-1.0, min(1.0, cosine))))

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    component_bias = (
        (mean([value[0] for value in biases]), mean([value[1] for value in biases]))
        if biases
        else None
    )
    return GoalForceMetricSummary(
        schema_version=GOAL_FORCE_METRICS_SCHEMA_VERSION,
        claim_boundary=GOAL_FORCE_METRICS_CLAIM_BOUNDARY,
        row_count=len(normalized_rows),
        unavailable_count=unavailable_count,
        censored_count=censored_count,
        exact_vector_count=len(exact_errors),
        direction_count=len(angles),
        magnitude_count=len(magnitudes),
        vector_mae=mean(exact_errors),
        vector_rmse=math.sqrt(mean(exact_squared_errors)) if exact_squared_errors else None,
        component_bias_xy=component_bias,
        angular_error_rad=mean(angles),
        cosine_similarity=mean(cosines),
        magnitude_mae=mean(magnitudes),
        relative_magnitude_mae=mean(relative_magnitudes),
    )


__all__ = [
    "GOAL_FORCE_METRICS_CLAIM_BOUNDARY",
    "GOAL_FORCE_METRICS_SCHEMA_VERSION",
    "GoalForceMetricRow",
    "GoalForceMetricSummary",
    "evaluate_goal_force_rows",
]
