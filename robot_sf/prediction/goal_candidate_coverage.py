# ruff: noqa: DOC201

"""Oracle-only evaluation of frozen public goal-candidate sets.

This module is deliberately separate from candidate generation.  Its truth inputs are
permitted only after an actor candidate set has been frozen, and its output is a
coverage diagnostic—not posterior accuracy, calibration, or a claim about pedestrian
preferences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from robot_sf.prediction._contract_utils import require_finite, require_non_negative, require_text
from robot_sf.prediction.goal_intention import GoalCandidate, GoalCandidateRole, GoalCandidateSet

GOAL_CANDIDATE_COVERAGE_SCHEMA_VERSION = "goal_candidate_coverage.v1"

Point = tuple[float, float]


def _xy(value: Point | Sequence[float] | None, field_name: str) -> Point | None:
    """Validate an optional finite point."""

    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    return (
        require_finite(value[0], f"{field_name}[0]"),
        require_finite(value[1], f"{field_name}[1]"),
    )


def _unit(value: Point | Sequence[float] | None, field_name: str) -> Point | None:
    """Validate and normalize an optional direction."""

    point = _xy(value, field_name)
    if point is None:
        return None
    norm = math.hypot(*point)
    if norm <= 0.0:
        raise ValueError(f"{field_name} must be non-zero")
    return (point[0] / norm, point[1] / norm)


def _distance(left: Point, right: Point) -> float:
    """Return Euclidean distance."""

    return math.hypot(left[0] - right[0], left[1] - right[1])


def _angle(left: Point, right: Point) -> float:
    """Return the unsigned angle between unit directions."""

    return math.acos(max(-1.0, min(1.0, left[0] * right[0] + left[1] * right[1])))


@dataclass(frozen=True, slots=True)
class OracleGoalTruth:
    """Truth tuple accepted only by the post-generation coverage evaluator."""

    active_position: Point | None = None
    final_position: Point | None = None
    direction: Point | None = None
    route_signature: str | None = None
    observed_position_global: Point | None = None

    def __post_init__(self) -> None:
        """Validate oracle geometry while keeping it out of actor APIs."""

        object.__setattr__(self, "active_position", _xy(self.active_position, "active_position"))
        object.__setattr__(self, "final_position", _xy(self.final_position, "final_position"))
        object.__setattr__(self, "direction", _unit(self.direction, "direction"))
        object.__setattr__(
            self,
            "observed_position_global",
            _xy(self.observed_position_global, "observed_position_global"),
        )
        if self.route_signature is not None:
            object.__setattr__(
                self, "route_signature", require_text(self.route_signature, "route_signature")
            )
        if self.active_position is None and self.final_position is None and self.direction is None:
            raise ValueError("oracle truth requires active_position, final_position, or direction")


@dataclass(frozen=True, slots=True)
class GoalCandidateCoverage:
    """Coverage-only result with explicit separation from inference quality."""

    active_position_covered: bool
    final_position_covered: bool
    direction_covered: bool
    route_signature_covered: bool
    top_k_covered: bool
    unknown_needed: bool
    unknown_present: bool
    candidate_count: int
    top_k: int
    path_tangent_vs_direct_line_angle_rad: float | None
    claim_boundary: str = "candidate_coverage_only"
    schema_version: str = GOAL_CANDIDATE_COVERAGE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-safe coverage receipt."""

        return {
            "schema_version": self.schema_version,
            "claim_boundary": self.claim_boundary,
            "active_position_covered": self.active_position_covered,
            "final_position_covered": self.final_position_covered,
            "direction_covered": self.direction_covered,
            "route_signature_covered": self.route_signature_covered,
            "top_k_covered": self.top_k_covered,
            "unknown_needed": self.unknown_needed,
            "unknown_present": self.unknown_present,
            "candidate_count": self.candidate_count,
            "top_k": self.top_k,
            "path_tangent_vs_direct_line_angle_rad": self.path_tangent_vs_direct_line_angle_rad,
        }


def _point_candidates(
    candidate_set: GoalCandidateSet, role: GoalCandidateRole
) -> tuple[GoalCandidate, ...]:
    """Select point candidates for one semantic coverage level."""

    if role is GoalCandidateRole.ACTIVE_WAYPOINT:
        roles = {GoalCandidateRole.ACTIVE_WAYPOINT, GoalCandidateRole.BOTH}
    else:
        roles = {
            GoalCandidateRole.FINAL_DESTINATION,
            GoalCandidateRole.ROUTE_ENDPOINT,
            GoalCandidateRole.BOTH,
        }
    return tuple(
        candidate
        for candidate in candidate_set.candidates
        if candidate.role in roles and candidate.position is not None
    )


def _prior_rank(candidate: GoalCandidate) -> tuple[float, str]:
    """Rank candidates by public prior and stable ID only."""

    return (-(candidate.prior_weight if candidate.prior_weight is not None else 1.0), candidate.id)


def evaluate_goal_candidate_coverage(
    candidate_set: GoalCandidateSet,
    truth: OracleGoalTruth,
    *,
    distance_tolerance_m: float = 0.5,
    direction_tolerance_rad: float = math.pi / 12.0,
    top_k: int = 5,
) -> GoalCandidateCoverage:
    """Compare a frozen actor candidate set with oracle truth for coverage only.

    The evaluator does not alter candidates, priors, or actor observations.  A missing
    finite match is reported as ``unknown_needed``; it is not converted into a
    prediction-quality failure.
    """

    if type(candidate_set) is not GoalCandidateSet:
        raise TypeError("candidate_set must be a GoalCandidateSet")
    if type(truth) is not OracleGoalTruth:
        raise TypeError("truth must be an OracleGoalTruth")
    tolerance = require_non_negative(distance_tolerance_m, "distance_tolerance_m")
    angle_tolerance = require_non_negative(direction_tolerance_rad, "direction_tolerance_rad")
    if angle_tolerance > math.pi:
        raise ValueError("direction_tolerance_rad must be at most pi")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    active = _point_candidates(candidate_set, GoalCandidateRole.ACTIVE_WAYPOINT)
    final = _point_candidates(candidate_set, GoalCandidateRole.FINAL_DESTINATION)
    active_covered = truth.active_position is not None and any(
        _distance(candidate.position, truth.active_position) <= tolerance for candidate in active
    )
    final_covered = truth.final_position is not None and any(
        _distance(candidate.position, truth.final_position) <= tolerance for candidate in final
    )
    route_covered = truth.route_signature is not None and any(
        candidate.route_signature == truth.route_signature for candidate in final
    )

    direction_covered = False
    if truth.direction is not None:
        direction_candidates = tuple(
            candidate
            for candidate in candidate_set.candidates
            if candidate.direction is not None or candidate.path_tangent is not None
        )
        direction_covered = any(
            _angle(
                candidate.direction or candidate.path_tangent,  # type: ignore[arg-type]
                truth.direction,
            )
            <= angle_tolerance + (candidate.angular_support_rad or 0.0)
            for candidate in direction_candidates
        )
        if not direction_covered and truth.observed_position_global is not None:
            direction_covered = any(
                candidate.position is not None
                and _distance(candidate.position, truth.observed_position_global) > 0.0
                and _angle(
                    (
                        (candidate.position[0] - truth.observed_position_global[0])
                        / _distance(candidate.position, truth.observed_position_global),
                        (candidate.position[1] - truth.observed_position_global[1])
                        / _distance(candidate.position, truth.observed_position_global),
                    ),
                    truth.direction,
                )
                <= angle_tolerance
                for candidate in (*active, *final)
            )

    matched = active_covered or final_covered or route_covered or direction_covered
    unknown_needed = not matched
    unknown_present = any(
        candidate.role is GoalCandidateRole.UNKNOWN for candidate in candidate_set.candidates
    )
    ranked = sorted(
        (
            candidate
            for candidate in candidate_set.candidates
            if candidate.role is not GoalCandidateRole.UNKNOWN
            and candidate.availability.value == "available"
        ),
        key=_prior_rank,
    )
    top_k_candidates = ranked[:top_k]
    top_k_covered = any(
        candidate in (*active, *final)
        and (
            (
                truth.active_position is not None
                and candidate.position is not None
                and _distance(candidate.position, truth.active_position) <= tolerance
            )
            or (
                truth.final_position is not None
                and candidate.position is not None
                and _distance(candidate.position, truth.final_position) <= tolerance
            )
            or (
                truth.route_signature is not None
                and candidate.route_signature == truth.route_signature
            )
        )
        for candidate in top_k_candidates
    )

    tangent_error: float | None = None
    if truth.observed_position_global is not None and truth.final_position is not None:
        direct = (
            truth.final_position[0] - truth.observed_position_global[0],
            truth.final_position[1] - truth.observed_position_global[1],
        )
        direct_unit = _unit(direct, "direct_line")
        tangent_errors = [
            _angle(candidate.path_tangent, direct_unit)  # type: ignore[arg-type]
            for candidate in final
            if candidate.path_tangent is not None and direct_unit is not None
        ]
        tangent_error = min(tangent_errors) if tangent_errors else None

    return GoalCandidateCoverage(
        active_position_covered=active_covered,
        final_position_covered=final_covered,
        direction_covered=direction_covered,
        route_signature_covered=route_covered,
        top_k_covered=top_k_covered,
        unknown_needed=unknown_needed,
        unknown_present=unknown_present,
        candidate_count=len(candidate_set.candidates),
        top_k=top_k,
        path_tangent_vs_direct_line_angle_rad=tangent_error,
    )


__all__ = [
    "GOAL_CANDIDATE_COVERAGE_SCHEMA_VERSION",
    "GoalCandidateCoverage",
    "OracleGoalTruth",
    "evaluate_goal_candidate_coverage",
]
