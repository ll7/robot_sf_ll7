"""Tests for the oracle-only candidate coverage boundary."""

from __future__ import annotations

import pytest

from robot_sf.prediction.goal_candidate_coverage import (
    OracleGoalTruth,
    evaluate_goal_candidate_coverage,
)
from robot_sf.prediction.goal_candidate_provider import (
    CandidatePathMode,
    GoalCandidateSource,
    PublicGoalCandidateRecord,
    generate_goal_candidates,
)
from robot_sf.prediction.goal_intention import GoalCandidateRole


def test_coverage_is_separate_from_generation_and_reports_path_tangent_error() -> None:
    """Coverage can compare a frozen set with truth without changing its bytes."""

    result = generate_goal_candidates(
        (
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,
                source_id="around-wall",
                position=(10.0, 0.0),
                route_signature="left",
                path_points=((0.0, 0.0), (0.0, 2.0), (10.0, 0.0)),
                path_mode=CandidatePathMode.PLANNER_PATH,
            ),
        ),
        observed_position_global=(0.0, 0.0),
    )
    before = result.candidate_set_digest
    coverage = evaluate_goal_candidate_coverage(
        result.candidate_set,
        OracleGoalTruth(
            active_position=(0.0, 2.0),
            final_position=(10.0, 0.0),
            direction=(0.0, 1.0),
            route_signature="left",
            observed_position_global=(0.0, 0.0),
        ),
    )

    assert coverage.active_position_covered is True
    assert coverage.final_position_covered is True
    assert coverage.direction_covered is True
    assert coverage.route_signature_covered is True
    assert coverage.top_k_covered is True
    assert coverage.unknown_needed is False
    assert coverage.path_tangent_vs_direct_line_angle_rad == pytest.approx(1.5707963267948966)
    assert coverage.claim_boundary == "candidate_coverage_only"
    assert result.candidate_set_digest == before


def test_missing_truth_match_reports_unknown_needed_when_unknown_is_present() -> None:
    """Withheld goals remain an unknown coverage case, not an oracle fallback."""

    result = generate_goal_candidates(
        (
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.MAP_DESTINATION_ZONE,
                source_id="east",
                position=(10.0, 0.0),
            ),
        )
    )
    coverage = evaluate_goal_candidate_coverage(
        result.candidate_set,
        OracleGoalTruth(final_position=(-10.0, 0.0)),
    )

    assert coverage.final_position_covered is False
    assert coverage.unknown_needed is True
    assert coverage.unknown_present is True


def test_coverage_rejects_invalid_inputs() -> None:
    """The evaluator keeps its truth boundary typed and bounded."""

    result = generate_goal_candidates(())
    with pytest.raises(ValueError, match="requires"):
        OracleGoalTruth()
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_goal_candidate_coverage(
            result.candidate_set, OracleGoalTruth(direction=(1.0, 0.0)), top_k=0
        )
    with pytest.raises(ValueError, match="at most pi"):
        evaluate_goal_candidate_coverage(
            result.candidate_set,
            OracleGoalTruth(direction=(1.0, 0.0)),
            direction_tolerance_rad=4.0,
        )


def test_coverage_distinguishes_active_and_final_roles() -> None:
    """A final endpoint match does not silently count as active-waypoint coverage."""

    result = generate_goal_candidates(
        (
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.MAP_DESTINATION_ZONE,
                source_id="end",
                role=GoalCandidateRole.FINAL_DESTINATION,
                position=(10.0, 0.0),
            ),
        )
    )
    coverage = evaluate_goal_candidate_coverage(
        result.candidate_set,
        OracleGoalTruth(active_position=(2.0, 0.0), final_position=(10.0, 0.0)),
    )

    assert coverage.active_position_covered is False
    assert coverage.final_position_covered is True
