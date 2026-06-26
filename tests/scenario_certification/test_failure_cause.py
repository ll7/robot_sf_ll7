"""Tests for the universally-failing scenario failure-cause classifier (issue #3484)."""

from __future__ import annotations

import pytest

from robot_sf.scenario_certification.failure_cause import (
    DYNAMIC_BLOCKING_OR_DEADLOCK,
    FAILURE_CAUSE_SCHEMA,
    FEASIBILITY_DIAGNOSTIC_EVIDENCE_SCHEMA,
    INDETERMINATE,
    INFEASIBLE_ROUTE,
    NOT_UNIVERSALLY_FAILING,
    PLANNER_LIMITED,
    TIME_LIMITED,
    VEHICLE_INFEASIBLE,
    DiagnosticLaneEvidence,
    FamilyDiagnostics,
    build_feasibility_diagnostic_evidence_report,
    classify_failure_cause,
    diagnostic_lane_evidence_to_dict,
    family_diagnostics_from_lane_evidence,
)


def test_not_universally_failing_when_a_planner_passed() -> None:
    """If any planner passed, the premise is violated and ranking is unaffected."""
    verdict = classify_failure_cause(FamilyDiagnostics(any_planner_succeeded=True))

    assert verdict["cause"] == NOT_UNIVERSALLY_FAILING
    assert verdict["comparable_for_ranking"] is False


def test_infeasible_route_takes_precedence() -> None:
    """An infeasible route is a scenario defect and wins over later checks."""
    verdict = classify_failure_cause(FamilyDiagnostics(route_feasible=False, oracle_solved=True))

    assert verdict["cause"] == INFEASIBLE_ROUTE
    assert verdict["comparable_for_ranking"] is False


def test_vehicle_infeasible_when_actor_free_run_fails() -> None:
    """Failure with no pedestrians is a vehicle-level defect, not a planner gap."""
    verdict = classify_failure_cause(
        FamilyDiagnostics(route_feasible=True, actor_free_solved=False)
    )

    assert verdict["cause"] == VEHICLE_INFEASIBLE


def test_time_limited_when_extended_time_solves() -> None:
    """Solvability under extended time means the time limit is the binding constraint."""
    verdict = classify_failure_cause(
        FamilyDiagnostics(route_feasible=True, actor_free_solved=True, extended_time_solved=True)
    )

    assert verdict["cause"] == TIME_LIMITED


def test_planner_limited_is_the_only_rankable_cause() -> None:
    """An oracle-solvable family with no planner success is a fair planner comparison."""
    verdict = classify_failure_cause(
        FamilyDiagnostics(
            route_feasible=True,
            actor_free_solved=True,
            extended_time_solved=False,
            oracle_solved=True,
        )
    )

    assert verdict["cause"] == PLANNER_LIMITED
    assert verdict["comparable_for_ranking"] is True
    assert verdict["evidence_complete"] is True


def test_dynamic_blocking_when_oracle_also_fails() -> None:
    """Route + actor-free succeed but the oracle fails ⇒ reciprocal deadlock/blocking."""
    verdict = classify_failure_cause(
        FamilyDiagnostics(
            route_feasible=True,
            actor_free_solved=True,
            extended_time_solved=False,
            oracle_solved=False,
        )
    )

    assert verdict["cause"] == DYNAMIC_BLOCKING_OR_DEADLOCK
    assert verdict["comparable_for_ranking"] is False


def test_indeterminate_when_diagnostics_missing() -> None:
    """With no diagnostics run, the verdict must be indeterminate, not a guess."""
    verdict = classify_failure_cause(FamilyDiagnostics())

    assert verdict["cause"] == INDETERMINATE
    assert verdict["evidence_complete"] is False


def test_verdict_is_schema_tagged_and_echoes_inputs() -> None:
    """The verdict must be versioned, diagnostic-labeled, and echo its inputs."""
    diagnostics = FamilyDiagnostics(route_feasible=False)
    verdict = classify_failure_cause(diagnostics)

    assert verdict["schema_version"] == FAILURE_CAUSE_SCHEMA
    assert verdict["evidence_kind"] == "diagnostic_proxy"
    assert verdict["inputs"]["route_feasible"] is False
    assert isinstance(verdict["rationale"], str) and verdict["rationale"]


def test_diagnostic_lane_evidence_row_normalizes_oracle_trajectory_alias() -> None:
    """Oracle-trajectory observations serialize without running any simulator work."""
    row = diagnostic_lane_evidence_to_dict(
        DiagnosticLaneEvidence(
            family_id="head_on_corridor",
            lane="oracle_trajectory",
            passed=True,
            source="synthetic-fixture",
            scenario_id="head_on_corridor_tight",
            evidence_ref="fixture://oracle-success",
        )
    )

    assert row["schema_version"] == FEASIBILITY_DIAGNOSTIC_EVIDENCE_SCHEMA
    assert row["lane"] == "oracle_or_scripted_controller"
    assert row["status"] == "passed"
    assert row["passed"] is True


def test_family_diagnostics_aggregate_route_actor_time_and_oracle_lanes() -> None:
    """Synthetic lane observations feed the existing failure-cause classifier."""
    diagnostics = family_diagnostics_from_lane_evidence(
        [
            DiagnosticLaneEvidence("bottleneck", "route_clearance", True, "fixture"),
            DiagnosticLaneEvidence("bottleneck", "actor_free_run", True, "fixture"),
            DiagnosticLaneEvidence("bottleneck", "extended_time_run", False, "fixture"),
            DiagnosticLaneEvidence("bottleneck", "oracle_trajectory", True, "fixture"),
        ]
    )

    verdict = classify_failure_cause(diagnostics)

    assert verdict["cause"] == PLANNER_LIMITED
    assert verdict["evidence_complete"] is True
    assert verdict["comparable_for_ranking"] is True


def test_evidence_report_rejects_conflicting_lane_observations() -> None:
    """Conflicting lane inputs fail closed instead of producing an arbitrary verdict."""
    with pytest.raises(ValueError, match="Conflicting feasibility diagnostic evidence"):
        build_feasibility_diagnostic_evidence_report(
            [
                DiagnosticLaneEvidence("cross_trap", "route_clearance", True, "fixture-a"),
                DiagnosticLaneEvidence("cross_trap", "route-clearance", False, "fixture-b"),
            ],
            source="synthetic-fixture",
        )


def test_evidence_report_keeps_claim_boundary_diagnostic_only() -> None:
    """The report schema must not present offline lane rows as benchmark evidence."""
    report = build_feasibility_diagnostic_evidence_report(
        [
            DiagnosticLaneEvidence("cross_trap", "route_clearance", False, "fixture"),
            DiagnosticLaneEvidence("bottleneck", "actor-free-run", None, "fixture"),
        ],
        source="synthetic-fixture",
    )

    assert report["schema_version"] == FEASIBILITY_DIAGNOSTIC_EVIDENCE_SCHEMA
    assert report["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
    assert report["row_count"] == 2
    assert report["rows"][1]["status"] == "not_run"
    assert report["family_verdicts"][0]["family_id"] == "bottleneck"
    assert report["family_verdicts"][0]["failure_cause_verdict"]["cause"] == INDETERMINATE
    assert report["family_verdicts"][1]["family_id"] == "cross_trap"
    assert report["family_verdicts"][1]["failure_cause_verdict"]["cause"] == INFEASIBLE_ROUTE
