"""Tests for the universally-failing scenario failure-cause classifier (issue #3484)."""

from __future__ import annotations

from robot_sf.scenario_certification.failure_cause import (
    DYNAMIC_BLOCKING_OR_DEADLOCK,
    FAILURE_CAUSE_SCHEMA,
    INDETERMINATE,
    INFEASIBLE_ROUTE,
    NOT_UNIVERSALLY_FAILING,
    PLANNER_LIMITED,
    TIME_LIMITED,
    VEHICLE_INFEASIBLE,
    FamilyDiagnostics,
    classify_failure_cause,
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
