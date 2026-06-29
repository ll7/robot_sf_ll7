"""Tests for issue #3484 scenario-failure cause classification."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.scenario_failure_cause import (
    SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION,
    VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK,
    VERDICT_INDETERMINATE,
    VERDICT_INFEASIBLE_ROUTE,
    VERDICT_PLANNER_LIMITED,
    VERDICT_TIME_LIMITED,
    VERDICT_VEHICLE_INFEASIBLE,
    ScenarioFailureDiagnostics,
    classify_scenario_failure_cause,
    diagnostics_from_mapping,
)


def test_route_clearance_failure_classifies_infeasible_route() -> None:
    """Route-clearance failure is not treated as planner-ranking evidence."""

    verdict = classify_scenario_failure_cause(
        ScenarioFailureDiagnostics(family="bottleneck", route_feasible=False)
    )

    assert verdict.schema_version == SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION
    assert verdict.verdict == VERDICT_INFEASIBLE_ROUTE
    assert verdict.comparable_for_ranking is False
    assert verdict.missing_diagnostics == ()


def test_actor_free_failure_classifies_vehicle_infeasible() -> None:
    """A route-feasible but actor-free-unsolved family is vehicle infeasible."""

    verdict = classify_scenario_failure_cause(
        ScenarioFailureDiagnostics(
            family="head_on_corridor",
            route_feasible=True,
            actor_free_solved=False,
        )
    )

    assert verdict.verdict == VERDICT_VEHICLE_INFEASIBLE
    assert verdict.comparable_for_ranking is False


def test_extended_time_success_classifies_time_limited() -> None:
    """Extended-time success separates horizon limits from planner limits."""

    verdict = classify_scenario_failure_cause(
        ScenarioFailureDiagnostics(
            family="cross_trap",
            route_feasible=True,
            actor_free_solved=True,
            extended_time_solved=True,
            oracle_solved=True,
        )
    )

    assert verdict.verdict == VERDICT_TIME_LIMITED
    assert verdict.comparable_for_ranking is False


def test_oracle_success_without_planner_success_classifies_planner_limited() -> None:
    """Only planner-limited rows are comparable for planner ranking."""

    verdict = classify_scenario_failure_cause(
        ScenarioFailureDiagnostics(
            family="cross_trap",
            route_feasible=True,
            actor_free_solved=True,
            extended_time_solved=False,
            oracle_solved=True,
            any_planner_succeeded=False,
        )
    )

    assert verdict.verdict == VERDICT_PLANNER_LIMITED
    assert verdict.comparable_for_ranking is True


def test_oracle_failure_after_route_and_actor_free_checks_classifies_dynamic_blocking() -> None:
    """Actor-related failures remain diagnostic rather than ranking evidence."""

    verdict = classify_scenario_failure_cause(
        ScenarioFailureDiagnostics(
            family="bottleneck",
            route_feasible=True,
            actor_free_solved=True,
            extended_time_solved=False,
            oracle_solved=False,
        )
    )

    assert verdict.verdict == VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK
    assert verdict.comparable_for_ranking is False


@pytest.mark.parametrize(
    ("diagnostics", "missing"),
    [
        (ScenarioFailureDiagnostics(family="bottleneck"), ("route_feasible",)),
        (
            ScenarioFailureDiagnostics(family="bottleneck", route_feasible=True),
            ("actor_free_solved",),
        ),
        (
            ScenarioFailureDiagnostics(
                family="bottleneck",
                route_feasible=True,
                actor_free_solved=True,
            ),
            ("oracle_solved", "extended_time_solved"),
        ),
        (
            ScenarioFailureDiagnostics(
                family="bottleneck",
                route_feasible=True,
                actor_free_solved=True,
                extended_time_solved=None,
                oracle_solved=False,
            ),
            ("extended_time_solved",),
        ),
    ],
)
def test_missing_required_diagnostics_fail_closed(
    diagnostics: ScenarioFailureDiagnostics,
    missing: tuple[str, ...],
) -> None:
    """Missing upstream diagnostics produce an indeterminate verdict."""

    verdict = classify_scenario_failure_cause(diagnostics)

    assert verdict.verdict == VERDICT_INDETERMINATE
    assert verdict.comparable_for_ranking is False
    assert verdict.missing_diagnostics == missing


def test_mapping_input_accepts_scenario_family_alias_and_serializes() -> None:
    """Benchmark summaries can feed mapping payloads directly."""

    verdict = classify_scenario_failure_cause(
        {
            "scenario_family": " bottleneck ",
            "route_feasible": True,
            "actor_free_solved": True,
            "extended_time_solved": False,
            "oracle_solved": True,
            "evidence_refs": ["output/diagnostics/summary.json"],
            "notes": ["diagnostic-only"],
        }
    )
    payload = verdict.as_dict()

    assert verdict.family == "bottleneck"
    assert payload["schema_version"] == SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION
    assert payload["diagnostics"]["evidence_refs"] == ["output/diagnostics/summary.json"]
    assert payload["diagnostics"]["notes"] == ["diagnostic-only"]


@pytest.mark.parametrize(
    "payload",
    [
        {"family": "bottleneck", "route_feasible": "yes"},
        {"family": "bottleneck", "any_planner_succeeded": 1},
        {"family": "bottleneck", "evidence_refs": [" "]},
        {"family": "bottleneck", "notes": [1]},
        {"route_feasible": True},
    ],
)
def test_mapping_input_rejects_ambiguous_payloads(payload: dict[str, object]) -> None:
    """Inputs fail closed instead of coercing ambiguous diagnostic values."""

    with pytest.raises(ValueError):
        diagnostics_from_mapping(payload)
