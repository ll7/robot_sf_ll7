"""Compatibility tests for benchmark scenario failure-cause imports."""

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
from robot_sf.scenario_certification.failure_cause import (
    FAILURE_CAUSE_SCHEMA,
    INDETERMINATE,
    FamilyDiagnostics,
    classify_failure_cause,
)


def test_benchmark_imports_use_canonical_schema_and_verdicts() -> None:
    """Benchmark compatibility names must not create a divergent schema."""

    assert SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION == FAILURE_CAUSE_SCHEMA
    assert VERDICT_INFEASIBLE_ROUTE == "infeasible_route"
    assert VERDICT_VEHICLE_INFEASIBLE == "vehicle_infeasible"
    assert VERDICT_TIME_LIMITED == "time_limited"
    assert VERDICT_PLANNER_LIMITED == "planner_limited"
    assert VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK == "dynamic_blocking_or_deadlock"
    assert VERDICT_INDETERMINATE == INDETERMINATE
    assert ScenarioFailureDiagnostics is FamilyDiagnostics


def test_benchmark_classifier_matches_canonical_owner() -> None:
    """Benchmark import surface delegates classification to the canonical owner."""

    diagnostics = ScenarioFailureDiagnostics(
        route_feasible=True,
        actor_free_solved=True,
        extended_time_solved=False,
        oracle_solved=False,
    )

    assert classify_scenario_failure_cause(diagnostics) == classify_failure_cause(diagnostics)


def test_benchmark_mapping_adapter_accepts_scenario_family_alias() -> None:
    """Benchmark summaries can still use the historic ``scenario_family`` key."""

    verdict = classify_scenario_failure_cause(
        {
            "scenario_family": "bottleneck",
            "route_feasible": True,
            "actor_free_solved": True,
            "extended_time_solved": None,
            "oracle_solved": False,
        }
    )

    assert verdict["schema_version"] == SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION
    assert verdict["cause"] == VERDICT_INDETERMINATE
    assert verdict["inputs"]["oracle_solved"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"family": "bottleneck", "route_feasible": "yes"},
        {"family": "bottleneck", "any_planner_succeeded": 1},
        {"family": 3484},
    ],
)
def test_mapping_adapter_rejects_ambiguous_payloads(payload: dict[str, object]) -> None:
    """Inputs fail closed instead of coercing ambiguous diagnostic values."""

    with pytest.raises(ValueError):
        diagnostics_from_mapping(payload)
