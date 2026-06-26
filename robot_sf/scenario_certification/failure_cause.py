"""Reclassify universally-failing scenario families by failure cause (issue #3484).

Some scenario families (bottleneck, cross-trap, head-on-corridor) are passed by *no*
benchmark planner. A row no method can pass is useful for diagnosis but weak for
comparative ranking, so before labeling such rows "universally hard" the cause must be
disambiguated: an infeasible route, a vehicle-level infeasibility, a binding time limit,
a genuine planner-capability gap, or reciprocal dynamic blocking.

This module provides the **pure decision layer** that consumes the per-family diagnostic
outcomes and emits a reproducible, versioned verdict. The sim-integrated diagnostic
*runners* (geometric clearance certification, oracle/scripted trajectory, actor-free run,
extended-time, difficulty ramp) are deliberate follow-ups; this layer is side-effect free
and changes no runtime/benchmark behavior.

Verdicts are versioned modeling choices, labeled diagnostic until validated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FAILURE_CAUSE_SCHEMA = "scenario_failure_cause.v1"
FEASIBILITY_DIAGNOSTIC_MANIFEST_SCHEMA = "scenario_feasibility_diagnostic_manifest.v1"

DIAGNOSTIC_STATUS_NOT_RUN = "not_run"
DIAGNOSTIC_STATUS_NEEDS_EVIDENCE = "needs_evidence"

REQUIRED_DIAGNOSTIC_LANES = (
    "route_clearance",
    "actor_free_run",
    "extended_time_run",
    "oracle_or_scripted_controller",
)


@dataclass(frozen=True, slots=True)
class FamilyDiagnostics:
    """Diagnostic outcomes for one universally-failing scenario family.

    Each boolean may be ``None`` when that diagnostic was not run; the classifier then
    degrades to ``indeterminate`` rather than guessing.

    Attributes:
        any_planner_succeeded: Whether any benchmark planner passed the family.
        route_feasible: Whether a collision-free path exists for the vehicle footprint.
        actor_free_solved: Whether the vehicle reaches goal with no pedestrians present.
        extended_time_solved: Whether the family is solved under an extended time limit.
        oracle_solved: Whether a privileged/scripted controller solves it within limits.
    """

    any_planner_succeeded: bool = False
    route_feasible: bool | None = None
    actor_free_solved: bool | None = None
    extended_time_solved: bool | None = None
    oracle_solved: bool | None = None


@dataclass(frozen=True, slots=True)
class FeasibilityDiagnosticRow:
    """Dry-run diagnostic planning row for one scenario in a candidate family.

    Rows created by this model are deliberately not benchmark evidence. They record
    which scenario-family entries need follow-up feasibility diagnostics and keep the
    failure-cause classifier in its fail-closed ``indeterminate`` state until real
    evidence is supplied.
    """

    scenario_id: str
    family_id: str
    source: str
    diagnostic_status: str = DIAGNOSTIC_STATUS_NOT_RUN
    evidence_status: str = DIAGNOSTIC_STATUS_NEEDS_EVIDENCE
    required_diagnostics: tuple[str, ...] = REQUIRED_DIAGNOSTIC_LANES
    notes: str = "Dry-run planning row only; no feasibility diagnostic was executed."


# Stable verdict vocabulary.
NOT_UNIVERSALLY_FAILING = "not_universally_failing"
INFEASIBLE_ROUTE = "infeasible_route"
VEHICLE_INFEASIBLE = "vehicle_infeasible"
TIME_LIMITED = "time_limited"
PLANNER_LIMITED = "planner_limited"
DYNAMIC_BLOCKING_OR_DEADLOCK = "dynamic_blocking_or_deadlock"
INDETERMINATE = "indeterminate"


def classify_failure_cause(diagnostics: FamilyDiagnostics) -> dict[str, Any]:
    """Map per-family diagnostics to a versioned failure-cause verdict.

    Decision order (first match wins):

    1. a planner did pass → ``not_universally_failing`` (premise violated);
    2. ``route_feasible is False`` → ``infeasible_route`` (scenario defect);
    3. ``actor_free_solved is False`` → ``vehicle_infeasible`` (footprint/adapter/limit
       defect at the vehicle level, before pedestrians);
    4. ``extended_time_solved is True`` → ``time_limited`` (the time limit binds);
    5. ``oracle_solved is True`` → ``planner_limited`` (valid scenario; the gap is planner
       capability — the only cause that is fair for comparative ranking);
    6. ``oracle_solved is False`` with a feasible route and actor-free success →
       ``dynamic_blocking_or_deadlock`` (reciprocal deadlock / unrealistic blocking);
    7. otherwise → ``indeterminate`` (insufficient diagnostics).

    Returns:
        dict[str, Any]: Versioned verdict with ``cause``, ``comparable_for_ranking``,
        ``evidence_complete``, and a human-readable ``rationale``.
    """
    cause, rationale = _decide(diagnostics)
    evidence_complete = None not in (
        diagnostics.route_feasible,
        diagnostics.actor_free_solved,
        diagnostics.extended_time_solved,
        diagnostics.oracle_solved,
    )
    return {
        "schema_version": FAILURE_CAUSE_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "cause": cause,
        # Only a genuine planner-capability gap makes a universally-failing row a fair
        # comparative-ranking signal; every other cause is a scenario/time/feasibility
        # defect and must be excluded from ranking claims.
        "comparable_for_ranking": cause == PLANNER_LIMITED,
        "evidence_complete": evidence_complete,
        "rationale": rationale,
        "inputs": {
            "any_planner_succeeded": diagnostics.any_planner_succeeded,
            "route_feasible": diagnostics.route_feasible,
            "actor_free_solved": diagnostics.actor_free_solved,
            "extended_time_solved": diagnostics.extended_time_solved,
            "oracle_solved": diagnostics.oracle_solved,
        },
    }


def build_feasibility_diagnostic_manifest(
    scenarios: Iterable[Mapping[str, Any]],
    *,
    source: str,
    family_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a dry-run feasibility diagnostic manifest for candidate families.

    The manifest prepares issue #3484 classification work without running a campaign
    or inferring scenario validity. Every row remains ``not_run`` and
    ``needs_evidence`` until route-clearance, actor-free, extended-time, and oracle
    diagnostics are populated by follow-up tooling.

    Returns:
        Versioned dry-run manifest with one row per selected scenario.
    """

    selected_families = {family.casefold() for family in family_ids or ()}
    rows: list[dict[str, Any]] = []
    skipped = 0

    for scenario in scenarios:
        family_id = _scenario_family_id(scenario)
        if selected_families and family_id.casefold() not in selected_families:
            skipped += 1
            continue
        row = FeasibilityDiagnosticRow(
            scenario_id=_scenario_id(scenario),
            family_id=family_id,
            source=source,
        )
        rows.append(feasibility_diagnostic_row_to_dict(row))

    return {
        "schema_version": FEASIBILITY_DIAGNOSTIC_MANIFEST_SCHEMA,
        "issue": "3484",
        "evidence_kind": "diagnostic_planning",
        "claim_boundary": "diagnostic_only_not_benchmark_evidence",
        "source": source,
        "family_filter": list(family_ids or []),
        "row_count": len(rows),
        "skipped_count": skipped,
        "required_diagnostics": list(REQUIRED_DIAGNOSTIC_LANES),
        "rows": rows,
    }


def feasibility_diagnostic_row_to_dict(row: FeasibilityDiagnosticRow) -> dict[str, Any]:
    """Convert a diagnostic planning row to JSON-safe primitives.

    Returns:
        Dictionary representation including a fail-closed indeterminate verdict.
    """

    verdict = classify_failure_cause(FamilyDiagnostics())
    return {
        "scenario_id": row.scenario_id,
        "family_id": row.family_id,
        "source": row.source,
        "diagnostic_status": row.diagnostic_status,
        "evidence_status": row.evidence_status,
        "required_diagnostics": list(row.required_diagnostics),
        "failure_cause_verdict": verdict,
        "notes": row.notes,
    }


def _scenario_family_id(scenario: Mapping[str, Any]) -> str:
    """Return the best available scenario-family identifier."""

    metadata = scenario.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    for value in (
        metadata_map.get("scenario_family"),
        metadata_map.get("family"),
        metadata_map.get("archetype"),
        scenario.get("scenario_family"),
        scenario.get("family"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Return a stable scenario identifier for manifest rows."""

    for key in ("scenario_id", "name", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _decide(d: FamilyDiagnostics) -> tuple[str, str]:
    """Return the (cause, rationale) for a diagnostics bundle."""
    if d.any_planner_succeeded:
        return (
            NOT_UNIVERSALLY_FAILING,
            "At least one planner passed; the family is not universally failing.",
        )
    if d.route_feasible is False:
        return (
            INFEASIBLE_ROUTE,
            "No collision-free path exists for the vehicle footprint; scenario defect.",
        )
    if d.actor_free_solved is False:
        return (
            VEHICLE_INFEASIBLE,
            "The vehicle cannot reach goal even with no pedestrians; vehicle-level defect.",
        )
    if d.extended_time_solved is True:
        return (
            TIME_LIMITED,
            "Solved under an extended time limit; the time limit is the binding constraint.",
        )
    if d.oracle_solved is True:
        return (
            PLANNER_LIMITED,
            "A privileged controller solves it within limits; the gap is planner capability.",
        )
    if d.oracle_solved is False and d.route_feasible is True and d.actor_free_solved is True:
        return (
            DYNAMIC_BLOCKING_OR_DEADLOCK,
            "Route and actor-free runs succeed but even an oracle fails with reactive "
            "pedestrians; reciprocal deadlock or unrealistic blocking.",
        )
    return (
        INDETERMINATE,
        "Insufficient diagnostics to disambiguate the failure cause.",
    )


__all__ = [
    "DIAGNOSTIC_STATUS_NEEDS_EVIDENCE",
    "DIAGNOSTIC_STATUS_NOT_RUN",
    "DYNAMIC_BLOCKING_OR_DEADLOCK",
    "FAILURE_CAUSE_SCHEMA",
    "FEASIBILITY_DIAGNOSTIC_MANIFEST_SCHEMA",
    "INDETERMINATE",
    "INFEASIBLE_ROUTE",
    "NOT_UNIVERSALLY_FAILING",
    "PLANNER_LIMITED",
    "REQUIRED_DIAGNOSTIC_LANES",
    "TIME_LIMITED",
    "VEHICLE_INFEASIBLE",
    "FamilyDiagnostics",
    "FeasibilityDiagnosticRow",
    "build_feasibility_diagnostic_manifest",
    "classify_failure_cause",
    "feasibility_diagnostic_row_to_dict",
]
