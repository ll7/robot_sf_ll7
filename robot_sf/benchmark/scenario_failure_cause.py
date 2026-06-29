"""Classify universally failing scenario-family diagnostics.

The classifier is intentionally pure: simulator and ledger runners can produce
the diagnostic booleans later, while this module fixes the versioned verdict
contract those runners should consume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION = "scenario_failure_cause.v1"

VERDICT_INFEASIBLE_ROUTE = "infeasible_route"
VERDICT_VEHICLE_INFEASIBLE = "vehicle_infeasible"
VERDICT_TIME_LIMITED = "time_limited"
VERDICT_PLANNER_LIMITED = "planner_limited"
VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK = "dynamic_blocking_or_deadlock"
VERDICT_INDETERMINATE = "indeterminate"

ScenarioFailureVerdict = Literal[
    "infeasible_route",
    "vehicle_infeasible",
    "time_limited",
    "planner_limited",
    "dynamic_blocking_or_deadlock",
    "indeterminate",
]


@dataclass(frozen=True, slots=True)
class ScenarioFailureDiagnostics:
    """Diagnostic outcomes for one universally failing scenario family."""

    family: str
    route_feasible: bool | None = None
    actor_free_solved: bool | None = None
    extended_time_solved: bool | None = None
    oracle_solved: bool | None = None
    any_planner_succeeded: bool = False
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ScenarioFailureCause:
    """Versioned failure-cause verdict for one scenario family."""

    family: str
    verdict: ScenarioFailureVerdict
    comparable_for_ranking: bool
    rationale: str
    missing_diagnostics: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: ScenarioFailureDiagnostics | None = None
    schema_version: str = SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly verdict payload."""

        payload = asdict(self)
        payload["missing_diagnostics"] = list(self.missing_diagnostics)
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict) and self.diagnostics is not None:
            diagnostics["evidence_refs"] = list(self.diagnostics.evidence_refs)
            diagnostics["notes"] = list(self.diagnostics.notes)
        if diagnostics is None:
            payload.pop("diagnostics")
        return payload


def diagnostics_from_mapping(payload: Mapping[str, Any]) -> ScenarioFailureDiagnostics:
    """Normalize a mapping into typed scenario-failure diagnostics.

    ``scenario_family`` is accepted as an alias for ``family`` because benchmark
    summaries commonly use that field name.

    Returns:
        Typed diagnostic outcomes.
    """

    family = payload.get("family", payload.get("scenario_family"))
    if not isinstance(family, str) or not family.strip():
        raise ValueError("scenario failure diagnostics require non-empty family")

    return ScenarioFailureDiagnostics(
        family=family.strip(),
        route_feasible=_optional_bool(payload, "route_feasible"),
        actor_free_solved=_optional_bool(payload, "actor_free_solved"),
        extended_time_solved=_optional_bool(payload, "extended_time_solved"),
        oracle_solved=_optional_bool(payload, "oracle_solved"),
        any_planner_succeeded=_required_bool(payload, "any_planner_succeeded", default=False),
        evidence_refs=_string_tuple(payload.get("evidence_refs", ()), field_name="evidence_refs"),
        notes=_string_tuple(payload.get("notes", ()), field_name="notes"),
    )


def classify_scenario_failure_cause(
    diagnostics: ScenarioFailureDiagnostics | Mapping[str, Any],
) -> ScenarioFailureCause:
    """Classify one scenario family from route, actor-free, time, and oracle diagnostics.

    Returns:
        Versioned failure-cause verdict.
    """

    normalized = (
        diagnostics_from_mapping(diagnostics) if isinstance(diagnostics, Mapping) else diagnostics
    )
    if not isinstance(normalized, ScenarioFailureDiagnostics):
        raise TypeError("diagnostics must be ScenarioFailureDiagnostics or mapping")
    if not isinstance(normalized.family, str) or not normalized.family.strip():
        raise ValueError("scenario failure diagnostics require non-empty family")
    if normalized.family != normalized.family.strip():
        normalized = replace(normalized, family=normalized.family.strip())

    verdict, rationale, missing = _classify_verdict(normalized)
    return ScenarioFailureCause(
        family=normalized.family,
        verdict=verdict,
        comparable_for_ranking=verdict == VERDICT_PLANNER_LIMITED,
        rationale=rationale,
        missing_diagnostics=tuple(missing),
        diagnostics=normalized,
    )


def _classify_verdict(
    diagnostics: ScenarioFailureDiagnostics,
) -> tuple[ScenarioFailureVerdict, str, list[str]]:
    """Return deterministic verdict, rationale, and missing diagnostic names.

    Returns:
        Verdict string, human-readable rationale, and missing diagnostic names.
    """

    if diagnostics.any_planner_succeeded:
        return (
            VERDICT_PLANNER_LIMITED,
            "At least one evaluated planner succeeded, so the row is not universally failing.",
            [],
        )

    if diagnostics.route_feasible is None:
        return _indeterminate("route clearance has not been certified", ["route_feasible"])
    if diagnostics.route_feasible is False:
        return (
            VERDICT_INFEASIBLE_ROUTE,
            "Route-clearance diagnostic found no collision-free vehicle-footprint path.",
            [],
        )

    if diagnostics.actor_free_solved is None:
        return _indeterminate(
            "actor-free vehicle feasibility has not been checked",
            ["actor_free_solved"],
        )
    if diagnostics.actor_free_solved is False:
        return (
            VERDICT_VEHICLE_INFEASIBLE,
            "Vehicle could not reach the goal even with pedestrians removed.",
            [],
        )

    if diagnostics.extended_time_solved is True:
        return (
            VERDICT_TIME_LIMITED,
            "The scenario solved only when the time horizon was extended.",
            [],
        )

    if diagnostics.oracle_solved is None:
        missing = ["oracle_solved"]
        if diagnostics.extended_time_solved is None:
            missing.append("extended_time_solved")
        return _indeterminate(
            "oracle trajectory diagnostic is required after route and actor-free checks pass",
            missing,
        )

    if diagnostics.oracle_solved:
        return (
            VERDICT_PLANNER_LIMITED,
            "Privileged/oracle trajectory solved the family while evaluated planners did not.",
            [],
        )

    return (
        VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK,
        "Route and actor-free checks pass, but privileged execution does not solve with actors.",
        [],
    )


def _indeterminate(
    reason: str, missing: list[str]
) -> tuple[ScenarioFailureVerdict, str, list[str]]:
    """Return a fail-closed indeterminate verdict.

    Returns:
        Indeterminate verdict tuple.
    """

    return VERDICT_INDETERMINATE, reason, missing


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    """Read an optional boolean field without coercing strings or integers.

    Returns:
        Boolean value or ``None`` when absent.
    """

    if key not in payload or payload[key] is None:
        return None
    if isinstance(payload[key], bool):
        return payload[key]
    raise ValueError(f"{key} must be boolean or null")


def _required_bool(payload: Mapping[str, Any], key: str, *, default: bool) -> bool:
    """Read a required/defaulted boolean field without lossy coercion.

    Returns:
        Boolean value.
    """

    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be boolean")


def _string_tuple(raw: object, *, field_name: str) -> tuple[str, ...]:
    """Normalize optional string sequences for provenance fields.

    Returns:
        Tuple of non-empty strings.
    """

    if raw is None:
        return ()
    if isinstance(raw, str):
        values: Sequence[object] = (raw,)
    elif isinstance(raw, Sequence):
        values = raw
    else:
        raise ValueError(f"{field_name} must be a string sequence")

    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"{field_name} entries must be strings")

    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} must not contain empty strings")
    return normalized
