"""Compatibility exports for scenario failure-cause classification.

The canonical classifier lives in
``robot_sf.scenario_certification.failure_cause``.  Keep this module as a thin
benchmark-facing import surface so benchmark callers cannot drift into a
second ``scenario_failure_cause.v1`` semantic contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robot_sf.scenario_certification.failure_cause import (
    DYNAMIC_BLOCKING_OR_DEADLOCK as VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK,
)
from robot_sf.scenario_certification.failure_cause import (
    FAILURE_CAUSE_SCHEMA as SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION,
)
from robot_sf.scenario_certification.failure_cause import (
    INDETERMINATE as VERDICT_INDETERMINATE,
)
from robot_sf.scenario_certification.failure_cause import (
    INFEASIBLE_ROUTE as VERDICT_INFEASIBLE_ROUTE,
)
from robot_sf.scenario_certification.failure_cause import (
    PLANNER_LIMITED as VERDICT_PLANNER_LIMITED,
)
from robot_sf.scenario_certification.failure_cause import (
    TIME_LIMITED as VERDICT_TIME_LIMITED,
)
from robot_sf.scenario_certification.failure_cause import (
    VEHICLE_INFEASIBLE as VERDICT_VEHICLE_INFEASIBLE,
)
from robot_sf.scenario_certification.failure_cause import (
    FamilyDiagnostics as ScenarioFailureDiagnostics,
)
from robot_sf.scenario_certification.failure_cause import classify_failure_cause

ScenarioFailureCause = dict[str, Any]


def diagnostics_from_mapping(payload: Mapping[str, Any]) -> ScenarioFailureDiagnostics:
    """Normalize benchmark-style mapping payloads to canonical diagnostics.

    Returns:
        Canonical failure-cause diagnostics.
    """

    family = payload.get("family", payload.get("scenario_family"))
    if family is not None and not isinstance(family, str):
        raise ValueError("scenario failure diagnostics family must be a string")
    if "scenario_family" in payload and "family" not in payload:
        payload = {**payload, "family": family}
    return ScenarioFailureDiagnostics(
        any_planner_succeeded=_required_bool(payload, "any_planner_succeeded", default=False),
        route_feasible=_optional_bool(payload, "route_feasible"),
        actor_free_solved=_optional_bool(payload, "actor_free_solved"),
        extended_time_solved=_optional_bool(payload, "extended_time_solved"),
        oracle_solved=_optional_bool(payload, "oracle_solved"),
    )


def classify_scenario_failure_cause(
    diagnostics: ScenarioFailureDiagnostics | Mapping[str, Any],
) -> ScenarioFailureCause:
    """Classify benchmark diagnostics with the canonical owner semantics.

    Returns:
        Canonical failure-cause verdict payload.
    """

    normalized = (
        diagnostics_from_mapping(diagnostics) if isinstance(diagnostics, Mapping) else diagnostics
    )
    return classify_failure_cause(normalized)


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    """Read optional boolean field without coercing strings or integers.

    Returns:
        Boolean value or ``None`` when absent.
    """

    if key not in payload or payload[key] is None:
        return None
    if isinstance(payload[key], bool):
        return payload[key]
    raise ValueError(f"{key} must be boolean or null")


def _required_bool(payload: Mapping[str, Any], key: str, *, default: bool) -> bool:
    """Read required/defaulted boolean field without lossy coercion.

    Returns:
        Boolean field value.
    """

    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be boolean")


__all__ = [
    "SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION",
    "VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK",
    "VERDICT_INDETERMINATE",
    "VERDICT_INFEASIBLE_ROUTE",
    "VERDICT_PLANNER_LIMITED",
    "VERDICT_TIME_LIMITED",
    "VERDICT_VEHICLE_INFEASIBLE",
    "ScenarioFailureCause",
    "ScenarioFailureDiagnostics",
    "classify_scenario_failure_cause",
    "diagnostics_from_mapping",
]
