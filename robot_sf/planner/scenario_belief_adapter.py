"""Planner-facing ScenarioBelief uncertainty projection helpers.

These helpers are diagnostic interface smoke, not benchmark evidence. They bridge
the uncertainty-preserving ScenarioBelief report into one planner-compatible observation shape
without changing legacy policy projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from robot_sf.representation import ScenarioBelief

SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION = "scenario-belief-planner-projection.v1"
SUPPORTED_UNCERTAINTY_PLANNER_KEYS = frozenset({"stream_gap"})


@dataclass(frozen=True)
class ScenarioBeliefPlannerProjection:
    """ScenarioBelief observation plus explicit planner uncertainty compatibility status."""

    observation: dict[str, Any]
    compatibility: dict[str, Any]


def _pedestrian_count(observation: dict[str, Any]) -> int | None:
    """Return the active pedestrian count from a SOCNAV_STRUCT-like observation."""
    pedestrians = observation.get("pedestrians")
    if not isinstance(pedestrians, dict):
        return None
    try:
        raw_count = np.asarray(pedestrians.get("count"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if raw_count.size == 0 or not np.isfinite(raw_count[0]):
        return None
    return max(0, int(raw_count[0]))


def _compatibility_payload(
    *,
    planner_key: str,
    status: str,
    reason: str | None = None,
    consumed_agent_count: int = 0,
) -> dict[str, Any]:
    """Return a deterministic planner-compatibility diagnostic payload."""
    payload: dict[str, Any] = {
        "schema_version": SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION,
        "planner_key": planner_key,
        "status": status,
        "uncertainty_consumed": status == "compatible",
        "consumed_agent_count": int(consumed_agent_count),
        "claim_boundary": "diagnostic_interface_smoke",
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def project_scenario_belief_for_planner(
    belief: ScenarioBelief,
    *,
    planner_key: str,
) -> ScenarioBeliefPlannerProjection:
    """Project ScenarioBelief into one planner observation with uncertainty compatibility status.

    Only ``stream_gap`` currently consumes the uncertainty sidecar under
    ``observation["pedestrians"]["uncertainty"]``. Unsupported planner keys fail closed by
    returning the legacy ``to_socnav_struct()`` observation without the sidecar and by recording
    an explicit unsupported status.

    Returns:
        ScenarioBeliefPlannerProjection: Observation plus diagnostic compatibility metadata.
    """
    observation = belief.to_socnav_struct()
    pedestrians = observation.get("pedestrians")
    if not isinstance(pedestrians, dict):
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="malformed_legacy_observation",
        )
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    if planner_key not in SUPPORTED_UNCERTAINTY_PLANNER_KEYS:
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="unsupported_uncertainty_planner",
        )
        pedestrians["uncertainty_compatibility"] = compatibility
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    count = _pedestrian_count(observation)
    if count is None:
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="malformed_pedestrian_count",
        )
        pedestrians["uncertainty_compatibility"] = compatibility
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    report = belief.to_uncertainty_report()
    rows = report.get("agents")
    if not isinstance(rows, list) or len(rows) < count:
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="malformed_uncertainty_report",
        )
        pedestrians["uncertainty_compatibility"] = compatibility
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    uncertainty_rows = [dict(row) for row in rows[:count]]
    pedestrians["uncertainty"] = uncertainty_rows
    compatibility = _compatibility_payload(
        planner_key=planner_key,
        status="compatible",
        consumed_agent_count=len(uncertainty_rows),
    )
    pedestrians["uncertainty_compatibility"] = compatibility
    return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)


__all__ = [
    "SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION",
    "SUPPORTED_UNCERTAINTY_PLANNER_KEYS",
    "ScenarioBeliefPlannerProjection",
    "project_scenario_belief_for_planner",
]
