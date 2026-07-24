"""Diagnostic social-compliance metric emission for episode records."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SOCIAL_COMPLIANCE_SCHEMA_VERSION = "social-compliance-metric-contract.v1"
SOCIAL_COMPLIANCE_CLAIM_CLASS = "diagnostic_proxy"

_METRIC_DEFINITIONS = {
    "pedestrian_deviation_mean_m": (
        "pedestrian_deviation",
        "meters",
        "tracked_pedestrian_steps_with_baseline",
        "matched pedestrian reference trajectory is unavailable",
    ),
    "flow_disruption_delay_s": (
        "flow_disruption",
        "seconds",
        "pedestrians_with_reference_arrival",
        "matched pedestrian arrival baseline is unavailable",
    ),
    "comfort_exposure_person_s": (
        "comfort_exposure",
        "person_seconds",
        "pedestrian_steps",
        "pedestrian positions, timestep, or comfort radius are unavailable",
    ),
    "legibility_progress_deficit_m": (
        "legibility_progress",
        "meters",
        "robot_steps_before_terminal",
        "reference robot progress profile is unavailable",
    ),
    "distributional_inconvenience_p90_p50_gap": (
        "distributional_inconvenience",
        "seconds",
        "pedestrians_with_delay_samples",
        "per-pedestrian delay samples are unavailable or under-supported",
    ),
}


def _metric_row(
    metric_id: str,
    *,
    status: str,
    value: float | None = None,
    support_count: int = 0,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Build one contract-shaped metric row.

    Returns:
        A JSON-serializable metric row with explicit status and support.
    """
    family, units, denominator, default_reason = _METRIC_DEFINITIONS[metric_id]
    row: dict[str, Any] = {
        "id": metric_id,
        "family": family,
        "claim_class": SOCIAL_COMPLIANCE_CLAIM_CLASS,
        "units": units,
        "denominator": denominator,
        "status": status,
        "support_count": int(support_count),
    }
    if status == "available":
        row["value"] = float(value) if value is not None else None
    else:
        row["unavailable_reason"] = unavailable_reason or default_reason
    return row


def unavailable_social_compliance_block(*, no_pedestrians: bool = False) -> dict[str, Any]:
    """Return a complete block with explicit unavailable/not-applicable rows."""
    rows: dict[str, dict[str, Any]] = {}
    for metric_id in _METRIC_DEFINITIONS:
        if metric_id == "comfort_exposure_person_s" and no_pedestrians:
            rows[metric_id] = _metric_row(
                metric_id,
                status="not_applicable",
                unavailable_reason="episode contains no pedestrians",
            )
        else:
            rows[metric_id] = _metric_row(metric_id, status="unavailable")
    return {
        "schema_version": SOCIAL_COMPLIANCE_SCHEMA_VERSION,
        "claim_class": SOCIAL_COMPLIANCE_CLAIM_CLASS,
        "metrics": rows,
    }


def build_social_compliance_episode_block(
    data: Any,
    *,
    comfort_radius_m: float = 1.2,
) -> dict[str, Any]:
    """Emit the contract block, computing only signals supported by native episode data.

    Returns:
        A versioned social-compliance block with all five contract families represented.
    """
    peds_pos = getattr(data, "peds_pos", None)
    robot_pos = getattr(data, "robot_pos", None)
    dt = getattr(data, "dt", None)
    if not isinstance(peds_pos, np.ndarray) or peds_pos.ndim != 3:
        return unavailable_social_compliance_block()
    if not isinstance(robot_pos, np.ndarray) or robot_pos.ndim != 2:
        return unavailable_social_compliance_block()
    if peds_pos.shape[0] != robot_pos.shape[0] or peds_pos.shape[2] < 2:
        return unavailable_social_compliance_block()
    if peds_pos.shape[1] == 0:
        return unavailable_social_compliance_block(no_pedestrians=True)
    try:
        timestep = float(dt)
        radius = float(comfort_radius_m)
        robot_radius = float(getattr(data, "robot_radius", None))
        ped_radius = float(getattr(data, "ped_radius", None))
    except (TypeError, ValueError):
        return unavailable_social_compliance_block()
    if (
        not math.isfinite(timestep)
        or timestep <= 0.0
        or not math.isfinite(radius)
        or radius < 0.0
        or not math.isfinite(robot_radius)
        or robot_radius < 0.0
        or not math.isfinite(ped_radius)
        or ped_radius < 0.0
    ):
        return unavailable_social_compliance_block()

    positions = peds_pos[..., :2]
    robot = robot_pos[:, None, :2]
    if not np.isfinite(positions).all() or not np.isfinite(robot).all():
        return unavailable_social_compliance_block()
    distances = np.linalg.norm(positions - robot, axis=2)
    surface_clearance = distances - robot_radius - ped_radius
    exposed_steps = int(np.count_nonzero(surface_clearance <= radius))
    support_count = int(positions.shape[0] * positions.shape[1])
    row = _metric_row(
        "comfort_exposure_person_s",
        status="available" if support_count else "unavailable",
        value=exposed_steps * timestep,
        support_count=support_count,
        unavailable_reason="no finite pedestrian position samples",
    )
    block = unavailable_social_compliance_block()
    block["metrics"]["comfort_exposure_person_s"] = row
    block["parameters"] = {"comfort_radius_m": radius, "timestep_seconds": timestep}
    return block


__all__ = [
    "SOCIAL_COMPLIANCE_CLAIM_CLASS",
    "SOCIAL_COMPLIANCE_SCHEMA_VERSION",
    "build_social_compliance_episode_block",
    "unavailable_social_compliance_block",
]
