"""Diagnostic event predicates for public-requirement scenario proxies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

EVENT_SCHEMA_VERSION = "public-requirement-events.v1"
CLAIM_BOUNDARY = "authored scenario trigger diagnostics only"


def extract_public_requirement_contract(scenario: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the scenario's public-requirement metadata contract, if present."""
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    public_requirement = metadata.get("public_requirement")
    if not isinstance(public_requirement, Mapping):
        return None
    return dict(public_requirement)


def evaluate_public_requirement_events(
    *,
    scenario: Mapping[str, Any],
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    ped_positions: np.ndarray,
    dt: float,
) -> dict[str, Any]:
    """Evaluate issue #3977 public-requirement diagnostics without changing metrics.

    Returns:
        Versioned diagnostic event payload for the scenario contract.
    """
    contract = extract_public_requirement_contract(scenario)
    if contract is None:
        return _base_event(
            category=None,
            event_type=None,
            status="not_applicable",
            triggered=False,
        )

    event_contract = contract.get("event_contract")
    if not isinstance(event_contract, Mapping):
        return _base_event(
            category=_string_or_none(contract.get("category")),
            event_type=None,
            status="unavailable",
            triggered=False,
            reason="metadata.public_requirement.event_contract missing or not a mapping",
        )

    event_type = _string_or_none(event_contract.get("type"))
    category = _string_or_none(contract.get("category"))
    base = _base_event(
        category=category,
        event_type=event_type,
        status="available",
        triggered=False,
        speed_limit_m_s=_optional_float(
            event_contract.get("speed_limit_m_s", contract.get("speed_limit_m_s"))
        ),
    )

    match event_type:
        case "pedestrian_steps_in_front":
            return _evaluate_actor_at_conflict_point(
                base=base,
                scenario=scenario,
                ped_positions=ped_positions,
                robot_positions=robot_positions,
                contract=event_contract,
                actor_key="pedestrian_id",
                dt=dt,
            )
        case "turn_or_start_stop_near_pedestrian":
            return _evaluate_turn_or_start_stop_near_pedestrian(
                base=base,
                robot_positions=robot_positions,
                contract=event_contract,
                dt=dt,
            )
        case "sudden_obstacle_proxy":
            return _evaluate_actor_at_conflict_point(
                base=base,
                scenario=scenario,
                ped_positions=ped_positions,
                robot_positions=robot_positions,
                contract=event_contract,
                actor_key="actor_id",
                dt=dt,
            )
        case "speed_limit_monitor":
            return _evaluate_speed_limit(
                base=base,
                robot_velocities=robot_velocities,
                contract=event_contract,
                dt=dt,
            )
        case _:
            base["status"] = "unavailable"
            base["reason"] = f"unsupported public_requirement event type: {event_type!r}"
            return base


def _base_event(
    *,
    category: str | None,
    event_type: str | None,
    status: str,
    triggered: bool,
    speed_limit_m_s: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "category": category,
        "event_type": event_type,
        "status": status,
        "triggered": bool(triggered),
        "trigger_step": None,
        "trigger_time_s": None,
        "speed_limit_m_s": speed_limit_m_s,
        "speed_limit_violation_count": 0,
        "max_speed_m_s": None,
        "max_excess_m_s": None,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if reason:
        event["reason"] = reason
    return event


def _evaluate_actor_at_conflict_point(
    *,
    base: dict[str, Any],
    scenario: Mapping[str, Any],
    ped_positions: np.ndarray,
    robot_positions: np.ndarray,
    contract: Mapping[str, Any],
    actor_key: str,
    dt: float,
) -> dict[str, Any]:
    actor_id = _string_or_none(contract.get(actor_key))
    actor_idx = _single_pedestrian_index(scenario, actor_id)
    conflict_point = _xy_array(contract.get("conflict_point") or contract.get("hold_ref_point"))
    if actor_idx is None or conflict_point is None:
        base["status"] = "unavailable"
        base["reason"] = f"{actor_key} and conflict_point are required"
        return base
    if ped_positions.ndim != 3 or actor_idx >= ped_positions.shape[1]:
        base["status"] = "unavailable"
        base["reason"] = f"pedestrian index for {actor_id!r} unavailable in trajectory"
        return base

    actor_xy = ped_positions[:, actor_idx, :]
    distances = np.linalg.norm(actor_xy - conflict_point[np.newaxis, :], axis=1)
    trigger_radius = _positive_float(contract.get("trigger_radius_m"), default=0.75)
    candidate_steps = np.flatnonzero(distances <= trigger_radius)
    robot_radius = _optional_float(contract.get("robot_trigger_radius_m"))
    if robot_radius is not None:
        robot_distances = np.linalg.norm(
            robot_positions[: distances.shape[0]] - conflict_point[np.newaxis, :], axis=1
        )
        candidate_steps = np.asarray(
            [step for step in candidate_steps if robot_distances[step] <= robot_radius],
            dtype=int,
        )
    return _triggered_from_steps(base, candidate_steps, dt=dt)


def _evaluate_turn_or_start_stop_near_pedestrian(
    *,
    base: dict[str, Any],
    robot_positions: np.ndarray,
    contract: Mapping[str, Any],
    dt: float,
) -> dict[str, Any]:
    conflict_point = _xy_array(contract.get("conflict_point") or contract.get("turn_point"))
    if conflict_point is None:
        base["status"] = "unavailable"
        base["reason"] = "conflict_point or turn_point is required"
        return base
    distances = np.linalg.norm(robot_positions - conflict_point[np.newaxis, :], axis=1)
    trigger_radius = _positive_float(contract.get("trigger_radius_m"), default=1.8)
    return _triggered_from_steps(base, np.flatnonzero(distances <= trigger_radius), dt=dt)


def _evaluate_speed_limit(
    *,
    base: dict[str, Any],
    robot_velocities: np.ndarray,
    contract: Mapping[str, Any],
    dt: float,
) -> dict[str, Any]:
    speed_limit = _optional_float(base.get("speed_limit_m_s"))
    if speed_limit is None:
        base["status"] = "unavailable"
        base["reason"] = "speed_limit_m_s is required"
        return base
    margin = _positive_float(contract.get("violation_margin_m_s"), default=0.0, allow_zero=True)
    velocities = np.asarray(robot_velocities, dtype=float)
    if velocities.ndim != 2 or velocities.shape[1] < 2 or velocities.shape[0] == 0:
        base["max_speed_m_s"] = 0.0
        base["max_excess_m_s"] = 0.0
        return base
    speeds = np.linalg.norm(velocities[:, :2], axis=1)
    finite_speeds = speeds[np.isfinite(speeds)]
    if finite_speeds.size == 0:
        base["max_speed_m_s"] = 0.0
        base["max_excess_m_s"] = 0.0
        return base
    max_speed = float(np.max(finite_speeds))
    excess = finite_speeds - speed_limit
    violation_steps = np.flatnonzero(excess > margin)
    base["max_speed_m_s"] = max_speed
    base["max_excess_m_s"] = max(0.0, float(np.max(excess)))
    base["speed_limit_violation_count"] = int(violation_steps.size)
    base["triggered"] = bool(violation_steps.size)
    if violation_steps.size:
        trigger_step = int(violation_steps[0])
        base["trigger_step"] = trigger_step
        base["trigger_time_s"] = float((trigger_step + 1) * dt)
    return base


def _triggered_from_steps(event: dict[str, Any], steps: np.ndarray, *, dt: float) -> dict[str, Any]:
    event["triggered"] = bool(steps.size)
    if steps.size:
        trigger_step = int(steps[0])
        event["trigger_step"] = trigger_step
        event["trigger_time_s"] = float((trigger_step + 1) * dt)
    return event


def _single_pedestrian_index(scenario: Mapping[str, Any], pedestrian_id: str | None) -> int | None:
    if not pedestrian_id:
        return None
    single_pedestrians = scenario.get("single_pedestrians")
    if not isinstance(single_pedestrians, list):
        return None
    for idx, pedestrian in enumerate(single_pedestrians):
        if isinstance(pedestrian, Mapping) and pedestrian.get("id") == pedestrian_id:
            return idx
    return None


def _xy_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _positive_float(value: Any, *, default: float, allow_zero: bool = False) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        return default
    if allow_zero and parsed >= 0.0:
        return parsed
    return parsed if parsed > 0.0 else default


__all__ = [
    "CLAIM_BOUNDARY",
    "EVENT_SCHEMA_VERSION",
    "evaluate_public_requirement_events",
    "extract_public_requirement_contract",
]
