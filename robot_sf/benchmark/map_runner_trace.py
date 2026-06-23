"""Trace and diagnostic helpers for benchmark map-runner episodes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


ROLLOVER_STABILITY_METADATA_KEY = "rollover_stability"


def _scenario_id(scenario: dict[str, Any]) -> str:
    """Resolve a scenario identifier from common manifest fields.

    Returns:
        str: Scenario id string, or ``"unknown"``.
    """
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def _first_float(value: Any, default: float = 0.0) -> float:
    """Return the first finite numeric value from scalar-or-sequence inputs."""

    if isinstance(value, int | float | np.integer | np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
        return _first_float(value[0], default=default)
    return default


def _metadata_bool(value: Any, *, default: bool = False) -> bool:
    """Return a conservative boolean for JSON/YAML metadata flags."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        return default
    if isinstance(value, int | np.integer):
        return bool(value)
    return default


def _observation_heading(obs: Any, *, default: float = 0.0) -> float:
    """Extract robot heading from structured or flat observations.

    Returns:
        Heading in radians, or the provided default when unavailable.
    """

    if isinstance(obs, dict):
        robot = obs.get("robot")
        if isinstance(robot, dict) and "heading" in robot:
            return _first_float(robot.get("heading"), default=default)
        if "robot_heading" in obs:
            return _first_float(obs.get("robot_heading"), default=default)
    return default


def _trace_pedestrians(
    positions: np.ndarray,
    previous_positions: np.ndarray | None,
    dt_seconds: float,
    intent_metadata: list[dict[str, Any] | None] | None = None,
    vru_metadata: list[dict[str, Any] | None] | None = None,
    robot_position: np.ndarray | None = None,
    robot_velocity: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build trace-export pedestrian frames from simulator position buffers.

    Returns:
        Renderer-neutral pedestrian frame entries.
    """

    if positions.size == 0:
        return []
    pedestrians: list[dict[str, Any]] = []
    metadata_count = max(len(intent_metadata or []), len(vru_metadata or []))
    single_offset = (
        max(0, int(np.asarray(positions).shape[0]) - metadata_count) if metadata_count else None
    )
    for ped_idx, ped_pos in enumerate(np.asarray(positions, dtype=float)):
        if (
            previous_positions is not None
            and previous_positions.shape == positions.shape
            and dt_seconds > 0.0
        ):
            velocity = (ped_pos - previous_positions[ped_idx]) / dt_seconds
        else:
            velocity = np.zeros(2, dtype=float)
        frame = {
            "id": int(ped_idx),
            "position": [float(ped_pos[0]), float(ped_pos[1])],
            "velocity": [float(velocity[0]), float(velocity[1])],
        }
        if single_offset is not None and ped_idx >= single_offset:
            single_idx = ped_idx - single_offset
            if 0 <= single_idx < len(intent_metadata or []):
                metadata = (intent_metadata or [])[single_idx]
                if metadata is not None:
                    frame.update(_intent_trace_payload(metadata, velocity))
            if 0 <= single_idx < len(vru_metadata or []):
                metadata = (vru_metadata or [])[single_idx]
                if metadata is not None:
                    frame.update(
                        _vru_trace_payload(
                            metadata,
                            ped_pos=ped_pos,
                            velocity=velocity,
                            robot_position=robot_position,
                            robot_velocity=robot_velocity,
                        )
                    )
        pedestrians.append(frame)
    return pedestrians


def _single_pedestrian_intent_metadata(scenario: dict[str, Any]) -> list[dict[str, Any] | None]:
    """Return authored intent metadata aligned with scenario single pedestrians."""
    scenario_metadata = (
        scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    )
    scenario_intent = (
        scenario_metadata.get("intent_conditioned_behavior")
        if isinstance(scenario_metadata.get("intent_conditioned_behavior"), dict)
        else {}
    )
    scenario_signal_state = (
        scenario_metadata.get("signal_state")
        if isinstance(scenario_metadata.get("signal_state"), dict)
        else None
    )
    single_peds = (
        scenario.get("single_pedestrians")
        if isinstance(scenario.get("single_pedestrians"), list)
        else []
    )
    result: list[dict[str, Any] | None] = []
    has_intent_metadata = False
    for idx, ped in enumerate(single_peds):
        if not isinstance(ped, dict):
            result.append(None)
            continue
        ped_metadata = ped.get("metadata") if isinstance(ped.get("metadata"), dict) else {}
        intent = (
            ped_metadata.get("intent_conditioned_behavior")
            if isinstance(ped_metadata.get("intent_conditioned_behavior"), dict)
            else {}
        )
        if not intent and not scenario_intent:
            result.append(None)
            continue
        wait_at = ped.get("wait_at") if isinstance(ped.get("wait_at"), list) else []
        trajectory = ped.get("trajectory") if isinstance(ped.get("trajectory"), list) else []
        phases = intent.get("intent_phases")
        if not isinstance(phases, list) or not phases:
            phases = ["waiting", "crossing"] if wait_at and trajectory else ["authored_motion"]
        label = str(
            intent.get("intent_label")
            or ("waiting_then_crossing" if wait_at and trajectory else "authored_single_pedestrian")
        )
        wait_intervals = [
            float(rule.get("wait_s"))
            for rule in wait_at
            if isinstance(rule, dict) and rule.get("wait_s") is not None
        ]
        has_intent_metadata = True
        result.append(
            {
                "single_index": idx,
                "pedestrian_id": str(ped.get("id") or f"single_{idx}"),
                "intent_label": label,
                "intent_phases": [str(phase) for phase in phases],
                "intent_source": str(intent.get("intent_source") or "authored_scenario_metadata"),
                "claim_boundary": str(
                    intent.get("claim_boundary")
                    or "Authored scenario metadata only; not data-grounded human intent evidence."
                ),
                "behavior_parameters": {
                    "trajectory_waypoint_count": len(trajectory),
                    "wait_at": wait_at,
                    "wait_interval_s": wait_intervals,
                    "start_delay_s": float(ped.get("start_delay_s", 0.0) or 0.0),
                    "speed_m_s": (
                        float(ped["speed_m_s"]) if ped.get("speed_m_s") is not None else None
                    ),
                    "role": ped.get("role"),
                    "role_target_id": ped.get("role_target_id"),
                },
                **({"signal_state": scenario_signal_state} if scenario_signal_state else {}),
            }
        )
    return result if has_intent_metadata else []


def _single_pedestrian_vru_metadata(scenario: dict[str, Any]) -> list[dict[str, Any] | None]:
    """Return authored fast-VRU metadata aligned with scenario single pedestrians."""
    single_peds = (
        scenario.get("single_pedestrians")
        if isinstance(scenario.get("single_pedestrians"), list)
        else []
    )
    result: list[dict[str, Any] | None] = []
    has_vru_metadata = False
    for idx, ped in enumerate(single_peds):
        if not isinstance(ped, dict):
            result.append(None)
            continue
        ped_metadata = ped.get("metadata") if isinstance(ped.get("metadata"), dict) else {}
        payload_key = "cyclist_like_vru"
        vru = ped_metadata.get("cyclist_like_vru")
        if not isinstance(vru, dict):
            fast_bicycle = ped_metadata.get("fast_bicycle_actor")
            if isinstance(fast_bicycle, dict):
                payload_key = "fast_bicycle_actor"
                vru = fast_bicycle
            else:
                vru = {}
        if not vru:
            result.append(None)
            continue
        speed_m_s = _metadata_float(vru, "speed_m_s", default=ped.get("speed_m_s"))
        acceleration_m_s2 = _metadata_float(vru, "acceleration_m_s2", default=0.0)
        actor_radius_m = _metadata_float(vru, "actor_radius_m", default=0.35)
        robot_radius_m = _metadata_float(vru, "robot_radius_m", default=0.3)
        has_vru_metadata = True
        result.append(
            {
                "single_index": idx,
                "pedestrian_id": str(ped.get("id") or f"single_{idx}"),
                "actor_type": str(
                    vru.get("actor_type")
                    or ("bicycle" if payload_key == "fast_bicycle_actor" else "cyclist_like_vru")
                ),
                "diagnostic_payload_key": payload_key,
                "speed_m_s": speed_m_s,
                "acceleration_m_s2": acceleration_m_s2,
                "actor_radius_m": actor_radius_m,
                "robot_radius_m": robot_radius_m,
                "interaction_role": str(vru.get("interaction_role") or "fast_moving_vru"),
                "interaction_class": str(
                    vru.get("interaction_class") or vru.get("interaction_role") or "fast_moving_vru"
                ),
                "diagnostic_metric_subset": [
                    str(item)
                    for item in (
                        vru.get("diagnostic_metric_subset")
                        if isinstance(vru.get("diagnostic_metric_subset"), list)
                        else [
                            "time_to_conflict_zone_s",
                            "clearance_m",
                            "pass_overtake_state",
                        ]
                    )
                ],
                "claim_boundary": str(
                    vru.get("claim_boundary")
                    or "Authored fast-VRU proxy metadata only; not cyclist realism or "
                    "planner-ranking benchmark evidence."
                ),
            }
        )
    return result if has_vru_metadata else []


def _metadata_float(mapping: Mapping[str, Any], key: str, *, default: Any) -> float:
    """Return a finite metadata float or a finite default."""
    value = mapping.get(key, default)
    try:
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    except (TypeError, ValueError):
        pass
    try:
        fallback = float(default) if default is not None else 0.0
    except (TypeError, ValueError):
        fallback = 0.0
    return fallback if math.isfinite(fallback) else 0.0


def _intent_trace_payload(metadata: dict[str, Any], velocity: np.ndarray) -> dict[str, Any]:
    """Build optional per-pedestrian authored-intent trace fields.

    Returns:
        dict[str, Any]: JSON-serializable intent fields for one trace pedestrian.
    """
    phases = (
        metadata.get("intent_phases") if isinstance(metadata.get("intent_phases"), list) else []
    )
    speed = float(np.linalg.norm(velocity))
    if "waiting" in phases and speed <= 1e-6:
        phase = "waiting"
    elif "crossing" in phases:
        phase = "crossing"
    elif phases:
        phase = str(phases[0])
    else:
        phase = "authored_motion"
    payload = {
        "pedestrian_id": metadata["pedestrian_id"],
        "intent_label": metadata["intent_label"],
        "intent_phase": phase,
        "intent_source": metadata["intent_source"],
        "claim_boundary": metadata["claim_boundary"],
        "behavior_parameters": metadata["behavior_parameters"],
    }
    proxy_payload = _signal_state_proxy_wrapper(
        metadata.get("signal_state"),
        phase,
        metadata["intent_label"],
        metadata["intent_source"],
    )
    if proxy_payload is not None:
        payload["signal_state"] = proxy_payload
    return payload


def _signal_state_trace_payload(signal_state: Any, intent_phase: str) -> dict[str, Any] | None:
    """Return proxy signal-state trace metadata for the current authored intent phase."""
    if not isinstance(signal_state, dict):
        return None
    phase_timeline = (
        signal_state.get("phase_timeline")
        if isinstance(signal_state.get("phase_timeline"), list)
        else []
    )
    matching_phase = next(
        (
            phase
            for phase in phase_timeline
            if isinstance(phase, dict) and phase.get("intent_phase") == intent_phase
        ),
        None,
    )
    if matching_phase is None:
        matching_phase = next(
            (phase for phase in phase_timeline if isinstance(phase, dict)),
            {},
        )
    return {
        "schema_version": str(signal_state.get("schema_version") or "signal-state-proxy.v1"),
        "status": str(signal_state.get("status") or "proxy_diagnostic_only"),
        "signal_id": str(signal_state.get("signal_id") or "unknown_signal"),
        "conflict_zone_id": str(signal_state.get("conflict_zone_id") or "unknown_conflict_zone"),
        "phase": str(matching_phase.get("phase") or "unknown"),
        "intent_phase": intent_phase,
        "robot_right_of_way": _metadata_bool(matching_phase.get("robot_right_of_way", False)),
        "pedestrian_right_of_way": _metadata_bool(
            matching_phase.get("pedestrian_right_of_way", False)
        ),
        "legality_state": str(matching_phase.get("legality_state") or "unknown"),
        "planner_observable": _metadata_bool(signal_state.get("planner_observable", False)),
        "observation_mode": str(signal_state.get("observation_mode") or "trace_metadata_only"),
        "benchmark_evidence": _metadata_bool(signal_state.get("benchmark_evidence", False)),
        "claim_boundary": str(
            signal_state.get("claim_boundary")
            or "Proxy signal-state metadata only; not benchmark evidence."
        ),
    }


_PROXY_TRACE_FIELDS_PRESENT = [
    "signal_phase",
    "pedestrian_intent",
    "robot_stop_or_yield_expectation",
    "claim_boundary",
    "trace_fields_present",
    "signal_state",
]

_SIGNAL_STATE_OBSERVABLE_SCHEMA = "signal-state-observable.v1"
_SIGNAL_STATE_OBSERVABLE_STATUS = "planner_observable_signal_state"
_SIGNAL_STATE_OBSERVABLE_MODE = "planner_observable"
_SIGNAL_STATE_OBSERVABLE_FIELDS = [
    "signal_id",
    "conflict_zone_id",
    "phase",
    "phase_elapsed_s",
    "phase_remaining_s",
    "robot_right_of_way",
    "pedestrian_right_of_way",
    "legality_state",
]
_SIGNAL_STATE_RECORDED_ONLY_FIELDS = [
    "schema_version",
    "status",
    "signal_id",
    "conflict_zone_id",
    "phase",
    "intent_phase",
    "robot_right_of_way",
    "pedestrian_right_of_way",
    "legality_state",
    "planner_observable",
    "observation_mode",
    "benchmark_evidence",
    "claim_boundary",
]


def _signal_state_promotion_contract(signal_state: Any) -> dict[str, Any]:
    """Classify signal-state metadata for benchmark promotion decisions.

    Returns:
        dict[str, Any]: Contract state plus planner-consumed and recorded-only fields.
    """
    if not isinstance(signal_state, dict):
        return {
            "contract_state": "unavailable",
            "planner_consumed_fields": [],
            "recorded_only_fields": [],
            "promotion_required_fields": list(_SIGNAL_STATE_OBSERVABLE_FIELDS),
            "fail_closed_reason": "signal_state_metadata_absent",
            "benchmark_evidence": False,
        }

    schema_version = str(signal_state.get("schema_version") or "")
    status = str(signal_state.get("status") or "")
    observation_mode = str(signal_state.get("observation_mode") or "")
    planner_observable = _metadata_bool(signal_state.get("planner_observable", False))
    benchmark_evidence = _metadata_bool(signal_state.get("benchmark_evidence", False))
    is_observable = (
        schema_version == _SIGNAL_STATE_OBSERVABLE_SCHEMA
        and status == _SIGNAL_STATE_OBSERVABLE_STATUS
        and observation_mode == _SIGNAL_STATE_OBSERVABLE_MODE
        and planner_observable
        and benchmark_evidence
    )
    if is_observable:
        return {
            "contract_state": "planner_observable",
            "planner_consumed_fields": list(_SIGNAL_STATE_OBSERVABLE_FIELDS),
            "recorded_only_fields": [],
            "promotion_required_fields": [],
            "fail_closed_reason": "",
            "benchmark_evidence": True,
        }

    return {
        "contract_state": "proxy_diagnostic",
        "planner_consumed_fields": [],
        "recorded_only_fields": list(_SIGNAL_STATE_RECORDED_ONLY_FIELDS),
        "promotion_required_fields": list(_SIGNAL_STATE_OBSERVABLE_FIELDS),
        "fail_closed_reason": (
            "signal_state_proxy_or_synthetic_not_planner_observable; "
            "do_not_count_as_signalized_benchmark_evidence"
        ),
        "benchmark_evidence": False,
    }


def _signal_state_for_metric_metadata(signal_state: Any) -> dict[str, Any] | None:
    """Return fail-closed signal-state metadata for runtime metric computation.

    Proxy signal metadata may be useful for trace diagnostics, but signal metrics may only enter
    denominators when the explicit planner-observable benchmark contract is met. Observable rows
    still need metric geometry fields; missing fields stay fail-closed in ``signal_metrics.py``.
    """
    if not isinstance(signal_state, dict):
        return None
    contract = _signal_state_promotion_contract(signal_state)
    if contract["contract_state"] != "planner_observable":
        return {
            "contract_state": contract["contract_state"],
            "benchmark_evidence": False,
        }
    metric_state: dict[str, Any] = {
        "contract_state": "planner_observable",
        "benchmark_evidence": True,
    }
    for key in ("timeline", "stop_line", "crosswalk_polygon"):
        if key in signal_state:
            metric_state[key] = signal_state[key]
    return metric_state


def _episode_metadata_for_signal_metrics(scenario: dict[str, Any]) -> dict[str, Any] | None:
    """Build optional episode metadata consumed by signal metrics.

    Returns:
        Optional episode metadata when the scenario carries metric-facing metadata.
    """
    metadata = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    episode_metadata: dict[str, Any] = {}
    signal_state = metadata.get("signal_state") if isinstance(metadata, dict) else None
    metric_signal_state = _signal_state_for_metric_metadata(signal_state)
    if metric_signal_state is not None:
        episode_metadata["signal_state"] = metric_signal_state

    rollover_stability = metadata.get(ROLLOVER_STABILITY_METADATA_KEY)
    if isinstance(rollover_stability, dict) and bool(rollover_stability.get("enabled", False)):
        episode_metadata[ROLLOVER_STABILITY_METADATA_KEY] = dict(rollover_stability)

    if not episode_metadata:
        return None
    return episode_metadata


def _synth_robot_stop_or_yield_expectation(
    robot_right_of_way: bool,
    pedestrian_right_of_way: bool,
    legality_state: str,
) -> str:
    """Synthesize a robot stop-or-yield expectation from proxy right-of-way fields.

    Returns:
        str: Diagnostic expectation label for trace/summary metadata.
    """
    if legality_state == "pedestrian_wait_required" and robot_right_of_way:
        return "proceed_clear"
    if legality_state == "pedestrian_crossing_allowed" and pedestrian_right_of_way:
        return "yield_to_pedestrian"
    if pedestrian_right_of_way:
        return "yield_to_pedestrian"
    return "proceed_clear"


def _signal_state_proxy_wrapper(
    signal_state: Any,
    intent_phase: str,
    intent_label: str,
    intent_source: str,
) -> dict[str, Any] | None:
    """Wrap signal-state proxy with bounded diagnostic fields for trace/summary export.

    This wrapper adds pedestrian_intent, robot_stop_or_yield_expectation,
    trace_fields_present, and claim_boundary=proxy_diagnostic on top of the
    existing signal-state trace payload. It does not add runtime simulation
    behavior or planner-observable signal-phase semantics.

    Returns:
        dict[str, Any] | None: Proxy diagnostic payload, or None when the
        input signal_state is absent.
    """
    base = _signal_state_trace_payload(signal_state, intent_phase)
    if base is None:
        return None

    pedestrian_intent = {
        "intent_label": intent_label,
        "intent_phase": intent_phase,
        "intent_source": intent_source,
    }
    robot_stop_or_yield_expectation = _synth_robot_stop_or_yield_expectation(
        base["robot_right_of_way"],
        base["pedestrian_right_of_way"],
        base["legality_state"],
    )
    return {
        **base,
        **_signal_state_promotion_contract(signal_state),
        "signal_phase": base["phase"],
        "pedestrian_intent": pedestrian_intent,
        "robot_stop_or_yield_expectation": robot_stop_or_yield_expectation,
        "trace_fields_present": list(_PROXY_TRACE_FIELDS_PRESENT),
        "claim_boundary": "proxy_diagnostic",
    }


def _vru_trace_payload(
    metadata: dict[str, Any],
    *,
    ped_pos: np.ndarray,
    velocity: np.ndarray,
    robot_position: np.ndarray | None,
    robot_velocity: np.ndarray | None,
) -> dict[str, Any]:
    """Build optional per-pedestrian cyclist-like VRU diagnostic trace fields.

    Returns:
        dict[str, Any]: JSON-serializable cyclist-like VRU diagnostic fields.
    """
    speed = float(np.linalg.norm(velocity))
    diagnostics: dict[str, Any] = {
        "speed_m_s": speed,
        "configured_speed_m_s": float(metadata["speed_m_s"]),
        "acceleration_m_s2": float(metadata["acceleration_m_s2"]),
    }
    if robot_position is not None:
        robot_pos = np.asarray(robot_position, dtype=float)
        robot_vel = (
            np.asarray(robot_velocity, dtype=float)
            if robot_velocity is not None
            else np.zeros(2, dtype=float)
        )
        offset = np.asarray(ped_pos, dtype=float) - robot_pos
        distance = float(np.linalg.norm(offset))
        relative_velocity = np.asarray(velocity, dtype=float) - robot_vel
        closing_speed = 0.0
        if distance > 1e-9:
            closing_speed = -float(np.dot(offset, relative_velocity) / distance)
        clearance = distance - float(metadata["actor_radius_m"]) - float(metadata["robot_radius_m"])
        diagnostics.update(
            {
                "distance_to_robot_m": distance,
                "relative_closing_speed_m_s": closing_speed,
                "time_to_conflict_zone_s": (
                    float(distance / closing_speed) if closing_speed > 1e-9 else None
                ),
                "clearance_m": clearance,
                "pass_overtake_state": _pass_overtake_state(closing_speed),
            }
        )
    payload_key = str(metadata.get("diagnostic_payload_key") or "cyclist_like_vru")
    return {
        "pedestrian_id": metadata["pedestrian_id"],
        "actor_type": metadata["actor_type"],
        "interaction_role": metadata["interaction_role"],
        "interaction_class": metadata.get("interaction_class", metadata["interaction_role"]),
        "claim_boundary": metadata["claim_boundary"],
        payload_key: diagnostics,
    }


def _pass_overtake_state(closing_speed_m_s: float) -> str:
    """Classify a one-step proxy pass/overtake state from relative closing speed.

    Returns:
        str: Diagnostic state for the proxy pass/overtake interaction.
    """
    if closing_speed_m_s > 0.1:
        return "approaching_conflict_zone"
    if closing_speed_m_s < -0.1:
        return "separating_after_pass"
    return "parallel_or_static_relative_motion"


def _intent_conditioned_behavior_summary(
    scenario: dict[str, Any],
    intent_metadata: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored intent-conditioned fixtures."""
    if not intent_metadata:
        return None
    summarized_pedestrians = [metadata for metadata in intent_metadata if metadata is not None]
    if not summarized_pedestrians:
        return None
    summary = {
        "schema_version": "intent-conditioned-behavior-summary.v1",
        "scenario_name": _scenario_id(scenario),
        "status": "diagnostic_metadata_only",
        "benchmark_evidence": False,
        "trace_field_source": "algorithm_metadata.simulation_step_trace.steps[].pedestrians[]",
        "claim_boundary": (
            "Authored intent metadata records fixture phases only; it is not data-grounded "
            "human behavior evidence and must not be used as a planner-ranking claim."
        ),
        "pedestrians": summarized_pedestrians,
    }
    signal_state = next(
        (
            metadata.get("signal_state")
            for metadata in summarized_pedestrians
            if isinstance(metadata.get("signal_state"), dict)
        ),
        None,
    )
    if isinstance(signal_state, dict):
        first_intent = next(
            (
                metadata
                for metadata in summarized_pedestrians
                if isinstance(metadata, dict) and metadata.get("intent_label")
            ),
            {},
        )
        phases = first_intent.get("intent_phases")
        first_phase = str(phases[0]) if isinstance(phases, list) and phases else "unknown"
        proxy_payload = _signal_state_proxy_wrapper(
            signal_state,
            first_phase,
            str(first_intent.get("intent_label", "unknown")),
            str(first_intent.get("intent_source", "unknown")),
        )
        summary["signal_state"] = (
            proxy_payload
            if proxy_payload is not None
            else {
                "schema_version": str(
                    signal_state.get("schema_version") or "signal-state-proxy.v1"
                ),
                "status": str(signal_state.get("status") or "proxy_diagnostic_only"),
                "signal_id": str(signal_state.get("signal_id") or "unknown_signal"),
                "conflict_zone_id": str(
                    signal_state.get("conflict_zone_id") or "unknown_conflict_zone"
                ),
                "planner_observable": _metadata_bool(signal_state.get("planner_observable", False)),
                "observation_mode": str(
                    signal_state.get("observation_mode") or "trace_metadata_only"
                ),
                "benchmark_evidence": _metadata_bool(signal_state.get("benchmark_evidence", False)),
                "claim_boundary": "proxy_diagnostic",
            }
        )
    return summary


def _cyclist_like_vru_summary(
    scenario: dict[str, Any],
    vru_metadata: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored cyclist-like VRU fixtures."""
    return _vru_diagnostic_summary(
        scenario,
        vru_metadata,
        payload_key="cyclist_like_vru",
        schema_version="cyclist-like-vru-smoke-summary.v1",
        claim_boundary=(
            "Authored cyclist-like VRU proxy metadata records speed, acceleration, "
            "time-to-conflict, clearance, and pass/overtake diagnostics only; it is not "
            "cyclist realism, cyclist behavior, or planner-ranking evidence."
        ),
    )


def _fast_bicycle_actor_summary(
    scenario: dict[str, Any],
    vru_metadata: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored fast-bicycle actor fixtures."""
    return _vru_diagnostic_summary(
        scenario,
        vru_metadata,
        payload_key="fast_bicycle_actor",
        schema_version="fast-bicycle-actor-summary.v1",
        claim_boundary=(
            "Authored fast-bicycle actor proxy metadata records speed, acceleration, "
            "time-to-conflict, clearance, and pass/overtake diagnostics only; it is not "
            "a full bicycle dynamics model, cyclist realism evidence, or planner-ranking evidence."
        ),
    )


def _vru_diagnostic_summary(
    scenario: dict[str, Any],
    vru_metadata: list[dict[str, Any] | None],
    *,
    payload_key: str,
    schema_version: str,
    claim_boundary: str,
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored fast-VRU fixtures."""
    if not vru_metadata:
        return None
    summarized = [
        metadata
        for metadata in vru_metadata
        if metadata is not None
        and (metadata.get("diagnostic_payload_key") or "cyclist_like_vru") == payload_key
    ]
    if not summarized:
        return None
    return {
        "schema_version": schema_version,
        "scenario_name": _scenario_id(scenario),
        "status": "diagnostic_metadata_only",
        "benchmark_evidence": False,
        "trace_field_source": "algorithm_metadata.simulation_step_trace.steps[].pedestrians[]",
        "claim_boundary": claim_boundary,
        "pedestrians": summarized,
    }


def _command_action_payload(command: Any) -> dict[str, Any]:
    """Normalize planner commands to trace-export selected_action fields.

    Returns:
        Linear and angular velocity action fields.
    """

    if isinstance(command, np.ndarray):
        command = command.tolist()
    if isinstance(command, dict) and command.get("command_kind") == "holonomic_vxy_world":
        return {
            "command_kind": "holonomic_vxy_world",
            "vx": _first_float(command.get("vx")),
            "vy": _first_float(command.get("vy")),
        }
    if isinstance(command, (list, tuple)) and len(command) >= 2:
        return {
            "linear_velocity": _first_float(command[0]),
            "angular_velocity": _first_float(command[1]),
        }
    return {"linear_velocity": 0.0, "angular_velocity": 0.0}
