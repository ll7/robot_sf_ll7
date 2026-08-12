"""Strict trace-table extraction for the bounded BRNE corridor diagnosis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from robot_sf.baselines.brne import BRNE_PINNED_SHA

TRACE_TABLE_SCHEMA_VERSION = "brne-trace-table.v1"
SIMULATION_TRACE_SCHEMA_VERSION = "simulation-step-trace.v1"
INTERACTION_EXPOSURE_RADIUS_M = 2.0
_ZERO_SPEED_EPSILON_M_S = 1.0e-9


def _finite(value: Any, *, field: str) -> float:
    """Return one finite float or fail closed.

    Returns:
        The finite numeric value.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite: {value!r}")
    return parsed


def _finite_pair(value: Any, *, field: str) -> list[float]:
    """Return a finite two-dimensional point or fail closed.

    Returns:
        A JSON-safe ``[x, y]`` list.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{field} must contain two coordinates")
    return [_finite(value[0], field=f"{field}[0]"), _finite(value[1], field=f"{field}[1]")]


def _optional_nonnegative_int(value: Any, *, field: str) -> int | None:
    """Validate an optional non-negative integer.

    Returns:
        The integer or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or null")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer or null") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != parsed):
        raise ValueError(f"{field} must be a non-negative integer or null")
    if isinstance(value, str) and value.strip() != str(parsed):
        raise ValueError(f"{field} must be a non-negative integer or null")
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer or null")
    return parsed


def _wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to the closed interval ``[-pi, pi]``.

    Returns:
        The wrapped angle in radians.
    """
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _validate_envelope(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the required simulation trace envelope.

    Returns:
        A normalized envelope used by the table builder.
    """
    if trace.get("schema_version") != SIMULATION_TRACE_SCHEMA_VERSION:
        raise ValueError("simulation-step-trace.v1 is required")
    dt = _finite(trace.get("dt"), field="simulation_step_trace.dt")
    if dt <= 0.0:
        raise ValueError("simulation_step_trace.dt must be positive")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("simulation_step_trace.steps must be a non-empty list")
    initial_goal_distance = _finite(
        trace.get("initial_goal_distance_m"), field="initial_goal_distance_m"
    )
    if initial_goal_distance < 0.0:
        raise ValueError("initial_goal_distance_m must be non-negative")
    goal_position = _finite_pair(trace.get("goal_position"), field="goal_position")
    initial_goal_position = _finite_pair(
        trace.get("initial_goal_position"), field="initial_goal_position"
    )
    initial_robot_position = _finite_pair(
        trace.get("initial_robot_position"), field="initial_robot_position"
    )
    if not math.isclose(
        initial_goal_distance,
        math.dist(initial_robot_position, initial_goal_position),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("initial_goal_distance_m is inconsistent with the initial goal and robot")
    robot_radius = _finite(trace.get("robot_radius_m"), field="robot_radius_m")
    ped_radius = _finite(trace.get("ped_radius_m"), field="ped_radius_m")
    if robot_radius < 0.0 or ped_radius < 0.0:
        raise ValueError("trace radii must be non-negative")
    reached_goal_step = _optional_nonnegative_int(
        trace.get("reached_goal_step"), field="reached_goal_step"
    )
    collision_step = _optional_nonnegative_int(trace.get("collision_step"), field="collision_step")
    termination_reason = trace.get("termination_reason")
    if not isinstance(termination_reason, str) or not termination_reason.strip():
        raise ValueError("termination_reason must be a non-empty string")
    return {
        "dt": dt,
        "steps": steps,
        "initial_goal_distance_m": initial_goal_distance,
        "goal_position": goal_position,
        "initial_goal_position": initial_goal_position,
        "initial_robot_position": initial_robot_position,
        "robot_radius_m": robot_radius,
        "ped_radius_m": ped_radius,
        "reached_goal_step": reached_goal_step,
        "collision_step": collision_step,
        "termination_reason": termination_reason,
    }


def _validate_pedestrians(value: Any, *, step_index: int) -> list[dict[str, Any]]:
    """Validate world-frame pedestrian positions and velocities.

    Returns:
        Normalized pedestrian frame entries.
    """
    if not isinstance(value, list):
        raise ValueError(f"steps[{step_index}].pedestrians must be a list")
    normalized: list[dict[str, Any]] = []
    for ped_index, pedestrian in enumerate(value):
        if not isinstance(pedestrian, Mapping):
            raise ValueError(f"steps[{step_index}].pedestrians[{ped_index}] must be a mapping")
        normalized.append(
            {
                "id": pedestrian.get("id", ped_index),
                "position": _finite_pair(
                    pedestrian.get("position"),
                    field=f"steps[{step_index}].pedestrians[{ped_index}].position",
                ),
                "velocity": _finite_pair(
                    pedestrian.get("velocity"),
                    field=f"steps[{step_index}].pedestrians[{ped_index}].velocity",
                ),
            }
        )
    return normalized


def _validate_action(value: Any, *, field: str) -> dict[str, float]:
    """Validate a selected linear/angular command.

    Returns:
        A normalized command mapping.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return {
        "linear_velocity_m_s": _finite(
            value.get("linear_velocity"), field=f"{field}.linear_velocity"
        ),
        "angular_velocity_rad_s": _finite(
            value.get("angular_velocity"), field=f"{field}.angular_velocity"
        ),
    }


def _validate_step(raw_step: Any, *, step_index: int, dt: float) -> dict[str, Any]:
    """Validate one retained trace step without inferring terminal outcomes.

    Returns:
        Normalized step fields.
    """
    if not isinstance(raw_step, Mapping):
        raise ValueError(f"steps[{step_index}] must be a mapping")
    if raw_step.get("step") != step_index:
        raise ValueError(f"steps[{step_index}].step must equal {step_index}")
    time_s = _finite(raw_step.get("time_s"), field=f"steps[{step_index}].time_s")
    if not math.isclose(time_s, (step_index + 1) * dt, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"steps[{step_index}].time_s is inconsistent with dt")
    robot = raw_step.get("robot")
    if not isinstance(robot, Mapping):
        raise ValueError(f"steps[{step_index}].robot must be a mapping")
    position = _finite_pair(robot.get("position"), field=f"steps[{step_index}].robot.position")
    velocity = _finite_pair(robot.get("velocity"), field=f"steps[{step_index}].robot.velocity")
    heading = _finite(robot.get("heading"), field=f"steps[{step_index}].robot.heading")
    goal_position = _finite_pair(
        raw_step.get("goal_position"), field=f"steps[{step_index}].goal_position"
    )
    planner = raw_step.get("planner")
    if not isinstance(planner, Mapping):
        raise ValueError(f"steps[{step_index}].planner must be a mapping")
    selected_action = _validate_action(
        planner.get("selected_action"), field=f"steps[{step_index}].planner.selected_action"
    )
    amv_raw = planner.get("amv")
    amv = None
    if amv_raw is not None:
        if not isinstance(amv_raw, Mapping):
            raise ValueError(f"steps[{step_index}].planner.amv must be a mapping")
        amv = {
            "requested_linear_m_s": _finite(
                amv_raw.get("requested_linear_m_s"),
                field=f"steps[{step_index}].planner.amv.requested_linear_m_s",
            ),
            "requested_angular_rad_s": _finite(
                amv_raw.get("requested_angular_rad_s"),
                field=f"steps[{step_index}].planner.amv.requested_angular_rad_s",
            ),
            "applied_linear_m_s": _finite(
                amv_raw.get("applied_linear_m_s"),
                field=f"steps[{step_index}].planner.amv.applied_linear_m_s",
            ),
            "applied_angular_rad_s": _finite(
                amv_raw.get("applied_angular_rad_s"),
                field=f"steps[{step_index}].planner.amv.applied_angular_rad_s",
            ),
            "command_clipped": bool(amv_raw.get("command_clipped", False)),
            "yaw_rate_saturated": bool(amv_raw.get("yaw_rate_saturated", False)),
        }
    rl = raw_step.get("rl")
    if not isinstance(rl, Mapping):
        raise ValueError(f"steps[{step_index}].rl must be a mapping")
    if not isinstance(rl.get("terminated"), bool) or not isinstance(rl.get("truncated"), bool):
        raise ValueError(f"steps[{step_index}].rl terminal flags must be booleans")
    reward = _finite(rl.get("reward"), field=f"steps[{step_index}].rl.reward")
    return {
        "step": step_index,
        "time_s": time_s,
        "position": position,
        "velocity": velocity,
        "heading": heading,
        "goal_position": goal_position,
        "pedestrians": _validate_pedestrians(raw_step.get("pedestrians"), step_index=step_index),
        "selected_action": selected_action,
        "amv": amv,
        "terminated": rl["terminated"],
        "truncated": rl["truncated"],
        "reward": reward,
    }


def _runtime_provenance(  # noqa: C901 - explicit fail-closed provenance gate
    metadata: Mapping[str, Any], *, expected_effective_num_samples: int | None
) -> dict[str, Any]:
    """Validate BRNE runtime and source provenance.

    Returns:
        The validated runtime provenance block.
    """
    diagnostic = metadata.get("brne_diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("BRNE diagnostic metadata is missing")
    if diagnostic.get("status") != "native_core_via_adapter":
        raise ValueError("BRNE diagnostic metadata is not native-core-via-adapter")
    runtime = metadata.get("planner_runtime")
    runtime_meta = runtime.get("planner_metadata") if isinstance(runtime, Mapping) else None
    if not isinstance(runtime_meta, Mapping):
        raise ValueError("BRNE planner runtime metadata is missing")
    if runtime_meta.get("status") != "ok" or runtime_meta.get("runtime_status") != "ok":
        raise ValueError("BRNE runtime is not successful")
    failure_count = runtime_meta.get("failure_count")
    if isinstance(failure_count, bool) or not isinstance(failure_count, int) or failure_count != 0:
        raise ValueError("BRNE runtime failure_count must be zero")
    effective = runtime_meta.get("effective_num_samples")
    if isinstance(effective, bool) or not isinstance(effective, int) or effective < 1:
        raise ValueError("BRNE effective_num_samples is missing or invalid")
    if expected_effective_num_samples is not None and effective != expected_effective_num_samples:
        raise ValueError("BRNE effective_num_samples does not match the frozen contract")
    for field in ("source_commit", "source_pin", "source_integrity"):
        if not isinstance(runtime_meta.get(field), str) or not runtime_meta[field].strip():
            raise ValueError(f"BRNE runtime provenance field {field} is missing")
    if runtime_meta["source_commit"] != BRNE_PINNED_SHA:
        raise ValueError("BRNE runtime source_commit does not match the frozen pin")
    if runtime_meta["source_pin"] != BRNE_PINNED_SHA:
        raise ValueError("BRNE runtime source_pin does not match the frozen pin")
    if runtime_meta.get("source_integrity") != "clean_pinned_worktree":
        raise ValueError("BRNE source integrity is not a clean pinned worktree")
    aggregation = runtime_meta.get("aggregation_layout")
    if not isinstance(aggregation, Mapping):
        planner_metadata = metadata.get("planner_metadata")
        aggregation = (
            planner_metadata.get("aggregation_layout")
            if isinstance(planner_metadata, Mapping)
            else None
        )
    if not isinstance(aggregation, Mapping):
        raise ValueError("BRNE aggregation layout metadata is missing")
    if (
        aggregation.get("method") != "weighted_first_command"
        or aggregation.get("ensemble_layout") != "plan_step_first"
    ):
        raise ValueError("BRNE aggregation layout metadata is invalid")
    return {
        "status": runtime_meta["status"],
        "runtime_status": runtime_meta["runtime_status"],
        "failure_count": failure_count,
        "failure_reasons": list(runtime_meta.get("failure_reasons", [])),
        "source_commit": runtime_meta["source_commit"],
        "source_pin": runtime_meta["source_pin"],
        "source_integrity": runtime_meta["source_integrity"],
        "effective_num_samples": effective,
        "step_count": runtime_meta.get("step_count"),
        "aggregation_layout": dict(aggregation),
    }


def _record_provenance(
    record: Mapping[str, Any],
    *,
    planner_key: str | None,
    expected_effective_num_samples: int | None,
) -> dict[str, Any] | None:
    """Validate row admission metadata and return BRNE provenance when applicable.

    Returns:
        A provenance mapping for BRNE or ``None`` for comparator rows.
    """
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("record.algorithm_metadata must be a mapping")
    status = str(metadata.get("status", "")).strip().lower()
    if status != "ok" or metadata.get("fallback_reason") or metadata.get("fallback_triggered"):
        raise ValueError("fallback, degraded, or non-ok rows are unavailable")
    if planner_key == "brne":
        return _runtime_provenance(
            metadata, expected_effective_num_samples=expected_effective_num_samples
        )
    return None


def _clearance(
    position: list[float], pedestrians: list[dict[str, Any]], robot_radius: float, ped_radius: float
) -> float | None:
    """Return minimum robot-pedestrian surface clearance.

    Returns:
        The minimum finite clearance or ``None`` when the frame has no pedestrians.
    """
    if not pedestrians:
        return None
    return min(
        math.dist(position, pedestrian["position"]) - robot_radius - ped_radius
        for pedestrian in pedestrians
    )


def _interaction_count(position: list[float], pedestrians: list[dict[str, Any]]) -> int:
    """Count pedestrians inside the canonical 2 m diagnostic exposure radius.

    Returns:
        Number of pedestrians in the interaction zone.
    """
    return sum(
        math.dist(position, pedestrian["position"]) <= INTERACTION_EXPOSURE_RADIUS_M
        for pedestrian in pedestrians
    )


def _step_row(
    step: dict[str, Any],
    *,
    previous: dict[str, Any] | None,
    initial_goal_distance: float,
    robot_radius: float,
    ped_radius: float,
    aggregation_layout: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one derived trace row.

    Returns:
        A JSON-safe per-step table row.
    """
    position = step["position"]
    velocity = step["velocity"]
    speed = math.hypot(*velocity)
    velocity_heading = (
        math.atan2(velocity[1], velocity[0]) if speed > _ZERO_SPEED_EPSILON_M_S else None
    )
    goal_position = step["goal_position"]
    goal_bearing = math.atan2(goal_position[1] - position[1], goal_position[0] - position[0])
    distance_to_goal = math.dist(position, goal_position)
    previous_distance = previous["distance_to_goal_m"] if previous else initial_goal_distance
    goal_switched = previous is not None and previous["goal_position"] != goal_position
    signed_progress_delta = (
        0.0 if goal_switched else previous_distance - distance_to_goal
    )
    selected = step["selected_action"]
    previous_selected = previous["selected_command"] if previous else None
    amv = step["amv"]
    previous_amv = previous.get("amv") if previous else None
    row = {
        "step": step["step"],
        "time_s": step["time_s"],
        "goal_position": goal_position,
        "robot_position": position,
        "robot_heading_rad": step["heading"],
        "velocity_derived_heading_rad": velocity_heading,
        "robot_velocity_world_m_s": velocity,
        "robot_speed_m_s": speed,
        "goal_bearing_rad": goal_bearing,
        "heading_goal_angular_difference_rad": _wrap_angle(goal_bearing - step["heading"]),
        "distance_to_goal_m": distance_to_goal,
        "signed_progress_delta_m": signed_progress_delta,
        "progress_from_start_m": initial_goal_distance - distance_to_goal,
        "goal_switched": goal_switched,
        "pedestrians_world": step["pedestrians"],
        "pedestrian_count": len(step["pedestrians"]),
        "interaction_count": _interaction_count(position, step["pedestrians"]),
        "min_clearance_m": _clearance(position, step["pedestrians"], robot_radius, ped_radius),
        "selected_command": selected,
        "selected_action_delta_linear_m_s": (
            selected["linear_velocity_m_s"] - previous_selected["linear_velocity_m_s"]
            if previous_selected is not None
            else None
        ),
        "selected_action_delta_angular_rad_s": (
            selected["angular_velocity_rad_s"] - previous_selected["angular_velocity_rad_s"]
            if previous_selected is not None
            else None
        ),
        "amv": amv,
        "amv_delta_linear_m_s": (
            amv["applied_linear_m_s"] - previous_amv["applied_linear_m_s"]
            if amv is not None and previous_amv is not None
            else None
        ),
        "amv_delta_angular_rad_s": (
            amv["applied_angular_rad_s"] - previous_amv["applied_angular_rad_s"]
            if amv is not None and previous_amv is not None
            else None
        ),
        "terminated": step["terminated"],
        "truncated": step["truncated"],
        "reward": step["reward"],
    }
    if aggregation_layout is not None:
        row["brne_aggregation_layout"] = aggregation_layout
    return row


def _phase_progress(rows: list[dict[str, Any]], *, interaction: bool | None) -> dict[str, Any]:
    """Aggregate signed progress over all, interaction, or non-interaction steps.

    Returns:
        A count and signed-progress summary.
    """
    selected = [
        row for row in rows if interaction is None or (row["interaction_count"] > 0) == interaction
    ]
    return {
        "steps": len(selected),
        "goal_switch_steps": sum(row["goal_switched"] for row in selected),
        "signed_progress_m": sum(row["signed_progress_delta_m"] for row in selected),
        "mean_signed_progress_delta_m": (
            sum(row["signed_progress_delta_m"] for row in selected) / len(selected)
            if selected
            else 0.0
        ),
    }


def build_trace_table(
    record: dict[str, Any],
    *,
    planner_key: str | None = None,
    expected_effective_num_samples: int | None = None,
) -> dict[str, Any]:
    """Build a strict mechanism table from one eligible episode record.

    Args:
        record: Episode record containing ``algorithm_metadata.simulation_step_trace``.
        planner_key: Frozen arm key; ``brne`` enables strict BRNE provenance checks.
        expected_effective_num_samples: Optional frozen BRNE sample-count requirement.

    Returns:
        A ``brne-trace-table.v1`` mapping with episode and per-step rows.

    Raises:
        ValueError: If any required trace or provenance field is missing or malformed.
    """
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("record.algorithm_metadata must be a mapping")
    raw_trace = metadata.get("simulation_step_trace")
    if not isinstance(raw_trace, Mapping):
        raise ValueError("simulation_step_trace.v1 is missing")
    envelope = _validate_envelope(raw_trace)
    resolved_planner = planner_key or str(record.get("algo") or "").strip().lower() or None
    provenance = _record_provenance(
        record,
        planner_key=resolved_planner,
        expected_effective_num_samples=expected_effective_num_samples,
    )
    aggregation_layout = provenance.get("aggregation_layout") if provenance else None
    previous: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for index, raw_step in enumerate(envelope["steps"]):
        step = _validate_step(raw_step, step_index=index, dt=envelope["dt"])
        row = _step_row(
            step,
            previous=previous,
            initial_goal_distance=envelope["initial_goal_distance_m"],
            robot_radius=envelope["robot_radius_m"],
            ped_radius=envelope["ped_radius_m"],
            aggregation_layout=aggregation_layout,
        )
        rows.append(row)
        previous = {
            "distance_to_goal_m": row["distance_to_goal_m"],
            "goal_position": row["goal_position"],
            "selected_command": row["selected_command"],
            "amv": row["amv"],
        }
    interaction_steps = sum(row["interaction_count"] > 0 for row in rows)
    clearances = [row["min_clearance_m"] for row in rows if row["min_clearance_m"] is not None]
    episode = {
        "episode_id": record.get("episode_id"),
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "planner": resolved_planner,
        "num_steps": len(rows),
        "dt": envelope["dt"],
        "goal_position": rows[-1]["goal_position"],
        "initial_goal_position": envelope["initial_goal_position"],
        "initial_robot_position": envelope["initial_robot_position"],
        "initial_goal_distance_m": envelope["initial_goal_distance_m"],
        "final_goal_distance_m": rows[-1]["distance_to_goal_m"],
        "displacement_m": math.dist(rows[0]["robot_position"], rows[-1]["robot_position"]),
        "path_length_m": sum(
            math.dist(previous_row["robot_position"], row["robot_position"])
            for previous_row, row in pairwise(rows)
        ),
        "min_clearance_m": min(clearances) if clearances else None,
        "interaction_exposure_radius_m": INTERACTION_EXPOSURE_RADIUS_M,
        "interaction_exposure_steps": interaction_steps,
        "interaction_exposure_fraction": interaction_steps / len(rows),
        "progress_by_phase": {
            "all": _phase_progress(rows, interaction=None),
            "interaction": _phase_progress(rows, interaction=True),
            "non_interaction": _phase_progress(rows, interaction=False),
        },
        "reached_goal_step": envelope["reached_goal_step"],
        "collision_step": envelope["collision_step"],
        "termination_reason": envelope["termination_reason"],
        "goal_reached": envelope["reached_goal_step"] is not None,
        "collision_detected": envelope["collision_step"] is not None,
    }
    result: dict[str, Any] = {
        "schema_version": TRACE_TABLE_SCHEMA_VERSION,
        "episode": episode,
        "steps": rows,
    }
    if provenance is not None:
        result["provenance"] = provenance
    return result


__all__ = ["TRACE_TABLE_SCHEMA_VERSION", "build_trace_table"]
