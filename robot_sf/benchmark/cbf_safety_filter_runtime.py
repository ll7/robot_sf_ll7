"""Runtime binding helpers for the opt-in CBF safety filter."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robot_sf.planner.cbf_safety_filter import (
    CbfSafetyFilterConfig,
    build_cbf_safety_filter,
)

CBF_SAFETY_FILTER_RUNTIME_STEP_SCHEMA = "cbf_safety_filter_runtime_step.v1"
CBF_SAFETY_FILTER_EPISODE_SUMMARY_SCHEMA = "cbf_safety_filter_episode_summary.v1"
CBF_OFF_ARM = "cbf_off"
CBF_COLLISION_CONE_ARM = "cbf_collision_cone_on"
CBF_DYNAMIC_PARABOLIC_V1_ARM = "cbf_dynamic_parabolic_v1_on"
CBF_VARIANT_COLLISION_CONE = "collision_cone"
CBF_VARIANT_DYNAMIC_PARABOLIC_V1 = "dynamic_parabolic_cbf_v1"

_PREDECLARED_CBF_CONFIG_BY_ARM = {
    CBF_COLLISION_CONE_ARM: CbfSafetyFilterConfig(
        enabled=True,
        variant=CBF_VARIANT_COLLISION_CONE,
    ),
    CBF_DYNAMIC_PARABOLIC_V1_ARM: CbfSafetyFilterConfig(
        enabled=True,
        variant=CBF_VARIANT_DYNAMIC_PARABOLIC_V1,
    ),
}
_PREDECLARED_THRESHOLD_FIELDS = (
    "variant",
    "alpha",
    "safety_margin",
    "robot_radius",
    "pedestrian_radius",
    "max_linear_speed",
    "max_angular_speed",
    "turn_gain",
    "max_projection_passes",
    "min_clearance_h",
    "dpcbf_lambda_gain",
    "dpcbf_mu_gain",
    "relative_speed_epsilon",
    "dpcbf_grid_samples",
)


@dataclass(frozen=True, slots=True)
class CBFSafetyFilterRuntimeConfig:
    """Explicit opt-in benchmark CBF safety-filter binding."""

    enabled: bool = False
    arm_key: str = CBF_OFF_ARM
    variant: str = CBF_VARIANT_COLLISION_CONE
    alpha: float = 1.0
    safety_margin: float = 0.15
    robot_radius: float = 0.3
    pedestrian_radius: float = 0.3
    max_linear_speed: float | None = None
    max_angular_speed: float | None = None
    turn_gain: float = 2.0
    max_projection_passes: int = 3
    min_clearance_h: float = 1.0e-6
    dpcbf_lambda_gain: float = 1.0
    dpcbf_mu_gain: float = 1.0
    relative_speed_epsilon: float = 1.0e-6
    dpcbf_grid_samples: int = 161
    fallback_mode: str = "stop_keep_turn"
    fail_on_native_action: bool = True
    fail_on_unsupported_command: bool = True
    record_step_trace: bool = False


def runtime_config_from_mapping(
    payload: Mapping[str, Any] | None,
) -> CBFSafetyFilterRuntimeConfig:
    """Normalize mapping input into a fail-closed runtime config.

    Returns:
        Validated CBF safety-filter runtime configuration.
    """

    if payload is None:
        return CBFSafetyFilterRuntimeConfig()
    if not isinstance(payload, Mapping):
        raise TypeError("cbf_safety_filter must be a mapping or None")
    allowed = {field.name for field in CBFSafetyFilterRuntimeConfig.__dataclass_fields__.values()}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown cbf_safety_filter keys: {unknown}")
    return validate_runtime_config(
        CBFSafetyFilterRuntimeConfig(**{key: payload[key] for key in payload})
    )


def validate_runtime_config(  # noqa: C901
    runtime: CBFSafetyFilterRuntimeConfig,
) -> CBFSafetyFilterRuntimeConfig:
    """Validate arm semantics and predeclared threshold equality.

    Returns:
        Validated CBF safety-filter runtime configuration.
    """

    arm_key = str(runtime.arm_key)
    enabled_arms = {CBF_COLLISION_CONE_ARM, CBF_DYNAMIC_PARABOLIC_V1_ARM}
    if arm_key not in {CBF_OFF_ARM, *enabled_arms}:
        raise ValueError(
            "cbf_safety_filter.arm_key must be cbf_off, cbf_collision_cone_on, "
            "or cbf_dynamic_parabolic_v1_on; "
            "threshold changes require a versioned experimental arm"
        )
    if bool(runtime.enabled) and arm_key not in enabled_arms:
        raise ValueError("cbf_safety_filter.enabled=True requires a predeclared enabled arm_key")
    if not bool(runtime.enabled) and arm_key != CBF_OFF_ARM:
        raise ValueError("cbf_safety_filter.enabled=False requires arm_key='cbf_off'")
    expected_config = _PREDECLARED_CBF_CONFIG_BY_ARM.get(arm_key)
    if bool(runtime.enabled) and expected_config is None:
        raise ValueError("cbf_safety_filter enabled arm must have predeclared config")
    if arm_key == CBF_OFF_ARM and runtime.variant not in {
        CBF_VARIANT_COLLISION_CONE,
        CBF_VARIANT_DYNAMIC_PARABOLIC_V1,
    }:
        raise ValueError("cbf_safety_filter.variant must be a known CBF variant")
    if runtime.fallback_mode != "stop_keep_turn":
        raise ValueError("cbf_safety_filter.fallback_mode must be stop_keep_turn")
    if bool(runtime.enabled):
        for field_name in _PREDECLARED_THRESHOLD_FIELDS:
            observed = getattr(runtime, field_name)
            expected = getattr(expected_config, field_name)
            if observed is None or expected is None:
                if observed != expected:
                    raise ValueError(
                        f"cbf_safety_filter.{arm_key} thresholds must match "
                        f"predeclared ablation config: {field_name}={expected}"
                    )
                continue
            if isinstance(expected, str):
                if str(observed) != expected:
                    raise ValueError(
                        f"cbf_safety_filter.{arm_key} thresholds must match "
                        f"predeclared ablation config: {field_name}={expected}"
                    )
                continue
            if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    f"cbf_safety_filter.{arm_key} thresholds must match "
                    f"predeclared ablation config: {field_name}={expected}"
                )
    return runtime


def cbf_filter_config(runtime: CBFSafetyFilterRuntimeConfig) -> CbfSafetyFilterConfig:
    """Return pure CBF filter config from runtime binding."""

    return CbfSafetyFilterConfig(
        enabled=bool(runtime.enabled),
        variant=str(runtime.variant),
        alpha=float(runtime.alpha),
        safety_margin=float(runtime.safety_margin),
        robot_radius=float(runtime.robot_radius),
        pedestrian_radius=float(runtime.pedestrian_radius),
        max_linear_speed=runtime.max_linear_speed,
        max_angular_speed=runtime.max_angular_speed,
        turn_gain=float(runtime.turn_gain),
        max_projection_passes=int(runtime.max_projection_passes),
        min_clearance_h=float(runtime.min_clearance_h),
        dpcbf_lambda_gain=float(runtime.dpcbf_lambda_gain),
        dpcbf_mu_gain=float(runtime.dpcbf_mu_gain),
        relative_speed_epsilon=float(runtime.relative_speed_epsilon),
        dpcbf_grid_samples=int(runtime.dpcbf_grid_samples),
    )


def _xy_rows(value: Any) -> np.ndarray:
    """Return finite ``(n, 2)`` rows from simulator pedestrian state."""

    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("cbf_safety_filter simulator pedestrian positions must be finite")
    if arr.ndim == 1:
        if arr.size % 2:
            raise ValueError("cbf_safety_filter pedestrian positions require even-length xy data")
        rows = arr.reshape((-1, 2))
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        rows = arr[:, :2]
    else:
        raise ValueError("cbf_safety_filter pedestrian positions must be shaped (n,2) or (n,3)")
    return np.asarray(rows, dtype=float)


def _robot_position(env: Any) -> np.ndarray:
    simulator = getattr(env, "simulator", None)
    if simulator is None or not hasattr(simulator, "robot_pos"):
        raise ValueError("cbf_safety_filter requires simulator.robot_pos")
    try:
        arr = np.asarray(simulator.robot_pos[0], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("cbf_safety_filter requires simulator.robot_pos[0] xy position") from exc
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        raise ValueError("cbf_safety_filter robot_pos must contain finite xy coordinates")
    return arr[:2]


def _pedestrian_positions(env: Any) -> np.ndarray:
    simulator = getattr(env, "simulator", None)
    return _xy_rows(getattr(simulator, "ped_pos", np.empty((0, 2), dtype=float)))


def _robot_heading(env: Any) -> float:
    simulator = getattr(env, "simulator", None)
    robot_poses = getattr(simulator, "robot_poses", None)
    if isinstance(robot_poses, list) and robot_poses:
        try:
            heading = float(robot_poses[0][1])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("cbf_safety_filter requires finite robot_poses heading") from exc
        if math.isfinite(heading):
            return heading
        raise ValueError("cbf_safety_filter requires finite robot_poses heading")
    robots = getattr(simulator, "robots", None)
    if isinstance(robots, list) and robots:
        theta = getattr(robots[0], "theta", None)
        if theta is not None:
            return float(np.asarray(theta, dtype=float).reshape(-1)[0])
    robot_pos = getattr(simulator, "robot_pos", None)
    try:
        arr = np.asarray(robot_pos[0], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("cbf_safety_filter requires robot heading from simulator") from exc
    if arr.size >= 3 and np.isfinite(arr[2]):
        return float(arr[2])
    raise ValueError("cbf_safety_filter requires finite robot heading")


def _pedestrian_velocities(
    current_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    dt: float,
) -> np.ndarray:
    if not dt > 0.0:
        raise ValueError("cbf_safety_filter dt must be positive")
    if previous_positions is None:
        return np.zeros_like(current_positions, dtype=float)
    previous = _xy_rows(previous_positions)
    velocities = np.zeros_like(current_positions, dtype=float)
    count = min(current_positions.shape[0], previous.shape[0])
    if count:
        velocities[:count] = (current_positions[:count] - previous[:count]) / float(dt)
    return velocities


def compute_cbf_observation_from_env(
    *,
    env: Any,
    config: Any,
    previous_ped_positions: np.ndarray | None,
    dt: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the current-state observation consumed by the pure CBF filter.

    Returns:
        Planner-style CBF observation and provenance metadata.
    """

    del config
    robot_pos = _robot_position(env)
    heading = _robot_heading(env)
    ped_positions = _pedestrian_positions(env)
    ped_velocities = _pedestrian_velocities(ped_positions, previous_ped_positions, dt)
    observation = {
        "robot": {
            "position": [float(robot_pos[0]), float(robot_pos[1])],
            "velocity": [0.0, 0.0],
            "heading": float(heading),
        },
        "agents": [
            {
                "position": [float(position[0]), float(position[1])],
                "velocity": [float(velocity[0]), float(velocity[1])],
            }
            for position, velocity in zip(ped_positions, ped_velocities, strict=False)
        ],
    }
    provenance = {
        "context_source": "simulator_state_pre_step",
        "pedestrian_velocity_identity": "row_order_finite_difference_no_stable_ids",
        "obstacle_count": int(ped_positions.shape[0]),
    }
    return observation, provenance


def _status_from_decision_label(label: str, *, intervened: bool) -> str:
    if label == "cbf_disabled":
        return "disabled"
    if intervened:
        return "filtered"
    if label in {"cbf_feasible", "cbf_no_constraints"}:
        return "pass_through"
    if label == "cbf_best_effort":
        return "fallback_infeasible"
    return "filtered"


def apply_runtime_cbf_safety_filter(
    *,
    command: Sequence[float],
    env: Any,
    config: Any,
    runtime: CBFSafetyFilterRuntimeConfig,
    previous_ped_positions: np.ndarray | None,
    step_idx: int,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Apply runtime CBF filtering and return a schema-tagged step record.

    Returns:
        Filtered ``(linear, angular)`` command and per-step evidence record.
    """

    if len(command) < 2:
        raise TypeError("cbf_safety_filter command must contain linear and angular velocity")
    observation, provenance = compute_cbf_observation_from_env(
        env=env,
        config=config,
        previous_ped_positions=previous_ped_positions,
        dt=float(config.sim_config.time_per_step_in_secs),
    )
    filter_ = build_cbf_safety_filter(cbf_filter_config(runtime))
    decision = filter_.filter_command(observation, (float(command[0]), float(command[1])))
    decision_meta = decision.to_metadata()
    qp_status = _status_from_decision_label(
        str(decision.decision_label),
        intervened=bool(decision.is_intervention),
    )
    record = {
        "schema_version": CBF_SAFETY_FILTER_RUNTIME_STEP_SCHEMA,
        "step": int(step_idx),
        "arm_key": str(runtime.arm_key),
        "enabled": bool(runtime.enabled),
        "variant": str(runtime.variant),
        "eligible_for_cbf_filter": True,
        "qp_status": qp_status,
        "qp_feasible": qp_status != "fallback_infeasible",
        "fallback_applied": qp_status == "fallback_infeasible",
        "intervened": bool(decision.is_intervention),
        "nominal_linear_velocity": float(command[0]),
        "nominal_angular_velocity": float(command[1]),
        "filtered_linear_velocity": float(decision.filtered_action[0]),
        "filtered_angular_velocity": float(decision.filtered_action[1]),
        "active_constraint_count": len(decision.violated_constraints),
        "min_barrier_before": decision.proposed_evaluation.get("min_cbf_margin"),
        "min_barrier_after": decision.selected_evaluation.get("min_cbf_margin"),
        "hard_constraint_violation": bool(decision.final_constraint_violation),
        "projection_method": decision.fallback_controller_state.get("projection_method"),
        **provenance,
    }
    if runtime.record_step_trace:
        record["decision"] = decision_meta
    return (
        (float(decision.filtered_action[0]), float(decision.filtered_action[1])),
        record,
    )


def ineligible_cbf_safety_filter_step_record(
    *,
    runtime: CBFSafetyFilterRuntimeConfig,
    step_idx: int,
    reason: str,
) -> dict[str, Any]:
    """Return a schema-tagged record for fail-open ineligible cases."""

    return {
        "schema_version": CBF_SAFETY_FILTER_RUNTIME_STEP_SCHEMA,
        "step": int(step_idx),
        "arm_key": str(runtime.arm_key),
        "enabled": bool(runtime.enabled),
        "variant": str(runtime.variant),
        "eligible_for_cbf_filter": False,
        "ineligible_reason": str(reason),
        "qp_status": "ineligible",
        "qp_feasible": False,
        "fallback_applied": False,
        "intervened": False,
        "nominal_linear_velocity": None,
        "nominal_angular_velocity": None,
        "filtered_linear_velocity": None,
        "filtered_angular_velocity": None,
        "active_constraint_count": 0,
        "min_barrier_before": None,
        "min_barrier_after": None,
        "hard_constraint_violation": False,
    }


def summarize_cbf_safety_filter_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    runtime: CBFSafetyFilterRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Summarize per-step CBF evidence for episode metadata and ledger provenance.

    Returns:
        Episode-level CBF safety-filter summary payload.
    """

    runtime = runtime or CBFSafetyFilterRuntimeConfig()
    step_count = len(trace)
    status_counts = Counter(str(record.get("qp_status", "unknown")) for record in trace)
    intervened_step_count = sum(1 for record in trace if bool(record.get("intervened", False)))
    infeasible_count = sum(1 for record in trace if not bool(record.get("qp_feasible", True)))
    fallback_count = sum(1 for record in trace if bool(record.get("fallback_applied", False)))
    claim_boundary = (
        "diagnostic opt-in Dynamic Parabolic CBF runtime arm; bounded comparison evidence "
        "only; not a formal safety certificate"
        if runtime.arm_key == CBF_DYNAMIC_PARABOLIC_V1_ARM
        else "diagnostic opt-in collision-cone CBF baseline; not a safety certificate"
    )
    summary = {
        "schema_version": CBF_SAFETY_FILTER_EPISODE_SUMMARY_SCHEMA,
        "enabled": bool(runtime.enabled),
        "arm_key": str(runtime.arm_key),
        "variant": str(runtime.variant),
        "thresholds": asdict(cbf_filter_config(runtime)),
        "step_count": int(step_count),
        "eligible_step_count": int(
            sum(1 for record in trace if bool(record.get("eligible_for_cbf_filter", True)))
        ),
        "intervened_step_count": int(intervened_step_count),
        "qp_infeasible_step_count": int(infeasible_count),
        "fallback_step_count": int(fallback_count),
        "intervention_rate": intervened_step_count / step_count if step_count else 0.0,
        "qp_infeasible_rate": infeasible_count / step_count if step_count else 0.0,
        "fallback_rate": fallback_count / step_count if step_count else 0.0,
        "status_counts": dict(status_counts),
        "claim_boundary": claim_boundary,
        "evidence_tier": "bounded_runtime_comparison",
        "fallback_rows_are_success_evidence": False,
        "context_source": "simulator_state_pre_step",
    }
    if runtime.record_step_trace:
        summary["steps"] = [dict(record) for record in trace]
    return summary


__all__ = [
    "CBF_COLLISION_CONE_ARM",
    "CBF_DYNAMIC_PARABOLIC_V1_ARM",
    "CBF_OFF_ARM",
    "CBF_SAFETY_FILTER_EPISODE_SUMMARY_SCHEMA",
    "CBF_SAFETY_FILTER_RUNTIME_STEP_SCHEMA",
    "CBFSafetyFilterRuntimeConfig",
    "apply_runtime_cbf_safety_filter",
    "cbf_filter_config",
    "compute_cbf_observation_from_env",
    "ineligible_cbf_safety_filter_step_record",
    "runtime_config_from_mapping",
    "summarize_cbf_safety_filter_trace",
]
