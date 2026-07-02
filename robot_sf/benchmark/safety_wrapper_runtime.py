"""Runtime binding helpers for the planner-agnostic safety wrapper.

The runtime layer is intentionally opt-in and planner-agnostic. It constructs the
pre-step :class:`robot_sf.robot.safety_wrapper.SafetyContext` from simulator
state that is already visible to benchmark execution, applies the pure wrapper
transform, and returns compact schema-tagged records for episode evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robot_sf.robot.safety_wrapper import (
    INTERVENTION_DISABLED,
    INTERVENTION_HARD_STOP,
    INTERVENTION_NONE,
    INTERVENTION_SPEED_CAP,
    SafetyContext,
    SafetyWrapperConfig,
    apply_safety_wrapper,
)

SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA = "safety_wrapper_runtime_step.v1"
SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA = "safety_wrapper_episode_summary.v1"
WRAPPER_OFF_ARM = "wrapper_off"
WRAPPER_ON_ARM = "wrapper_on"
_PREDECLARED_WRAPPER_CONFIG = SafetyWrapperConfig()
_PREDECLARED_THRESHOLD_FIELDS = (
    "pedestrian_caution_radius_m",
    "capped_speed_m_s",
    "ttc_veto_threshold_s",
    "clearance_veto_m",
)


@dataclass(frozen=True, slots=True)
class SafetyWrapperRuntimeConfig:
    """Explicit opt-in runtime configuration for benchmark wrapper binding."""

    enabled: bool = False
    arm_key: str = "wrapper_off"
    pedestrian_caution_radius_m: float = 2.0
    capped_speed_m_s: float = 0.5
    ttc_veto_threshold_s: float = 1.0
    clearance_veto_m: float = 0.3
    fail_on_native_action: bool = True
    fail_on_unsupported_command: bool = True
    record_step_trace: bool = False
    false_stop_lookahead_s: float = 2.0


def runtime_config_from_mapping(
    payload: Mapping[str, Any] | None,
) -> SafetyWrapperRuntimeConfig:
    """Normalize optional mapping input into runtime config.

    ``None`` and empty mappings keep the wrapper disabled, preserving benchmark
    behavior unless a caller explicitly opts in.

    Returns:
        Runtime config with fixed thresholds and fail-closed controls.
    """

    if payload is None:
        return SafetyWrapperRuntimeConfig()
    if not isinstance(payload, Mapping):
        raise TypeError("safety_wrapper must be a mapping or None")
    allowed = {field.name for field in SafetyWrapperRuntimeConfig.__dataclass_fields__.values()}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown safety_wrapper keys: {unknown}")
    return validate_runtime_config(
        SafetyWrapperRuntimeConfig(**{key: payload[key] for key in payload})
    )


def validate_runtime_config(runtime: SafetyWrapperRuntimeConfig) -> SafetyWrapperRuntimeConfig:
    """Validate runtime wrapper arm semantics against predeclared ablation contract.

    Returns:
        Validated runtime config unchanged.
    """

    arm_key = str(runtime.arm_key)
    if arm_key not in {WRAPPER_OFF_ARM, WRAPPER_ON_ARM}:
        raise ValueError(
            "safety_wrapper.arm_key must be wrapper_off or wrapper_on; "
            "threshold changes require a versioned experimental arm"
        )
    if bool(runtime.enabled) and arm_key != WRAPPER_ON_ARM:
        raise ValueError("safety_wrapper.enabled=True requires arm_key='wrapper_on'")
    if not bool(runtime.enabled) and arm_key != WRAPPER_OFF_ARM:
        raise ValueError("safety_wrapper.enabled=False requires arm_key='wrapper_off'")
    if bool(runtime.enabled):
        for field_name in _PREDECLARED_THRESHOLD_FIELDS:
            observed = float(getattr(runtime, field_name))
            expected = float(getattr(_PREDECLARED_WRAPPER_CONFIG, field_name))
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    "safety_wrapper.wrapper_on thresholds must match "
                    f"predeclared ablation config: {field_name}={expected}"
                )
    return runtime


def safety_wrapper_config(runtime: SafetyWrapperRuntimeConfig) -> SafetyWrapperConfig:
    """Return the pure wrapper config with fixed predeclared thresholds."""

    return SafetyWrapperConfig(
        enabled=bool(runtime.enabled),
        pedestrian_caution_radius_m=float(runtime.pedestrian_caution_radius_m),
        capped_speed_m_s=float(runtime.capped_speed_m_s),
        ttc_veto_threshold_s=float(runtime.ttc_veto_threshold_s),
        clearance_veto_m=float(runtime.clearance_veto_m),
    )


def _xy_rows(value: Any) -> np.ndarray:
    """Return finite ``(n, 2)`` rows from simulator pedestrian state."""

    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("safety_wrapper simulator pedestrian positions must be finite")
    if arr.ndim == 1:
        if arr.size % 2:
            raise ValueError("safety_wrapper pedestrian positions require even-length flat xy data")
        rows = arr.reshape((-1, 2))
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        rows = arr[:, :2]
    else:
        raise ValueError("safety_wrapper pedestrian positions must be shaped (n,2) or (n,3)")
    return np.asarray(rows, dtype=float)


def _robot_position(env: Any) -> np.ndarray:
    simulator = getattr(env, "simulator", None)
    if simulator is None or not hasattr(simulator, "robot_pos"):
        raise ValueError("safety_wrapper requires simulator.robot_pos")
    robot_pos = simulator.robot_pos
    try:
        arr = np.asarray(robot_pos[0], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("safety_wrapper requires simulator.robot_pos[0] xy position") from exc
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        raise ValueError("safety_wrapper robot_pos must contain finite xy coordinates")
    return arr[:2]


def _pedestrian_positions(env: Any) -> np.ndarray:
    simulator = getattr(env, "simulator", None)
    return _xy_rows(getattr(simulator, "ped_pos", np.empty((0, 2), dtype=float)))


def _robot_heading(env: Any) -> float:
    simulator = getattr(env, "simulator", None)
    robots = getattr(simulator, "robots", None)
    if isinstance(robots, list) and robots:
        pose = getattr(robots[0], "pose", None)
        if pose is not None and len(pose) >= 3:
            return float(pose[2])
    heading = getattr(simulator, "robot_theta", None)
    if heading is not None:
        arr = np.asarray(heading, dtype=float).reshape(-1)
        if arr.size:
            return float(arr[0])
    return 0.0


def _radius(config: Any, *names: str, default: float) -> float:
    sim_config = getattr(config, "sim_config", None)
    for name in names:
        value = getattr(sim_config, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, float(value))
    return float(default)


def _robot_radius(config: Any) -> float:
    robot_config = getattr(config, "robot_config", None)
    robot_config_radius = getattr(robot_config, "radius", None)
    return _radius(
        config,
        "robot_radius",
        default=(
            float(robot_config_radius)
            if isinstance(robot_config_radius, (int, float))
            and math.isfinite(float(robot_config_radius))
            else 1.0
        ),
    )


def _pedestrian_velocities(
    current_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    dt: float,
) -> np.ndarray:
    if not dt > 0.0:
        raise ValueError("safety_wrapper dt must be positive for pre-step context")
    if previous_positions is None:
        return np.zeros_like(current_positions, dtype=float)
    previous = _xy_rows(previous_positions)
    velocities = np.zeros_like(current_positions, dtype=float)
    count = min(current_positions.shape[0], previous.shape[0])
    if count:
        velocities[:count] = (current_positions[:count] - previous[:count]) / float(dt)
    return velocities


def _min_positive_ttc(
    *,
    robot_pos: np.ndarray,
    robot_velocity: np.ndarray,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
) -> float | None:
    min_ttc: float | None = None
    for ped_pos, ped_velocity in zip(ped_positions, ped_velocities, strict=False):
        relative_pos = np.asarray(ped_pos, dtype=float) - robot_pos
        relative_velocity = np.asarray(ped_velocity, dtype=float) - robot_velocity
        speed_sq = float(np.dot(relative_velocity, relative_velocity))
        if speed_sq <= 1.0e-12:
            continue
        closing = float(np.dot(relative_pos, relative_velocity))
        if closing >= 0.0:
            continue
        ttc = -closing / speed_sq
        if ttc > 0.0 and (min_ttc is None or ttc < min_ttc):
            min_ttc = float(ttc)
    return min_ttc


def compute_safety_context_from_env(
    *,
    env: Any,
    config: Any,
    command: Sequence[float],
    previous_ped_positions: np.ndarray | None,
    dt: float,
) -> tuple[SafetyContext, dict[str, Any]]:
    """Build pre-step safety context from simulator state.

    Definitions:
    ``min_pedestrian_distance_m`` is the current center-to-center distance to
    the nearest pedestrian, or ``math.inf`` when no pedestrians exist.
    ``min_clearance_m`` is the one-step predicted surface clearance against
    pedestrians, subtracting robot and pedestrian radii. Obstacle samples are not
    consumed in this runtime slice, so provenance records pedestrians as the only
    clearance source.
    ``min_ttc_s`` is the minimum positive time to closest approach under the
    commanded robot velocity and finite-difference pedestrian velocity. It is
    ``None`` when relative motion is not closing or unavailable.

    Returns:
        Safety context plus provenance for the context calculation.
    """

    if not dt > 0.0:
        raise ValueError("safety_wrapper dt must be positive for pre-step context")
    robot_pos = _robot_position(env)
    ped_positions = _pedestrian_positions(env)
    if len(command) < 2:
        raise TypeError("safety_wrapper command must contain at least (linear, angular)")
    linear_velocity = float(command[0])
    heading = _robot_heading(env)
    robot_velocity = np.array(
        [linear_velocity * math.cos(heading), linear_velocity * math.sin(heading)],
        dtype=float,
    )
    ped_velocities = _pedestrian_velocities(ped_positions, previous_ped_positions, dt)
    robot_radius = _robot_radius(config)
    ped_radius = _radius(config, "ped_radius", "pedestrian_radius", default=0.4)
    radius_sum = robot_radius + ped_radius

    if ped_positions.size:
        distances = np.linalg.norm(ped_positions - robot_pos, axis=1)
        min_pedestrian_distance = float(np.min(distances))
        next_robot_pos = robot_pos + robot_velocity * float(dt)
        next_ped_positions = ped_positions + ped_velocities * float(dt)
        next_distances = np.linalg.norm(next_ped_positions - next_robot_pos, axis=1)
        min_clearance = float(np.min(next_distances) - radius_sum)
    else:
        min_pedestrian_distance = math.inf
        min_clearance = math.inf

    context = SafetyContext(
        min_pedestrian_distance_m=min_pedestrian_distance,
        min_clearance_m=min_clearance,
        min_ttc_s=_min_positive_ttc(
            robot_pos=robot_pos,
            robot_velocity=robot_velocity,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
        ),
    )
    provenance = {
        "context_source": "simulator_state_pre_step",
        "ttc_source": "finite_difference_pedestrian_velocity",
        "pedestrian_velocity_identity": "row_order_finite_difference_no_stable_ids",
        "clearance_sources": ["pedestrians"],
        "robot_radius_m": robot_radius,
        "pedestrian_radius_m": ped_radius,
        "pedestrian_count": int(ped_positions.shape[0]),
    }
    return context, provenance


def apply_runtime_safety_wrapper(
    *,
    command: Sequence[float],
    env: Any,
    config: Any,
    runtime: SafetyWrapperRuntimeConfig,
    previous_ped_positions: np.ndarray | None,
    step_idx: int,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Apply the opt-in wrapper to a two-component absolute command.

    Returns:
        Corrected ``(linear, angular)`` command plus schema-tagged step record.
    """

    context, context_provenance = compute_safety_context_from_env(
        env=env,
        config=config,
        command=command,
        previous_ped_positions=previous_ped_positions,
        dt=float(config.sim_config.time_per_step_in_secs),
    )
    wrapper_record = apply_safety_wrapper(
        float(command[0]),
        float(command[1]),
        context,
        safety_wrapper_config(runtime),
    )
    wrapper_schema_version = wrapper_record.pop("schema_version", None)
    step_record = {
        "schema_version": SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA,
        "wrapper_schema_version": wrapper_schema_version,
        "step": int(step_idx),
        "arm_key": str(runtime.arm_key),
        "enabled": bool(runtime.enabled),
        "eligible_for_wrapper": True,
        **context_provenance,
        **wrapper_record,
    }
    corrected = (
        float(wrapper_record["corrected_linear_velocity"]),
        float(wrapper_record["corrected_angular_velocity"]),
    )
    return corrected, step_record


def ineligible_safety_wrapper_step_record(
    *,
    runtime: SafetyWrapperRuntimeConfig,
    step_idx: int,
    reason: str,
) -> dict[str, Any]:
    """Return a schema-tagged wrapper trace record for fail-open ineligible steps."""

    return {
        "schema_version": SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA,
        "wrapper_schema_version": None,
        "step": int(step_idx),
        "arm_key": str(runtime.arm_key),
        "enabled": bool(runtime.enabled),
        "eligible_for_wrapper": False,
        "ineligible_reason": str(reason),
        "intervention": INTERVENTION_DISABLED,
        "intervened": False,
        "context": None,
    }


def summarize_safety_wrapper_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    runtime: SafetyWrapperRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Summarize per-step wrapper evidence for episode metadata and ledger provenance.

    Returns:
        Compact episode-level safety-wrapper intervention summary.
    """

    runtime = runtime or SafetyWrapperRuntimeConfig()
    counts = Counter(str(record.get("intervention", INTERVENTION_DISABLED)) for record in trace)
    intervened_records = [record for record in trace if bool(record.get("intervened", False))]
    speed_cap_steps = [
        int(record["step"])
        for record in trace
        if record.get("intervention") == INTERVENTION_SPEED_CAP and "step" in record
    ]
    hard_stop_steps = [
        int(record["step"])
        for record in trace
        if record.get("intervention") == INTERVENTION_HARD_STOP and "step" in record
    ]
    clearances = [
        float(record["context"]["min_clearance_m"])
        for record in trace
        if isinstance(record.get("context"), Mapping)
        and math.isfinite(float(record["context"]["min_clearance_m"]))
    ]
    ttcs = [
        float(record["context"]["min_ttc_s"])
        for record in trace
        if isinstance(record.get("context"), Mapping)
        and record["context"].get("min_ttc_s") is not None
        and math.isfinite(float(record["context"]["min_ttc_s"]))
    ]
    thresholds = asdict(safety_wrapper_config(runtime))
    step_count = len(trace)
    eligible_step_count = sum(
        1 for record in trace if bool(record.get("eligible_for_wrapper", True))
    )
    summary: dict[str, Any] = {
        "schema_version": SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA,
        "arm_key": str(runtime.arm_key),
        "enabled": bool(runtime.enabled),
        "thresholds_source": "predeclared_fixed_no_per_planner_tuning",
        "thresholds": thresholds,
        "false_stop_lookahead_s": float(runtime.false_stop_lookahead_s),
        "false_stop_analysis_supported": False,
        "step_count": int(step_count),
        "eligible_step_count": int(eligible_step_count),
        "intervention_counts": {
            INTERVENTION_DISABLED: int(counts.get(INTERVENTION_DISABLED, 0)),
            INTERVENTION_NONE: int(counts.get(INTERVENTION_NONE, 0)),
            INTERVENTION_SPEED_CAP: int(counts.get(INTERVENTION_SPEED_CAP, 0)),
            INTERVENTION_HARD_STOP: int(counts.get(INTERVENTION_HARD_STOP, 0)),
        },
        "intervened_step_count": len(intervened_records),
        "intervention_rate": float(len(intervened_records) / step_count) if step_count else 0.0,
        "first_intervention_step": (
            int(intervened_records[0]["step"]) if intervened_records else None
        ),
        "first_speed_cap_step": min(speed_cap_steps) if speed_cap_steps else None,
        "first_hard_stop_step": min(hard_stop_steps) if hard_stop_steps else None,
        "min_context_clearance_m": min(clearances) if clearances else None,
        "min_context_ttc_s": min(ttcs) if ttcs else None,
    }
    if runtime.record_step_trace:
        summary["step_trace"] = [dict(record) for record in trace]
    return summary


__all__ = [
    "SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA",
    "SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA",
    "SafetyWrapperRuntimeConfig",
    "apply_runtime_safety_wrapper",
    "compute_safety_context_from_env",
    "ineligible_safety_wrapper_step_record",
    "runtime_config_from_mapping",
    "safety_wrapper_config",
    "summarize_safety_wrapper_trace",
]
