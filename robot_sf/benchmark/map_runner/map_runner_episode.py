"""Episode execution helpers for map-based benchmark batches."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - runtime type-hint consumers resolve Path
from typing import Any, cast

import numpy as np
from loguru import logger

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
    resolve_learned_checkpoint_observation_contract,
)
from robot_sf.benchmark.ammv_feasibility import evaluate_artifact_command_feasibility
from robot_sf.benchmark.analysis_trace import (
    build_analysis_trace,
    telemetry_from_scenario,
)
from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.failure_mechanism_taxonomy import unknown_failure_mechanism_record
from robot_sf.benchmark.group_space_metrics import group_specs_from_map
from robot_sf.benchmark.interaction_exposure import (
    InteractionExposureError,
    compute_interaction_exposure_fields,
    not_derivable_interaction_exposure,
)
from robot_sf.benchmark.latency.latency_stress import (
    LatencyMeasurementHarness,
    LatencyStressProfile,
    not_available_latency_metrics,
)
from robot_sf.benchmark.map_runner.map_runner_env import (
    apply_active_observation_mode_to_env_config as _apply_active_observation_mode_to_env_config,
)
from robot_sf.benchmark.map_runner.map_runner_env import (
    apply_policy_env_observation_overrides as _apply_policy_env_observation_overrides,
)
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config as _build_env_config
from robot_sf.benchmark.map_runner.map_runner_env import (
    validate_sensor_fusion_adapter_config as _validate_sensor_fusion_adapter_config,
)
from robot_sf.benchmark.map_runner.map_runner_identity import (
    _compute_map_episode_id,
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.map_runner.map_runner_metrics import (
    floor_collision_metrics_from_flags as _floor_collision_metrics_from_flags,
)
from robot_sf.benchmark.map_runner.map_runner_metrics import (
    normalize_pedestrian_impact_controls as _normalize_pedestrian_impact_controls,
)
from robot_sf.benchmark.map_runner.map_runner_native_command import (
    native_command_metadata_for_record,
)
from robot_sf.benchmark.map_runner.map_runner_observations import (
    normalize_xy_rows as _normalize_xy_rows,
)
from robot_sf.benchmark.map_runner.map_runner_static_deadlock import (
    static_deadlock_trace_fields as _static_deadlock_trace_fields,
)
from robot_sf.benchmark.map_runner.map_runner_trace import (
    _command_action_payload,
    _cyclist_like_vru_summary,
    _episode_metadata_for_signal_metrics,
    _fast_bicycle_actor_summary,
    _intent_conditioned_behavior_summary,
    _observation_heading,
    _single_pedestrian_intent_metadata,
    _single_pedestrian_vru_metadata,
    _trace_pedestrians,
)
from robot_sf.benchmark.map_runner.map_runner_view_integrity import (
    DegeneratePlannerViewError,
    evaluate_effective_view_integrity,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    DEFAULT_KINEMATICS as _DEFAULT_KINEMATICS,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    policy_command_to_env_action as _policy_command_to_env_action,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    robot_kinematics_label as _robot_kinematics_label,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    robot_max_speed as _robot_max_speed,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    stack_ped_positions as _stack_ped_positions,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import vel_and_acc as _vel_and_acc
from robot_sf.benchmark.map_runner_policies.map_runner_policy_metadata import (
    finalize_feasibility_metadata as _finalize_feasibility_metadata,
)
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    _apply_planner_selector_v2_context,
    _apply_scenario_uncertainty_envelope_config,
    _parse_algo_config,
    _resolve_policy_search_candidate_runtime,
)
from robot_sf.benchmark.map_runner_policies.map_runner_profile_metadata import (
    load_latency_profile as _load_latency_stress_profile,
)
from robot_sf.benchmark.map_runner_policies.map_runner_profile_metadata import (
    load_synthetic_actuation_profile as _load_synthetic_actuation_profile,
)
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, post_process_metrics
from robot_sf.benchmark.observation_noise import (
    ObservationNoiseState,
    apply_observation_noise,
    make_observation_noise_rng,
    make_observation_noise_state,
    merge_observation_noise_stats,
    new_observation_noise_stats,
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.ped_model_sensitivity import (
    attach_pedestrian_model_fields,
    build_pedestrian_model_provenance,
)
from robot_sf.benchmark.pedestrian_control_trace import (
    attach_pedestrian_control_trace,
)
from robot_sf.benchmark.planner_command_contract import (
    validate_planner_contract as _validate_planner_contract,
)
from robot_sf.benchmark.public_requirement_events import evaluate_public_requirement_events
from robot_sf.benchmark.result_provenance import build_simulator_settings_provenance
from robot_sf.benchmark.safety.cbf_safety_filter_runtime import (
    CBFSafetyFilterRuntimeConfig,
    apply_runtime_cbf_safety_filter,
    ineligible_cbf_safety_filter_step_record,
    summarize_cbf_safety_filter_trace,
)
from robot_sf.benchmark.safety.cbf_safety_filter_runtime import (
    runtime_config_from_mapping as cbf_runtime_config_from_mapping,
)
from robot_sf.benchmark.safety.safety_predicates import (
    late_evasive_predicate,
    occlusion_near_miss_predicate,
    oscillatory_control_predicate,
)
from robot_sf.benchmark.safety.safety_wrapper_runtime import (
    SafetyWrapperRuntimeConfig,
    apply_runtime_safety_wrapper,
    ineligible_safety_wrapper_step_record,
    make_deadlock_recovery_monitor,
    runtime_config_from_mapping,
    summarize_safety_wrapper_trace,
)
from robot_sf.benchmark.synthetic_actuation import (
    SyntheticActuationController,
    SyntheticActuationProfile,
    not_available_saturation_metrics,
)
from robot_sf.benchmark.termination_reason import (
    build_outcome_payload,
    collision_event,
    outcome_contradictions,
    resolve_termination_reason,
    route_complete_success,
    status_from_termination_reason,
)
from robot_sf.benchmark.thresholds import ensure_metric_parameters
from robot_sf.benchmark.tracking_precision_contract import (
    apply_speed_contract,
    apply_tracking_precision_spec,
    make_tracking_precision_rng,
    minimum_separation,
    normalize_tracking_precision_spec,
    tracking_precision_hash,
)
from robot_sf.benchmark.types import (
    AlgoMeta,
    EpisodeRecordDict,
    NoiseConfig,
    NoiseSpec,
    PlannerDecisionTrace,
    PlannerDecisionTraceEntry,
    PlannerRuntime,
    TrackingPrecisionSpec,
)
from robot_sf.benchmark.utils import (
    _config_hash,
    _git_hash_fallback,
    attach_track_metadata,
    normalize_track_field,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig  # noqa: TC001
from robot_sf.planner.safety_shield import shield_metrics_from_stats
from robot_sf.robot.safety_wrapper import DeadlockRecoveryMonitor  # noqa: TC001

# Policy builders are migrated incrementally; the episode boundary narrows the
# legacy plain-dict metadata to ``AlgoMeta`` after enrichment.
PolicyBuilder = Callable[..., tuple[Any, AlgoMeta | dict[str, Any]]]
PedestrianControlTraceLabelBuilder = Callable[[int], list[dict[str, Any]]]
_OBSTACLE_FORCE_LAW_RUNTIME_RECORD_SCHEMA = "obstacle_force_law_runtime_record.v1"


@dataclass(frozen=True, slots=True)
class VisibilityEvidenceTrace:
    """Episode-level visibility evidence arrays consumed by safety predicates."""

    visibility: np.ndarray | None = None
    track_confidence: np.ndarray | None = None
    status: str = "unavailable"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _CollisionEventContext:
    """Immutable context for per-step collision-event typing."""

    dt_seconds: float
    map_def: Any
    robot_radius: float
    ped_radius: float


def _point_to_segment_distance(point: np.ndarray, segment: Any) -> float:
    """Return Euclidean distance from ``point`` to one line segment."""
    try:
        segment_arr = np.asarray(segment, dtype=float)
    except (TypeError, ValueError):
        return float("inf")
    if segment_arr.shape != (2, 2):
        return float("inf")
    start = segment_arr[0]
    end = segment_arr[1]
    line_vec = end - start
    denom = float(np.dot(line_vec, line_vec))
    if denom <= 0.0:
        return float(np.linalg.norm(point - start))
    t_val = float(np.dot(point - start, line_vec) / denom)
    closest = start + np.clip(t_val, 0.0, 1.0) * line_vec
    return float(np.linalg.norm(point - closest))


def _closest_segment_partner_id(
    point: np.ndarray,
    segments: list[Any],
    *,
    prefix: str,
) -> str | None:
    """Return a stable partner id for the nearest segment, when available."""
    if not segments:
        return None
    indexed_distances = [
        (_point_to_segment_distance(point, segment), index)
        for index, segment in enumerate(segments)
    ]
    finite = [(distance, index) for distance, index in indexed_distances if math.isfinite(distance)]
    if not finite:
        return None
    _, best_index = min(finite)
    return f"{prefix}:{best_index}"


def _map_obstacle_segments(map_def: Any) -> list[Any]:
    """Return flattened obstacle segments from the live map definition."""
    obstacles = getattr(map_def, "obstacles", None)
    if not isinstance(obstacles, list):
        return []
    segments: list[Any] = []
    for obstacle in obstacles:
        lines = getattr(obstacle, "lines", None)
        if isinstance(lines, list):
            segments.extend(lines)
    return segments


def _point_inside_map_bounds(point: np.ndarray, map_def: Any) -> bool:
    """Return whether ``point`` lies inside the declared rectangular map bounds."""
    width = getattr(map_def, "width", None)
    height = getattr(map_def, "height", None)
    if not isinstance(width, int | float) or not isinstance(height, int | float):
        return True
    if not math.isfinite(float(width)) or not math.isfinite(float(height)):
        return True
    return 0.0 <= float(point[0]) <= float(width) and 0.0 <= float(point[1]) <= float(height)


def _step_collision_events(
    *,
    step_idx: int,
    robot_pos: np.ndarray,
    previous_robot_pos: np.ndarray | None,
    ped_positions: np.ndarray,
    previous_ped_positions: np.ndarray | None,
    meta: Mapping[str, Any],
    context: _CollisionEventContext,
) -> list[dict[str, Any]]:
    """Return typed collision-event records for one simulator step."""
    events: list[dict[str, Any]] = []
    collision_time = float((step_idx + 1) * context.dt_seconds)
    if previous_robot_pos is not None and context.dt_seconds > 0.0:
        robot_velocity = (robot_pos - previous_robot_pos) / context.dt_seconds
    else:
        robot_velocity = np.zeros(2, dtype=float)

    if bool(meta.get("is_pedestrian_collision", False)):
        ped_array = np.asarray(ped_positions, dtype=float).reshape(-1, 2)
        partner_id: str | None = None
        relative_speed = float(np.linalg.norm(robot_velocity))
        # Filter non-finite pedestrian slots (padded/absent pedestrians) before
        # selecting the contact partner: np.argmin over a NaN-containing array
        # returns a NaN index, which would propagate NaN into
        # relative_speed_at_contact and violate the non-finite safety rule.
        finite_indices = (
            np.where(np.all(np.isfinite(ped_array), axis=1))[0]
            if ped_array.size
            else np.empty(0, dtype=int)
        )
        if finite_indices.size:
            finite_positions = ped_array[finite_indices]
            ped_distances = np.linalg.norm(finite_positions - robot_pos[np.newaxis, :], axis=1)
            contact_threshold = max(0.0, context.robot_radius + context.ped_radius) + 1.0e-6
            contact_candidates = np.where(ped_distances <= contact_threshold)[0]
            if contact_candidates.size:
                nearest = int(contact_candidates[np.argmin(ped_distances[contact_candidates])])
            else:
                nearest = int(np.argmin(ped_distances))
            ped_index = int(finite_indices[nearest])
            partner_id = str(ped_index)
            ped_velocity = np.zeros(2, dtype=float)
            if (
                previous_ped_positions is not None
                and previous_ped_positions.shape == ped_array.shape
                and context.dt_seconds > 0.0
                and np.all(np.isfinite(previous_ped_positions[ped_index]))
            ):
                ped_velocity = (
                    ped_array[ped_index] - previous_ped_positions[ped_index]
                ) / context.dt_seconds
            relative_speed = float(np.linalg.norm(robot_velocity - ped_velocity))
        events.append(
            {
                "collision_partner_type": "pedestrian",
                "collision_partner_id": partner_id,
                "collision_time": collision_time,
                "relative_speed_at_contact": relative_speed,
                "clearance_series_source": "runtime.step.pedestrian_positions",
                "exact_event_source": "runtime.step.meta.is_pedestrian_collision",
            }
        )

    if bool(meta.get("is_obstacle_collision", False)):
        bounds = list(getattr(context.map_def, "bounds", [])) if context.map_def is not None else []
        obstacle_segments = _map_obstacle_segments(context.map_def)
        in_bounds = (
            _point_inside_map_bounds(robot_pos, context.map_def)
            if context.map_def is not None
            else True
        )
        partner_type = "static_geometry" if in_bounds else "boundary"
        partner_type_override = str(meta.get("collision_partner_type") or "").strip()
        if partner_type_override in {"static_geometry", "boundary", "goal_artifact"}:
            partner_type = partner_type_override
        partner_id = (
            str(meta.get("collision_partner_id"))
            if meta.get("collision_partner_id") is not None
            else _closest_segment_partner_id(
                robot_pos,
                obstacle_segments if partner_type == "static_geometry" else bounds,
                prefix="obstacle" if partner_type == "static_geometry" else "boundary",
            )
        )
        events.append(
            {
                "collision_partner_type": partner_type,
                "collision_partner_id": partner_id,
                "collision_time": collision_time,
                "relative_speed_at_contact": float(np.linalg.norm(robot_velocity)),
                "clearance_series_source": (
                    "runtime.step.map.obstacles"
                    if partner_type in {"static_geometry", "goal_artifact"}
                    else "runtime.step.map.bounds"
                ),
                "exact_event_source": "runtime.step.meta.is_obstacle_collision",
            }
        )

    return events


def _nearest_hazard_distances(
    robot_pos_arr: np.ndarray,
    ped_pos_arr: np.ndarray,
) -> np.ndarray:
    """Return nearest pedestrian distance per episode step."""
    step_count = int(robot_pos_arr.shape[0])
    if step_count == 0:
        return np.asarray([], dtype=float)
    if ped_pos_arr.ndim < 3 or ped_pos_arr.shape[1] == 0:
        return np.full(step_count, 1.0e9, dtype=float)
    peds = ped_pos_arr[:step_count]
    robot = robot_pos_arr[:step_count, np.newaxis, :]
    return np.min(np.linalg.norm(peds - robot, axis=2), axis=1)


def _observed_pedestrian_positions(obs: Any) -> np.ndarray | None:
    """Return planner-facing pedestrian positions when the observation exposes them."""
    if not isinstance(obs, Mapping):
        return None
    pedestrians = obs.get("pedestrians")
    if isinstance(pedestrians, Mapping) and "positions" in pedestrians:
        return _normalize_xy_rows(pedestrians.get("positions"))
    if "pedestrian_positions" in obs:
        return _normalize_xy_rows(obs.get("pedestrian_positions"))
    if "ped_positions" in obs:
        return _normalize_xy_rows(obs.get("ped_positions"))
    return None


def _write_observed_pedestrian_positions(obs: Any, positions: np.ndarray) -> bool:
    """Update planner-facing pedestrian positions when observation exposes them.

    Returns:
        True when a supported observation position field was updated.
    """
    if not isinstance(obs, dict):
        return False
    pedestrians = obs.get("pedestrians")
    if isinstance(pedestrians, dict) and "positions" in pedestrians:
        pedestrians["positions"] = positions.tolist()
        pedestrians["count"] = int(positions.shape[0])
        return True
    if "pedestrian_positions" in obs:
        obs["pedestrian_positions"] = positions.tolist()
        return True
    if "ped_positions" in obs:
        obs["ped_positions"] = positions.tolist()
        return True
    return False


def _apply_tracking_precision_to_observation(
    obs: dict[str, Any],
    spec: TrackingPrecisionSpec,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Apply default-off MOTP drift mask to planner-facing tracked actors.

    Returns:
        Observation plus the planner-facing positions used for corrupted-distance metrics.
    """
    positions = _observed_pedestrian_positions(obs)
    if positions is None:
        return obs, None
    if not bool(spec.get("enabled", False)):
        return obs, positions
    corrupted = apply_tracking_precision_spec(
        positions,
        cast("dict[str, Any]", spec),
        rng,
    )
    _write_observed_pedestrian_positions(obs, corrupted)
    return obs, corrupted


def _visibility_evidence_for_step(
    *,
    peds: np.ndarray,
    obs: Any,
    config: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str | None]:
    """Match simulator pedestrians to planner-facing observations for trace labels.

    Returns:
        Tuple of visible mask, track-confidence values, evidence status, and reason.
    """
    peds = (
        np.asarray(peds, dtype=float).reshape(-1, 2) if np.asarray(peds).size else np.zeros((0, 2))
    )
    if peds.shape[0] == 0:
        return (
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=float),
            "not_applicable",
            "no_pedestrians",
        )

    settings = getattr(config, "observation_visibility", None)
    if settings is None or not bool(getattr(settings, "enabled", False)):
        return None, None, "not_applicable", "observation_visibility_disabled"

    observed = _observed_pedestrian_positions(obs)
    if observed is None:
        return None, None, "unavailable", "planner_observation_missing_pedestrian_positions"

    visible = np.zeros((peds.shape[0],), dtype=bool)
    if observed.shape[0] > 0:
        noise_std = float(getattr(settings, "tracking_noise_std_m", 0.0) or 0.0)
        match_tolerance_m = max(1.0e-4, 3.0 * noise_std + 1.0e-3)
        distances = np.linalg.norm(peds[:, np.newaxis, :] - observed[np.newaxis, :, :], axis=2)
        visible = np.min(distances, axis=1) <= match_tolerance_m
    confidence = visible.astype(float)
    return visible, confidence, "available", None


def _annotate_trace_visibility(
    pedestrians: list[dict[str, Any]],
    *,
    visible: np.ndarray | None,
    track_confidence: np.ndarray | None,
    evidence_status: str,
    evidence_reason: str | None,
) -> list[dict[str, Any]]:
    """Attach per-pedestrian visibility labels to trace frames.

    Returns:
        The same frame list with visibility fields attached to each pedestrian.
    """
    for idx, frame in enumerate(pedestrians):
        if visible is None or track_confidence is None:
            frame["visibility_state"] = evidence_status
            frame["track_confidence"] = None
        else:
            is_visible = bool(visible[idx]) if idx < visible.shape[0] else False
            frame["visibility_state"] = "visible" if is_visible else "occluded"
            frame["track_confidence"] = (
                float(track_confidence[idx]) if idx < track_confidence.shape[0] else 0.0
            )
        frame["visibility_evidence_status"] = evidence_status
        frame["visibility_evidence_reason"] = evidence_reason
    return pedestrians


def _stack_visibility_values(
    values: list[np.ndarray | None],
    *,
    fill_value: float,
    dtype: Any,
) -> np.ndarray | None:
    """Stack per-step pedestrian scalar labels, preserving missing-evidence state.

    Returns:
        ``(steps, pedestrians)`` array, or ``None`` when any step lacks evidence.
    """
    if not values or any(value is None for value in values):
        return None
    width = max((int(np.asarray(value).reshape(-1).shape[0]) for value in values), default=0)
    stacked = np.full((len(values), width), fill_value, dtype=dtype)
    for row_idx, value in enumerate(values):
        arr = np.asarray(value, dtype=dtype).reshape(-1)
        stacked[row_idx, : arr.shape[0]] = arr
    return stacked


def _nearest_hazard_visibility_signals(
    *,
    robot_pos_arr: np.ndarray,
    ped_pos_arr: np.ndarray,
    visibility_arr: np.ndarray | None,
    track_confidence_arr: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return nearest-pedestrian visibility/confidence signals for the predicate.

    Returns:
        Tuple of per-step nearest-hazard visibility and confidence arrays.
    """
    if visibility_arr is None or track_confidence_arr is None:
        return None, None
    step_count = min(int(robot_pos_arr.shape[0]), int(ped_pos_arr.shape[0]))
    if step_count == 0 or ped_pos_arr.ndim < 3 or ped_pos_arr.shape[1] == 0:
        return np.zeros((step_count,), dtype=bool), np.zeros((step_count,), dtype=float)
    peds = np.asarray(ped_pos_arr[:step_count], dtype=float)
    robot = np.asarray(robot_pos_arr[:step_count], dtype=float)[:, np.newaxis, :]
    distances = np.linalg.norm(peds - robot, axis=2)
    distances[~np.isfinite(distances)] = np.inf
    nearest = np.argmin(distances, axis=1)
    visible = np.zeros((step_count,), dtype=bool)
    confidence = np.zeros((step_count,), dtype=float)
    for step_idx, ped_idx in enumerate(nearest):
        if not np.isfinite(distances[step_idx, ped_idx]):
            continue
        if step_idx < visibility_arr.shape[0] and ped_idx < visibility_arr.shape[1]:
            visible[step_idx] = bool(visibility_arr[step_idx, ped_idx])
        if step_idx < track_confidence_arr.shape[0] and ped_idx < track_confidence_arr.shape[1]:
            confidence[step_idx] = float(track_confidence_arr[step_idx, ped_idx])
    return visible, confidence


def _safety_predicates_for_episode(
    *,
    robot_pos_arr: np.ndarray,
    robot_vel_arr: np.ndarray,
    robot_headings: list[float],
    ped_pos_arr: np.ndarray,
    dt: float,
    command_sources: list[str | None] | None = None,
    visibility_evidence: VisibilityEvidenceTrace | None = None,
) -> dict[str, dict[str, Any]]:
    """Build diagnostic safety predicate records for a completed episode.

    Returns:
        Mapping of ledger predicate keys to versioned predicate records.
    """
    step_count = min(len(robot_headings), int(robot_pos_arr.shape[0]))
    if step_count < 2 or not dt > 0.0:
        return {}

    positions = np.asarray(robot_pos_arr[:step_count], dtype=float)
    headings = np.asarray(robot_headings[:step_count], dtype=float)
    velocities = np.asarray(robot_vel_arr[:step_count], dtype=float)
    speeds = np.linalg.norm(velocities, axis=1) if velocities.size else np.zeros(step_count)
    hazard_distances = _nearest_hazard_distances(positions, ped_pos_arr)[:step_count]

    visibility_evidence = visibility_evidence or VisibilityEvidenceTrace()
    hazard_visible, track_confidence = _nearest_hazard_visibility_signals(
        robot_pos_arr=positions,
        ped_pos_arr=ped_pos_arr[:step_count],
        visibility_arr=visibility_evidence.visibility,
        track_confidence_arr=visibility_evidence.track_confidence,
    )

    # The late-evasive diagnostic measures latency from first hazard visibility and
    # always requires a concrete per-step visibility signal. When per-step occlusion
    # evidence is unavailable (the default map-runner path), retain the prior
    # all-visible assumption so this predicate keeps computing; only the
    # occlusion-near-miss predicate distinguishes unavailable visibility evidence.
    late_evasive_visible = (
        hazard_visible if hazard_visible is not None else np.ones(step_count, dtype=bool)
    )

    return {
        "oscillatory_control_predicate": oscillatory_control_predicate(
            positions,
            headings,
            speeds,
            dt=dt,
            command_sources=command_sources,
        ),
        "late_evasive_predicate": late_evasive_predicate(
            hazard_distances,
            late_evasive_visible,
            speeds,
            dt=dt,
        ),
        "occlusion_near_miss_predicate": occlusion_near_miss_predicate(
            hazard_distances,
            hazard_visible,
            track_confidence,
            speeds,
            dt=dt,
            visibility_evidence_status=visibility_evidence.status,
            visibility_evidence_reason=visibility_evidence.reason,
        ),
    }


def _episode_metadata_for_benchmark_metrics(
    scenario: dict[str, Any],
    map_def: Any,
) -> dict[str, Any] | None:
    """Merge signal-metric metadata with declared social-group geometry.

    Extends :func:`_episode_metadata_for_signal_metrics` with an additive
    ``social_groups`` payload sourced from the runtime map definition, so
    group-space intrusion metrics can be computed without changing the episode
    result schema. Returns ``None`` when neither signal metadata nor social
    groups are present, preserving existing default behavior.

    Returns:
        Optional merged episode metadata for benchmark metrics.
    """
    episode_metadata = _episode_metadata_for_signal_metrics(scenario) or {}
    group_specs = group_specs_from_map(map_def) if map_def is not None else []
    if group_specs:
        episode_metadata = deepcopy(episode_metadata)
        episode_metadata["social_groups"] = {
            "schema_version": "social-groups.v1",
            "groups": group_specs,
        }
    return episode_metadata or None


# Diagnostic interaction-exposure defaults for write-time instrumentation
# (issue #4242 AC #2). The 2.0 m radius mirrors the existing proxemic near/far
# split used by ``experimental_ped_impact_metrics`` so the writer is grounded in
# an existing repository convention rather than an arbitrary threshold. Both
# values are recorded on every emitted row so downstream tooling can override or
# re-derive without guessing which radius/threshold produced the value.
_INTERACTION_EXPOSURE_RADIUS_M = 2.0
_LOW_EXPOSURE_SUCCESS_THRESHOLD = 0.2


def _finite_pedestrian_frames(
    ped_pos_arr: np.ndarray,
    step_count: int,
) -> list[list[tuple[float, float]]]:
    """Convert a padded ``(T, K, 2)`` pedestrian tensor to per-step finite points.

    ``stack_ped_positions`` pads absent pedestrians with NaN; the interaction
    exposure helper rejects non-finite coordinates, so padding is dropped here
    and each frame is aligned to ``step_count`` to match the robot trace length.

    Returns:
        One list of finite ``(x, y)`` pedestrian points per step.
    """
    frames: list[list[tuple[float, float]]] = []
    peds = np.asarray(ped_pos_arr, dtype=float)
    if peds.ndim == 3 and peds.shape[0] >= 1:
        for frame in peds[:step_count]:
            frames.append(
                [
                    (float(px), float(py))
                    for px, py in frame
                    if math.isfinite(px) and math.isfinite(py)
                ]
            )
    if len(frames) < step_count:
        frames.extend([] for _ in range(step_count - len(frames)))
    return frames[:step_count]


def _episode_evidence_fields(
    *,
    robot_pos_arr: np.ndarray,
    ped_pos_arr: np.ndarray,
    dt: float,
    success: bool,
) -> dict[str, Any]:
    """Build native failure-mechanism and interaction-exposure schema blocks.

    Write-time instrumentation for issue #4242 AC #2. The blocks are attached to
    every map-runner episode record so new campaigns natively carry the
    ``failure_mechanism_taxonomy.v1`` and ``interaction_exposure.v1`` fields
    instead of omitting them.

    Fail-closed policy:

    - Failure mechanism is always ``unknown`` at write time. A single map-runner
      episode is not a paired-trace mechanism analysis, so no trace-verified
      label is asserted and geometry/scenario names are never substituted. A
      trace-verified label must come from the mechanism cross-cut path, not this
      writer.
    - Interaction exposure is computed from the episode's own recorded
      trajectory (its real trace, not imputation). When the trajectory support
      is missing or malformed, an explicit ``not_derivable`` block is emitted
      rather than fabricated zeros.

    Returns:
        Mapping with ``failure_mechanism`` and ``interaction_exposure`` blocks.
    """
    mechanism = unknown_failure_mechanism_record("not_derivable_from_single_episode_record")

    robot = np.asarray(robot_pos_arr, dtype=float)
    if robot.ndim != 2 or robot.shape[0] == 0 or robot.shape[1] != 2:
        exposure = not_derivable_interaction_exposure("not_derivable_missing_trace")
        return {"failure_mechanism": mechanism, "interaction_exposure": exposure}

    step_count = int(robot.shape[0])
    robot_frames = [(float(x), float(y)) for x, y in robot]
    ped_frames = _finite_pedestrian_frames(ped_pos_arr, step_count)
    try:
        exposure = compute_interaction_exposure_fields(
            robot_positions=robot_frames,
            pedestrian_positions=ped_frames,
            dt=float(dt),
            exposure_radius_m=_INTERACTION_EXPOSURE_RADIUS_M,
            low_exposure_success_threshold=_LOW_EXPOSURE_SUCCESS_THRESHOLD,
            success=bool(success),
        )
    except (InteractionExposureError, ValueError, TypeError):
        # Instrumentation must never break the episode writer; fail closed.
        exposure = not_derivable_interaction_exposure("not_derivable_missing_trace")
    return {"failure_mechanism": mechanism, "interaction_exposure": exposure}


def _finite_trace_float(value: Any) -> float | None:
    """Return a finite float for compact episode diagnostics."""
    if isinstance(value, int | float | np.integer | np.floating):
        candidate = float(value)
        if math.isfinite(candidate):
            return candidate
    return None


@dataclass(slots=True)
class _TopologyGuidedEpisodeAccumulator:
    """Mutable accumulator for compact topology-guided episode diagnostics."""

    topology_steps: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    fallback_reason_counts: Counter[str] = field(default_factory=Counter)
    no_candidate_reason_counts: Counter[str] = field(default_factory=Counter)
    selected_counts: Counter[str] = field(default_factory=Counter)
    near_parity_reason_counts: Counter[str] = field(default_factory=Counter)
    lane_status_counts: Counter[str] = field(default_factory=Counter)
    candidate_availability_status_counts: Counter[str] = field(default_factory=Counter)
    candidate_unavailable_reason_counts: Counter[str] = field(default_factory=Counter)
    candidate_outcome_counts: Counter[str] = field(default_factory=Counter)
    configured_fallback_steps: int = 0
    candidate_counts: list[int] = field(default_factory=list)
    route_progress_values: list[float] = field(default_factory=list)
    selected_sequence: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    topology_command_influence_steps: int = 0


def _update_topology_candidate_availability_fields(
    accumulator: _TopologyGuidedEpisodeAccumulator,
    candidate_availability: Any,
) -> None:
    """Fold explicit topology candidate availability into the episode accumulator."""
    if not isinstance(candidate_availability, dict):
        return
    candidate_status = str(candidate_availability.get("status", "unknown"))
    accumulator.candidate_availability_status_counts[candidate_status] += 1
    candidate_reason = candidate_availability.get("reason")
    if candidate_status != "available" and candidate_reason is not None:
        accumulator.candidate_unavailable_reason_counts[str(candidate_reason)] += 1
    candidate_outcome = str(candidate_availability.get("outcome", "unknown"))
    accumulator.candidate_outcome_counts[candidate_outcome] += 1
    if bool(candidate_availability.get("fallback_used")):
        accumulator.configured_fallback_steps += 1


def _update_topology_guided_episode_fields(
    accumulator: _TopologyGuidedEpisodeAccumulator,
    *,
    step: PlannerDecisionTraceEntry,
    topology: dict[str, Any],
) -> None:
    """Fold one topology-guided planner-step row into the episode accumulator."""
    accumulator.topology_steps += 1
    status = str(topology.get("status", "unknown"))
    accumulator.status_counts[status] += 1

    reason = topology.get("reason")
    if status != "ok" and reason is not None:
        accumulator.no_candidate_reason_counts[str(reason)] += 1
    fallback_reason = step.get("topology_fallback_reason")
    if fallback_reason is not None:
        accumulator.fallback_reason_counts[str(fallback_reason)] += 1

    lane_status = step.get("topology_lane_status")
    if lane_status is not None:
        accumulator.lane_status_counts[str(lane_status)] += 1
    _update_topology_candidate_availability_fields(
        accumulator, step.get("topology_candidate_availability")
    )
    count = topology.get("hypothesis_count")
    if isinstance(count, int | np.integer):
        accumulator.candidate_counts.append(int(count))

    selected = topology.get("selected_hypothesis_id")
    if selected is not None:
        selected_key = str(selected)
        accumulator.selected_counts[selected_key] += 1
        accumulator.selected_sequence.append(selected_key)
    near_parity_reason = topology.get("near_parity_gate_reason")
    if near_parity_reason is not None:
        accumulator.near_parity_reason_counts[str(near_parity_reason)] += 1

    progress = _finite_trace_float(step.get("route_progress_from_start_m"))
    if progress is not None:
        accumulator.route_progress_values.append(progress)
    step_config = step.get("topology_guided_config")
    if isinstance(step_config, dict):
        accumulator.config.update(step_config)
    if isinstance(step.get("topology_command_influence"), dict):
        accumulator.topology_command_influence_steps += 1


def _collect_topology_guided_episode_fields(
    planner_decision_trace: list[PlannerDecisionTraceEntry],
) -> _TopologyGuidedEpisodeAccumulator | None:
    """Collect topology-guided fields from reduced planner-step rows.

    Returns:
        Accumulated topology fields, or ``None`` when the trace contains no topology lane rows.
    """
    accumulator = _TopologyGuidedEpisodeAccumulator()
    for step in planner_decision_trace:
        topology = step.get("topology_guided")
        if not isinstance(topology, dict):
            continue
        _update_topology_guided_episode_fields(accumulator, step=step, topology=topology)

    if accumulator.topology_steps == 0:
        return None
    return accumulator


def _topology_route_progress_summary(
    *,
    route_progress_values: list[float],
    selected_switch_count: int,
    min_progress_delta: float,
    stall_window_steps: int,
    fallback_only: bool,
) -> dict[str, Any]:
    """Summarize route progress and classify terminal stall/progress reason.

    Returns:
        Route-progress fields for the topology-guided episode diagnostic block.
    """
    stagnant_steps = 0
    max_stagnant_steps = 0
    previous_progress: float | None = None
    for progress in route_progress_values:
        if previous_progress is None or progress - previous_progress >= min_progress_delta:
            stagnant_steps = 0
        else:
            stagnant_steps += 1
            max_stagnant_steps = max(max_stagnant_steps, stagnant_steps)
        previous_progress = progress

    route_progress_delta = (
        route_progress_values[-1] - route_progress_values[0]
        if len(route_progress_values) >= 2
        else 0.0
    )
    if fallback_only:
        terminal_reason = "fallback_only"
    elif route_progress_delta >= min_progress_delta:
        terminal_reason = "goal_progress"
    elif max_stagnant_steps >= stall_window_steps and selected_switch_count > 0:
        terminal_reason = "near_parity_churn"
    elif max_stagnant_steps >= stall_window_steps:
        terminal_reason = "true_stall"
    else:
        terminal_reason = "no_stall_observed"

    return {
        "observed_steps": len(route_progress_values),
        "initial_m": route_progress_values[0] if route_progress_values else None,
        "final_m": route_progress_values[-1] if route_progress_values else None,
        "delta_m": float(route_progress_delta),
        "min_progress_delta_m": float(min_progress_delta),
        "stall_window_steps": int(stall_window_steps),
        "max_stagnant_steps": int(max_stagnant_steps),
        "terminal_reason": terminal_reason,
    }


def _topology_guided_episode_diagnostics(
    planner_decision_trace: list[PlannerDecisionTraceEntry],
) -> dict[str, Any] | None:
    """Aggregate topology-guided lane diagnostics from reduced planner-step rows.

    The block is diagnostic-only by construction: fallback-only operation remains explicit and
    cannot be confused with benchmark-strength topology-lane success.

    Returns:
        Compact episode-level topology diagnostics, or ``None`` when no topology rows exist.
    """
    accumulator = _collect_topology_guided_episode_fields(planner_decision_trace)
    if accumulator is None:
        return None
    selected_switch_count = sum(
        1
        for previous, current in zip(
            accumulator.selected_sequence, accumulator.selected_sequence[1:], strict=False
        )
        if previous != current
    )
    min_progress_delta = _finite_trace_float(accumulator.config.get("min_route_progress_delta_m"))
    if min_progress_delta is None:
        min_progress_delta = 0.05
    stall_window_steps = int(accumulator.config.get("stall_window_steps", 20) or 20)
    fallback_steps = sum(
        count for status, count in accumulator.status_counts.items() if status != "ok"
    )
    fallback_used = fallback_steps > 0 or bool(
        accumulator.lane_status_counts.get("fallback_only", 0)
    )
    fallback_only = accumulator.topology_steps == fallback_steps or (
        accumulator.lane_status_counts.get("fallback_only", 0) == accumulator.topology_steps
    )
    route_progress = _topology_route_progress_summary(
        route_progress_values=accumulator.route_progress_values,
        selected_switch_count=selected_switch_count,
        min_progress_delta=min_progress_delta,
        stall_window_steps=stall_window_steps,
        fallback_only=fallback_only,
    )

    return {
        "schema_version": "topology-guided-episode-diagnostics.v1",
        "claim_boundary": str(accumulator.config.get("claim_boundary", "diagnostic_only")),
        "diagnostic_only": bool(accumulator.config.get("diagnostic_only", True)),
        "hypothesis_available": bool(accumulator.status_counts.get("ok", 0)),
        "hypothesis_available_steps": int(accumulator.status_counts.get("ok", 0)),
        "fallback_used": bool(fallback_used),
        "fallback_steps": int(fallback_steps),
        "status_counts": dict(sorted(accumulator.status_counts.items())),
        "lane_status_counts": dict(sorted(accumulator.lane_status_counts.items())),
        "candidate_availability_status_counts": dict(
            sorted(accumulator.candidate_availability_status_counts.items())
        ),
        "candidate_unavailable_reasons": dict(
            sorted(accumulator.candidate_unavailable_reason_counts.items())
        ),
        "candidate_outcome_counts": dict(sorted(accumulator.candidate_outcome_counts.items())),
        "configured_fallback_steps": int(accumulator.configured_fallback_steps),
        "no_candidate_reasons": dict(sorted(accumulator.no_candidate_reason_counts.items())),
        "fallback_reasons": dict(sorted(accumulator.fallback_reason_counts.items())),
        "candidate_counts": {
            "observed_steps": len(accumulator.candidate_counts),
            "min": min(accumulator.candidate_counts) if accumulator.candidate_counts else None,
            "max": max(accumulator.candidate_counts) if accumulator.candidate_counts else None,
            "last": accumulator.candidate_counts[-1] if accumulator.candidate_counts else None,
        },
        "selected_candidate_counts": dict(sorted(accumulator.selected_counts.items())),
        "selected_candidate_switch_count": int(selected_switch_count),
        "topology_command_influence_steps": int(accumulator.topology_command_influence_steps),
        "arbitration_weight": _finite_trace_float(accumulator.config.get("arbitration_weight")),
        "near_parity_margin": _finite_trace_float(
            accumulator.config.get(
                "near_parity_margin",
                accumulator.config.get("near_parity_route_distance_slack_ratio"),
            )
        ),
        "near_parity_gate_reason_counts": dict(
            sorted(accumulator.near_parity_reason_counts.items())
        ),
        "route_progress": route_progress,
    }


def _apply_safety_wrapper_step(
    command: Any,
    *,
    runtime: Any,
    env: Any,
    config: Any,
    step_idx: int,
    step_is_native: bool,
    previous_ped_positions: np.ndarray | None,
    deadlock_monitor: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run one safety-wrapper correction or record an ineligible step.

    Error/fallback path for ``safety_wrapper``: native actions and unsupported command
    shapes either raise (when the runtime is configured to fail closed) or emit an
    ineligible step record; otherwise the runtime corrects the command in place.

    Returns:
        tuple[Any, dict[str, Any]]: ``(command, record)`` where ``command`` is the
        corrected command (tail preserved) on the applied path, or the unchanged
        command on an ineligible path; ``record`` is appended to the wrapper trace.
    """
    if step_is_native:
        if runtime.fail_on_native_action:
            raise ValueError(
                "safety_wrapper.enabled requires absolute commands; "
                "native environment actions cannot be wrapped safely"
            )
        return command, ineligible_safety_wrapper_step_record(
            runtime=runtime,
            step_idx=step_idx,
            reason="native_environment_action",
        )
    if not isinstance(command, (tuple, list, np.ndarray)) or len(command) < 2:
        if runtime.fail_on_unsupported_command:
            raise TypeError(
                "safety_wrapper.enabled expects commands shaped like "
                "(linear_velocity, angular_velocity)"
            )
        return command, ineligible_safety_wrapper_step_record(
            runtime=runtime,
            step_idx=step_idx,
            reason="unsupported_command_shape",
        )
    corrected_command, wrapper_record = apply_runtime_safety_wrapper(
        command=command,
        env=env,
        config=config,
        runtime=runtime,
        previous_ped_positions=previous_ped_positions,
        step_idx=step_idx,
        deadlock_monitor=deadlock_monitor,
    )
    corrected = (
        corrected_command[0],
        corrected_command[1],
        *tuple(command[2:]),
    )
    return corrected, wrapper_record


def _apply_cbf_safety_filter_step(
    command: Any,
    *,
    runtime: Any,
    env: Any,
    config: Any,
    step_idx: int,
    step_is_native: bool,
    previous_ped_positions: np.ndarray | None,
) -> tuple[Any, dict[str, Any]]:
    """Run one CBF safety-filter correction or record an ineligible step.

    Error/fallback path for ``cbf_safety_filter``: native actions and unsupported
    command shapes either raise (when the runtime is configured to fail closed) or
    emit an ineligible step record; otherwise the CBF filter corrects the command.

    Returns:
        tuple[Any, dict[str, Any]]: ``(command, record)`` where ``command`` is the
        corrected command (tail preserved) on the applied path, or the unchanged
        command on an ineligible path; ``record`` is appended to the filter trace.
    """
    if step_is_native:
        if runtime.fail_on_native_action:
            raise ValueError(
                "cbf_safety_filter.enabled requires absolute commands; "
                "native environment actions cannot be filtered safely"
            )
        return command, ineligible_cbf_safety_filter_step_record(
            runtime=runtime,
            step_idx=step_idx,
            reason="native_environment_action",
        )
    if not isinstance(command, (tuple, list, np.ndarray)) or len(command) < 2:
        if runtime.fail_on_unsupported_command:
            raise TypeError(
                "cbf_safety_filter.enabled expects commands shaped like "
                "(linear_velocity, angular_velocity)"
            )
        return command, ineligible_cbf_safety_filter_step_record(
            runtime=runtime,
            step_idx=step_idx,
            reason="unsupported_command_shape",
        )
    corrected_command, cbf_record = apply_runtime_cbf_safety_filter(
        command=command,
        env=env,
        config=config,
        runtime=runtime,
        previous_ped_positions=previous_ped_positions,
        step_idx=step_idx,
    )
    corrected = (
        corrected_command[0],
        corrected_command[1],
        *tuple(command[2:]),
    )
    return corrected, cbf_record


def _min_finite_or_inf(values: list[float]) -> float:
    """Return the minimum finite value, falling back to ``+inf`` when none are finite.

    Non-finite entries (NaN, +inf, -inf) are filtered out so a stray NaN in the
    per-step separation stream cannot produce order-dependent, non-deterministic
    results from ``min()``. An empty or all-non-finite list yields ``+inf``.

    Args:
        values: Per-step float measurements (may contain non-finite values).

    Returns:
        float: The minimum finite value, or ``float("inf")`` when no finite value exists.
    """
    finite = [v for v in values if math.isfinite(v)]
    return float(min(finite)) if finite else float("inf")


def _build_tracking_precision_summary(
    *,
    spec: TrackingPrecisionSpec,
    records: list[dict[str, Any]],
    min_separation_corrupted_values: list[float],
) -> dict[str, Any]:
    """Build the tracking-precision summary block for episode algorithm metadata.

    Args:
        spec: Normalized tracking-precision spec.
        records: Per-step tracking-precision records emitted during the episode loop.
        min_separation_corrupted_values: Per-step min robot-ped separation under corrupted obs.

    Returns:
        dict[str, Any]: Tracking-precision summary with contract-honored rates and the
        last step record (when present).
    """
    summary: dict[str, Any] = {
        "spec": spec,
        "hash": tracking_precision_hash(cast("dict[str, Any]", spec)),
        "step_count": len(records),
        "min_separation_corrupted_m": _min_finite_or_inf(min_separation_corrupted_values),
        "contract_honored": (
            all(bool(record.get("contract_honored", False)) for record in records)
            if records
            else True
        ),
        "contract_honored_rate": (
            float(sum(bool(record.get("contract_honored", False)) for record in records))
            / float(len(records))
            if records
            else 1.0
        ),
    }
    if records:
        summary["last_step"] = dict(records[-1])
    return summary


@dataclass(frozen=True, slots=True)
class _EpisodeRunContext:
    """Resolved inputs and runtime config for one episode run.

    Bundles the normalization/env-config/horizon/profile/policy-cfg resolution phase of
    ``run_map_episode`` so the episode loop and metadata assembly receive a single
    immutable context object instead of recomputing the same locals inline.
    """

    scenario: dict[str, Any]
    scenario_id: str
    ts_start: str
    start_time: float
    ped_impact_radius_m: float
    ped_impact_window_steps: int
    benchmark_track: str | None
    track_schema_version: str | None
    noise_spec: NoiseSpec
    noise_rng: np.random.Generator
    noise_state: ObservationNoiseState
    noise_stats: dict[str, int]
    tracking_precision_spec: TrackingPrecisionSpec
    tracking_precision_rng: np.random.Generator
    safety_wrapper_runtime: SafetyWrapperRuntimeConfig
    cbf_runtime: CBFSafetyFilterRuntimeConfig
    safety_wrapper_deadlock_monitor: DeadlockRecoveryMonitor | None
    config: RobotSimulationConfig
    horizon_val: int
    robot_kinematics: str
    robot_command_mode: str
    actuation_profile: SyntheticActuationProfile | None
    latency_profile: LatencyStressProfile | None
    algo: str
    policy_cfg: dict[str, Any]


def _resolve_episode_run_context(  # noqa: PLR0913
    *,
    scenario: dict[str, Any],
    seed: int,
    horizon: int | None,
    dt: float | None,
    algo: str,
    scenario_path: Path,
    algo_config: dict[str, Any] | None,
    algo_config_path: str | None,
    experimental_ped_impact: bool,
    ped_impact_radius_m: float,
    ped_impact_window_steps: int,
    observation_mode: str | None,
    observation_level: str | None,
    benchmark_track: str | None,
    track_schema_version: str | None,
    observation_noise: dict[str, Any] | None,
    tracking_precision: dict[str, Any] | None,
    synthetic_actuation_profile: dict[str, Any] | None,
    latency_stress_profile: dict[str, Any] | None,
    safety_wrapper: dict[str, Any] | None,
    cbf_safety_filter: dict[str, Any] | None,
) -> _EpisodeRunContext:
    """Normalize episode inputs, build the env config, and resolve the policy cfg.

    Returns:
        _EpisodeRunContext: Immutable bundle of resolved scenario/track/noise/profile/
        kinematics/policy-cfg values consumed by the rest of ``run_map_episode``.
    """
    ped_impact_radius_m, ped_impact_window_steps = _normalize_pedestrian_impact_controls(
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
    )
    ts_start = datetime.now(UTC).isoformat()
    start_time = time.time()
    scenario = _scenario_with_episode_seed_defaults(scenario, seed=seed)
    scenario_id = str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )
    benchmark_track = normalize_track_field(benchmark_track, field_name="benchmark_track")
    track_schema_version = normalize_track_field(
        track_schema_version,
        field_name="track_schema_version",
    )
    noise_spec = cast("NoiseSpec", normalize_observation_noise_spec(observation_noise))
    noise_rng = make_observation_noise_rng(
        cast("dict[str, Any]", noise_spec), seed=seed, scenario_id=scenario_id
    )
    noise_state = make_observation_noise_state(cast("dict[str, Any]", noise_spec))
    noise_stats = new_observation_noise_stats()
    tracking_precision_spec = cast(
        "TrackingPrecisionSpec",
        normalize_tracking_precision_spec(tracking_precision),
    )
    tracking_precision_rng = make_tracking_precision_rng(
        cast("dict[str, Any]", tracking_precision_spec),
        seed=seed,
        scenario_id=scenario_id,
    )
    safety_wrapper_runtime = runtime_config_from_mapping(safety_wrapper)
    cbf_runtime = cbf_runtime_config_from_mapping(cbf_safety_filter)
    if safety_wrapper_runtime.enabled and cbf_runtime.enabled:
        raise ValueError(
            "safety_wrapper and cbf_safety_filter cannot both be enabled in #3948 first slice"
        )
    safety_wrapper_deadlock_monitor = make_deadlock_recovery_monitor(safety_wrapper_runtime)
    config = _build_env_config(scenario, scenario_path=scenario_path)
    max_steps = int(scenario.get("simulation_config", {}).get("max_episode_steps", 0) or 0)
    horizon_val = int(horizon) if horizon and horizon > 0 else max_steps
    if horizon_val <= 0:
        horizon_val = 200
    if dt is not None and dt > 0:
        config.sim_config.time_per_step_in_secs = float(dt)

    robot_kinematics = _robot_kinematics_label(config)
    actuation_profile = _load_synthetic_actuation_profile(synthetic_actuation_profile)
    latency_profile = _load_latency_stress_profile(latency_stress_profile)
    if actuation_profile is not None and robot_kinematics != _DEFAULT_KINEMATICS:
        raise ValueError(
            "synthetic_actuation_profile requires differential_drive scenarios; "
            f"got {robot_kinematics!r} for scenario {scenario_id!r}"
        )
    if (
        latency_profile is not None
        and latency_profile.action_delay_steps > 0
        and robot_kinematics != _DEFAULT_KINEMATICS
    ):
        raise ValueError(
            "latency_stress_profile.action_delay_steps requires differential_drive scenarios; "
            f"got {robot_kinematics!r} for scenario {scenario_id!r}"
        )
    robot_command_mode = (
        str(getattr(getattr(config, "robot_config", None), "command_mode", "vx_vy")).strip().lower()
    )
    raw_policy_cfg = (
        dict(algo_config) if algo_config is not None else _parse_algo_config(algo_config_path)
    )
    algo, policy_cfg = _resolve_policy_search_candidate_runtime(
        default_algo=algo,
        algo_config_path=algo_config_path,
        algo_config=raw_policy_cfg,
        scenario=scenario,
    )
    policy_cfg = _apply_planner_selector_v2_context(
        algo,
        policy_cfg,
        scenario=scenario,
        seed=int(seed),
    )
    policy_cfg = _apply_scenario_uncertainty_envelope_config(algo, policy_cfg, scenario)
    return _EpisodeRunContext(
        scenario=scenario,
        scenario_id=scenario_id,
        ts_start=ts_start,
        start_time=start_time,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        noise_spec=noise_spec,
        noise_rng=noise_rng,
        noise_state=noise_state,
        noise_stats=noise_stats,
        tracking_precision_spec=tracking_precision_spec,
        tracking_precision_rng=tracking_precision_rng,
        safety_wrapper_runtime=safety_wrapper_runtime,
        cbf_runtime=cbf_runtime,
        safety_wrapper_deadlock_monitor=safety_wrapper_deadlock_monitor,
        config=config,
        horizon_val=horizon_val,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
        actuation_profile=actuation_profile,
        latency_profile=latency_profile,
        algo=algo,
        policy_cfg=policy_cfg,
    )


@dataclass(frozen=True, slots=True)
class _EpisodePostLoopResult:
    """Trajectory arrays and raw metrics computed after the episode step loop.

    Bundles the post-loop phase of ``run_map_episode``: trajectory stacking, visibility
    evidence reduction, safety predicates, obstacle sampling, and raw metric computation.
    """

    robot_pos_arr: np.ndarray
    robot_vel_arr: np.ndarray
    robot_acc_arr: np.ndarray
    ped_pos_arr: np.ndarray
    ped_forces_arr: np.ndarray
    safety_predicates: dict[str, Any]
    obstacles: Any
    shortest_path: float
    metrics_raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _MetadataFinalizationOptions:
    """Runtime options consumed while finalizing an episode record."""

    actuation_controller: Any
    active_observation_mode: str
    active_observation_level: str
    single_pedestrian_intent_metadata: Any
    single_pedestrian_vru_metadata: Any
    record_forces: bool
    record_planner_decision_trace: bool
    record_simulation_step_trace: bool


def _compute_post_loop_metrics(  # noqa: PLR0913
    *,
    robot_positions: list[np.ndarray],
    robot_headings: list[float],
    hybrid_command_sources: list[str | None] | None = None,
    ped_positions: list[np.ndarray],
    ped_forces: list[np.ndarray],
    visibility_trace: list[np.ndarray | None],
    track_confidence_trace: list[np.ndarray | None],
    visibility_evidence_statuses: list[str],
    visibility_evidence_reasons: list[str | None],
    reached_goal_step: int | None,
    collision_seen: bool,
    ped_collision_seen: bool,
    obstacle_collision_seen: bool,
    robot_collision_seen: bool,
    map_def: Any,
    goal_vec: np.ndarray,
    scenario: dict[str, Any],
    config: Any,
    horizon_val: int,
    record_forces: bool,
    experimental_ped_impact: bool,
    ped_impact_radius_m: float,
    ped_impact_window_steps: int,
) -> _EpisodePostLoopResult:
    """Stack the episode trajectory, derive safety predicates, and compute raw metrics.

    Returns:
        _EpisodePostLoopResult: Trajectory arrays plus safety predicates, sampled
        obstacles, shortest-path length, and the raw (pre-post-process) metrics dict.
    """
    robot_pos_arr = np.asarray(robot_positions, dtype=float)
    robot_vel_arr, robot_acc_arr = _vel_and_acc(
        robot_pos_arr, config.sim_config.time_per_step_in_secs
    )
    ped_pos_arr = _stack_ped_positions(ped_positions)
    ped_forces_arr = (
        _stack_ped_positions(ped_forces, fill_value=np.nan)
        if record_forces
        else np.zeros_like(ped_pos_arr, dtype=float)
    )
    visibility_arr = _stack_visibility_values(
        visibility_trace,
        fill_value=False,
        dtype=bool,
    )
    track_confidence_arr = _stack_visibility_values(
        track_confidence_trace,
        fill_value=0.0,
        dtype=float,
    )
    if "unavailable" in visibility_evidence_statuses:
        visibility_evidence_status = "unavailable"
    elif visibility_evidence_statuses and all(
        status == "not_applicable" for status in visibility_evidence_statuses
    ):
        visibility_evidence_status = "not_applicable"
    else:
        visibility_evidence_status = "available"
    visibility_evidence_reason = next(
        (reason for reason in visibility_evidence_reasons if reason),
        None,
    )
    safety_predicates = _safety_predicates_for_episode(
        robot_pos_arr=robot_pos_arr,
        robot_vel_arr=robot_vel_arr,
        robot_headings=robot_headings,
        ped_pos_arr=ped_pos_arr,
        dt=float(config.sim_config.time_per_step_in_secs),
        command_sources=hybrid_command_sources,
        visibility_evidence=VisibilityEvidenceTrace(
            visibility=visibility_arr,
            track_confidence=track_confidence_arr,
            status=visibility_evidence_status,
            reason=visibility_evidence_reason,
        ),
    )

    obstacles = (
        sample_obstacle_points(map_def.obstacles, map_def.bounds) if map_def is not None else None
    )
    if robot_pos_arr.size:
        shortest_path = compute_shortest_path_length(map_def, robot_pos_arr[0], goal_vec)
    else:
        shortest_path = float("nan")

    if robot_pos_arr.size == 0:
        metrics_raw = {
            "success": 0.0,
            "time_to_goal_norm": float("nan"),
            "collisions": 0.0,
        }
    else:
        robot_config = getattr(config, "robot_config", None)
        ep = EpisodeData(
            robot_pos=robot_pos_arr,
            robot_vel=robot_vel_arr,
            robot_acc=robot_acc_arr,
            peds_pos=ped_pos_arr,
            ped_forces=ped_forces_arr,
            obstacles=obstacles,
            goal=goal_vec,
            dt=float(config.sim_config.time_per_step_in_secs),
            reached_goal_step=reached_goal_step,
            robot_radius=float(getattr(robot_config, "radius", 1.0)),
            ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),
            episode_metadata=_episode_metadata_for_benchmark_metrics(scenario, map_def),
        )
        metrics_raw = compute_all_metrics(
            ep,
            horizon=horizon_val,
            shortest_path_len=shortest_path,
            robot_max_speed=_robot_max_speed(config),
            experimental_ped_impact=experimental_ped_impact,
            ped_impact_radius_m=ped_impact_radius_m,
            ped_impact_window_steps=ped_impact_window_steps,
        )
    _floor_collision_metrics_from_flags(
        metrics_raw,
        collision_seen=collision_seen,
        ped_collision_seen=ped_collision_seen,
        obstacle_collision_seen=obstacle_collision_seen,
        robot_collision_seen=robot_collision_seen,
    )
    return _EpisodePostLoopResult(
        robot_pos_arr=robot_pos_arr,
        robot_vel_arr=robot_vel_arr,
        robot_acc_arr=robot_acc_arr,
        ped_pos_arr=ped_pos_arr,
        ped_forces_arr=ped_forces_arr,
        safety_predicates=safety_predicates,
        obstacles=obstacles,
        shortest_path=shortest_path,
        metrics_raw=metrics_raw,
    )


@dataclass(frozen=True, slots=True)
class _PolicyContract:
    """Resolved policy callable, planner lifecycle hooks, and observation contract.

    Bundles the policy/observation-contract preparation phase of ``run_map_episode``
    so the step-loop and metadata-finalization phases receive a single immutable
    object instead of recomputing these inline.
    """

    policy_fn: Callable[..., Any]
    algo_meta: AlgoMeta
    planner_close: Callable[..., Any] | None
    planner_reset: Callable[..., Any] | None
    planner_bind_env: Callable[..., Any] | None
    planner_stats: Callable[..., Any] | None
    planner_native_action: bool
    actuation_controller: SyntheticActuationController | None
    active_observation_mode: str
    active_observation_level: str
    single_pedestrian_intent_metadata: list[dict[str, Any] | None]
    single_pedestrian_vru_metadata: list[dict[str, Any] | None]


def _prepare_policy_and_observation_contract(  # noqa: PLR0913
    *,
    scenario: dict[str, Any],
    algo: str,
    policy_cfg: dict[str, Any],
    config: RobotSimulationConfig,
    observation_mode: str | None,
    observation_level: str | None,
    robot_kinematics: str,
    robot_command_mode: str,
    adapter_impact_eval: bool,
    benchmark_track: str | None,
    track_schema_version: str | None,
    actuation_profile: SyntheticActuationProfile | None,
    policy_builder: PolicyBuilder,
) -> _PolicyContract:
    """Resolve the learned observation contract, build the policy, and derive hooks.

    Returns:
        _PolicyContract: Immutable bundle of the policy callable, enriched algorithm
        metadata, planner lifecycle hooks, the synthetic-actuation controller, and the
        active observation mode/level plus single-pedestrian intent/VRU metadata.
    """
    learned_observation_contract = resolve_learned_checkpoint_observation_contract(
        algo,
        policy_cfg,
        observation_mode=observation_mode,
        observation_level=observation_level,
    )
    active_observation_mode = str(learned_observation_contract["active_observation_mode"])
    resolved_observation_level = observation_level
    if resolved_observation_level is None:
        resolved_observation_level = learned_observation_contract.get("observation_level_key")
    _apply_active_observation_mode_to_env_config(
        config,
        active_observation_mode=active_observation_mode,
    )
    _apply_policy_env_observation_overrides(config, policy_cfg)
    _validate_sensor_fusion_adapter_config(
        algo=algo,
        active_observation_mode=active_observation_mode,
        algo_config=policy_cfg,
    )
    _validate_planner_contract(
        algo=algo,
        robot_kinematics=robot_kinematics,
        algo_config=policy_cfg,
        observation_mode=active_observation_mode,
        observation_level=observation_level,
    )
    extra_kwargs = {}
    if algo.lower().strip() == "native_command":
        sim_cfg = getattr(config, "sim_config", None)
        extra_kwargs = {
            "scenario_id": scenario.get("name", "unknown"),
            "seed": int(getattr(sim_cfg, "seed", 0) or 0),
            "horizon": int(getattr(sim_cfg, "max_episode_steps", 120) or 120),
            "dt": float(getattr(sim_cfg, "time_per_step_in_secs", 0.1) or 0.1),
            "observation_mode": active_observation_mode,
            "observation_level": resolved_observation_level,
        }

    policy_fn, algo_meta = policy_builder(
        algo,
        policy_cfg,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
        adapter_impact_eval=adapter_impact_eval,
        **extra_kwargs,
    )
    algo_meta = cast(
        "AlgoMeta",
        enrich_algorithm_metadata(
            algo=algo,
            metadata=cast("dict[str, Any]", algo_meta),
            robot_kinematics=robot_kinematics,
            observation_mode=active_observation_mode,
            observation_level=resolved_observation_level,
        ),
    )
    # Latency instrumentation resolves the planner configuration hash from the callable so
    # cached policies remain provenance-bound when a new harness is activated per episode.
    policy_fn._meta = algo_meta
    algo_meta["learned_checkpoint_observation_contract"] = learned_observation_contract
    active_observation_level = str(algo_meta["observation_level"]["key"])
    attach_track_metadata(
        cast("dict[str, Any]", algo_meta),
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        observation_level=active_observation_level,
        observation_mode=active_observation_mode,
    )
    planner_close = getattr(policy_fn, "_planner_close", None)
    planner_reset = getattr(policy_fn, "_planner_reset", None)
    planner_bind_env = getattr(policy_fn, "_planner_bind_env", None)
    planner_stats = getattr(policy_fn, "_planner_stats", None)
    planner_native_action = getattr(policy_fn, "_planner_native_env_action", False)
    actuation_controller = (
        SyntheticActuationController(
            profile=actuation_profile, dt=config.sim_config.time_per_step_in_secs
        )
        if actuation_profile is not None
        else None
    )
    single_pedestrian_intent_metadata = _single_pedestrian_intent_metadata(scenario)
    single_pedestrian_vru_metadata = _single_pedestrian_vru_metadata(scenario)

    return _PolicyContract(
        policy_fn=policy_fn,
        algo_meta=algo_meta,
        planner_close=planner_close,
        planner_reset=planner_reset,
        planner_bind_env=planner_bind_env,
        planner_stats=planner_stats,
        planner_native_action=planner_native_action,
        actuation_controller=actuation_controller,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        single_pedestrian_intent_metadata=single_pedestrian_intent_metadata,
        single_pedestrian_vru_metadata=single_pedestrian_vru_metadata,
    )


@dataclass(frozen=True, slots=True)
class _EpisodeStepLoopResult:
    """All outputs of the episode step-loop phase of ``run_map_episode``.

    Bundles the trajectory position/heading/force lists, visibility evidence
    traces, per-step instrumentation traces, collision/termination outcome flags,
    the final map_def/goal_vec, the effective-view integrity probe result, and the
    planner runtime snapshot captured at teardown.
    """

    map_def: Any
    goal_vec: np.ndarray
    initial_robot_pos: np.ndarray
    initial_robot_heading: float
    initial_ped_positions: np.ndarray
    initial_robot_velocity: np.ndarray | None
    initial_ped_velocities: np.ndarray | None
    trace_actor_ids: list[str] | None
    initial_goal_distance: float
    reached_goal_step: int | None
    termination_reason: str
    collision_seen: bool
    ped_collision_seen: bool
    obstacle_collision_seen: bool
    robot_collision_seen: bool
    timeout_seen: bool
    collision_events: list[dict[str, Any]]
    robot_positions: list[np.ndarray]
    robot_headings: list[float]
    ped_positions: list[np.ndarray]
    ped_forces: list[np.ndarray]
    visibility_trace: list[np.ndarray | None]
    track_confidence_trace: list[np.ndarray | None]
    visibility_evidence_statuses: list[str]
    visibility_evidence_reasons: list[str | None]
    tracking_precision_records: list[dict[str, Any]]
    min_separation_corrupted_values: list[float]
    safety_wrapper_trace: list[dict[str, Any]]
    cbf_filter_trace: list[dict[str, Any]]
    ammv_command_actions: list[dict[str, Any]]
    synthetic_actuation_trace: list[dict[str, Any]]
    hybrid_command_sources: list[str | None] | None
    planner_decision_trace: list[PlannerDecisionTraceEntry]
    simulation_step_trace: list[dict[str, Any]]
    view_integrity: dict[str, Any] | None
    planner_runtime_snapshot: dict[str, Any] | None
    obstacle_force_law_metadata: dict[str, Any] | None


@dataclass(slots=True)
class _StepLoopState:
    """Mutable state accumulated across episode step-loop iterations."""

    obs: Any
    current_command: tuple[float, float] = (0.0, 0.0)
    view_integrity: dict[str, Any] | None = None
    collision_seen: bool = False
    ped_collision_seen: bool = False
    obstacle_collision_seen: bool = False
    robot_collision_seen: bool = False
    timeout_seen: bool = False
    reached_goal_step: int | None = None
    termination_reason: str = "max_steps"
    previous_trace_robot_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    previous_trace_ped_pos: np.ndarray | None = None
    previous_trace_heading: float = 0.0
    previous_collision_robot_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    previous_collision_ped_pos: np.ndarray | None = None
    robot_positions: list[np.ndarray] = field(default_factory=list)
    robot_headings: list[float] = field(default_factory=list)
    ped_positions: list[np.ndarray] = field(default_factory=list)
    ped_forces: list[np.ndarray] = field(default_factory=list)
    collision_events: list[dict[str, Any]] = field(default_factory=list)
    visibility_trace: list[np.ndarray | None] = field(default_factory=list)
    track_confidence_trace: list[np.ndarray | None] = field(default_factory=list)
    visibility_evidence_statuses: list[str] = field(default_factory=list)
    visibility_evidence_reasons: list[str | None] = field(default_factory=list)
    tracking_precision_records: list[dict[str, Any]] = field(default_factory=list)
    min_separation_corrupted_values: list[float] = field(default_factory=list)
    safety_wrapper_trace: list[dict[str, Any]] = field(default_factory=list)
    cbf_filter_trace: list[dict[str, Any]] = field(default_factory=list)
    ammv_command_actions: list[dict[str, Any]] = field(default_factory=list)
    synthetic_actuation_trace: list[dict[str, Any]] = field(default_factory=list)
    hybrid_command_sources: list[str | None] | None = None
    planner_decision_trace: list[PlannerDecisionTraceEntry] = field(default_factory=list)
    simulation_step_trace: list[dict[str, Any]] = field(default_factory=list)
    map_def: Any = None
    goal_vec: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    initial_robot_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    initial_robot_heading: float = 0.0
    initial_ped_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    initial_robot_velocity: np.ndarray | None = None
    initial_ped_velocities: np.ndarray | None = None
    trace_actor_ids: list[str] | None = field(default_factory=list)
    initial_goal_distance: float = 0.0
    planner_runtime_snapshot: dict[str, Any] | None = None
    simulator_obstacle_force_law_metadata: dict[str, Any] | None = None
    planner_obstacle_force_law_metadata: dict[str, Any] | None = None


def _read_obstacle_force_law_metadata(env: Any) -> dict[str, Any] | None:
    """Read a serializable obstacle-law payload from the active simulator.

    Returns:
        Simulator metadata payload, or ``None`` when the simulator has no accessor.
    """
    simulator = getattr(env, "simulator", None)
    metadata_fn = getattr(simulator, "obstacle_force_law_metadata", None)
    if not callable(metadata_fn):
        return None
    payload = metadata_fn()
    return dict(payload) if isinstance(payload, Mapping) else None


def _read_policy_obstacle_force_law_metadata(policy_fn: Any) -> dict[str, Any] | None:
    """Read obstacle-law metadata directly from a policy's planner adapter.

    Returns:
        Planner metadata payload, or ``None`` when the policy has no accessor.
    """
    adapter = getattr(policy_fn, "_planner_adapter", None)
    metadata_fn = getattr(adapter, "obstacle_force_law_metadata", None)
    if not callable(metadata_fn):
        return None
    payload = metadata_fn()
    return dict(payload) if isinstance(payload, Mapping) else None


def _obstacle_force_site_metadata(payload: Any) -> dict[str, Any] | None:
    """Return one site payload when it carries an unambiguous site identifier."""
    if not isinstance(payload, Mapping):
        return None
    site = payload.get("site")
    if not isinstance(site, str) or not site.strip():
        return None
    return dict(payload)


def _build_obstacle_force_runtime_record(
    *,
    simulator_metadata: Mapping[str, Any] | None,
    planner_metadata: Mapping[str, Any] | None,
    planner_runtime_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Combine site-specific law metadata into the durable episode payload.

    Returns:
        Versioned runtime record, or ``None`` when no site payload is available.
    """
    sites: dict[str, dict[str, Any]] = {}
    for candidate in (
        _obstacle_force_site_metadata(simulator_metadata),
        _obstacle_force_site_metadata(planner_metadata),
        _obstacle_force_site_metadata(
            planner_runtime_snapshot.get("obstacle_force_law")
            if isinstance(planner_runtime_snapshot, Mapping)
            else None
        ),
    ):
        if candidate is not None:
            site = str(candidate["site"])
            previous = sites.get(site)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"conflicting obstacle-force metadata for site {site!r}; "
                    "refusing to overwrite a runtime snapshot"
                )
            sites[site] = candidate

    if not sites:
        return None
    return {
        "schema_version": _OBSTACLE_FORCE_LAW_RUNTIME_RECORD_SCHEMA,
        "sites": sites,
    }


@dataclass(frozen=True, slots=True)
class _StepLoopConfig:
    """Read-only per-episode configuration for step-loop helpers."""

    config: RobotSimulationConfig
    policy_fn: Any
    planner_native_action: Any
    noise_spec: Any
    noise_rng: np.random.Generator
    noise_state: Any
    noise_stats: dict[str, int]
    tracking_precision_spec: TrackingPrecisionSpec
    tracking_precision_rng: np.random.Generator
    safety_wrapper_runtime: SafetyWrapperRuntimeConfig
    safety_wrapper_deadlock_monitor: Any
    cbf_runtime: CBFSafetyFilterRuntimeConfig
    actuation_controller: Any
    algo_meta: AlgoMeta
    record_forces: bool
    record_planner_decision_trace: bool
    record_simulation_step_trace: bool
    single_pedestrian_intent_metadata: Any
    single_pedestrian_vru_metadata: Any
    hybrid_source_field: str | None
    active_harness: Any
    collision_event_context: _CollisionEventContext


@dataclass(frozen=True, slots=True)
class _StepLoopSetupArgs:
    """Inputs for environment setup and execution of one episode step loop."""

    seed: int
    scenario: dict[str, object] | None
    config: RobotSimulationConfig
    horizon_val: int
    planner_runtime: PlannerRuntime
    noise: NoiseConfig
    tracking_precision_spec: TrackingPrecisionSpec
    tracking_precision_rng: np.random.Generator
    safety_wrapper_runtime: SafetyWrapperRuntimeConfig
    safety_wrapper_deadlock_monitor: DeadlockRecoveryMonitor | None
    cbf_runtime: CBFSafetyFilterRuntimeConfig
    actuation_controller: SyntheticActuationController | None
    algo_meta: AlgoMeta
    record_forces: bool
    record_planner_decision_trace: bool
    record_simulation_step_trace: bool
    single_pedestrian_intent_metadata: Any
    single_pedestrian_vru_metadata: Any
    pedestrian_control_trace_label_builder: PedestrianControlTraceLabelBuilder | None
    expected_population_size: int | None
    hybrid_source_field: str | None


@dataclass(slots=True)
class _StepSimResult:
    """Per-step simulation outputs consumed by trace and termination helpers."""

    robot_pos: np.ndarray
    peds: np.ndarray
    forces_arr: np.ndarray | None
    heading: float
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    step_visible: np.ndarray | None
    step_confidence: np.ndarray | None
    step_visibility_status: str
    step_visibility_reason: str | None
    selected_action_payload: dict[str, Any]
    applied_environment_action_payload: dict[str, Any]
    actuation_step: Any
    planner_step_decision: dict[str, Any] | None


def _prepare_episode_env(  # noqa: C901
    env: Any,
    *,
    seed: int,
    scenario: dict[str, object] | None,
    planner_bind_env: Any,
    planner_reset: Any,
    expected_population_size: int | None,
    pedestrian_control_trace_label_builder: PedestrianControlTraceLabelBuilder | None,
) -> Any:
    """Reset the environment, validate population, and bind the planner.

    Returns:
        The initial observation from ``env.reset``.
    """
    obs, _ = env.reset(seed=int(seed))
    instantiated_count: int | None = None
    if expected_population_size is not None:
        if scenario is None:
            raise ValueError("population-size validation requires the episode scenario")
        instantiated_count = int(np.asarray(env.simulator.ped_pos).reshape(-1, 2).shape[0])
        # Issue #5666 acceptance: a forced population must instantiate exactly
        # the declared count. A silent divergence is what wasted a full compute
        # cycle, so fail loudly instead of letting the mismatch propagate.
        if instantiated_count != expected_population_size:
            raise AssertionError(
                f"instantiated pedestrian count {instantiated_count} does not match forced "
                f"population_size {expected_population_size}; the declared population was not "
                "realized by the simulator"
            )
    if pedestrian_control_trace_label_builder is not None:
        if scenario is None:
            raise ValueError("runtime trace label building requires the episode scenario")
        if instantiated_count is None:
            instantiated_count = int(np.asarray(env.simulator.ped_pos).reshape(-1, 2).shape[0])
        runtime_labels = pedestrian_control_trace_label_builder(instantiated_count)
        if len(runtime_labels) != instantiated_count:
            raise ValueError(
                "runtime pedestrian control trace label builder must return one label per "
                f"instantiated pedestrian (got {len(runtime_labels)}, expected "
                f"{instantiated_count})"
            )
        scenario["pedestrian_control_trace_labels"] = runtime_labels
    if expected_population_size is not None and scenario is not None:
        simulation_config = scenario.get("simulation_config")
        if isinstance(simulation_config, dict):
            # Record the *instantiated* count so the readiness gate and any
            # future triage can see declared-vs-actual without re-running.
            simulation_config["population_size"] = instantiated_count
            simulation_config["instantiated_population_size"] = instantiated_count
            simulation_config["declared_population_size"] = expected_population_size
    if callable(planner_bind_env):
        planner_bind_env(env)
    if callable(planner_reset):
        planner_reset(seed=int(seed))
    return obs


def _init_step_loop_state(
    *,
    obs: Any,
    env: Any,
    config: RobotSimulationConfig,
    hybrid_source_field: str | None,
) -> _StepLoopState:
    """Create the mutable step-loop state from the post-reset environment.

    Returns:
        _StepLoopState: Initialized mutable state bundle.
    """
    map_def = getattr(env.simulator, "map_def", None)
    goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
    initial_robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
    initial_ped_positions = np.array(env.simulator.ped_pos, dtype=float, copy=True).reshape(-1, 2)
    initial_robot_velocity = _initial_robot_velocity(env.simulator)
    initial_ped_velocities = _initial_ped_velocities(env.simulator, len(initial_ped_positions))
    trace_actor_ids = _initial_pedestrian_actor_ids(env.simulator, len(initial_ped_positions))
    initial_goal_distance = float(np.linalg.norm(initial_robot_pos - goal_vec))
    state = _StepLoopState(obs=obs)
    state.hybrid_command_sources = [] if hybrid_source_field is not None else None
    state.previous_trace_robot_pos = np.array(initial_robot_pos, dtype=float, copy=True)
    state.previous_trace_heading = _reset_robot_heading(env.simulator, obs)
    state.initial_robot_heading = state.previous_trace_heading
    state.previous_collision_robot_pos = np.array(initial_robot_pos, dtype=float, copy=True)
    state.map_def = map_def
    state.goal_vec = goal_vec
    state.initial_robot_pos = initial_robot_pos
    state.initial_ped_positions = initial_ped_positions
    state.initial_robot_velocity = initial_robot_velocity
    state.initial_ped_velocities = initial_ped_velocities
    state.trace_actor_ids = trace_actor_ids
    state.initial_goal_distance = initial_goal_distance
    return state


def _reset_robot_heading(simulator: Any, obs: Any) -> float:
    """Read a reset heading from simulator state before observation fallback.

    Returns:
        The finite reset heading in radians, or ``0.0`` when no heading is
        available from the simulator or observation.
    """

    for name in ("robot_heading", "robot_theta", "heading"):
        value = getattr(simulator, name, None)
        try:
            numeric = float(np.asarray(value).reshape(-1)[0]) if value is not None else None
        except (TypeError, ValueError, IndexError):
            numeric = None
        if numeric is not None and np.isfinite(numeric):
            return numeric
    return _observation_heading(obs)


def _initial_robot_velocity(simulator: Any) -> np.ndarray | None:  # noqa: C901
    """Read the reset robot velocity without assuming a zero initial state.

    Returns:
        The finite reset velocity, or ``None`` when the simulator does not expose it.
    """

    direct = getattr(simulator, "robot_velocity_xy", None)
    if direct is not None:
        try:
            value = np.asarray(direct, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            value = np.asarray([], dtype=float)
        if value.size >= 2 and np.isfinite(value[:2]).all():
            return value[:2].copy()
    robots = getattr(simulator, "robots", None)
    if isinstance(robots, (list, tuple)) and robots:
        state = getattr(robots[0], "state", None)
        pose = getattr(state, "pose", None)
        heading = None
        if isinstance(pose, (list, tuple)) and len(pose) > 1:
            try:
                heading = float(pose[1])
            except (TypeError, ValueError):
                heading = None
        for name in ("velocity_xy", "robot_velocity_xy"):
            value = getattr(state, name, None)
            if value is not None:
                try:
                    array = np.asarray(value, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if array.size >= 2 and np.isfinite(array[:2]).all():
                    return array[:2].copy()
        polar = getattr(state, "velocity", None)
        if heading is not None:
            try:
                values = np.asarray(polar, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                values = np.asarray([], dtype=float)
            if values.size >= 1 and np.isfinite(values[0]):
                speed = float(values[0])
                return np.asarray([speed * np.cos(heading), speed * np.sin(heading)], dtype=float)
    return None


def _initial_ped_velocities(simulator: Any, count: int) -> np.ndarray | None:
    """Read reset pedestrian velocities, preserving unavailable state explicitly.

    Returns:
        A finite ``(count, 2)`` array, or ``None`` when reset velocities are unavailable.
    """

    value = getattr(simulator, "ped_vel", None)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if array.shape[0] < count or not np.isfinite(array[:count]).all():
        return None
    return array[:count].copy()


def _initial_pedestrian_actor_ids(simulator: Any, count: int) -> list[str] | None:
    """Return stable simulator-slot IDs for the reset pedestrian population.

    The benchmark simulator stores pedestrian state in fixed rows.  When an
    explicit identity registry is exposed, preserve it; otherwise the row
    index is promoted to a namespaced simulator-slot identity.  The latter is
    deliberately distinct from the legacy frame ``id`` field so positional
    v1 traces cannot be mistaken for stable identities by the analysis gate.

    Returns:
        Stable, unique identifiers aligned with ``simulator.ped_pos``.  ``None``
        means that an exposed registry was malformed and the trace must remain
        unavailable rather than falling back to a guessed identity.
    """
    if count <= 0:
        return []
    owners = (simulator, getattr(simulator, "pysf_state", None))
    for owner in owners:
        if owner is None:
            continue
        for name in ("pedestrian_ids", "ped_ids", "ped_actor_ids"):
            raw = getattr(owner, name, None)
            if raw is None:
                continue
            if isinstance(raw, (str, bytes)):
                return None
            try:
                values = list(raw)
            except TypeError:
                return None
            if len(values) != count:
                return None
            identifiers = [str(value).strip() for value in values]
            if all(
                identifier
                and identifier.lower() not in {"none", "nan", "null"}
                and not isinstance(value, bool)
                for value, identifier in zip(values, identifiers, strict=True)
            ) and len(set(identifiers)) == len(identifiers):
                return identifiers
            return None
    return [f"simulator-slot-{index}" for index in range(count)]


def _finite_positive_float(value: Any) -> float | None:
    """Return a finite positive geometry value without substituting a default."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0.0 else None


def _make_collision_event_context(
    config: RobotSimulationConfig,
    map_def: Any,
) -> _CollisionEventContext:
    """Build the per-episode collision-event typing context.

    Returns:
        _CollisionEventContext: Immutable context for per-step collision typing.
    """
    # Normalize optional radii before float(): getattr returns the default
    # only when the attribute is absent, so an explicit None in config would
    # otherwise raise TypeError in float(None).
    robot_radius_val = getattr(getattr(config, "robot_config", None), "radius", 1.0)
    robot_radius = float(robot_radius_val if robot_radius_val is not None else 1.0)
    ped_radius_val = getattr(config.sim_config, "ped_radius", 0.4)
    ped_radius = float(ped_radius_val if ped_radius_val is not None else 0.4)
    return _CollisionEventContext(
        dt_seconds=float(config.sim_config.time_per_step_in_secs),
        map_def=map_def,
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )


def _step_policy_inference(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    env: Any,
) -> tuple[Any, np.ndarray | None]:
    """Apply observation noise, tracking precision, and policy inference.

    Returns:
        Tuple of (policy_command, corrupted_ped_positions).
    """
    policy_obs, step_noise_stats = apply_observation_noise(
        state.obs,
        cast("dict[str, Any]", slc.noise_spec),
        slc.noise_rng,
        slc.noise_state,
    )
    merge_observation_noise_stats(slc.noise_stats, step_noise_stats)
    policy_obs, corrupted_ped_positions = _apply_tracking_precision_to_observation(
        policy_obs,
        slc.tracking_precision_spec,
        slc.tracking_precision_rng,
    )
    robot_reference = np.asarray(env.simulator.robot_pos[0], dtype=float)
    if corrupted_ped_positions is not None:
        state.min_separation_corrupted_values.append(
            minimum_separation(corrupted_ped_positions, robot_reference)
        )
    policy_command = slc.policy_fn(policy_obs)
    if state.view_integrity is None:
        # Runtime fail-closed guard (#3634): probe the planner's effective observation view
        # once. The extractor signature is deterministic across steps, so a single probe
        # detects a silent-blind planner before any benchmark metrics are recorded. Fail
        # closed per docs/context/issue_691_benchmark_fallback_policy.md instead of emitting
        # results produced by a planner that drives blind to the pedestrians it was shown.
        integrity = evaluate_effective_view_integrity(
            policy_fn=slc.policy_fn,
            observation=policy_obs,
            algo_meta=slc.algo_meta,
        )
        state.view_integrity = integrity.to_metadata()
        if integrity.degraded:
            raise DegeneratePlannerViewError(integrity)
    return policy_command, corrupted_ped_positions


def _step_hybrid_and_planner_stats(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    planner_stats: Any,
) -> tuple[bool, dict[str, Any] | None]:
    """Sample hybrid handoff telemetry and planner stats for one step.

    Returns:
        Tuple of (step_is_native, planner_step_decision).
    """
    planner_step_decision = None
    # Hybrid handoff telemetry is part of the episode predicate contract, so
    # sample it even when the larger planner-decision trace is not requested.
    if (slc.record_planner_decision_trace or state.hybrid_command_sources is not None) and callable(
        planner_stats
    ):
        try:
            planner_stats_payload = planner_stats()
        except (RuntimeError, ValueError, TypeError):
            planner_stats_payload = None
        if isinstance(planner_stats_payload, dict) and isinstance(
            planner_stats_payload.get("last_decision"), dict
        ):
            planner_step_decision = dict(planner_stats_payload["last_decision"])
    if state.hybrid_command_sources is not None:
        source = (
            planner_step_decision.get(slc.hybrid_source_field)
            if planner_step_decision is not None and slc.hybrid_source_field is not None
            else None
        )
        normalized_source = str(source).strip() if source is not None else ""
        state.hybrid_command_sources.append(normalized_source or None)
    # Use per-step flag when available (e.g. SAC with fallback); fall back to the
    # static cached value for planners that set _planner_native_env_action once.
    step_is_native = getattr(slc.policy_fn, "_last_step_native", slc.planner_native_action)
    return bool(step_is_native), planner_step_decision


def _step_actuation_and_tracking(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    policy_command: Any,
    step_is_native: bool,
) -> tuple[Any, Any]:
    """Apply synthetic actuation and tracking-precision speed contract.

    Returns:
        Tuple of (policy_command, actuation_step).
    """
    actuation_step = None
    if slc.actuation_controller is not None and step_is_native:
        raise ValueError(
            "synthetic_actuation_profile requires absolute differential-drive commands; "
            "native env actions cannot be wrapped safely"
        )
    if slc.actuation_controller is not None:
        if not isinstance(policy_command, (tuple, list, np.ndarray)) or len(policy_command) < 2:
            raise TypeError(
                "synthetic_actuation_profile expects planner commands shaped like "
                "(linear_velocity, angular_velocity)"
            )
        actuation_step = slc.actuation_controller.apply(
            current_command=state.current_command,
            requested_command=(float(policy_command[0]), float(policy_command[1])),
        )
        policy_command = actuation_step.applied_command
        state.current_command = actuation_step.applied_command
    if (
        bool(slc.tracking_precision_spec.get("enabled", False))
        and not step_is_native
        and isinstance(policy_command, (tuple, list, np.ndarray))
        and len(policy_command) >= 2
    ):
        applied_linear, tracking_record = apply_speed_contract(
            float(policy_command[0]),
            float(slc.tracking_precision_spec["target_motp_m"]),
            cast("dict[str, Any]", slc.tracking_precision_spec),
        )
        policy_command = (
            applied_linear,
            float(policy_command[1]),
            *tuple(policy_command[2:]),
        )
        state.tracking_precision_records.append(tracking_record)
    return policy_command, actuation_step


def _step_safety_filters(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    policy_command: Any,
    step_is_native: bool,
    env: Any,
    step_idx: int,
) -> Any:
    """Apply safety wrapper and CBF safety filter to the command.

    Returns:
        The (possibly corrected) policy command.
    """
    if slc.safety_wrapper_runtime.enabled:
        policy_command, wrapper_record = _apply_safety_wrapper_step(
            policy_command,
            runtime=slc.safety_wrapper_runtime,
            env=env,
            config=slc.config,
            step_idx=step_idx,
            step_is_native=step_is_native,
            previous_ped_positions=state.previous_trace_ped_pos,
            deadlock_monitor=slc.safety_wrapper_deadlock_monitor,
        )
        state.safety_wrapper_trace.append(wrapper_record)
    if slc.cbf_runtime.enabled:
        policy_command, cbf_record = _apply_cbf_safety_filter_step(
            policy_command,
            runtime=slc.cbf_runtime,
            env=env,
            config=slc.config,
            step_idx=step_idx,
            step_is_native=step_is_native,
            previous_ped_positions=state.previous_trace_ped_pos,
        )
        state.cbf_filter_trace.append(cbf_record)
    return policy_command


def _step_convert_and_execute(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    policy_command: Any,
    step_is_native: bool,
    env: Any,
) -> tuple[Any, float, bool, bool, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Convert command to env action, execute step, and update state.obs.

    Returns:
        Tuple of (obs, reward, terminated, truncated, info, selected_action_payload,
        applied_environment_action_payload).
    """
    selected_action_payload = _command_action_payload(policy_command)
    state.ammv_command_actions.append(selected_action_payload)
    action_conversion_start = time.perf_counter() if slc.active_harness is not None else None
    if step_is_native:
        # Policy already outputs native env actions (e.g. delta velocities);
        # skip the absolute->delta conversion done by _policy_command_to_env_action.
        action = np.asarray(policy_command, dtype=np.float32)
    else:
        action = _policy_command_to_env_action(
            env=env,
            config=slc.config,
            command=policy_command,
        )
    applied_environment_action_payload = _command_action_payload(action)
    if slc.active_harness is not None and action_conversion_start is not None:
        slc.active_harness.add_time(
            "action_conversion", (time.perf_counter() - action_conversion_start) * 1000.0
        )
    if slc.active_harness is not None:
        slc.active_harness.end_cycle()
    obs, reward, terminated, truncated, info = env.step(action)
    state.obs = obs
    return (
        obs,
        reward,
        terminated,
        truncated,
        info,
        selected_action_payload,
        applied_environment_action_payload,
    )


def _step_snapshot_and_record(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    env: Any,
    obs: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    float,
    np.ndarray | None,
    np.ndarray | None,
    str,
    str | None,
]:
    """Snapshot mutable simulator buffers and record positions/headings/visibility.

    Returns:
        Tuple of (robot_pos, peds, forces_arr, heading, step_visible,
        step_confidence, step_visibility_status, step_visibility_reason).
    """
    # Snapshot mutable simulator buffers; do not keep view aliases across steps.
    robot_pos = np.array(env.simulator.robot_pos[0], dtype=float, copy=True)
    peds = np.array(env.simulator.ped_pos, dtype=float, copy=True)
    forces_arr: np.ndarray | None = None
    if slc.record_forces:
        forces = getattr(env.simulator, "last_ped_forces", None)
        if forces is None:
            forces_arr = np.zeros_like(peds, dtype=float)
        else:
            forces_arr = np.array(forces, dtype=float, copy=True)
            if forces_arr.shape != peds.shape:
                forces_arr = np.zeros_like(peds, dtype=float)
    state.robot_positions.append(robot_pos)
    state.ped_positions.append(peds)
    if slc.record_forces and forces_arr is not None:
        state.ped_forces.append(forces_arr)
    heading = _observation_heading(obs, default=state.previous_trace_heading)
    state.robot_headings.append(float(heading))
    (
        step_visible,
        step_confidence,
        step_visibility_status,
        step_visibility_reason,
    ) = _visibility_evidence_for_step(peds=peds, obs=obs, config=slc.config)
    state.visibility_trace.append(step_visible)
    state.track_confidence_trace.append(step_confidence)
    state.visibility_evidence_statuses.append(step_visibility_status)
    state.visibility_evidence_reasons.append(step_visibility_reason)
    return (
        robot_pos,
        peds,
        forces_arr,
        float(heading),
        step_visible,
        step_confidence,
        step_visibility_status,
        step_visibility_reason,
    )


def _step_build_simulation_trace(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    step_idx: int,
    sim: _StepSimResult,
) -> None:
    """Append one simulation-step-trace entry when trace recording is enabled."""
    if not slc.record_simulation_step_trace:
        return
    dt_seconds = float(slc.config.sim_config.time_per_step_in_secs)
    robot_velocity = (
        (sim.robot_pos - state.previous_trace_robot_pos) / dt_seconds
        if dt_seconds > 0.0
        else np.zeros(2, dtype=float)
    )
    planner_payload: dict[str, Any] = {
        "event": "step",
        "selected_action": sim.selected_action_payload,
        "applied_environment_action": sim.applied_environment_action_payload,
    }
    if sim.actuation_step is not None:
        planner_payload["amv"] = {
            "requested_linear_m_s": float(sim.actuation_step.requested_command[0]),
            "requested_angular_rad_s": float(sim.actuation_step.requested_command[1]),
            "applied_linear_m_s": float(sim.actuation_step.applied_command[0]),
            "applied_angular_rad_s": float(sim.actuation_step.applied_command[1]),
            "command_clipped": bool(sim.actuation_step.command_clipped),
            "yaw_rate_saturated": bool(sim.actuation_step.yaw_rate_saturated),
        }
    if slc.record_forces and sim.forces_arr is not None and sim.peds.size:
        planner_payload["ammv"] = {
            "pedestrian_force_vectors": [
                [float(force[0]), float(force[1])] for force in sim.forces_arr
            ]
        }
    trace_pedestrians = _annotate_trace_visibility(
        _trace_pedestrians(
            sim.peds,
            state.previous_trace_ped_pos,
            dt_seconds,
            slc.single_pedestrian_intent_metadata,
            slc.single_pedestrian_vru_metadata,
            sim.robot_pos,
            robot_velocity,
            state.trace_actor_ids,
        ),
        visible=sim.step_visible,
        track_confidence=sim.step_confidence,
        evidence_status=sim.step_visibility_status,
        evidence_reason=sim.step_visibility_reason,
    )
    trace_entry: dict[str, Any] = {
        "step": int(step_idx),
        "time_s": float((step_idx + 1) * dt_seconds),
        "robot": {
            "position": [float(sim.robot_pos[0]), float(sim.robot_pos[1])],
            "heading": float(sim.heading),
            "velocity": [float(robot_velocity[0]), float(robot_velocity[1])],
        },
        "pedestrians": trace_pedestrians,
        "planner": planner_payload,
        "rl": {
            "reward": float(sim.reward),
            "terminated": bool(sim.terminated),
            "truncated": bool(sim.truncated),
        },
    }
    sim_info = getattr(sim, "info", None)
    oracle_trace = sim_info.get("oracle_transition_trace") if isinstance(sim_info, dict) else None
    if oracle_trace is not None:
        # Preserve the evaluator-only trace as a sibling of planner data. It is
        # never copied into observations or planner decision payloads.
        trace_entry["oracle_transition_trace"] = oracle_trace
    state.simulation_step_trace.append(trace_entry)
    state.previous_trace_robot_pos = np.array(sim.robot_pos, dtype=float, copy=True)
    state.previous_trace_ped_pos = np.array(sim.peds, dtype=float, copy=True)
    state.previous_trace_heading = float(sim.heading)


def _step_build_actuation_trace(
    state: _StepLoopState,
    *,
    step_idx: int,
    sim: _StepSimResult,
) -> None:
    """Append one synthetic-actuation trace entry when actuation is active."""
    if sim.actuation_step is None:
        return
    distance_to_goal = float(np.linalg.norm(sim.robot_pos - state.goal_vec))
    route_progress = float(state.initial_goal_distance - distance_to_goal)
    progress_ratio = (
        route_progress / state.initial_goal_distance if state.initial_goal_distance > 1e-9 else 0.0
    )
    state.synthetic_actuation_trace.append(
        {
            "step": int(step_idx),
            "requested_linear_m_s": float(sim.actuation_step.requested_command[0]),
            "requested_angular_rad_s": float(sim.actuation_step.requested_command[1]),
            "applied_linear_m_s": float(sim.actuation_step.applied_command[0]),
            "applied_angular_rad_s": float(sim.actuation_step.applied_command[1]),
            "command_clipped": bool(sim.actuation_step.command_clipped),
            "yaw_rate_saturated": bool(sim.actuation_step.yaw_rate_saturated),
            "linear_accel_applied_m_s2": float(sim.actuation_step.linear_accel_applied_m_s2),
            "angular_accel_applied_rad_s2": float(sim.actuation_step.angular_accel_applied_rad_s2),
            "distance_to_goal_m": distance_to_goal,
            "route_progress_from_start_m": route_progress,
            "route_progress_ratio": float(progress_ratio),
            "robot_x_m": float(sim.robot_pos[0]),
            "robot_y_m": float(sim.robot_pos[1]),
        }
    )


def _finite_planner_decision_value(value: Any) -> float | None:
    """Return a finite planner diagnostic value or ``None``."""
    if isinstance(value, int | float | np.integer | np.floating) and math.isfinite(float(value)):
        return float(value)
    return None


def _planner_decision_counter_mapping(raw: Any, *, field: str) -> dict[str, int]:
    """Normalize a non-negative integer trace counter mapping or fail closed.

    Returns:
        A string-keyed counter mapping.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"planner decision {field} must be a mapping")
    normalized: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(value, int | np.integer) or isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"planner decision {field}.{key} must be a non-negative integer")
        normalized[str(key)] = int(value)
    return normalized


def _planner_decision_nested_counter_mapping(raw: Any, *, field: str) -> dict[str, dict[str, int]]:
    """Normalize nested per-source trace counters or fail closed.

    Returns:
        A string-keyed mapping of counter mappings.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"planner decision {field} must be a mapping")
    normalized: dict[str, dict[str, int]] = {}
    for source, counts in raw.items():
        normalized[str(source)] = _planner_decision_counter_mapping(
            counts,
            field=f"{field}.{source}",
        )
    return normalized


def _step_build_planner_decision_entry(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    step_idx: int,
    sim: _StepSimResult,
) -> None:
    """Append one planner-decision-trace entry when trace recording is enabled."""
    if not slc.record_planner_decision_trace or sim.planner_step_decision is None:
        return
    psd = sim.planner_step_decision
    selected_terms = psd.get("selected_terms")
    selected_terms = selected_terms if isinstance(selected_terms, dict) else {}
    progress_windows_raw = psd.get("progress_windows")
    progress_windows = progress_windows_raw if isinstance(progress_windows_raw, dict) else {}
    selected_command = psd.get("selected_command")
    selected_command = selected_command if isinstance(selected_command, list) else []
    rejection_counts = _planner_decision_counter_mapping(
        psd.get("rejection_counts"), field="rejection_counts"
    )
    moving_rejection_counts = _planner_decision_counter_mapping(
        psd.get("moving_rejection_counts"), field="moving_rejection_counts"
    )
    rejection_counts_by_source = _planner_decision_nested_counter_mapping(
        psd.get("rejection_counts_by_source"), field="rejection_counts_by_source"
    )
    distance_to_goal = float(np.linalg.norm(sim.robot_pos - state.goal_vec))
    step_decision: dict[str, Any] = {
        "step": int(step_idx),
        "selected_source": str(psd.get("selected_source", "unknown")),
        "planner_mode": str(psd.get("planner_mode", "unknown")),
        "selected_command": [
            float(value)
            for value in selected_command[:2]
            if isinstance(value, int | float | np.integer | np.floating)
        ],
        "selected_score": float(psd["selected_score"])
        if isinstance(psd.get("selected_score"), int | float | np.integer | np.floating)
        and math.isfinite(float(psd["selected_score"]))
        else None,
        "static_recenter": float(selected_terms.get("static_recenter", 0.0)),
        "route_arc_progress": float(selected_terms.get("route_arc_progress", 0.0)),
        "goal_progress": float(selected_terms.get("goal_progress", 0.0)),
        "progress_windows": {
            str(key): float(value)
            for key, value in progress_windows.items()
            if isinstance(value, int | float | np.integer | np.floating)
        },
        "rejection_counts": rejection_counts,
        "moving_rejection_counts": moving_rejection_counts,
        "rejection_counts_by_source": rejection_counts_by_source,
        "nearest_pedestrian_distance_m": _finite_planner_decision_value(
            psd.get("nearest_pedestrian_distance")
        ),
        "nearest_static_obstacle_distance_m": _finite_planner_decision_value(
            psd.get("nearest_static_obstacle_distance")
        ),
        "distance_to_goal_m": distance_to_goal,
        "route_progress_from_start_m": float(state.initial_goal_distance - distance_to_goal),
        "robot_x_m": float(sim.robot_pos[0]),
        "robot_y_m": float(sim.robot_pos[1]),
    }
    _step_planner_decision_topology_keys(step_decision, psd)
    _step_planner_decision_dwa_keys(step_decision, psd)
    state.planner_decision_trace.append(cast("PlannerDecisionTraceEntry", step_decision))


def _step_planner_decision_topology_keys(
    step_decision: dict[str, Any],
    planner_step_decision: dict[str, Any],
) -> None:
    """Copy topology-guided pass-through keys into the step decision entry."""
    for key in (
        "topology_guided",
        "topology_guided_config",
        "topology_lane_status",
        "topology_fallback_status",
        "topology_fallback_reason",
        "topology_candidate_availability",
        "topology_command_influence",
    ):
        value = planner_step_decision.get(key)
        if value is not None:
            step_decision[key] = deepcopy(value)
    topology_guided = step_decision.get("topology_guided")
    if isinstance(topology_guided, dict):
        corridor = planner_step_decision.get("planner_route_corridor")
        if isinstance(corridor, dict):
            config_payload = corridor.get("topology_guided_config")
            if isinstance(config_payload, dict):
                step_decision["topology_guided_config"] = deepcopy(config_payload)
        fallback_config = planner_step_decision.get("topology_guided_config")
        if "topology_guided_config" not in step_decision and isinstance(fallback_config, dict):
            step_decision["topology_guided_config"] = deepcopy(fallback_config)


def _step_planner_decision_dwa_keys(
    step_decision: dict[str, Any],
    planner_step_decision: dict[str, Any],
) -> None:
    """Copy additive DWA adapter diagnostic keys into the step decision entry.

    Additive, planner-agnostic pass-through for adapter diagnostics that do
    not map onto the topology-guided fields (issue #5298 DWA trace).
    Only present when the underlying adapter populates them, so other
    planners' traces are unchanged.
    """
    for dwa_key in (
        "constraint_reason",
        "candidate_total",
        "candidate_feasible",
        "candidate_infeasible",
        "feasible_score_min",
        "feasible_score_max",
        "dynamic_window",
        "target_goal",
        "global_route_probe_activated",
    ):
        value = planner_step_decision.get(dwa_key)
        if value is not None:
            step_decision[dwa_key] = deepcopy(value)


def _step_collision_and_termination(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    step_idx: int,
    sim: _StepSimResult,
) -> bool:
    """Update collision/termination state and return whether the loop should break.

    Returns:
        bool: True when the episode loop should break.
    """
    meta = sim.info.get("meta", {}) if isinstance(sim.info, dict) else {}
    step_collision = collision_event(sim.info)
    step_route_complete = route_complete_success(sim.info)
    step_success = step_route_complete and not step_collision
    step_timeout = bool(meta.get("is_timesteps_exceeded", False))
    state.collision_seen = state.collision_seen or step_collision
    state.ped_collision_seen = state.ped_collision_seen or bool(
        meta.get("is_pedestrian_collision", False)
    )
    state.obstacle_collision_seen = state.obstacle_collision_seen or bool(
        meta.get("is_obstacle_collision", False)
    )
    state.robot_collision_seen = state.robot_collision_seen or bool(
        meta.get("is_robot_collision", False)
    )
    state.timeout_seen = state.timeout_seen or step_timeout
    state.collision_events.extend(
        _step_collision_events(
            step_idx=step_idx,
            robot_pos=sim.robot_pos,
            previous_robot_pos=state.previous_collision_robot_pos,
            ped_positions=sim.peds,
            previous_ped_positions=state.previous_collision_ped_pos,
            meta=meta,
            context=slc.collision_event_context,
        )
    )
    state.previous_collision_robot_pos = np.array(sim.robot_pos, dtype=float, copy=True)
    state.previous_collision_ped_pos = np.array(sim.peds, dtype=float, copy=True)
    if state.reached_goal_step is None and step_success:
        state.reached_goal_step = step_idx
    if step_success:
        state.termination_reason = resolve_termination_reason(
            terminated=True,
            truncated=False,
            success=True,
            collision=step_collision,
        )
        return True
    if sim.terminated or sim.truncated:
        state.termination_reason = resolve_termination_reason(
            terminated=bool(sim.terminated),
            truncated=bool(sim.truncated),
            success=step_success,
            collision=step_collision,
        )
        return True
    return False


def _teardown_step_loop(
    env: Any,
    *,
    planner_stats: Any,
    planner_close: Any,
    state: _StepLoopState | None = None,
) -> None:
    """Capture planner runtime snapshot, close planner and environment.

    ``state`` is optional so teardown stays safe when the step-loop state was
    never initialised (e.g. ``env.reset()`` raised during setup): the planner
    hooks and environment are still released even though there is no state
    object to attach the runtime snapshot to, matching the pre-refactor
    finally-block behaviour.
    """
    if callable(planner_stats):
        try:
            planner_stats_payload = planner_stats()
        except (RuntimeError, ValueError, TypeError):
            logger.debug("Planner stats hook failed before close", exc_info=True)
            planner_stats_payload = None
        if isinstance(planner_stats_payload, dict) and state is not None:
            state.planner_runtime_snapshot = dict(planner_stats_payload)
    if callable(planner_close):
        try:
            planner_close()
        except (RuntimeError, ValueError, TypeError):
            logger.debug("Planner close hook failed", exc_info=True)
    env.close()


def _build_step_loop_result(state: _StepLoopState) -> _EpisodeStepLoopResult:
    """Construct the immutable step-loop result bundle from accumulated state.

    Returns:
        _EpisodeStepLoopResult: Immutable bundle of trajectory and outcome data.
    """
    return _EpisodeStepLoopResult(
        map_def=state.map_def,
        goal_vec=state.goal_vec,
        initial_robot_pos=state.initial_robot_pos,
        initial_robot_heading=state.initial_robot_heading,
        initial_ped_positions=state.initial_ped_positions,
        initial_robot_velocity=state.initial_robot_velocity,
        initial_ped_velocities=state.initial_ped_velocities,
        trace_actor_ids=(
            list(state.trace_actor_ids) if state.trace_actor_ids is not None else None
        ),
        initial_goal_distance=state.initial_goal_distance,
        reached_goal_step=state.reached_goal_step,
        termination_reason=state.termination_reason,
        collision_seen=state.collision_seen,
        ped_collision_seen=state.ped_collision_seen,
        obstacle_collision_seen=state.obstacle_collision_seen,
        robot_collision_seen=state.robot_collision_seen,
        timeout_seen=state.timeout_seen,
        collision_events=state.collision_events,
        robot_positions=state.robot_positions,
        robot_headings=state.robot_headings,
        ped_positions=state.ped_positions,
        ped_forces=state.ped_forces,
        visibility_trace=state.visibility_trace,
        track_confidence_trace=state.track_confidence_trace,
        visibility_evidence_statuses=state.visibility_evidence_statuses,
        visibility_evidence_reasons=state.visibility_evidence_reasons,
        tracking_precision_records=state.tracking_precision_records,
        min_separation_corrupted_values=state.min_separation_corrupted_values,
        safety_wrapper_trace=state.safety_wrapper_trace,
        cbf_filter_trace=state.cbf_filter_trace,
        ammv_command_actions=state.ammv_command_actions,
        synthetic_actuation_trace=state.synthetic_actuation_trace,
        hybrid_command_sources=state.hybrid_command_sources,
        planner_decision_trace=state.planner_decision_trace,
        simulation_step_trace=state.simulation_step_trace,
        view_integrity=state.view_integrity,
        planner_runtime_snapshot=state.planner_runtime_snapshot,
        obstacle_force_law_metadata=_build_obstacle_force_runtime_record(
            simulator_metadata=state.simulator_obstacle_force_law_metadata,
            planner_metadata=state.planner_obstacle_force_law_metadata,
            planner_runtime_snapshot=state.planner_runtime_snapshot,
        ),
    )


def _make_step_loop_config(  # noqa: PLR0913
    *,
    config: RobotSimulationConfig,
    policy_fn: Any,
    planner_runtime: PlannerRuntime,
    noise: NoiseConfig,
    tracking_precision_spec: TrackingPrecisionSpec,
    tracking_precision_rng: np.random.Generator,
    safety_wrapper_runtime: SafetyWrapperRuntimeConfig,
    safety_wrapper_deadlock_monitor: DeadlockRecoveryMonitor | None,
    cbf_runtime: CBFSafetyFilterRuntimeConfig,
    actuation_controller: SyntheticActuationController | None,
    algo_meta: AlgoMeta,
    record_forces: bool,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
    single_pedestrian_intent_metadata: Any,
    single_pedestrian_vru_metadata: Any,
    hybrid_source_field: str | None,
    active_harness: Any,
    collision_event_context: _CollisionEventContext,
) -> _StepLoopConfig:
    """Build the read-only step-loop configuration bundle.

    Returns:
        _StepLoopConfig: Immutable per-episode configuration.
    """
    return _StepLoopConfig(
        config=config,
        policy_fn=policy_fn,
        planner_native_action=planner_runtime.planner_native_action,
        noise_spec=noise.spec,
        noise_rng=noise.rng,
        noise_state=noise.state,
        noise_stats=noise.stats,
        tracking_precision_spec=tracking_precision_spec,
        tracking_precision_rng=tracking_precision_rng,
        safety_wrapper_runtime=safety_wrapper_runtime,
        safety_wrapper_deadlock_monitor=safety_wrapper_deadlock_monitor,
        cbf_runtime=cbf_runtime,
        actuation_controller=actuation_controller,
        algo_meta=algo_meta,
        record_forces=record_forces,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
        single_pedestrian_intent_metadata=single_pedestrian_intent_metadata,
        single_pedestrian_vru_metadata=single_pedestrian_vru_metadata,
        hybrid_source_field=hybrid_source_field,
        active_harness=active_harness,
        collision_event_context=collision_event_context,
    )


def _execute_step_loop(
    state: _StepLoopState,
    slc: _StepLoopConfig,
    *,
    env: Any,
    planner_stats: Any,
    horizon_val: int,
) -> None:
    """Run the per-step episode loop, mutating ``state`` in place."""
    for step_idx in range(horizon_val):
        if slc.active_harness is not None:
            slc.active_harness.start_cycle()
        policy_command, _ = _step_policy_inference(state, slc, env=env)
        step_is_native, planner_step_decision = _step_hybrid_and_planner_stats(
            state,
            slc,
            planner_stats=planner_stats,
        )
        policy_command, actuation_step = _step_actuation_and_tracking(
            state,
            slc,
            policy_command=policy_command,
            step_is_native=step_is_native,
        )
        policy_command = _step_safety_filters(
            state,
            slc,
            policy_command=policy_command,
            step_is_native=step_is_native,
            env=env,
            step_idx=step_idx,
        )
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
            sel_payload,
            applied_environment_action_payload,
        ) = _step_convert_and_execute(
            state,
            slc,
            policy_command=policy_command,
            step_is_native=step_is_native,
            env=env,
        )
        (robot_pos, peds, forces_arr, heading, s_vis, s_conf, s_stat, s_reason) = (
            _step_snapshot_and_record(
                state,
                slc,
                env=env,
                obs=obs,
            )
        )
        sim = _StepSimResult(
            robot_pos=robot_pos,
            peds=peds,
            forces_arr=forces_arr,
            heading=heading,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            step_visible=s_vis,
            step_confidence=s_conf,
            step_visibility_status=s_stat,
            step_visibility_reason=s_reason,
            selected_action_payload=sel_payload,
            applied_environment_action_payload=applied_environment_action_payload,
            actuation_step=actuation_step,
            planner_step_decision=planner_step_decision,
        )
        _step_build_simulation_trace(state, slc, step_idx=step_idx, sim=sim)
        _step_build_actuation_trace(state, step_idx=step_idx, sim=sim)
        _step_build_planner_decision_entry(state, slc, step_idx=step_idx, sim=sim)
        if _step_collision_and_termination(state, slc, step_idx=step_idx, sim=sim):
            break


def _setup_and_run_step_loop(args: _StepLoopSetupArgs) -> _EpisodeStepLoopResult:
    """Create env, run the step loop, teardown, and return the result bundle.

    Returns:
        _EpisodeStepLoopResult: Immutable bundle of trajectory and outcome data.
    """
    policy_fn = args.planner_runtime.policy_fn
    env = make_robot_env(config=args.config, seed=int(args.seed), debug=False)
    state: _StepLoopState | None = None
    try:
        active_harness = LatencyMeasurementHarness.get_current()
        if active_harness is not None:
            policy_fn = active_harness.wrap_policy(policy_fn)
        obs = _prepare_episode_env(
            env,
            seed=args.seed,
            scenario=args.scenario,
            planner_bind_env=args.planner_runtime.planner_bind_env,
            planner_reset=args.planner_runtime.planner_reset,
            expected_population_size=args.expected_population_size,
            pedestrian_control_trace_label_builder=args.pedestrian_control_trace_label_builder,
        )
        state = _init_step_loop_state(
            obs=obs,
            env=env,
            config=args.config,
            hybrid_source_field=args.hybrid_source_field,
        )
        slc = _make_step_loop_config(
            config=args.config,
            policy_fn=policy_fn,
            planner_runtime=args.planner_runtime,
            noise=args.noise,
            tracking_precision_spec=args.tracking_precision_spec,
            tracking_precision_rng=args.tracking_precision_rng,
            safety_wrapper_runtime=args.safety_wrapper_runtime,
            safety_wrapper_deadlock_monitor=args.safety_wrapper_deadlock_monitor,
            cbf_runtime=args.cbf_runtime,
            actuation_controller=args.actuation_controller,
            algo_meta=args.algo_meta,
            record_forces=args.record_forces,
            record_planner_decision_trace=args.record_planner_decision_trace,
            record_simulation_step_trace=args.record_simulation_step_trace,
            single_pedestrian_intent_metadata=args.single_pedestrian_intent_metadata,
            single_pedestrian_vru_metadata=args.single_pedestrian_vru_metadata,
            hybrid_source_field=args.hybrid_source_field,
            active_harness=active_harness,
            collision_event_context=_make_collision_event_context(args.config, state.map_def),
        )
        _execute_step_loop(
            state,
            slc,
            env=env,
            planner_stats=args.planner_runtime.planner_stats,
            horizon_val=args.horizon_val,
        )
        state.planner_obstacle_force_law_metadata = _read_policy_obstacle_force_law_metadata(
            args.planner_runtime.policy_fn
        )
        if getattr(env, "simulator", None) is not None:
            state.simulator_obstacle_force_law_metadata = _read_obstacle_force_law_metadata(env)
            state.map_def = env.simulator.map_def
            state.goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
    finally:
        _teardown_step_loop(
            env,
            planner_stats=args.planner_runtime.planner_stats,
            planner_close=args.planner_runtime.planner_close,
            state=state,
        )
    # ``state`` is only ``None`` when setup raised before the loop body ran; in
    # that case the exception propagates through ``finally`` and control never
    # reaches here, so the narrowed type below is sound.
    assert state is not None
    return _build_step_loop_result(state)


def _run_episode_step_loop(  # noqa: PLR0913
    *,
    seed: int,
    scenario: dict[str, object] | None = None,
    config: RobotSimulationConfig,
    horizon_val: int,
    planner_runtime: PlannerRuntime,
    noise: NoiseConfig,
    tracking_precision_spec: TrackingPrecisionSpec,
    tracking_precision_rng: np.random.Generator,
    safety_wrapper_runtime: SafetyWrapperRuntimeConfig,
    safety_wrapper_deadlock_monitor: DeadlockRecoveryMonitor | None,
    cbf_runtime: CBFSafetyFilterRuntimeConfig,
    actuation_controller: SyntheticActuationController | None,
    algo_meta: AlgoMeta,
    record_forces: bool,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
    single_pedestrian_intent_metadata: list[dict[str, object] | None],
    single_pedestrian_vru_metadata: list[dict[str, object] | None],
    pedestrian_control_trace_label_builder: PedestrianControlTraceLabelBuilder | None = None,
    expected_population_size: int | None = None,
) -> _EpisodeStepLoopResult:
    """Run the env reset, the per-step episode loop, and planner/env teardown.

    Returns:
        _EpisodeStepLoopResult: Immutable bundle of every trajectory, trace, and
        outcome flag produced by the step loop, plus the planner runtime snapshot
        captured in the ``finally`` teardown.
    """
    _algo_key = (
        str(algo_meta.get("canonical_algorithm", algo_meta.get("algorithm", ""))).strip().lower()
    )
    hybrid_source_field = {
        "hybrid_portfolio": "selected_head",
        "hybrid_rule_local_planner": "selected_source",
    }.get(_algo_key)
    return _setup_and_run_step_loop(
        _StepLoopSetupArgs(
            seed=seed,
            scenario=scenario,
            config=config,
            horizon_val=horizon_val,
            planner_runtime=planner_runtime,
            noise=noise,
            tracking_precision_spec=tracking_precision_spec,
            tracking_precision_rng=tracking_precision_rng,
            safety_wrapper_runtime=safety_wrapper_runtime,
            safety_wrapper_deadlock_monitor=safety_wrapper_deadlock_monitor,
            cbf_runtime=cbf_runtime,
            actuation_controller=actuation_controller,
            algo_meta=algo_meta,
            record_forces=record_forces,
            record_planner_decision_trace=record_planner_decision_trace,
            record_simulation_step_trace=record_simulation_step_trace,
            single_pedestrian_intent_metadata=single_pedestrian_intent_metadata,
            single_pedestrian_vru_metadata=single_pedestrian_vru_metadata,
            pedestrian_control_trace_label_builder=pedestrian_control_trace_label_builder,
            expected_population_size=expected_population_size,
            hybrid_source_field=hybrid_source_field,
        )
    )


def _finalize_adapter_impact_metadata(
    algo_meta: AlgoMeta,
    *,
    algo: str,
    robot_kinematics: str,
    active_observation_mode: str,
    active_observation_level: str,
    benchmark_track: str | None,
    track_schema_version: str | None,
) -> AlgoMeta:
    """Resolve adapter-impact status and re-enrich metadata when applicable.

    Returns:
        AlgoMeta: The enriched algorithm metadata.
    """
    impact = algo_meta.get("adapter_impact")
    if isinstance(impact, dict) and bool(impact.get("requested", False)):
        native_steps = int(impact.get("native_steps", 0))
        adapted_steps = int(impact.get("adapted_steps", 0))
        total = native_steps + adapted_steps
        if total > 0:
            execution_mode = infer_execution_mode_from_counts(native_steps, adapted_steps)
            impact["status"] = "complete"
            impact["execution_mode"] = execution_mode
            impact["adapter_fraction"] = float(adapted_steps / total)
            algo_meta = cast(
                "AlgoMeta",
                enrich_algorithm_metadata(
                    algo=algo,
                    metadata=cast("dict[str, Any]", algo_meta),
                    execution_mode=execution_mode,
                    robot_kinematics=robot_kinematics,
                    observation_mode=active_observation_mode,
                    observation_level=active_observation_level,
                ),
            )
            attach_track_metadata(
                cast("dict[str, Any]", algo_meta),
                benchmark_track=benchmark_track,
                track_schema_version=track_schema_version,
                observation_level=active_observation_level,
                observation_mode=active_observation_mode,
            )
        else:
            impact["status"] = "not_applicable"
            impact["adapter_fraction"] = 0.0
    return algo_meta


def _finalize_planner_runtime_metadata(
    algo_meta: AlgoMeta,
    planner_runtime_snapshot: dict[str, Any] | None,
    *,
    algo: str,
    robot_kinematics: str,
    active_observation_mode: str,
    active_observation_level: str,
    benchmark_track: str | None,
    track_schema_version: str | None,
) -> AlgoMeta:
    """Attach planner runtime snapshot and foresight prediction to metadata.

    Returns:
        AlgoMeta: The enriched algorithm metadata.
    """
    if not isinstance(planner_runtime_snapshot, dict):
        return algo_meta
    algo_meta["planner_runtime"] = planner_runtime_snapshot
    foresight = planner_runtime_snapshot.get("foresight_prediction")
    if isinstance(foresight, Mapping):
        # Issue #6190: policy builders expose live predictive-foresight
        # diagnostics through the standard planner-runtime snapshot. Copy
        # that episode-time provenance into the canonical metadata block
        # before enrichment so a model-load fallback becomes structurally
        # evidence-ineligible in map-runner records as well.
        algo_meta["foresight_prediction"] = dict(foresight)
        algo_meta = cast(
            "AlgoMeta",
            enrich_algorithm_metadata(
                algo=algo,
                metadata=cast("dict[str, Any]", algo_meta),
                robot_kinematics=robot_kinematics,
                observation_mode=active_observation_mode,
                observation_level=active_observation_level,
            ),
        )
        attach_track_metadata(
            cast("dict[str, Any]", algo_meta),
            benchmark_track=benchmark_track,
            track_schema_version=track_schema_version,
            observation_level=active_observation_level,
            observation_mode=active_observation_mode,
        )
    return algo_meta


def _finalize_trace_metadata(  # noqa: PLR0913
    algo_meta: AlgoMeta,
    *,
    config: RobotSimulationConfig,
    initial_goal_distance: float,
    planner_decision_trace: list[PlannerDecisionTraceEntry],
    simulation_step_trace: list[dict[str, Any]],
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
    record_forces: bool,
    scenario: dict[str, Any],
    ped_pos_arr: np.ndarray,
    ped_forces_arr: np.ndarray,
    robot_pos_arr: np.ndarray,
    robot_config: Any,
    initial_robot_pos: np.ndarray,
    initial_robot_heading: float,
    initial_ped_positions: np.ndarray,
    initial_robot_velocity: np.ndarray | None,
    initial_ped_velocities: np.ndarray | None,
    trace_actor_ids: list[str] | None,
    horizon_val: int,
    termination_reason: str,
    safety_events: list[dict[str, Any]],
) -> None:
    """Attach planner-decision and simulation-step traces to algorithm metadata."""
    if record_planner_decision_trace:
        planner_trace: PlannerDecisionTrace = {
            "schema_version": "planner-decision-trace.v1",
            "dt": float(config.sim_config.time_per_step_in_secs),
            "initial_goal_distance_m": initial_goal_distance,
            "steps": planner_decision_trace,
        }
        algo_meta["planner_decision_trace"] = planner_trace
        topology_episode = _topology_guided_episode_diagnostics(planner_decision_trace)
        if topology_episode is not None:
            algo_meta["topology_guided_episode"] = topology_episode
    if record_simulation_step_trace:
        algo_meta["simulation_step_trace"] = {
            "schema_version": "simulation-step-trace.v1",
            "dt": float(config.sim_config.time_per_step_in_secs),
            "initial_goal_distance_m": initial_goal_distance,
            "steps": simulation_step_trace,
        }
        attach_pedestrian_control_trace(
            cast("dict[str, Any]", algo_meta),
            scenario=scenario,
            ped_positions=ped_pos_arr,
            ped_forces=ped_forces_arr if record_forces else None,
            dt=float(config.sim_config.time_per_step_in_secs),
            robot_positions=robot_pos_arr,
            robot_radius=float(getattr(robot_config, "radius", 1.0)),
            ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),
        )
        telemetry = telemetry_from_scenario(scenario)
        if telemetry.analysis_enabled:
            robot_radius = _finite_positive_float(getattr(robot_config, "radius", None))
            pedestrian_radius = _finite_positive_float(
                getattr(config.sim_config, "ped_radius", None)
            )
            if robot_radius is None or pedestrian_radius is None:
                algo_meta["analysis_trace_unavailable"] = {
                    "status": "unavailable",
                    "reason": "actor_radius_unavailable",
                }
                algo_meta["telemetry"] = telemetry.to_mapping()
                return
            upstream_reference = algo_meta.get("upstream_reference")
            planner_commit = (
                upstream_reference.get("commit")
                if isinstance(upstream_reference, Mapping)
                else None
            )
            if not planner_commit:
                planner_commit = algo_meta.get("planner_commit")
            analysis_trace = build_analysis_trace(
                steps=simulation_step_trace,
                initial_robot_position=initial_robot_pos,
                initial_robot_heading=initial_robot_heading,
                initial_pedestrians=initial_ped_positions,
                initial_robot_velocity=initial_robot_velocity,
                initial_pedestrian_velocities=initial_ped_velocities,
                initial_pedestrian_ids=trace_actor_ids,
                initial_pedestrian_id_source=(
                    "simulator_slot" if trace_actor_ids is not None else None
                ),
                dt=float(config.sim_config.time_per_step_in_secs),
                horizon=int(horizon_val),
                robot_radius_m=robot_radius,
                pedestrian_radius_m=pedestrian_radius,
                scenario=scenario,
                planner=str(algo_meta.get("algorithm") or algo_meta.get("algo") or "unknown"),
                planner_commit=str(planner_commit) if planner_commit else None,
                config_hash=_config_hash(
                    {
                        key: value
                        for key, value in scenario.items()
                        if key not in {"seed", "repeats"}
                    }
                ),
                git_hash=_git_hash_fallback(),
                termination_reason=termination_reason,
                safety_events=safety_events,
            )
            algo_meta["analysis_trace"] = analysis_trace
            algo_meta["telemetry"] = telemetry.to_mapping()


def _finalize_safety_summaries(  # noqa: PLR0913
    algo_meta: AlgoMeta,
    *,
    tracking_precision_spec: TrackingPrecisionSpec,
    tracking_precision_records: list[dict[str, Any]],
    min_separation_corrupted_values: list[float],
    safety_wrapper_runtime: SafetyWrapperRuntimeConfig,
    safety_wrapper_trace: list[dict[str, Any]],
    cbf_runtime: CBFSafetyFilterRuntimeConfig,
    cbf_filter_trace: list[dict[str, Any]],
    config: RobotSimulationConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Build tracking-precision, safety-wrapper, and CBF summaries.

    Returns:
        Tuple of (tracking_precision_summary, safety_wrapper_summary, cbf_filter_summary).
    """
    tracking_precision_summary = _build_tracking_precision_summary(
        spec=tracking_precision_spec,
        records=tracking_precision_records,
        min_separation_corrupted_values=min_separation_corrupted_values,
    )
    algo_meta["tracking_precision"] = tracking_precision_summary
    safety_wrapper_summary: dict[str, Any] | None = None
    if safety_wrapper_runtime.enabled:
        safety_wrapper_summary = summarize_safety_wrapper_trace(
            safety_wrapper_trace,
            runtime=safety_wrapper_runtime,
            time_per_step_s=float(config.sim_config.time_per_step_in_secs),
        )
        algo_meta["safety_wrapper"] = safety_wrapper_summary
    cbf_filter_summary: dict[str, Any] | None = None
    if cbf_runtime.enabled:
        cbf_filter_summary = summarize_cbf_safety_filter_trace(
            cbf_filter_trace,
            runtime=cbf_runtime,
        )
        algo_meta["cbf_safety_filter"] = cbf_filter_summary
    return tracking_precision_summary, safety_wrapper_summary, cbf_filter_summary


def _finalize_behavior_metadata(  # noqa: PLR0913
    algo_meta: AlgoMeta,
    *,
    scenario: dict[str, Any],
    single_pedestrian_intent_metadata: Any,
    single_pedestrian_vru_metadata: Any,
    actuation_controller: Any,
    actuation_profile: Any,
    synthetic_actuation_trace: list[dict[str, Any]],
    latency_profile: Any,
    config: RobotSimulationConfig,
    initial_goal_distance: float,
    robot_pos_arr: np.ndarray,
    robot_vel_arr: np.ndarray,
    ped_pos_arr: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach behavior summaries to algorithm metadata.

    Returns:
        Tuple of (actuation_summary, public_requirement_events).
    """
    intent_summary = _intent_conditioned_behavior_summary(
        scenario,
        single_pedestrian_intent_metadata,
    )
    if intent_summary is not None:
        algo_meta["intent_conditioned_behavior"] = intent_summary
    vru_summary = _cyclist_like_vru_summary(scenario, single_pedestrian_vru_metadata)
    if vru_summary is not None:
        algo_meta["cyclist_like_vru"] = vru_summary
    fast_bicycle_summary = _fast_bicycle_actor_summary(scenario, single_pedestrian_vru_metadata)
    if fast_bicycle_summary is not None:
        algo_meta["fast_bicycle_actor"] = fast_bicycle_summary
    actuation_summary: dict[str, Any] = not_available_saturation_metrics()
    if actuation_controller is not None and actuation_profile is not None:
        actuation_summary = actuation_controller.summary()
        algo_meta["synthetic_actuation"] = {
            "profile": actuation_profile.to_metadata(),
            "summary": dict(actuation_summary),
            "trace": {
                "schema_version": "synthetic-actuation-step-trace.v1",
                "dt": float(config.sim_config.time_per_step_in_secs),
                "initial_goal_distance_m": initial_goal_distance,
                "steps": synthetic_actuation_trace,
            },
        }
    if latency_profile is not None:
        algo_meta["latency_stress"] = {
            "profile": latency_profile.to_metadata(dt=config.sim_config.time_per_step_in_secs),
            "metrics": not_available_latency_metrics(),
        }
    public_requirement_events = evaluate_public_requirement_events(
        scenario=scenario,
        robot_positions=robot_pos_arr,
        robot_velocities=robot_vel_arr,
        ped_positions=ped_pos_arr,
        dt=float(config.sim_config.time_per_step_in_secs),
    )
    if public_requirement_events["status"] != "not_applicable":
        algo_meta["public_requirement"] = public_requirement_events
    visibility_settings = getattr(config, "observation_visibility", None)
    if visibility_settings is not None and hasattr(visibility_settings, "to_metadata"):
        algo_meta["observation_visibility"] = visibility_settings.to_metadata()
    return actuation_summary, public_requirement_events


def _finalize_episode_metrics(  # noqa: PLR0913
    metrics_raw: dict[str, Any],
    *,
    algo_meta: AlgoMeta,
    actuation_controller: Any,
    actuation_summary: dict[str, Any],
    tracking_precision_summary: dict[str, Any],
    tracking_precision_spec: TrackingPrecisionSpec,
    safety_wrapper_summary: dict[str, Any] | None,
    cbf_filter_summary: dict[str, Any] | None,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    """Post-process raw metrics and attach actuation/tracking/wrapper fields.

    Returns:
        dict[str, Any]: The finalized metrics dictionary.
    """
    shield_stats = algo_meta.get("shield_stats")
    if isinstance(shield_stats, dict):
        metrics_raw.update(shield_metrics_from_stats(shield_stats))
    metrics = post_process_metrics(
        metrics_raw,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )
    if actuation_controller is not None:
        for metric_name, metric_value in actuation_summary.items():
            if metric_name in {
                "schema_version",
                "status",
                "step_count",
                "command_clip_steps",
                "yaw_rate_saturation_steps",
            }:
                continue
            metrics[metric_name] = metric_value
    metrics["min_separation_corrupted_m"] = tracking_precision_summary["min_separation_corrupted_m"]
    metrics["tracking_contract_honored"] = bool(tracking_precision_summary["contract_honored"])
    metrics["tracking_contract_honored_rate"] = float(
        tracking_precision_summary["contract_honored_rate"]
    )
    metrics["tracking_target_motp_m"] = float(tracking_precision_spec["target_motp_m"])
    if safety_wrapper_summary is not None:
        metrics["wrapper_intervention_rate"] = float(safety_wrapper_summary["intervention_rate"])
    if cbf_filter_summary is not None:
        metrics["cbf_filter_intervention_rate"] = float(cbf_filter_summary["intervention_rate"])
        metrics["cbf_filter_qp_infeasible_rate"] = float(cbf_filter_summary["qp_infeasible_rate"])
        metrics["cbf_filter_fallback_rate"] = float(cbf_filter_summary["fallback_rate"])
    metrics["metric_values"] = _paired_effect_metric_values(
        metrics_raw=metrics_raw,
        metrics=metrics,
    )
    return metrics


def _paired_effect_metric_values(
    *,
    metrics_raw: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, float | None]:
    """Emit the #6970 paired-effect retained-row ``metric_values`` mapping.

    Derives the five fields whose source predicates exist on the map_runner
    path today. A field whose source is unavailable is omitted from the mapping
    (never a fabricated zero) so the existing #6970 gate fails closed. The
    remaining three contract fields (``false_positive_stop_rate``,
    ``stop_yield_latency_s``, ``progress_at_timeout``) await the versioned
    counterfactual-window contract clarification and are not emitted here.

    Returns:
        The retained-row ``metric_values`` mapping with only the fields whose
        sources were available, or an empty mapping when no source is present.
    """
    raw_success = metrics_raw.get("success")
    raw_collisions = metrics_raw.get("collisions")
    raw_near_misses = metrics_raw.get("near_misses")
    raw_min_distance = metrics_raw.get("min_distance")

    values: dict[str, float | None] = {}
    if isinstance(raw_collisions, (int, float)) and not isinstance(raw_collisions, bool):
        values["exact_collision_probability"] = 1.0 if raw_collisions > 0 else 0.0
    if isinstance(raw_near_misses, (int, float)) and not isinstance(raw_near_misses, bool):
        values["near_miss_probability"] = 1.0 if raw_near_misses > 0 else 0.0
    if isinstance(raw_min_distance, (int, float)) and not isinstance(raw_min_distance, bool):
        values["min_predicted_separation_m"] = float(raw_min_distance)
    if isinstance(raw_success, (int, float)) and not isinstance(raw_success, bool):
        values["completion_probability"] = 1.0 if raw_success > 0 else 0.0
    if isinstance(metrics.get("wrapper_intervention_rate"), (int, float)):
        values["wrapper_intervention_rate"] = float(metrics["wrapper_intervention_rate"])
    return values


def _build_episode_record_dict(  # noqa: PLR0913
    *,
    scenario_id: str,
    seed: int,
    scenario_params: dict[str, Any],
    metrics: dict[str, Any],
    safety_predicates: dict[str, Any],
    public_requirement_events: dict[str, Any],
    algo_meta: AlgoMeta,
    noise_spec: Any,
    noise_stats: dict[str, int],
    tracking_precision_spec: TrackingPrecisionSpec,
    algo: str,
    active_observation_mode: str,
    active_observation_level: str,
    ts_start: str,
    ts_end: str,
    status: str,
    steps_taken: int,
    horizon_val: int,
    wall_time: float,
    termination_reason: str,
    outcome: dict[str, Any],
    contradictions: list[str],
    view_integrity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the core episode record dictionary.

    Returns:
        dict[str, Any]: The episode record before provenance attachment.
    """
    analysis_trace = algo_meta.get("analysis_trace")
    analysis_trace = analysis_trace if isinstance(analysis_trace, Mapping) else {}
    provenance = {
        "artifact_uri": scenario_params.get("artifact_uri"),
        "artifact_sha256": analysis_trace.get("artifact_sha256"),
        "map_digest": analysis_trace.get("map_digest"),
        "scenario_digest": analysis_trace.get("scenario_digest"),
        "config_hash": analysis_trace.get("config_hash") or _config_hash(scenario_params),
        "git_hash": analysis_trace.get("git_hash") or _git_hash_fallback(),
        "planner_commit": analysis_trace.get("planner_commit"),
        "telemetry_profile": algo_meta.get("telemetry"),
    }
    return {
        "version": "v1",
        "episode_id": _compute_map_episode_id(scenario_params, seed),
        "scenario_id": scenario_id,
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        "safety_predicates": safety_predicates,
        "public_requirement": public_requirement_events,
        "algorithm_metadata": algo_meta,
        "observation_noise": noise_spec,
        "observation_noise_hash": observation_noise_hash(cast("dict[str, Any]", noise_spec)),
        "observation_noise_stats": noise_stats,
        "tracking_precision": tracking_precision_spec,
        "tracking_precision_hash": tracking_precision_hash(
            cast("dict[str, Any]", tracking_precision_spec)
        ),
        "algo": algo,
        "observation_mode": active_observation_mode,
        "observation_level": active_observation_level,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "provenance": provenance,
        "timestamps": {"start": ts_start, "end": ts_end},
        "status": status,
        "steps": steps_taken,
        "horizon": horizon_val,
        "wall_time_sec": wall_time,
        "timing": {"steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0},
        "termination_reason": termination_reason,
        "outcome": outcome,
        "integrity": {
            "contradictions": contradictions,
            "effective_view": view_integrity,
        },
    }


def _finalize_metadata_outputs(
    algo_meta: AlgoMeta,
    *,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    options: _MetadataFinalizationOptions,
) -> tuple[
    AlgoMeta,
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
]:
    """Finalize trace, safety, and behavior metadata after runtime metadata.

    Returns:
        Tuple of (algo_meta, tp_summary, sw_summary, cbf_summary,
        actuation_summary, public_requirement_events).
    """
    config = ctx.config
    robot_pos_arr = post_loop.robot_pos_arr
    robot_config = getattr(config, "robot_config", None) if robot_pos_arr.size else None
    _finalize_trace_metadata(
        algo_meta,
        config=config,
        initial_goal_distance=loop_result.initial_goal_distance,
        planner_decision_trace=loop_result.planner_decision_trace,
        simulation_step_trace=loop_result.simulation_step_trace,
        record_planner_decision_trace=options.record_planner_decision_trace,
        record_simulation_step_trace=options.record_simulation_step_trace,
        record_forces=options.record_forces,
        scenario=ctx.scenario,
        ped_pos_arr=post_loop.ped_pos_arr,
        ped_forces_arr=post_loop.ped_forces_arr,
        robot_pos_arr=robot_pos_arr,
        robot_config=robot_config,
        initial_robot_pos=loop_result.initial_robot_pos,
        initial_robot_heading=loop_result.initial_robot_heading,
        initial_ped_positions=loop_result.initial_ped_positions,
        initial_robot_velocity=loop_result.initial_robot_velocity,
        initial_ped_velocities=loop_result.initial_ped_velocities,
        trace_actor_ids=loop_result.trace_actor_ids,
        horizon_val=ctx.horizon_val,
        termination_reason=loop_result.termination_reason,
        safety_events=loop_result.collision_events,
    )
    tp_summary, sw_summary, cbf_summary = _finalize_safety_summaries(
        algo_meta,
        tracking_precision_spec=ctx.tracking_precision_spec,
        tracking_precision_records=loop_result.tracking_precision_records,
        min_separation_corrupted_values=loop_result.min_separation_corrupted_values,
        safety_wrapper_runtime=ctx.safety_wrapper_runtime,
        safety_wrapper_trace=loop_result.safety_wrapper_trace,
        cbf_runtime=ctx.cbf_runtime,
        cbf_filter_trace=loop_result.cbf_filter_trace,
        config=config,
    )
    actuation_summary, public_requirement_events = _finalize_behavior_metadata(
        algo_meta,
        scenario=ctx.scenario,
        single_pedestrian_intent_metadata=options.single_pedestrian_intent_metadata,
        single_pedestrian_vru_metadata=options.single_pedestrian_vru_metadata,
        actuation_controller=options.actuation_controller,
        actuation_profile=ctx.actuation_profile,
        synthetic_actuation_trace=loop_result.synthetic_actuation_trace,
        latency_profile=ctx.latency_profile,
        config=config,
        initial_goal_distance=loop_result.initial_goal_distance,
        robot_pos_arr=robot_pos_arr,
        robot_vel_arr=post_loop.robot_vel_arr,
        ped_pos_arr=post_loop.ped_pos_arr,
    )
    return (
        algo_meta,
        tp_summary,
        sw_summary,
        cbf_summary,
        actuation_summary,
        public_requirement_events,
    )


def _finalize_metadata_phase(
    algo_meta: AlgoMeta,
    *,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    options: _MetadataFinalizationOptions,
) -> tuple[
    AlgoMeta,
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
]:
    """Finalize runtime metadata before trace, safety, and behavior outputs.

    Returns:
        Tuple containing finalized metadata and derived summaries.
    """
    algo_meta = _finalize_adapter_impact_metadata(
        algo_meta,
        algo=ctx.algo,
        robot_kinematics=ctx.robot_kinematics,
        active_observation_mode=options.active_observation_mode,
        active_observation_level=options.active_observation_level,
        benchmark_track=ctx.benchmark_track,
        track_schema_version=ctx.track_schema_version,
    )
    _finalize_feasibility_metadata(cast("dict[str, Any]", algo_meta))
    algo_meta["ammv_feasibility"] = evaluate_artifact_command_feasibility(
        loop_result.ammv_command_actions
    )
    algo_meta = _finalize_planner_runtime_metadata(
        algo_meta,
        loop_result.planner_runtime_snapshot,
        algo=ctx.algo,
        robot_kinematics=ctx.robot_kinematics,
        active_observation_mode=options.active_observation_mode,
        active_observation_level=options.active_observation_level,
        benchmark_track=ctx.benchmark_track,
        track_schema_version=ctx.track_schema_version,
    )
    if loop_result.obstacle_force_law_metadata is not None:
        algo_meta["obstacle_force_law"] = deepcopy(loop_result.obstacle_force_law_metadata)
    return _finalize_metadata_outputs(
        algo_meta,
        ctx=ctx,
        loop_result=loop_result,
        post_loop=post_loop,
        options=options,
    )


def _build_scenario_params(  # noqa: PLR0913
    ctx: _EpisodeRunContext,
    *,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    active_observation_mode: str,
    active_observation_level: str,
    safety_wrapper: dict[str, Any] | None,
    cbf_safety_filter: dict[str, Any] | None,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
) -> dict[str, Any]:
    """Build the scenario identity payload for the episode record.

    Returns:
        dict[str, Any]: The scenario identity payload.
    """
    config = ctx.config
    return _scenario_identity_payload(
        ctx.scenario,
        algo=ctx.algo,
        algo_config=ctx.policy_cfg,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        observation_mode=active_observation_mode,
        observation_level=active_observation_level,
        benchmark_track=ctx.benchmark_track,
        track_schema_version=ctx.track_schema_version,
        observation_noise=cast("dict[str, Any]", ctx.noise_spec),
        tracking_precision=cast("dict[str, Any]", ctx.tracking_precision_spec),
        synthetic_actuation_profile=(
            ctx.actuation_profile.to_metadata() if ctx.actuation_profile is not None else None
        ),
        latency_stress_profile=(
            ctx.latency_profile.to_metadata(dt=config.sim_config.time_per_step_in_secs)
            if ctx.latency_profile is not None
            else None
        ),
        safety_wrapper=dict(safety_wrapper) if safety_wrapper is not None else None,
        cbf_safety_filter=dict(cbf_safety_filter) if cbf_safety_filter is not None else None,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
    )


def _finalize_record_provenance(  # noqa: PLR0913
    record: dict[str, Any],
    *,
    algo_meta: AlgoMeta,
    config: RobotSimulationConfig,
    policy_cfg: dict[str, Any],
    scenario: dict[str, Any],
    scenario_id: str,
    seed: int,
    scenario_params: dict[str, Any],
    robot_pos_arr: np.ndarray,
    ped_pos_arr: np.ndarray,
    goal_vec: np.ndarray,
    initial_goal_distance: float,
    termination_reason: str,
    outcome: dict[str, Any],
    collision_events: list[dict[str, Any]],
    planner_decision_trace: list[PlannerDecisionTraceEntry],
    route_complete: bool,
    collision_seen: bool,
    horizon_val: int,
    record_forces: bool,
    active_observation_mode: str,
    active_observation_level: str,
    noise_spec: Any,
    tracking_precision_spec: TrackingPrecisionSpec,
    benchmark_track: str | None,
    track_schema_version: str | None,
) -> None:
    """Attach provenance, evidence, event ledger, and track fields to the record."""
    pedestrian_model_provenance = build_pedestrian_model_provenance(
        sim_config=config.sim_config,
        policy_cfg=policy_cfg,
        algorithm_metadata=algo_meta,
    )
    attach_pedestrian_model_fields(record, pedestrian_model_provenance)
    _finalize_record_deadlock_and_evidence(
        record,
        algo_meta=algo_meta,
        config=config,
        scenario=scenario,
        robot_pos_arr=robot_pos_arr,
        ped_pos_arr=ped_pos_arr,
        goal_vec=goal_vec,
        initial_goal_distance=initial_goal_distance,
        termination_reason=termination_reason,
        outcome=outcome,
        collision_events=collision_events,
        planner_decision_trace=planner_decision_trace,
        route_complete=route_complete,
        collision_seen=collision_seen,
    )
    if benchmark_track is not None:
        record["benchmark_track"] = benchmark_track
    if track_schema_version is not None:
        record["track_schema_version"] = track_schema_version
    _finalize_result_provenance_block(
        record,
        scenario_id=scenario_id,
        seed=seed,
        scenario_params=scenario_params,
        config=config,
        horizon_val=horizon_val,
        record_forces=record_forces,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        noise_spec=noise_spec,
        tracking_precision_spec=tracking_precision_spec,
    )
    ensure_metric_parameters(record)


def _finalize_record_deadlock_and_evidence(  # noqa: PLR0913
    record: dict[str, Any],
    *,
    algo_meta: AlgoMeta,
    config: RobotSimulationConfig,
    scenario: dict[str, Any],
    robot_pos_arr: np.ndarray,
    ped_pos_arr: np.ndarray,
    goal_vec: np.ndarray,
    initial_goal_distance: float,
    termination_reason: str,
    outcome: dict[str, Any],
    collision_events: list[dict[str, Any]],
    planner_decision_trace: list[PlannerDecisionTraceEntry],
    route_complete: bool,
    collision_seen: bool,
) -> None:
    """Attach static-deadlock, native-command, and episode-evidence fields."""
    static_deadlock_fields = _static_deadlock_trace_fields(
        scenario,
        robot_pos_arr=robot_pos_arr,
        goal_vec=goal_vec,
        initial_goal_distance=initial_goal_distance,
        termination_reason=termination_reason,
        outcome=outcome,
        planner_decision_trace=planner_decision_trace,
    )
    record.update(static_deadlock_fields)
    # Keep native-command diagnostics in the canonical algorithm metadata block used by
    # the issue #5416 analyzer. The generic deadlock metric already lives under
    # ``metrics.deadlock``/``metrics.deadlock_stall``; the native detector's typed
    # trace is nested under the native command metadata rather than emitted as a
    # misleading top-level replacement.
    is_native_nc, deadlock_field, planner_diag = native_command_metadata_for_record(
        cast("dict[str, Any]", algo_meta)
    )
    if is_native_nc:
        algo_meta["planner_diagnostics"] = planner_diag
        native_metadata = algo_meta.get("native_command")
        if isinstance(native_metadata, dict):
            native_metadata["deadlock"] = deadlock_field
    # Write-time episode-row instrumentation for issue #4242 AC #2: emit native
    # failure-mechanism (fail-closed unknown) and interaction-exposure (computed
    # from this episode's trajectory) schema blocks so new campaigns carry them.
    record.update(
        _episode_evidence_fields(
            robot_pos_arr=robot_pos_arr,
            ped_pos_arr=ped_pos_arr,
            dt=float(config.sim_config.time_per_step_in_secs),
            success=route_complete and not collision_seen,
        )
    )
    record["event_ledger"] = build_event_ledger(record, collision_events=collision_events)


def _finalize_result_provenance_block(  # noqa: PLR0913
    record: dict[str, Any],
    *,
    scenario_id: str,
    seed: int,
    scenario_params: dict[str, Any],
    config: RobotSimulationConfig,
    horizon_val: int,
    record_forces: bool,
    active_observation_mode: str,
    active_observation_level: str,
    noise_spec: Any,
    tracking_precision_spec: TrackingPrecisionSpec,
) -> None:
    """Attach the result_provenance schema block to the record."""
    record["result_provenance"] = {
        "schema_version": "benchmark_row_provenance.v1",
        "scenario_id": scenario_id,
        "seed": int(seed),
        "config_hash": _config_hash(scenario_params),
        "repo_commit": _git_hash_fallback(),
        "simulator_settings": build_simulator_settings_provenance(
            horizon=horizon_val,
            dt=float(config.sim_config.time_per_step_in_secs),
            record_forces=bool(record_forces),
            active_observation_mode=active_observation_mode,
            active_observation_level=active_observation_level,
            noise_hash=observation_noise_hash(cast("dict[str, Any]", noise_spec)),
            tracking_precision_hash=tracking_precision_hash(
                cast("dict[str, Any]", tracking_precision_spec)
            ),
        ),
        "postprocessing": [
            {"step": "compute_all_metrics", "status": "completed"},
            {"step": "post_process_metrics", "status": "completed"},
        ],
    }


def _finalize_assembled_record_provenance(  # noqa: PLR0913
    record: dict[str, Any],
    *,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    algo_meta: AlgoMeta,
    scenario_params: dict[str, Any],
    outcome: dict[str, Any],
    record_forces: bool,
    active_observation_mode: str,
    active_observation_level: str,
) -> None:
    """Attach provenance and episode-evidence metadata to an assembled record."""
    _finalize_record_provenance(
        record,
        algo_meta=algo_meta,
        config=ctx.config,
        policy_cfg=ctx.policy_cfg,
        scenario=ctx.scenario,
        scenario_id=ctx.scenario_id,
        seed=record["seed"],
        scenario_params=scenario_params,
        robot_pos_arr=post_loop.robot_pos_arr,
        ped_pos_arr=post_loop.ped_pos_arr,
        goal_vec=loop_result.goal_vec,
        initial_goal_distance=loop_result.initial_goal_distance,
        termination_reason=loop_result.termination_reason,
        outcome=outcome,
        collision_events=loop_result.collision_events,
        planner_decision_trace=loop_result.planner_decision_trace,
        route_complete=loop_result.reached_goal_step is not None,
        collision_seen=loop_result.collision_seen,
        horizon_val=ctx.horizon_val,
        record_forces=record_forces,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        noise_spec=ctx.noise_spec,
        tracking_precision_spec=ctx.tracking_precision_spec,
        benchmark_track=ctx.benchmark_track,
        track_schema_version=ctx.track_schema_version,
    )


def _episode_outcome(loop_result: _EpisodeStepLoopResult) -> tuple[dict[str, bool], str]:
    """Build the standardized outcome payload and status for one step-loop result.

    Returns:
        Tuple of outcome payload and status label.
    """
    route_complete = loop_result.reached_goal_step is not None
    collision_seen = loop_result.collision_seen
    timeout_event = (
        not route_complete
        and not collision_seen
        and (
            loop_result.timeout_seen
            or loop_result.termination_reason
            in {
                "truncated",
                "max_steps",
            }
        )
    )
    return (
        build_outcome_payload(
            route_complete=route_complete,
            collision=collision_seen,
            timeout=timeout_event,
        ),
        status_from_termination_reason(loop_result.termination_reason),
    )


def _assemble_episode_record(  # noqa: PLR0913
    *,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    algo_meta: AlgoMeta,
    metrics: dict[str, Any],
    public_requirement_events: dict[str, Any],
    scenario_params: dict[str, Any],
    active_observation_mode: str,
    active_observation_level: str,
    seed: int,
    record_forces: bool,
) -> dict[str, Any]:
    """Build and finalize an episode record.

    Returns:
        dict[str, Any]: The finalized episode record.
    """
    robot_pos_arr = post_loop.robot_pos_arr
    steps_taken = int(robot_pos_arr.shape[0])
    wall_time = float(max(1e-9, time.time() - ctx.start_time))
    outcome, status = _episode_outcome(loop_result)
    contradictions = outcome_contradictions(
        termination_reason=loop_result.termination_reason,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        raise ValueError(
            f"Episode integrity contradictions for scenario '{ctx.scenario_id}', seed={seed}: "
            + "; ".join(contradictions)
        )
    record = _build_episode_record_dict(
        scenario_id=ctx.scenario_id,
        seed=seed,
        scenario_params=scenario_params,
        metrics=metrics,
        safety_predicates=post_loop.safety_predicates,
        public_requirement_events=public_requirement_events,
        algo_meta=algo_meta,
        noise_spec=ctx.noise_spec,
        noise_stats=ctx.noise_stats,
        tracking_precision_spec=ctx.tracking_precision_spec,
        algo=ctx.algo,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        ts_start=ctx.ts_start,
        ts_end=datetime.now(UTC).isoformat(),
        status=status,
        steps_taken=steps_taken,
        horizon_val=ctx.horizon_val,
        wall_time=wall_time,
        termination_reason=loop_result.termination_reason,
        outcome=outcome,
        contradictions=contradictions,
        view_integrity=loop_result.view_integrity,
    )
    runtime_law = record.get("algorithm_metadata", {}).get("obstacle_force_law")
    if isinstance(runtime_law, dict) and isinstance(runtime_law.get("sites"), dict):
        for site_metadata in runtime_law["sites"].values():
            if isinstance(site_metadata, dict):
                site_metadata.setdefault("config_hash", record["config_hash"])
                site_metadata.setdefault("source_commit", record["git_hash"])
    _finalize_assembled_record_provenance(
        record,
        ctx=ctx,
        loop_result=loop_result,
        post_loop=post_loop,
        algo_meta=algo_meta,
        scenario_params=scenario_params,
        outcome=outcome,
        record_forces=record_forces,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
    )
    return record


def _finalize_record_inner(  # noqa: PLR0913
    *,
    algo_meta: AlgoMeta,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    actuation_controller: Any,
    active_observation_mode: str,
    active_observation_level: str,
    single_pedestrian_intent_metadata: Any,
    single_pedestrian_vru_metadata: Any,
    seed: int,
    horizon: int | None,
    dt: float | None,
    safety_wrapper: dict[str, Any] | None,
    cbf_safety_filter: dict[str, Any] | None,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    record_forces: bool,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
) -> dict[str, Any]:
    """Run metadata, metrics, scenario-params, and record-assembly phases.

    Returns:
        dict[str, Any]: The finalized episode record.
    """
    (
        algo_meta,
        tp_summary,
        sw_summary,
        cbf_summary,
        actuation_summary,
        public_requirement_events,
    ) = _finalize_metadata_phase(
        algo_meta,
        ctx=ctx,
        loop_result=loop_result,
        post_loop=post_loop,
        options=_MetadataFinalizationOptions(
            actuation_controller=actuation_controller,
            active_observation_mode=active_observation_mode,
            active_observation_level=active_observation_level,
            single_pedestrian_intent_metadata=single_pedestrian_intent_metadata,
            single_pedestrian_vru_metadata=single_pedestrian_vru_metadata,
            record_forces=record_forces,
            record_planner_decision_trace=record_planner_decision_trace,
            record_simulation_step_trace=record_simulation_step_trace,
        ),
    )
    return _finalize_metrics_and_assemble_record(
        ctx=ctx,
        loop_result=loop_result,
        post_loop=post_loop,
        algo_meta=algo_meta,
        tracking_precision_summary=tp_summary,
        safety_wrapper_summary=sw_summary,
        cbf_filter_summary=cbf_summary,
        actuation_summary=actuation_summary,
        public_requirement_events=public_requirement_events,
        actuation_controller=actuation_controller,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        seed=seed,
        horizon=horizon,
        dt=dt,
        safety_wrapper=safety_wrapper,
        cbf_safety_filter=cbf_safety_filter,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        record_forces=record_forces,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
    )


def _finalize_metrics_and_assemble_record(  # noqa: PLR0913
    *,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    algo_meta: AlgoMeta,
    tracking_precision_summary: dict[str, Any],
    safety_wrapper_summary: dict[str, Any] | None,
    cbf_filter_summary: dict[str, Any] | None,
    actuation_summary: dict[str, Any],
    public_requirement_events: dict[str, Any],
    actuation_controller: Any,
    active_observation_mode: str,
    active_observation_level: str,
    seed: int,
    horizon: int | None,
    dt: float | None,
    safety_wrapper: dict[str, Any] | None,
    cbf_safety_filter: dict[str, Any] | None,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    record_forces: bool,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
) -> dict[str, Any]:
    """Build metrics, scenario parameters, and the final episode record.

    Returns:
        dict[str, Any]: The finalized episode record.
    """
    metrics = _finalize_episode_metrics(
        post_loop.metrics_raw,
        algo_meta=algo_meta,
        actuation_controller=actuation_controller,
        actuation_summary=actuation_summary,
        tracking_precision_summary=tracking_precision_summary,
        tracking_precision_spec=ctx.tracking_precision_spec,
        safety_wrapper_summary=safety_wrapper_summary,
        cbf_filter_summary=cbf_filter_summary,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )
    scenario_params = _build_scenario_params(
        ctx,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        safety_wrapper=safety_wrapper,
        cbf_safety_filter=cbf_safety_filter,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
    )
    return _assemble_episode_record(
        ctx=ctx,
        loop_result=loop_result,
        post_loop=post_loop,
        algo_meta=algo_meta,
        metrics=metrics,
        public_requirement_events=public_requirement_events,
        scenario_params=scenario_params,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        seed=seed,
        record_forces=record_forces,
    )


def _finalize_episode_record(  # noqa: PLR0913
    *,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    algo_meta: AlgoMeta,
    actuation_controller: Any,
    active_observation_mode: str,
    active_observation_level: str,
    single_pedestrian_intent_metadata: Any,
    single_pedestrian_vru_metadata: Any,
    seed: int,
    horizon: int | None,
    dt: float | None,
    safety_wrapper: dict[str, Any] | None,
    cbf_safety_filter: dict[str, Any] | None,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    record_forces: bool,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
) -> EpisodeRecordDict:
    """Assemble the benchmark JSONL record from the step-loop and post-loop results.

    Returns:
        EpisodeRecordDict: The finalized episode record with metrics, provenance, and
        planner metadata, mirroring the prior inline metadata-finalization phase.
    """
    # Finalization phase: isolate the episode metadata from the builder-provided
    # ``algo_meta`` so the finalization writes below cannot leak back into a
    # builder that reuses/caches the same dict across episodes (#4954).
    algo_meta = deepcopy(algo_meta)
    return cast(
        "EpisodeRecordDict",
        _finalize_record_inner(
            algo_meta=algo_meta,
            ctx=ctx,
            loop_result=loop_result,
            post_loop=post_loop,
            actuation_controller=actuation_controller,
            active_observation_mode=active_observation_mode,
            active_observation_level=active_observation_level,
            single_pedestrian_intent_metadata=single_pedestrian_intent_metadata,
            single_pedestrian_vru_metadata=single_pedestrian_vru_metadata,
            seed=seed,
            horizon=horizon,
            dt=dt,
            safety_wrapper=safety_wrapper,
            cbf_safety_filter=cbf_safety_filter,
            snqi_weights=snqi_weights,
            snqi_baseline=snqi_baseline,
            record_forces=record_forces,
            record_planner_decision_trace=record_planner_decision_trace,
            record_simulation_step_trace=record_simulation_step_trace,
        ),
    )


def run_map_episode(  # noqa: PLR0913
    scenario: dict[str, Any],
    seed: int,
    *,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    algo: str,
    scenario_path: Path,
    algo_config: dict[str, Any] | None = None,
    algo_config_path: str | None = None,
    adapter_impact_eval: bool = False,
    experimental_ped_impact: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    benchmark_track: str | None = None,
    track_schema_version: str | None = None,
    observation_noise: dict[str, Any] | None = None,
    tracking_precision: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    safety_wrapper: dict[str, Any] | None = None,
    cbf_safety_filter: dict[str, Any] | None = None,
    record_planner_decision_trace: bool = False,
    record_simulation_step_trace: bool = False,
    pedestrian_control_trace_label_builder: PedestrianControlTraceLabelBuilder | None = None,
    close_policy: bool = True,
    policy_builder: PolicyBuilder,
) -> EpisodeRecordDict:
    """Run one scenario/seed episode and return a benchmark JSONL record.

    Returns:
        EpisodeRecordDict: Episode record with metrics, provenance, and planner metadata.
    """
    ctx = _resolve_episode_run_context(
        scenario=scenario,
        seed=seed,
        horizon=horizon,
        dt=dt,
        algo=algo,
        scenario_path=scenario_path,
        algo_config=algo_config,
        algo_config_path=algo_config_path,
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
        observation_mode=observation_mode,
        observation_level=observation_level,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        observation_noise=observation_noise,
        tracking_precision=tracking_precision,
        synthetic_actuation_profile=synthetic_actuation_profile,
        latency_stress_profile=latency_stress_profile,
        safety_wrapper=safety_wrapper,
        cbf_safety_filter=cbf_safety_filter,
    )
    scenario = ctx.scenario
    telemetry_profile = telemetry_from_scenario(scenario)
    # The profile is a recording choice only.  It enables the legacy step trace
    # capture path but never changes policy inputs, actions, or simulator state.
    record_simulation_step_trace = bool(
        record_simulation_step_trace or telemetry_profile.analysis_enabled
    )
    ped_impact_radius_m = ctx.ped_impact_radius_m
    ped_impact_window_steps = ctx.ped_impact_window_steps
    benchmark_track = ctx.benchmark_track
    track_schema_version = ctx.track_schema_version
    noise_spec = ctx.noise_spec
    noise_rng = ctx.noise_rng
    noise_state = ctx.noise_state
    noise_stats = ctx.noise_stats
    tracking_precision_spec = ctx.tracking_precision_spec
    tracking_precision_rng = ctx.tracking_precision_rng
    safety_wrapper_runtime = ctx.safety_wrapper_runtime
    cbf_runtime = ctx.cbf_runtime
    safety_wrapper_deadlock_monitor = ctx.safety_wrapper_deadlock_monitor
    config = ctx.config
    horizon_val = ctx.horizon_val
    robot_kinematics = ctx.robot_kinematics
    actuation_profile = ctx.actuation_profile
    robot_command_mode = ctx.robot_command_mode
    algo = ctx.algo
    policy_cfg = ctx.policy_cfg
    # When a control-trace label builder is supplied the population is *forced* to
    # the declared ``population_size`` (issue #5666): the simulator honors it as an
    # exact ``force_population_size`` so the scenario instantiates exactly the
    # declared pedestrians regardless of map area or ``ped_density``. We must NOT
    # reset ``config.sim_config.population_size`` to ``None`` here (the prior bug):
    # doing so silently dropped the forced count and let ``ped_density * area`` win.
    simulation_config = scenario.get("simulation_config")
    expected_population_size = (
        int(simulation_config["population_size"])
        if isinstance(simulation_config, dict)
        and simulation_config.get("population_size") is not None
        else None
    )
    if pedestrian_control_trace_label_builder is not None:
        # Labels are rebuilt from the actually instantiated count after reset, so
        # the pre-reset labels must not leak in and misalign the trace.
        config.sim_config.pedestrian_control_trace_labels = None
    policy_contract = _prepare_policy_and_observation_contract(
        scenario=scenario,
        algo=algo,
        policy_cfg=policy_cfg,
        config=config,
        observation_mode=observation_mode,
        observation_level=observation_level,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
        adapter_impact_eval=adapter_impact_eval,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        actuation_profile=actuation_profile,
        policy_builder=policy_builder,
    )
    loop_result = _run_episode_step_loop(
        seed=seed,
        scenario=scenario,
        config=config,
        horizon_val=horizon_val,
        planner_runtime=PlannerRuntime(
            policy_fn=policy_contract.policy_fn,
            planner_bind_env=policy_contract.planner_bind_env,
            planner_reset=policy_contract.planner_reset,
            planner_close=policy_contract.planner_close if close_policy else None,
            planner_stats=policy_contract.planner_stats,
            planner_native_action=policy_contract.planner_native_action,
        ),
        noise=NoiseConfig(
            spec=noise_spec,
            rng=noise_rng,
            state=noise_state,
            stats=noise_stats,
        ),
        tracking_precision_spec=tracking_precision_spec,
        tracking_precision_rng=tracking_precision_rng,
        safety_wrapper_runtime=safety_wrapper_runtime,
        safety_wrapper_deadlock_monitor=safety_wrapper_deadlock_monitor,
        cbf_runtime=cbf_runtime,
        actuation_controller=policy_contract.actuation_controller,
        algo_meta=policy_contract.algo_meta,
        record_forces=record_forces,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
        single_pedestrian_intent_metadata=policy_contract.single_pedestrian_intent_metadata,
        single_pedestrian_vru_metadata=policy_contract.single_pedestrian_vru_metadata,
        pedestrian_control_trace_label_builder=pedestrian_control_trace_label_builder,
        expected_population_size=expected_population_size,
    )
    post_loop = _compute_post_loop_metrics(
        robot_positions=loop_result.robot_positions,
        robot_headings=loop_result.robot_headings,
        hybrid_command_sources=loop_result.hybrid_command_sources,
        ped_positions=loop_result.ped_positions,
        ped_forces=loop_result.ped_forces,
        visibility_trace=loop_result.visibility_trace,
        track_confidence_trace=loop_result.track_confidence_trace,
        visibility_evidence_statuses=loop_result.visibility_evidence_statuses,
        visibility_evidence_reasons=loop_result.visibility_evidence_reasons,
        reached_goal_step=loop_result.reached_goal_step,
        collision_seen=loop_result.collision_seen,
        ped_collision_seen=loop_result.ped_collision_seen,
        obstacle_collision_seen=loop_result.obstacle_collision_seen,
        robot_collision_seen=loop_result.robot_collision_seen,
        map_def=loop_result.map_def,
        goal_vec=loop_result.goal_vec,
        scenario=scenario,
        config=config,
        horizon_val=horizon_val,
        record_forces=record_forces,
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
    )
    return _finalize_episode_record(
        ctx=ctx,
        loop_result=loop_result,
        post_loop=post_loop,
        algo_meta=policy_contract.algo_meta,
        actuation_controller=policy_contract.actuation_controller,
        active_observation_mode=policy_contract.active_observation_mode,
        active_observation_level=policy_contract.active_observation_level,
        single_pedestrian_intent_metadata=policy_contract.single_pedestrian_intent_metadata,
        single_pedestrian_vru_metadata=policy_contract.single_pedestrian_vru_metadata,
        seed=seed,
        horizon=horizon,
        dt=dt,
        safety_wrapper=safety_wrapper,
        cbf_safety_filter=cbf_safety_filter,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        record_forces=record_forces,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
    )


__all__ = ["run_map_episode"]
