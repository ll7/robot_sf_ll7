"""Episode execution helpers for map-based benchmark batches."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
    resolve_learned_checkpoint_observation_contract,
)
from robot_sf.benchmark.ammv_feasibility import evaluate_artifact_command_feasibility
from robot_sf.benchmark.cbf_safety_filter_runtime import (
    apply_runtime_cbf_safety_filter,
    ineligible_cbf_safety_filter_step_record,
    summarize_cbf_safety_filter_trace,
)
from robot_sf.benchmark.cbf_safety_filter_runtime import (
    runtime_config_from_mapping as cbf_runtime_config_from_mapping,
)
from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.failure_mechanism_taxonomy import unknown_failure_mechanism_record
from robot_sf.benchmark.group_space_metrics import group_specs_from_map
from robot_sf.benchmark.interaction_exposure import (
    InteractionExposureError,
    compute_interaction_exposure_fields,
    not_derivable_interaction_exposure,
)
from robot_sf.benchmark.latency_stress import (
    LatencyMeasurementHarness,
    not_available_latency_metrics,
)
from robot_sf.benchmark.map_runner_actions import DEFAULT_KINEMATICS as _DEFAULT_KINEMATICS
from robot_sf.benchmark.map_runner_actions import (
    policy_command_to_env_action as _policy_command_to_env_action,
)
from robot_sf.benchmark.map_runner_actions import robot_kinematics_label as _robot_kinematics_label
from robot_sf.benchmark.map_runner_actions import robot_max_speed as _robot_max_speed
from robot_sf.benchmark.map_runner_actions import stack_ped_positions as _stack_ped_positions
from robot_sf.benchmark.map_runner_actions import vel_and_acc as _vel_and_acc
from robot_sf.benchmark.map_runner_env import (
    apply_active_observation_mode_to_env_config as _apply_active_observation_mode_to_env_config,
)
from robot_sf.benchmark.map_runner_env import (
    apply_policy_env_observation_overrides as _apply_policy_env_observation_overrides,
)
from robot_sf.benchmark.map_runner_env import build_env_config as _build_env_config
from robot_sf.benchmark.map_runner_env import (
    validate_sensor_fusion_adapter_config as _validate_sensor_fusion_adapter_config,
)
from robot_sf.benchmark.map_runner_identity import (
    _compute_map_episode_id,
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.map_runner_metrics import (
    floor_collision_metrics_from_flags as _floor_collision_metrics_from_flags,
)
from robot_sf.benchmark.map_runner_metrics import (
    normalize_pedestrian_impact_controls as _normalize_pedestrian_impact_controls,
)
from robot_sf.benchmark.map_runner_observations import normalize_xy_rows as _normalize_xy_rows
from robot_sf.benchmark.map_runner_policy_metadata import (
    finalize_feasibility_metadata as _finalize_feasibility_metadata,
)
from robot_sf.benchmark.map_runner_policy_resolution import (
    _apply_planner_selector_v2_context,
    _apply_scenario_uncertainty_envelope_config,
    _parse_algo_config,
    _resolve_policy_search_candidate_runtime,
)
from robot_sf.benchmark.map_runner_profile_metadata import (
    load_latency_profile as _load_latency_stress_profile,
)
from robot_sf.benchmark.map_runner_profile_metadata import (
    load_synthetic_actuation_profile as _load_synthetic_actuation_profile,
)
from robot_sf.benchmark.map_runner_static_deadlock import (
    static_deadlock_trace_fields as _static_deadlock_trace_fields,
)
from robot_sf.benchmark.map_runner_trace import (
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
from robot_sf.benchmark.map_runner_view_integrity import (
    DegeneratePlannerViewError,
    evaluate_effective_view_integrity,
)
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, post_process_metrics
from robot_sf.benchmark.observation_noise import (
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
from robot_sf.benchmark.safety_predicates import (
    late_evasive_predicate,
    occlusion_near_miss_predicate,
    oscillatory_control_predicate,
)
from robot_sf.benchmark.safety_wrapper_runtime import (
    apply_runtime_safety_wrapper,
    ineligible_safety_wrapper_step_record,
    make_deadlock_recovery_monitor,
    runtime_config_from_mapping,
    summarize_safety_wrapper_trace,
)
from robot_sf.benchmark.synthetic_actuation import (
    SyntheticActuationController,
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
from robot_sf.benchmark.utils import (
    _config_hash,
    _git_hash_fallback,
    attach_track_metadata,
    normalize_track_field,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.safety_shield import shield_metrics_from_stats

if TYPE_CHECKING:
    from pathlib import Path

PolicyBuilder = Callable[..., tuple[Any, dict[str, Any]]]


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
    spec: dict[str, Any],
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
    corrupted = apply_tracking_precision_spec(positions, spec, rng)
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
    step: dict[str, Any],
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
    planner_decision_trace: list[dict[str, Any]],
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
    planner_decision_trace: list[dict[str, Any]],
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
    spec: dict[str, Any],
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
        "hash": tracking_precision_hash(spec),
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
    noise_spec: dict[str, Any]
    noise_rng: Any
    noise_state: Any
    noise_stats: Any
    tracking_precision_spec: dict[str, Any]
    tracking_precision_rng: Any
    safety_wrapper_runtime: Any
    cbf_runtime: Any
    safety_wrapper_deadlock_monitor: Any
    config: Any
    horizon_val: int
    robot_kinematics: str
    robot_command_mode: str
    actuation_profile: Any
    latency_profile: Any
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
    noise_spec = normalize_observation_noise_spec(observation_noise)
    noise_rng = make_observation_noise_rng(noise_spec, seed=seed, scenario_id=scenario_id)
    noise_state = make_observation_noise_state(noise_spec)
    noise_stats = new_observation_noise_stats()
    tracking_precision_spec = normalize_tracking_precision_spec(tracking_precision)
    tracking_precision_rng = make_tracking_precision_rng(
        tracking_precision_spec,
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

    policy_fn: Any
    algo_meta: dict[str, Any]
    planner_close: Any
    planner_reset: Any
    planner_bind_env: Any
    planner_stats: Any
    planner_native_action: bool
    actuation_controller: Any
    active_observation_mode: str
    active_observation_level: str
    single_pedestrian_intent_metadata: Any
    single_pedestrian_vru_metadata: Any


def _prepare_policy_and_observation_contract(  # noqa: PLR0913
    *,
    scenario: dict[str, Any],
    algo: str,
    policy_cfg: dict[str, Any],
    config: Any,
    observation_mode: str | None,
    observation_level: str | None,
    robot_kinematics: str,
    robot_command_mode: str,
    adapter_impact_eval: bool,
    benchmark_track: str | None,
    track_schema_version: str | None,
    actuation_profile: Any,
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
    policy_fn, algo_meta = policy_builder(
        algo,
        policy_cfg,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
        adapter_impact_eval=adapter_impact_eval,
    )
    algo_meta = enrich_algorithm_metadata(
        algo=algo,
        metadata=algo_meta,
        robot_kinematics=robot_kinematics,
        observation_mode=active_observation_mode,
        observation_level=resolved_observation_level,
    )
    # Latency instrumentation resolves the planner configuration hash from the callable so
    # cached policies remain provenance-bound when a new harness is activated per episode.
    policy_fn._meta = algo_meta
    algo_meta["learned_checkpoint_observation_contract"] = learned_observation_contract
    active_observation_level = str(algo_meta["observation_level"]["key"])
    attach_track_metadata(
        algo_meta,
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
    planner_decision_trace: list[dict[str, Any]]
    simulation_step_trace: list[dict[str, Any]]
    view_integrity: dict[str, Any] | None
    planner_runtime_snapshot: dict[str, Any] | None


def _run_episode_step_loop(  # noqa: C901,PLR0912,PLR0913,PLR0915
    *,
    seed: int,
    config: Any,
    horizon_val: int,
    policy_fn: Any,
    planner_bind_env: Any,
    planner_reset: Any,
    planner_close: Any,
    planner_stats: Any,
    planner_native_action: bool,
    noise_spec: dict[str, Any],
    noise_rng: Any,
    noise_state: Any,
    noise_stats: Any,
    tracking_precision_spec: dict[str, Any],
    tracking_precision_rng: Any,
    safety_wrapper_runtime: Any,
    safety_wrapper_deadlock_monitor: Any,
    cbf_runtime: Any,
    actuation_controller: Any,
    algo_meta: dict[str, Any],
    record_forces: bool,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
    single_pedestrian_intent_metadata: Any,
    single_pedestrian_vru_metadata: Any,
) -> _EpisodeStepLoopResult:
    """Run the env reset, the per-step episode loop, and planner/env teardown.

    Returns:
        _EpisodeStepLoopResult: Immutable bundle of every trajectory, trace, and
        outcome flag produced by the step loop, plus the planner runtime snapshot
        captured in the ``finally`` teardown.
    """
    # Per-episode instrumentation buffers (populated during the step loop below).
    tracking_precision_records: list[dict[str, Any]] = []
    safety_wrapper_trace: list[dict[str, Any]] = []
    cbf_filter_trace: list[dict[str, Any]] = []
    min_separation_corrupted_values: list[float] = []
    planner_runtime_snapshot: dict[str, Any] | None = None
    current_command = (0.0, 0.0)
    synthetic_actuation_trace: list[dict[str, Any]] = []
    canonical_algorithm = (
        str(algo_meta.get("canonical_algorithm", algo_meta.get("algorithm", ""))).strip().lower()
    )
    hybrid_source_field = {
        "hybrid_portfolio": "selected_head",
        "hybrid_rule_local_planner": "selected_source",
    }.get(canonical_algorithm)
    hybrid_command_sources: list[str | None] | None = (
        [] if hybrid_source_field is not None else None
    )
    planner_decision_trace: list[dict[str, Any]] = []
    simulation_step_trace: list[dict[str, Any]] = []
    visibility_trace: list[np.ndarray | None] = []
    track_confidence_trace: list[np.ndarray | None] = []
    visibility_evidence_statuses: list[str] = []
    visibility_evidence_reasons: list[str | None] = []
    env = make_robot_env(config=config, seed=int(seed), debug=False)
    try:
        active_harness = LatencyMeasurementHarness.get_current()
        if active_harness is not None:
            policy_fn = active_harness.wrap_policy(policy_fn)

        obs, _ = env.reset(seed=int(seed))
        if callable(planner_bind_env):
            planner_bind_env(env)
        if callable(planner_reset):
            planner_reset(seed=int(seed))

        robot_positions: list[np.ndarray] = []
        ped_positions: list[np.ndarray] = []
        ped_forces: list[np.ndarray] = []
        reached_goal_step: int | None = None
        termination_reason = "max_steps"
        collision_seen = False
        ped_collision_seen = False
        obstacle_collision_seen = False
        robot_collision_seen = False
        timeout_seen = False
        collision_events: list[dict[str, Any]] = []

        map_def = getattr(env.simulator, "map_def", None)
        goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
        initial_robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
        initial_goal_distance = float(np.linalg.norm(initial_robot_pos - goal_vec))
        # Normalize optional radii before float(): getattr returns the default
        # only when the attribute is absent, so an explicit None in config would
        # otherwise raise TypeError in float(None).
        robot_radius_val = getattr(getattr(config, "robot_config", None), "radius", 1.0)
        robot_radius = float(robot_radius_val if robot_radius_val is not None else 1.0)
        ped_radius_val = getattr(config.sim_config, "ped_radius", 0.4)
        ped_radius = float(ped_radius_val if ped_radius_val is not None else 0.4)
        collision_event_context = _CollisionEventContext(
            dt_seconds=float(config.sim_config.time_per_step_in_secs),
            map_def=map_def,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
        )
        previous_trace_robot_pos = np.array(initial_robot_pos, dtype=float, copy=True)
        previous_trace_ped_pos: np.ndarray | None = None
        previous_trace_heading = _observation_heading(obs)
        previous_collision_robot_pos = np.array(initial_robot_pos, dtype=float, copy=True)
        previous_collision_ped_pos: np.ndarray | None = None
        robot_headings: list[float] = []
        ammv_command_actions: list[dict[str, Any]] = []
        view_integrity: dict[str, Any] | None = None
        for step_idx in range(horizon_val):
            if active_harness is not None:
                active_harness.start_cycle()
            policy_obs, step_noise_stats = apply_observation_noise(
                obs,
                noise_spec,
                noise_rng,
                noise_state,
            )
            merge_observation_noise_stats(noise_stats, step_noise_stats)
            policy_obs, corrupted_ped_positions = _apply_tracking_precision_to_observation(
                policy_obs,
                tracking_precision_spec,
                tracking_precision_rng,
            )
            robot_reference = np.asarray(env.simulator.robot_pos[0], dtype=float)
            if corrupted_ped_positions is not None:
                min_separation_corrupted_values.append(
                    minimum_separation(corrupted_ped_positions, robot_reference)
                )
            policy_command = policy_fn(policy_obs)
            if view_integrity is None:
                # Runtime fail-closed guard (#3634): probe the planner's effective observation view
                # once. The extractor signature is deterministic across steps, so a single probe
                # detects a silent-blind planner before any benchmark metrics are recorded. Fail
                # closed per docs/context/issue_691_benchmark_fallback_policy.md instead of emitting
                # results produced by a planner that drives blind to the pedestrians it was shown.
                integrity = evaluate_effective_view_integrity(
                    policy_fn=policy_fn,
                    observation=policy_obs,
                    algo_meta=algo_meta,
                )
                view_integrity = integrity.to_metadata()
                if integrity.degraded:
                    raise DegeneratePlannerViewError(integrity)
            actuation_step = None
            planner_step_decision = None
            # Hybrid handoff telemetry is part of the episode predicate contract, so
            # sample it even when the larger planner-decision trace is not requested.
            if (record_planner_decision_trace or hybrid_command_sources is not None) and callable(
                planner_stats
            ):
                try:
                    planner_runtime = planner_stats()
                except (RuntimeError, ValueError, TypeError):
                    planner_runtime = None
                if isinstance(planner_runtime, dict) and isinstance(
                    planner_runtime.get("last_decision"), dict
                ):
                    planner_step_decision = dict(planner_runtime["last_decision"])
            if hybrid_command_sources is not None:
                source = (
                    planner_step_decision.get(hybrid_source_field)
                    if planner_step_decision is not None and hybrid_source_field is not None
                    else None
                )
                normalized_source = str(source).strip() if source is not None else ""
                hybrid_command_sources.append(normalized_source or None)
            # Use per-step flag when available (e.g. SAC with fallback); fall back to the
            # static cached value for planners that set _planner_native_env_action once.
            step_is_native = getattr(policy_fn, "_last_step_native", planner_native_action)
            if actuation_controller is not None and step_is_native:
                raise ValueError(
                    "synthetic_actuation_profile requires absolute differential-drive commands; "
                    "native env actions cannot be wrapped safely"
                )
            if actuation_controller is not None:
                if (
                    not isinstance(policy_command, (tuple, list, np.ndarray))
                    or len(policy_command) < 2
                ):
                    raise TypeError(
                        "synthetic_actuation_profile expects planner commands shaped like "
                        "(linear_velocity, angular_velocity)"
                    )
                actuation_step = actuation_controller.apply(
                    current_command=current_command,
                    requested_command=(float(policy_command[0]), float(policy_command[1])),
                )
                policy_command = actuation_step.applied_command
                current_command = actuation_step.applied_command
            if (
                bool(tracking_precision_spec.get("enabled", False))
                and not step_is_native
                and isinstance(policy_command, (tuple, list, np.ndarray))
                and len(policy_command) >= 2
            ):
                applied_linear, tracking_record = apply_speed_contract(
                    float(policy_command[0]),
                    float(tracking_precision_spec["target_motp_m"]),
                    tracking_precision_spec,
                )
                policy_command = (
                    applied_linear,
                    float(policy_command[1]),
                    *tuple(policy_command[2:]),
                )
                tracking_precision_records.append(tracking_record)
            if safety_wrapper_runtime.enabled:
                policy_command, wrapper_record = _apply_safety_wrapper_step(
                    policy_command,
                    runtime=safety_wrapper_runtime,
                    env=env,
                    config=config,
                    step_idx=step_idx,
                    step_is_native=step_is_native,
                    previous_ped_positions=previous_trace_ped_pos,
                    deadlock_monitor=safety_wrapper_deadlock_monitor,
                )
                safety_wrapper_trace.append(wrapper_record)
            if cbf_runtime.enabled:
                policy_command, cbf_record = _apply_cbf_safety_filter_step(
                    policy_command,
                    runtime=cbf_runtime,
                    env=env,
                    config=config,
                    step_idx=step_idx,
                    step_is_native=step_is_native,
                    previous_ped_positions=previous_trace_ped_pos,
                )
                cbf_filter_trace.append(cbf_record)
            selected_action_payload = _command_action_payload(policy_command)
            ammv_command_actions.append(selected_action_payload)
            action_conversion_start = time.perf_counter() if active_harness is not None else None
            if step_is_native:
                # Policy already outputs native env actions (e.g. delta velocities);
                # skip the absolute→delta conversion done by _policy_command_to_env_action.
                action = np.asarray(policy_command, dtype=np.float32)
            else:
                action = _policy_command_to_env_action(
                    env=env,
                    config=config,
                    command=policy_command,
                )
            if active_harness is not None and action_conversion_start is not None:
                active_harness.add_time(
                    "action_conversion", (time.perf_counter() - action_conversion_start) * 1000.0
                )
            if active_harness is not None:
                active_harness.end_cycle()
            obs, reward, terminated, truncated, info = env.step(action)

            # Snapshot mutable simulator buffers; do not keep view aliases across steps.
            robot_pos = np.array(env.simulator.robot_pos[0], dtype=float, copy=True)
            peds = np.array(env.simulator.ped_pos, dtype=float, copy=True)
            if record_forces:
                forces = getattr(env.simulator, "last_ped_forces", None)
                if forces is None:
                    forces_arr = np.zeros_like(peds, dtype=float)
                else:
                    forces_arr = np.array(forces, dtype=float, copy=True)
                    if forces_arr.shape != peds.shape:
                        forces_arr = np.zeros_like(peds, dtype=float)

            robot_positions.append(robot_pos)
            ped_positions.append(peds)
            if record_forces:
                ped_forces.append(forces_arr)
            heading = _observation_heading(obs, default=previous_trace_heading)
            robot_headings.append(float(heading))
            (
                step_visible,
                step_confidence,
                step_visibility_status,
                step_visibility_reason,
            ) = _visibility_evidence_for_step(peds=peds, obs=obs, config=config)
            visibility_trace.append(step_visible)
            track_confidence_trace.append(step_confidence)
            visibility_evidence_statuses.append(step_visibility_status)
            visibility_evidence_reasons.append(step_visibility_reason)
            if record_simulation_step_trace:
                dt_seconds = float(config.sim_config.time_per_step_in_secs)
                robot_velocity = (
                    (robot_pos - previous_trace_robot_pos) / dt_seconds
                    if dt_seconds > 0.0
                    else np.zeros(2, dtype=float)
                )
                planner_payload: dict[str, Any] = {
                    "event": "step",
                    "selected_action": selected_action_payload,
                }
                if actuation_step is not None:
                    planner_payload["amv"] = {
                        "requested_linear_m_s": float(actuation_step.requested_command[0]),
                        "requested_angular_rad_s": float(actuation_step.requested_command[1]),
                        "applied_linear_m_s": float(actuation_step.applied_command[0]),
                        "applied_angular_rad_s": float(actuation_step.applied_command[1]),
                        "command_clipped": bool(actuation_step.command_clipped),
                        "yaw_rate_saturated": bool(actuation_step.yaw_rate_saturated),
                    }
                if record_forces and peds.size:
                    planner_payload["ammv"] = {
                        "pedestrian_force_vectors": [
                            [float(force[0]), float(force[1])] for force in forces_arr
                        ]
                    }
                trace_pedestrians = _annotate_trace_visibility(
                    _trace_pedestrians(
                        peds,
                        previous_trace_ped_pos,
                        dt_seconds,
                        single_pedestrian_intent_metadata,
                        single_pedestrian_vru_metadata,
                        robot_pos,
                        robot_velocity,
                    ),
                    visible=step_visible,
                    track_confidence=step_confidence,
                    evidence_status=step_visibility_status,
                    evidence_reason=step_visibility_reason,
                )
                simulation_step_trace.append(
                    {
                        "step": int(step_idx),
                        "time_s": float((step_idx + 1) * dt_seconds),
                        "robot": {
                            "position": [float(robot_pos[0]), float(robot_pos[1])],
                            "heading": float(heading),
                            "velocity": [float(robot_velocity[0]), float(robot_velocity[1])],
                        },
                        "pedestrians": trace_pedestrians,
                        "planner": planner_payload,
                        "rl": {
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        },
                    }
                )
                previous_trace_robot_pos = np.array(robot_pos, dtype=float, copy=True)
                previous_trace_ped_pos = np.array(peds, dtype=float, copy=True)
                previous_trace_heading = float(heading)
            if actuation_step is not None:
                distance_to_goal = float(np.linalg.norm(robot_pos - goal_vec))
                route_progress = float(initial_goal_distance - distance_to_goal)
                progress_ratio = (
                    route_progress / initial_goal_distance if initial_goal_distance > 1e-9 else 0.0
                )
                synthetic_actuation_trace.append(
                    {
                        "step": int(step_idx),
                        "requested_linear_m_s": float(actuation_step.requested_command[0]),
                        "requested_angular_rad_s": float(actuation_step.requested_command[1]),
                        "applied_linear_m_s": float(actuation_step.applied_command[0]),
                        "applied_angular_rad_s": float(actuation_step.applied_command[1]),
                        "command_clipped": bool(actuation_step.command_clipped),
                        "yaw_rate_saturated": bool(actuation_step.yaw_rate_saturated),
                        "linear_accel_applied_m_s2": float(
                            actuation_step.linear_accel_applied_m_s2
                        ),
                        "angular_accel_applied_rad_s2": float(
                            actuation_step.angular_accel_applied_rad_s2
                        ),
                        "distance_to_goal_m": distance_to_goal,
                        "route_progress_from_start_m": route_progress,
                        "route_progress_ratio": float(progress_ratio),
                        "robot_x_m": float(robot_pos[0]),
                        "robot_y_m": float(robot_pos[1]),
                    }
                )
            if record_planner_decision_trace and planner_step_decision is not None:
                selected_terms = planner_step_decision.get("selected_terms")
                selected_terms = selected_terms if isinstance(selected_terms, dict) else {}
                progress_windows_raw = planner_step_decision.get("progress_windows")
                progress_windows = (
                    progress_windows_raw if isinstance(progress_windows_raw, dict) else {}
                )
                selected_command = planner_step_decision.get("selected_command")
                selected_command = selected_command if isinstance(selected_command, list) else []
                distance_to_goal = float(np.linalg.norm(robot_pos - goal_vec))
                step_decision = {
                    "step": int(step_idx),
                    "selected_source": str(planner_step_decision.get("selected_source", "unknown")),
                    "selected_command": [
                        float(value)
                        for value in selected_command[:2]
                        if isinstance(value, int | float | np.integer | np.floating)
                    ],
                    "selected_score": float(planner_step_decision["selected_score"])
                    if isinstance(
                        planner_step_decision.get("selected_score"),
                        int | float | np.integer | np.floating,
                    )
                    and math.isfinite(float(planner_step_decision["selected_score"]))
                    else None,
                    "static_recenter": float(selected_terms.get("static_recenter", 0.0)),
                    "route_arc_progress": float(selected_terms.get("route_arc_progress", 0.0)),
                    "goal_progress": float(selected_terms.get("goal_progress", 0.0)),
                    "progress_windows": {
                        str(key): float(value)
                        for key, value in progress_windows.items()
                        if isinstance(value, int | float | np.integer | np.floating)
                    },
                    "distance_to_goal_m": distance_to_goal,
                    "route_progress_from_start_m": float(initial_goal_distance - distance_to_goal),
                    "robot_x_m": float(robot_pos[0]),
                    "robot_y_m": float(robot_pos[1]),
                }
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
                    if "topology_guided_config" not in step_decision and isinstance(
                        fallback_config, dict
                    ):
                        step_decision["topology_guided_config"] = deepcopy(fallback_config)
                # Additive, planner-agnostic pass-through for adapter diagnostics that do
                # not map onto the topology-guided fields above (issue #5298 DWA trace).
                # Only present when the underlying adapter populates them, so other
                # planners' traces are unchanged.
                for _dwa_key in (
                    "constraint_reason",
                    "candidate_total",
                    "candidate_feasible",
                    "candidate_infeasible",
                    "feasible_score_min",
                    "feasible_score_max",
                    "dynamic_window",
                    "target_goal",
                ):
                    _value = planner_step_decision.get(_dwa_key)
                    if _value is not None:
                        step_decision[_dwa_key] = deepcopy(_value)
                planner_decision_trace.append(step_decision)

            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            step_collision = collision_event(info)
            step_route_complete = route_complete_success(info)
            step_success = step_route_complete and not step_collision
            step_timeout = bool(meta.get("is_timesteps_exceeded", False))
            collision_seen = collision_seen or step_collision
            ped_collision_seen = ped_collision_seen or bool(
                meta.get("is_pedestrian_collision", False)
            )
            obstacle_collision_seen = obstacle_collision_seen or bool(
                meta.get("is_obstacle_collision", False)
            )
            robot_collision_seen = robot_collision_seen or bool(
                meta.get("is_robot_collision", False)
            )
            timeout_seen = timeout_seen or step_timeout
            collision_events.extend(
                _step_collision_events(
                    step_idx=step_idx,
                    robot_pos=robot_pos,
                    previous_robot_pos=previous_collision_robot_pos,
                    ped_positions=peds,
                    previous_ped_positions=previous_collision_ped_pos,
                    meta=meta,
                    context=collision_event_context,
                )
            )
            previous_collision_robot_pos = np.array(robot_pos, dtype=float, copy=True)
            previous_collision_ped_pos = np.array(peds, dtype=float, copy=True)
            if reached_goal_step is None and step_success:
                reached_goal_step = step_idx
            if step_success:
                termination_reason = resolve_termination_reason(
                    terminated=True,
                    truncated=False,
                    success=True,
                    collision=step_collision,
                )
                break
            if terminated or truncated:
                termination_reason = resolve_termination_reason(
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=step_success,
                    collision=step_collision,
                )
                break
        if getattr(env, "simulator", None) is not None:
            map_def = env.simulator.map_def
            goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
    finally:
        if callable(planner_stats):
            try:
                planner_runtime = planner_stats()
            except (RuntimeError, ValueError, TypeError):
                logger.debug("Planner stats hook failed before close", exc_info=True)
                planner_runtime = None
            if isinstance(planner_runtime, dict):
                planner_runtime_snapshot = dict(planner_runtime)
        if callable(planner_close):
            try:
                planner_close()
            except (RuntimeError, ValueError, TypeError):
                logger.debug("Planner close hook failed", exc_info=True)
        env.close()

    return _EpisodeStepLoopResult(
        map_def=map_def,
        goal_vec=goal_vec,
        initial_robot_pos=initial_robot_pos,
        initial_goal_distance=initial_goal_distance,
        reached_goal_step=reached_goal_step,
        termination_reason=termination_reason,
        collision_seen=collision_seen,
        ped_collision_seen=ped_collision_seen,
        obstacle_collision_seen=obstacle_collision_seen,
        robot_collision_seen=robot_collision_seen,
        timeout_seen=timeout_seen,
        collision_events=collision_events,
        robot_positions=robot_positions,
        robot_headings=robot_headings,
        ped_positions=ped_positions,
        ped_forces=ped_forces,
        visibility_trace=visibility_trace,
        track_confidence_trace=track_confidence_trace,
        visibility_evidence_statuses=visibility_evidence_statuses,
        visibility_evidence_reasons=visibility_evidence_reasons,
        tracking_precision_records=tracking_precision_records,
        min_separation_corrupted_values=min_separation_corrupted_values,
        safety_wrapper_trace=safety_wrapper_trace,
        cbf_filter_trace=cbf_filter_trace,
        ammv_command_actions=ammv_command_actions,
        synthetic_actuation_trace=synthetic_actuation_trace,
        hybrid_command_sources=hybrid_command_sources,
        planner_decision_trace=planner_decision_trace,
        simulation_step_trace=simulation_step_trace,
        view_integrity=view_integrity,
        planner_runtime_snapshot=planner_runtime_snapshot,
    )


def _finalize_episode_record(  # noqa: C901,PLR0912,PLR0913,PLR0915
    *,
    ctx: _EpisodeRunContext,
    loop_result: _EpisodeStepLoopResult,
    post_loop: _EpisodePostLoopResult,
    algo_meta: dict[str, Any],
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
    """Assemble the benchmark JSONL record from the step-loop and post-loop results.

    Returns:
        dict[str, Any]: The finalized episode record with metrics, provenance, and
        planner metadata, mirroring the prior inline metadata-finalization phase.
    """
    scenario = ctx.scenario
    scenario_id = ctx.scenario_id
    ts_start = ctx.ts_start
    start_time = ctx.start_time
    benchmark_track = ctx.benchmark_track
    track_schema_version = ctx.track_schema_version
    noise_spec = ctx.noise_spec
    noise_stats = ctx.noise_stats
    tracking_precision_spec = ctx.tracking_precision_spec
    safety_wrapper_runtime = ctx.safety_wrapper_runtime
    cbf_runtime = ctx.cbf_runtime
    config = ctx.config
    horizon_val = ctx.horizon_val
    robot_kinematics = ctx.robot_kinematics
    actuation_profile = ctx.actuation_profile
    latency_profile = ctx.latency_profile
    algo = ctx.algo
    policy_cfg = ctx.policy_cfg
    ammv_command_actions = loop_result.ammv_command_actions
    planner_runtime_snapshot = loop_result.planner_runtime_snapshot
    planner_decision_trace = loop_result.planner_decision_trace
    simulation_step_trace = loop_result.simulation_step_trace
    tracking_precision_records = loop_result.tracking_precision_records
    min_separation_corrupted_values = loop_result.min_separation_corrupted_values
    safety_wrapper_trace = loop_result.safety_wrapper_trace
    cbf_filter_trace = loop_result.cbf_filter_trace
    synthetic_actuation_trace = loop_result.synthetic_actuation_trace
    initial_goal_distance = loop_result.initial_goal_distance
    reached_goal_step = loop_result.reached_goal_step
    termination_reason = loop_result.termination_reason
    collision_seen = loop_result.collision_seen
    timeout_seen = loop_result.timeout_seen
    goal_vec = loop_result.goal_vec
    view_integrity = loop_result.view_integrity
    collision_events = loop_result.collision_events
    actuation_summary: dict[str, Any] = not_available_saturation_metrics()
    robot_pos_arr = post_loop.robot_pos_arr
    robot_vel_arr = post_loop.robot_vel_arr
    ped_pos_arr = post_loop.ped_pos_arr
    ped_forces_arr = post_loop.ped_forces_arr
    safety_predicates = post_loop.safety_predicates
    metrics_raw = post_loop.metrics_raw
    if robot_pos_arr.size:
        robot_config = getattr(config, "robot_config", None)
    # Finalization phase: isolate the episode metadata from the builder-provided
    # ``algo_meta`` so the finalization writes below (adapter_impact status,
    # tracking_precision, safety_wrapper, planner_runtime, etc.) cannot leak back
    # into a builder that reuses/caches the same dict across episodes (#4954).
    # The episode loop has finished, so every runtime write the policy made into
    # ``algo_meta`` (e.g. adapter_impact counters, shield_stats) is already present
    # and is captured by this deep copy. ``enrich_algorithm_metadata`` only shallow-
    # copies, so nested mutable structures (notably ``adapter_impact``) would
    # otherwise still be shared with the builder and mutated in place here.
    algo_meta = deepcopy(algo_meta)
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
            algo_meta = enrich_algorithm_metadata(
                algo=algo,
                metadata=algo_meta,
                execution_mode=execution_mode,
                robot_kinematics=robot_kinematics,
                observation_mode=active_observation_mode,
                observation_level=active_observation_level,
            )
            attach_track_metadata(
                algo_meta,
                benchmark_track=benchmark_track,
                track_schema_version=track_schema_version,
                observation_level=active_observation_level,
                observation_mode=active_observation_mode,
            )
        else:
            impact["status"] = "not_applicable"
            impact["adapter_fraction"] = 0.0
    _finalize_feasibility_metadata(algo_meta)
    algo_meta["ammv_feasibility"] = evaluate_artifact_command_feasibility(ammv_command_actions)
    if isinstance(planner_runtime_snapshot, dict):
        algo_meta["planner_runtime"] = planner_runtime_snapshot
    if record_planner_decision_trace:
        algo_meta["planner_decision_trace"] = {
            "schema_version": "planner-decision-trace.v1",
            "dt": float(config.sim_config.time_per_step_in_secs),
            "initial_goal_distance_m": initial_goal_distance,
            "steps": planner_decision_trace,
        }
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
            algo_meta,
            scenario=scenario,
            ped_positions=ped_pos_arr,
            ped_forces=ped_forces_arr if record_forces else None,
            dt=float(config.sim_config.time_per_step_in_secs),
            robot_positions=robot_pos_arr,
            robot_radius=float(getattr(robot_config, "radius", 1.0)),
            ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),
        )
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
    intent_summary = _intent_conditioned_behavior_summary(
        scenario,
        single_pedestrian_intent_metadata,
    )
    if intent_summary is not None:
        algo_meta["intent_conditioned_behavior"] = intent_summary
    vru_summary = _cyclist_like_vru_summary(
        scenario,
        single_pedestrian_vru_metadata,
    )
    if vru_summary is not None:
        algo_meta["cyclist_like_vru"] = vru_summary
    fast_bicycle_summary = _fast_bicycle_actor_summary(
        scenario,
        single_pedestrian_vru_metadata,
    )
    if fast_bicycle_summary is not None:
        algo_meta["fast_bicycle_actor"] = fast_bicycle_summary
    if actuation_controller is not None:
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

    ts_end = datetime.now(UTC).isoformat()
    scenario_params = _scenario_identity_payload(
        scenario,
        algo=algo,
        algo_config=policy_cfg,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        observation_mode=active_observation_mode,
        observation_level=active_observation_level,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        observation_noise=noise_spec,
        tracking_precision=tracking_precision_spec,
        synthetic_actuation_profile=(
            actuation_profile.to_metadata() if actuation_profile is not None else None
        ),
        latency_stress_profile=(
            latency_profile.to_metadata(dt=config.sim_config.time_per_step_in_secs)
            if latency_profile is not None
            else None
        ),
        safety_wrapper=dict(safety_wrapper) if safety_wrapper is not None else None,
        cbf_safety_filter=dict(cbf_safety_filter) if cbf_safety_filter is not None else None,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
    )
    steps_taken = int(robot_pos_arr.shape[0])
    wall_time = float(max(1e-9, time.time() - start_time))
    timing = {"steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0}
    route_complete = reached_goal_step is not None
    timeout_event = timeout_seen or termination_reason in {"truncated", "max_steps"}
    outcome = build_outcome_payload(
        route_complete=route_complete,
        collision=collision_seen,
        timeout=timeout_event,
    )
    status = status_from_termination_reason(termination_reason)
    contradictions = outcome_contradictions(
        termination_reason=termination_reason,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        raise ValueError(
            f"Episode integrity contradictions for scenario '{scenario_id}', seed={seed}: "
            + "; ".join(contradictions)
        )
    static_deadlock_fields = _static_deadlock_trace_fields(
        scenario,
        robot_pos_arr=robot_pos_arr,
        goal_vec=goal_vec,
        initial_goal_distance=initial_goal_distance,
        termination_reason=termination_reason,
        outcome=outcome,
        planner_decision_trace=planner_decision_trace,
    )
    record = {
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
        "observation_noise_hash": observation_noise_hash(noise_spec),
        "observation_noise_stats": noise_stats,
        "tracking_precision": tracking_precision_spec,
        "tracking_precision_hash": tracking_precision_hash(tracking_precision_spec),
        "algo": algo,
        "observation_mode": active_observation_mode,
        "observation_level": active_observation_level,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {"start": ts_start, "end": ts_end},
        "status": status,
        "steps": steps_taken,
        "horizon": horizon_val,
        "wall_time_sec": wall_time,
        "timing": timing,
        "termination_reason": termination_reason,
        "outcome": outcome,
        "integrity": {
            "contradictions": contradictions,
            "effective_view": view_integrity,
        },
    }
    pedestrian_model_provenance = build_pedestrian_model_provenance(
        sim_config=config.sim_config,
        policy_cfg=policy_cfg,
        algorithm_metadata=algo_meta,
    )
    attach_pedestrian_model_fields(record, pedestrian_model_provenance)
    record.update(static_deadlock_fields)
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
    if benchmark_track is not None:
        record["benchmark_track"] = benchmark_track
    if track_schema_version is not None:
        record["track_schema_version"] = track_schema_version

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
            noise_hash=observation_noise_hash(noise_spec),
            tracking_precision_hash=tracking_precision_hash(tracking_precision_spec),
        ),
        "postprocessing": [
            {"step": "compute_all_metrics", "status": "completed"},
            {"step": "post_process_metrics", "status": "completed"},
        ],
    }
    ensure_metric_parameters(record)
    return record


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
    close_policy: bool = True,
    policy_builder: PolicyBuilder,
) -> dict[str, Any]:
    """Run one scenario/seed episode and return a benchmark JSONL record.

    Returns:
        dict[str, Any]: Episode record with metrics, provenance, and planner metadata.
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
        config=config,
        horizon_val=horizon_val,
        policy_fn=policy_contract.policy_fn,
        planner_bind_env=policy_contract.planner_bind_env,
        planner_reset=policy_contract.planner_reset,
        planner_close=policy_contract.planner_close if close_policy else None,
        planner_stats=policy_contract.planner_stats,
        planner_native_action=policy_contract.planner_native_action,
        noise_spec=noise_spec,
        noise_rng=noise_rng,
        noise_state=noise_state,
        noise_stats=noise_stats,
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
