"""Episode execution helpers for map-based benchmark batches."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
    resolve_learned_checkpoint_observation_contract,
)
from robot_sf.benchmark.latency_stress import not_available_latency_metrics
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
from robot_sf.benchmark.map_runner_policy_metadata import (
    finalize_feasibility_metadata as _finalize_feasibility_metadata,
)
from robot_sf.benchmark.map_runner_policy_resolution import (
    _apply_planner_selector_v2_context,
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
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, post_process_metrics
from robot_sf.benchmark.observation_noise import (
    apply_observation_noise,
    make_observation_noise_rng,
    merge_observation_noise_stats,
    new_observation_noise_stats,
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.pedestrian_control_trace import (
    attach_pedestrian_control_trace,
)
from robot_sf.benchmark.planner_command_contract import (
    validate_planner_contract as _validate_planner_contract,
)
from robot_sf.benchmark.safety_predicates import (
    late_evasive_predicate,
    occlusion_near_miss_predicate,
    oscillatory_control_predicate,
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


def _safety_predicates_for_episode(
    *,
    robot_pos_arr: np.ndarray,
    robot_vel_arr: np.ndarray,
    robot_headings: list[float],
    ped_pos_arr: np.ndarray,
    dt: float,
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

    # Map-runner currently has simulator full-state traces, not per-step occlusion labels.
    # Treat actors as visible with full track confidence and let the predicate emit false
    # unless future visibility evidence shows an occluded actor emerging near a miss.
    hazard_visible = np.ones(step_count, dtype=bool)
    track_confidence = np.ones(step_count, dtype=float)

    return {
        "oscillatory_control_predicate": oscillatory_control_predicate(
            positions,
            headings,
            speeds,
            dt=dt,
        ),
        "late_evasive_predicate": late_evasive_predicate(
            hazard_distances,
            hazard_visible,
            speeds,
            dt=dt,
        ),
        "occlusion_near_miss_predicate": occlusion_near_miss_predicate(
            hazard_distances,
            hazard_visible,
            track_confidence,
            speeds,
            dt=dt,
        ),
    }


def run_map_episode(  # noqa: C901,PLR0912,PLR0913,PLR0915
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
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    record_planner_decision_trace: bool = False,
    record_simulation_step_trace: bool = False,
    policy_builder: PolicyBuilder,
) -> dict[str, Any]:
    """Run one scenario/seed episode and return a benchmark JSONL record.

    Returns:
        dict[str, Any]: Episode record with metrics, provenance, and planner metadata.
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
    noise_stats = new_observation_noise_stats()
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

    planner_runtime_snapshot: dict[str, Any] | None = None
    actuation_controller = (
        SyntheticActuationController(
            profile=actuation_profile, dt=config.sim_config.time_per_step_in_secs
        )
        if actuation_profile is not None
        else None
    )
    current_command = (0.0, 0.0)
    actuation_summary: dict[str, Any] = not_available_saturation_metrics()
    synthetic_actuation_trace: list[dict[str, Any]] = []
    planner_decision_trace: list[dict[str, Any]] = []
    simulation_step_trace: list[dict[str, Any]] = []
    single_pedestrian_intent_metadata = _single_pedestrian_intent_metadata(scenario)
    single_pedestrian_vru_metadata = _single_pedestrian_vru_metadata(scenario)

    env = make_robot_env(config=config, seed=int(seed), debug=False)
    try:
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

        map_def = None
        goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
        initial_robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
        initial_goal_distance = float(np.linalg.norm(initial_robot_pos - goal_vec))
        previous_trace_robot_pos = np.array(initial_robot_pos, dtype=float, copy=True)
        previous_trace_ped_pos: np.ndarray | None = None
        previous_trace_heading = _observation_heading(obs)
        robot_headings: list[float] = []
        for step_idx in range(horizon_val):
            policy_obs, step_noise_stats = apply_observation_noise(obs, noise_spec, noise_rng)
            merge_observation_noise_stats(noise_stats, step_noise_stats)
            policy_command = policy_fn(policy_obs)
            actuation_step = None
            planner_step_decision = None
            if record_planner_decision_trace and callable(planner_stats):
                try:
                    planner_runtime = planner_stats()
                except (RuntimeError, ValueError, TypeError):
                    planner_runtime = None
                if isinstance(planner_runtime, dict) and isinstance(
                    planner_runtime.get("last_decision"), dict
                ):
                    planner_step_decision = dict(planner_runtime["last_decision"])
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
            obs, _reward, terminated, truncated, info = env.step(action)

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
            if record_simulation_step_trace:
                dt_seconds = float(config.sim_config.time_per_step_in_secs)
                robot_velocity = (
                    (robot_pos - previous_trace_robot_pos) / dt_seconds
                    if dt_seconds > 0.0
                    else np.zeros(2, dtype=float)
                )
                planner_payload: dict[str, Any] = {
                    "event": "step",
                    "selected_action": _command_action_payload(policy_command),
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
                simulation_step_trace.append(
                    {
                        "step": int(step_idx),
                        "time_s": float((step_idx + 1) * dt_seconds),
                        "robot": {
                            "position": [float(robot_pos[0]), float(robot_pos[1])],
                            "heading": float(heading),
                            "velocity": [float(robot_velocity[0]), float(robot_velocity[1])],
                        },
                        "pedestrians": _trace_pedestrians(
                            peds,
                            previous_trace_ped_pos,
                            dt_seconds,
                            single_pedestrian_intent_metadata,
                            single_pedestrian_vru_metadata,
                            robot_pos,
                            robot_velocity,
                        ),
                        "planner": planner_payload,
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
            if planner_step_decision is not None:
                selected_terms = planner_step_decision.get("selected_terms")
                selected_terms = selected_terms if isinstance(selected_terms, dict) else {}
                progress_windows_raw = planner_step_decision.get("progress_windows")
                progress_windows = (
                    progress_windows_raw if isinstance(progress_windows_raw, dict) else {}
                )
                selected_command = planner_step_decision.get("selected_command")
                selected_command = selected_command if isinstance(selected_command, list) else []
                distance_to_goal = float(np.linalg.norm(robot_pos - goal_vec))
                planner_decision_trace.append(
                    {
                        "step": int(step_idx),
                        "selected_source": str(
                            planner_step_decision.get("selected_source", "unknown")
                        ),
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
                        "route_progress_from_start_m": float(
                            initial_goal_distance - distance_to_goal
                        ),
                        "robot_x_m": float(robot_pos[0]),
                        "robot_y_m": float(robot_pos[1]),
                    }
                )

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
    safety_predicates = _safety_predicates_for_episode(
        robot_pos_arr=robot_pos_arr,
        robot_vel_arr=robot_vel_arr,
        robot_headings=robot_headings,
        ped_pos_arr=ped_pos_arr,
        dt=float(config.sim_config.time_per_step_in_secs),
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
            episode_metadata=_episode_metadata_for_signal_metrics(scenario),
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
    if isinstance(planner_runtime_snapshot, dict):
        algo_meta["planner_runtime"] = planner_runtime_snapshot
    if record_planner_decision_trace:
        algo_meta["planner_decision_trace"] = {
            "schema_version": "planner-decision-trace.v1",
            "dt": float(config.sim_config.time_per_step_in_secs),
            "initial_goal_distance_m": initial_goal_distance,
            "steps": planner_decision_trace,
        }
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
        )
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
        synthetic_actuation_profile=(
            actuation_profile.to_metadata() if actuation_profile is not None else None
        ),
        latency_stress_profile=(
            latency_profile.to_metadata(dt=config.sim_config.time_per_step_in_secs)
            if latency_profile is not None
            else None
        ),
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
        "algorithm_metadata": algo_meta,
        "observation_noise": noise_spec,
        "observation_noise_hash": observation_noise_hash(noise_spec),
        "observation_noise_stats": noise_stats,
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
        "integrity": {"contradictions": contradictions},
    }
    record.update(static_deadlock_fields)
    if benchmark_track is not None:
        record["benchmark_track"] = benchmark_track
    if track_schema_version is not None:
        record["track_schema_version"] = track_schema_version
    ensure_metric_parameters(record)
    return record


__all__ = ["run_map_episode"]
