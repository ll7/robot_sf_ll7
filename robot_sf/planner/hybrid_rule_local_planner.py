"""Deterministic hybrid-rule local planner family.

This module starts the ``hybrid_rule_local_planner`` family with the v0 minimal
variant: a dynamic-window style local planner with explicit safety filtering and
diagnostic score decomposition.  It intentionally avoids training, learned
weights, and benchmark-result fitting.
"""

from __future__ import annotations

import copy
from collections import Counter, deque
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from robot_sf.common.math_utils import wrap_angle_pi as _wrap_angle
from robot_sf.nav.occupancy import circle_collides_any_lines
from robot_sf.nav.proxemic_costmap import (
    ProxemicCostmapConfig,
    build_proxemic_costmap_config,
    proxemic_cost_at_points,
)
from robot_sf.nav.proxemic_costmap import (
    config_hash as proxemic_config_hash,
)
from robot_sf.planner.grid_route import GridRoutePlannerAdapter, GridRoutePlannerConfig
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin

_SUPPORTED_VARIANTS = {
    "hybrid_rule_v0_minimal",
    "hybrid_rule_v1_dwa_social",
    "hybrid_rule_v2_orca_guided",
    "hybrid_rule_v3_teb_like_rollout",
    "hybrid_rule_v4_recovery_aware",
    "hybrid_rule_v5_ensemble_selector",
    "tentabot_value_scorer_v0",
    "tentabot_value_scorer_v1_static_gated",
    "tentabot_value_scorer_v2_route_arc",
    "tentabot_value_scorer_v3_trace_recovery",
    "actuation_aware_hybrid_rule_v0",
}
_EPS = 1e-9


def _clip01(value: float) -> float:
    """Clamp a scalar to the normalized ``[0, 1]`` range.

    Returns:
        float: Clamped value.
    """
    return float(np.clip(float(value), 0.0, 1.0))


def _finite_or_none(value: Any) -> float | None:
    """Return finite float values for JSON-safe diagnostics."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _ego_velocity_to_world(velocity: np.ndarray, heading: float) -> np.ndarray:
    """Convert robot-ego-frame planar velocities to world-frame velocities.

    SocNav structured observations report pedestrian velocities in the robot ego
    frame while positions are world-frame. Dynamic collision prediction requires
    both quantities in the same frame.

    Returns:
        np.ndarray: World-frame velocity array with shape ``(N, 2)``.
    """
    if velocity.size == 0:
        return velocity
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    world = np.empty_like(velocity, dtype=float)
    world[:, 0] = cos_h * velocity[:, 0] - sin_h * velocity[:, 1]
    world[:, 1] = sin_h * velocity[:, 0] + cos_h * velocity[:, 1]
    return world


@dataclass(frozen=True)
class HybridRuleCandidate:
    """Candidate unicycle command with an optional planned rollout sequence."""

    linear: float
    angular: float
    source: str
    rollout_sequence: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class _ObstacleClearanceContext:
    """Per-plan static-obstacle clearance lookup."""

    observation_id: int
    grid: np.ndarray
    meta: dict[str, Any]
    channel: int
    resolution: float
    clearance_cells: np.ndarray | None


@dataclass(frozen=True)
class _ContinuousStaticContext:
    """Environment-backed continuous static-obstacle collision surface."""

    width: float
    height: float
    obstacle_segments: np.ndarray


@dataclass
class HybridRuleLocalPlannerConfig:
    """Manual constants for ``hybrid_rule_v0_minimal``.

    Units:
    - speeds are metres/second and radians/second,
    - accelerations are metres/second^2 and radians/second^2,
    - distances are metres,
    - horizons and rollout step sizes are seconds.
    """

    planner_variant: str = "hybrid_rule_v0_minimal"
    max_linear_speed: float = 2.0
    max_angular_speed: float = 1.2
    max_linear_accel: float = 2.0
    max_linear_decel: float = 2.5
    max_angular_accel: float = 4.0
    control_period: float = 0.6
    rollout_dt: float = 0.2
    rollout_horizon: float = 1.6
    hard_collision_horizon: float = 1.2
    soft_prediction_horizon: float = 3.0
    goal_tolerance: float = 0.25
    waypoint_switch_distance: float = 0.9

    linear_samples: int = 7
    angular_samples: int = 9
    lookahead_distances: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

    robot_radius_default: float = 0.3
    pedestrian_radius_default: float = 0.3
    hard_safety_margin: float = 0.05
    static_hard_safety_margin: float = -1.0
    desired_static_clearance: float = 0.7
    desired_dynamic_clearance: float = 0.9
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12
    proxemic_costmap_enabled: bool = False
    proxemic_costmap_personal_radius: float = 0.45
    proxemic_costmap_social_radius: float = 1.2
    proxemic_costmap_personal_weight: float = 1.0
    proxemic_costmap_social_weight: float = 0.35
    proxemic_costmap_velocity_elongation_factor: float = 0.0
    proxemic_costmap_max_cost: float = 10.0
    proxemic_costmap_decay_function: str = "linear"
    proxemic_costmap_weight: float = 0.0

    stop_distance_human: float = 0.5
    slow_distance_human: float = 1.0
    moderate_distance_human: float = 2.0
    very_slow_speed: float = 0.15
    moderate_speed: float = 0.6
    near_human_angular_limit_distance: float = 0.8
    near_human_max_angular_speed: float = 0.7

    goal_progress_weight: float = 4.0
    path_alignment_weight: float = 0.8
    speed_preference_weight: float = 0.6
    static_clearance_weight: float = 1.5
    dynamic_clearance_weight: float = 1.8
    ttc_weight: float = 0.8
    heading_smoothness_weight: float = 0.4
    velocity_smoothness_weight: float = 0.3
    control_effort_weight: float = 0.2
    deadlock_escape_weight: float = 0.0
    route_arc_progress_weight: float = 0.0
    route_guide_commitment_weight: float = 0.0
    corridor_subgoal_route_progress_weight: float = 0.0
    corridor_subgoal_centering_weight: float = 0.0
    corridor_subgoal_tangent_alignment_weight: float = 0.0
    corridor_subgoal_clearance_weight: float = 0.0
    corridor_subgoal_continuity_weight: float = 0.0
    freezing_weight: float = 0.8
    oscillation_weight: float = 0.5

    freezing_speed_threshold: float = 0.05
    goal_far_distance: float = 0.8
    oscillation_window: int = 6
    top_k_diagnostics: int = 5
    deadlock_progress_threshold: float = 0.05
    deadlock_rotation_threshold: float = 0.15
    route_guide_enabled: bool = False
    route_guide_obstacle_inflation_cells: int = 3
    route_guide_waypoint_lookahead_cells: int = 5
    route_guide_clearance_penalty_weight: float = 0.5
    corridor_subgoal_enabled: bool = False
    corridor_subgoal_speed: float = 0.25
    corridor_subgoal_turn_in_place_error: float = 0.25
    corridor_subgoal_heading_gain: float = 1.0
    corridor_subgoal_min_nearest_ped_distance: float = 1.0
    corridor_subgoal_use_continuous_static_check: bool = True
    continuous_static_clearance_enabled: bool = False
    corridor_subgoal_static_clearance_buffer: float = 0.2
    corridor_subgoal_goal_stall_progress_3s: float = 0.05
    corridor_subgoal_route_stall_progress_3s: float = 0.05
    corridor_subgoal_route_regression_1s: float = -0.05
    corridor_subgoal_min_route_remaining_distance: float = 0.5
    corridor_subgoal_max_lateral_offset: float = 0.5
    route_trace_recovery_enabled: bool = False
    route_trace_recovery_goal_stall_progress_3s: float = 0.05
    route_trace_recovery_route_stall_progress_3s: float = 0.05
    route_trace_recovery_route_regression_1s: float = -0.05
    route_trace_recovery_min_nearest_ped_distance: float = 1.0
    route_trace_recovery_min_route_remaining_distance: float = 0.5
    route_trace_recovery_hold_steps: int = 2
    recovery_enabled: bool = False
    recovery_reorient_angular_speed: float = 0.6

    # Ablation-only static-clearance escape knobs kept so the rejected
    # 2026-04-30 policy-search candidates can be reproduced. They default off
    # and are not part of the selected waypoint2 policy.
    static_clearance_escape_enabled: bool = False
    static_clearance_escape_tolerance: float = 0.05
    static_clearance_escape_max_speed: float = 0.6
    static_clearance_escape_min_clearance: float = 0.5
    static_recenter_enabled: bool = False
    static_recenter_weight: float = 0.0
    static_recenter_probe_speed: float = 0.3
    static_corridor_transit_enabled: bool = False
    static_corridor_transit_initial_band: float = 0.05
    static_corridor_transit_tolerance: float = 0.05
    static_corridor_transit_min_progress_3s: float = 0.0
    static_corridor_transit_weight: float = 0.0

    # Ablation-only route-commitment bonus. It remains configurable for
    # diagnostics, but benchmark evidence rejected it due static collisions.
    route_guide_commitment_progress_threshold: float = 0.5

    # Experimental clean-room value-scorer metadata for issue #1387. The v0
    # scorer uses the existing Robot SF candidate lattice and hand-auditable
    # linear score terms; it does not import upstream Tentabot code, data, or
    # weights.
    value_scorer_profile: str = "manual_linear"
    value_scorer_training_source: str = "hand_scored_hybrid_rule_teacher"
    static_safety_gate_enabled: bool = False
    static_safety_gate_min_clearance: float = 0.55
    static_safety_gate_progress_threshold: float = 0.05
    static_safety_gate_clearance_tolerance: float = 0.02
    static_safety_gate_penalty: float = 12.0
    static_safety_gate_all_sources: bool = False

    # Experimental AMV synthetic-actuation scoring for issue #1807. These
    # values intentionally mirror the diagnostic profile shape used by issue
    # #1556; they are not calibrated hardware limits.
    actuation_score_enabled: bool = False
    actuation_profile_name: str = "amv-actuation-stress-v0"
    actuation_claim_scope: str = "synthetic-diagnostic-only"
    actuation_max_linear_accel: float = 2.0
    actuation_max_linear_decel: float = 2.5
    actuation_max_yaw_rate: float = 1.2
    actuation_max_angular_accel: float = 4.0
    actuation_clip_risk_weight: float = 0.0


class HybridRuleLocalPlannerAdapter(OccupancyAwarePlannerMixin):
    """Minimal deterministic hybrid-rule local planner."""

    def __init__(self, config: HybridRuleLocalPlannerConfig | None = None) -> None:
        """Initialize the planner with manual constants."""
        self.config = config or HybridRuleLocalPlannerConfig()
        if self.config.planner_variant not in _SUPPORTED_VARIANTS:
            supported = ", ".join(sorted(_SUPPORTED_VARIANTS))
            raise ValueError(
                f"Unsupported hybrid rule planner variant "
                f"'{self.config.planner_variant}'. Supported variants: {supported}."
            )
        self._route_guide = (
            GridRoutePlannerAdapter(
                GridRoutePlannerConfig(
                    max_linear_speed=float(self.config.max_linear_speed),
                    max_angular_speed=float(self.config.max_angular_speed),
                    goal_tolerance=float(self.config.goal_tolerance),
                    waypoint_lookahead_cells=int(self.config.route_guide_waypoint_lookahead_cells),
                    obstacle_inflation_cells=int(self.config.route_guide_obstacle_inflation_cells),
                    clearance_penalty_weight=float(
                        self.config.route_guide_clearance_penalty_weight
                    ),
                )
            )
            if bool(self.config.route_guide_enabled)
            else None
        )
        self._continuous_static_context: _ContinuousStaticContext | None = None
        self.reset()

    @property
    def _proxemic_costmap_config(self) -> ProxemicCostmapConfig:
        """Return typed proxemic-layer config from planner fields."""
        return ProxemicCostmapConfig(
            enabled=bool(self.config.proxemic_costmap_enabled),
            personal_radius=float(self.config.proxemic_costmap_personal_radius),
            social_radius=float(self.config.proxemic_costmap_social_radius),
            personal_weight=float(self.config.proxemic_costmap_personal_weight),
            social_weight=float(self.config.proxemic_costmap_social_weight),
            velocity_elongation_factor=float(
                self.config.proxemic_costmap_velocity_elongation_factor
            ),
            max_cost=float(self.config.proxemic_costmap_max_cost),
            decay_function=str(self.config.proxemic_costmap_decay_function),
        )

    def _proxemic_costmap_metadata(self) -> dict[str, Any]:
        """Return episode metadata for the opt-in proxemic soft-cost layer."""
        config = self._proxemic_costmap_config
        return {
            "enabled": bool(config.enabled),
            "status": "ok" if config.enabled else "disabled",
            "config_hash": proxemic_config_hash(config),
            "weight": float(self.config.proxemic_costmap_weight),
            "soft_cost_only": True,
            "fallback_or_degraded": False,
        }

    def bind_env(self, env: Any) -> None:
        """Bind environment obstacle geometry for continuous static-collision checks."""
        simulator = getattr(env, "simulator", None)
        map_def = getattr(simulator, "map_def", None)
        get_obstacle_lines = getattr(simulator, "get_obstacle_lines", None)
        if map_def is None or not callable(get_obstacle_lines):
            self._continuous_static_context = None
            return
        try:
            width = float(map_def.width)
            height = float(map_def.height)
            obstacle_segments = np.asarray(get_obstacle_lines(), dtype=float).reshape(-1, 4)
        except (AttributeError, TypeError, ValueError):
            self._continuous_static_context = None
            return
        if width <= 0.0 or height <= 0.0 or not np.isfinite(width) or not np.isfinite(height):
            self._continuous_static_context = None
            return
        self._continuous_static_context = _ContinuousStaticContext(
            width=width,
            height=height,
            obstacle_segments=obstacle_segments,
        )

    def reset(self, *, seed: int | None = None) -> None:
        """Clear per-episode deterministic state.

        ``seed`` is accepted for map-runner compatibility; the planner does not
        use randomness.
        """
        _ = seed
        self._step_index = 0
        self._last_command = (0.0, 0.0)
        self._recent_commands: deque[tuple[float, float]] = deque(
            maxlen=max(int(self.config.oscillation_window), 2)
        )
        self._progress_history: deque[tuple[float, float]] = deque(maxlen=256)
        self._route_distance_history: deque[tuple[float, float]] = deque(maxlen=256)
        self._route_trace_recovery_hold_remaining = 0
        self._selected_source_counts: Counter[str] = Counter()
        self._rejection_counts: Counter[str] = Counter()
        self._unavailable_counts: Counter[str] = Counter()
        self._fallback_count = 0
        self._last_decision: dict[str, Any] | None = None
        self._clearance_context: _ObstacleClearanceContext | None = None

    def _extract_state(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Extract the structured planner state from map-runner observations.

        Returns:
            dict[str, Any]: Robot, goal, pedestrian, and timing state.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        current_speed = float(self._as_1d_float(robot_state.get("speed", [0.0]), pad=1)[0])
        robot_radius = float(
            self._as_1d_float(
                robot_state.get("radius", [self.config.robot_radius_default]),
                pad=1,
                default=self.config.robot_radius_default,
            )[0]
        )

        goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        goal_next = self._as_1d_float(goal_state.get("next", goal_current), pad=2)[:2]
        current_dist = float(np.linalg.norm(goal_current - robot_pos))
        next_dist = float(np.linalg.norm(goal_next - robot_pos))
        if next_dist > 1e-6 and current_dist <= float(self.config.waypoint_switch_distance):
            goal = goal_next
        else:
            goal = goal_current

        ped_pos = np.asarray(
            [] if ped_state.get("positions") is None else ped_state.get("positions"),
            dtype=float,
        )
        ped_vel = np.asarray(
            [] if ped_state.get("velocities") is None else ped_state.get("velocities"),
            dtype=float,
        )
        if ped_pos.ndim == 1 and ped_pos.size % 2 == 0:
            ped_pos = ped_pos.reshape(-1, 2)
        if ped_vel.ndim == 1 and ped_vel.size % 2 == 0:
            ped_vel = ped_vel.reshape(-1, 2)
        if ped_pos.ndim != 2 or ped_pos.shape[-1] != 2:
            ped_pos = np.zeros((0, 2), dtype=float)
        count_arr = self._as_1d_float(ped_state.get("count", [ped_pos.shape[0]]), pad=1)
        ped_count = max(0, min(int(count_arr[0]), ped_pos.shape[0]))
        ped_pos = ped_pos[:ped_count]
        if ped_vel.ndim != 2 or ped_vel.shape[-1] != 2:
            ped_vel = np.zeros_like(ped_pos)
        else:
            if ped_vel.shape[0] < ped_count:
                ped_vel = np.pad(
                    ped_vel,
                    ((0, ped_count - ped_vel.shape[0]), (0, 0)),
                    constant_values=0.0,
                )
            ped_vel = ped_vel[:ped_count]
        if "robot" in observation:
            ped_vel = _ego_velocity_to_world(ped_vel, heading)
        ped_radius = float(
            self._as_1d_float(
                ped_state.get("radius", [self.config.pedestrian_radius_default]),
                pad=1,
                default=self.config.pedestrian_radius_default,
            )[0]
        )

        sim = observation.get("sim") if isinstance(observation.get("sim"), dict) else {}
        dt_raw = sim.get("timestep", observation.get("dt", self.config.rollout_dt))
        dt = float(self._as_1d_float(dt_raw, pad=1, default=self.config.rollout_dt)[0])
        dt = max(dt, 1e-3)

        return {
            "robot_pos": robot_pos,
            "heading": heading,
            "current_speed": current_speed,
            "goal": goal,
            "ped_pos": ped_pos,
            "ped_vel": ped_vel,
            "robot_radius": robot_radius,
            "ped_radius": ped_radius,
            "dt": dt,
            "observation": observation,
        }

    def _nearest_ped_distance(self, robot_pos: np.ndarray, ped_pos: np.ndarray) -> float:
        """Return nearest pedestrian distance in metres."""
        if ped_pos.size == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)))

    def _human_speed_cap(self, nearest_ped: float) -> float:
        """Apply documented distance-based speed caps around humans.

        Returns:
            float: Maximum allowed linear speed for the current crowd proximity.
        """
        if nearest_ped < float(self.config.stop_distance_human):
            return min(float(self.config.very_slow_speed), float(self.config.max_linear_speed))
        if nearest_ped < float(self.config.slow_distance_human):
            return min(float(self.config.very_slow_speed), float(self.config.max_linear_speed))
        if nearest_ped < float(self.config.moderate_distance_human):
            return min(float(self.config.moderate_speed), float(self.config.max_linear_speed))
        return float(self.config.max_linear_speed)

    def _dynamic_window(
        self, current_speed: float, speed_cap: float
    ) -> tuple[float, float, float, float]:
        """Return feasible linear and angular bounds for the current control period."""
        period = max(float(self.config.control_period), 1e-3)
        v_min = max(0.0, float(current_speed) - float(self.config.max_linear_decel) * period)
        v_max = min(
            float(speed_cap),
            float(current_speed) + float(self.config.max_linear_accel) * period,
        )
        v_min = min(v_min, v_max)
        last_w = float(self._last_command[1])
        w_delta = float(self.config.max_angular_accel) * period
        w_min = max(-float(self.config.max_angular_speed), last_w - w_delta)
        w_max = min(float(self.config.max_angular_speed), last_w + w_delta)
        return v_min, v_max, w_min, w_max

    def _candidate_key(self, candidate: HybridRuleCandidate) -> tuple[Any, ...]:
        """Quantize a candidate command for duplicate removal.

        Returns:
            tuple[Any, ...]: Rounded command and sequence key.
        """
        rollout_key = tuple(
            (round(float(duration), 3), round(float(linear), 4), round(float(angular), 4))
            for duration, linear, angular in candidate.rollout_sequence
        )
        return round(float(candidate.linear), 4), round(float(candidate.angular), 4), rollout_key

    def _clip_rollout_sequence(
        self,
        sequence: tuple[tuple[float, float, float], ...],
        *,
        speed_cap: float,
    ) -> tuple[tuple[float, float, float], ...]:
        """Return finite, bounded rollout-sequence segments."""
        segments: list[tuple[float, float, float]] = []
        max_linear = max(min(float(speed_cap), float(self.config.max_linear_speed)), 0.0)
        max_angular = max(float(self.config.max_angular_speed), 0.0)
        for segment in sequence:
            try:
                duration, linear, angular = segment
                duration = float(duration)
                linear = float(linear)
                angular = float(angular)
            except (TypeError, ValueError):
                continue
            if (
                not (np.isfinite(duration) and np.isfinite(linear) and np.isfinite(angular))
                or duration <= _EPS
            ):
                continue
            segments.append(
                (
                    duration,
                    float(np.clip(linear, 0.0, max_linear)),
                    float(np.clip(angular, -max_angular, max_angular)),
                )
            )
        return tuple(segments)

    def _clip_candidate(
        self,
        candidate: HybridRuleCandidate,
        *,
        speed_cap: float,
    ) -> HybridRuleCandidate:
        """Return a candidate clipped to planner speed and turn-rate limits."""
        max_linear = max(min(float(speed_cap), float(self.config.max_linear_speed)), 0.0)
        max_angular = max(float(self.config.max_angular_speed), 0.0)
        return HybridRuleCandidate(
            float(np.clip(candidate.linear, 0.0, max_linear)),
            float(np.clip(candidate.angular, -max_angular, max_angular)),
            candidate.source,
            self._clip_rollout_sequence(candidate.rollout_sequence, speed_cap=speed_cap),
        )

    def _candidate_rollout_commands(
        self,
        candidate: HybridRuleCandidate,
        *,
        dt: float,
        steps: int,
    ) -> list[tuple[float, float]]:
        """Expand a candidate's rollout sequence to per-step commands.

        Returns:
            list[tuple[float, float]]: Linear/angular command for each rollout step.
        """
        if not candidate.rollout_sequence:
            return [(float(candidate.linear), float(candidate.angular)) for _ in range(steps)]

        segments = candidate.rollout_sequence
        commands: list[tuple[float, float]] = []
        segment_index = 0
        segment_end = float(segments[0][0])
        for step_idx in range(steps):
            elapsed = float(step_idx) * float(dt)
            while segment_index + 1 < len(segments) and elapsed >= segment_end - _EPS:
                segment_index += 1
                segment_end += float(segments[segment_index][0])
            _duration, linear, angular = segments[segment_index]
            commands.append((float(linear), float(angular)))
        return commands

    def _candidate_rollout_plan(
        self,
        candidate: HybridRuleCandidate,
    ) -> tuple[float, int, list[tuple[float, float]]]:
        """Return rollout integration settings and commands for a candidate.

        Returns:
            tuple[float, int, list[tuple[float, float]]]: Step size, step count,
            and per-step commands.
        """
        dt = max(float(self.config.rollout_dt), 1e-3)
        steps = max(int(np.ceil(float(self.config.rollout_horizon) / dt)), 1)
        return dt, steps, self._candidate_rollout_commands(candidate, dt=dt, steps=steps)

    @staticmethod
    def _rollout_linear_stats(rollout_commands: list[tuple[float, float]]) -> tuple[float, float]:
        """Return mean and max planned linear speed for scoring.

        Returns:
            tuple[float, float]: Mean and maximum linear speed across rollout steps.
        """
        linear_values = [command[0] for command in rollout_commands]
        return float(np.mean(linear_values)), float(np.max(linear_values))

    def _candidate_source_priority(self, source: str) -> int:
        """Return a source priority for preserving specialized duplicate commands."""
        return {
            "corridor_subgoal": 30,
            "route_guide": 20,
        }.get(source, 10)

    def _route_point(self, route_corridor: dict[str, Any] | None, key: str) -> np.ndarray | None:
        """Read one finite ``(x, y)`` point from route-corridor diagnostics.

        Returns:
            np.ndarray | None: Point array, or ``None`` when unavailable.
        """
        if not isinstance(route_corridor, dict):
            return None
        raw = route_corridor.get(key)
        try:
            point = np.asarray(raw, dtype=float).reshape(-1)[:2]
        except (TypeError, ValueError):
            return None
        if point.shape[0] != 2 or not np.all(np.isfinite(point)):
            return None
        return point

    def _route_float(self, route_corridor: dict[str, Any] | None, key: str) -> float | None:
        """Read one finite scalar from route-corridor diagnostics.

        Returns:
            float | None: Finite scalar, or ``None`` when unavailable.
        """
        if not isinstance(route_corridor, dict):
            return None
        value = route_corridor.get(key)
        if isinstance(value, int | float | np.integer | np.floating) and np.isfinite(value):
            return float(value)
        return None

    def _route_progress_pair(
        self, route_corridor: dict[str, Any] | None
    ) -> tuple[float, float] | None:
        """Read finite 1s and 3s route-arc progress diagnostics.

        Returns:
            tuple[float, float] | None: Progress over 1s and 3s, or ``None`` when unavailable.
        """
        if not isinstance(route_corridor, dict):
            return None
        route_progress = route_corridor.get("route_arc_progress_windows")
        if not isinstance(route_progress, dict):
            return None
        try:
            route_progress_1s = float(route_progress["1s"])
            route_progress_3s = float(route_progress["3s"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(route_progress_1s) or not np.isfinite(route_progress_3s):
            return None
        return route_progress_1s, route_progress_3s

    def _route_tangent_heading(self, route_corridor: dict[str, Any] | None) -> float | None:
        """Return a finite route tangent heading, deriving it from route points if needed."""
        heading = self._route_float(route_corridor, "route_tangent_heading")
        if heading is not None:
            return heading
        start = self._route_point(route_corridor, "route_start_world")
        stop = self._route_point(route_corridor, "route_next_world")
        if stop is None:
            stop = self._route_point(route_corridor, "route_waypoint_world")
        if start is None or stop is None:
            return None
        delta = stop - start
        if float(np.linalg.norm(delta)) <= _EPS:
            return None
        return float(np.arctan2(delta[1], delta[0]))

    @staticmethod
    def _lateral_offset_to_segment(
        point: np.ndarray,
        segment_start: np.ndarray,
        segment_stop: np.ndarray,
    ) -> float | None:
        """Return lateral distance from ``point`` to a world-space segment."""
        segment = segment_stop - segment_start
        length = float(np.linalg.norm(segment))
        if length <= _EPS:
            return None
        relative = point - segment_start
        return float(abs(segment[0] * relative[1] - segment[1] * relative[0]) / length)

    def _corridor_subgoal_activation(
        self,
        *,
        route_corridor: dict[str, Any] | None,
        progress_windows: dict[str, float],
        nearest_ped: float,
    ) -> dict[str, Any]:
        """Return fail-closed activation diagnostics for route-corridor subgoals."""
        diagnostics: dict[str, Any] = {
            "enabled": bool(self.config.corridor_subgoal_enabled),
            "active": False,
            "reason": "disabled",
            "candidate_count": 0,
            "nearest_pedestrian_distance": _finite_or_none(nearest_ped),
            "min_nearest_pedestrian_distance": float(
                self.config.corridor_subgoal_min_nearest_ped_distance
            ),
        }
        if not bool(self.config.corridor_subgoal_enabled):
            return diagnostics
        if self._route_guide is None:
            diagnostics["reason"] = "route_guide_disabled"
            return diagnostics
        if not isinstance(route_corridor, dict):
            diagnostics["reason"] = "missing_route_geometry"
            return diagnostics
        if nearest_ped < float(self.config.corridor_subgoal_min_nearest_ped_distance):
            diagnostics["reason"] = "near_pedestrian"
            return diagnostics
        route_remaining = self._route_float(route_corridor, "route_remaining_distance")
        diagnostics["route_remaining_distance"] = _finite_or_none(route_remaining)
        if route_remaining is None or route_remaining < float(
            self.config.corridor_subgoal_min_route_remaining_distance
        ):
            diagnostics["reason"] = "route_too_short"
            return diagnostics
        waypoint = self._route_point(route_corridor, "route_waypoint_world")
        tangent_heading = self._route_tangent_heading(route_corridor)
        if waypoint is None or tangent_heading is None:
            diagnostics["reason"] = "incomplete_route_geometry"
            return diagnostics
        lateral_offset = self._route_float(route_corridor, "robot_lateral_offset_to_corridor")
        diagnostics["robot_lateral_offset_to_corridor"] = _finite_or_none(lateral_offset)
        if lateral_offset is not None and lateral_offset > float(
            self.config.corridor_subgoal_max_lateral_offset
        ):
            diagnostics["reason"] = "outside_corridor_band"
            return diagnostics

        goal_progress_1s = float(progress_windows.get("1s", 0.0))
        goal_progress_3s = float(progress_windows.get("3s", 0.0))
        route_progress_pair = self._route_progress_pair(route_corridor)
        if route_progress_pair is None:
            diagnostics["reason"] = "missing_route_progress"
            return diagnostics
        route_progress_1s, route_progress_3s = route_progress_pair
        diagnostics.update(
            {
                "goal_progress_1s": goal_progress_1s,
                "goal_progress_3s": goal_progress_3s,
                "route_arc_progress_1s": route_progress_1s,
                "route_arc_progress_3s": route_progress_3s,
            }
        )
        goal_progress_3s_threshold = float(self.config.corridor_subgoal_goal_stall_progress_3s)
        goal_progress_1s_threshold = goal_progress_3s_threshold / 3.0
        goal_stalled = (
            goal_progress_1s <= goal_progress_1s_threshold
            and goal_progress_3s <= goal_progress_3s_threshold
        )
        route_stalled = route_progress_3s <= float(
            self.config.corridor_subgoal_route_stall_progress_3s
        )
        route_regressing = route_progress_1s <= float(
            self.config.corridor_subgoal_route_regression_1s
        )
        diagnostics.update(
            {
                "goal_stalled": bool(goal_stalled),
                "route_stalled": bool(route_stalled),
                "route_regressing": bool(route_regressing),
            }
        )
        if not (goal_stalled and (route_stalled or route_regressing)):
            diagnostics["reason"] = "progress_not_stalled"
            return diagnostics
        diagnostics["active"] = True
        diagnostics["reason"] = "active"
        return diagnostics

    def _route_trace_recovery_signal(  # noqa: C901
        self,
        *,
        route_corridor: dict[str, Any] | None,
        progress_windows: dict[str, float],
        nearest_ped: float,
    ) -> dict[str, Any]:
        """Return fail-closed trace-level route recovery diagnostics."""
        diagnostics: dict[str, Any] = {
            "enabled": bool(self.config.route_trace_recovery_enabled),
            "active": False,
            "reason": "disabled",
            "selected": False,
            "nearest_pedestrian_distance": _finite_or_none(nearest_ped),
            "min_nearest_pedestrian_distance": float(
                self.config.route_trace_recovery_min_nearest_ped_distance
            ),
        }
        if not bool(self.config.route_trace_recovery_enabled):
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics
        if self._route_guide is None:
            diagnostics["reason"] = "route_guide_disabled"
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics
        if not isinstance(route_corridor, dict):
            diagnostics["reason"] = "missing_route_geometry"
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics
        if nearest_ped < float(self.config.route_trace_recovery_min_nearest_ped_distance):
            diagnostics["reason"] = "near_pedestrian"
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics
        route_remaining = self._route_float(route_corridor, "route_remaining_distance")
        diagnostics["route_remaining_distance"] = _finite_or_none(route_remaining)
        if route_remaining is None or route_remaining < float(
            self.config.route_trace_recovery_min_route_remaining_distance
        ):
            diagnostics["reason"] = "route_too_short"
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics
        tangent_heading = self._route_tangent_heading(route_corridor)
        waypoint = self._route_point(route_corridor, "route_waypoint_world")
        if tangent_heading is None or waypoint is None:
            diagnostics["reason"] = "incomplete_route_geometry"
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics
        lateral_offset = self._route_float(route_corridor, "robot_lateral_offset_to_corridor")
        diagnostics["robot_lateral_offset_to_corridor"] = _finite_or_none(lateral_offset)
        if lateral_offset is not None and lateral_offset > float(
            self.config.corridor_subgoal_max_lateral_offset
        ):
            diagnostics["reason"] = "outside_corridor_band"
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics
        route_progress_pair = self._route_progress_pair(route_corridor)
        if route_progress_pair is None:
            diagnostics["reason"] = "missing_route_progress"
            self._route_trace_recovery_hold_remaining = 0
            return diagnostics

        goal_progress_3s = float(progress_windows.get("3s", 0.0))
        route_progress_1s, route_progress_3s = route_progress_pair
        goal_stalled = goal_progress_3s <= float(
            self.config.route_trace_recovery_goal_stall_progress_3s
        )
        route_stalled = route_progress_3s <= float(
            self.config.route_trace_recovery_route_stall_progress_3s
        )
        route_regressing = route_progress_1s <= float(
            self.config.route_trace_recovery_route_regression_1s
        )
        trigger = bool(route_regressing or (goal_stalled and route_stalled))
        diagnostics.update(
            {
                "goal_progress_3s": goal_progress_3s,
                "route_arc_progress_1s": route_progress_1s,
                "route_arc_progress_3s": route_progress_3s,
                "goal_stalled": bool(goal_stalled),
                "route_stalled": bool(route_stalled),
                "route_regressing": bool(route_regressing),
                "hold_remaining": int(self._route_trace_recovery_hold_remaining),
            }
        )
        if trigger:
            self._route_trace_recovery_hold_remaining = max(
                int(self.config.route_trace_recovery_hold_steps), 0
            )
            diagnostics["active"] = True
            diagnostics["reason"] = "route_regressing" if route_regressing else "route_stalled"
            diagnostics["hold_remaining"] = int(self._route_trace_recovery_hold_remaining)
            return diagnostics
        if self._route_trace_recovery_hold_remaining > 0:
            self._route_trace_recovery_hold_remaining -= 1
            diagnostics["active"] = True
            diagnostics["reason"] = "hold"
            diagnostics["hold_remaining"] = int(self._route_trace_recovery_hold_remaining)
            return diagnostics

        diagnostics["reason"] = "progress_not_stalled"
        return diagnostics

    def _corridor_subgoal_activation_for_trace_recovery(
        self,
        corridor_subgoal: dict[str, Any],
        route_trace_recovery: dict[str, Any],
    ) -> dict[str, Any]:
        """Return subgoal activation widened by a route-trace recovery signal."""
        if bool(corridor_subgoal.get("active")) or not bool(route_trace_recovery.get("active")):
            return corridor_subgoal
        widened = dict(corridor_subgoal)
        widened.update(
            {
                "active": True,
                "reason": "route_trace_recovery",
                "route_trace_recovery_reason": route_trace_recovery.get("reason"),
            }
        )
        return widened

    def _select_route_trace_recovery_evaluation(
        self,
        accepted: list[dict[str, Any]],
        route_trace_recovery: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Select an already accepted route-recovery candidate for trace mode.

        Returns:
            dict[str, Any] | None: Selected accepted evaluation, or ``None`` when
            trace recovery is inactive or no accepted recovery candidate exists.
        """
        if not bool(route_trace_recovery.get("active")):
            return None
        priority = {"corridor_subgoal": 0, "route_guide": 1}
        candidates: list[dict[str, Any]] = []
        for evaluation in accepted:
            candidate = evaluation.get("candidate")
            if not isinstance(candidate, HybridRuleCandidate) or candidate.source not in priority:
                continue
            route_progress = float(evaluation.get("terms", {}).get("route_arc_progress", 0.0))
            if route_progress <= 0.0:
                continue
            candidates.append(evaluation)
        if not candidates:
            route_trace_recovery["selected_reason"] = "no_accepted_recovery_candidate"
            return None
        candidates.sort(
            key=lambda item: (
                priority[item["candidate"].source],
                -float(item.get("terms", {}).get("route_arc_progress", 0.0)),
                -float(item.get("score", 0.0)),
            )
        )
        selected = candidates[0]
        route_trace_recovery.update(
            {
                "selected": True,
                "selected_source": selected["candidate"].source,
                "selected_reason": "accepted_recovery_candidate",
            }
        )
        return selected

    def _corridor_subgoal_candidates(
        self,
        *,
        state: dict[str, Any],
        speed_cap: float,
        route_corridor: dict[str, Any] | None,
        activation: dict[str, Any] | None,
        bounds: tuple[float, float, float, float],
    ) -> list[HybridRuleCandidate]:
        """Generate route-corridor recovery candidates when activation is satisfied.

        Returns:
            list[HybridRuleCandidate]: Candidate commands from the subgoal primitive.
        """
        if not activation or not bool(activation.get("active")):
            return []
        v_min, v_max, w_min, w_max = bounds
        robot_pos = state["robot_pos"]
        heading = float(state["heading"])
        waypoint = self._route_point(route_corridor, "route_waypoint_world")
        tangent_heading = self._route_tangent_heading(route_corridor)
        if waypoint is None or tangent_heading is None:
            return []

        waypoint_vec = waypoint - robot_pos
        waypoint_heading = (
            tangent_heading
            if float(np.linalg.norm(waypoint_vec)) <= _EPS
            else float(np.arctan2(waypoint_vec[1], waypoint_vec[0]))
        )
        tangent_error = _wrap_angle(tangent_heading - heading)
        waypoint_error = _wrap_angle(waypoint_heading - heading)
        if abs(tangent_error) >= float(self.config.corridor_subgoal_turn_in_place_error):
            desired_heading_error = tangent_error
        else:
            desired_heading_error = 0.5 * tangent_error + 0.5 * waypoint_error
        desired_angular = float(
            np.clip(
                float(self.config.corridor_subgoal_heading_gain)
                * desired_heading_error
                / max(float(self.config.control_period), _EPS),
                w_min,
                w_max,
            )
        )
        desired_speed = min(
            float(speed_cap),
            float(self.config.corridor_subgoal_speed),
            float(self.config.max_linear_speed),
        )
        candidates: list[HybridRuleCandidate] = []
        if abs(desired_heading_error) >= float(self.config.corridor_subgoal_turn_in_place_error):
            rollout_horizon = max(float(self.config.rollout_horizon), float(self.config.rollout_dt))
            turn_duration = rollout_horizon
            if abs(desired_angular) > _EPS:
                turn_duration = min(
                    abs(desired_heading_error) / abs(desired_angular), rollout_horizon
                )
            segments: list[tuple[float, float, float]] = [(turn_duration, 0.0, desired_angular)]
            forward_duration = rollout_horizon - turn_duration
            if forward_duration > _EPS and desired_speed > float(
                self.config.freezing_speed_threshold
            ):
                segments.append((forward_duration, desired_speed, 0.0))
            candidates.append(
                HybridRuleCandidate(
                    0.0,
                    desired_angular,
                    "corridor_subgoal",
                    tuple(segments),
                )
            )
            return candidates
        if desired_speed > float(self.config.freezing_speed_threshold):
            alignment = max(0.0, np.cos(desired_heading_error))
            linear = float(np.clip(desired_speed * alignment, v_min, v_max))
            if linear > float(self.config.freezing_speed_threshold):
                candidates.append(
                    HybridRuleCandidate(
                        linear,
                        desired_angular,
                        "corridor_subgoal",
                        ((float(self.config.rollout_horizon), linear, desired_angular),),
                    )
                )
        return candidates

    def _generate_candidates(
        self,
        state: dict[str, Any],
        speed_cap: float,
        *,
        route_corridor: dict[str, Any] | None = None,
        corridor_subgoal: dict[str, Any] | None = None,
    ) -> list[HybridRuleCandidate]:
        """Generate deterministic DWA, path-following, and safety candidates.

        Returns:
            list[HybridRuleCandidate]: Unique candidate commands.
        """
        v_min, v_max, w_min, w_max = self._dynamic_window(state["current_speed"], speed_cap)
        candidates: list[HybridRuleCandidate] = []

        for linear in np.linspace(v_min, v_max, max(int(self.config.linear_samples), 2)):
            for angular in np.linspace(w_min, w_max, max(int(self.config.angular_samples), 3)):
                candidates.append(
                    HybridRuleCandidate(float(linear), float(angular), "dynamic_window")
                )

        robot_pos = state["robot_pos"]
        goal = state["goal"]
        heading = float(state["heading"])
        goal_vec = goal - robot_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist > _EPS:
            goal_heading = float(np.arctan2(goal_vec[1], goal_vec[0]))
            heading_error = _wrap_angle(goal_heading - heading)
            for lookahead in self.config.lookahead_distances:
                lookahead_ratio = _clip01(goal_dist / max(float(lookahead), _EPS))
                desired_v = min(speed_cap, float(self.config.max_linear_speed) * lookahead_ratio)
                desired_v *= max(0.15, 1.0 - min(abs(heading_error), np.pi / 2.0) / (np.pi / 2.0))
                desired_w = np.clip(
                    1.4 * heading_error,
                    -float(self.config.max_angular_speed),
                    float(self.config.max_angular_speed),
                )
                candidates.append(
                    HybridRuleCandidate(
                        float(np.clip(desired_v, v_min, v_max)),
                        float(np.clip(desired_w, w_min, w_max)),
                        f"path_follow_{lookahead:.1f}m",
                    )
                )

        if self._route_guide is not None:
            route_linear, route_angular = self._route_guide.plan(state["observation"])
            candidates.append(
                HybridRuleCandidate(
                    float(np.clip(route_linear, 0.0, speed_cap)),
                    float(
                        np.clip(
                            route_angular,
                            -float(self.config.max_angular_speed),
                            float(self.config.max_angular_speed),
                        )
                    ),
                    "route_guide",
                )
            )

        subgoal_candidates = self._corridor_subgoal_candidates(
            state=state,
            speed_cap=speed_cap,
            route_corridor=route_corridor,
            activation=corridor_subgoal,
            bounds=(v_min, v_max, w_min, w_max),
        )
        if corridor_subgoal is not None:
            corridor_subgoal["candidate_count"] = len(subgoal_candidates)
        candidates.extend(subgoal_candidates)

        candidates.extend(
            [
                HybridRuleCandidate(0.0, 0.0, "stop"),
                HybridRuleCandidate(
                    min(float(self.config.very_slow_speed), speed_cap), 0.0, "creep"
                ),
                HybridRuleCandidate(0.0, float(np.clip(0.45, w_min, w_max)), "rotate_left"),
                HybridRuleCandidate(0.0, float(np.clip(-0.45, w_min, w_max)), "rotate_right"),
            ]
        )

        unique: dict[tuple[Any, ...], HybridRuleCandidate] = {}
        for candidate in candidates:
            clipped = self._clip_candidate(candidate, speed_cap=speed_cap)
            key = self._candidate_key(clipped)
            existing = unique.get(key)
            if existing is None or self._candidate_source_priority(
                clipped.source
            ) > self._candidate_source_priority(existing.source):
                unique[key] = clipped
        return list(unique.values())

    def _build_clearance_context(
        self, observation: dict[str, Any]
    ) -> _ObstacleClearanceContext | None:
        """Build a per-plan obstacle clearance lookup from the occupancy grid.

        Returns:
            _ObstacleClearanceContext | None: Lookup context, or ``None`` when
            the observation has no usable static-obstacle grid.
        """
        payload = self._obstacle_grid_payload(observation)
        if payload is None:
            return None
        grid, meta, channel, resolution = payload
        channel_grid = np.asarray(grid[channel], dtype=float)
        blocked = channel_grid >= float(self.config.obstacle_threshold)
        clearance_cells = None if not np.any(blocked) else distance_transform_edt(~blocked)
        return _ObstacleClearanceContext(
            observation_id=id(observation),
            grid=grid,
            meta=meta,
            channel=channel,
            resolution=resolution,
            clearance_cells=clearance_cells,
        )

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, Any]) -> float:
        """Approximate static-obstacle clearance from occupancy-grid observations.

        Returns:
            float: Clearance in metres, or infinity when no obstacle grid is available.
        """
        context = (
            self._clearance_context
            if self._clearance_context is not None
            and self._clearance_context.observation_id == id(observation)
            else self._build_clearance_context(observation)
        )
        if context is None:
            return float("inf")
        indices = self._world_to_grid(
            point,
            context.meta,
            grid_shape=(context.grid.shape[1], context.grid.shape[2]),
        )
        if indices is None:
            return 0.0
        row, col = indices
        if context.clearance_cells is None:
            return float("inf")
        clearance_cells = float(context.clearance_cells[row, col])
        if clearance_cells <= 0.0:
            return 0.0
        radius = max(int(self.config.obstacle_search_cells), 1)
        if clearance_cells > float(radius):
            return float("inf")
        return float(clearance_cells * max(float(context.resolution), 1e-6))

    def _continuous_static_collision(self, point: np.ndarray, radius: float) -> bool | None:
        """Return continuous obstacle collision status when environment geometry is bound."""
        context = self._continuous_static_context
        if context is None:
            return None
        position = np.asarray(point, dtype=float).reshape(-1)[:2]
        if position.shape[0] != 2 or not np.all(np.isfinite(position)):
            return True
        x, y = float(position[0]), float(position[1])
        if not (0.0 <= x <= context.width and 0.0 <= y <= context.height):
            return True
        collision_radius = max(float(radius), 0.0)
        return bool(
            circle_collides_any_lines(((x, y), collision_radius), context.obstacle_segments)
        )

    def _static_collision_rejection(
        self,
        *,
        candidate: HybridRuleCandidate,
        observation: dict[str, Any],
        robot_pos: np.ndarray,
        hard_static_clearance: float,
        use_continuous_static_check: bool,
        t: float,
    ) -> dict[str, Any] | None:
        """Return an occupied-cell or continuous-obstacle rejection for a rollout pose."""
        obstacle_value = 0.0
        payload = self._obstacle_grid_payload(observation)
        if payload is not None:
            grid, meta, channel, _resolution = payload
            obstacle_value = self._grid_value(robot_pos, grid, meta, channel)
        if obstacle_value >= float(self.config.obstacle_threshold):
            return {
                "accepted": False,
                "reason": "static_collision",
                "candidate": candidate,
                "obstacle_value": float(obstacle_value),
                "time": float(t),
            }
        continuous_static_collision = (
            self._continuous_static_collision(robot_pos, hard_static_clearance)
            if use_continuous_static_check
            else None
        )
        if continuous_static_collision is True:
            return {
                "accepted": False,
                "reason": "static_collision",
                "candidate": candidate,
                "continuous_static_collision": True,
                "hard_static_clearance": float(hard_static_clearance),
                "time": float(t),
            }
        return None

    def _ttc_proxy(
        self,
        *,
        robot_pos: np.ndarray,
        heading: float,
        command: HybridRuleCandidate,
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
    ) -> float:
        """Compute a conservative closest-approach TTC proxy.

        Returns:
            float: Predicted TTC in seconds, or infinity when no risky approach is detected.
        """
        if ped_pos.size == 0:
            return float("inf")
        robot_vel = np.array(
            [command.linear * np.cos(heading), command.linear * np.sin(heading)],
            dtype=float,
        )
        rel_pos = ped_pos - robot_pos[None, :]
        rel_vel = ped_vel - robot_vel[None, :]
        speed_sq = np.sum(rel_vel * rel_vel, axis=1)
        valid = speed_sq > 1e-8
        if not np.any(valid):
            return float("inf")
        ttc = -np.sum(rel_pos[valid] * rel_vel[valid], axis=1) / speed_sq[valid]
        ttc = np.clip(ttc, 0.0, float(self.config.soft_prediction_horizon))
        closest = rel_pos[valid] + rel_vel[valid] * ttc[:, None]
        closest_dist = np.linalg.norm(closest, axis=1)
        risky = closest_dist <= float(self.config.desired_dynamic_clearance)
        if not np.any(risky):
            return float("inf")
        positive = ttc[risky]
        positive = positive[positive > 1e-6]
        if positive.size == 0:
            return 0.0
        return float(np.min(positive))

    def _oscillation_penalty(self, angular: float) -> float:
        """Penalize recent alternating turn commands.

        Returns:
            float: Normalized oscillation penalty in ``[0, 1]``.
        """
        recent_turns = [np.sign(cmd[1]) for cmd in self._recent_commands if abs(cmd[1]) > 0.15]
        if len(recent_turns) < 2 or abs(angular) <= 0.15:
            return 0.0
        alternations = sum(1 for prev, cur in pairwise(recent_turns) if prev * cur < 0)
        candidate_flip = 1 if recent_turns[-1] * np.sign(angular) < 0 else 0
        denom = max(len(recent_turns), 1)
        return _clip01((alternations + candidate_flip) / denom)

    def _progress_windows(self, current_time: float, goal_distance: float) -> dict[str, float]:
        """Return sliding-window goal-distance progress diagnostics."""
        return self._distance_progress_windows(
            self._progress_history,
            current_time=current_time,
            current_distance=goal_distance,
        )

    @staticmethod
    def _distance_progress_windows(
        history: deque[tuple[float, float]],
        *,
        current_time: float,
        current_distance: float,
    ) -> dict[str, float]:
        """Return sliding-window distance-progress diagnostics for a history."""
        windows: dict[str, float] = {}
        for seconds in (1.0, 3.0, 5.0):
            reference = None
            for time_value, distance_value in history:
                if time_value >= current_time - seconds:
                    reference = distance_value
                    break
            if reference is None and history:
                reference = history[0][1]
            windows[f"{seconds:.0f}s"] = (
                0.0 if reference is None else float(reference - current_distance)
            )
        return windows

    def _route_corridor_diagnostics(
        self,
        observation: dict[str, Any],
        *,
        current_time: float,
    ) -> dict[str, Any] | None:
        """Return additive route-corridor diagnostics when route guide is enabled."""
        if self._route_guide is None:
            return None
        route_geometry = getattr(self._route_guide, "route_geometry", None)
        if not callable(route_geometry):
            return None
        try:
            diagnostics = route_geometry(observation)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None
        if not isinstance(diagnostics, dict):
            return None
        route_remaining = diagnostics.get("route_remaining_distance")
        if isinstance(route_remaining, int | float | np.integer | np.floating) and np.isfinite(
            route_remaining
        ):
            remaining = float(route_remaining)
            self._route_distance_history.append((current_time, remaining))
            diagnostics = dict(diagnostics)
            diagnostics["route_arc_progress_windows"] = self._distance_progress_windows(
                self._route_distance_history,
                current_time=current_time,
                current_distance=remaining,
            )
        return diagnostics

    def _static_clearance_escape_allowed(
        self,
        *,
        candidate: HybridRuleCandidate,
        initial_clearance: float,
        current_min_clearance: float,
        hard_static_clearance: float,
    ) -> bool:
        """Allow slow escape from an already conservative static-clearance violation.

        Returns:
            bool: True when the candidate is a bounded, non-worsening escape command.
        """
        if not bool(self.config.static_clearance_escape_enabled):
            return False
        if not np.isfinite(initial_clearance) or initial_clearance <= 0.0:
            return False
        min_escape_clearance = float(self.config.static_clearance_escape_min_clearance)
        if initial_clearance < min_escape_clearance:
            return False
        if current_min_clearance < min_escape_clearance:
            return False
        if candidate.linear <= float(self.config.freezing_speed_threshold):
            return False
        if candidate.linear > float(self.config.static_clearance_escape_max_speed):
            return False
        tolerance = max(float(self.config.static_clearance_escape_tolerance), 0.0)
        if initial_clearance > hard_static_clearance:
            return False
        return bool(current_min_clearance + tolerance >= initial_clearance)

    def _static_corridor_transit_allowed(
        self,
        *,
        candidate: HybridRuleCandidate,
        initial_clearance: float,
        current_min_clearance: float,
        hard_static_clearance: float,
        step_progress: float,
        progress_windows: dict[str, float],
    ) -> bool:
        """Allow slow progress through a narrow conservative static-clearance band.

        Returns:
            bool: True when a candidate preserves occupied-cell collision rejection,
            remains above an explicit minimum clearance, and only enters the hard
            clearance band by a small configured tolerance while making progress.
        """
        if not bool(self.config.static_corridor_transit_enabled):
            return False
        if candidate.linear <= float(self.config.freezing_speed_threshold):
            return False
        if candidate.linear > float(self.config.static_clearance_escape_max_speed):
            return False
        if step_progress <= 0.0:
            return False
        min_recent_progress = max(float(self.config.static_corridor_transit_min_progress_3s), 0.0)
        if float(progress_windows.get("3s", 0.0)) < min_recent_progress:
            return False
        if not np.isfinite(initial_clearance) or not np.isfinite(current_min_clearance):
            return False
        if initial_clearance <= hard_static_clearance:
            return False
        initial_band = max(float(self.config.static_corridor_transit_initial_band), 0.0)
        if initial_clearance > hard_static_clearance + initial_band:
            return False
        min_transit_clearance = float(self.config.static_clearance_escape_min_clearance)
        if current_min_clearance < min_transit_clearance:
            return False
        tolerance = max(float(self.config.static_corridor_transit_tolerance), 0.0)
        return bool(current_min_clearance + tolerance >= hard_static_clearance)

    def _static_clearance_violation_policy(
        self,
        *,
        candidate: HybridRuleCandidate,
        initial_clearance: float,
        current_min_clearance: float,
        hard_static_clearance: float,
        step_progress: float,
        progress_windows: dict[str, float],
    ) -> str | None:
        """Classify an allowed conservative static-clearance band exception.

        Returns:
            str | None: ``"escape"`` or ``"corridor_transit"`` when a guarded
            exception applies, otherwise ``None``.
        """
        if candidate.source == "corridor_subgoal":
            return None
        if self._static_clearance_escape_allowed(
            candidate=candidate,
            initial_clearance=initial_clearance,
            current_min_clearance=current_min_clearance,
            hard_static_clearance=hard_static_clearance,
        ):
            return "escape"
        if self._static_corridor_transit_allowed(
            candidate=candidate,
            initial_clearance=initial_clearance,
            current_min_clearance=current_min_clearance,
            hard_static_clearance=hard_static_clearance,
            step_progress=step_progress,
            progress_windows=progress_windows,
        ):
            return "corridor_transit"
        return None

    def _static_recenter_probe_score(
        self,
        *,
        candidate: HybridRuleCandidate,
        observation: dict[str, Any],
        state: dict[str, Any],
        hard_static_clearance: float,
    ) -> float:
        """Score rotate-in-place candidates by the next safe forward heading.

        Returns:
            float: ``1.0`` when the rotation creates a forward rollout that stays
            outside the hard static-clearance band, otherwise ``0.0``.
        """
        if not bool(self.config.static_recenter_enabled):
            return 0.0
        if candidate.linear > float(self.config.freezing_speed_threshold):
            return 0.0
        if abs(candidate.angular) < float(self.config.deadlock_rotation_threshold):
            return 0.0

        dt = max(float(self.config.rollout_dt), 1e-3)
        steps = max(int(np.ceil(float(self.config.rollout_horizon) / dt)), 1)
        probe_speed = min(
            max(float(self.config.static_recenter_probe_speed), 0.0),
            max(float(self.config.static_clearance_escape_max_speed), 0.0),
            max(float(self.config.max_linear_speed), 0.0),
        )
        if probe_speed <= float(self.config.freezing_speed_threshold):
            return 0.0

        robot_pos = np.array(state["robot_pos"], dtype=float)
        heading = _wrap_angle(float(state["heading"]) + float(candidate.angular) * dt * steps)
        for _step_idx in range(steps):
            robot_pos = (
                robot_pos
                + np.array([probe_speed * np.cos(heading), probe_speed * np.sin(heading)]) * dt
            )
            payload = self._obstacle_grid_payload(observation)
            if payload is not None:
                grid, meta, channel, _resolution = payload
                if self._grid_value(robot_pos, grid, meta, channel) >= float(
                    self.config.obstacle_threshold
                ):
                    return 0.0
            if self._min_obstacle_clearance(robot_pos, observation) <= hard_static_clearance:
                return 0.0
        return 1.0

    def _static_recenter_term(
        self,
        *,
        candidate: HybridRuleCandidate,
        observation: dict[str, Any],
        state: dict[str, Any],
        hard_static_clearance: float,
        stalled_progress: bool,
        start_dist: float,
        nearest_ped: float,
    ) -> float:
        """Return the scored static-recenter term when stalled away from pedestrians."""
        if not (
            stalled_progress
            and start_dist > float(self.config.goal_far_distance)
            and nearest_ped >= float(self.config.slow_distance_human)
        ):
            return 0.0
        return self._static_recenter_probe_score(
            candidate=candidate,
            observation=observation,
            state=state,
            hard_static_clearance=hard_static_clearance,
        )

    def _route_guide_commitment_term(
        self,
        *,
        candidate: HybridRuleCandidate,
        progress_windows: dict[str, float],
        start_dist: float,
    ) -> float:
        """Return the route-guide commitment bonus term for stalled route candidates."""
        if candidate.source != "route_guide":
            return 0.0
        if float(progress_windows.get("3s", 0.0)) > float(
            self.config.route_guide_commitment_progress_threshold
        ) or start_dist <= float(self.config.goal_far_distance):
            return 0.0
        return 1.0

    def _zero_corridor_subgoal_terms(self) -> dict[str, float]:
        """Return neutral route-corridor score terms for non-subgoal candidates."""
        return {
            "corridor_subgoal_route_progress": 0.0,
            "corridor_subgoal_centering": 0.0,
            "corridor_subgoal_tangent_alignment": 0.0,
            "corridor_subgoal_clearance_margin": 0.0,
            "corridor_subgoal_continuity": 0.0,
        }

    def _corridor_subgoal_score_terms(
        self,
        *,
        candidate: HybridRuleCandidate,
        route_corridor: dict[str, Any] | None,
        state: dict[str, Any],
        end_pos: np.ndarray,
        end_heading: float,
        min_static_clearance: float,
        hard_static_clearance: float,
    ) -> dict[str, float]:
        """Score route-corridor progress terms for accepted subgoal candidates.

        Returns:
            dict[str, float]: Normalized score terms for the route-corridor primitive.
        """
        terms = self._zero_corridor_subgoal_terms()
        if candidate.source != "corridor_subgoal" or not isinstance(route_corridor, dict):
            return terms
        tangent_heading = self._route_tangent_heading(route_corridor)
        if tangent_heading is None:
            return terms
        route_dir = np.array([np.cos(tangent_heading), np.sin(tangent_heading)], dtype=float)
        displacement = end_pos - state["robot_pos"]
        route_progress = float(np.dot(displacement, route_dir))
        max_route_progress = max(
            float(self.config.max_linear_speed) * float(self.config.rollout_horizon),
            _EPS,
        )
        terms["corridor_subgoal_route_progress"] = float(
            np.clip(route_progress / max_route_progress, -1.0, 1.0)
        )

        heading_error = abs(_wrap_angle(tangent_heading - end_heading))
        terms["corridor_subgoal_tangent_alignment"] = float(0.5 + 0.5 * np.cos(heading_error))

        start = self._route_point(route_corridor, "route_start_world")
        stop = self._route_point(route_corridor, "route_next_world")
        if stop is None:
            stop = self._route_point(route_corridor, "route_waypoint_world")
        if start is not None and stop is not None:
            end_offset = self._lateral_offset_to_segment(end_pos, start, stop)
            if end_offset is not None:
                start_offset = self._route_float(route_corridor, "robot_lateral_offset_to_corridor")
                corridor_width = self._route_float(route_corridor, "corridor_width_estimate")
                half_width = max(
                    0.5
                    * (
                        corridor_width
                        if corridor_width is not None
                        else 2.0 * float(self.config.desired_static_clearance)
                    ),
                    _EPS,
                )
                if start_offset is None:
                    terms["corridor_subgoal_centering"] = 1.0 - _clip01(end_offset / half_width)
                else:
                    terms["corridor_subgoal_centering"] = _clip01(
                        0.5 + (start_offset - end_offset) / half_width
                    )

        if np.isinf(min_static_clearance):
            terms["corridor_subgoal_clearance_margin"] = 1.0
        else:
            terms["corridor_subgoal_clearance_margin"] = _clip01(
                (float(min_static_clearance) - float(hard_static_clearance))
                / max(float(self.config.desired_static_clearance), _EPS)
            )
        velocity_delta = abs(candidate.linear - float(self._last_command[0]))
        angular_delta = abs(candidate.angular - float(self._last_command[1]))
        terms["corridor_subgoal_continuity"] = 1.0 - 0.5 * (
            _clip01(velocity_delta / max(float(self.config.max_linear_speed), _EPS))
            + _clip01(angular_delta / max(float(self.config.max_angular_speed), _EPS))
        )
        return terms

    def _actuation_scoring_metadata(self) -> dict[str, Any] | None:
        """Return synthetic actuation-scoring metadata for exploratory variants."""
        if not bool(self.config.actuation_score_enabled):
            return None
        return {
            "profile": str(self.config.actuation_profile_name),
            "claim_scope": str(self.config.actuation_claim_scope),
            "interpretation": "synthetic_diagnostic_only",
            "hardware_calibrated": False,
            "paper_facing": False,
            "projection_policy": "score_penalty_only",
            "limits": {
                "max_linear_accel_m_s2": float(self.config.actuation_max_linear_accel),
                "max_linear_decel_m_s2": float(self.config.actuation_max_linear_decel),
                "max_yaw_rate_rad_s": float(self.config.actuation_max_yaw_rate),
                "max_angular_accel_rad_s2": float(self.config.actuation_max_angular_accel),
            },
        }

    def _actuation_terms(
        self,
        *,
        candidate: HybridRuleCandidate,
        state: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any] | None]:
        """Score synthetic actuation-envelope risk without replacing diagnostics.

        Returns:
            Scoring terms plus optional per-candidate diagnostic metadata.
        """
        if not bool(self.config.actuation_score_enabled):
            return {
                "actuation_feasibility": 1.0,
                "actuation_clip_risk": 0.0,
            }, None

        dt = max(float(state.get("dt", self.config.rollout_dt)), 1e-3)
        current_linear = float(state.get("current_speed", self._last_command[0]))
        current_angular = float(self._last_command[1])
        target_linear = float(candidate.linear)
        target_angular = float(candidate.angular)

        allowed_linear_up = max(float(self.config.actuation_max_linear_accel), _EPS) * dt
        allowed_linear_down = max(float(self.config.actuation_max_linear_decel), _EPS) * dt
        linear_delta = target_linear - current_linear
        if linear_delta >= 0.0:
            projected_linear = current_linear + min(linear_delta, allowed_linear_up)
            linear_norm = allowed_linear_up
        else:
            projected_linear = current_linear + max(linear_delta, -allowed_linear_down)
            linear_norm = allowed_linear_down

        yaw_limit = max(float(self.config.actuation_max_yaw_rate), _EPS)
        bounded_target_angular = float(np.clip(target_angular, -yaw_limit, yaw_limit))
        allowed_angular_delta = max(float(self.config.actuation_max_angular_accel), _EPS) * dt
        angular_delta = bounded_target_angular - current_angular
        projected_angular = current_angular + float(
            np.clip(angular_delta, -allowed_angular_delta, allowed_angular_delta)
        )

        linear_projection_delta = abs(projected_linear - target_linear)
        yaw_projection_delta = abs(bounded_target_angular - target_angular)
        angular_projection_delta = abs(projected_angular - bounded_target_angular)
        linear_clip_risk = _clip01(linear_projection_delta / max(linear_norm, _EPS))
        yaw_clip_risk = _clip01(yaw_projection_delta / max(yaw_limit, _EPS))
        angular_clip_risk = _clip01(angular_projection_delta / max(allowed_angular_delta, _EPS))
        clip_risk = max(linear_clip_risk, yaw_clip_risk, angular_clip_risk)
        feasibility = 1.0 - clip_risk
        diagnostics = {
            "profile": str(self.config.actuation_profile_name),
            "claim_scope": str(self.config.actuation_claim_scope),
            "requested_command": [target_linear, target_angular],
            "projected_command": [float(projected_linear), float(projected_angular)],
            "command_clipped": bool(clip_risk > 0.0),
            "yaw_rate_saturated": bool(yaw_projection_delta > _EPS),
            "linear_accel_required_m_s2": float(linear_delta / dt),
            "angular_accel_required_rad_s2": float((target_angular - current_angular) / dt),
            "clip_risk": float(clip_risk),
            "linear_clip_risk": float(linear_clip_risk),
            "yaw_rate_clip_risk": float(yaw_clip_risk),
            "angular_accel_clip_risk": float(angular_clip_risk),
            "interpretation": "synthetic_diagnostic_only",
        }
        return {
            "actuation_feasibility": float(feasibility),
            "actuation_clip_risk": float(clip_risk),
        }, diagnostics

    def _static_safety_gate(
        self,
        *,
        candidate: HybridRuleCandidate,
        progress: float,
        progress_metric: str,
        initial_static_clearance: float,
        min_static_clearance: float,
    ) -> dict[str, Any] | None:
        """Return optional static-safety demotion diagnostics for value-scorer variants."""
        if not bool(self.config.static_safety_gate_enabled):
            return None
        gated_sources = {"route_guide", "corridor_subgoal"}
        source_gated = bool(self.config.static_safety_gate_all_sources) or (
            candidate.source in gated_sources
        )
        if not source_gated:
            return {
                "enabled": True,
                "source_gated": False,
                "tier": "not_applicable",
                "reason": "source_not_gated",
                "penalty": 0.0,
            }

        min_clearance = _finite_or_none(min_static_clearance)
        initial_clearance = _finite_or_none(initial_static_clearance)
        if min_clearance is None:
            return {
                "enabled": True,
                "source_gated": True,
                "tier": "clear",
                "reason": "no_static_obstacle_clearance_limit",
                "penalty": 0.0,
            }

        low_clearance = min_clearance < float(self.config.static_safety_gate_min_clearance)
        if not low_clearance:
            return {
                "enabled": True,
                "source_gated": True,
                "tier": "clear",
                "reason": "above_min_clearance",
                "min_static_clearance": min_clearance,
                "penalty": 0.0,
            }

        positive_progress = progress > float(self.config.static_safety_gate_progress_threshold)
        non_worsening_clearance = (
            initial_clearance is None
            or min_clearance + float(self.config.static_safety_gate_clearance_tolerance)
            >= initial_clearance
        )
        if positive_progress and non_worsening_clearance:
            return {
                "enabled": True,
                "source_gated": True,
                "tier": "guarded_progress",
                "reason": "low_clearance_progress_nonworsening",
                "min_static_clearance": min_clearance,
                "initial_static_clearance": initial_clearance,
                "progress": float(progress),
                "progress_metric": progress_metric,
                "penalty": 0.0,
            }
        return {
            "enabled": True,
            "source_gated": True,
            "tier": "low_clearance_demoted",
            "reason": "low_clearance_without_safe_progress",
            "min_static_clearance": min_clearance,
            "initial_static_clearance": initial_clearance,
            "progress": float(progress),
            "progress_metric": progress_metric,
            "penalty": float(self.config.static_safety_gate_penalty),
        }

    def _static_safety_gate_progress(
        self,
        *,
        route_corridor: dict[str, Any] | None,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        fallback_goal_progress: float,
    ) -> tuple[float, str]:
        """Return route-local static-gate progress when route geometry is available."""
        tangent_heading = self._route_tangent_heading(route_corridor)
        if tangent_heading is None:
            return float(fallback_goal_progress), "goal_distance"
        route_dir = np.array([np.cos(tangent_heading), np.sin(tangent_heading)], dtype=float)
        return float(np.dot(end_pos - start_pos, route_dir)), "route_local"

    def _route_arc_progress_term(
        self,
        *,
        route_corridor: dict[str, Any] | None,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        max_progress: float,
    ) -> float:
        """Return normalized route-tangent progress for any accepted candidate."""
        tangent_heading = self._route_tangent_heading(route_corridor)
        if tangent_heading is None:
            return 0.0
        route_dir = np.array([np.cos(tangent_heading), np.sin(tangent_heading)], dtype=float)
        route_progress = float(np.dot(end_pos - start_pos, route_dir))
        return float(np.clip(route_progress / max(max_progress, _EPS), -1.0, 1.0))

    def _evaluate_candidate(  # noqa: C901, PLR0915
        self,
        *,
        candidate: HybridRuleCandidate,
        observation: dict[str, Any],
        state: dict[str, Any],
        speed_cap: float,
        nearest_ped: float,
        progress_windows: dict[str, float] | None = None,
        route_corridor: dict[str, Any] | None = None,
        strict_static_clearance: bool = False,
    ) -> dict[str, Any]:
        """Filter and score one candidate command.

        Returns:
            dict[str, Any]: Rejection reason or accepted score/term diagnostics.
        """
        if nearest_ped < float(self.config.near_human_angular_limit_distance) and abs(
            candidate.angular
        ) > float(self.config.near_human_max_angular_speed):
            return {
                "accepted": False,
                "reason": "excessive_angular_near_human",
                "candidate": candidate,
            }

        dt, _steps, rollout_commands = self._candidate_rollout_plan(candidate)
        start_pos = np.array(state["robot_pos"], dtype=float)
        robot_pos = np.array(start_pos, dtype=float)
        heading = float(state["heading"])
        goal = state["goal"]
        ped_pos = state["ped_pos"]
        ped_vel = state["ped_vel"]
        collision_radius = (
            float(state["robot_radius"])
            + float(state["ped_radius"])
            + float(self.config.hard_safety_margin)
        )

        start_dist = float(np.linalg.norm(goal - robot_pos))
        static_margin = (
            float(self.config.static_hard_safety_margin)
            if float(self.config.static_hard_safety_margin) >= 0.0
            else float(self.config.hard_safety_margin)
        )
        hard_static_clearance = float(state["robot_radius"]) + static_margin
        corridor_clearance_buffer = (
            max(float(self.config.corridor_subgoal_static_clearance_buffer), 0.0)
            if candidate.source == "corridor_subgoal" or strict_static_clearance
            else 0.0
        )
        required_static_clearance = hard_static_clearance + corridor_clearance_buffer
        use_continuous_static_check = (
            bool(self.config.corridor_subgoal_use_continuous_static_check)
            and self._continuous_static_context is not None
            and (
                candidate.source == "corridor_subgoal"
                or strict_static_clearance
                or bool(self.config.continuous_static_clearance_enabled)
            )
        )
        initial_static_clearance = self._min_obstacle_clearance(robot_pos, observation)
        min_static_clearance = float("inf")
        min_dynamic_clearance = float("inf")
        static_clearance_exception_terms: set[str] = set()
        proxemic_enabled = bool(self.config.proxemic_costmap_enabled)
        proxemic_costmap_config = self._proxemic_costmap_config if proxemic_enabled else None
        rollout_points = [np.array(robot_pos, dtype=float)] if proxemic_enabled else None

        for step_idx, (step_linear, step_angular) in enumerate(rollout_commands):
            t = (step_idx + 1) * dt
            robot_pos = (
                robot_pos
                + np.array(
                    [step_linear * np.cos(heading), step_linear * np.sin(heading)],
                    dtype=float,
                )
                * dt
            )
            heading = _wrap_angle(heading + step_angular * dt)
            if proxemic_enabled and rollout_points is not None:
                rollout_points.append(np.array(robot_pos, dtype=float))

            static_rejection = self._static_collision_rejection(
                candidate=candidate,
                observation=observation,
                robot_pos=robot_pos,
                hard_static_clearance=hard_static_clearance,
                use_continuous_static_check=use_continuous_static_check,
                t=t,
            )
            if static_rejection is not None:
                return static_rejection
            min_static_clearance = min(
                min_static_clearance,
                self._min_obstacle_clearance(robot_pos, observation),
            )
            if min_static_clearance <= required_static_clearance:
                if use_continuous_static_check:
                    continue
                static_violation_policy = (
                    None
                    if strict_static_clearance or corridor_clearance_buffer > 0.0
                    else self._static_clearance_violation_policy(
                        candidate=candidate,
                        initial_clearance=initial_static_clearance,
                        current_min_clearance=min_static_clearance,
                        hard_static_clearance=hard_static_clearance,
                        step_progress=start_dist - float(np.linalg.norm(goal - robot_pos)),
                        progress_windows=progress_windows,
                    )
                )
                if static_violation_policy is None:
                    return {
                        "accepted": False,
                        "reason": "static_clearance",
                        "candidate": candidate,
                        "min_static_clearance": float(min_static_clearance),
                        "hard_static_clearance": float(hard_static_clearance),
                        "required_static_clearance": float(required_static_clearance),
                        "time": float(t),
                    }
                static_clearance_exception_terms.add(static_violation_policy)

            if (
                "escape" in static_clearance_exception_terms
                and initial_static_clearance <= hard_static_clearance
                and min_static_clearance + float(self.config.static_clearance_escape_tolerance)
                < initial_static_clearance
            ):
                return {
                    "accepted": False,
                    "reason": "static_clearance",
                    "candidate": candidate,
                    "min_static_clearance": float(min_static_clearance),
                    "hard_static_clearance": float(hard_static_clearance),
                    "time": float(t),
                }

            if ped_pos.size > 0:
                ped_future = ped_pos + ped_vel * t
                distances = np.linalg.norm(ped_future - robot_pos[None, :], axis=1)
                min_dynamic_clearance = min(min_dynamic_clearance, float(np.min(distances)))
                if (
                    t <= float(self.config.hard_collision_horizon)
                    and min_dynamic_clearance <= collision_radius
                ):
                    return {
                        "accepted": False,
                        "reason": "dynamic_collision",
                        "candidate": candidate,
                        "min_dynamic_clearance": float(min_dynamic_clearance),
                        "collision_radius": float(collision_radius),
                        "time": float(t),
                    }

        end_dist = float(np.linalg.norm(goal - robot_pos))
        progress = start_dist - end_dist
        static_gate_progress, static_gate_progress_metric = self._static_safety_gate_progress(
            route_corridor=route_corridor,
            start_pos=start_pos,
            end_pos=robot_pos,
            fallback_goal_progress=progress,
        )
        max_progress = max(
            float(self.config.max_linear_speed) * float(self.config.rollout_horizon), _EPS
        )
        goal_vec = goal - robot_pos
        goal_heading = (
            heading
            if np.linalg.norm(goal_vec) <= _EPS
            else float(np.arctan2(goal_vec[1], goal_vec[0]))
        )
        heading_error = abs(_wrap_angle(goal_heading - heading))
        ttc = self._ttc_proxy(
            robot_pos=state["robot_pos"],
            heading=float(state["heading"]),
            command=candidate,
            ped_pos=ped_pos,
            ped_vel=ped_vel,
        )

        dynamic_margin = min_dynamic_clearance - collision_radius
        desired_dynamic_margin = max(
            float(self.config.desired_dynamic_clearance) - collision_radius, _EPS
        )
        static_clearance = (
            1.0
            if np.isinf(min_static_clearance)
            else _clip01(
                min_static_clearance / max(float(self.config.desired_static_clearance), _EPS)
            )
        )
        rollout_mean_linear, rollout_max_linear = self._rollout_linear_stats(rollout_commands)
        dynamic_clearance = (
            1.0
            if np.isinf(min_dynamic_clearance)
            else _clip01(dynamic_margin / desired_dynamic_margin)
        )
        velocity_delta = abs(candidate.linear - float(state["current_speed"]))
        angular_delta = abs(candidate.angular - float(self._last_command[1]))
        progress_windows = progress_windows or {}
        stalled_progress = float(progress_windows.get("3s", 0.0)) <= float(
            self.config.deadlock_progress_threshold
        )
        deadlock_escape = (
            1.0
            if bool(self.config.recovery_enabled)
            and stalled_progress
            and start_dist > float(self.config.goal_far_distance)
            and nearest_ped >= float(self.config.slow_distance_human)
            and candidate.linear <= float(self.config.freezing_speed_threshold)
            and abs(candidate.angular) >= float(self.config.deadlock_rotation_threshold)
            else 0.0
        )
        static_recenter = self._static_recenter_term(
            candidate=candidate,
            observation=observation,
            state=state,
            hard_static_clearance=hard_static_clearance,
            stalled_progress=stalled_progress,
            start_dist=start_dist,
            nearest_ped=nearest_ped,
        )
        route_guide_commitment = self._route_guide_commitment_term(
            candidate=candidate,
            progress_windows=progress_windows,
            start_dist=start_dist,
        )
        corridor_subgoal_terms = self._corridor_subgoal_score_terms(
            candidate=candidate,
            route_corridor=route_corridor,
            state=state,
            end_pos=robot_pos,
            end_heading=heading,
            min_static_clearance=min_static_clearance,
            hard_static_clearance=hard_static_clearance,
        )
        actuation_terms, actuation_diagnostics = self._actuation_terms(
            candidate=candidate,
            state=state,
        )
        if proxemic_costmap_config is not None and rollout_points is not None:
            proxemic_cost_values = proxemic_cost_at_points(
                np.asarray(rollout_points, dtype=float),
                ped_pos,
                ped_vel,
                proxemic_costmap_config,
            )
            proxemic_cost_mean = float(np.mean(proxemic_cost_values))
            proxemic_cost_max = float(np.max(proxemic_cost_values))
            proxemic_cost_norm = _clip01(
                proxemic_cost_mean / max(float(proxemic_costmap_config.max_cost), _EPS)
            )
        else:
            proxemic_cost_mean = 0.0
            proxemic_cost_max = 0.0
            proxemic_cost_norm = 0.0
        static_safety_gate = self._static_safety_gate(
            candidate=candidate,
            progress=static_gate_progress,
            progress_metric=static_gate_progress_metric,
            initial_static_clearance=initial_static_clearance,
            min_static_clearance=min_static_clearance,
        )
        static_safety_gate_penalty = (
            float(static_safety_gate.get("penalty", 0.0))
            if isinstance(static_safety_gate, dict)
            else 0.0
        )
        terms = {
            "goal_progress": float(np.clip(progress / max_progress, -1.0, 1.0)),
            "route_arc_progress": self._route_arc_progress_term(
                route_corridor=route_corridor,
                start_pos=start_pos,
                end_pos=robot_pos,
                max_progress=max_progress,
            ),
            "path_alignment": float(np.cos(heading_error)),
            "speed_preference": _clip01(rollout_mean_linear / max(speed_cap, _EPS)),
            "static_clearance": static_clearance,
            "dynamic_clearance": dynamic_clearance,
            "time_to_collision_margin": 1.0
            if np.isinf(ttc)
            else _clip01(ttc / max(float(self.config.soft_prediction_horizon), _EPS)),
            "heading_smoothness": 1.0
            - _clip01(angular_delta / max(float(self.config.max_angular_speed), _EPS)),
            "velocity_smoothness": 1.0
            - _clip01(velocity_delta / max(float(self.config.max_linear_speed), _EPS)),
            "control_effort": 1.0
            - 0.5
            * (
                _clip01(candidate.linear / max(float(self.config.max_linear_speed), _EPS))
                + _clip01(abs(candidate.angular) / max(float(self.config.max_angular_speed), _EPS))
            ),
            "freezing_penalty": 1.0
            if rollout_max_linear <= float(self.config.freezing_speed_threshold)
            and start_dist > float(self.config.goal_far_distance)
            else 0.0,
            "oscillation_penalty": self._oscillation_penalty(candidate.angular),
            "deadlock_escape": deadlock_escape,
            "static_recenter": static_recenter,
            "static_clearance_escape": 1.0 if "escape" in static_clearance_exception_terms else 0.0,
            "static_corridor_transit": 1.0
            if "corridor_transit" in static_clearance_exception_terms
            else 0.0,
            "route_guide_commitment": route_guide_commitment,
            "static_safety_gate_penalty": static_safety_gate_penalty,
            "proxemic_cost": proxemic_cost_norm,
            **corridor_subgoal_terms,
            **actuation_terms,
        }
        raw_value_score = (
            float(self.config.goal_progress_weight) * terms["goal_progress"]
            + float(self.config.path_alignment_weight) * terms["path_alignment"]
            + float(self.config.speed_preference_weight) * terms["speed_preference"]
            + float(self.config.static_clearance_weight) * terms["static_clearance"]
            + float(self.config.dynamic_clearance_weight) * terms["dynamic_clearance"]
            + float(self.config.ttc_weight) * terms["time_to_collision_margin"]
            + float(self.config.heading_smoothness_weight) * terms["heading_smoothness"]
            + float(self.config.velocity_smoothness_weight) * terms["velocity_smoothness"]
            + float(self.config.control_effort_weight) * terms["control_effort"]
            + float(self.config.deadlock_escape_weight) * terms["deadlock_escape"]
            + float(self.config.route_arc_progress_weight) * terms["route_arc_progress"]
            + float(self.config.static_recenter_weight) * terms["static_recenter"]
            + float(self.config.static_corridor_transit_weight) * terms["static_corridor_transit"]
            + float(self.config.route_guide_commitment_weight) * terms["route_guide_commitment"]
            + float(self.config.corridor_subgoal_route_progress_weight)
            * terms["corridor_subgoal_route_progress"]
            + float(self.config.corridor_subgoal_centering_weight)
            * terms["corridor_subgoal_centering"]
            + float(self.config.corridor_subgoal_tangent_alignment_weight)
            * terms["corridor_subgoal_tangent_alignment"]
            + float(self.config.corridor_subgoal_clearance_weight)
            * terms["corridor_subgoal_clearance_margin"]
            + float(self.config.corridor_subgoal_continuity_weight)
            * terms["corridor_subgoal_continuity"]
            - float(self.config.actuation_clip_risk_weight) * terms["actuation_clip_risk"]
            - float(self.config.proxemic_costmap_weight) * terms["proxemic_cost"]
            - float(self.config.freezing_weight) * terms["freezing_penalty"]
            - float(self.config.oscillation_weight) * terms["oscillation_penalty"]
        )
        score = raw_value_score - static_safety_gate_penalty
        return {
            "accepted": True,
            "candidate": candidate,
            "score": float(score),
            "raw_value_score": float(raw_value_score),
            "terms": terms,
            "min_static_clearance": min_static_clearance,
            "min_dynamic_clearance": min_dynamic_clearance,
            "predicted_ttc": ttc,
            "continuous_static_checked": bool(use_continuous_static_check),
            "proxemic_cost_summary": {
                "enabled": bool(proxemic_costmap_config.enabled)
                if proxemic_costmap_config is not None
                else False,
                "mean": proxemic_cost_mean,
                "max": proxemic_cost_max,
                "normalized_mean": proxemic_cost_norm,
            },
            "actuation_diagnostics": actuation_diagnostics,
            "static_safety_gate": static_safety_gate,
        }

    def _candidate_diagnostic(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        """Return a compact JSON-safe candidate diagnostic row."""
        candidate = evaluation["candidate"]
        row = {
            "command": [float(candidate.linear), float(candidate.angular)],
            "source": candidate.source,
            "score": float(evaluation["score"]),
            "raw_value_score": float(evaluation.get("raw_value_score", evaluation["score"])),
            "terms": {key: float(value) for key, value in evaluation["terms"].items()},
            "min_static_clearance": _finite_or_none(evaluation.get("min_static_clearance")),
            "min_dynamic_clearance": _finite_or_none(evaluation.get("min_dynamic_clearance")),
            "predicted_ttc": _finite_or_none(evaluation.get("predicted_ttc")),
        }
        if candidate.rollout_sequence:
            row["rollout_sequence"] = [
                [float(duration), float(linear), float(angular)]
                for duration, linear, angular in candidate.rollout_sequence
            ]
        if "continuous_static_checked" in evaluation:
            row["continuous_static_checked"] = bool(evaluation.get("continuous_static_checked"))
        if "proxemic_cost_summary" in evaluation:
            row["proxemic_cost_summary"] = evaluation.get("proxemic_cost_summary")
        if isinstance(evaluation.get("actuation_diagnostics"), dict):
            row["actuation_diagnostics"] = dict(evaluation["actuation_diagnostics"])
        if isinstance(evaluation.get("static_safety_gate"), dict):
            row["static_safety_gate"] = dict(evaluation["static_safety_gate"])
        return row

    def _rejection_diagnostic(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        """Return a compact JSON-safe rejected-candidate diagnostic row."""
        candidate = evaluation.get("candidate")
        row: dict[str, Any] = {
            "reason": str(evaluation.get("reason", "unknown")),
        }
        if isinstance(candidate, HybridRuleCandidate):
            row["command"] = [float(candidate.linear), float(candidate.angular)]
            row["source"] = candidate.source
            if candidate.rollout_sequence:
                row["rollout_sequence"] = [
                    [float(duration), float(linear), float(angular)]
                    for duration, linear, angular in candidate.rollout_sequence
                ]
        for key in (
            "min_static_clearance",
            "hard_static_clearance",
            "required_static_clearance",
            "min_dynamic_clearance",
            "collision_radius",
            "obstacle_value",
            "time",
        ):
            value = evaluation.get(key)
            if isinstance(value, int | float | np.integer | np.floating):
                row[key] = float(value)
        if "continuous_static_collision" in evaluation:
            row["continuous_static_collision"] = bool(evaluation.get("continuous_static_collision"))
        return row

    def _evaluate_candidates_for_plan(
        self,
        *,
        candidates: list[HybridRuleCandidate],
        observation: dict[str, Any],
        state: dict[str, Any],
        speed_cap: float,
        nearest_ped: float,
        progress_windows: dict[str, float],
        route_corridor: dict[str, Any] | None = None,
        corridor_subgoal: dict[str, Any] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        Counter[str],
        Counter[str],
        dict[str, dict[str, int]],
        list[dict[str, Any]],
    ]:
        """Evaluate a candidate set and collect rejection diagnostics.

        Returns:
            tuple: Accepted evaluations, aggregate rejections, moving-command
            rejections, source-level rejections, and compact examples.
        """
        accepted: list[dict[str, Any]] = []
        rejection_counts: Counter[str] = Counter()
        moving_rejection_counts: Counter[str] = Counter()
        rejection_counts_by_source: dict[str, Counter[str]] = {}
        rejected_examples: list[dict[str, Any]] = []

        for candidate in candidates:
            evaluation = self._evaluate_candidate(
                candidate=candidate,
                observation=observation,
                state=state,
                speed_cap=speed_cap,
                nearest_ped=nearest_ped,
                progress_windows=progress_windows,
                route_corridor=route_corridor,
                strict_static_clearance=bool(corridor_subgoal and corridor_subgoal.get("active")),
            )
            if bool(evaluation.get("accepted")):
                accepted.append(evaluation)
                continue

            reason = str(evaluation.get("reason", "unknown"))
            rejection_counts[reason] += 1
            source_counts = rejection_counts_by_source.setdefault(candidate.source, Counter())
            source_counts[reason] += 1
            if candidate.linear > float(self.config.freezing_speed_threshold):
                moving_rejection_counts[reason] += 1
            if len(rejected_examples) < max(int(self.config.top_k_diagnostics), 1):
                rejected_examples.append(self._rejection_diagnostic(evaluation))

        return (
            accepted,
            rejection_counts,
            moving_rejection_counts,
            {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(rejection_counts_by_source.items())
            },
            rejected_examples,
        )

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the selected ``(linear, angular)`` command."""
        state = self._extract_state(observation)
        robot_pos = state["robot_pos"]
        goal = state["goal"]
        goal_distance = float(np.linalg.norm(goal - robot_pos))
        current_time = float(self._step_index) * float(state["dt"])
        self._progress_history.append((current_time, goal_distance))
        progress_windows = self._progress_windows(current_time, goal_distance)
        if goal_distance <= float(self.config.goal_tolerance):
            command = (0.0, 0.0)
            self._record_decision(
                mode="GOAL_STOP",
                command=command,
                source="goal_reached",
                score=0.0,
                terms={},
                rejection_counts={},
                top_k=[],
                nearest_ped=float("inf"),
                nearest_static=float("inf"),
                predicted_ttc=float("inf"),
                progress_windows=progress_windows,
                route_corridor=None,
                corridor_subgoal=None,
                route_trace_recovery=None,
            )
            return command

        route_corridor = self._route_corridor_diagnostics(
            state["observation"],
            current_time=current_time,
        )
        nearest_ped = self._nearest_ped_distance(robot_pos, state["ped_pos"])
        speed_cap = self._human_speed_cap(nearest_ped)
        self._clearance_context = self._build_clearance_context(observation)
        corridor_subgoal = self._corridor_subgoal_activation(
            route_corridor=route_corridor,
            progress_windows=progress_windows,
            nearest_ped=nearest_ped,
        )
        route_trace_recovery = self._route_trace_recovery_signal(
            route_corridor=route_corridor,
            progress_windows=progress_windows,
            nearest_ped=nearest_ped,
        )
        corridor_subgoal_for_candidates = self._corridor_subgoal_activation_for_trace_recovery(
            corridor_subgoal,
            route_trace_recovery,
        )
        candidates = self._generate_candidates(
            state,
            speed_cap,
            route_corridor=route_corridor,
            corridor_subgoal=corridor_subgoal_for_candidates,
        )
        (
            accepted,
            rejection_counts,
            moving_rejection_counts,
            rejection_counts_by_source,
            rejected_examples,
        ) = self._evaluate_candidates_for_plan(
            candidates=candidates,
            observation=observation,
            state=state,
            speed_cap=speed_cap,
            nearest_ped=nearest_ped,
            progress_windows=progress_windows,
            route_corridor=route_corridor,
            corridor_subgoal=corridor_subgoal_for_candidates,
        )

        self._rejection_counts.update(rejection_counts)
        if accepted:
            accepted.sort(key=lambda item: float(item["score"]), reverse=True)
            recovery_best = self._select_route_trace_recovery_evaluation(
                accepted,
                route_trace_recovery,
            )
            best = recovery_best if recovery_best is not None else accepted[0]
            candidate = best["candidate"]
            command = (float(candidate.linear), float(candidate.angular))
            mode = "ROUTE_TRACE_RECOVERY" if recovery_best is not None else "NORMAL"
            source = candidate.source
            score = float(best["score"])
            terms = best["terms"]
            nearest_static = best["min_static_clearance"]
            predicted_ttc = best["predicted_ttc"]
            actuation_diagnostics = best.get("actuation_diagnostics")
            static_safety_gate = best.get("static_safety_gate")
            top_k = [
                self._candidate_diagnostic(item)
                for item in accepted[: max(int(self.config.top_k_diagnostics), 1)]
            ]
        else:
            self._fallback_count += 1
            mode = "EMERGENCY_STOP"
            source = "all_candidates_rejected"
            command = (0.0, 0.0)
            if self._static_recovery_allowed(rejection_counts, nearest_ped):
                goal_vec = goal - robot_pos
                goal_heading = float(np.arctan2(goal_vec[1], goal_vec[0]))
                heading_error = _wrap_angle(goal_heading - float(state["heading"]))
                turn_sign = 1.0 if heading_error >= 0.0 else -1.0
                command = (
                    0.0,
                    float(turn_sign * abs(float(self.config.recovery_reorient_angular_speed))),
                )
                mode = "REORIENT"
                source = "static_reorient"
            score = 0.0
            terms = {"freezing_penalty": 1.0}
            nearest_static = self._min_obstacle_clearance(robot_pos, observation)
            actuation_diagnostics = None
            static_safety_gate = None
            predicted_ttc = self._ttc_proxy(
                robot_pos=robot_pos,
                heading=float(state["heading"]),
                command=HybridRuleCandidate(0.0, 0.0, "emergency_stop"),
                ped_pos=state["ped_pos"],
                ped_vel=state["ped_vel"],
            )
            top_k = []

        self._record_decision(
            mode=mode,
            command=command,
            source=source,
            score=score,
            terms=terms,
            rejection_counts=dict(rejection_counts),
            top_k=top_k,
            nearest_ped=nearest_ped,
            nearest_static=nearest_static,
            predicted_ttc=predicted_ttc,
            progress_windows=progress_windows,
            route_corridor=route_corridor,
            corridor_subgoal=corridor_subgoal_for_candidates,
            route_trace_recovery=route_trace_recovery,
            actuation_diagnostics=actuation_diagnostics,
            static_safety_gate=static_safety_gate,
            rejected_examples=rejected_examples,
            moving_rejection_counts=dict(moving_rejection_counts),
            rejection_counts_by_source=rejection_counts_by_source,
        )
        return command

    def _static_recovery_allowed(self, rejection_counts: Counter[str], nearest_ped: float) -> bool:
        """Return whether a rotate-in-place static recovery is safe enough to try.

        Returns:
            bool: True when static-clearance rejection dominates and no pedestrian is too close.
        """
        if not bool(self.config.recovery_enabled):
            return False
        if nearest_ped < float(self.config.slow_distance_human):
            return False
        static_rejections = int(rejection_counts.get("static_clearance", 0)) + int(
            rejection_counts.get("static_collision", 0)
        )
        dynamic_rejections = int(rejection_counts.get("dynamic_collision", 0))
        return static_rejections > 0 and static_rejections >= dynamic_rejections

    def _record_decision(  # noqa: PLR0913
        self,
        *,
        mode: str,
        command: tuple[float, float],
        source: str,
        score: float,
        terms: dict[str, Any],
        rejection_counts: dict[str, int],
        top_k: list[dict[str, Any]],
        nearest_ped: float,
        nearest_static: float,
        predicted_ttc: float,
        progress_windows: dict[str, float],
        route_corridor: dict[str, Any] | None = None,
        corridor_subgoal: dict[str, Any] | None = None,
        route_trace_recovery: dict[str, Any] | None = None,
        actuation_diagnostics: dict[str, Any] | None = None,
        static_safety_gate: dict[str, Any] | None = None,
        rejected_examples: list[dict[str, Any]] | None = None,
        moving_rejection_counts: dict[str, int] | None = None,
        rejection_counts_by_source: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Persist selected-command diagnostics and update state."""
        self._last_command = (float(command[0]), float(command[1]))
        self._recent_commands.append(self._last_command)
        self._selected_source_counts[str(source)] += 1
        self._step_index += 1
        unavailable_counts, unavailable_examples = self._unavailable_diagnostics(corridor_subgoal)
        self._unavailable_counts.update(unavailable_counts)
        self._last_decision = {
            "planner_variant": self.config.planner_variant,
            "value_scorer": self._value_scorer_metadata(),
            "proxemic_costmap": self._proxemic_costmap_metadata(),
            "planner_mode": mode,
            "selected_command": [float(command[0]), float(command[1])],
            "selected_source": source,
            "selected_score": float(score),
            "selected_terms": {
                key: float(value)
                for key, value in terms.items()
                if isinstance(value, int | float | np.integer | np.floating)
            },
            "top_k": top_k,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "moving_rejection_counts": dict(sorted((moving_rejection_counts or {}).items())),
            "rejection_counts_by_source": rejection_counts_by_source or {},
            "rejected_examples": rejected_examples or [],
            "unavailable_counts": dict(sorted(unavailable_counts.items())),
            "unavailable_examples": unavailable_examples,
            "nearest_pedestrian_distance": _finite_or_none(nearest_ped),
            "nearest_static_obstacle_distance": _finite_or_none(nearest_static),
            "predicted_ttc": _finite_or_none(predicted_ttc),
            "progress_windows": {key: float(value) for key, value in progress_windows.items()},
            "route_corridor": route_corridor,
            "corridor_subgoal": corridor_subgoal,
            "route_trace_recovery": route_trace_recovery,
            "selected_actuation_diagnostics": actuation_diagnostics,
            "selected_static_safety_gate": static_safety_gate,
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return aggregate planner diagnostics for benchmark episode metadata."""
        return {
            "planner_variant": self.config.planner_variant,
            "value_scorer": self._value_scorer_metadata(),
            "proxemic_costmap": self._proxemic_costmap_metadata(),
            "actuation_scoring": self._actuation_scoring_metadata(),
            "steps": int(self._step_index),
            "selected_source_counts": dict(sorted(self._selected_source_counts.items())),
            "rejection_counts": dict(sorted(self._rejection_counts.items())),
            "unavailable_counts": dict(sorted(self._unavailable_counts.items())),
            "fallback_count": int(self._fallback_count),
            "last_decision": dict(self._last_decision) if self._last_decision else None,
        }

    def last_decision(self) -> dict[str, Any] | None:
        """Return the latest selected-command diagnostics for step-level tooling."""
        return dict(self._last_decision) if self._last_decision else None

    def _value_scorer_metadata(self) -> dict[str, Any] | None:
        """Return clean-room value-scorer metadata for experimental variants."""
        if not self.config.planner_variant.startswith("tentabot_value_scorer_"):
            return None
        return {
            "profile": str(self.config.value_scorer_profile),
            "training_source": str(self.config.value_scorer_training_source),
            "candidate_lattice": "hybrid_rule_local_planner",
            "static_safety_gate_enabled": bool(self.config.static_safety_gate_enabled),
            "route_trace_recovery_enabled": bool(self.config.route_trace_recovery_enabled),
            "source_parity_claim": False,
            "upstream_code_used": False,
            "observation_scope": (
                "route_progress, static_clearance, pedestrian_distance_ttc, "
                "route_arc_progress, smoothness, and command_bounds"
            ),
        }

    def _unavailable_diagnostics(
        self, corridor_subgoal: dict[str, Any] | None
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Return candidate-source unavailable diagnostics for optional sources."""
        counts: dict[str, int] = {}
        examples: list[dict[str, Any]] = []
        if isinstance(corridor_subgoal, dict) and not bool(corridor_subgoal.get("active")):
            reason = str(corridor_subgoal.get("reason", "unavailable"))
            counts["corridor_subgoal"] = 1
            examples.append({"source": "corridor_subgoal", "reason": reason})
        return counts, examples


def _expand_nested_proxemic_costmap_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten optional nested proxemic-layer config into planner fields.

    Returns:
        dict[str, Any]: Planner config mapping with proxemic fields expanded.
    """
    proxemic_layer = raw.pop("proxemic_costmap", None)
    if proxemic_layer is None:
        return raw
    if not isinstance(proxemic_layer, dict):
        raise ValueError("proxemic_costmap must be a mapping when provided")
    proxemic_config = build_proxemic_costmap_config(proxemic_layer)
    raw.update(
        {
            "proxemic_costmap_enabled": proxemic_config.enabled,
            "proxemic_costmap_personal_radius": proxemic_config.personal_radius,
            "proxemic_costmap_social_radius": proxemic_config.social_radius,
            "proxemic_costmap_personal_weight": proxemic_config.personal_weight,
            "proxemic_costmap_social_weight": proxemic_config.social_weight,
            "proxemic_costmap_velocity_elongation_factor": (
                proxemic_config.velocity_elongation_factor
            ),
            "proxemic_costmap_max_cost": proxemic_config.max_cost,
            "proxemic_costmap_decay_function": proxemic_config.decay_function,
        }
    )
    return raw


def build_hybrid_rule_local_planner_config(
    cfg: dict[str, Any] | None,
) -> HybridRuleLocalPlannerConfig:
    """Build a typed config from a YAML mapping.

    Returns:
        HybridRuleLocalPlannerConfig: Parsed planner config.
    """
    if not isinstance(cfg, dict):
        return HybridRuleLocalPlannerConfig()

    raw = _expand_nested_proxemic_costmap_config(copy.deepcopy(cfg))
    defaults = HybridRuleLocalPlannerConfig()
    field_map = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in field_map:
            continue
        if key == "lookahead_distances":
            if isinstance(value, list | tuple):
                kwargs[key] = tuple(float(item) for item in value)
            continue
        default = field_map[key]
        if isinstance(default, bool):
            kwargs[key] = _coerce_config_bool(key, value)
        elif isinstance(default, int):
            kwargs[key] = int(value)
        elif isinstance(default, float):
            kwargs[key] = float(value)
        elif isinstance(default, str):
            kwargs[key] = str(value)
        else:
            kwargs[key] = value
    return HybridRuleLocalPlannerConfig(**kwargs)


def _coerce_config_bool(key: str, value: Any) -> bool:
    """Parse booleans from YAML/CLI-style config values without Python truthiness traps.

    Returns:
        bool: Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"Expected boolean-compatible value for hybrid rule config '{key}'.")


__all__ = [
    "HybridRuleCandidate",
    "HybridRuleLocalPlannerAdapter",
    "HybridRuleLocalPlannerConfig",
    "build_hybrid_rule_local_planner_config",
]
