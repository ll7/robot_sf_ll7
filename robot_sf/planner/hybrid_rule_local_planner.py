"""Deterministic hybrid-rule local planner family.

This module starts the ``hybrid_rule_local_planner`` family with the v0 minimal
variant: a dynamic-window style local planner with explicit safety filtering and
diagnostic score decomposition.  It intentionally avoids training, learned
weights, and benchmark-result fitting.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from robot_sf.planner.grid_route import GridRoutePlannerAdapter, GridRoutePlannerConfig
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin

_SUPPORTED_VARIANTS = {
    "hybrid_rule_v0_minimal",
    "hybrid_rule_v1_dwa_social",
    "hybrid_rule_v2_orca_guided",
    "hybrid_rule_v3_teb_like_rollout",
    "hybrid_rule_v4_recovery_aware",
    "hybrid_rule_v5_ensemble_selector",
}
_EPS = 1e-9


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]`` radians.

    Returns:
        float: Wrapped angle in radians.
    """
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


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
    """Candidate constant unicycle command for short-horizon rollout."""

    linear: float
    angular: float
    source: str


@dataclass(frozen=True)
class _ObstacleClearanceContext:
    """Per-plan static-obstacle clearance lookup."""

    observation_id: int
    grid: np.ndarray
    meta: dict[str, Any]
    channel: int
    resolution: float
    clearance_cells: np.ndarray | None


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
    route_guide_commitment_weight: float = 0.0
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

    # Ablation-only route-commitment bonus. It remains configurable for
    # diagnostics, but benchmark evidence rejected it due static collisions.
    route_guide_commitment_progress_threshold: float = 0.5


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
        self.reset()

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
        self._selected_source_counts: Counter[str] = Counter()
        self._rejection_counts: Counter[str] = Counter()
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

    def _candidate_key(self, candidate: HybridRuleCandidate) -> tuple[float, float]:
        """Quantize a candidate command for duplicate removal.

        Returns:
            tuple[float, float]: Rounded linear/angular command key.
        """
        return round(float(candidate.linear), 4), round(float(candidate.angular), 4)

    def _generate_candidates(
        self, state: dict[str, Any], speed_cap: float
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

        unique: dict[tuple[float, float], HybridRuleCandidate] = {}
        for candidate in candidates:
            clipped = HybridRuleCandidate(
                float(np.clip(candidate.linear, 0.0, speed_cap)),
                float(
                    np.clip(
                        candidate.angular,
                        -float(self.config.max_angular_speed),
                        float(self.config.max_angular_speed),
                    )
                ),
                candidate.source,
            )
            unique.setdefault(self._candidate_key(clipped), clipped)
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
        windows: dict[str, float] = {}
        for seconds in (1.0, 3.0, 5.0):
            reference = None
            for time_value, distance_value in self._progress_history:
                if time_value >= current_time - seconds:
                    reference = distance_value
                    break
            if reference is None and self._progress_history:
                reference = self._progress_history[0][1]
            windows[f"{seconds:.0f}s"] = (
                0.0 if reference is None else float(reference - goal_distance)
            )
        return windows

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

    def _evaluate_candidate(
        self,
        *,
        candidate: HybridRuleCandidate,
        observation: dict[str, Any],
        state: dict[str, Any],
        speed_cap: float,
        nearest_ped: float,
        progress_windows: dict[str, float] | None = None,
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

        dt = max(float(self.config.rollout_dt), 1e-3)
        steps = max(int(np.ceil(float(self.config.rollout_horizon) / dt)), 1)
        robot_pos = np.array(state["robot_pos"], dtype=float)
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
        initial_static_clearance = self._min_obstacle_clearance(robot_pos, observation)
        min_static_clearance = float("inf")
        min_dynamic_clearance = float("inf")
        static_clearance_escape_used = False

        for step_idx in range(steps):
            t = (step_idx + 1) * dt
            robot_pos = (
                robot_pos
                + np.array(
                    [candidate.linear * np.cos(heading), candidate.linear * np.sin(heading)],
                    dtype=float,
                )
                * dt
            )
            heading = _wrap_angle(heading + candidate.angular * dt)

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
            min_static_clearance = min(
                min_static_clearance,
                self._min_obstacle_clearance(robot_pos, observation),
            )
            if min_static_clearance <= hard_static_clearance:
                if self._static_clearance_escape_allowed(
                    candidate=candidate,
                    initial_clearance=initial_static_clearance,
                    current_min_clearance=min_static_clearance,
                    hard_static_clearance=hard_static_clearance,
                ):
                    static_clearance_escape_used = True
                else:
                    return {
                        "accepted": False,
                        "reason": "static_clearance",
                        "candidate": candidate,
                        "min_static_clearance": float(min_static_clearance),
                        "hard_static_clearance": float(hard_static_clearance),
                        "time": float(t),
                    }

            if (
                static_clearance_escape_used
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
        static_recenter = (
            self._static_recenter_probe_score(
                candidate=candidate,
                observation=observation,
                state=state,
                hard_static_clearance=hard_static_clearance,
            )
            if stalled_progress
            and start_dist > float(self.config.goal_far_distance)
            and nearest_ped >= float(self.config.slow_distance_human)
            else 0.0
        )
        route_guide_commitment = (
            1.0
            if candidate.source == "route_guide"
            and float(progress_windows.get("3s", 0.0))
            <= float(self.config.route_guide_commitment_progress_threshold)
            and start_dist > float(self.config.goal_far_distance)
            else 0.0
        )
        terms = {
            "goal_progress": float(np.clip(progress / max_progress, -1.0, 1.0)),
            "path_alignment": float(np.cos(heading_error)),
            "speed_preference": _clip01(candidate.linear / max(speed_cap, _EPS)),
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
            if candidate.linear <= float(self.config.freezing_speed_threshold)
            and start_dist > float(self.config.goal_far_distance)
            else 0.0,
            "oscillation_penalty": self._oscillation_penalty(candidate.angular),
            "deadlock_escape": deadlock_escape,
            "static_recenter": static_recenter,
            "static_clearance_escape": 1.0 if static_clearance_escape_used else 0.0,
            "route_guide_commitment": route_guide_commitment,
        }
        score = (
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
            + float(self.config.static_recenter_weight) * terms["static_recenter"]
            + float(self.config.route_guide_commitment_weight) * terms["route_guide_commitment"]
            - float(self.config.freezing_weight) * terms["freezing_penalty"]
            - float(self.config.oscillation_weight) * terms["oscillation_penalty"]
        )
        return {
            "accepted": True,
            "candidate": candidate,
            "score": float(score),
            "terms": terms,
            "min_static_clearance": min_static_clearance,
            "min_dynamic_clearance": min_dynamic_clearance,
            "predicted_ttc": ttc,
        }

    def _candidate_diagnostic(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        """Return a compact JSON-safe candidate diagnostic row."""
        candidate = evaluation["candidate"]
        return {
            "command": [float(candidate.linear), float(candidate.angular)],
            "source": candidate.source,
            "score": float(evaluation["score"]),
            "terms": {key: float(value) for key, value in evaluation["terms"].items()},
            "min_static_clearance": _finite_or_none(evaluation.get("min_static_clearance")),
            "min_dynamic_clearance": _finite_or_none(evaluation.get("min_dynamic_clearance")),
            "predicted_ttc": _finite_or_none(evaluation.get("predicted_ttc")),
        }

    def _rejection_diagnostic(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        """Return a compact JSON-safe rejected-candidate diagnostic row."""
        candidate = evaluation.get("candidate")
        row: dict[str, Any] = {
            "reason": str(evaluation.get("reason", "unknown")),
        }
        if isinstance(candidate, HybridRuleCandidate):
            row["command"] = [float(candidate.linear), float(candidate.angular)]
            row["source"] = candidate.source
        for key in (
            "min_static_clearance",
            "hard_static_clearance",
            "min_dynamic_clearance",
            "collision_radius",
            "obstacle_value",
            "time",
        ):
            value = evaluation.get(key)
            if isinstance(value, int | float | np.integer | np.floating):
                row[key] = float(value)
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
            )
            return command

        nearest_ped = self._nearest_ped_distance(robot_pos, state["ped_pos"])
        speed_cap = self._human_speed_cap(nearest_ped)
        self._clearance_context = self._build_clearance_context(observation)
        candidates = self._generate_candidates(state, speed_cap)
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
        )

        self._rejection_counts.update(rejection_counts)
        if accepted:
            accepted.sort(key=lambda item: float(item["score"]), reverse=True)
            best = accepted[0]
            candidate = best["candidate"]
            command = (float(candidate.linear), float(candidate.angular))
            mode = "NORMAL"
            source = candidate.source
            score = float(best["score"])
            terms = best["terms"]
            nearest_static = best["min_static_clearance"]
            predicted_ttc = best["predicted_ttc"]
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
        rejected_examples: list[dict[str, Any]] | None = None,
        moving_rejection_counts: dict[str, int] | None = None,
        rejection_counts_by_source: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Persist selected-command diagnostics and update state."""
        self._last_command = (float(command[0]), float(command[1]))
        self._recent_commands.append(self._last_command)
        self._selected_source_counts[str(source)] += 1
        self._step_index += 1
        self._last_decision = {
            "planner_variant": self.config.planner_variant,
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
            "nearest_pedestrian_distance": _finite_or_none(nearest_ped),
            "nearest_static_obstacle_distance": _finite_or_none(nearest_static),
            "predicted_ttc": _finite_or_none(predicted_ttc),
            "progress_windows": {key: float(value) for key, value in progress_windows.items()},
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return aggregate planner diagnostics for benchmark episode metadata."""
        return {
            "planner_variant": self.config.planner_variant,
            "steps": int(self._step_index),
            "selected_source_counts": dict(sorted(self._selected_source_counts.items())),
            "rejection_counts": dict(sorted(self._rejection_counts.items())),
            "fallback_count": int(self._fallback_count),
            "last_decision": dict(self._last_decision) if self._last_decision else None,
        }

    def last_decision(self) -> dict[str, Any] | None:
        """Return the latest selected-command diagnostics for step-level tooling."""
        return dict(self._last_decision) if self._last_decision else None


def build_hybrid_rule_local_planner_config(
    cfg: dict[str, Any] | None,
) -> HybridRuleLocalPlannerConfig:
    """Build a typed config from a YAML mapping.

    Returns:
        HybridRuleLocalPlannerConfig: Parsed planner config.
    """
    if not isinstance(cfg, dict):
        return HybridRuleLocalPlannerConfig()

    raw = dict(cfg)
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
