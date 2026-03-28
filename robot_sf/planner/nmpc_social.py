"""Testing-only native NMPC-style social local planner."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.optimize import Bounds, minimize

from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


@dataclass
class NMPCSocialConfig:
    """Configuration for the native NMPC-style social local planner."""

    max_linear_speed: float = 0.9
    max_angular_speed: float = 1.1
    horizon_steps: int = 6
    rollout_dt: float = 0.25
    goal_tolerance: float = 0.25

    path_goal_weight: float = 1.8
    terminal_goal_weight: float = 4.5
    progress_reward_weight: float = 2.0
    heading_weight: float = 0.35
    control_effort_weight: float = 0.05
    smoothness_weight: float = 0.12

    pedestrian_clearance_weight: float = 3.8
    obstacle_clearance_weight: float = 3.2
    occupancy_cost_weight: float = 1.2
    collision_cost_kappa: float = 10.0
    pedestrian_margin: float = 0.55
    obstacle_margin: float = 0.45

    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12
    avoidance_turn_bias_weight: float = 0.25
    symmetry_break_bias: float = 0.2

    solver_ftol: float = 1e-3
    solver_max_iterations: int = 32
    warm_start: bool = True
    fallback_to_stop: bool = True


@dataclass
class _RolloutContext:
    """Static rollout context for one NMPC optimization call."""

    robot_pos: np.ndarray
    heading: float
    current_speed: float
    goal: np.ndarray
    ped_positions: np.ndarray
    ped_velocities: np.ndarray
    robot_radius: float
    ped_radius: float
    observation: dict[str, Any]


class NMPCSocialPlannerAdapter(OccupancyAwarePlannerMixin):
    """Short-horizon deterministic optimizer over unicycle control sequences."""

    _EPS = 1e-6

    def __init__(self, config: NMPCSocialConfig | None = None) -> None:
        """Initialize the adapter with optional config overrides."""
        self.config = config or NMPCSocialConfig()
        self.reset()

    def reset(self) -> None:
        """Clear any per-episode optimizer warm-start state."""
        self._last_solution: np.ndarray | None = None

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Extract and sanitize robot/goal/pedestrian state from an observation.

        Returns:
            tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, float, float]:
            Robot position, heading, speed, active goal, pedestrian positions, pedestrian
            velocities, robot radius, and pedestrian radius.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        speed = float(self._as_1d_float(robot_state.get("speed", [0.0]), pad=1)[0])
        robot_radius = float(self._as_1d_float(robot_state.get("radius", [0.25]), pad=1)[0])
        goal_next = self._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        goal = goal_next if np.linalg.norm(goal_next - robot_pos) > self._EPS else goal_current

        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_positions.ndim == 1 and ped_positions.size % 2 == 0:
            ped_positions = ped_positions.reshape(-1, 2)
        if ped_velocities.ndim == 1 and ped_velocities.size % 2 == 0:
            ped_velocities = ped_velocities.reshape(-1, 2)
        if ped_positions.ndim != 2 or ped_positions.shape[-1] != 2:
            ped_positions = np.zeros((0, 2), dtype=float)
        if ped_velocities.ndim != 2 or ped_velocities.shape[-1] != 2:
            ped_velocities = np.zeros((0, 2), dtype=float)
        count = int(self._as_1d_float(ped_state.get("count", [ped_positions.shape[0]]), pad=1)[0])
        count = max(0, min(count, ped_positions.shape[0]))
        ped_positions = ped_positions[:count]
        if ped_velocities.shape[0] != ped_positions.shape[0]:
            ped_velocities = np.zeros_like(ped_positions)
        else:
            ped_velocities = ped_velocities[:count]
        ped_radius = float(
            self._as_1d_float(ped_state.get("radius", [0.25]), pad=1, default=0.25)[0]
        )
        return (
            robot_pos,
            heading,
            speed,
            goal,
            ped_positions,
            ped_velocities,
            robot_radius,
            ped_radius,
        )

    def _initial_guess(
        self,
        *,
        goal_heading_error: float,
        current_speed: float,
        goal_distance: float,
        preferred_turn: float,
    ) -> np.ndarray:
        """Build the optimizer initial guess, optionally warm-started from prior solve.

        Returns:
            np.ndarray: Flattened `(v, w)` control sequence seed for the solver.
        """
        horizon = max(int(self.config.horizon_steps), 1)
        if (
            bool(self.config.warm_start)
            and self._last_solution is not None
            and self._last_solution.size == horizon * 2
        ):
            shifted = np.empty_like(self._last_solution)
            shifted[:-2] = self._last_solution[2:]
            shifted[-2:] = self._last_solution[-2:]
            return shifted

        desired_speed = min(
            float(self.config.max_linear_speed),
            max(0.15, min(goal_distance / max(horizon * float(self.config.rollout_dt), 1e-3), 1.0)),
        )
        desired_speed = max(
            desired_speed, min(current_speed + 0.1, float(self.config.max_linear_speed))
        )
        desired_turn = np.clip(
            goal_heading_error / max(horizon * float(self.config.rollout_dt), 1e-3),
            -float(self.config.max_angular_speed),
            float(self.config.max_angular_speed),
        )
        desired_turn = float(
            np.clip(
                desired_turn + preferred_turn * float(self.config.symmetry_break_bias),
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        guess = np.zeros(horizon * 2, dtype=float)
        guess[0::2] = desired_speed
        guess[1::2] = desired_turn
        return guess

    def _predict_pedestrians(
        self, ped_positions: np.ndarray, ped_velocities: np.ndarray, step_idx: int
    ) -> np.ndarray:
        """Predict pedestrian positions under a constant-velocity assumption.

        Returns:
            np.ndarray: Predicted pedestrian positions for the requested rollout step.
        """
        if ped_positions.size == 0:
            return ped_positions
        t = float(step_idx + 1) * float(self.config.rollout_dt)
        return ped_positions + ped_velocities * t

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, Any]) -> float:
        """Return the nearest obstacle clearance around a point in meters."""
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return float("inf")
        grid, meta = payload
        channel = self._preferred_channel(meta)
        if channel < 0 or channel >= grid.shape[0]:
            return float("inf")

        rc = self._world_to_grid(point, meta, grid_shape=(grid.shape[1], grid.shape[2]))
        if rc is None:
            return 0.0
        row, col = rc
        channel_grid = np.asarray(grid[channel], dtype=float)
        threshold = float(self.config.obstacle_threshold)
        if channel_grid[row, col] >= threshold:
            return 0.0

        radius = max(int(self.config.obstacle_search_cells), 1)
        r0 = max(0, row - radius)
        r1 = min(channel_grid.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(channel_grid.shape[1], col + radius + 1)
        window = channel_grid[r0:r1, c0:c1]
        obs_idx = np.argwhere(window >= threshold)
        if obs_idx.size == 0:
            return float("inf")

        dr = obs_idx[:, 0] + r0 - row
        dc = obs_idx[:, 1] + c0 - col
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        return float(np.min(np.sqrt(dr.astype(float) ** 2 + dc.astype(float) ** 2)) * resolution)

    def _occupancy_cost(self, point: np.ndarray, observation: dict[str, Any]) -> float:
        """Return raw occupancy at a point for soft obstacle shaping."""
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return 0.0
        grid, meta = payload
        channel = self._preferred_channel(meta)
        if channel < 0:
            return 0.0
        return self._grid_value(point, grid, meta, channel)

    def _soft_collision_cost(self, clearance: float, margin: float) -> float:
        """Convert a clearance margin into a bounded soft collision penalty.

        Returns:
            float: Logistic penalty in `[0, 1]` for the supplied clearance.
        """
        if not np.isfinite(clearance):
            return 0.0
        return float(
            1.0 / (1.0 + np.exp(float(self.config.collision_cost_kappa) * (clearance - margin)))
        )

    def _preferred_avoidance_turn(
        self,
        *,
        robot_pos: np.ndarray,
        heading: float,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
    ) -> float:
        """Estimate the preferred turn direction for near-field pedestrian avoidance.

        Returns:
            float: `-1.0`, `0.0`, or `1.0` indicating right, neutral, or left preference.
        """
        if ped_positions.size == 0:
            return 0.0
        forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
        preference = 0.0
        closest_dist = float("inf")
        for ped_pos, ped_vel in zip(ped_positions, ped_velocities, strict=False):
            rel = ped_pos - robot_pos
            along = float(np.dot(rel, forward))
            if along <= 0.0:
                continue
            dist = float(np.linalg.norm(rel))
            if dist > 1.5 or dist >= closest_dist:
                continue
            lateral = float(forward[0] * rel[1] - forward[1] * rel[0])
            rel_motion = ped_vel - forward * max(float(np.linalg.norm(ped_vel)), 0.2)
            lateral_motion = float(forward[0] * rel_motion[1] - forward[1] * rel_motion[0])
            preference = lateral + 0.5 * lateral_motion
            if abs(preference) <= self._EPS:
                preference = float(self.config.symmetry_break_bias)
            closest_dist = dist
        if abs(preference) <= self._EPS:
            return 0.0
        return 1.0 if preference > 0.0 else -1.0

    def _rollout_cost(self, controls_flat: np.ndarray, *, context: _RolloutContext) -> float:
        """Evaluate the objective for a flattened control sequence.

        Returns:
            float: Scalar optimization objective value for the rollout.
        """
        controls = np.asarray(controls_flat, dtype=float).reshape(-1, 2)
        dt = float(self.config.rollout_dt)
        x = np.asarray(context.robot_pos, dtype=float).copy()
        theta = float(context.heading)
        start_dist = float(np.linalg.norm(context.goal - x))
        cumulative_goal_dist = 0.0
        cost = 0.0

        prev_v = float(context.current_speed)
        prev_w = 0.0
        preferred_turn = self._preferred_avoidance_turn(
            robot_pos=context.robot_pos,
            heading=context.heading,
            ped_positions=context.ped_positions,
            ped_velocities=context.ped_velocities,
        )
        for step_idx, (v_raw, w_raw) in enumerate(controls):
            v = float(np.clip(v_raw, 0.0, float(self.config.max_linear_speed)))
            w = float(
                np.clip(
                    w_raw,
                    -float(self.config.max_angular_speed),
                    float(self.config.max_angular_speed),
                )
            )
            theta = _wrap_angle(theta + w * dt)
            x = x + np.asarray([v * np.cos(theta) * dt, v * np.sin(theta) * dt], dtype=float)

            goal_dist = float(np.linalg.norm(context.goal - x))
            cumulative_goal_dist += goal_dist
            goal_heading = float(np.arctan2(context.goal[1] - x[1], context.goal[0] - x[0]))
            heading_error = abs(_wrap_angle(goal_heading - theta))
            cost += float(self.config.heading_weight) * heading_error
            cost += float(self.config.control_effort_weight) * (0.35 * v * v + 0.15 * w * w)
            cost += float(self.config.smoothness_weight) * (
                abs(v - prev_v) + 0.25 * abs(w - prev_w)
            )
            if preferred_turn != 0.0:
                cost -= float(self.config.avoidance_turn_bias_weight) * preferred_turn * w
            prev_v = v
            prev_w = w

            ped_t = self._predict_pedestrians(
                context.ped_positions, context.ped_velocities, step_idx
            )
            if ped_t.size > 0:
                dists = np.linalg.norm(ped_t - x[None, :], axis=1)
                min_ped_clearance = float(np.min(dists)) - (
                    context.robot_radius + context.ped_radius
                )
                cost += float(self.config.pedestrian_clearance_weight) * self._soft_collision_cost(
                    min_ped_clearance,
                    float(self.config.pedestrian_margin),
                )

            obstacle_clearance = (
                self._min_obstacle_clearance(x, context.observation) - context.robot_radius
            )
            cost += float(self.config.obstacle_clearance_weight) * self._soft_collision_cost(
                obstacle_clearance,
                float(self.config.obstacle_margin),
            )
            cost += float(self.config.occupancy_cost_weight) * self._occupancy_cost(
                x, context.observation
            )

        end_dist = float(np.linalg.norm(context.goal - x))
        progress = start_dist - end_dist
        cost += float(self.config.path_goal_weight) * cumulative_goal_dist
        cost += float(self.config.terminal_goal_weight) * end_dist
        cost -= float(self.config.progress_reward_weight) * progress
        return float(cost)

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the first command of the locally optimized NMPC sequence."""
        (
            robot_pos,
            heading,
            speed,
            goal,
            ped_positions,
            ped_velocities,
            robot_radius,
            ped_radius,
        ) = self._extract_state(observation)
        goal_delta = goal - robot_pos
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        goal_heading = float(np.arctan2(goal_delta[1], goal_delta[0]))
        goal_heading_error = _wrap_angle(goal_heading - heading)
        preferred_turn = self._preferred_avoidance_turn(
            robot_pos=robot_pos,
            heading=heading,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
        )
        context = _RolloutContext(
            robot_pos=robot_pos,
            heading=heading,
            current_speed=speed,
            goal=goal,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
            observation=observation,
        )
        x0 = self._initial_guess(
            goal_heading_error=goal_heading_error,
            current_speed=speed,
            goal_distance=goal_distance,
            preferred_turn=preferred_turn,
        )
        horizon = max(int(self.config.horizon_steps), 1)
        lower = np.empty(horizon * 2, dtype=float)
        upper = np.empty(horizon * 2, dtype=float)
        lower[0::2] = 0.0
        upper[0::2] = float(self.config.max_linear_speed)
        lower[1::2] = -float(self.config.max_angular_speed)
        upper[1::2] = float(self.config.max_angular_speed)

        result = minimize(
            lambda u: self._rollout_cost(u, context=context),
            x0,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            options={
                "ftol": float(self.config.solver_ftol),
                "maxiter": int(self.config.solver_max_iterations),
                "disp": False,
            },
        )
        if not bool(result.success):
            self._last_solution = None
            if bool(self.config.fallback_to_stop):
                return 0.0, 0.0
            action = np.asarray(result.x if result.x is not None else x0, dtype=float)
        else:
            action = np.asarray(result.x, dtype=float)
            self._last_solution = action.copy()

        return (
            float(np.clip(action[0], 0.0, float(self.config.max_linear_speed))),
            float(
                np.clip(
                    action[1],
                    -float(self.config.max_angular_speed),
                    float(self.config.max_angular_speed),
                )
            ),
        )


def build_nmpc_social_config(cfg: dict[str, Any] | None) -> NMPCSocialConfig:
    """Build NMPC planner config from a mapping payload.

    Returns:
        NMPCSocialConfig: Parsed planner config with numeric overrides applied.
    """
    if not isinstance(cfg, dict):
        return NMPCSocialConfig()
    defaults = NMPCSocialConfig()
    numeric_casts = {
        "max_linear_speed": float,
        "max_angular_speed": float,
        "horizon_steps": int,
        "rollout_dt": float,
        "goal_tolerance": float,
        "path_goal_weight": float,
        "terminal_goal_weight": float,
        "progress_reward_weight": float,
        "heading_weight": float,
        "control_effort_weight": float,
        "smoothness_weight": float,
        "pedestrian_clearance_weight": float,
        "obstacle_clearance_weight": float,
        "occupancy_cost_weight": float,
        "collision_cost_kappa": float,
        "pedestrian_margin": float,
        "obstacle_margin": float,
        "obstacle_threshold": float,
        "obstacle_search_cells": int,
        "avoidance_turn_bias_weight": float,
        "symmetry_break_bias": float,
        "solver_ftol": float,
        "solver_max_iterations": int,
        "warm_start": bool,
        "fallback_to_stop": bool,
    }
    kwargs = {}
    for field in fields(NMPCSocialConfig):
        value = cfg.get(field.name, getattr(defaults, field.name))
        caster = numeric_casts.get(field.name)
        kwargs[field.name] = caster(value) if caster is not None else value
    return NMPCSocialConfig(**kwargs)
