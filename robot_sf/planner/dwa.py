"""Deterministic Dynamic Window Approach (DWA) local-planner baseline.

The planner samples unicycle commands that are reachable from the current command
within one control period, rolls them out, and scores goal alignment, clearance,
velocity, and progress. It is an in-repository classical baseline, not an
upstream-wrapper claim and not benchmark-performance evidence by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


@dataclass
class DWAPlannerConfig:
    """Tunable parameters for the classical DWA baseline."""

    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.2
    max_linear_acceleration: float = 0.8
    max_angular_acceleration: float = 1.5
    control_dt: float = 0.2
    prediction_dt: float = 0.1
    prediction_steps: int = 15
    linear_samples: int = 7
    angular_samples: int = 9
    goal_tolerance: float = 0.25
    robot_radius: float = 0.25
    pedestrian_radius: float = 0.30
    safety_margin: float = 0.10
    clearance_distance: float = 2.0
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12
    heading_weight: float = 0.8
    clearance_weight: float = 1.2
    velocity_weight: float = 0.25
    progress_weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid dynamics and rollout parameters before planning begins."""
        numeric_values = {
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "max_linear_acceleration": self.max_linear_acceleration,
            "max_angular_acceleration": self.max_angular_acceleration,
            "control_dt": self.control_dt,
            "prediction_dt": self.prediction_dt,
            "goal_tolerance": self.goal_tolerance,
            "robot_radius": self.robot_radius,
            "pedestrian_radius": self.pedestrian_radius,
            "safety_margin": self.safety_margin,
            "clearance_distance": self.clearance_distance,
            "obstacle_threshold": self.obstacle_threshold,
            "heading_weight": self.heading_weight,
            "clearance_weight": self.clearance_weight,
            "velocity_weight": self.velocity_weight,
            "progress_weight": self.progress_weight,
        }
        invalid = [name for name, value in numeric_values.items() if not isfinite(float(value))]
        if invalid:
            raise ValueError(f"DWA configuration values must be finite: {', '.join(invalid)}")
        positive = ("max_linear_speed", "max_angular_speed", "control_dt", "prediction_dt")
        if any(float(numeric_values[name]) <= 0.0 for name in positive):
            raise ValueError(
                f"DWA configuration values must be greater than zero: {', '.join(positive)}"
            )
        nonnegative = set(numeric_values) - set(positive)
        if any(float(numeric_values[name]) < 0.0 for name in nonnegative):
            raise ValueError("DWA configuration values must not be negative")
        if not 0.0 <= float(self.obstacle_threshold) <= 1.0:
            raise ValueError("obstacle_threshold must be in [0, 1]")
        if any(
            int(value) < 1
            for value in (self.prediction_steps, self.linear_samples, self.angular_samples)
        ):
            raise ValueError(
                "prediction_steps, linear_samples, and angular_samples must be at least one"
            )
        if int(self.obstacle_search_cells) < 1:
            raise ValueError("obstacle_search_cells must be at least one")


class DWAPlannerAdapter(OccupancyAwarePlannerMixin):
    """Classical Dynamic Window Approach planner producing bounded ``(v, omega)`` actions."""

    def __init__(self, config: DWAPlannerConfig | None = None) -> None:
        """Initialize the deterministic planner with optional parameter overrides."""
        self.config = config or DWAPlannerConfig()

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray]:
        """Return robot pose/command state, active goal, and pedestrian positions."""
        robot, goal, pedestrians = self._socnav_fields(observation)
        robot = robot or {}
        goal = goal or {}
        pedestrians = pedestrians or {}
        robot_pos = self._as_1d_float(robot.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot.get("heading", [0.0]), pad=1)[0])
        linear_speed = float(self._as_1d_float(robot.get("speed", [0.0]), pad=1)[0])
        angular_speed = float(
            self._as_1d_float(robot.get("angular_velocity", robot.get("omega", [0.0])), pad=1)[0]
        )

        goal_current = self._as_1d_float(goal.get("current", [0.0, 0.0]), pad=2)[:2]
        goal_next = self._as_1d_float(goal.get("next", [0.0, 0.0]), pad=2)[:2]
        active_goal = (
            goal_next
            if np.linalg.norm(goal_next - robot_pos) > float(self.config.goal_tolerance)
            else goal_current
        )

        raw_positions = np.asarray(pedestrians.get("positions", []), dtype=float)
        if raw_positions.ndim == 1 and raw_positions.size % 2 == 0:
            raw_positions = raw_positions.reshape(-1, 2)
        if raw_positions.ndim != 2 or raw_positions.shape[-1] != 2:
            raw_positions = np.zeros((0, 2), dtype=float)
        count = max(
            int(self._as_1d_float(pedestrians.get("count", [raw_positions.shape[0]]), pad=1)[0]),
            0,
        )
        return robot_pos, heading, linear_speed, angular_speed, active_goal, raw_positions[:count]

    def _dynamic_window(
        self, linear_speed: float, angular_speed: float
    ) -> tuple[float, float, float, float]:
        """Return reachable linear/angular bounds for the next control period."""
        dt = max(float(self.config.control_dt), 0.0)
        v_delta = max(float(self.config.max_linear_acceleration), 0.0) * dt
        w_delta = max(float(self.config.max_angular_acceleration), 0.0) * dt
        v_reach_min, v_reach_max = linear_speed - v_delta, linear_speed + v_delta
        w_reach_min, w_reach_max = angular_speed - w_delta, angular_speed + w_delta
        v_min, v_max = _reachable_interval(
            v_reach_min, v_reach_max, 0.0, self.config.max_linear_speed
        )
        w_min, w_max = _reachable_interval(
            w_reach_min,
            w_reach_max,
            -self.config.max_angular_speed,
            self.config.max_angular_speed,
        )
        return (
            v_min,
            v_max,
            w_min,
            w_max,
        )

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, Any]) -> float:
        """Return grid-derived obstacle clearance, or infinity when no grid is available."""
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
        if not (0 <= row < channel_grid.shape[0] and 0 <= col < channel_grid.shape[1]):
            return 0.0
        if channel_grid[row, col] >= float(self.config.obstacle_threshold):
            return 0.0
        radius = max(int(self.config.obstacle_search_cells), 1)
        r0, r1 = max(0, row - radius), min(channel_grid.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(channel_grid.shape[1], col + radius + 1)
        obstacle_indices = np.argwhere(channel_grid[r0:r1, c0:c1] >= self.config.obstacle_threshold)
        if obstacle_indices.size == 0:
            return float("inf")
        row_delta = obstacle_indices[:, 0] + r0 - row
        col_delta = obstacle_indices[:, 1] + c0 - col
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        return float(np.min(np.hypot(row_delta, col_delta)) * max(resolution, 1e-6))

    def _rollout_score(
        self,
        *,
        robot_pos: np.ndarray,
        heading: float,
        goal: np.ndarray,
        pedestrian_positions: np.ndarray,
        command: tuple[float, float],
        observation: dict[str, Any],
    ) -> float:
        """Score a constant command rollout; a collision candidate is infeasible.

        Returns:
            Higher-is-better score, or negative infinity for an unsafe rollout.
        """
        position = np.array(robot_pos, dtype=float)
        orientation = float(heading)
        start_distance = float(np.linalg.norm(goal - position))
        min_clearance = float("inf")
        for _ in range(max(int(self.config.prediction_steps), 1)):
            position += np.array(
                [
                    command[0] * np.cos(orientation) * float(self.config.prediction_dt),
                    command[0] * np.sin(orientation) * float(self.config.prediction_dt),
                ]
            )
            orientation = wrap_angle_pi(orientation + command[1] * float(self.config.prediction_dt))
            if pedestrian_positions.size:
                pedestrian_clearance = (
                    float(np.min(np.linalg.norm(pedestrian_positions - position[None, :], axis=1)))
                    - float(self.config.robot_radius)
                    - float(self.config.pedestrian_radius)
                )
                min_clearance = min(min_clearance, pedestrian_clearance)
            min_clearance = min(
                min_clearance,
                self._min_obstacle_clearance(position, observation)
                - float(self.config.robot_radius),
            )
        if min_clearance <= float(self.config.safety_margin):
            return float("-inf")
        end_distance = float(np.linalg.norm(goal - position))
        desired_heading = float(np.arctan2(goal[1] - position[1], goal[0] - position[0]))
        heading_score = float(np.cos(wrap_angle_pi(desired_heading - orientation)))
        clearance_score = min(min_clearance, float(self.config.clearance_distance)) / max(
            float(self.config.clearance_distance), 1e-6
        )
        velocity_score = command[0] / max(float(self.config.max_linear_speed), 1e-6)
        return (
            float(self.config.heading_weight) * heading_score
            + float(self.config.clearance_weight) * clearance_score
            + float(self.config.velocity_weight) * velocity_score
            + float(self.config.progress_weight) * (start_distance - end_distance)
        )

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Select the highest-scoring dynamically reachable unicycle command.

        Returns:
            Bounded linear and angular velocity command ``(v, omega)``.
        """
        robot_pos, heading, linear_speed, angular_speed, goal, pedestrians = self._extract_state(
            observation
        )
        if np.linalg.norm(goal - robot_pos) <= float(self.config.goal_tolerance):
            return 0.0, 0.0
        v_min, v_max, w_min, w_max = self._dynamic_window(linear_speed, angular_speed)
        linear_candidates = np.linspace(v_min, v_max, int(self.config.linear_samples))
        angular_candidates = np.linspace(w_min, w_max, int(self.config.angular_samples))
        best_score = float("-inf")
        best_command = (0.0, 0.0)
        for linear in linear_candidates:
            for angular in angular_candidates:
                command = (float(linear), float(angular))
                score = self._rollout_score(
                    robot_pos=robot_pos,
                    heading=heading,
                    goal=goal,
                    pedestrian_positions=pedestrians,
                    command=command,
                    observation=observation,
                )
                if score > best_score:
                    best_score, best_command = score, command
        return best_command


def _reachable_interval(
    reachable_min: float, reachable_max: float, limit_min: float, limit_max: float
) -> tuple[float, float]:
    """Intersect reachable and configured ranges, preserving acceleration reachability when disjoint.

    Returns:
        Lower and upper candidate bounds, equal when only the nearest reachable value is valid.
    """
    if reachable_max < limit_min:
        return reachable_max, reachable_max
    if reachable_min > limit_max:
        return reachable_min, reachable_min
    return max(reachable_min, limit_min), min(reachable_max, limit_max)


def build_dwa_config(cfg: dict[str, Any] | None) -> DWAPlannerConfig:
    """Build a DWA config from an algorithm-config mapping.

    Returns:
        Parsed DWA configuration using defaults for omitted parameters.
    """
    if not isinstance(cfg, dict):
        return DWAPlannerConfig()
    defaults = DWAPlannerConfig()
    return DWAPlannerConfig(
        max_linear_speed=float(cfg.get("max_linear_speed", defaults.max_linear_speed)),
        max_angular_speed=float(cfg.get("max_angular_speed", defaults.max_angular_speed)),
        max_linear_acceleration=float(
            cfg.get("max_linear_acceleration", defaults.max_linear_acceleration)
        ),
        max_angular_acceleration=float(
            cfg.get("max_angular_acceleration", defaults.max_angular_acceleration)
        ),
        control_dt=float(cfg.get("control_dt", defaults.control_dt)),
        prediction_dt=float(cfg.get("prediction_dt", defaults.prediction_dt)),
        prediction_steps=int(cfg.get("prediction_steps", defaults.prediction_steps)),
        linear_samples=int(cfg.get("linear_samples", defaults.linear_samples)),
        angular_samples=int(cfg.get("angular_samples", defaults.angular_samples)),
        goal_tolerance=float(cfg.get("goal_tolerance", defaults.goal_tolerance)),
        robot_radius=float(cfg.get("robot_radius", defaults.robot_radius)),
        pedestrian_radius=float(cfg.get("pedestrian_radius", defaults.pedestrian_radius)),
        safety_margin=float(cfg.get("safety_margin", defaults.safety_margin)),
        clearance_distance=float(cfg.get("clearance_distance", defaults.clearance_distance)),
        obstacle_threshold=float(cfg.get("obstacle_threshold", defaults.obstacle_threshold)),
        obstacle_search_cells=int(cfg.get("obstacle_search_cells", defaults.obstacle_search_cells)),
        heading_weight=float(cfg.get("heading_weight", defaults.heading_weight)),
        clearance_weight=float(cfg.get("clearance_weight", defaults.clearance_weight)),
        velocity_weight=float(cfg.get("velocity_weight", defaults.velocity_weight)),
        progress_weight=float(cfg.get("progress_weight", defaults.progress_weight)),
    )
