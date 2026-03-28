"""Testing-only native safety-barrier local planner.

This planner is a clean-room, static-obstacle-first controller inspired by
barrier-style safety filters. It does not claim upstream `safe_control`
equivalence or CBF-QP fidelity. The implementation combines a goal-directed
unicycle command with conservative occupancy-grid probes to reduce speed,
commit to a turn direction, or stop when immediate obstacle risk is high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``.

    Returns:
        float: Wrapped angle.
    """
    wrapped = (float(angle) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped)


@dataclass
class SafetyBarrierPlannerConfig:
    """Configuration for :class:`SafetyBarrierPlannerAdapter`."""

    max_linear_speed: float = 0.8
    max_angular_speed: float = 1.2
    goal_tolerance: float = 0.25
    safe_distance: float = 1.0
    stop_distance: float = 0.35
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12
    forward_clearance_weight: float = 1.5
    turn_away_gain: float = 1.1
    progress_weight: float = 0.8
    heading_weight: float = 1.4
    recovery_turn_rate: float = 0.9


class SafetyBarrierPlannerAdapter(OccupancyAwarePlannerMixin):
    """Clean-room static-obstacle-first barrier-style planner."""

    def __init__(self, config: SafetyBarrierPlannerConfig | None = None) -> None:
        """Initialize the planner with conservative defaults."""
        self.config = config or SafetyBarrierPlannerConfig()
        self._turn_commit = 0.0

    def _select_goal_point(
        self,
        *,
        robot_pos: np.ndarray,
        goal_current: np.ndarray,
        goal_next: np.ndarray,
    ) -> np.ndarray:
        """Select the active route target from current/next goal fields.

        Returns:
            np.ndarray: Active goal point for nominal tracking.
        """
        current_dist = float(np.linalg.norm(goal_current - robot_pos))
        next_dist = float(np.linalg.norm(goal_next - robot_pos))
        current_valid = bool(np.isfinite(goal_current).all()) and current_dist > 1e-6
        next_valid = bool(np.isfinite(goal_next).all()) and next_dist > 1e-6

        if current_valid and current_dist > float(self.config.goal_tolerance):
            return goal_current
        if next_valid:
            return goal_next
        return goal_current if current_valid else goal_next

    def _extract_robot_goal(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Extract robot and goal state with safe defaults.

        Returns:
            tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
                Robot position, heading, speed, radius, selected goal point,
                current goal, and next goal.
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        speed = float(self._as_1d_float(robot_state.get("speed", [0.0]), pad=1)[0])
        radius = float(self._as_1d_float(robot_state.get("radius", [0.3]), pad=1)[0])

        goal_next = self._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        goal = self._select_goal_point(
            robot_pos=robot_pos,
            goal_current=goal_current,
            goal_next=goal_next,
        )
        return robot_pos, heading, speed, radius, goal, goal_current, goal_next

    def _nominal_command(
        self,
        robot_pos: np.ndarray,
        heading: float,
        goal: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Compute the unconstrained goal-directed command.

        Returns:
            tuple[float, float, float, float]:
                Linear speed, angular speed, goal distance, and heading error.
        """
        goal_vec = goal - robot_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= float(self.config.goal_tolerance):
            return 0.0, 0.0, goal_dist, 0.0

        desired_heading = float(np.arctan2(goal_vec[1], goal_vec[0]))
        heading_error = _wrap_angle(desired_heading - heading)
        angular = float(
            np.clip(
                float(self.config.heading_weight) * heading_error,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        alignment = max(0.0, np.cos(heading_error))
        linear = float(
            np.clip(
                float(self.config.progress_weight) * goal_dist * alignment * alignment,
                0.0,
                float(self.config.max_linear_speed),
            )
        )
        if abs(heading_error) > 0.75:
            linear = 0.0
        return linear, angular, goal_dist, heading_error

    def _ray_profile(
        self,
        *,
        robot_pos: np.ndarray,
        heading: float,
        meta: dict[str, Any],
        grid: np.ndarray,
        channel_idx: int,
        start_distance: float,
        max_distance: float,
        angle_offset: float,
    ) -> tuple[float, float]:
        """Return first occupied distance and mean occupancy along a ray."""
        if channel_idx < 0:
            return float("inf"), 0.0
        if max_distance <= start_distance:
            return float("inf"), 0.0

        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        step = max(resolution * 0.75, 0.05)
        heading_world = float(heading + angle_offset)
        direction = np.array([np.cos(heading_world), np.sin(heading_world)], dtype=float)
        distances = np.arange(start_distance, max_distance + step * 0.5, step, dtype=float)
        if distances.size == 0:
            return float("inf"), 0.0

        occupied_distance = float("inf")
        occupancies: list[float] = []
        threshold = float(self.config.obstacle_threshold)
        for distance in distances:
            point = robot_pos + direction * float(distance)
            value = float(self._grid_value(point, grid, meta, channel_idx))
            occupancies.append(value)
            if value >= threshold:
                occupied_distance = float(distance)
                break
        mean_occupancy = float(np.mean(occupancies)) if occupancies else 0.0
        return occupied_distance, mean_occupancy

    def _resolve_turn_sign(self, imbalance: float, heading_error: float) -> float:
        """Resolve a committed turn direction for symmetric or near-symmetric obstacles.

        Returns:
            float: Signed turn direction where positive values turn left.
        """
        if abs(imbalance) > 1e-3:
            sign = -1.0 if imbalance > 0.0 else 1.0
            self._turn_commit = sign
            return sign
        if abs(self._turn_commit) > 1e-3:
            return float(self._turn_commit)
        if abs(heading_error) > 1e-3:
            sign = 1.0 if heading_error >= 0.0 else -1.0
            self._turn_commit = sign
            return sign
        self._turn_commit = 1.0
        return 1.0

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a bounded ``(v, omega)`` command with conservative obstacle handling."""
        try:
            robot_pos, heading, speed, radius, goal, _goal_current, _goal_next = (
                self._extract_robot_goal(observation)
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return 0.0, 0.0

        payload = self._extract_grid_payload(observation)
        grid = None
        meta: dict[str, Any] | None = None
        channel_idx = -1
        if payload is not None:
            grid, meta = payload
            channel_idx = self._preferred_channel(meta)
        linear, angular, goal_dist, heading_error = self._nominal_command(robot_pos, heading, goal)
        if goal_dist <= float(self.config.goal_tolerance):
            self._turn_commit = 0.0
            return 0.0, 0.0

        if payload is None or grid is None or meta is None:
            return linear, angular

        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        max_probe = max(
            float(self.config.safe_distance) * 2.0,
            float(self.config.stop_distance) * 2.0,
            float(self.config.obstacle_search_cells) * max(resolution, 1e-6),
        )
        start_distance = max(radius, 0.05)
        forward_clearance, _ = self._ray_profile(
            robot_pos=robot_pos,
            heading=heading,
            meta=meta,
            grid=grid,
            channel_idx=channel_idx,
            start_distance=start_distance,
            max_distance=max_probe,
            angle_offset=0.0,
        )
        left_clearance, left_occ = self._ray_profile(
            robot_pos=robot_pos,
            heading=heading,
            meta=meta,
            grid=grid,
            channel_idx=channel_idx,
            start_distance=start_distance,
            max_distance=max_probe,
            angle_offset=0.65,
        )
        right_clearance, right_occ = self._ray_profile(
            robot_pos=robot_pos,
            heading=heading,
            meta=meta,
            grid=grid,
            channel_idx=channel_idx,
            start_distance=start_distance,
            max_distance=max_probe,
            angle_offset=-0.65,
        )
        front_left_clearance, front_left_occ = self._ray_profile(
            robot_pos=robot_pos,
            heading=heading,
            meta=meta,
            grid=grid,
            channel_idx=channel_idx,
            start_distance=start_distance,
            max_distance=max_probe,
            angle_offset=0.35,
        )
        front_right_clearance, front_right_occ = self._ray_profile(
            robot_pos=robot_pos,
            heading=heading,
            meta=meta,
            grid=grid,
            channel_idx=channel_idx,
            start_distance=start_distance,
            max_distance=max_probe,
            angle_offset=-0.35,
        )
        imbalance = float(left_occ - right_occ)
        near_lateral_clearance = min(
            left_clearance,
            right_clearance,
            front_left_clearance,
            front_right_clearance,
        )
        near_occ = max(left_occ, right_occ, front_left_occ, front_right_occ)
        if near_occ > 0.05 and (
            forward_clearance < float(self.config.safe_distance) * 1.25
            or near_lateral_clearance < float(self.config.safe_distance) * 1.5
        ):
            turn_sign = self._resolve_turn_sign(imbalance, heading_error)
            angular += (
                turn_sign
                * float(self.config.turn_away_gain)
                * min(
                    near_occ,
                    1.0,
                )
                * 0.5
            )

        effective_clearance = min(forward_clearance, front_left_clearance, front_right_clearance)

        if effective_clearance <= float(self.config.stop_distance):
            turn_sign = self._resolve_turn_sign(imbalance, heading_error)
            return 0.0, float(
                np.clip(
                    turn_sign * float(self.config.recovery_turn_rate),
                    -float(self.config.max_angular_speed),
                    float(self.config.max_angular_speed),
                )
            )

        if np.isfinite(effective_clearance) and effective_clearance < float(
            self.config.safe_distance
        ):
            norm = (effective_clearance - float(self.config.stop_distance)) / max(
                float(self.config.safe_distance) - float(self.config.stop_distance),
                1e-6,
            )
            speed_scale = float(np.clip(norm, 0.0, 1.0)) ** max(
                float(self.config.forward_clearance_weight), 1e-6
            )
            turn_sign = self._resolve_turn_sign(imbalance, heading_error)
            linear = min(linear, float(self.config.max_linear_speed)) * speed_scale
            angular += turn_sign * float(self.config.turn_away_gain) * (1.0 - speed_scale * 0.5)
        else:
            self._turn_commit = 0.0

        linear = float(np.clip(linear, 0.0, float(self.config.max_linear_speed)))
        angular = float(
            np.clip(
                angular, -float(self.config.max_angular_speed), float(self.config.max_angular_speed)
            )
        )
        if not np.isfinite(linear) or not np.isfinite(angular):
            return 0.0, 0.0

        # Avoid issuing accelerating forward commands while already rotating sharply in place.
        if abs(angular) > 0.9 * float(self.config.max_angular_speed):
            linear = min(linear, max(speed * 0.5, 0.15))
        return linear, angular


def build_safety_barrier_config(cfg: dict[str, Any] | None) -> SafetyBarrierPlannerConfig:
    """Build :class:`SafetyBarrierPlannerConfig` from mapping payload.

    Returns:
        SafetyBarrierPlannerConfig: Parsed planner config.
    """
    if not isinstance(cfg, dict):
        return SafetyBarrierPlannerConfig()

    return SafetyBarrierPlannerConfig(
        max_linear_speed=float(cfg.get("max_linear_speed", 0.8)),
        max_angular_speed=float(cfg.get("max_angular_speed", 1.2)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        safe_distance=float(cfg.get("safe_distance", 1.0)),
        stop_distance=float(cfg.get("stop_distance", 0.35)),
        obstacle_threshold=float(cfg.get("obstacle_threshold", 0.5)),
        obstacle_search_cells=int(cfg.get("obstacle_search_cells", 12)),
        forward_clearance_weight=float(cfg.get("forward_clearance_weight", 1.5)),
        turn_away_gain=float(cfg.get("turn_away_gain", 1.1)),
        progress_weight=float(cfg.get("progress_weight", 0.8)),
        heading_weight=float(cfg.get("heading_weight", 1.4)),
        recovery_turn_rate=float(cfg.get("recovery_turn_rate", 0.9)),
    )


__all__ = [
    "SafetyBarrierPlannerAdapter",
    "SafetyBarrierPlannerConfig",
    "build_safety_barrier_config",
]
