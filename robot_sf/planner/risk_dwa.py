"""Risk-aware dynamic-window local planner adapter.

This adapter is non-learning and scores constant unicycle commands over a short
rollout horizon against progress, heading alignment, pedestrian/obstacle
clearance, TTC proxy, and smoothness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


def _wrap_angle(angle: float) -> float:
    """Wrap angle to ``[-pi, pi]``.

    Returns:
        float: Wrapped angle.
    """
    wrapped = (float(angle) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped)


def _safe_mean(values: np.ndarray) -> float:
    """Return finite mean or ``0.0`` for empty/invalid arrays."""
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    return mean if np.isfinite(mean) else 0.0


@dataclass
class RiskDWAPlannerConfig:
    """Configuration for :class:`RiskDWAPlannerAdapter`."""

    max_linear_speed: float = 1.2
    max_angular_speed: float = 1.2
    rollout_dt: float = 0.2
    rollout_steps: int = 8
    goal_tolerance: float = 0.25

    linear_candidates: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    angular_candidates: tuple[float, ...] = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)

    goal_progress_weight: float = 4.0
    heading_weight: float = 0.8
    ped_clearance_weight: float = 1.6
    obstacle_clearance_weight: float = 1.2
    smoothness_weight: float = 0.15
    ttc_weight: float = 0.3

    safe_distance: float = 0.35
    near_distance: float = 0.7
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12

    near_field_distance: float = 2.5
    density_norm_count: float = 8.0
    near_field_speed_cap: float = 0.6

    progress_escape_enabled: bool = True
    progress_escape_distance: float = 1.0
    progress_escape_speed: float = 0.45
    progress_escape_heading_gain: float = 1.4


class RiskDWAPlannerAdapter(OccupancyAwarePlannerMixin):
    """Deterministic, non-learning dynamic-window style planner."""

    def __init__(self, config: RiskDWAPlannerConfig | None = None) -> None:
        """Initialize adapter with an optional config override."""
        self.config = config or RiskDWAPlannerConfig()

    def _extract_robot_goal_ped(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Extract robot/goal/pedestrian state from observation.

        Returns:
            tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
                Robot position, heading, goal position, pedestrian positions, pedestrian velocities.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)

        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])

        goal_next = self._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        goal = goal_next if np.linalg.norm(goal_next - robot_pos) > 1e-6 else goal_current

        ped_positions_raw = ped_state.get("positions")
        ped_velocities_raw = ped_state.get("velocities")
        ped_pos = np.asarray([] if ped_positions_raw is None else ped_positions_raw, dtype=float)
        ped_vel = np.asarray([] if ped_velocities_raw is None else ped_velocities_raw, dtype=float)
        if ped_pos.ndim != 2 or ped_pos.shape[-1] != 2:
            ped_pos = np.zeros((0, 2), dtype=float)
        if ped_vel.ndim != 2 or ped_vel.shape[-1] != 2:
            ped_vel = np.zeros_like(ped_pos)
        if ped_vel.shape[0] != ped_pos.shape[0]:
            ped_vel = np.zeros_like(ped_pos)

        return robot_pos, heading, goal, ped_pos, ped_vel

    def _crowd_density_scale(self, robot_pos: np.ndarray, ped_pos: np.ndarray) -> float:
        """Return speed scaling factor in dense near field."""
        if ped_pos.size == 0:
            return 1.0
        dists = np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)
        near_count = float(np.sum(dists <= float(self.config.near_field_distance)))
        density_ratio = near_count / max(float(self.config.density_norm_count), 1.0)
        cap_ratio = float(self.config.near_field_speed_cap) / max(
            float(self.config.max_linear_speed), 1e-6
        )
        return float(np.clip(1.0 - density_ratio, cap_ratio, 1.0))

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, Any]) -> float:
        """Approximate obstacle clearance from occupancy grid payload.

        Returns:
            float: Clearance in meters (`inf` when unavailable/no nearby obstacle).
        """
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
        cell_dist = np.sqrt(dr.astype(float) ** 2 + dc.astype(float) ** 2)
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        return float(np.min(cell_dist) * max(resolution, 1e-6))

    def _ttc_proxy(
        self,
        robot_pos: np.ndarray,
        command: tuple[float, float],
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
        heading: float,
    ) -> float:
        """Compute minimum positive TTC approximation for a command.

        Returns:
            float: Minimum positive TTC in seconds (`inf` when no collision trajectory).
        """
        if ped_pos.size == 0:
            return float("inf")
        v = float(command[0])
        robot_vel = np.array([v * np.cos(heading), v * np.sin(heading)], dtype=float)
        rel_pos = ped_pos - robot_pos[None, :]
        rel_vel = ped_vel - robot_vel[None, :]
        rel_speed_sq = np.sum(rel_vel * rel_vel, axis=1)
        valid = rel_speed_sq > 1e-6
        if not np.any(valid):
            return float("inf")
        ttc = -np.sum(rel_pos[valid] * rel_vel[valid], axis=1) / rel_speed_sq[valid]
        ttc = ttc[ttc > 0.0]
        if ttc.size == 0:
            return float("inf")
        return float(np.min(ttc))

    def _rollout_score(
        self,
        *,
        robot_pos: np.ndarray,
        heading: float,
        goal: np.ndarray,
        command: tuple[float, float],
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
        observation: dict[str, Any],
        current_speed: float,
    ) -> float:
        """Evaluate one constant command across rollout horizon.

        Returns:
            float: Scalar score where higher is better.
        """
        dt = float(self.config.rollout_dt)
        steps = max(int(self.config.rollout_steps), 1)
        x = np.array(robot_pos, dtype=float)
        theta = float(heading)
        start_dist = float(np.linalg.norm(goal - x))
        min_ped_clear = float("inf")
        min_obs_clear = float("inf")

        for step in range(steps):
            t = (step + 1) * dt
            x = x + np.array(
                [
                    float(command[0]) * np.cos(theta) * dt,
                    float(command[0]) * np.sin(theta) * dt,
                ]
            )
            theta = _wrap_angle(theta + float(command[1]) * dt)

            if ped_pos.size > 0:
                ped_t = ped_pos + ped_vel * t
                ped_dist = np.linalg.norm(ped_t - x[None, :], axis=1)
                min_ped_clear = min(min_ped_clear, float(np.min(ped_dist)))
            min_obs_clear = min(min_obs_clear, self._min_obstacle_clearance(x, observation))

        end_dist = float(np.linalg.norm(goal - x))
        progress = start_dist - end_dist
        goal_heading = float(np.arctan2(goal[1] - x[1], goal[0] - x[0]))
        heading_score = float(np.cos(_wrap_angle(goal_heading - theta)))
        smooth_penalty = abs(float(command[0]) - float(current_speed)) + 0.2 * abs(
            float(command[1])
        )

        ped_risk = max(0.0, float(self.config.near_distance) - min_ped_clear)
        obs_risk = max(0.0, float(self.config.near_distance) - min_obs_clear)
        ttc = self._ttc_proxy(robot_pos, command, ped_pos, ped_vel, heading)
        ttc_term = 0.0 if not np.isfinite(ttc) else 1.0 / max(ttc, 1e-3)

        return (
            float(self.config.goal_progress_weight) * progress
            + float(self.config.heading_weight) * heading_score
            + float(self.config.ped_clearance_weight) * min(min_ped_clear, 2.0)
            + float(self.config.obstacle_clearance_weight) * min(min_obs_clear, 2.0)
            - 2.0 * ped_risk
            - 1.5 * obs_risk
            - float(self.config.ttc_weight) * ttc_term
            - float(self.config.smoothness_weight) * smooth_penalty
        )

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return best unicycle command `(v, omega)` for the current observation."""
        robot_pos, heading, goal, ped_pos, ped_vel = self._extract_robot_goal_ped(observation)
        to_goal = float(np.linalg.norm(goal - robot_pos))
        if to_goal <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        current_speed = float(
            self._as_1d_float(self._socnav_fields(observation)[0].get("speed", [0.0]), pad=1)[0]
        )
        density_scale = self._crowd_density_scale(robot_pos, ped_pos)

        speed_cap = min(
            float(self.config.max_linear_speed),
            float(self.config.max_linear_speed) * density_scale,
        )
        best_score = float("-inf")
        best_cmd = (0.0, 0.0)

        for v_raw in self.config.linear_candidates:
            v = float(np.clip(v_raw, 0.0, speed_cap))
            for w_raw in self.config.angular_candidates:
                w = float(
                    np.clip(w_raw, -self.config.max_angular_speed, self.config.max_angular_speed)
                )
                score = self._rollout_score(
                    robot_pos=robot_pos,
                    heading=heading,
                    goal=goal,
                    command=(v, w),
                    ped_pos=ped_pos,
                    ped_vel=ped_vel,
                    observation=observation,
                    current_speed=current_speed,
                )
                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)

        if bool(self.config.progress_escape_enabled):
            if (
                to_goal > float(self.config.progress_escape_distance)
                and best_cmd[0] < float(self.config.progress_escape_speed) * 0.6
            ):
                goal_heading = float(np.arctan2(goal[1] - robot_pos[1], goal[0] - robot_pos[0]))
                heading_err = _wrap_angle(goal_heading - heading)
                escape_v = float(np.clip(self.config.progress_escape_speed, 0.0, speed_cap))
                escape_w = float(
                    np.clip(
                        heading_err * float(self.config.progress_escape_heading_gain),
                        -float(self.config.max_angular_speed),
                        float(self.config.max_angular_speed),
                    )
                )
                escape_score = self._rollout_score(
                    robot_pos=robot_pos,
                    heading=heading,
                    goal=goal,
                    command=(escape_v, escape_w),
                    ped_pos=ped_pos,
                    ped_vel=ped_vel,
                    observation=observation,
                    current_speed=current_speed,
                )
                if escape_score > best_score:
                    best_score = escape_score
                    best_cmd = (escape_v, escape_w)

        return best_cmd


def build_risk_dwa_config(cfg: dict[str, Any] | None) -> RiskDWAPlannerConfig:
    """Build :class:`RiskDWAPlannerConfig` from mapping payload.

    Returns:
        RiskDWAPlannerConfig: Parsed planner config.
    """
    if not isinstance(cfg, dict):
        return RiskDWAPlannerConfig()

    linear = cfg.get("linear_candidates", RiskDWAPlannerConfig.linear_candidates)
    angular = cfg.get("angular_candidates", RiskDWAPlannerConfig.angular_candidates)
    linear_candidates = (
        tuple(float(v) for v in linear)
        if isinstance(linear, (list, tuple))
        else RiskDWAPlannerConfig.linear_candidates
    )
    angular_candidates = (
        tuple(float(v) for v in angular)
        if isinstance(angular, (list, tuple))
        else RiskDWAPlannerConfig.angular_candidates
    )

    return RiskDWAPlannerConfig(
        max_linear_speed=float(cfg.get("max_linear_speed", 1.2)),
        max_angular_speed=float(cfg.get("max_angular_speed", 1.2)),
        rollout_dt=float(cfg.get("rollout_dt", 0.2)),
        rollout_steps=int(cfg.get("rollout_steps", 8)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        linear_candidates=linear_candidates,
        angular_candidates=angular_candidates,
        goal_progress_weight=float(cfg.get("goal_progress_weight", 4.0)),
        heading_weight=float(cfg.get("heading_weight", 0.8)),
        ped_clearance_weight=float(cfg.get("ped_clearance_weight", 1.6)),
        obstacle_clearance_weight=float(cfg.get("obstacle_clearance_weight", 1.2)),
        smoothness_weight=float(cfg.get("smoothness_weight", 0.15)),
        ttc_weight=float(cfg.get("ttc_weight", 0.3)),
        safe_distance=float(cfg.get("safe_distance", 0.35)),
        near_distance=float(cfg.get("near_distance", 0.7)),
        obstacle_threshold=float(cfg.get("obstacle_threshold", 0.5)),
        obstacle_search_cells=int(cfg.get("obstacle_search_cells", 12)),
        near_field_distance=float(cfg.get("near_field_distance", 2.5)),
        density_norm_count=float(cfg.get("density_norm_count", 8.0)),
        near_field_speed_cap=float(cfg.get("near_field_speed_cap", 0.6)),
        progress_escape_enabled=bool(cfg.get("progress_escape_enabled", True)),
        progress_escape_distance=float(cfg.get("progress_escape_distance", 1.0)),
        progress_escape_speed=float(cfg.get("progress_escape_speed", 0.45)),
        progress_escape_heading_gain=float(cfg.get("progress_escape_heading_gain", 1.4)),
    )


__all__ = [
    "RiskDWAPlannerAdapter",
    "RiskDWAPlannerConfig",
    "build_risk_dwa_config",
]
