"""Safety guard for PPO benchmark actions.

This adapter keeps PPO as the primary action source and only intervenes when a
short-horizon rollout predicts unsafe pedestrian or obstacle clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, _wrap_angle, build_risk_dwa_config
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


@dataclass
class GuardedPPOConfig:
    """Configuration for the PPO safety guard."""

    rollout_dt: float = 0.2
    rollout_steps: int = 6
    goal_tolerance: float = 0.25
    near_field_distance: float = 2.0
    hard_ped_clearance: float = 0.58
    first_step_ped_clearance: float = 0.72
    hard_obstacle_clearance: float = 0.30
    min_ttc: float = 0.70
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12


class GuardedPPOAdapter(OccupancyAwarePlannerMixin):
    """Intervene on PPO actions only when they violate short-horizon safety checks."""

    def __init__(
        self,
        config: GuardedPPOConfig | None = None,
        *,
        fallback_adapter: RiskDWAPlannerAdapter | None = None,
    ) -> None:
        """Initialize guard and fallback local planner."""
        self.config = config or GuardedPPOConfig()
        self.fallback_adapter = fallback_adapter or RiskDWAPlannerAdapter()

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Extract robot, goal, and pedestrian state from structured observation.

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
        if ped_vel.ndim != 2 or ped_vel.shape[-1] != 2 or ped_vel.shape[0] != ped_pos.shape[0]:
            ped_vel = np.zeros_like(ped_pos)
        return robot_pos, heading, goal, ped_pos, ped_vel

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, Any]) -> float:
        """Approximate obstacle clearance from occupancy grid payload.

        Returns:
            float: Clearance in meters, or ``inf`` when unavailable.
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

    def _evaluate_command(
        self, observation: dict[str, Any], command: tuple[float, float]
    ) -> dict[str, float | bool]:
        """Evaluate a command over a short rollout horizon.

        Returns:
            dict[str, float | bool]: Safety summary including `safe` and clearance metrics.
        """
        robot_pos, heading, goal, ped_pos, ped_vel = self._extract_state(observation)
        dt = float(self.config.rollout_dt)
        steps = max(int(self.config.rollout_steps), 1)
        x = np.array(robot_pos, dtype=float)
        theta = float(heading)
        start_dist = float(np.linalg.norm(goal - robot_pos))
        min_ped_clear = float("inf")
        first_ped_clear = float("inf")
        min_obs_clear = float("inf")
        min_ttc = float("inf")

        for step in range(steps):
            t = (step + 1) * dt
            x = x + np.array(
                [
                    float(command[0]) * np.cos(theta) * dt,
                    float(command[0]) * np.sin(theta) * dt,
                ],
                dtype=float,
            )
            theta = _wrap_angle(theta + float(command[1]) * dt)

            if ped_pos.size > 0:
                ped_t = ped_pos + ped_vel * t
                ped_dist = np.linalg.norm(ped_t - x[None, :], axis=1)
                if ped_dist.size > 0:
                    min_ped_clear = min(min_ped_clear, float(np.min(ped_dist)))
                    if step == 0:
                        first_ped_clear = min(first_ped_clear, float(np.min(ped_dist)))

                    robot_vel = np.array(
                        [
                            float(command[0]) * np.cos(theta),
                            float(command[0]) * np.sin(theta),
                        ],
                        dtype=float,
                    )
                    rel_pos = ped_t - x[None, :]
                    rel_vel = ped_vel - robot_vel[None, :]
                    rel_speed_sq = np.sum(rel_vel * rel_vel, axis=1)
                    valid = rel_speed_sq > 1e-6
                    if np.any(valid):
                        ttc = -np.sum(rel_pos[valid] * rel_vel[valid], axis=1) / rel_speed_sq[valid]
                        ttc = ttc[ttc > 0.0]
                        if ttc.size > 0:
                            min_ttc = min(min_ttc, float(np.min(ttc)))
            min_obs_clear = min(min_obs_clear, self._min_obstacle_clearance(x, observation))

        end_dist = float(np.linalg.norm(goal - x))
        progress = start_dist - end_dist
        safe = (
            min_ped_clear >= float(self.config.hard_ped_clearance)
            and first_ped_clear >= float(self.config.first_step_ped_clearance)
            and min_obs_clear >= float(self.config.hard_obstacle_clearance)
            and (not np.isfinite(min_ttc) or min_ttc >= float(self.config.min_ttc))
        )
        return {
            "safe": bool(safe),
            "progress": float(progress),
            "min_ped_clear": float(min_ped_clear),
            "first_ped_clear": float(first_ped_clear),
            "min_obs_clear": float(min_obs_clear),
            "min_ttc": float(min_ttc),
        }

    def choose_command(
        self, observation: dict[str, Any], ppo_command: tuple[float, float]
    ) -> tuple[tuple[float, float], str]:
        """Choose between PPO, fallback planner, and stop.

        Returns:
            tuple[tuple[float, float], str]: Selected command and decision label.
        """
        robot_pos, _heading, goal, ped_pos, _ped_vel = self._extract_state(observation)
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.config.goal_tolerance):
            return (0.0, 0.0), "goal_reached"

        if ped_pos.size > 0:
            current_min_dist = float(np.min(np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)))
        else:
            current_min_dist = float("inf")

        if current_min_dist > float(self.config.near_field_distance):
            return (float(ppo_command[0]), float(ppo_command[1])), "ppo_clear"

        ppo_eval = self._evaluate_command(observation, ppo_command)
        if bool(ppo_eval["safe"]):
            return (float(ppo_command[0]), float(ppo_command[1])), "ppo_safe"

        fallback_command = self.fallback_adapter.plan(observation)
        fallback_eval = self._evaluate_command(observation, fallback_command)
        if bool(fallback_eval["safe"]):
            return (float(fallback_command[0]), float(fallback_command[1])), "fallback_safe"

        stop_eval = self._evaluate_command(observation, (0.0, 0.0))
        if bool(stop_eval["safe"]):
            return (0.0, 0.0), "stop_safe"

        if float(fallback_eval["min_ped_clear"]) > max(
            float(ppo_eval["min_ped_clear"]), float(stop_eval["min_ped_clear"])
        ):
            return (float(fallback_command[0]), float(fallback_command[1])), "fallback_best_effort"
        return (0.0, 0.0), "stop_best_effort"


def build_guarded_ppo_config(cfg: dict[str, Any] | None) -> GuardedPPOConfig:
    """Build :class:`GuardedPPOConfig` from mapping payload.

    Returns:
        GuardedPPOConfig: Parsed guard configuration.
    """
    if not isinstance(cfg, dict):
        return GuardedPPOConfig()
    return GuardedPPOConfig(
        rollout_dt=float(cfg.get("guard_rollout_dt", 0.2)),
        rollout_steps=int(cfg.get("guard_rollout_steps", 6)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        near_field_distance=float(cfg.get("guard_near_field_distance", 2.0)),
        hard_ped_clearance=float(cfg.get("guard_hard_ped_clearance", 0.58)),
        first_step_ped_clearance=float(cfg.get("guard_first_step_ped_clearance", 0.72)),
        hard_obstacle_clearance=float(cfg.get("guard_hard_obstacle_clearance", 0.30)),
        min_ttc=float(cfg.get("guard_min_ttc", 0.70)),
        obstacle_threshold=float(cfg.get("guard_obstacle_threshold", 0.5)),
        obstacle_search_cells=int(cfg.get("guard_obstacle_search_cells", 12)),
    )


def build_guarded_ppo_fallback(cfg: dict[str, Any] | None) -> RiskDWAPlannerAdapter:
    """Build fallback local planner for guarded PPO.

    Returns:
        RiskDWAPlannerAdapter: Fallback adapter used when PPO action is unsafe.
    """
    root = cfg if isinstance(cfg, dict) else {}
    fallback_cfg = root.get("fallback_risk_dwa")
    if not isinstance(fallback_cfg, dict):
        fallback_cfg = {}
    return RiskDWAPlannerAdapter(config=build_risk_dwa_config(fallback_cfg))


__all__ = [
    "GuardedPPOAdapter",
    "GuardedPPOConfig",
    "build_guarded_ppo_config",
    "build_guarded_ppo_fallback",
]
