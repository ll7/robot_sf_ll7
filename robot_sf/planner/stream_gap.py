"""Gap-acceptance local planner for pedestrian stream crossing.

This planner detects pedestrians occupying the goal corridor, estimates future
free windows with a simple constant-velocity projection, and switches between
wait/approach/commit modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``.

    Returns:
        float: Wrapped angle.
    """
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class StreamGapPlannerConfig:
    """Configuration for :class:`StreamGapPlannerAdapter`."""

    max_linear_speed: float = 1.2
    max_angular_speed: float = 1.2
    goal_tolerance: float = 0.25
    heading_gain: float = 1.6
    turn_in_place_angle: float = 0.7

    forward_lookahead: float = 4.0
    rear_margin: float = 0.5
    corridor_half_width: float = 0.85
    emergency_clearance: float = 0.55

    sample_dt: float = 0.2
    sample_horizon: float = 4.0
    safe_gap_time: float = 1.0
    approach_gap_time: float = 0.8

    wait_speed: float = 0.0
    creep_speed: float = 0.12
    approach_speed: float = 0.35
    commit_speed: float = 0.95
    commit_hold_steps: int = 6


class StreamGapPlannerAdapter:
    """Wait-for-gap and commit planner for lateral pedestrian streams."""

    def __init__(self, config: StreamGapPlannerConfig | None = None) -> None:
        """Initialize adapter and commit-mode state."""
        self.config = config or StreamGapPlannerConfig()
        self._commit_steps_remaining = 0

    @staticmethod
    def _as_xy(values: Any) -> np.ndarray:
        arr = np.asarray([] if values is None else values, dtype=float)
        if arr.ndim != 2 or arr.shape[-1] != 2:
            return np.zeros((0, 2), dtype=float)
        return arr

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
        goal = observation.get("goal") if isinstance(observation.get("goal"), dict) else {}
        pedestrians = (
            observation.get("pedestrians")
            if isinstance(observation.get("pedestrians"), dict)
            else {}
        )

        robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
        if robot_pos.size < 2:
            robot_pos = np.zeros(2, dtype=float)
        heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])

        goal_next = np.asarray(goal.get("next", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
        goal_current = np.asarray(goal.get("current", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
        goal_pos = goal_next if goal_next.size == 2 else np.zeros(2, dtype=float)
        if goal_pos.size != 2 or np.linalg.norm(goal_pos - robot_pos) <= 1e-6:
            goal_pos = goal_current if goal_current.size == 2 else np.zeros(2, dtype=float)

        ped_pos = self._as_xy(pedestrians.get("positions"))
        ped_vel = self._as_xy(pedestrians.get("velocities"))
        if ped_vel.shape != ped_pos.shape:
            ped_vel = np.zeros_like(ped_pos)

        count_arr = np.asarray(pedestrians.get("count", [ped_pos.shape[0]]), dtype=float).reshape(
            -1
        )
        count = int(count_arr[0]) if count_arr.size else ped_pos.shape[0]
        count = max(0, min(count, ped_pos.shape[0]))
        return robot_pos, heading, goal_pos, ped_pos[:count], ped_vel[:count]

    def _goal_frame(
        self,
        *,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        goal_vec = goal_pos - robot_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = goal_vec / goal_dist
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        rel = ped_pos - robot_pos[None, :]
        along = rel @ direction
        cross = rel @ lateral
        vel_along = ped_vel @ direction
        vel_cross = ped_vel @ lateral
        return along, cross, np.stack([vel_along, vel_cross], axis=1)

    def _blocked_mask(
        self,
        *,
        along: np.ndarray,
        cross: np.ndarray,
    ) -> np.ndarray:
        return (
            (along >= -float(self.config.rear_margin))
            & (along <= float(self.config.forward_lookahead))
            & (np.abs(cross) <= float(self.config.corridor_half_width))
        )

    def _gap_start_time(
        self,
        *,
        along: np.ndarray,
        cross: np.ndarray,
        vel_goal_frame: np.ndarray,
    ) -> tuple[float | None, bool, float]:
        dt = max(float(self.config.sample_dt), 1e-3)
        horizon = max(float(self.config.sample_horizon), dt)
        times = np.arange(0.0, horizon + dt * 0.5, dt, dtype=float)
        blocked = np.zeros(times.shape[0], dtype=bool)

        if along.size == 0:
            return 0.0, False, float("inf")

        min_distance = float("inf")
        for idx, t in enumerate(times):
            along_t = along + vel_goal_frame[:, 0] * t
            cross_t = cross + vel_goal_frame[:, 1] * t
            inside = self._blocked_mask(along=along_t, cross=cross_t)
            blocked[idx] = bool(np.any(inside))
            if inside.any():
                min_distance = min(
                    min_distance,
                    float(np.min(np.sqrt(along_t[inside] ** 2 + cross_t[inside] ** 2))),
                )

        free_steps_needed = max(1, int(np.ceil(float(self.config.safe_gap_time) / dt)))
        for start in range(blocked.shape[0]):
            end = start + free_steps_needed
            if end > blocked.shape[0]:
                break
            if not bool(np.any(blocked[start:end])):
                return float(times[start]), bool(blocked[0]), min_distance
        return None, bool(blocked[0]), min_distance

    def _heading_command(
        self, robot_pos: np.ndarray, heading: float, goal_pos: np.ndarray
    ) -> tuple[float, float]:
        goal_heading = float(np.arctan2(goal_pos[1] - robot_pos[1], goal_pos[0] - robot_pos[0]))
        heading_err = _wrap_angle(goal_heading - heading)
        angular = float(
            np.clip(
                float(self.config.heading_gain) * heading_err,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        return heading_err, angular

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a unicycle command using wait/approach/commit gap logic."""
        robot_pos, heading, goal_pos, ped_pos, ped_vel = self._extract_state(observation)
        goal_dist = float(np.linalg.norm(goal_pos - robot_pos))
        if goal_dist <= float(self.config.goal_tolerance):
            self._commit_steps_remaining = 0
            return 0.0, 0.0

        heading_err, angular = self._heading_command(robot_pos, heading, goal_pos)
        if abs(heading_err) >= float(self.config.turn_in_place_angle):
            self._commit_steps_remaining = 0
            return 0.0, angular

        along, cross, vel_goal_frame = self._goal_frame(
            robot_pos=robot_pos,
            goal_pos=goal_pos,
            ped_pos=ped_pos,
            ped_vel=ped_vel,
        )
        gap_start, blocked_now, min_distance = self._gap_start_time(
            along=along,
            cross=cross,
            vel_goal_frame=vel_goal_frame,
        )

        if np.isfinite(min_distance) and min_distance <= float(self.config.emergency_clearance):
            self._commit_steps_remaining = 0
            return float(self.config.wait_speed), angular

        if self._commit_steps_remaining > 0 and not blocked_now:
            self._commit_steps_remaining -= 1
            return float(self.config.commit_speed), angular

        if gap_start == 0.0:
            self._commit_steps_remaining = max(int(self.config.commit_hold_steps) - 1, 0)
            return float(self.config.commit_speed), angular

        if gap_start is not None and gap_start <= float(self.config.approach_gap_time):
            self._commit_steps_remaining = 0
            return float(self.config.approach_speed), angular

        if blocked_now:
            self._commit_steps_remaining = 0
            return float(self.config.wait_speed), angular

        self._commit_steps_remaining = 0
        return float(self.config.creep_speed), angular


def build_stream_gap_config(cfg: dict[str, Any] | None) -> StreamGapPlannerConfig:
    """Build :class:`StreamGapPlannerConfig` from a mapping.

    Returns:
        StreamGapPlannerConfig: Parsed planner configuration.
    """
    if not isinstance(cfg, dict):
        return StreamGapPlannerConfig()
    return StreamGapPlannerConfig(
        max_linear_speed=float(cfg.get("max_linear_speed", 1.2)),
        max_angular_speed=float(cfg.get("max_angular_speed", 1.2)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        heading_gain=float(cfg.get("heading_gain", 1.6)),
        turn_in_place_angle=float(cfg.get("turn_in_place_angle", 0.7)),
        forward_lookahead=float(cfg.get("forward_lookahead", 4.0)),
        rear_margin=float(cfg.get("rear_margin", 0.5)),
        corridor_half_width=float(cfg.get("corridor_half_width", 0.85)),
        emergency_clearance=float(cfg.get("emergency_clearance", 0.55)),
        sample_dt=float(cfg.get("sample_dt", 0.2)),
        sample_horizon=float(cfg.get("sample_horizon", 4.0)),
        safe_gap_time=float(cfg.get("safe_gap_time", 1.0)),
        approach_gap_time=float(cfg.get("approach_gap_time", 0.8)),
        wait_speed=float(cfg.get("wait_speed", 0.0)),
        creep_speed=float(cfg.get("creep_speed", 0.12)),
        approach_speed=float(cfg.get("approach_speed", 0.35)),
        commit_speed=float(cfg.get("commit_speed", 0.95)),
        commit_hold_steps=int(cfg.get("commit_hold_steps", 6)),
    )


__all__ = ["StreamGapPlannerAdapter", "StreamGapPlannerConfig", "build_stream_gap_config"]
