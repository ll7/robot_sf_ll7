"""Testing-only corridor-commitment planner inspired by TEB-like local optimization."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import atan2, pi
from typing import Any

import numpy as np

from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


def _wrap_angle(angle: float) -> float:
    """Wrap angle to ``[-pi, pi]``.

    Returns:
        float: Wrapped angle in radians.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class TEBCommitmentConfig:
    """Config for the native corridor-commitment local planner."""

    max_linear_speed: float = 0.9
    max_angular_speed: float = 1.2
    angular_gain: float = 1.4
    goal_tolerance: float = 0.25
    probe_distance: float = 1.0
    side_offset: float = 0.5
    occupancy_threshold: float = 0.5
    commit_gain: float = 0.6
    commit_persistence_steps: int = 10
    symmetry_bias: float = 0.1
    progress_epsilon: float = 0.03
    low_speed_threshold: float = 0.12
    clearance_speed_gain: float = 0.45


class TEBCommitmentPlannerAdapter(OccupancyAwarePlannerMixin):
    """Short-horizon corridor planner with side-choice commitment."""

    _EPS = 1e-6

    def __init__(self, config: TEBCommitmentConfig | None = None) -> None:
        """Initialize the planner with optional config overrides."""
        self.config = config or TEBCommitmentConfig()
        self.reset()

    def reset(self) -> None:
        """Clear per-episode progress and commitment state."""
        self._last_goal_distance: float | None = None
        self._last_goal: np.ndarray | None = None
        self._commit_side = 0
        self._commit_ttl = 0

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """Return a unit vector or zeros when the norm is near zero."""
        norm = float(np.linalg.norm(vec))
        if norm <= TEBCommitmentPlannerAdapter._EPS:
            return np.zeros(2, dtype=float)
        return vec / norm

    def _normalize_ped_positions(self, ped_state: dict[str, Any]) -> np.ndarray:
        """Return a sanitized ``(N, 2)`` pedestrian position array."""
        ped_positions_raw = ped_state.get("positions")
        ped_positions = np.asarray(
            [] if ped_positions_raw is None else ped_positions_raw, dtype=float
        )
        if ped_positions.ndim == 1 and ped_positions.size % 2 == 0:
            ped_positions = ped_positions.reshape(-1, 2)
        if ped_positions.ndim != 2 or ped_positions.shape[-1] != 2:
            return np.zeros((0, 2), dtype=float)
        count = int(self._as_1d_float(ped_state.get("count", [ped_positions.shape[0]]), pad=1)[0])
        count = max(0, min(count, ped_positions.shape[0]))
        return ped_positions[:count]

    def _forward_blocked(
        self, observation: dict[str, Any], robot_pos: np.ndarray, forward: np.ndarray
    ) -> bool:
        """Return whether the forward occupancy probe is blocked."""
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return False
        grid, meta = payload
        channel = self._grid_channel_index(meta, "obstacles")
        if channel < 0:
            channel = self._preferred_channel(meta)
        if channel < 0:
            return False
        center = robot_pos + forward * float(self.config.probe_distance)
        return self._grid_value(center, grid, meta, channel) >= float(
            self.config.occupancy_threshold
        )

    def _stalled(self, *, robot_speed: float, goal_distance: float) -> bool:
        """Return whether the robot is making too little progress at low speed."""
        if self._last_goal_distance is None:
            return False
        return robot_speed <= float(self.config.low_speed_threshold) and (
            self._last_goal_distance - goal_distance <= float(self.config.progress_epsilon)
        )

    def _command_from_heading(
        self, *, forward: np.ndarray, robot_heading: float, blocked: bool
    ) -> tuple[float, float]:
        """Project the chosen corridor heading into a bounded unicycle command.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        desired_heading = atan2(forward[1], forward[0])
        heading_error = _wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                float(self.config.angular_gain) * heading_error,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        linear = float(self.config.max_linear_speed)
        if blocked:
            linear *= float(self.config.clearance_speed_gain)
        linear *= max(0.2, 1.0 - min(abs(heading_error), pi / 2) / (pi / 2))
        linear = float(np.clip(linear, 0.0, float(self.config.max_linear_speed)))
        return linear, angular

    def _choose_side(
        self,
        *,
        robot_pos: np.ndarray,
        forward: np.ndarray,
        lateral: np.ndarray,
        ped_positions: np.ndarray,
        observation: dict[str, Any],
    ) -> int:
        if self._commit_ttl > 0:
            self._commit_ttl -= 1
        if self._commit_ttl > 0:
            return self._commit_side
        # Positive score means the left corridor looks more favorable than the right.
        score = float(self.config.symmetry_bias)
        if ped_positions.size:
            offsets = ped_positions - robot_pos[None, :]
            score += -float(np.sum(offsets @ lateral))
        payload = self._extract_grid_payload(observation)
        if payload is not None:
            grid, meta = payload
            channel = self._grid_channel_index(meta, "obstacles")
            if channel < 0:
                channel = self._preferred_channel(meta)
            if channel >= 0:
                left = robot_pos + forward * float(self.config.probe_distance)
                left = left + lateral * float(self.config.side_offset)
                right = robot_pos + forward * float(self.config.probe_distance)
                right = right - lateral * float(self.config.side_offset)
                score += self._grid_value(right, grid, meta, channel)
                score -= self._grid_value(left, grid, meta, channel)
        self._commit_side = 1 if score >= 0.0 else -1
        self._commit_ttl = max(int(self.config.commit_persistence_steps) - 1, 0)
        return self._commit_side

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a committed corridor-following ``(v, w)`` command.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        robot_speed = float(self._as_1d_float(robot_state.get("speed", [0.0]), pad=1)[0])
        goal = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        if self._last_goal is not None and np.linalg.norm(goal - self._last_goal) > self._EPS:
            self._last_goal_distance = None
            self._commit_side = 0
            self._commit_ttl = 0
        self._last_goal = goal.copy()
        goal_delta = goal - robot_pos
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        ped_positions = self._normalize_ped_positions(ped_state)

        forward = self._normalize(goal_delta)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        blocked = self._forward_blocked(observation, robot_pos, forward)
        stalled = self._stalled(robot_speed=robot_speed, goal_distance=goal_distance)
        self._last_goal_distance = goal_distance

        if blocked or stalled:
            side = self._choose_side(
                robot_pos=robot_pos,
                forward=forward,
                lateral=lateral,
                ped_positions=ped_positions,
                observation=observation,
            )
            forward = self._normalize(forward + lateral * side * float(self.config.commit_gain))

        return self._command_from_heading(
            forward=forward, robot_heading=robot_heading, blocked=blocked
        )


def build_teb_commitment_config(cfg: dict[str, Any] | None) -> TEBCommitmentConfig:
    """Build the native TEB-style config from a mapping payload.

    Returns:
        TEBCommitmentConfig: Parsed planner config.
    """
    if not isinstance(cfg, dict):
        return TEBCommitmentConfig()
    defaults = TEBCommitmentConfig()
    numeric_casts = {
        "max_linear_speed": float,
        "max_angular_speed": float,
        "angular_gain": float,
        "goal_tolerance": float,
        "probe_distance": float,
        "side_offset": float,
        "occupancy_threshold": float,
        "commit_gain": float,
        "commit_persistence_steps": int,
        "symmetry_bias": float,
        "progress_epsilon": float,
        "low_speed_threshold": float,
        "clearance_speed_gain": float,
    }
    kwargs = {}
    for field in fields(TEBCommitmentConfig):
        value = cfg.get(field.name, getattr(defaults, field.name))
        caster = numeric_casts.get(field.name)
        kwargs[field.name] = caster(value) if caster is not None else value
    return TEBCommitmentConfig(**kwargs)


__all__ = [
    "TEBCommitmentConfig",
    "TEBCommitmentPlannerAdapter",
    "build_teb_commitment_config",
]
