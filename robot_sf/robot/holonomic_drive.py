"""Holonomic robot model supporting ``vx/vy`` and ``v/omega`` command modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from robot_sf.common.types import PolarVec2D, RobotPose

_COMMAND_MODES = {"vx_vy", "unicycle_vw"}


@dataclass
class HolonomicDriveSettings:
    """Configuration settings for a holonomic robot."""

    radius: float = 1.0
    max_speed: float = 2.0
    max_angular_speed: float = 1.0
    command_mode: str = "vx_vy"

    def __post_init__(self) -> None:
        """Validate kinematic limits and command mode."""
        if self.radius <= 0:
            raise ValueError("Holonomic robot radius must be positive.")
        if self.max_speed <= 0:
            raise ValueError("Holonomic robot max_speed must be positive.")
        if self.max_angular_speed <= 0:
            raise ValueError("Holonomic robot max_angular_speed must be positive.")
        mode = str(self.command_mode).strip().lower()
        if mode not in _COMMAND_MODES:
            raise ValueError(
                f"Holonomic command_mode must be one of {sorted(_COMMAND_MODES)}, got '{mode}'."
            )
        self.command_mode = mode


@dataclass
class HolonomicDriveState:
    """State for a holonomic robot."""

    pose: RobotPose = ((0.0, 0.0), 0.0)
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    velocity_vw: PolarVec2D = (0.0, 0.0)


@dataclass
class HolonomicDriveRobot:
    """Holonomic robot with dual control semantics."""

    config: HolonomicDriveSettings
    state: HolonomicDriveState = field(default_factory=HolonomicDriveState)

    @property
    def observation_space(self) -> spaces.Box:
        """Return observation bounds for velocity and angular speed."""
        high = np.array(
            [self.config.max_speed, self.config.max_speed, self.config.max_angular_speed],
            dtype=np.float32,
        )
        low = -high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Return action-space bounds for configured holonomic command mode."""
        if self.config.command_mode == "vx_vy":
            high = np.array([self.config.max_speed, self.config.max_speed], dtype=np.float32)
        else:
            high = np.array(
                [self.config.max_speed, self.config.max_angular_speed], dtype=np.float32
            )
        low = -high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def pos(self) -> tuple[float, float]:
        """Return current Cartesian position."""
        return self.state.pose[0]

    @property
    def pose(self) -> tuple[tuple[float, float], float]:
        """Return current pose ``((x, y), heading)``."""
        return self.state.pose

    @property
    def current_speed(self) -> tuple[float, float]:
        """Return current ``(linear_speed, angular_speed)``."""
        return self.state.velocity_vw

    def apply_action(self, action: tuple[float, float], d_t: float) -> None:
        """Apply one holonomic action for ``d_t`` seconds."""
        x, y = self.state.pose[0]
        heading = float(self.state.pose[1])
        dt = max(float(d_t), 1e-9)

        cmd_x = float(action[0])
        cmd_y = float(action[1])
        if self.config.command_mode == "vx_vy":
            vx = float(np.clip(cmd_x, -self.config.max_speed, self.config.max_speed))
            vy = float(np.clip(cmd_y, -self.config.max_speed, self.config.max_speed))
            speed = float(np.hypot(vx, vy))
            if speed > self.config.max_speed:
                scale = self.config.max_speed / max(speed, 1e-9)
                vx *= scale
                vy *= scale
            omega = 0.0
            if speed > 1e-9:
                heading = float(np.arctan2(vy, vx))
        else:
            v = float(np.clip(cmd_x, -self.config.max_speed, self.config.max_speed))
            omega = float(
                np.clip(cmd_y, -self.config.max_angular_speed, self.config.max_angular_speed)
            )
            heading = float((heading + omega * dt + np.pi) % (2.0 * np.pi) - np.pi)
            vx = v * cos(heading)
            vy = v * sin(heading)

        self.state.pose = ((x + vx * dt, y + vy * dt), heading)
        self.state.velocity_xy = (vx, vy)
        self.state.velocity_vw = (float(np.hypot(vx, vy)), omega)

    def reset_state(self, new_pose: tuple[tuple[float, float], float]) -> None:
        """Reset the robot state to a new pose and zero velocity."""
        self.state = HolonomicDriveState(pose=new_pose)

    def parse_action(self, action: np.ndarray) -> tuple[float, float]:
        """Parse a NumPy action array into a ``(float, float)`` tuple.

        Returns:
            tuple[float, float]: Parsed action tuple.
        """
        return (float(action[0]), float(action[1]))
