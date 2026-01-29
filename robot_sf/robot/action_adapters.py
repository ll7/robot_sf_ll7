"""Action-space adapters for mapping planner outputs into robot commands.

This module provides small, reusable conversions between holonomic velocity
commands (vx, vy) and non-holonomic differential-drive actions (v, omega).
The adapters are intentionally lightweight and can be swapped out or tuned
for specific controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_sf.common.types import RobotPose


@dataclass
class DiffDriveAdapterConfig:
    """Configuration for holonomic-to-diff-drive action conversion."""

    angular_gain: float = 1.5
    heading_slowdown: float = 0.6
    min_speed: float = 1e-4
    allow_backwards: bool = False


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi].

    Returns:
        float: Wrapped angle in radians.
    """
    return (angle + pi) % (2 * pi) - pi


def holonomic_to_diff_drive_action(
    velocity: np.ndarray,
    pose: RobotPose,
    *,
    max_linear_speed: float,
    max_angular_speed: float,
    config: DiffDriveAdapterConfig | None = None,
) -> np.ndarray:
    """Convert holonomic velocity into a differential-drive (v, omega) action.

    Args:
        velocity: Holonomic velocity vector (vx, vy) in world coordinates.
        pose: Robot pose ``((x, y), heading)`` in world coordinates.
        max_linear_speed: Robot maximum linear speed.
        max_angular_speed: Robot maximum angular speed.
        config: Optional adapter configuration.

    Returns:
        np.ndarray: Differential-drive action ``[v, omega]``.
    """
    cfg = config or DiffDriveAdapterConfig()
    velocity = np.asarray(velocity, dtype=float).reshape(2)
    speed = float(np.linalg.norm(velocity))
    if speed < cfg.min_speed:
        return np.zeros(2, dtype=float)

    heading = float(pose[1])
    desired_heading = atan2(velocity[1], velocity[0])
    heading_error = _wrap_angle(desired_heading - heading)

    angular = float(
        np.clip(cfg.angular_gain * heading_error, -max_angular_speed, max_angular_speed),
    )
    slowdown = max(0.0, 1.0 - cfg.heading_slowdown * abs(heading_error) / pi)
    linear = speed * slowdown
    if cfg.allow_backwards and abs(heading_error) > (pi / 2):
        linear *= -1.0
    if cfg.allow_backwards:
        linear = float(np.clip(linear, -max_linear_speed, max_linear_speed))
    else:
        linear = float(np.clip(linear, 0.0, max_linear_speed))

    return np.array([linear, angular], dtype=float)


__all__ = ["DiffDriveAdapterConfig", "holonomic_to_diff_drive_action"]
