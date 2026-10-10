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

from robot_sf.common.math_utils import clip_scalar
from robot_sf.common.math_utils import wrap_angle_pi as _wrap_angle

if TYPE_CHECKING:
    from robot_sf.common.types import RobotPose


@dataclass
class DiffDriveAdapterConfig:
    """Configuration for holonomic-to-diff-drive action conversion."""

    angular_gain: float = 1.5
    heading_slowdown: float = 0.6
    min_speed: float = 1e-4
    allow_backwards: bool = False


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
        clip_scalar(cfg.angular_gain * heading_error, -max_angular_speed, max_angular_speed),
    )
    slowdown = max(0.0, 1.0 - cfg.heading_slowdown * abs(heading_error) / pi)
    linear = speed * slowdown
    if cfg.allow_backwards and abs(heading_error) > (pi / 2):
        linear *= -1.0
    if cfg.allow_backwards:
        linear = float(clip_scalar(linear, -max_linear_speed, max_linear_speed))
    else:
        linear = float(clip_scalar(linear, 0.0, max_linear_speed))

    return np.array([linear, angular], dtype=float)


__all__ = ["DiffDriveAdapterConfig", "holonomic_to_diff_drive_action"]


def ppo_delta_to_velocity_target(
    action: np.ndarray,
    current_speed: np.ndarray,
    *,
    max_linear_speed: float,
    max_angular_speed: float,
) -> np.ndarray:
    """Sum signed PPO velocity deltas before applying the no-reverse target limits.

    Deltas are velocities per policy step, not accelerations and not multiplied by dt.
    The plant subsequently enforces its acceleration limits.

    Returns:
        The bounded target linear and angular velocity.
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    speed = np.asarray(current_speed, dtype=float).reshape(-1)
    if action.size != 2 or not np.all(np.isfinite(action)):
        raise ValueError("velocity_delta requires two finite policy outputs")
    if speed.size != 2 or not np.all(np.isfinite(speed)):
        raise ValueError("velocity_delta requires finite current robot_speed (v, omega)")
    return np.clip(speed + action, [0.0, -max_angular_speed], [max_linear_speed, max_angular_speed])


def unicycle_velocity_target_to_acceleration(
    target: np.ndarray,
    current_speed: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Convert a velocity target to the drive request before plant acceleration clipping.

    Returns:
        Requested linear and angular accelerations for this step.

    Raises:
        ValueError: If the timestep is nonfinite or nonpositive.
    """
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestep must be finite and positive")
    return (np.asarray(target, dtype=float) - np.asarray(current_speed, dtype=float)) / dt
