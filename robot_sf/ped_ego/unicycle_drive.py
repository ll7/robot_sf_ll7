"""This module describes a unicycle drive pedestrian model."""

from dataclasses import dataclass, field
from math import cos, sin

import numpy as np
from gymnasium import spaces

from robot_sf.common.math_utils import clip_scalar, normalize_angle_atan2
from robot_sf.common.types import PedPose, PolarVec2D, UnicycleAction, Vec2D


@dataclass
class UnicycleDriveSettings:
    """
    A class that defines the settings for a unicycle drive pedestrian.
    """

    radius: float = 0.4  # Collision radius, not relevant for kinematics
    max_steer: float = 0.78  # Maximum angular velocity command, kept for compatibility
    max_velocity: float = 3.0  # Maximum forward velocity

    max_accel: float = 1.0  # Maximum acceleration
    allow_backwards: bool = False  # Whether backwards movement is allowed

    @property
    def min_velocity(self) -> float:
        """
        Get the minimum velocity of the pedestrian.

        If backwards movement is allowed, the minimum velocity is -max_velocity.
        Otherwise, the minimum velocity is 0.
        """
        return -self.max_velocity if self.allow_backwards else 0.0


@dataclass
class UnicycleDriveState:
    """A class that represents the state of a unicycle drive pedestrian."""

    pose: PedPose = field(default_factory=lambda: ((0.0, 0.0), 0.0))
    velocity: float = 0.0

    @property
    def pos(self) -> Vec2D:
        """Get the position of the pedestrian.


        Returns:
            Vec2D: The (x, y) position of the pedestrian.
        """
        return self.pose[0]

    @property
    def orient(self) -> float:
        """Get the orientation of the pedestrian in radians."""
        return self.pose[1]

    @property
    def current_speed(self) -> PolarVec2D:
        """Get the current speed and orientation of the pedestrian."""
        return (self.velocity, self.orient)


@dataclass
class UnicycleMotion:
    """An implementation of the unicycle model for pedestrian movement.

    based on: https://msl.cs.uiuc.edu/planning/node660.html"""

    config: UnicycleDriveSettings

    def move(self, state: UnicycleDriveState, action: UnicycleAction, d_t: float):
        """Update the unicycle state by applying the provided action for ``d_t`` seconds.

        Args:
            state (UnicycleDriveState): The current unicycle state (mutated in place).
            action (UnicycleAction): Tuple with acceleration and angular velocity commands.
            d_t (float): Duration in seconds for which the action is applied.

        Notes:
            This method mutates ``state`` directly and does not return a value.
        """
        acceleration, angular_velocity = action
        (x, y), orient = state.pose
        velocity = state.velocity

        # Apply limits to the acceleration and calculate new velocity
        acceleration = clip_scalar(acceleration, -self.config.max_accel, self.config.max_accel)
        new_velocity = velocity + d_t * acceleration
        new_velocity = clip_scalar(new_velocity, self.config.min_velocity, self.config.max_velocity)

        # Apply limits to the angular velocity command.
        angular_velocity = clip_scalar(
            angular_velocity,
            -self.config.max_steer,
            self.config.max_steer,
        )

        # Normalize new orientation to ensure it stays within the valid range
        new_orient = normalize_angle_atan2(orient + angular_velocity * d_t)

        # Update position coordinates based on current orientation and speed
        # Use the updated forward speed (new_velocity) for translational motion
        v = float(new_velocity)
        new_x = x + v * cos(orient) * d_t
        new_y = y + v * sin(orient) * d_t

        # Update state with new pose and velocity
        state.pose = ((new_x, new_y), new_orient)
        state.velocity = float(new_velocity)


@dataclass
class UnicycleDrivePedestrian:
    """Representing a pedestrian with unicycle driving behavior"""

    config: UnicycleDriveSettings
    state: UnicycleDriveState = field(default_factory=UnicycleDriveState)
    movement: UnicycleMotion = field(init=False)

    def __post_init__(self):
        """Initialize the UnicycleMotion"""
        self.movement = UnicycleMotion(self.config)

    @property
    def observation_space(self) -> spaces.Box:
        """Action-independent observation bounds for the unicycle pedestrian.

        Returns:
            spaces.Box: 2D continuous box for speed and angular-velocity constraints.
        """
        high = np.array([self.config.max_velocity, self.config.max_steer], dtype=np.float32)
        low = np.array([self.config.min_velocity, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Action bounds accepted by the unicycle pedestrian.

        Returns:
            spaces.Box: 2D continuous box with acceleration and angular-velocity limits.
        """
        high = np.array([self.config.max_accel, self.config.max_steer], dtype=np.float32)
        low = np.array([-self.config.max_accel, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def pos(self) -> Vec2D:
        """Current 2D position of the pedestrian in world coordinates."""
        return self.state.pose[0]

    @property
    def pose(self) -> PedPose:
        """Current pose as ``((x, y), heading)``."""
        return self.state.pose

    @property
    def current_speed(self) -> PolarVec2D:
        """Current speed (m/s) and travel direction."""
        return self.state.current_speed

    def apply_action(self, action: UnicycleAction, d_t: float):
        """Integrate one control command for the configured time step.

        Args:
            action: Unicycle control tuple ``(acceleration, angular velocity)``.
            d_t: Integration duration in seconds.
        """
        self.movement.move(self.state, action, d_t)

    def reset_state(self, new_pose: PedPose):
        """Reset internal state to a pose and zero speed.

        Args:
            new_pose: New pedestrian pose used to reinitialize state.
        """
        self.state = UnicycleDriveState(new_pose, 0)

    def parse_action(self, action: np.ndarray) -> UnicycleAction:
        """Convert a 2-element action array to the unicycle action tuple.

        Args:
            action: Array-like with ``[acceleration, angular velocity]``.

        Returns:
            UnicycleAction: Parsed acceleration and angular velocity command tuple.
        """
        return (action[0], action[1])
