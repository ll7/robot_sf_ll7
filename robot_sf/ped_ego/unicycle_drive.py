from math import atan2, sin, cos, tan
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces

from robot_sf.util.types import Vec2D, PolarVec2D, PedPose, UnicycleAction


@dataclass
class UnicycleDriveSettings:
    """
    A class that defines the settings for a unicycle drive pedestrian.
    """

    radius: float = 0.4  # Collision radius, not relevant for kinematics
    max_steer: float = 0.78  # Maximum steering angle (45 degrees in radians)
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
    velocity: PolarVec2D = field(default_factory=lambda: (0.0, 0.0))

    @property
    def pos(self) -> Vec2D:
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
        """
        Update the state of the unicycle given an action and time duration.

        Args:
            state (UnicycleDriveState): The current state of the unicycle.
            action (UnicycleAction): The action to take, contains acceleration
                                    and steering angle.
            d_t (float): The time duration for which to apply the action.

        Returns:
            None: The method updates the state in-place.
        """
        acceleration, steering_angle = action
        (x, y), orient = state.pose
        velocity = state.velocity

        # Apply limits to the acceleration and calculate new velocity
        acceleration = np.clip(acceleration, -self.config.max_accel, self.config.max_accel)
        new_velocity = velocity + d_t * acceleration
        new_velocity = np.clip(new_velocity, self.config.min_velocity, self.config.max_velocity)

        # Apply limits to the steering angle
        steering_angle = np.clip(steering_angle, -self.config.max_steer, self.config.max_steer)

        # Calculate angular velocity based on velocity and steering angle
        angular_velocity = new_velocity * tan(steering_angle)

        # Normalize new orientation to ensure it stays within the valid range
        new_orient = self._norm_angle(orient + angular_velocity * d_t)

        # Update position coordinates based on current orientation and speed
        new_x = x + velocity * cos(orient) * d_t
        new_y = y + velocity * sin(orient) * d_t

        # Update state with new pose and velocity
        state.pose = ((new_x, new_y), new_orient)
        state.velocity = new_velocity

    def _norm_angle(self, angle: float) -> float:
        """
        Normalize an angle to be within the range [-π, π].

        Args:
            angle (float): The angle to normalize.

        Returns:
            float: Normalized angle within range [-π, π].
        TODO: I think that this function is implemented in multiple places in the codebase.
        """
        return atan2(sin(angle), cos(angle))


@dataclass
class UnicycleDrivePedestrian:
    """Representing a pedestrian with unicycle driving behavior"""

    config: UnicycleDriveSettings
    state: UnicycleDriveState = field(default_factory=UnicycleDriveState)
    movement: UnicycleMotion = field(init=False)

    def __post_init__(self):
        self.movement = UnicycleMotion(self.config)

    @property
    def observation_space(self) -> spaces.Box:
        high = np.array([self.config.max_velocity, self.config.max_steer], dtype=np.float32)
        low = np.array([self.config.min_velocity, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        high = np.array([self.config.max_accel, self.config.max_steer], dtype=np.float32)
        low = np.array([-self.config.max_accel, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def pos(self) -> Vec2D:
        return self.state.pose[0]

    @property
    def pose(self) -> PedPose:
        return self.state.pose

    @property
    def current_speed(self) -> PolarVec2D:
        return self.state.current_speed

    def apply_action(self, action: UnicycleAction, d_t: float):
        self.movement.move(self.state, action, d_t)

    def reset_state(self, new_pose: PedPose):
        self.state = UnicycleDriveState(new_pose, 0)

    def parse_action(self, action: np.ndarray) -> UnicycleAction:
        return (action[0], action[1])
