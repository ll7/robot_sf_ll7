"""TODO docstring. Document this module."""

from dataclasses import dataclass, field
from math import atan2, cos, sin, tan

import numpy as np
from gymnasium import spaces

from robot_sf.common.types import BicycleAction, PolarVec2D, RobotPose, Vec2D


@dataclass
class BicycleDriveSettings:
    """
    A class that defines the settings for a bicycle drive robot.
    """

    radius: float = 1.0  # Collision radius, not relevant for kinematics
    wheelbase: float = 1.0  # Distance between front and rear wheels
    max_steer: float = 0.78  # Maximum steering angle (45 degrees in radians)
    max_velocity: float = 3.0  # Maximum forward velocity
    max_accel: float = 1.0  # Maximum acceleration
    allow_backwards: bool = False  # Whether backwards movement is allowed

    @property
    def min_velocity(self) -> float:
        """
        Get the minimum velocity of the robot.

        If backwards movement is allowed, the minimum velocity is -max_velocity.
        Otherwise, the minimum velocity is 0.
        """
        return -self.max_velocity if self.allow_backwards else 0.0


@dataclass
class BicycleDriveState:
    """A class that represents the state of a bicycle drive robot."""

    pose: RobotPose
    velocity: float = field(default=0)

    @property
    def pos(self) -> Vec2D:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.pose[0]

    @property
    def orient(self) -> float:
        """Get the orientation of the robot in radians."""
        return self.pose[1]

    @property
    def current_speed(self) -> PolarVec2D:
        """Get the current speed and orientation of the robot."""
        return (self.velocity, self.orient)


@dataclass
class BicycleMotion:
    """An implementation of the bicycle model for motion
    with front and rear wheels for modeling e.g. an e-scooter.

    Snippet taken from https://github.com/winstxnhdw/KinematicBicycleModel."""

    config: BicycleDriveSettings

    def move(self, state: BicycleDriveState, action: BicycleAction, d_t: float):
        """Update the bicycle state by applying the provided action for ``d_t`` seconds.

        Args:
            state (BicycleDriveState): The current bicycle state (mutated in place).
            action (BicycleAction): Tuple with acceleration and steering commands.
            d_t (float): Duration in seconds for which the action is applied.

        Notes:
            This method mutates ``state`` directly and does not return a value.
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
        angular_velocity = new_velocity * tan(steering_angle) / self.config.wheelbase

        # Update position coordinates based on current orientation and speed
        new_x = x + velocity * cos(orient) * d_t
        new_y = y + velocity * sin(orient) * d_t

        # Normalize new orientation to ensure it stays within the valid range
        new_orient = self._norm_angle(orient + angular_velocity * d_t)

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
class BicycleDriveRobot:
    """Representing a robot with bicycle driving behavior"""

    config: BicycleDriveSettings
    state: BicycleDriveState = field(
        default_factory=lambda: BicycleDriveState(pose=((0.0, 0.0), 0.0)),
    )
    movement: BicycleMotion = field(init=False)

    def __post_init__(self):
        """TODO docstring. Document this function."""
        self.movement = BicycleMotion(self.config)

    @property
    def observation_space(self) -> spaces.Box:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        high = np.array([self.config.max_velocity, self.config.max_steer], dtype=np.float32)
        low = np.array([self.config.min_velocity, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        high = np.array([self.config.max_accel, self.config.max_steer], dtype=np.float32)
        low = np.array([-self.config.max_accel, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def pos(self) -> Vec2D:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.state.pose[0]

    @property
    def pose(self) -> RobotPose:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.state.pose

    @property
    def current_speed(self) -> PolarVec2D:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.state.current_speed

    def apply_action(self, action: BicycleAction, d_t: float):
        """TODO docstring. Document this function.

        Args:
            action: TODO docstring.
            d_t: TODO docstring.
        """
        self.movement.move(self.state, action, d_t)

    def reset_state(self, new_pose: RobotPose):
        """TODO docstring. Document this function.

        Args:
            new_pose: TODO docstring.
        """
        self.state = BicycleDriveState(new_pose, 0)

    def parse_action(self, action: np.ndarray) -> BicycleAction:
        """TODO docstring. Document this function.

        Args:
            action: TODO docstring.

        Returns:
            TODO docstring.
        """
        return (action[0], action[1])
