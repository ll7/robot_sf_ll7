"""Bicycle drive model for vehicle dynamics simulation."""

from dataclasses import dataclass, field
from math import cos, sin, tan

import numpy as np
from gymnasium import spaces

from robot_sf.common.math_utils import clip_scalar, normalize_angle_atan2
from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS
from robot_sf.common.types import BicycleAction, PolarVec2D, RobotPose, Vec2D


@dataclass
class BicycleDriveSettings:
    """
    A class that defines the settings for a bicycle drive robot.
    """

    # Collision radius for physics/metrics, not used in kinematics.
    # References the authoritative default from robot_sf.common.robot_defaults.
    radius: float = DEFAULT_ROBOT_RADIUS
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
    angular_velocity: float = field(default=0)

    @property
    def pos(self) -> Vec2D:
        """Return the current 2D position.


        Returns:
            The ``(x, y)`` position component of the bicycle pose.
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

    @property
    def current_yaw_rate(self) -> float:
        """Return the last executed bicycle yaw rate in radians per second."""
        return self.angular_velocity


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
        acceleration = clip_scalar(acceleration, -self.config.max_accel, self.config.max_accel)
        new_velocity = velocity + d_t * acceleration
        new_velocity = clip_scalar(new_velocity, self.config.min_velocity, self.config.max_velocity)

        # Apply limits to the steering angle
        steering_angle = clip_scalar(steering_angle, -self.config.max_steer, self.config.max_steer)

        # Calculate angular velocity based on velocity and steering angle
        angular_velocity = new_velocity * tan(steering_angle) / self.config.wheelbase

        # Update position coordinates based on current orientation and speed
        new_x = x + velocity * cos(orient) * d_t
        new_y = y + velocity * sin(orient) * d_t

        # Normalize new orientation to ensure it stays within the valid range
        new_orient = normalize_angle_atan2(orient + angular_velocity * d_t)

        # Update state with new pose and velocity
        state.pose = ((new_x, new_y), new_orient)
        state.velocity = new_velocity
        state.angular_velocity = angular_velocity


@dataclass
class BicycleDriveRobot:
    """Representing a robot with bicycle driving behavior"""

    config: BicycleDriveSettings
    state: BicycleDriveState = field(
        default_factory=lambda: BicycleDriveState(pose=((0.0, 0.0), 0.0)),
    )
    movement: BicycleMotion = field(init=False)

    def __post_init__(self):
        """Create the motion helper bound to this robot's drive settings."""
        self.movement = BicycleMotion(self.config)

    @property
    def observation_space(self) -> spaces.Box:
        """Return the bounded observation space for velocity and steering state.

        Returns:
            A Box with lower and upper bounds derived from the configured velocity
            and steering limits.
        """
        high = np.array([self.config.max_velocity, self.config.max_steer], dtype=np.float32)
        low = np.array([self.config.min_velocity, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Return the bounded action space for acceleration and steering commands.

        Returns:
            A Box with lower and upper bounds derived from the configured
            acceleration and steering limits.
        """
        high = np.array([self.config.max_accel, self.config.max_steer], dtype=np.float32)
        low = np.array([-self.config.max_accel, -self.config.max_steer], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def pos(self) -> Vec2D:
        """Return the current 2D position of the robot.

        Returns:
            The ``(x, y)`` position from the current bicycle-drive pose.
        """
        return self.state.pose[0]

    @property
    def pose(self) -> RobotPose:
        """Return the full robot pose.

        Returns:
            The current ``((x, y), theta)`` pose tuple.
        """
        return self.state.pose

    @property
    def current_speed(self) -> PolarVec2D:
        """Return the robot's scalar speed paired with its current heading.

        Returns:
            The current speed/orientation tuple exposed by the drive state.
        """
        return self.state.current_speed

    @property
    def current_yaw_rate(self) -> float:
        """Return the last executed bicycle yaw rate in radians per second."""
        return self.state.current_yaw_rate

    def apply_action(self, action: BicycleAction, d_t: float):
        """Apply action and advance simulation by dt.

        Args:
            action: (velocity, steering_angle) tuple
            d_t: Time step duration
        """
        self.movement.move(self.state, action, d_t)

    def reset_state(self, new_pose: RobotPose):
        """Update vehicle state directly.

        Args:
            new_pose: New (x, y, theta) pose tuple
        """
        self.state = BicycleDriveState(new_pose, 0)

    def parse_action(self, action: np.ndarray) -> BicycleAction:
        """Convert a raw action array into the bicycle-drive action tuple.

        Args:
            action: Array-like acceleration and steering command.

        Returns:
            The ``(acceleration, steering_angle)`` tuple used by the drive model.
        """
        return (action[0], action[1])
