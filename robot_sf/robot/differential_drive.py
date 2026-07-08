"""Differential Drive Robot Model"""

from dataclasses import dataclass, field
from math import cos, sin

import numpy as np
from gymnasium import spaces

from robot_sf.common.math_utils import clip_scalar
from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS
from robot_sf.common.types import DifferentialDriveAction, PolarVec2D, RobotPose, Vec2D

WheelSpeedState = tuple[float, float]
"""Left and right wheel angular velocities in radians per second."""


@dataclass
class DifferentialDriveSettings:
    """
    Configuration settings for a differential drive robot, including physical
    characteristics and speed limitations.
    """

    # Robot collision radius used by physics/metrics (not kinematics or grid rasterization).
    # References the authoritative default from robot_sf.common.robot_defaults.
    radius: float = DEFAULT_ROBOT_RADIUS
    # Maximum linear velocity the robot can achieve
    max_linear_speed: float = 2.0
    # Maximum angular velocity the robot can achieve
    max_angular_speed: float = 1.0
    # Radius of the wheels (used to calculate rotations from linear movement)
    wheel_radius: float = 0.05
    # Distance between the centers of the two wheels of the robot
    interaxis_length: float = 0.3
    # Whether backwards motion is allowed (enables negative linear speed)
    allow_backwards: bool = False
    # Maximum linear acceleration (m/s^2) accepted as an action; the per-step
    # velocity delta is max_linear_accel * d_t (see issue #3711)
    max_linear_accel: float = 1.0
    # Maximum angular acceleration (rad/s^2) accepted as an action; the per-step
    # angular velocity delta is max_angular_accel * d_t (see issue #3711)
    max_angular_accel: float = 1.0

    def __post_init__(self):
        """
        Post-initialization processing to ensure valid configuration values.

        Raises:
            ValueError: If any of the provided settings are not within the
                        expected positive, non-zero ranges.
        """
        if self.radius <= 0:
            raise ValueError("Robot's radius must be positive and non-zero!")
        if self.wheel_radius <= 0:
            raise ValueError("Robot's wheel radius must be positive and non-zero!")
        if self.max_linear_speed <= 0 or self.max_angular_speed <= 0:
            raise ValueError(
                "Robot's max. linear and angular speeds must be positive and non-zero!",
            )
        if self.max_linear_accel <= 0 or self.max_angular_accel <= 0:
            raise ValueError(
                "Robot's max. linear and angular accelerations must be positive and non-zero!",
            )
        if self.interaxis_length <= 0:
            raise ValueError("Robot's interaxis length must be positive and non-zero!")

    @property
    def min_linear_speed(self) -> float:
        """Return the minimum linear speed based on allow_backwards."""
        return -self.max_linear_speed if self.allow_backwards else 0.0


@dataclass
class DifferentialDriveState:
    """Differential drive model for wheeled vehicles."""

    pose: RobotPose = ((0.0, 0.0), 0.0)
    velocity: PolarVec2D = (0.0, 0.0)
    last_wheel_speeds: WheelSpeedState = (0.0, 0.0)
    """Previous left/right wheel angular velocities in radians per second."""
    wheel_speeds: WheelSpeedState = (0.0, 0.0)
    """Current left/right wheel angular velocities in radians per second."""


@dataclass
class DifferentialDriveMotion:
    """
    Class providing functionality for simulating or controlling the motion
    of a differential drive robot based on its settings and current state.
    """

    config: DifferentialDriveSettings

    def move(self, state: DifferentialDriveState, action: PolarVec2D, d_t: float):
        """
        Updates the robot's state including position and velocity based on
        the action taken and elapsed time.

        :param state: The current state of the differential drive.
        :param action: The commanded linear/angular acceleration (a_linear, a_angular).
        :param d_t: The time elapsed since the last update in seconds.
        """
        robot_vel = self._robot_velocity(state.velocity, action, d_t)
        new_wheel_speeds = self._resulting_wheel_speeds(robot_vel)
        distance = self._covered_distance(state.wheel_speeds, new_wheel_speeds, d_t)
        new_orient = self._new_orientation(state.pose[1], state.wheel_speeds, new_wheel_speeds, d_t)
        state.pose = self._compute_odometry(state.pose, (distance, new_orient))
        state.last_wheel_speeds = state.wheel_speeds
        state.wheel_speeds = new_wheel_speeds
        state.velocity = robot_vel

    def _robot_velocity(self, velocity: PolarVec2D, action: PolarVec2D, d_t: float) -> PolarVec2D:
        """
        Computes the new polar velocity vector from the current velocity and the
        commanded acceleration integrated over the timestep.

        The action is interpreted as a commanded acceleration ``(a_linear,
        a_angular)``. It is first clipped to the configured acceleration limits,
        then integrated over ``d_t`` to obtain the velocity delta
        (``max_delta = max_accel * d_t``). Scaling by ``d_t`` keeps the velocity
        update timestep-invariant: the same acceleration command applied over the
        same elapsed time produces the same velocity change regardless of how the
        interval is subdivided. Without this scaling, smaller timesteps inflated
        the effective acceleration and produced unstable velocity spikes at high
        update frequencies (see issue #3711).

        Args:
            velocity: The current velocity of the robot (linear speed, angular speed).
            action: The commanded acceleration to apply (a_linear, a_angular).
            d_t: The time elapsed since the last update in seconds.

        Returns:
            PolarVec2D: The new velocity clipped by configured accel and speed limits.
        """
        linear_accel = clip_scalar(
            action[0], -self.config.max_linear_accel, self.config.max_linear_accel
        )
        angular_accel = clip_scalar(
            action[1], -self.config.max_angular_accel, self.config.max_angular_accel
        )
        # Integrate the commanded acceleration over the timestep so the per-step
        # velocity delta scales with d_t (max_delta = max_accel * d_t).
        dot_x = velocity[0] + linear_accel * d_t
        dot_orient = velocity[1] + angular_accel * d_t
        dot_x = clip_scalar(dot_x, self.config.min_linear_speed, self.config.max_linear_speed)
        angular_max = self.config.max_angular_speed
        dot_orient = clip_scalar(dot_orient, -angular_max, angular_max)
        return dot_x, dot_orient

    def _resulting_wheel_speeds(self, movement: PolarVec2D) -> WheelSpeedState:
        """
        Calculates the wheel speeds resulting from a given robot movement.

        Args:
            movement: The robot movement as a polar vector ``(speed, orientation)``.

        Returns:
            tuple[float, float]: The wheel speeds ``(left, right)`` in radians/sec.
        """
        dot_x, dot_orient = movement
        diff = self.config.interaxis_length * dot_orient / 2
        new_left_wheel_speed = (dot_x - diff) / self.config.wheel_radius
        new_right_wheel_speed = (dot_x + diff) / self.config.wheel_radius
        return new_left_wheel_speed, new_right_wheel_speed

    def _covered_distance(
        self,
        last_wheel_speeds: WheelSpeedState,
        new_wheel_speeds: WheelSpeedState,
        d_t: float,
    ) -> float:
        """
        Computes the distance covered by the robot over the time interval ``d_t``.

        Args:
            last_wheel_speeds: The previous left/right wheel angular velocities in rad/s.
            new_wheel_speeds: The updated left/right wheel angular velocities in rad/s.
            d_t: The time elapsed since last speeds measurement.

        Returns:
            float: The covered distance during ``d_t`` (meters).
        """
        avg_left_speed = (last_wheel_speeds[0] + new_wheel_speeds[0]) / 2
        avg_right_speed = (last_wheel_speeds[1] + new_wheel_speeds[1]) / 2
        average_velocity = (avg_left_speed + avg_right_speed) / 2
        # Linear velocity v = r * (wl + wr) / 2. Using trapezoidal integration yields
        # v_avg = r * average_velocity (where average_velocity = (wl + wr) / 2).
        # References: Dudek & Jenkin (CMP Robotics), Siegwart & Nourbakhsh (AMR).
        return self.config.wheel_radius * average_velocity * d_t

    def _new_orientation(
        self,
        robot_orient: float,
        last_wheel_speeds: WheelSpeedState,
        wheel_speeds: WheelSpeedState,
        d_t: float,
    ) -> float:
        """
        Determines the new orientation of the robot after updating its wheel speeds.

        Args:
            robot_orient: The prior orientation angle of the robot (radians).
            last_wheel_speeds: The previous left/right wheel angular velocities in rad/s.
            wheel_speeds: The new left/right wheel angular velocities in rad/s.
            d_t: Time elapsed (seconds).

        Returns:
            float: The updated orientation angle (radians).
        """
        last_wheel_speed_left, last_wheel_speed_right = last_wheel_speeds
        wheel_speed_left, wheel_speed_right = wheel_speeds

        # Angular velocity omega = r / L * (wr - wl). Use trapezoidal integration
        # over wheel speeds for the step.
        # References: Dudek & Jenkin (CMP Robotics), Siegwart & Nourbakhsh (AMR).
        right_left_diff = (last_wheel_speed_right + wheel_speed_right) / 2 - (
            last_wheel_speed_left + wheel_speed_left
        ) / 2
        diff = self.config.wheel_radius / self.config.interaxis_length * right_left_diff * d_t
        new_orient = robot_orient + diff
        return new_orient

    def _compute_odometry(self, old_pose: RobotPose, movement: PolarVec2D) -> RobotPose:
        """
        Computes and returns the new odometry of the robot.

        Args:
            old_pose: The previous pose of the robot ``((x, y), orientation)``.
            movement: The movement made by the robot ``(distance, new orientation)``.

        Returns:
            RobotPose: The updated pose ``((x, y), orientation)``.
        """
        distance_covered, new_orient = movement
        (robot_x, robot_y), old_orient = old_pose
        rel_rotation = (old_orient + new_orient) / 2
        new_x = robot_x + distance_covered * cos(rel_rotation)
        new_y = robot_y + distance_covered * sin(rel_rotation)
        return (new_x, new_y), new_orient


@dataclass
class DifferentialDriveRobot:
    """
    A robot with differential drive behavior that defines its movement,
    action and observation space.
    """

    config: DifferentialDriveSettings  # Configuration settings for the robot
    # Default state of the robot on initialization
    state: DifferentialDriveState = field(default_factory=DifferentialDriveState)
    # Movement logic for the robot based on the config; initialized post-creation
    movement: DifferentialDriveMotion = field(init=False)

    def __post_init__(self):
        """Initializes the movement behavior of the robot after class creation."""
        self.movement = DifferentialDriveMotion(self.config)

    @property
    def observation_space(self) -> spaces.Box:
        """
        Defines the observation space for the robot based on its configuration.

        Returns:
            Box: An instance of gymnasium.spaces.Box representing the continuous
                 observation space where each observation is a vector containing
                 linear and angular speeds.
        """
        high = np.array(
            [self.config.max_linear_speed, self.config.max_angular_speed],
            dtype=np.float32,
        )
        low = np.array(
            [self.config.min_linear_speed, -self.config.max_angular_speed],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """
        Defines the action space for the robot based on its configuration.

        Returns:
            Box: An instance of gymnasium.spaces.Box representing the continuous
            action space where each action is a vector containing commanded
            linear and angular accelerations (integrated over d_t per step).
        """
        high = np.array(
            [self.config.max_linear_accel, self.config.max_angular_accel],
            dtype=np.float32,
        )
        low = np.array(
            [-self.config.max_linear_accel, -self.config.max_angular_accel],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def pos(self) -> Vec2D:
        """
        Current position of the robot.

        Returns:
            Vec2D: The x-y coordinates representing the current position.
        """
        return self.state.pose[0]

    @property
    def pose(self) -> RobotPose:
        """
        Current pose (position and orientation) of the robot.

        Returns:
            RobotPose: The pose representing the robot's current position and
                orientation.
        """
        return self.state.pose

    @property
    def current_speed(self) -> PolarVec2D:
        """
        Current speed of the robot in polar coordinates (linear and angular).

        Returns:
            PolarVec2D: The current speed of the robot.
        """
        return self.state.velocity

    def apply_action(self, action: DifferentialDriveAction, d_t: float):
        """
        Applies an action to the robot over a time interval.

        Args:
            action: The action to be applied represented by commanded linear and
                    angular accelerations.
            d_t: The duration in seconds over which to apply the action.
        """
        self.movement.move(self.state, action, d_t)

    def reset_state(self, new_pose: RobotPose):
        """
        Resets the robot's state to a new pose.

        Args:
            new_pose: The new pose to set as the current state.
        """
        self.state = DifferentialDriveState(new_pose)

    def parse_action(self, action: np.ndarray) -> DifferentialDriveAction:
        """
        Parses a numpy array into a DifferentialDriveAction.

        Args:
            action: A numpy array containing linear and angular components.

        Returns:
            DifferentialDriveAction: The tuple with linear and angular velocity.
        """
        return (action[0], action[1])
