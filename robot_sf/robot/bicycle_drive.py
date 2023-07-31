from math import atan2, sin, cos, tan
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from gym import spaces


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


@dataclass
class BicycleDriveSettings:
    wheelbase: float=1.0
    max_steer: float=0.78 # 45 deg
    max_velocity: float=3.0
    max_accel: float=1.0
    allow_backwards: bool=False
    radius: float=1.0

    @property
    def min_velocity(self) -> float:
        return -self.max_velocity if self.allow_backwards else 0.0


@dataclass
class BicycleDriveState:
    pose: RobotPose
    velocity: float

    @property
    def pos(self) -> Vec2D:
        return self.pose[0]

    @property
    def orient(self) -> float:
        return self.pose[1]

    @property
    def current_speed(self) -> PolarVec2D:
        return (self.velocity, self.orient)


BicycleAction = Tuple[float, float] # (acceleration, steering angle)


@dataclass
class BicycleMotion:
    """An implementation of the bicycle model for motion
    with front and rear wheels for modeling e.g. an e-scooter.

    Snippet taken from https://github.com/winstxnhdw/KinematicBicycleModel."""

    config: BicycleDriveSettings

    def move(self, state: BicycleDriveState, action: BicycleAction, d_t: float):
        acceleration, steering_angle = action
        (x, y), orient = state.pose
        velocity = state.velocity

        acceleration = np.clip(acceleration, -self.config.max_accel, self.config.max_accel)
        new_velocity = velocity + d_t * acceleration
        new_velocity = np.clip(new_velocity, self.config.min_velocity, self.config.max_velocity)
        steering_angle = np.clip(steering_angle, -self.config.max_steer, self.config.max_steer)
        angular_velocity = new_velocity * tan(steering_angle) / self.config.wheelbase

        new_x = x + velocity * cos(orient) * d_t
        new_y = y + velocity * sin(orient) * d_t
        new_orient = self._norm_angle(orient + angular_velocity * d_t)

        state.pose = ((new_x, new_y), new_orient)
        state.velocity = new_velocity

    def _norm_angle(self, angle: float) -> float:
        return atan2(sin(angle), cos(angle))


@dataclass
class BicycleDriveRobot():
    """Representing a robot with bicycle driving behavior"""

    config: BicycleDriveSettings
    state: BicycleDriveState = field(default=BicycleDriveState(((0, 0), 0), 0))
    movement: BicycleMotion = field(init=False)

    def __post_init__(self):
        self.movement = BicycleMotion(self.config)

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
    def pose(self) -> RobotPose:
        return self.state.pose

    @property
    def current_speed(self) -> PolarVec2D:
        return self.state.current_speed

    def apply_action(self, action: BicycleAction, d_t: float):
        self.movement.move(self.state, action, d_t)

    def reset_state(self, new_pose: RobotPose):
        self.state = BicycleDriveState(new_pose, 0)

    def parse_action(self, action: np.ndarray) -> BicycleAction:
        return (action[0], action[1])
