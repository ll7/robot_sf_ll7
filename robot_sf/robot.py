from math import dist, atan2, sin, cos
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[PolarVec2D, float]


def rel_pos(pose: RobotPose, target_coords: Vec2D) -> PolarVec2D:
    t_x, t_y = target_coords
    (r_x, r_y), orient = pose
    distance = dist(target_coords, (r_x, r_y))

    angle = atan2(t_y - r_y, t_x - r_x) - orient
    angle = (angle + np.pi) % (2 * np.pi) -np.pi
    return distance, angle


@dataclass
class WheelSpeedState:
    left: float
    right: float


@dataclass
class RobotSettings:
    radius: float
    max_linear_speed: float
    max_angular_speed: float
    rob_collision_radius: float = 0.7
    wheel_radius: float = 0.05
    interaxis_length: float = 0.3


@dataclass
class RobotState:
    config: RobotSettings
    current_speed: PolarVec2D
    current_pose: RobotPose
    last_wheels_speed: WheelSpeedState
    wheels_speed: WheelSpeedState

    # TODO: think of adding a markdown file describing what's happening here

    def resulting_movement(self, action: PolarVec2D) -> Tuple[PolarVec2D, bool]:
        dot_x = self.current_speed[0] + action[0]
        dot_orient = self.current_speed[1] + action[1]
        clipped = dot_x < 0 or dot_x > self.config.max_linear_speed or \
            abs(dot_orient) > self.config.max_angular_speed

        dot_x = np.clip(dot_x, 0, self.config.max_linear_speed)
        angular_max = self.config.max_angular_speed
        dot_orient = np.clip(dot_orient, -angular_max, angular_max)
        return (dot_x, dot_orient), clipped

    def update_robot_speed(self, movement: PolarVec2D):
        dot_x, dot_orient = movement
        diff = self.config.interaxis_length * dot_orient / 2
        self.wheels_speed.left = (dot_x - diff) / self.config.wheel_radius
        self.last_wheels_speed.right = (dot_x + diff) / self.config.wheel_radius
        self.current_speed = (dot_x, dot_orient)

    def compute_odometry(self, t_s: float):
        right_left_diff = -(self.last_wheels_speed.left + self.wheels_speed.left) / 2 \
            + (self.last_wheels_speed.right + self.wheels_speed.right) / 2

        new_orient = self.current_pose[1] \
            + self.config.wheel_radius / self.config.interaxis_length \
                * right_left_diff * t_s

        new_x_local = self.config.wheel_radius / 2 \
                * ((self.last_wheels_speed.left + self.wheels_speed.left) / 2 \
            + (self.last_wheels_speed.right + self.wheels_speed.right) / 2) * t_s

        rel_rotation = (new_orient + self.current_pose[1]) / 2
        pos_x, pos_y = self.current_pose[0]
        new_x = pos_x + new_x_local * cos(rel_rotation)
        new_y = pos_y + new_x_local * sin(rel_rotation)
        self.current_pose = ((new_x, new_y), new_orient)
        self.last_wheels_speed = self.wheels_speed


@dataclass
class DifferentialDriveRobot():
    """Representing a robot with differential driving behavior"""

    spawn_pose: RobotPose
    goal: Vec2D
    config: RobotSettings
    state: RobotState = field(init=False)

    def __post_init__(self):
        self.state = RobotState(
            self.config,
            (0, 0),
            self.spawn_pose,
            WheelSpeedState(0, 0),
            WheelSpeedState(0, 0))

    @property
    def pos(self) -> Vec2D:
        return self.state.current_pose[0]

    @property
    def pose(self) -> RobotPose:
        return self.state.current_pose

    @property
    def dist_to_goal(self) -> float:
        return dist(self.pose[0], self.goal)

    @property
    def current_speed(self) -> PolarVec2D:
        return self.state.current_speed

    def apply_action(self, action: PolarVec2D, d_t: float) -> Tuple[PolarVec2D, bool]:
        movement, clipped = self.state.resulting_movement(action)
        self.state.update_robot_speed(movement)
        self.state.compute_odometry(d_t)
        return movement, clipped
