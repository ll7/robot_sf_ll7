from math import sin, cos
from dataclasses import dataclass, field
from typing import Tuple, Protocol

import numpy as np

from robot_sf.vector import RobotPose, PolarVec2D


Vec2D = Tuple[float, float]


class MapImpl(Protocol):
    def is_obstacle_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        raise NotImplementedError()

    def is_pedestrians_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        raise NotImplementedError()

    def is_in_bounds(self, world_x: float, world_y: float) -> bool:
        raise NotImplementedError()


class ScannerImpl(Protocol):
    def get_scan(self, pose: RobotPose) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class WheelSpeedState:
    left: float
    right: float


@dataclass
class RobotSettings:
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
    last_pose: RobotPose
    last_wheels_speed: WheelSpeedState
    wheels_speed: WheelSpeedState

    # TODO: think of adding a markdown file describing what's happening here

    def resulting_movement(self, action: PolarVec2D) -> Tuple[PolarVec2D, bool]:
        dot_x = self.current_speed.dist + action.dist
        dot_orient = self.current_speed.orient + action.orient
        clipped = dot_x < 0 or dot_x > self.config.max_linear_speed or \
            abs(dot_orient) > self.config.max_angular_speed

        dot_x = np.clip(dot_x, 0, self.config.max_linear_speed)
        angular_max = self.config.max_angular_speed
        dot_orient = np.clip(dot_orient, -angular_max, angular_max)
        return PolarVec2D(dot_x, dot_orient), clipped

    def update_robot_speed(self, movement: PolarVec2D):
        # compute Kinematics
        dot_x, dot_orient = movement.dist, movement.orient
        diff = self.config.interaxis_length * dot_orient / 2
        self.wheels_speed.left = (dot_x - diff) / self.config.wheel_radius
        self.last_wheels_speed.right = (dot_x + diff) / self.config.wheel_radius

        # update current speed
        self.current_speed.dist = dot_x
        self.current_speed.orient = dot_orient

    def compute_odometry(self, t_s: float):
        right_left_diff = -(self.last_wheels_speed.left + self.wheels_speed.left) / 2 \
            + (self.last_wheels_speed.right + self.wheels_speed.right) / 2

        new_orient = self.last_pose.orient \
            + self.config.wheel_radius / self.config.interaxis_length \
                * right_left_diff * t_s

        new_x_local = self.config.wheel_radius / 2 \
                * ((self.last_wheels_speed.left + self.wheels_speed.left) / 2 \
            + (self.last_wheels_speed.right + self.wheels_speed.right) / 2) * t_s

        rel_rotation = (new_orient + self.last_pose.orient) / 2
        x, y = self.current_pose.pos
        new_x = x + new_x_local * cos(rel_rotation)
        new_y = y + new_x_local * sin(rel_rotation)
        self.current_pose.pos = (new_x, new_y)
        self.current_pose.orient = new_orient

        # update old values
        self.last_pose = self.current_pose
        self.last_wheels_speed = self.wheels_speed


@dataclass
class DifferentialDriveRobot():
    """Representing a robot with differential driving behavior"""
    spawn_pose: RobotPose
    scanner: ScannerImpl
    _map: MapImpl
    config: RobotSettings
    state: RobotState = field(init=False)

    def __post_init__(self):
        self.state = RobotState(
            self.config,
            PolarVec2D(0, 0),
            self.spawn_pose,
            self.spawn_pose,
            WheelSpeedState(0, 0),
            WheelSpeedState(0, 0))

    @property
    def pos(self) -> Vec2D:
        return self.state.current_pose.pos

    @property
    def pose(self) -> RobotPose:
        return self.state.current_pose

    @property
    def is_out_of_bounds(self):
        """checks if robot went out of bounds """
        pos_x, pos_y = self.state.current_pose.coords
        return not self._map.is_in_bounds(pos_x, pos_y)

    def apply_action(self, action: PolarVec2D, d_t: float) -> Tuple[PolarVec2D, bool]:
        movement, clipped = self.state.resulting_movement(action)
        self.state.update_robot_speed(movement)
        self.state.compute_odometry(d_t)
        return movement, clipped

    def get_scan(self):
        return self.scanner.get_scan(self.state.current_pose)

    def is_obstacle_collision(self, collision_distance: float) -> bool:
        return self._map.is_obstacle_collision(
            self.state.current_pose.pos, collision_distance)

    def is_pedestrians_collision(self, collision_distance: float) -> bool:
        return self._map.is_pedestrians_collision(
            self.state.current_pose.pos, collision_distance)

    def is_target_reached(self, target_coordinates: Vec2D, tolerance: float):
        # TODO: think about whether the robot should know its goal
        #       -> maybe model this as a class "NagivationRequest" or similar
        return self.state.current_pose.rel_pos(target_coordinates)[0] <= tolerance
