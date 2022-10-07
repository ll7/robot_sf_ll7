from math import sin, cos
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from robot_sf.vector import RobotPose, MovementVec2D, Vec2D
from robot_sf.map import BinaryOccupancyGrid
from robot_sf.range_sensor import LiDARscanner, LidarScannerSettings


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
    map: BinaryOccupancyGrid
    current_speed: MovementVec2D
    current_pose: RobotPose
    last_pose: RobotPose
    last_wheels_speed: WheelSpeedState
    wheels_speed: WheelSpeedState

    # TODO: think of adding a markdown file describing what's happening here

    def resulting_movement(self, action: MovementVec2D) -> Tuple[MovementVec2D, bool]:
        dot_x = self.current_speed.dist + action.dist
        dot_orient = self.current_speed.orient + action.orient
        clipped = dot_x < 0 or dot_x > self.config.max_linear_speed or \
            abs(dot_orient) > self.config.max_angular_speed

        dot_x = np.clip(dot_x, 0, self.config.max_linear_speed)
        angular_max = self.config.max_angular_speed
        dot_orient = np.clip(dot_orient, -angular_max, angular_max)
        return MovementVec2D(dot_x, dot_orient), clipped

    def update_robot_speed(self, movement: MovementVec2D):
        # compute Kinematics
        dot_x, dot_orient = movement.dist, movement.orient
        diff = self.config.interaxis_length * dot_orient / 2 # TODO: rename this
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
        self.current_pose.pos.x += new_x_local * cos(rel_rotation)
        self.current_pose.pos.y += new_x_local * sin(rel_rotation)
        self.current_pose.orient = new_orient

        # update old values
        self.last_pose = self.current_pose
        self.last_wheels_speed = self.wheels_speed


@dataclass
class DifferentialDriveRobot():
    """Representing a robot with differential driving behavior"""
    spawn_pose: RobotPose
    config: RobotSettings
    lidar_settings: LidarScannerSettings
    map: BinaryOccupancyGrid
    scanner: LiDARscanner = field(init=False)
    state: RobotState = field(init=False)

    def __post_init__(self):
        self.scanner = LiDARscanner(self.lidar_settings)
        self.state = RobotState(
            self.config,
            self.map,
            MovementVec2D(0, 0),
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

    def apply_action(self, action: MovementVec2D, d_t: float) -> Tuple[MovementVec2D, bool]:
        movement, clipped = self.state.resulting_movement(action)
        self.state.update_robot_speed(movement)
        self.state.compute_odometry(d_t)
        return movement, clipped

    def get_scan(self, scan_noise: List[float]):
        return self.scanner.get_scan(self.state.current_pose, self.map, scan_noise)

    def is_out_of_bounds(self, margin = 0):
        """checks if robot went out of bounds """
        return not self.map.check_if_valid_world_coordinates(
            self.state.current_pose.coords, margin).any()

    def is_obstacle_collision(self, collision_distance: float) -> bool:
        return self.map.is_obstacle_collision(self.state.current_pose.pos, collision_distance)

    def is_pedestrians_collision(self, collision_distance: float) -> bool:
        return self.map.is_pedestrians_collision(self.state.current_pose.pos, collision_distance)

    def is_target_reached(self, target_coordinates: np.ndarray, tolerance: float):
        # TODO: think about whether the robot should know his goal's coords
        #       -> maybe model this as a class "NagivationRequest" or similar
        return self.state.current_pose.target_rel_position(target_coordinates)[0] <= tolerance
