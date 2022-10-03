from math import sin, cos
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from robot_sf.vector import Vec2D
from robot_sf.map import BinaryOccupancyGrid
from robot_sf.range_sensor import LiDARscanner, LidarScannerSettings
from robot_sf.utils.utilities import norm_angles


@dataclass
class SpeedState:
    """Representing directed movement as 2D polar coords"""
    dist: float
    orient: float

    @property
    def vector(self) -> Vec2D:
        return Vec2D(self.dist * cos(self.orient), self.dist * sin(self.orient))


@dataclass
class RobotPose:
    pos: Vec2D
    orient: float

    @property
    def coords(self) -> List[float]:
        return [self.pos.x, self.pos.y]

    @property
    def coords_with_orient(self) -> List[float]:
        return [self.pos.x, self.pos.y, self.orient]

    # TODO: rename this
    def target_rel_position(self, target_coords: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        dists: np.ndarray = np.linalg.norm(target_coords - np.array(self.coords))
        x_offsets = target_coords[0] - self.pos.x
        y_offsets = target_coords[1] - self.pos.y
        angles = np.arctan2(y_offsets, x_offsets) - self.orient
        angles = norm_angles(angles)
        return dists, angles


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
    current_speed: SpeedState
    current_pose: RobotPose
    last_pose: RobotPose
    last_wheels_speed: WheelSpeedState
    wheels_speed: WheelSpeedState


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
            SpeedState(0, 0),
            self.spawn_pose,
            self.spawn_pose,
            WheelSpeedState(0, 0),
            WheelSpeedState(0, 0))

    # TODO: move this function inside the robot state
    def update_robot_speed(self, dot_x: float, dot_orient: float):
        if dot_x > self.config.max_linear_speed or dot_x < -self.config.max_linear_speed:
            dot_x = np.sign(dot_x) * self.config.max_linear_speed

        if dot_orient > self.config.max_angular_speed or dot_orient < -self.config.max_angular_speed:
            dot_orient = np.sign(dot_orient) * self.config.max_angular_speed

        # compute Kinematics
        self.state.wheels_speed.left = (dot_x - self.config.interaxis_length * dot_orient / 2) / self.config.wheel_radius
        self.state.last_wheels_speed.right = (dot_x + self.config.interaxis_length * dot_orient / 2) / self.config.wheel_radius

        # update current speed
        self.state.current_speed.dist = dot_x
        self.state.current_speed.orient = dot_orient

    def compute_odometry(self, t_s: float):
        # TODO: apply law of demeter
        right_left_diff = -(self.state.last_wheels_speed.left + self.state.wheels_speed.left) / 2 \
            + (self.state.last_wheels_speed.right + self.state.wheels_speed.right) / 2
        new_orient = self.state.last_pose.orient \
            + self.config.wheel_radius / self.config.interaxis_length * right_left_diff * t_s

        new_x_local = self.config.wheel_radius / 2 * ((self.state.last_wheels_speed.left + self.state.wheels_speed.left) / 2 \
            + (self.state.last_wheels_speed.right + self.state.wheels_speed.right) / 2) * t_s

        self.state.current_pose.pos.x += new_x_local * cos((new_orient + self.state.last_pose.orient) / 2)
        self.state.current_pose.pos.y += new_x_local * sin((new_orient + self.state.last_pose.orient) / 2)
        self.state.current_pose.orient = new_orient

        # update old values
        self.state.last_pose = self.state.current_pose
        self.state.last_wheels_speed = self.state.wheels_speed

    def clear_old_var(self):
        self.state.last_wheels_speed = WheelSpeedState(0, 0)
        self.state.wheels_speed = WheelSpeedState(0, 0)

    def get_scan(self, scan_noise: List[float]):
        return self.scanner.get_scan(
            self.state.current_pose.pos.x,
            self.state.current_pose.pos.y,
            self.state.current_pose.orient,
            self.map,
            scan_noise)

    def check_out_of_bounds(self, margin = 0):
        """checks if robot went out of bounds """
        return not self.map.check_if_valid_world_coordinates(
            self.state.current_pose.coords, margin).any()

    def check_collision(self, collision_distance: float):
        # check for collision with map objects (when distance from robots is less than radius)
        return self.check_pedestrians_collision(collision_distance) \
            or self.check_obstacle_collision(collision_distance)

    def check_obstacle_collision(self, collision_distance: float) -> bool:
        return self.map.check_obstacle_collision(self.state.current_pose.pos, collision_distance)

    def check_pedestrians_collision(self, collision_distance: float) -> bool:
        return self.map.check_pedestrians_collision(self.state.current_pose.pos, collision_distance)

    def target_rel_position(self, target_coordinates):
        return self.state.current_pose.target_rel_position(target_coordinates)

    def check_target_reached(self, target_coordinates, tolerance=0.2):
        return self.state.current_pose.target_rel_position(target_coordinates)[0] <= tolerance

    # ==============================================================================
    #         debug information, not required to run the simulation
    # ==============================================================================

    # def get_peds_distances(self):
    #     """returns vector with distances of all pedestrians from robot"""
    #     idx_x, idx_y = np.where(self.map.occupancy_raw == True)
    #     idxs_peds = np.concatenate((idx_x[:,np.newaxis], idx_y[:, np.newaxis]), axis=1)
    #     world_coords_peds = self.map.convert_grid_to_world(idxs_peds)
    #     return np.sqrt(np.sum((world_coords_peds - self.state.current_pose.coords)**2, axis=1))

    # def get_distances_and_angles(self, peds_only: bool=False):
    #     """returns vector with distances of all pedestrians from robot"""
    #     occpuancy = self.map.occupancy_raw if peds_only \
    #         else np.logical_or(self.map.occupancy_raw, self.map.occupancy_fixed_raw)
    #     idx_x, idx_y = np.where(occpuancy)

    #     idxs_objs = np.concatenate((idx_x[:,np.newaxis], idx_y[:,np.newaxis]), axis=1)
    #     world_coords_objs = self.map.convert_grid_to_world(idxs_objs)

    #     # compute angles
    #     pose = self.state.current_pose
    #     dst =  np.sqrt(np.sum((world_coords_objs - pose.coords)**2, axis = 1))
    #     x_offsets = world_coords_objs[:,0] - pose.pos.x
    #     y_offsets = world_coords_objs[:,1] - pose.pos.y
    #     alphas = np.arctan2(y_offsets, x_offsets) - pose.orient
    #     alphas = norm_angles(alphas)

    #     return (alphas, dst)

    # def chunk(self, n_sections, peds_only=False):
    #     alphas, dst = self.get_distances_and_angles(peds_only)
    #     bins = np.pi * (2 * np.arange(0, n_sections + 1) / n_sections - 1)
    #     sector_id = np.arange(0, n_sections - 1)
    #     distances = n_sections * [1.] # TODO: is this for casting as float array?!

    #     inds = np.digitize(alphas, bins)
    #     for j in range(len(sector_id)):
    #         if j in inds:
    #             distances[j] = min(1.0, round(min(dst[np.where(inds == j)]) / self.scanner.range[1], 2))
    #     return distances
