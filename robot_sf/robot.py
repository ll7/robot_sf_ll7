from math import sin, cos
from dataclasses import dataclass
from typing import List

import numpy as np

from robot_sf.map import BinaryOccupancyGrid, fill_surrounding
from robot_sf.range_sensor import LiDARscanner, LidarScannerSettings
from robot_sf.utils.utilities import norm_angles


@dataclass
class Vec2D:
    x: float
    y: float


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


class DifferentialDriveRobot():
    """Representing a robot with differential driving behavior"""

    def __init__(self,
            spawn_pose: RobotPose,
            robot_settings: RobotSettings,
            lidar_settings: LidarScannerSettings,
            robot_map: BinaryOccupancyGrid = None):
        self.config = robot_settings
        self.lidar_config = lidar_settings
        self.map = robot_map if robot_map else BinaryOccupancyGrid()
        self.scanner = LiDARscanner(lidar_settings)

        # transform this into (dist, orient) tuple representing directed 2D movement
        self.current_speed = SpeedState(0, 0)

        self.current_pose = spawn_pose
        self.last_pose = spawn_pose
        self.robot_occupancy = self.get_robot_occupancy()

        self.last_wheels_speed = WheelSpeedState(0, 0)
        self.wheels_speed = self.last_wheels_speed

    @property
    def get_current_pose(self) -> RobotPose:
        return self.current_pose

    def set_robot_pose(self, pose: RobotPose):
        # TODO: pose computation should be self-contained -> move it inside this class
        self.current_pose = pose
        self.last_pose = self.current_pose
        self.clear_old_var()
        self.update_robot_occupancy()

    def update_robot_speed(self, dot_x: float, dot_orient: float):
        # TODO: velocity computation should be self-contained -> move it inside this class
        if dot_x > self.config.max_linear_speed or dot_x < -self.config.max_linear_speed:
            dot_x = np.sign(dot_x) * self.config.max_linear_speed

        if dot_orient > self.config.max_angular_speed or dot_orient < -self.config.max_angular_speed:
            dot_orient = np.sign(dot_orient) * self.config.max_angular_speed

        # compute Kinematics
        self.wheels_speed.left = (dot_x - self.config.interaxis_length * dot_orient / 2) / self.config.wheel_radius
        self.last_wheels_speed.right = (dot_x + self.config.interaxis_length * dot_orient / 2) / self.config.wheel_radius

        # update current speed
        self.current_speed.dist = dot_x
        self.current_speed.orient = dot_orient

    def compute_odometry(self, t_s: float):
        # TODO: odometry computation should be self-contained -> move it inside this class
        right_left_diff = -(self.last_wheels_speed.left + self.wheels_speed.left) / 2 \
            + (self.last_wheels_speed.right + self.wheels_speed.right) / 2
        new_orient = self.last_pose.orient \
            + self.config.wheel_radius / self.config.interaxis_length * right_left_diff * t_s

        new_x_local = self.config.wheel_radius / 2 * ((self.last_wheels_speed.left + self.wheels_speed.left) / 2 \
            + (self.last_wheels_speed.right + self.wheels_speed.right) / 2) * t_s

        self.current_pose.pos.x += new_x_local * cos((new_orient + self.last_pose.orient) / 2)
        self.current_pose.pos.y += new_x_local * sin((new_orient + self.last_pose.orient) / 2)
        self.current_pose.orient = new_orient

        # update old values
        self.last_pose = self.current_pose
        self.last_wheels_speed = self.wheels_speed

        # update robot occupancy map
        self.update_robot_occupancy()

    def clear_old_var(self):
        self.last_wheels_speed = WheelSpeedState(0, 0)
        self.wheels_speed = WheelSpeedState(0, 0)

    def get_scan(self, scan_noise: List[float]):
        return self.scanner.get_scan(
            self.current_pose.pos.x,
            self.current_pose.pos.y,
            self.current_pose.orient,
            self.map,
            scan_noise)

    def check_out_of_bounds(self, margin = 0):
        """checks if robot went out of bounds """
        return not self.map.check_if_valid_world_coordinates(
            self.current_pose.coords, margin).any()

    def get_peds_distances(self):
        """returns vector with distances of all pedestrians from robot"""
        idx_x, idx_y = np.where(self.map.occupancy_raw == True)
        idxs_peds = np.concatenate((idx_x[:,np.newaxis], idx_y[:, np.newaxis]), axis=1)
        world_coords_peds = self.map.convert_grid_to_world(idxs_peds)
        return np.sqrt(np.sum((world_coords_peds - self.current_pose.coords)**2, axis=1))

    def get_distances_and_angles(self, peds_only: bool=False):
        """returns vector with distances of all pedestrians from robot"""
        occpuancy = self.map.occupancy_raw if peds_only \
            else np.logical_or(self.map.occupancy_raw, self.map.occupancy_fixed_raw)
        idx_x, idx_y = np.where(occpuancy)

        idxs_objs = np.concatenate((idx_x[:,np.newaxis], idx_y[:,np.newaxis]), axis=1)
        world_coords_objs = self.map.convert_grid_to_world(idxs_objs)

        # compute angles
        pose = self.current_pose
        dst =  np.sqrt(np.sum((world_coords_objs - pose.coords)**2, axis = 1))
        x_offsets = world_coords_objs[:,0] - pose.pos.x
        y_offsets = world_coords_objs[:,1] - pose.pos.y
        alphas = np.arctan2(y_offsets, x_offsets) - pose.orient
        alphas = norm_angles(alphas)

        return (alphas, dst)

    def chunk(self, n_sections, peds_only=False):
        alphas, dst = self.get_distances_and_angles(peds_only)
        bins = np.pi * (2 * np.arange(0, n_sections + 1) / n_sections - 1)
        sector_id = np.arange(0,n_sections-1)
        distances = n_sections * [1.] # TODO: is this for casting as float array?!

        inds = np.digitize(alphas, bins)
        for j in range(len(sector_id)):
            if j in inds:
                distances[j] = min(1.0, round(min(dst[np.where(inds == j)]) / self.scanner.range[1], 2))
        # TODO: rewrite this for loop as list comprehension
        return distances

    def update_robot_occupancy(self):
        # internal update only uses attribut collision distance
        self.robot_occupancy = self.get_robot_occupancy()

    def get_robot_occupancy(self, coll_distance: float=None) -> np.ndarray:
        # TODO: figure out whether this function occasionally throws an exception

        coll_distance = coll_distance if coll_distance else self.config.rob_collision_radius

        rob_matrix = np.zeros(self.map.occupancy.shape, dtype=bool)
        idx = self.map.convert_world_to_grid_no_error(
            np.array(self.current_pose.coords)[np.newaxis, :])
        self.robot_idx = idx
        int_radius_step = round(self.map.map_resolution * coll_distance)
        return fill_surrounding(rob_matrix, int_radius_step, idx)

    def check_obstacle_collision(self, collision_distance: float=None) -> bool:
        if self.map is None:
            return False
        occupancy = self.get_robot_occupancy(collision_distance) \
            if collision_distance else self.robot_occupancy
        return np.logical_and(self.map.occupancy_fixed, occupancy).any()

    def check_pedestrians_collision(self, collision_distance: float=None) -> bool:
        if self.map is None:
            return False
        occupancy = self.get_robot_occupancy(collision_distance) \
            if collision_distance else self.robot_occupancy
        return np.logical_and(self.map.occupancy, occupancy).any()

    def check_collision(self, collision_distance: float=None):
        # check for collision with map objects (when distance from robots is less than radius)
        if self.map is None:
            return False
        return self.check_pedestrians_collision(collision_distance) \
            or self.check_obstacle_collision(collision_distance)

    def target_rel_position(self, target_coordinates, current_pose: RobotPose=None):
        current_pose = current_pose if current_pose else self.current_pose

        dists = np.linalg.norm(target_coordinates - np.array(current_pose.coords))
        x_offsets = target_coordinates[1] - current_pose.pos.y
        y_offsets = target_coordinates[0] - current_pose.pos.x
        angles = np.arctan2(y_offsets, x_offsets) - current_pose.orient
        angles = norm_angles(angles)

        return dists, angles

    def check_target_reach(self, target_coordinates, tolerance = 0.2):
        return self.target_rel_position(target_coordinates)[0] <= tolerance
