from dataclasses import dataclass, field
from math import pi, sin, cos
from typing import List, Tuple

import numpy as np
import numba

from robot_sf.map import BinaryOccupancyGrid
from robot_sf.vector import RobotPose


GridPoint = Tuple[int, int]
GridLine = Tuple[GridPoint, GridPoint]


@dataclass
class Range:
    lower: float
    upper: float


@dataclass
class LidarScannerSettings:
    """Representing LiDAR sensor configuration settings."""

    max_scan_dist: float
    visual_angle_portion: float # info: value between 0 and 1
    lidar_n_rays: int # info: needs to be a multiple of 4!!!
    angle_opening: Range=field(init=False)
    scan_length: int=field(init=False)
    scan_noise: List[float]=field(default_factory=lambda: [0, 0])

    def __post_init__(self):
        if self.lidar_n_rays % 4 != 0 and self.lidar_n_rays > 0:
            raise ValueError('Amount of rays needs to be divisible by 4!')

        if not 0 < self.visual_angle_portion <= 1:
            raise ValueError('Scan angle portion needs to be within (0, 1)!')

        self.scan_length = int(self.visual_angle_portion * self.lidar_n_rays)
        self.angle_opening = Range(
            -np.pi * self.visual_angle_portion,
             np.pi * self.visual_angle_portion)


@numba.njit(fastmath=True)
def bresenham_line(out_x: np.ndarray, out_y: np.ndarray,
                   p1: GridPoint, p2: GridPoint) -> int:
    """Jack Bresenham's algorithm (1962) to draw a line with 0/1 pixels
    between the given points p1 and p2 on a 2D grid of given size."""

    (x_0, y_0), (x_1, y_1) = p1, p2

    sign_x = 1 if x_0 < x_1 else -1
    sign_y = 1 if y_0 < y_1 else -1
    diff_x = abs(x_1 - x_0)
    diff_y = -abs(y_1 - y_0)
    error = diff_x + diff_y

    i = 0
    while True:
        out_x[i] = x_0
        out_y[i] = y_0
        i += 1
        if x_0 == x_1 and y_0 == y_1:
            break
        e_2 = 2 * error
        if e_2 >= diff_y:
            if x_0 == x_1:
                break
            error = error + diff_y
            x_0 = x_0 + sign_x

        if e_2 <= diff_x:
            if y_0 == y_1:
                break
            error = error + diff_x
            y_0 = y_0 + sign_y

    return i


def grid_cell_of(x: float, y: float) -> GridPoint:
    """Convert continuous 2D coordinates to the grid cell they belong to."""
    return int(x), int(y)


def normalize_angle(angle: float) -> float:
    """Normalize the given angle (in radians) between [0, 2*pi)."""
    while angle < 0:
        angle += 2 * pi
    while angle >= 2 * pi:
        angle -= 2 * pi
    return angle


def grid_boundary_hit(start_point: GridPoint, orient: float,
                      max_x: int, max_y: int, scan_scale: float=0.1) -> GridPoint:
    """Project a line from the given start point towards the grid's boundaries
    in the given direction. Therefore go in small steps to avoid skipping the bound.

    Returns the grid cell's coordinates where the line projection hit the grid bounds.

    Info: this is a really slow algorithm, but it's only performed once on
    simulator launch, so don't care for performance optimization."""
    x, y = start_point
    scan_vec = (cos(orient) * scan_scale, sin(orient) * scan_scale)
    is_in_bounds = lambda x, y: 0 <= x < max_x and 0 <= y < max_y
    while True:
        new_x, new_y = x + scan_vec[0], y + scan_vec[1]
        if not is_in_bounds(new_x, new_y):
            break
        x, y = new_x, new_y
    return grid_cell_of(x, y)


def compute_distances_cache(width: int, height: int, resolution: int) -> np.ndarray:
    """Precompute the distances of grid cells from the middle (width+1, height+1).

    Returns an array of shape (width * resolution * 2 + 1, height * resolution * 2 + 1)"""

    x = np.arange(-width  * resolution, width  * resolution + 1)[:, np.newaxis]
    y = np.arange(-height * resolution, height * resolution + 1)[np.newaxis, :]
    x, y = np.meshgrid(x, y)
    xy = np.stack((y, x), axis=2)
    dists = np.power((xy[:, :, 0] / resolution)**2 + (xy[:, :, 1] / resolution)**2, 0.5)
    return dists


@numba.njit(parallel=True, fastmath=True)
def simple_raycast(first_ray_id: int, occupancy: numba.types.bool_[:, :], cached_end_pos: np.ndarray,
                   cached_distances: np.ndarray, scan_length: int, lidar_n_rays: int,
                   scanner_position: np.ndarray, max_scan_dist: float) -> np.ndarray:

    width, height = occupancy.shape
    x, y = scanner_position
    offset = np.array([width + 1 - x, height + 1 - y])
    distances = cached_distances[offset[0]:offset[0]+width, offset[1]:offset[1]+height]
    end_pos = cached_end_pos - offset
    ray_x, ray_y = np.zeros((width), dtype=np.int64), np.zeros((height), dtype=np.int64)

    ranges = np.zeros((scan_length))
    for i in range(scan_length):
        angle_id = (first_ray_id + i) % lidar_n_rays
        num_points = bresenham_line(ray_x, ray_y, scanner_position, end_pos[angle_id])
        temp_range = max_scan_dist
        for j in range(num_points):
            x, y = ray_x[j], ray_y[j]
            if occupancy[x, y]:
                collision_distance = distances[x, y]
                temp_range = min(temp_range, collision_distance)
        ranges[i] = temp_range

    return ranges


@numba.njit(parallel=True, fastmath=True)
def range_postprocessing(ranges: np.ndarray, scan_length: int,
                         scan_noise: np.ndarray, max_scan_dist: float) -> np.ndarray:
    not_lost_scans = np.where(np.random.random(scan_length) >= scan_noise[0], 1, 0)
    corrupt_scans = np.where(np.random.random(scan_length) < scan_noise[1], 1, 0)
    ranges = np.where(not_lost_scans, ranges, max_scan_dist)
    corrupt_scans_mask = np.bitwise_and(corrupt_scans, not_lost_scans)
    ranges = np.where(corrupt_scans_mask, ranges * np.random.random(scan_length), ranges)
    return ranges


@dataclass
class LidarScanner():
    """Representing a simulated radial LiDAR scanner operating in a 2D plane."""
    settings: LidarScannerSettings
    robot_map: BinaryOccupancyGrid

    def __post_init__(self):
        if (self.robot_map.map_width * self.robot_map.map_resolution * 2 + 1) \
                * (self.robot_map.map_height * self.robot_map.map_resolution * 2 + 1) \
                    * self.settings.lidar_n_rays > 5e8:
            print("WARNING: The ray cache will allocate more than 500 MB of memory!")

        self.cached_distances = compute_distances_cache(
            self.robot_map.map_width, self.robot_map.map_height, self.robot_map.map_resolution)
        self.cached_angles = np.linspace(0, 2*pi, self.settings.lidar_n_rays + 1)[:-1]
        middle = (self.robot_map.map_width + 1, self.robot_map.map_width + 1)
        w, h = self.robot_map.map_width * 2 + 1, self.robot_map.map_width * 2 + 1
        self.cached_end_pos = np.array([grid_boundary_hit(middle, phi, w, h)
                                        for phi in self.cached_angles])

        self.get_object_occupancy = lambda: self.robot_map.occupancy_overall_xy

    def angle_id(self, orient: float) -> float:
        """Retrieve the ray index related to the given scan direction."""
        orient = normalize_angle(orient)
        return int((orient / (2 * pi)) * self.settings.lidar_n_rays)

    def get_scan(self, pose: RobotPose) -> np.ndarray:
        """This method takes in input the state of the robot
        and an input map (map object) and returns a data structure
        containing the sensor readings"""

        start_pt = np.squeeze(self.robot_map.convert_world_to_grid_no_error(
            np.expand_dims(np.array(pose.coords), axis=0)))
        scan_noise = np.array(self.settings.scan_noise)
        scan_length = self.settings.scan_length

        # perform raycast
        first_ray_id = self.angle_id(pose.orient + self.settings.angle_opening.lower)
        occupancy = self.get_object_occupancy()
        ranges = simple_raycast(first_ray_id, occupancy, self.cached_end_pos, self.cached_distances,
                                scan_length, self.settings.lidar_n_rays,
                                start_pt, self.settings.max_scan_dist)

        # simulate lost scans and signal noise
        ranges = range_postprocessing(ranges, scan_length, scan_noise, self.settings.max_scan_dist)
        return ranges
