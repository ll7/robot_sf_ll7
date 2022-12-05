from dataclasses import dataclass, field
from math import pi, sin, cos
from typing import List, Tuple

import numpy as np
import numba

import logging
logging.getLogger('numba').setLevel(logging.WARNING)

from robot_sf.map_discrete import BinaryOccupancyGrid
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
    lidar_n_rays: int
    angle_opening: Range=field(init=False)
    scan_length: int=field(init=False)
    scan_noise: List[float]=field(default_factory=lambda: [0, 0])

    def __post_init__(self):
        # if self.lidar_n_rays % 4 != 0 and self.lidar_n_rays > 0:
        #     raise ValueError('Amount of rays needs to be divisible by 4!')

        if not 0 < self.visual_angle_portion <= 1:
            raise ValueError('Scan angle portion needs to be within (0, 1]!')

        self.scan_length = int(self.visual_angle_portion * self.lidar_n_rays)
        self.angle_opening = Range(
            -np.pi * self.visual_angle_portion,
             np.pi * self.visual_angle_portion)


# @numba.njit(fastmath=True)
def bresenham_line(out_x: np.ndarray, out_y: np.ndarray,
                   p_1: GridPoint, p_2: GridPoint) -> int:
    """Jack Bresenham's algorithm (1962) to draw a line with 0/1 pixels
    between the given points p1 and p2 on a 2D grid of given size.

    Note: In this implementation the output grid is sparsified such that
    only the pixel coordinates are returned instead of a boolean grid.
    For performance reasons, the output arrays are passed as parameters
    such that the calling scope only needs to allocate once per entire scan."""

    (x_0, y_0), (x_1, y_1) = p_1, p_2

    sign_x = 1 if x_0 < x_1 else -1
    sign_y = 1 if y_0 < y_1 else -1
    diff_x = abs(x_1 - x_0)
    diff_y = -abs(y_1 - y_0)
    error = diff_x + diff_y

    i = 0
    while i < out_x.shape[0]:
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


def grid_cell_of(pos_x: float, pos_y: float) -> GridPoint:
    """Convert continuous 2D coordinates to the grid cell they belong to."""
    return int(pos_x), int(pos_y)


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
    pos_x, pos_y = start_point
    scan_vec = (cos(orient) * scan_scale, sin(orient) * scan_scale)
    is_in_bounds = lambda x, y: 0 <= x < max_x and 0 <= y < max_y
    while True:
        new_x, new_y = pos_x + scan_vec[0], pos_y + scan_vec[1]
        if not is_in_bounds(new_x, new_y):
            break
        pos_x, pos_y = new_x, new_y
    return grid_cell_of(pos_x, pos_y)


def compute_distances_cache(width: float, height: float, x_units: int, y_units: int) -> np.ndarray:
    """Precompute the distances of grid cells from the middle (x_units+1, y_units+1)
    ranging from 0 at the middle to sqrt(width^2 + height^2) at the edges.

    Returns an array of shape (x_units * 2 + 1, y_units * 2 + 1)"""
    x_coords = np.linspace(-1 , 1, num=2*x_units + 1)[:, np.newaxis] * width
    y_coords = np.linspace(-1 , 1, num=2*y_units + 1)[np.newaxis, :] * height
    x_coords, y_coords = np.meshgrid(x_coords, y_coords)
    all_xy = np.stack((y_coords, x_coords), axis=2)
    dists = np.power((all_xy[:, :, 0] + 0.5)**2 + (all_xy[:, :, 1] + 0.5)**2, 0.5)
    return dists


# @numba.njit(parallel=True, fastmath=True)
def raycast(first_ray_id: int, occupancy: numba.types.bool_[:, :], cached_end_pos: np.ndarray,
            cached_distances: np.ndarray, scan_length: int, lidar_n_rays: int,
            scanner_position: np.ndarray, max_scan_dist: float) -> np.ndarray:
    """Perform a given amount of radial ray casts in a 2D plane
    for a fixed amount of equally distributed, discrete directions.
    The scan angles are constrained by the first_ray_id and the scan_length.

    All scan directions are defined as a line between a twice bigger grid's
    center and a point at the grid's boundary (coordinates are pre-computed).
    Considering the scanner's position within the world's grid, the bigger
    grid's middle is shifted to the scanner position by subtracting an offset.
    Also the distances from the middle to any point of the twice bigger
    grid are cached, so they only need to be looked up.

    As scan procedure, Bresenham's line algorithm yields the grid coordinates
    occupied by the ray (line between the scanner's position and the grid bounds).
    It's critical to the algorithm's performance that the coordinates are
    exchanged in a sparse format as (x, y) coordinates. Next, the occupancy
    is simply looked up at the ray's coordinates; if it's a hit, the distance
    is also just looked up from the distances cache. Finally, the scanned
    distances are capped by the maximum scan range."""

    width, height = occupancy.shape
    pos_x, pos_y = width + 1, height + 1 # scanner is always at the middle of the bigger grid
    x_offset, y_offset = width - scanner_position[0], height - scanner_position[1]
    ray_x, ray_y = np.zeros((width + 1), dtype=np.int64), np.zeros((height + 1), dtype=np.int64)

    ranges = np.zeros((scan_length))
    for i in range(scan_length):
        angle_id = (first_ray_id + i) % lidar_n_rays
        num_points = bresenham_line(ray_x, ray_y, (pos_x, pos_y), cached_end_pos[angle_id])
        temp_range = max_scan_dist
        for j in range(num_points):
            pos_x, pos_y = ray_x[j], ray_y[j] # ray coords in big grid
            rel_pos_x, rel_pos_y = pos_x - x_offset, pos_y - y_offset # ray coords in small grid
            if not 0 <= rel_pos_x < width or not 0 <= rel_pos_y < height:
                break
            if occupancy[rel_pos_x, rel_pos_y]:
                coll_dist = cached_distances[pos_x, pos_y] # cached distances from mid of big grid
                temp_range = min(coll_dist, max_scan_dist)
                break
        ranges[i] = temp_range

    return ranges


@numba.njit(parallel=True, fastmath=True)
def range_postprocessing(ranges: np.ndarray, scan_length: int,
                         scan_noise: np.ndarray, max_scan_dist: float) -> np.ndarray:
    """Postprocess the raycast results to simulate a noisy scan result."""
    not_lost_scans = np.where(np.random.random(scan_length) >= scan_noise[0], 1, 0)
    corrupt_scans = np.where(np.random.random(scan_length) < scan_noise[1], 1, 0)
    ranges = np.where(not_lost_scans, ranges, max_scan_dist)
    corrupt_scans_mask = np.bitwise_and(corrupt_scans, not_lost_scans)
    ranges = np.where(corrupt_scans_mask, ranges * np.random.random(scan_length), ranges)
    return ranges


@dataclass
class LidarScanner():
    """Representing a simulated radial LiDAR scanner operating
    in a 2D plane on binary occupancy grids."""
    settings: LidarScannerSettings
    robot_map: BinaryOccupancyGrid

    def __post_init__(self):
        self.cached_distances = compute_distances_cache(
            self.robot_map.box_size, self.robot_map.box_size,
            self.robot_map.grid_width, self.robot_map.grid_height)
        self.cached_angles = np.linspace(0, 2*pi, self.settings.lidar_n_rays + 1)[:-1]
        middle = (self.robot_map.grid_width, self.robot_map.grid_height)
        width, height = self.robot_map.grid_width * 2 - 1, self.robot_map.grid_height * 2 - 1
        self.cached_end_pos = np.array([grid_boundary_hit(middle, phi, width, height)
                                        for phi in self.cached_angles])
        self.get_object_occupancy = lambda: self.robot_map.occupancy_overall

    def _angle_id(self, orient: float) -> float:
        """Retrieve the ray index related to the given scan direction."""
        orient = normalize_angle(orient)
        return int((orient / (2 * pi)) * self.settings.lidar_n_rays)

    def get_scan(self, pose: RobotPose) -> np.ndarray:
        """This method takes in input the state of the robot
        and an input map (map object) and returns a data structure
        containing the sensor readings"""

        pos_x, pos_y = pose.coords
        start_pt = self.robot_map.world_coords_to_grid_cell(pos_x, pos_y)
        scan_noise = np.array(self.settings.scan_noise)
        scan_length = self.settings.scan_length

        # perform raycast
        first_ray_id = self._angle_id(pose.orient + self.settings.angle_opening.lower)
        occupancy = self.get_object_occupancy()
        ranges = raycast(first_ray_id, occupancy, self.cached_end_pos, self.cached_distances,
                         scan_length, self.settings.lidar_n_rays,
                         start_pt, self.settings.max_scan_dist)

        # simulate lost scans and signal noise
        ranges = range_postprocessing(ranges, scan_length, scan_noise, self.settings.max_scan_dist)
        return ranges
