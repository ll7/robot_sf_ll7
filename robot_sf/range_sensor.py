from dataclasses import dataclass, field
from math import pi, sin, cos, ceil
from typing import List, Tuple

import numpy as np

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


def bresenham_line(grid_width: int, grid_height: int, p1: GridPoint, p2: GridPoint) -> np.ndarray:
    """Jack Bresenham's algorithm (1962) to draw a line with 0/1 pixels
    between the given points p1 and p2 on a 2D grid of given size."""

    occupancy = np.zeros((grid_width, grid_height), dtype=bool)
    (x_0, y_0), (x_1, y_1) = p1, p2

    sign_x = 1 if x_0 < x_1 else -1
    sign_y = 1 if y_0 < y_1 else -1
    diff_x = abs(x_1 - x_0)
    diff_y = -abs(y_1 - y_0)
    error = diff_x + diff_y

    while True:
        occupancy[x_0, y_0] = 1
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

    return occupancy


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


def compute_ray_cache(width: int, height: int, resolution: int, lidar_n_rays: int) -> np.ndarray:
    """Precompute a given amount of ray occupancies for a map of given size.

    IMPORTANT: lidar_n_rays needs to be divisible by 4.
    WARNING: This might allocate several GiB of memory, depending on
    the grid's size, resolution and the amount of rays.

    This implementation makes use of the fact that quadrants can be easily
    distinguished by x/y comparison, so it initializes 4 rays per cache,
    each shifted by 90 degrees. Thus, there are lidar_n_rays / 4 outgoing caches
    of shape (width * resolution * 2 + 1, height * resolution * 2 + 1)."""

    grid_width, grid_height = width * resolution * 2 + 1, height * resolution * 2 + 1
    rays_per_quadrant = ceil(lidar_n_rays / 4)
    shape = (rays_per_quadrant, grid_width, grid_height)
    ray_caches = np.zeros(shape, dtype=bool)

    middle = (grid_width // 2 + 1, grid_height // 2 + 1)
    for i, phi in enumerate(np.linspace(0, pi/2, rays_per_quadrant + 1)[:-1]):
        end_point_q1 = grid_boundary_hit(middle, phi,            grid_width, grid_height)
        end_point_q2 = grid_boundary_hit(middle, phi + pi * 0.5, grid_width, grid_height)
        end_point_q3 = grid_boundary_hit(middle, phi + pi,       grid_width, grid_height)
        end_point_q4 = grid_boundary_hit(middle, phi + pi * 1.5, grid_width, grid_height)
        ray_q1 = bresenham_line(grid_width, grid_height, middle, end_point_q1)
        ray_q2 = bresenham_line(grid_width, grid_height, middle, end_point_q2)
        ray_q3 = bresenham_line(grid_width, grid_height, middle, end_point_q3)
        ray_q4 = bresenham_line(grid_width, grid_height, middle, end_point_q4)
        ray_caches[i] = np.bitwise_and(ray_q1, np.bitwise_and(
            ray_q2, np.bitwise_and(ray_q3, ray_q4)))

    return ray_caches


def compute_distances_cache(width: int, height: int, resolution: int) -> np.ndarray:
    """Precompute the distances of grid cells from the middle (width+1, height+1).

    Returns an array of shape (width * resolution * 2 + 1, height * resolution * 2 + 1)"""

    x = np.arange(-width  * resolution, width  * resolution + 1)[:, np.newaxis]
    y = np.arange(-height * resolution, height * resolution + 1)[np.newaxis, :]
    x, y = np.meshgrid(x, y)
    xy = np.stack((y, x), axis=2)
    dists = np.power((xy[:, :, 0] / resolution)**2 + (xy[:, :, 1] / resolution)**2, 0.5)
    return dists


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

        self.cached_rays = compute_ray_cache(
            self.robot_map.map_width, self.robot_map.map_height,
            self.robot_map.map_resolution, self.settings.lidar_n_rays)
        self.cached_distances = compute_distances_cache(
            self.robot_map.map_width, self.robot_map.map_height, self.robot_map.map_resolution)

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
        scan_noise = self.settings.scan_noise
        scan_length = self.settings.scan_length

        # perform vectorized raycast
        min_angle = pose.orient + self.settings.angle_opening.lower
        max_angle = pose.orient + self.settings.angle_opening.upper
        first_ray_id, last_ray_id = self.angle_id(min_angle), self.angle_id(max_angle)
        all_ranges = self.batched_raycast(start_pt, self.settings.max_scan_dist)
        if last_ray_id < first_ray_id:
            ranges = np.concatenate((all_ranges[first_ray_id:], all_ranges[:last_ray_id]))
        else:
            ranges = all_ranges[first_ray_id:last_ray_id]

        # simulate lost scans and signal noise
        not_lost_scans = np.where(np.random.random(scan_length) >= scan_noise[0], 1, 0)
        corrupt_scans = np.where(np.random.random(scan_length) < scan_noise[1], 1, 0)
        ranges = np.where(not_lost_scans, ranges, self.settings.max_scan_dist)
        corrupt_scans_mask = np.bitwise_and(corrupt_scans, not_lost_scans)
        ranges = np.where(corrupt_scans_mask, ranges * np.random.random(scan_length), ranges)
        return ranges

    def batched_raycast(self, scanner_position: List[float], max_scan_dist: float) -> np.ndarray:
        # shift the occupancy such that the robot's
        # position is at the middle of a 2x bigger grid
        x, y = scanner_position
        orig_occupancy = self.get_object_occupancy()
        width, height = orig_occupancy.shape
        offset = (width + 1 - x, height + 1 - y)
        occupancy = np.zeros((width * 2 + 1, height * 2 + 1), dtype=bool)
        occupancy[offset[0]:offset[0]+width, offset[1]:offset[1]+height] = orig_occupancy

        # apply the cached ray masks by bitwise AND, then retrieve distances
        collisions_mask = np.bitwise_and(self.cached_rays, occupancy)
        collision_dists = np.where(collisions_mask, self.cached_distances, max_scan_dist)

        # evaluate the minimal distance per ray (one ray per quadrant)
        ranges_q1 = np.min(np.min(collision_dists[:, width+1:2*width+1, 0:height+1], axis=1), axis=1)
        ranges_q2 = np.min(np.min(collision_dists[:, 0:width+1, 0:height+1], axis=1), axis=1)
        ranges_q3 = np.min(np.min(collision_dists[:, 0:width+1, height+1:2*height+1], axis=1), axis=1)
        ranges_q4 = np.min(np.min(collision_dists[:, width+1:2*width+1, height+1:2*height+1], axis=1), axis=1)
        ranges_all = np.concatenate((ranges_q1, ranges_q2, ranges_q3, ranges_q4))

        # cap range distances at max scan range
        return np.where(ranges_all > max_scan_dist, max_scan_dist, ranges_all)
