from math import cos, sin
from typing import List, Tuple
from dataclasses import dataclass, field

import numpy as np
import numba

from robot_sf.vector import RobotPose
from robot_sf.map_continuous import ContinuousOccupancy
from robot_sf.geometry import circle_line_intersection_distance, \
                              lineseg_line_intersection_distance


Vec2D = Tuple[float, float]


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


@numba.njit(fastmath=True)
def raycast_pedestrians(out_ranges: np.ndarray, scanner_pos: Tuple[float, float],
                        ped_pos: np.ndarray, ped_radius: float, ray_angles: np.ndarray):
    # TODO: pre-compute ray unit vectors
    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        for pos in ped_pos:
            ped_circle = ((pos[0], pos[1]), ped_radius)
            coll_dist = circle_line_intersection_distance(ped_circle, scanner_pos, unit_vec)
            out_ranges[i] = min(coll_dist, out_ranges[i])


@numba.njit(fastmath=True)
def raycast_obstacles(out_ranges: np.ndarray, scanner_pos: Vec2D,
                      obstacles: np.ndarray, ray_angles: np.ndarray):
    if len(obstacles.shape) != 2 or obstacles.shape[0] == 0 or obstacles.shape[1] != 4:
        return

    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        for s_x, s_y, e_x, e_y in obstacles:
            obst_lineseg = ((s_x, s_y), (e_x, e_y))
            coll_dist = lineseg_line_intersection_distance(obst_lineseg, scanner_pos, unit_vec)
            out_ranges[i] = min(coll_dist, out_ranges[i])


@numba.njit()
def raycast(scanner_pos: Tuple[float, float], obstacles: np.ndarray,
            ped_pos: np.ndarray, ped_radius: float, ray_angles: np.ndarray) -> np.ndarray:
    out_ranges = np.full((ray_angles.shape[0]), np.inf)
    raycast_pedestrians(out_ranges, scanner_pos, ped_pos, ped_radius, ray_angles)
    raycast_obstacles(out_ranges, scanner_pos, obstacles, ray_angles)
    return out_ranges


@numba.njit(parallel=True, fastmath=True)
def range_postprocessing(ranges: np.ndarray, scan_length: int,
                         scan_noise: np.ndarray, max_scan_dist: float) -> np.ndarray:
    """Postprocess the raycast results to simulate a noisy scan result."""
    not_lost_scans = np.where(np.random.random(scan_length) >= scan_noise[0], 1, 0)
    corrupt_scans = np.where(np.random.random(scan_length) < scan_noise[1], 1, 0)
    ranges = np.clip(ranges, 0, max_scan_dist)
    ranges = np.where(not_lost_scans, ranges, max_scan_dist)
    corrupt_scans_mask = np.bitwise_and(corrupt_scans, not_lost_scans)
    ranges = np.where(corrupt_scans_mask, ranges * np.random.random(scan_length), ranges)
    return ranges


@dataclass
class ContinuousLidarScanner():
    """Representing a simulated radial LiDAR scanner operating
    in a 2D plane on binary occupancy grids."""
    settings: LidarScannerSettings
    robot_map: ContinuousOccupancy

    def __post_init__(self):
        self.cached_angles = np.linspace(0, 2*np.pi, self.settings.lidar_n_rays + 1)[:-1]

    def get_scan(self, pose: RobotPose) -> np.ndarray:
        """This method takes in input the state of the robot
        and an input map (map object) and returns a data structure
        containing the sensor readings"""

        pos_x, pos_y = pose.coords
        robot_orient = pose.orient
        scan_noise = np.array(self.settings.scan_noise)
        scan_length = self.settings.scan_length

        ped_pos = self.robot_map.pedestrian_coords
        obstacles = self.robot_map.obstacle_coords

        lower = robot_orient + self.settings.angle_opening.lower
        upper = robot_orient + self.settings.angle_opening.upper
        ray_angles = np.linspace(lower, upper, self.settings.lidar_n_rays + 1)[:-1]
        ray_angles = np.array([(angle + np.pi*2) % (np.pi*2) for angle in ray_angles])

        ranges = raycast((pos_x, pos_y), obstacles, ped_pos, 0.4, ray_angles)
        ranges = range_postprocessing(ranges, scan_length, scan_noise, self.settings.max_scan_dist)
        return ranges
