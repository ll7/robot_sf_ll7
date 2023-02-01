from math import cos, sin
from typing import List, Tuple
from dataclasses import dataclass, field

import numpy as np
import numba

from robot_sf.occupancy import ContinuousOccupancy
from robot_sf.geometry import circle_line_intersection_distance, \
                              lineseg_line_intersection_distance


Vec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]
Range = Tuple[float, float]


@dataclass
class LidarScannerSettings:
    """Representing LiDAR sensor configuration settings."""

    max_scan_dist: float = 10.0
    visual_angle_portion: float = 1.0
    lidar_n_rays: int = 272
    scan_noise: List[float] = field(default_factory=lambda: [0.005, 0.002])
    angle_opening: Range = field(init=False)
    scan_length: int = field(init=False)

    def __post_init__(self):
        if not 0 < self.visual_angle_portion <= 1:
            raise ValueError('Scan angle portion needs to be within (0, 1]!')
        if self.max_scan_dist <= 0:
            raise ValueError("Max. scan distance mustn't be negative or zero!")
        if self.lidar_n_rays <= 0:
            raise ValueError("Amount of LiDAR rays mustn't be negative or zero!")
        if any([not 0 <= prob <= 1 for prob in self.scan_noise]):
            raise ValueError("Scan noise probabilities must be within [0, 1]!")

        self.scan_length = self.lidar_n_rays
        self.angle_opening = (-np.pi * self.visual_angle_portion, np.pi * self.visual_angle_portion)


@numba.njit(fastmath=True)
def euclid_dist_sq(v_1: Vec2D, v_2: Vec2D) -> float:
    return (v_1[0] - v_2[0])**2 + (v_1[1] - v_2[1])**2


@numba.njit(fastmath=True)
def raycast_pedestrians(
        out_ranges: np.ndarray, scanner_pos: Vec2D, max_scan_range: float,
        ped_positions: np.ndarray, ped_radius: float, ray_angles: np.ndarray):

    if len(ped_positions.shape) != 2 or ped_positions.shape[0] == 0 \
            or ped_positions.shape[1] != 2:
        return

    scanner_pos_np = np.array([scanner_pos[0], scanner_pos[1]])
    threshold_sq = max_scan_range**2
    relative_ped_pos = ped_positions - scanner_pos_np
    dist_sq = np.sum(relative_ped_pos**2, axis=1)
    ped_dist_mask = np.where(dist_sq <= threshold_sq)[0]
    close_ped_pos = relative_ped_pos[ped_dist_mask]

    if len(ped_dist_mask) == 0:
        return

    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        cos_sims = close_ped_pos[:, 0] * unit_vec[0] \
            + close_ped_pos[:, 1] * unit_vec[1]

        ped_dir_mask = np.where(cos_sims >= 0)[0]
        joined_mask = ped_dist_mask[ped_dir_mask]
        relevant_ped_pos = relative_ped_pos[joined_mask]

        for pos in relevant_ped_pos:
            ped_circle = ((pos[0], pos[1]), ped_radius)
            coll_dist = circle_line_intersection_distance(
                ped_circle, (0.0, 0.0), unit_vec)
            out_ranges[i] = min(coll_dist, out_ranges[i])


@numba.njit(fastmath=True)
def raycast_obstacles(
        out_ranges: np.ndarray, scanner_pos: Vec2D,
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
def raycast(scanner_pos: Vec2D, obstacles: np.ndarray, max_scan_range: float,
            ped_pos: np.ndarray, ped_radius: float, ray_angles: np.ndarray) -> np.ndarray:
    """Cast rays in the directions of all given angles outgoing from
    the scanner's position and detect the minimal collision distance
    with either a pedestrian or an obstacle (or in case there's no collision,
    just return the maximum scan range)."""
    out_ranges = np.full((ray_angles.shape[0]), np.inf)
    raycast_pedestrians(out_ranges, scanner_pos, max_scan_range, ped_pos, ped_radius, ray_angles)
    raycast_obstacles(out_ranges, scanner_pos, obstacles, ray_angles)
    return out_ranges


@numba.njit(fastmath=True)
def range_postprocessing(out_ranges: np.ndarray, scan_noise: np.ndarray, max_scan_dist: float):
    """Postprocess the raycast results to simulate a noisy scan result."""
    prob_scan_loss, prob_scan_corruption = scan_noise
    for i in range(out_ranges.shape[0]):
        out_ranges[i] = min(out_ranges[i], max_scan_dist)
        if np.random.random() < prob_scan_loss:
            out_ranges[i] = max_scan_dist
        elif np.random.random() < prob_scan_corruption:
            out_ranges[i] = out_ranges[i] * np.random.random()


@dataclass
class ContinuousLidarScanner():
    """Representing a simulated radial LiDAR scanner operating
    in a 2D plane on a continuous occupancy with explicit objects.

    The occupancy contains the robot (as circle), a set of pedestrians
    (as circles) and a set of static obstacles (as 2D lines)"""

    settings: LidarScannerSettings
    robot_map: ContinuousOccupancy

    def __post_init__(self):
        self.cached_angles = np.linspace(0, 2*np.pi, self.settings.lidar_n_rays + 1)[:-1]

    def get_scan(self, pose: RobotPose) -> np.ndarray:
        """This method takes in input the state of the robot
        and an input map (map object) and returns a data structure
        containing the sensor readings"""

        (pos_x, pos_y), robot_orient = pose
        scan_noise = np.array(self.settings.scan_noise)
        scan_dist = self.settings.max_scan_dist

        ped_pos = self.robot_map.pedestrian_coords
        obstacles = self.robot_map.obstacle_coords

        lower = robot_orient + self.settings.angle_opening[0]
        upper = robot_orient + self.settings.angle_opening[1]
        ray_angles = np.linspace(lower, upper, self.settings.lidar_n_rays + 1)[:-1]
        ray_angles = np.array([(angle + np.pi*2) % (np.pi*2) for angle in ray_angles])

        ranges = raycast(
            (pos_x, pos_y), obstacles, scan_dist, ped_pos,
            self.robot_map.ped_radius, ray_angles)
        range_postprocessing(ranges, scan_noise, scan_dist)
        return ranges
