from dataclasses import dataclass, field
from typing import List

import numpy as np

from robot_sf.map import BinaryOccupancyGrid
from robot_sf.utils.utilities import norm_angles
from robot_sf.vector import RobotPose


@dataclass
class Range:
    lower: float
    upper: float


@dataclass
class LidarScannerSettings:
    lidar_range: int
    visualization_angle_portion: float
    lidar_n_rays: int
    distance_noise: float=0
    scan_range: Range=field(init=False)
    angle_opening: Range=field(init=False)
    angle_increment: float=field(init=False)

    def __post_init__(self):
        self.scan_range = Range(0, self.lidar_range)
        self.angle_opening = Range(
            -np.pi * self.visualization_angle_portion,
             np.pi * self.visualization_angle_portion)
        self.angle_increment = \
            (self.angle_opening.upper - self.angle_opening.lower) \
                / (self.lidar_n_rays - 1)


class LiDARscanner():
    """The following class is responsible of creating a LiDAR scanner object"""

    def __init__(self, settings: LidarScannerSettings):
        scan_range = [settings.scan_range.lower, settings.scan_range.upper]
        angle_opening = [settings.angle_opening.lower, settings.angle_opening.upper]

        self.scan_range = scan_range
        self.angle_opening = angle_opening
        self.angle_increment = settings.angle_increment
        self.distance_noise = 0
        self.angle_noise = 0

        self._cached_angles = np.arange(
            self.angle_opening[0],
            self.angle_opening[1] + self.angle_increment,
            self.angle_increment).tolist()
        self.num_readings = len(self._cached_angles)

    def get_scan(self, pose: RobotPose, input_map: BinaryOccupancyGrid,
                 scan_noise: List[float]=None) -> np.ndarray:
        """This method takes in input the state of the robot
        and an input map (map object) and returns a data structure
        containing the sensor readings"""

        start_pt = np.array(pose.coords)
        scan_length = self.num_readings
        scan_noise = scan_noise if scan_noise else [0, 0]

        lost_scans = np.where(np.random.random(scan_length) < scan_noise[0], 1, 0)
        corrupt_scans = np.where(np.random.random(scan_length) < scan_noise[1], 1, 0)

        # TODO: don't rotate by robot orientation, use pre-computed angles / offsets
        angles = norm_angles(np.array([phi + pose.orient for phi in self._cached_angles]))
        offset_vecs = np.column_stack((np.cos(angles), np.sin(angles)))
        end_pts = offset_vecs + start_pt

        # TODO: vectorize the raycast as well to compute it in one go
        collisions = np.zeros((scan_length), dtype=np.bool)
        intercepts = np.zeros((scan_length, 2))
        for i in range(scan_length):
            ray_index = input_map.raycast(np.array([start_pt]), np.array([end_pts[i]]))
            collisions[i], intercepts[i] = input_map.does_ray_collide(ray_index)

        # initialize ranges with full range (no collision or lost scan)
        ranges = np.full((self.num_readings), self.scan_range[1], dtype=float)

        # ray did collide with obstacle -> compute distance
        detected_collisions_mask = np.bitwise_and(collisions, np.invert(lost_scans))
        ranges = np.where(detected_collisions_mask, np.linalg.norm(intercepts - start_pt), ranges)

        # ray cast is corrupt or lost -> put noise on the signal
        corrupt_scans_mask = np.bitwise_and(corrupt_scans, np.invert(lost_scans))
        ranges = np.where(corrupt_scans_mask, ranges * np.random.random(scan_length), ranges)

        return ranges
