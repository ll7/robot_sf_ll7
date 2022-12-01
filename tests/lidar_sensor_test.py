from math import pi, cos, sin
from typing import Tuple

import numpy as np
from pytest import approx

from robot_sf.map import BinaryOccupancyGrid
from robot_sf.range_sensor import LidarScanner, LidarScannerSettings
from robot_sf.vector import RobotPose, Vec2D


NO_SCAN_NOISE = [0.0, 0.0]
Point2D = Tuple[float, float]


def rotate(point: Point2D, rot_center: Point2D, rot_angle_rad: float) -> Point2D:
    x, y = point[0] - rot_center[0], point[1] - rot_center[1]
    s, c = sin(rot_angle_rad), cos(rot_angle_rad)
    x, y = x * c - y * s, x * s + y * c
    return x + rot_center[0], y + rot_center[1]


def test_scanner_detects_obstacles():
    # construct obstacles to form a 360-edged, isosceles polygon that's
    # located at the map's center where each ray hits the polygon
    # orthogonally after a distance of 2.0
    lidar_n_rays = 360
    cached_angles = np.linspace(0, 2*pi, lidar_n_rays + 1)[:-1]
    obs_starts = np.array([rotate((2, 1), (0, 0), rot) for rot in cached_angles])
    obs_ends = np.array([rotate((2, -1), (0, 0), rot) for rot in cached_angles])
    obstacles = np.concatenate((obs_starts[:, 0:1], obs_ends[:, 0:1],
                                obs_starts[:, 1:2], obs_ends[:, 1:2]), axis=1)
    occupancy = BinaryOccupancyGrid(10, 10, 10, 10, lambda: obstacles, lambda: np.array([]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = LidarScanner(settings, occupancy)
    scan = scanner.get_scan(RobotPose(Vec2D(0, 0), 0))

    exp_dist, deviation = 2.0, 0.2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist), deviation)
