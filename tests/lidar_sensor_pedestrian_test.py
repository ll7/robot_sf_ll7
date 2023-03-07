from math import pi, cos, sin
from typing import Tuple

import numpy as np
from pytest import approx

from robot_sf.occupancy import ContinuousOccupancy
from robot_sf.range_sensor import ContinuousLidarScanner, LidarScannerSettings


NO_SCAN_NOISE = [0.0, 0.0]
Vec2D = Tuple[float, float]
Point2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


def rotate(point: Point2D, rot_center: Point2D, rot_angle_rad: float) -> Point2D:
    x, y = point[0] - rot_center[0], point[1] - rot_center[1]
    s, c = sin(rot_angle_rad), cos(rot_angle_rad)
    x, y = x * c - y * s, x * s + y * c
    return x + rot_center[0], y + rot_center[1]


def test_scanner_detects_single_pedestrian():
    lidar_n_rays = 1
    pedestrians = np.array([[2.4, 0]])
    occupancy = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: pedestrians)
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_multiple_equidist_pedestrians_from_center():
    # construct 360 pedestrian circles arranges as isosceles,
    # scanned from the map's center where each ray hits a circle
    # orthogonally after a distance of 2.0
    lidar_n_rays = 360
    cached_angles = np.linspace(0, 2*pi, lidar_n_rays + 1)[:-1]
    ped_pos = np.array([rotate((2.4, 0), (0, 0), rot) for rot in cached_angles])
    occupancy = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: ped_pos)
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), 0))

    exp_dist = 2.0
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_only_closest_pedestrian():
    lidar_n_rays = 1
    pedestrians = np.array([[2.4, 0], [3.4, 0]])
    occupancy = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: pedestrians)
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_nothing_when_there_is_nothing():
    lidar_n_rays = 1
    occupancy = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 5
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_nothing_when_ray_pointing_to_other_side():
    lidar_n_rays = 1
    pedestrians = np.array([[2.4, 0]])
    occupancy = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: pedestrians)
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), 0))

    exp_dist = 5
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))
