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


def test_scanner_detects_single_obstacle_orthogonal_orientation():
    lidar_n_rays = 1
    obstacles = np.array([[2, 1, 2, -1]])
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: obstacles, lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_obstacle_other_orientation_superpositioned():
    lidar_n_rays = 1
    obstacles = np.array([[0, 1, 0, -1]])
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: obstacles, lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 0
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_obstacle_same_orientation_superpositioned():
    lidar_n_rays = 1
    obstacles = np.array([[1, 0, -1, 0]])
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: obstacles, lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 0
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_obstacle_same_orientation_not_superpositioned():
    lidar_n_rays = 1
    obstacles = np.array([[3, 0, 4, 0]])
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: obstacles, lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 3
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_multiple_equidist_obstacles_from_center():
    # construct obstacles to form a 360-edged, isosceles polygon that's
    # located at the map's center where each ray hits the polygon
    # orthogonally after a distance of 2.0
    lidar_n_rays = 360
    cached_angles = np.linspace(0, 2*pi, lidar_n_rays + 1)[:-1]
    obs_starts = np.array([rotate((2, 1), (0, 0), rot) for rot in cached_angles])
    obs_ends = np.array([rotate((2, -1), (0, 0), rot) for rot in cached_angles])
    obstacles = np.concatenate((obs_starts[:, 0:1], obs_starts[:, 1:2],
                                obs_ends[:, 0:1], obs_ends[:, 1:2]), axis=1)
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: obstacles, lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), 0))

    exp_dist = 2.0
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_multiple_equidist_obstacles_randomly_shifted():
    # construct obstacles to form a 360-edged, isosceles polygon that's
    # located at a random center where each ray hits the polygon
    # orthogonally after a distance of 2.0

    shift_x, shift_y = np.random.uniform(0, 8, size=(2))
    lidar_n_rays = 360
    cached_angles = np.linspace(0, 2*pi, lidar_n_rays + 1)[:-1]
    obs_starts = np.array([rotate((2, 1), (0, 0), rot) for rot in cached_angles])
    obs_ends = np.array([rotate((2, -1), (0, 0), rot) for rot in cached_angles])
    obstacles = np.concatenate((obs_starts[:, 0:1], obs_starts[:, 1:2],
                                obs_ends[:, 0:1], obs_ends[:, 1:2]), axis=1)
    obstacles[:, 0] += shift_x
    obstacles[:, 2] += shift_x
    obstacles[:, 1] += shift_y
    obstacles[:, 3] += shift_y
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: obstacles, lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((shift_x, shift_y), 0))

    exp_dist = 2.0
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_max_range_when_nothing_found():
    max_scan_range = 5
    lidar_n_rays = 360
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: np.array([[]]))
    settings = LidarScannerSettings(max_scan_range, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), 0))

    exp_dist = max_scan_range
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_only_closest_obstacle():
    lidar_n_rays = 1
    obstacles = np.array([[2, 1, 2, -1], [3, 1, 3, -1]])
    occupancy = ContinuousOccupancy(10, lambda: None, lambda: None, lambda: obstacles, lambda: np.array([[]]))
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scanner = ContinuousLidarScanner(settings, occupancy)
    scan = scanner.get_scan(((0, 0), pi))

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))
