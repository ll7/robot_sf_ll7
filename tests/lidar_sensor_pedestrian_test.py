"""Tests for lidar ray scanning against pedestrian circles."""

from math import cos, pi, sin

import numpy as np
from pytest import approx

from robot_sf.common.types import Point2D
from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.sensor.range_sensor import LidarScannerSettings, lidar_ray_scan

NO_SCAN_NOISE = [0.0, 0.0]


def rotate(point: Point2D, rot_center: Point2D, rot_angle_rad: float) -> Point2D:
    """Rotate a 2D point around a center by an angle in radians.

    Args:
        point: Point to rotate as (x, y).
        rot_center: Rotation center as (x, y).
        rot_angle_rad: Counterclockwise rotation angle in radians.

    Returns:
        Rotated point as (x, y).
    """
    x, y = point[0] - rot_center[0], point[1] - rot_center[1]
    s, c = sin(rot_angle_rad), cos(rot_angle_rad)
    x, y = x * c - y * s, x * s + y * c
    return x + rot_center[0], y + rot_center[1]


def test_scanner_detects_single_pedestrian():
    """A pedestrian at (2.4, 0) yields a scan distance of 2."""
    lidar_n_rays = 1
    pedestrians = np.array([[2.4, 0]])
    occupancy = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: pedestrians,
    )
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scan, _ = lidar_ray_scan(((0, 0), pi), occupancy, settings)

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_multiple_equidist_pedestrians_from_center():
    # construct 360 pedestrian circles arranges as isosceles,
    # scanned from the map's center where each ray hits a circle
    # orthogonally after a distance of 2.0
    """360 pedestrian circles around the map center return distance 2 on all rays."""
    lidar_n_rays = 360
    cached_angles = np.linspace(0, 2 * pi, lidar_n_rays + 1)[:-1]
    ped_pos = np.array([rotate((2.4, 0), (0, 0), rot) for rot in cached_angles])
    occupancy = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: ped_pos,
    )
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scan, _ = lidar_ray_scan(((0, 0), 0), occupancy, settings)

    exp_dist = 2.0
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_only_closest_pedestrian():
    """With pedestrians at x=2.4 and x=3.4 the ray reports the nearer distance 2."""
    lidar_n_rays = 1
    pedestrians = np.array([[2.4, 0], [3.4, 0]])
    occupancy = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: pedestrians,
    )
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scan, _ = lidar_ray_scan(((0, 0), pi), occupancy, settings)

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_nothing_when_there_is_nothing():
    """An occupancy map with no obstacles or pedestrians returns the max range 5."""
    lidar_n_rays = 1
    occupancy = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: np.array([[]]),
    )
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scan, _ = lidar_ray_scan(((0, 0), pi), occupancy, settings)

    exp_dist = 5
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detects_nothing_when_ray_pointing_to_other_side():
    """A ray pointing away from the only pedestrian returns the max range 5."""
    lidar_n_rays = 1
    pedestrians = np.array([[2.4, 0]])
    occupancy = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: pedestrians,
    )
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scan, _ = lidar_ray_scan(((0, 0), 0), occupancy, settings)

    exp_dist = 5
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_detect_robot():
    """An ego-pedestrian occupancy with enemy at (3, 0) and radius 1 returns distance 2."""
    lidar_n_rays = 1
    occupancy = EgoPedContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: np.array([[]]),
        get_enemy_coords=lambda: (3, 0),
        enemy_radius=1,
    )
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scan, _ = lidar_ray_scan(((0, 0), pi), occupancy, settings)

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))


def test_scanner_robot_detect_ego_ped():
    """A ray toward +x picks the nearest circle among two pedestrians and the ego pedestrian."""
    lidar_n_rays = 1
    ped_pos = np.array([[3.4, 0], [4.4, 0]])
    ego_ped_pos = (2.4, 0)
    occupancy = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: np.vstack((ped_pos, np.array([ego_ped_pos]))),
    )
    settings = LidarScannerSettings(5, 1, lidar_n_rays, scan_noise=NO_SCAN_NOISE)

    scan, _ = lidar_ray_scan(((0, 0), pi), occupancy, settings)

    exp_dist = 2
    assert scan.shape[0] == lidar_n_rays
    assert scan == approx(np.full((lidar_n_rays), exp_dist))
