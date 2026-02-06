"""Tests for LiDAR range sensor utilities and helpers."""

import numpy as np
import pytest

from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.sensor.range_sensor import (
    LidarScannerSettings,
    circle_line_intersection_distance,
    euclid_dist,
    lidar_ray_scan,
    lidar_sensor_space,
    lineseg_line_intersection_distance,
    range_postprocessing,
    raycast,
    raycast_obstacles,
    raycast_pedestrians,
)


def _py_func(func):
    """Return the pure-Python implementation when numba jit is present."""
    return getattr(func, "py_func", func)


# Circle-line intersection tests


def test_intersection_at_origin():
    """Ray from origin hits circle boundary at radius length."""
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (0.0, 0.0)  # Ray starts at origin
    ray_vec = (1.0, 0.0)  # Ray points along the x-axis
    assert _py_func(circle_line_intersection_distance)(circle, origin, ray_vec) == 1.0


def test_no_intersection():
    """Ray that misses the circle returns infinite distance."""
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (2.0, 2.0)  # Ray starts outside the circle
    ray_vec = (1.0, 0.0)  # Ray points along the x-axis
    assert _py_func(circle_line_intersection_distance)(circle, origin, ray_vec) == float("inf")


def test_intersection_at_circle_perimeter():
    """Diagonal ray from origin intersects unit circle at distance 1."""
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (0.0, 0.0)  # Ray starts at origin
    ray_vec = (1.0, 1.0)  # Ray points diagonally
    assert _py_func(circle_line_intersection_distance)(circle, origin, ray_vec) == 1.0


def test_negative_ray_direction():
    """Ray pointing toward circle returns zero distance at origin edge."""
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (1.0, 0.0)  # Ray starts at x=1
    ray_vec = (-1.0, 0.0)  # Ray points along the negative x-axis
    assert _py_func(circle_line_intersection_distance)(circle, origin, ray_vec) == 0.0


def test_lineseg_line_intersection_distance_hit():
    """Ray intersects vertical segment one unit away."""
    segment = ((1.0, -1.0), (1.0, 1.0))
    sensor_pos = (0.0, 0.0)
    ray_vec = (1.0, 0.0)
    assert _py_func(lineseg_line_intersection_distance)(segment, sensor_pos, ray_vec) == 1.0


def test_lineseg_line_intersection_distance_parallel():
    """Parallel segment yields infinite intersection distance."""
    segment = ((0.0, 1.0), (1.0, 1.0))
    sensor_pos = (0.0, 0.0)
    ray_vec = (1.0, 0.0)
    assert _py_func(lineseg_line_intersection_distance)(segment, sensor_pos, ray_vec) == float(
        "inf"
    )


################################################################################
# Euclidean distance tests


def test_same_point():
    """Euclidean distance is zero for identical points."""
    vec_1 = (0.0, 0.0)
    vec_2 = (0.0, 0.0)
    assert euclid_dist(vec_1, vec_2) == 0.0


def test_unit_distance():
    """Euclidean distance is one for unit offset points."""
    vec_1 = (0.0, 0.0)
    vec_2 = (1.0, 0.0)
    assert euclid_dist(vec_1, vec_2) == 1.0


def test_negative_coordinates():
    """Euclidean distance handles negative coordinates."""
    vec_1 = (0.0, 0.0)
    vec_2 = (-1.0, -1.0)
    assert euclid_dist(vec_1, vec_2) == (2**0.5)


def test_non_integer_distance():
    """Euclidean distance returns sqrt(2) for diagonal unit offsets."""
    vec_1 = (0.0, 0.0)
    vec_2 = (1.0, 1.0)
    assert euclid_dist(vec_1, vec_2) == (2**0.5)


################################################################################
def test_no_pedestrians():
    """Empty pedestrian array leaves ranges unchanged."""
    out_ranges = np.array([10.0, 10.0])
    scanner_pos = (0.0, 0.0)
    max_scan_range = 10.0
    ped_positions = np.empty((0, 2), dtype=np.float64)
    ped_radius = 1.0
    ray_angles = np.array([0.0, np.pi / 2])

    _py_func(raycast_pedestrians)(
        out_ranges,
        scanner_pos,
        max_scan_range,
        ped_positions,
        ped_radius,
        ray_angles,
    )

    assert np.all(out_ranges == 10.0)


def test_pedestrian_in_range():
    """Nearest pedestrian along a ray reduces the output range."""
    out_ranges = np.array([10.0, 10.0])
    scanner_pos = (0.0, 0.0)
    max_scan_range = 10.0
    ped_positions = np.array([[5.0, 0.0]])
    ped_radius = 1.0
    ray_angles = np.array([0.0, np.pi / 2])

    _py_func(raycast_pedestrians)(
        out_ranges,
        scanner_pos,
        max_scan_range,
        ped_positions,
        ped_radius,
        ray_angles,
    )

    assert out_ranges[0] == 4.0
    assert out_ranges[1] == 10.0


def test_pedestrian_out_of_range():
    """Pedestrians beyond max range leave outputs unchanged."""
    out_ranges = np.array([10.0, 10.0])
    scanner_pos = (0.0, 0.0)
    max_scan_range = 10.0
    ped_positions = np.array([[15.0, 0.0]])
    ped_radius = 1.0
    ray_angles = np.array([0.0, np.pi / 2])

    _py_func(raycast_pedestrians)(
        out_ranges,
        scanner_pos,
        max_scan_range,
        ped_positions,
        ped_radius,
        ray_angles,
    )

    assert np.all(out_ranges == 10.0)


def test_raycast_obstacles_hits():
    """Obstacle segment along ray updates range with hit distance."""
    out_ranges = np.array([10.0])
    scanner_pos = (0.0, 0.0)
    obstacles = np.array([[5.0, -1.0, 5.0, 1.0]])
    ray_angles = np.array([0.0])

    _py_func(raycast_obstacles)(out_ranges, scanner_pos, obstacles, ray_angles)

    assert out_ranges[0] == 5.0


def test_raycast_with_enemy_pos():
    """Enemy position path returns edge distance in raycast."""
    scanner_pos = (0.0, 0.0)
    obstacles = np.empty((0, 4), dtype=np.float64)
    max_scan_range = 10.0
    ped_pos = np.empty((0, 2), dtype=np.float64)
    ped_radius = 1.0
    ray_angles = np.array([0.0])
    enemy_pos = np.array([[2.0, 0.0]])
    enemy_radius = 0.5

    out_ranges = _py_func(raycast)(
        scanner_pos,
        obstacles,
        max_scan_range,
        ped_pos,
        ped_radius,
        ray_angles,
        enemy_pos=enemy_pos,
        enemy_radius=enemy_radius,
    )

    assert np.isclose(out_ranges[0], 1.5)


def test_range_postprocessing_clamps_to_max():
    """Postprocessing clamps distances above max to max."""
    out_ranges = np.array([5.0, 12.0])
    _py_func(range_postprocessing)(out_ranges, np.array([0.0, 0.0]), 10.0)
    assert np.allclose(out_ranges, [5.0, 10.0])


def test_range_postprocessing_loss_sets_max():
    """Scan loss sets ranges to max distance."""
    out_ranges = np.array([3.0, 7.0])
    _py_func(range_postprocessing)(out_ranges, np.array([1.0, 0.0]), 10.0)
    assert np.allclose(out_ranges, [10.0, 10.0])


def test_range_postprocessing_corruption_scales_down():
    """Scan corruption keeps values within [0, max]."""
    out_ranges = np.array([4.0, 8.0])
    _py_func(range_postprocessing)(out_ranges, np.array([0.0, 1.0]), 10.0)
    assert np.all(out_ranges >= 0.0)
    assert np.all(out_ranges <= 10.0)


def test_lidar_ray_scan_basic_obstacle():
    """LiDAR scan returns expected hit distance against obstacle."""
    obstacle_coords = np.array([[3.0, 0.0, 3.0, 2.0]])
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
    )
    settings = LidarScannerSettings(max_scan_dist=10.0, num_rays=2, scan_noise=[0.0, 0.0])
    ranges, ray_angles = lidar_ray_scan(((1.0, 1.0), 0.0), occ, settings)

    assert ranges.shape == (2,)
    assert ray_angles.shape == (2,)
    assert np.any(np.isclose(ranges, 2.0))


def test_lidar_ray_scan_enemy_branch():
    """Ego-ped occupancy path detects enemy radius."""
    obstacle_coords = np.empty((0, 4), dtype=np.float64)
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = EgoPedContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
        get_enemy_coords=lambda: (2.0, 1.0),
        enemy_radius=0.5,
    )
    settings = LidarScannerSettings(max_scan_dist=10.0, num_rays=2, scan_noise=[0.0, 0.0])
    ranges, _ray_angles = lidar_ray_scan(((1.0, 1.0), 0.0), occ, settings)
    assert np.any(np.isclose(ranges, 0.5))


def test_lidar_ray_scan_detects_other_robots_when_enabled():
    """LiDAR scan should include dynamic robot circles when enabled."""
    obstacle_coords = np.empty((0, 4), dtype=np.float64)
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
        get_dynamic_objects=lambda: [((2.0, 1.0), 0.5)],
    )
    settings = LidarScannerSettings(
        max_scan_dist=10.0,
        num_rays=2,
        scan_noise=[0.0, 0.0],
        detect_other_robots=True,
    )
    ranges, _ray_angles = lidar_ray_scan(((1.0, 1.0), 0.0), occ, settings)
    assert np.any(ranges < settings.max_scan_dist)


def test_lidar_ray_scan_ignores_other_robots_when_disabled():
    """LiDAR scan should ignore dynamic robots when detection is disabled."""
    obstacle_coords = np.empty((0, 4), dtype=np.float64)
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
        get_dynamic_objects=lambda: [((2.0, 1.0), 0.5)],
    )
    settings = LidarScannerSettings(
        max_scan_dist=10.0,
        num_rays=2,
        scan_noise=[0.0, 0.0],
        detect_other_robots=False,
    )
    ranges, _ray_angles = lidar_ray_scan(((1.0, 1.0), 0.0), occ, settings)
    assert np.allclose(ranges, settings.max_scan_dist)


def test_lidar_ray_scan_no_false_self_hit_when_dynamic_list_excludes_self():
    """LiDAR should report no dynamic hit when the dynamic callback excludes self robot."""
    obstacle_coords = np.empty((0, 4), dtype=np.float64)
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
        get_dynamic_objects=lambda: [],
    )
    settings = LidarScannerSettings(
        max_scan_dist=10.0,
        num_rays=2,
        scan_noise=[0.0, 0.0],
        detect_other_robots=True,
    )
    ranges, _ray_angles = lidar_ray_scan(((1.0, 1.0), 0.0), occ, settings)
    assert np.allclose(ranges, settings.max_scan_dist)


def test_lidar_sensor_space_bounds():
    """Sensor space bounds match expected shape and limits."""
    space = lidar_sensor_space(3, 7.0)
    assert space.shape == (3,)
    assert np.allclose(space.low, [0.0, 0.0, 0.0])
    assert np.allclose(space.high, [7.0, 7.0, 7.0])


def test_lidar_settings_validation_errors():
    """Invalid settings raise ValueError for each constraint."""
    with pytest.raises(ValueError):
        LidarScannerSettings(visual_angle_portion=0.0)
    with pytest.raises(ValueError):
        LidarScannerSettings(max_scan_dist=0.0)
    with pytest.raises(ValueError):
        LidarScannerSettings(num_rays=0)
    with pytest.raises(ValueError):
        LidarScannerSettings(scan_noise=[-0.1, 0.2])
    with pytest.raises(ValueError):
        LidarScannerSettings(scan_noise=[0.1, 1.5])
