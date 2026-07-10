"""Tests for LiDAR range sensor utilities and helpers."""

import numpy as np
import pytest

from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.sensor.range_sensor import (
    LidarScannerSettings,
    circle_line_intersection_distance,
    euclid_dist,
    lidar_ray_offsets,
    lidar_ray_scan,
    lidar_ray_scan_ranges_only,
    lidar_sensor_space,
    lineseg_line_intersection_distance,
    range_postprocessing,
    raycast,
    raycast_circles,
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


def test_raycast_circles_filters_and_keeps_nearest_hits():
    """Circle raycasting handles range, direction, and nearest-hit filtering."""
    out_ranges = np.array([10.0, 10.0, 10.0])
    scanner_pos = (1.0, 1.0)
    max_scan_range = 10.0
    circles = np.array(
        [
            [5.0, 1.0, 0.5],  # hit at 3.5 along +x
            [3.0, 1.0, 0.25],  # nearer hit at 1.75 along +x
            [1.0, 5.0, 0.5],  # hit at 3.5 along +y
            [-3.0, 1.0, 0.5],  # behind +x ray
            [20.0, 1.0, 1.0],  # outside max_scan_range
        ],
        dtype=np.float64,
    )
    ray_angles = np.array([0.0, np.pi / 2, np.pi])

    _py_func(raycast_circles)(out_ranges, scanner_pos, max_scan_range, circles, ray_angles)

    assert np.allclose(out_ranges, [1.75, 3.5, 3.5])


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


def test_lidar_ray_scan_wraps_large_heading_with_endpoint_convention():
    """LiDAR ray angles should wrap arbitrary headings without adding the upper endpoint."""
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: np.empty((0, 4), dtype=np.float64),
        get_pedestrian_coords=lambda: np.empty((0, 2), dtype=np.float64),
    )
    settings = LidarScannerSettings(
        max_scan_dist=10.0,
        num_rays=4,
        scan_noise=[0.0, 0.0],
        visual_angle_portion=1.0,
    )

    _ranges, ray_angles = lidar_ray_scan(((1.0, 1.0), 8.5 * np.pi), occ, settings)

    expected = np.mod(8.5 * np.pi + np.array([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0]), 2.0 * np.pi)
    np.testing.assert_allclose(ray_angles, expected)


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


def test_lidar_settings_ego_pedestrian_lidar():
    """Ego pedestrian 120-degree lidar has correct extended range and narrow FOV."""
    lidar = LidarScannerSettings.ego_pedestrian_lidar()

    # Check range is extended
    assert lidar.max_scan_dist == 30.0

    # Check angle portion is 1/3 (120 degrees total, -60 to +60)
    assert np.isclose(lidar.visual_angle_portion, 1.0 / 3.0)

    # Check angle opening is correct (-60° to +60°)
    assert np.isclose(lidar.angle_opening[0], -np.pi / 3)  # -60 degrees
    assert np.isclose(lidar.angle_opening[1], np.pi / 3)  # +60 degrees

    # Check ray count is consistent
    assert lidar.num_rays == 272

    # Verify it's a valid configuration (no validation errors)
    assert lidar.max_scan_dist > 0
    assert 0 < lidar.visual_angle_portion <= 1
    assert lidar.num_rays > 0


def test_lidar_settings_default():
    """Default lidar factory method creates 360-degree, 10m range configuration."""
    lidar = LidarScannerSettings.default()

    # Check range is standard
    assert lidar.max_scan_dist == 10.0

    # Check angle portion is 1.0 (full 360 degrees)
    assert np.isclose(lidar.visual_angle_portion, 1.0)

    # Check angle opening is full circle (-180° to +180°)
    assert np.isclose(lidar.angle_opening[0], -np.pi)  # -180 degrees
    assert np.isclose(lidar.angle_opening[1], np.pi)  # +180 degrees

    # Check ray count is consistent
    assert lidar.num_rays == 272

    # Verify it's a valid configuration
    assert lidar.max_scan_dist > 0
    assert 0 < lidar.visual_angle_portion <= 1
    assert lidar.num_rays > 0


def test_lidar_settings_factory_methods_comparison():
    """Verify factory methods create appropriately different configurations."""
    default_lidar = LidarScannerSettings.default()
    ego_ped_lidar = LidarScannerSettings.ego_pedestrian_lidar()

    # Ego pedestrian should have longer range
    assert ego_ped_lidar.max_scan_dist > default_lidar.max_scan_dist

    # Ego pedestrian should have narrower FOV
    assert ego_ped_lidar.visual_angle_portion < default_lidar.visual_angle_portion

    # Same ray count for consistency
    assert ego_ped_lidar.num_rays == default_lidar.num_rays

    # Angle opening should reflect the difference
    assert abs(ego_ped_lidar.angle_opening[0]) < abs(default_lidar.angle_opening[0])
    assert abs(ego_ped_lidar.angle_opening[1]) < abs(default_lidar.angle_opening[1])


def test_ray_offsets_precomputed_equivalence():
    """Precomputed offsets preserve absolute ray angles from the old linspace.

    Verifies the refactoring preserves ray-angle values across headings,
    allowing only sub-ULP arithmetic-order differences.
    """
    settings = LidarScannerSettings(num_rays=10, visual_angle_portion=1.0, scan_noise=[0.0, 0.0])

    headings = [
        0.0,
        np.pi / 4,
        np.pi,
        -np.pi / 2,
        2 * np.pi,
        -3 * np.pi,
        7 * np.pi / 4,
        -2 * np.pi + np.pi / 6,
    ]

    for heading in headings:
        # Old method: direct linspace from (heading + opening[0]) to (heading + opening[1])
        lower = heading + settings.angle_opening[0]
        upper = heading + settings.angle_opening[1]
        expected = np.linspace(lower, upper, settings.num_rays + 1)[:-1]
        expected = np.array([(a + np.pi * 2) % (np.pi * 2) for a in expected])

        # New method: heading added to precomputed offsets, same wrapping
        actual = heading + settings.ray_offsets
        actual = np.array([(a + np.pi * 2) % (np.pi * 2) for a in actual])

        assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12), (
            f"ray_angles mismatch at heading={heading:.6f}\n"
            f"  expected[0]={expected[0]:.15f}  actual[0]={actual[0]:.15f}\n"
            f"  expected[-1]={expected[-1]:.15f}  actual[-1]={actual[-1]:.15f}"
        )


def test_ray_offsets_length_and_range():
    """Precomputed offsets have correct length and stay within opening bounds."""
    settings = LidarScannerSettings(num_rays=64, visual_angle_portion=0.5, scan_noise=[0.0, 0.0])

    assert len(settings.ray_offsets) == settings.num_rays
    assert np.isclose(settings.ray_offsets[0], settings.angle_opening[0])
    assert settings.ray_offsets[-1] < settings.angle_opening[1]
    assert np.all(settings.ray_offsets >= settings.angle_opening[0])
    assert np.all(settings.ray_offsets <= settings.angle_opening[1])
    assert np.all(np.diff(settings.ray_offsets) > 0)


def test_ray_offsets_rebuilt_on_reinit():
    """Ray offsets are rebuilt when a new settings instance is created."""
    s1 = LidarScannerSettings(num_rays=8, visual_angle_portion=1.0, scan_noise=[0.0, 0.0])
    s2 = LidarScannerSettings(num_rays=16, visual_angle_portion=0.5, scan_noise=[0.0, 0.0])

    assert len(s1.ray_offsets) == 8
    assert len(s2.ray_offsets) == 16
    assert not np.array_equal(s1.ray_offsets, s2.ray_offsets)
    assert s2.ray_offsets[0] > s1.ray_offsets[0]  # narrower FOV has less negative start


def test_lidar_ray_scan_uses_precomputed_offsets():
    """Integration: lidar_ray_scan with precomputed offsets matches old direct linspace."""
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
    settings = LidarScannerSettings(num_rays=4, visual_angle_portion=1.0, scan_noise=[0.0, 0.0])

    for heading in [0.0, np.pi / 3, -np.pi / 2, np.pi, 3 * np.pi / 2]:
        # Old method
        lower = heading + settings.angle_opening[0]
        upper = heading + settings.angle_opening[1]
        expected_angles = np.linspace(lower, upper, settings.num_rays + 1)[:-1]
        expected_angles = np.array([(a + np.pi * 2) % (np.pi * 2) for a in expected_angles])

        # New method through public API
        ranges, actual_angles = lidar_ray_scan(((1.0, 1.0), heading), occ, settings)

        assert np.allclose(actual_angles, expected_angles, atol=1e-12, rtol=1e-12), (
            f"ray_angles mismatch at heading={heading:.6f}"
        )
        assert ranges.shape == (settings.num_rays,)


################################################################################
# lidar_ray_offsets tests


def test_lidar_ray_offsets_endpoint_convention():
    """Cached helper produces the same values as direct relative linspace."""
    num_rays, portion = 4, 0.3
    half = np.pi * portion
    expected = np.linspace(-half, half, num_rays + 1)[:-1]

    result = lidar_ray_offsets(num_rays, portion)
    assert np.allclose(result, expected, atol=1e-12, rtol=1e-12)
    assert result.shape == (num_rays,)


def test_lidar_ray_offsets_cache_identity():
    """Same arguments return the identical cached array object."""
    a = lidar_ray_offsets(8, 0.5)
    b = lidar_ray_offsets(8, 0.5)
    assert a is b


def test_lidar_ray_offsets_different_portion():
    """Different visual_angle_portion produces distinct cached arrays."""
    a = lidar_ray_offsets(8, 0.5)
    b = lidar_ray_offsets(8, 1.0)
    assert not np.array_equal(a, b)


def test_lidar_ray_offsets_read_only():
    """Cached array is read-only; in-place mutation raises ValueError."""
    result = lidar_ray_offsets(8, 0.5)
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        result[0] = 0.0


def test_lidar_ray_scan_orientation_equivalence():
    """Ray angles at zero orientation match normalized cached offsets."""
    obstacle_coords = np.empty((0, 4), dtype=np.float64)
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (0.0, 0.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
    )
    settings = LidarScannerSettings(num_rays=8, visual_angle_portion=1.0, scan_noise=[0.0, 0.0])
    _ranges, ray_angles = lidar_ray_scan(((0.0, 0.0), 0.0), occ, settings)
    expected = np.mod(lidar_ray_offsets(8, 1.0), 2.0 * np.pi)
    assert np.allclose(ray_angles, expected, atol=1e-12, rtol=1e-12)


def test_scan_noise_array_values_match_noise():
    """scan_noise_array element-wise equals the original scan_noise list."""
    settings = LidarScannerSettings(scan_noise=[0.1, 0.2])
    assert np.allclose(settings.scan_noise_array, [0.1, 0.2])
    assert settings.scan_noise_array.dtype == np.float64


def test_scan_noise_array_is_read_only():
    """scan_noise_array is not writeable to protect shared settings state."""
    settings = LidarScannerSettings(scan_noise=[0.005, 0.002])
    assert not settings.scan_noise_array.flags.writeable
    with pytest.raises(ValueError):
        settings.scan_noise_array[0] = 0.0


def test_scan_noise_array_zero_noise_preserved():
    """Zero-noise scan produces unchanged deterministic ranges via scan_noise_array."""
    obstacle_coords = np.array([[3.0, -1.0, 3.0, 1.0]])
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
    )
    settings = LidarScannerSettings(max_scan_dist=10.0, num_rays=4, scan_noise=[0.0, 0.0])
    ranges_1, _ = lidar_ray_scan(((1.0, 1.0), 0.0), occ, settings)
    ranges_2, _ = lidar_ray_scan(((1.0, 1.0), 0.0), occ, settings)
    assert np.array_equal(ranges_1, ranges_2)


def test_lidar_ray_scan_ranges_only_matches_first_element():
    """Ranges-only scan output equals lidar_ray_scan(...)[0] for various poses."""
    obstacle_coords = np.array([[3.0, -1.0, 3.0, 1.0]])
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
    )
    settings = LidarScannerSettings(max_scan_dist=10.0, num_rays=8, scan_noise=[0.0, 0.0])

    for heading in [0.0, np.pi / 4, np.pi, -np.pi / 2, 2.5 * np.pi]:
        pose = ((1.0, 1.0), heading)
        ranges_ref = lidar_ray_scan(pose, occ, settings)[0]
        ranges_only = lidar_ray_scan_ranges_only(pose, occ, settings)
        np.testing.assert_array_equal(ranges_only, ranges_ref)


def test_lidar_ray_scan_ranges_only_thread_local_buffer_reuse():
    """Repeated calls with different headings correctly reuse the thread-local buffer."""
    obstacle_coords = np.array([[3.0, -1.0, 3.0, 1.0]])
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
    )
    settings = LidarScannerSettings(max_scan_dist=10.0, num_rays=4, scan_noise=[0.0, 0.0])

    heading_a = 0.0
    heading_b = np.pi / 3

    ranges_a = lidar_ray_scan_ranges_only(((1.0, 1.0), heading_a), occ, settings)
    ranges_b = lidar_ray_scan_ranges_only(((1.0, 1.0), heading_b), occ, settings)

    ref_a = lidar_ray_scan(((1.0, 1.0), heading_a), occ, settings)[0]
    ref_b = lidar_ray_scan(((1.0, 1.0), heading_b), occ, settings)[0]

    np.testing.assert_array_equal(ranges_a, ref_a)
    np.testing.assert_array_equal(ranges_b, ref_b)


def test_lidar_ray_scan_angles_isolated_from_ranges_only_buffer():
    """Public lidar_ray_scan returned angles remain correct after thread-local buffer mutation."""
    obstacle_coords = np.array([[3.0, -1.0, 3.0, 1.0]])
    ped_coords = np.empty((0, 2), dtype=np.float64)
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: obstacle_coords,
        get_pedestrian_coords=lambda: ped_coords,
    )
    settings = LidarScannerSettings(
        max_scan_dist=10.0,
        num_rays=4,
        visual_angle_portion=1.0,
        scan_noise=[0.0, 0.0],
    )

    heading_1 = 0.0
    heading_2 = np.pi / 2

    # First scan with full API
    _ranges_1, angles_1 = lidar_ray_scan(((1.0, 1.0), heading_1), occ, settings)

    # Ranges-only call mutates the active thread's private buffer.
    lidar_ray_scan_ranges_only(((1.0, 1.0), heading_2), occ, settings)

    # Second full scan — must not alias or be contaminated by that buffer.
    _ranges_2, angles_2 = lidar_ray_scan(((1.0, 1.0), heading_1), occ, settings)

    expected = np.mod(heading_1 + settings.ray_offsets, 2.0 * np.pi)
    np.testing.assert_allclose(angles_1, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(angles_2, expected, atol=1e-12, rtol=1e-12)
