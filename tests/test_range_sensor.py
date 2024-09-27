import numpy as np
from robot_sf.sensor.range_sensor import (
    circle_line_intersection_distance,
    euclid_dist,
    raycast_pedestrians
    )

# Circle-line intersection tests


def test_intersection_at_origin():
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (0.0, 0.0)  # Ray starts at origin
    ray_vec = (1.0, 0.0)  # Ray points along the x-axis
    assert circle_line_intersection_distance(circle, origin, ray_vec) == 1.0


def test_no_intersection():
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (2.0, 2.0)  # Ray starts outside the circle
    ray_vec = (1.0, 0.0)  # Ray points along the x-axis
    assert circle_line_intersection_distance(circle, origin, ray_vec) == float('inf')


def test_intersection_at_circle_perimeter():
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (0.0, 0.0)  # Ray starts at origin
    ray_vec = (1.0, 1.0)  # Ray points diagonally
    assert circle_line_intersection_distance(circle, origin, ray_vec) == 1.0


def test_negative_ray_direction():
    circle = ((0.0, 0.0), 1.0)  # Circle centered at origin with radius 1
    origin = (1.0, 0.0)  # Ray starts at x=1
    ray_vec = (-1.0, 0.0)  # Ray points along the negative x-axis
    assert circle_line_intersection_distance(circle, origin, ray_vec) == 0.0

################################################################################
# Euclidean distance tests


def test_same_point():
    vec_1 = (0.0, 0.0)
    vec_2 = (0.0, 0.0)
    assert euclid_dist(vec_1, vec_2) == 0.0


def test_unit_distance():
    vec_1 = (0.0, 0.0)
    vec_2 = (1.0, 0.0)
    assert euclid_dist(vec_1, vec_2) == 1.0


def test_negative_coordinates():
    vec_1 = (0.0, 0.0)
    vec_2 = (-1.0, -1.0)
    assert euclid_dist(vec_1, vec_2) == (2**0.5)


def test_non_integer_distance():
    vec_1 = (0.0, 0.0)
    vec_2 = (1.0, 1.0)
    assert euclid_dist(vec_1, vec_2) == (2**0.5)

################################################################################
# Raycasting pedestrians tests
# def test_no_pedestrians():
#     out_ranges = np.array([10.0, 10.0])
#     scanner_pos = (0.0, 0.0)
#     max_scan_range = 10.0
#     ped_positions = np.array([], dtype=np.float64)
#     ped_radius = 1.0
#     ray_angles = np.array([0.0, np.pi / 2])

#     raycast_pedestrians(
#         out_ranges,
#         scanner_pos,
#         max_scan_range,
#         ped_positions,
#         ped_radius,
#         ray_angles
#         )

#     assert np.all(out_ranges == 10.0)
# TODO testing with no pedestrians did not work


def test_pedestrian_in_range():
    out_ranges = np.array([10.0, 10.0])
    scanner_pos = (0.0, 0.0)
    max_scan_range = 10.0
    ped_positions = np.array([[5.0, 0.0]])
    ped_radius = 1.0
    ray_angles = np.array([0.0, np.pi / 2])

    raycast_pedestrians(
        out_ranges,
        scanner_pos,
        max_scan_range,
        ped_positions,
        ped_radius,
        ray_angles)

    assert out_ranges[0] == 4.0
    assert out_ranges[1] == 10.0


def test_pedestrian_out_of_range():
    out_ranges = np.array([10.0, 10.0])
    scanner_pos = (0.0, 0.0)
    max_scan_range = 10.0
    ped_positions = np.array([[15.0, 0.0]])
    ped_radius = 1.0
    ray_angles = np.array([0.0, np.pi / 2])

    raycast_pedestrians(
        out_ranges,
        scanner_pos,
        max_scan_range,
        ped_positions,
        ped_radius,
        ray_angles)

    assert np.all(out_ranges == 10.0)
