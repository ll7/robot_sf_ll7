from math import dist
from typing import Tuple
import numpy as np
from robot_sf.occupancy import ContinuousOccupancy

Vec2D = Tuple[float, float]


def test_create_map():
    _map = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: np.array([[]]))
    assert _map is not None


def test_is_in_bounds():
    _map = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([[]]), lambda: np.array([[]]))
    assert _map.is_in_bounds(0, 0) == True
    assert _map.is_in_bounds(10, 0) == True
    assert _map.is_in_bounds(0, 10) == True
    assert _map.is_in_bounds(10, 10) == True
    assert _map.is_in_bounds(0, 0) == True
    assert _map.is_in_bounds(5, 5) == True


def test_is_out_of_bounds():
    _map = ContinuousOccupancy(10, 10, lambda: None, lambda: None, lambda: np.array([]), lambda: np.array([[]]))
    assert not _map.is_in_bounds(10.0000001, 10)
    assert not _map.is_in_bounds(10, 10.0000001)
    assert not _map.is_in_bounds(-10.0000001, 10)
    assert not _map.is_in_bounds(10, -10.0000001)


def test_is_collision_with_obstacle_segment_fully_contained_inside_circle():
    obstacle_pos = np.random.uniform(-10, 10, size=(1, 4))
    robot_pos = (obstacle_pos[0, 0], obstacle_pos[0, 1])
    _map = ContinuousOccupancy(10, 10, lambda: robot_pos, lambda: None, lambda: obstacle_pos, lambda: np.array([[20, 20]]), robot_radius=2)
    assert _map.is_robot_collision


def test_is_collision_with_obstacle_segment_outside_circle():
    obstacle_pos = np.random.uniform(-10, 10, size=(1, 4))
    middle = np.squeeze((obstacle_pos[0: :2] + obstacle_pos[0: 2:]) / 2)
    robot_pos = (middle[0], middle[1])
    radius = dist(obstacle_pos[0, :2], obstacle_pos[0, 2:]) / 2.1
    _map = ContinuousOccupancy(10, 10, lambda: robot_pos, lambda: None, lambda: obstacle_pos, lambda: np.array([[20, 20]]), radius)
    assert _map.is_robot_collision


def test_is_collision_with_pedestrian():
    ped_pos = np.random.uniform(-10, 10, size=(2))
    robot_pos = (ped_pos[0], ped_pos[1])
    _map = ContinuousOccupancy(40, 40, lambda: robot_pos, lambda: None, lambda: np.array([[]]), lambda: np.array([ped_pos]), robot_radius=2)
    assert _map.is_robot_collision
