import numpy as np
from robot_sf.map import BinaryOccupancyGrid
from robot_sf.vector import Vec2D


def test_create_map():
    _map = BinaryOccupancyGrid(40, 40, 10, 10, lambda: np.array([]), lambda: np.array([]))
    assert _map is not None


def test_is_in_bounds():
    _map = BinaryOccupancyGrid(40, 40, 10, 10, lambda: np.array([]), lambda: np.array([]))
    assert _map.is_in_bounds(-10, -10) == True
    assert _map.is_in_bounds(10, -10) == True
    assert _map.is_in_bounds(-10, 10) == True
    assert _map.is_in_bounds(10, 10) == True
    assert _map.is_in_bounds(0, 0) == True
    assert _map.is_in_bounds(-5, -5) == True
    assert _map.is_in_bounds(5, -5) == True
    assert _map.is_in_bounds(-5, 5) == True
    assert _map.is_in_bounds(5, 5) == True


def test_is_out_of_bounds():
    _map = BinaryOccupancyGrid(40, 40, 10, 10, lambda: np.array([]), lambda: np.array([]))
    assert not _map.is_in_bounds(10.0000001, 10)
    assert not _map.is_in_bounds(10, 10.0000001)
    assert not _map.is_in_bounds(-10.0000001, 10)
    assert not _map.is_in_bounds(10, -10.0000001)


def test_is_collision_with_obstacle():
    obstacle_pos = np.random.uniform(-10, 10, size=(2))
    robot_pos = Vec2D(obstacle_pos[0], obstacle_pos[1])
    _map = BinaryOccupancyGrid(40, 40, 10, 10, lambda: [obstacle_pos], lambda: np.array([]))
    assert _map.is_obstacle_collision(robot_pos, 2)


def test_is_collision_with_pedestrian():
    ped_pos = np.random.uniform(-10, 10, size=(2))
    robot_pos = Vec2D(ped_pos[0], ped_pos[1])
    _map = BinaryOccupancyGrid(40, 40, 10, 10, lambda: np.array([]), lambda: np.array([ped_pos]))
    assert _map.is_pedestrians_collision(robot_pos, 2)
