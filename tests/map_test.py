import numpy as np
from robot_sf.map import BinaryOccupancyGrid


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


def test_is_collision():
    pass
