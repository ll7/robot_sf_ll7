import numpy as np
from robot_sf.map import BinaryOccupancyGrid


def test_create_map():
    _map = BinaryOccupancyGrid(40, 40, 10, 10, lambda: np.array([]), lambda: np.array([]))
    assert True
