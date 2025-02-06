"""
This module implements the rendering of the lidar sensor.
"""

import numpy as np
from typing import Tuple

Vec2D = Tuple[float, float]


def render_lidar(
    robot_pos: Vec2D, distances: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """
    Render the LIDAR scanning visualization using vectorized NumPy operations.

    Args:
        robot_pos: Robot position in the environment.
        distances: Distances to obstacles detected by the LIDAR.
        directions: Angles at which the LIDAR rays are cast.

    Returns:
        ray_vecs_np: (N, 2, 2) array of ray start and end point positions.
    """
    # Compute the endpoints for each ray visualization
    x_offsets = np.cos(directions) * distances
    y_offsets = np.sin(directions) * distances

    # Starting points are all the robot_pos
    n = len(distances)
    start_points = np.repeat(np.array(robot_pos)[np.newaxis, :], n, axis=0)
    # Compute the end points by adding displacements to the robot's position
    end_points = start_points + np.column_stack((x_offsets, y_offsets))
    # Stack the start and end points along a new axis (axis=1)
    ray_vecs_np = np.stack((start_points, end_points), axis=1)

    return ray_vecs_np
