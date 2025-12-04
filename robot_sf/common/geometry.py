"""Geometry utilities shared across the robot_sf package."""

import math

import numba

from robot_sf.common.types import Vec2D


@numba.njit(fastmath=True)
def euclid_dist(vec_1: Vec2D, vec_2: Vec2D) -> float:
    """
    Compute the Euclidean distance between two 2D vectors.

    The helper is numba-jitted because it sits in hot paths across sensors,
    occupancy checks, and pedestrian force calculations.

    Args:
        vec_1: First 2D vector.
        vec_2: Second 2D vector.

    Returns:
        float: Euclidean distance between ``vec_1`` and ``vec_2``.
    """
    return math.hypot(vec_1[0] - vec_2[0], vec_1[1] - vec_2[1])
