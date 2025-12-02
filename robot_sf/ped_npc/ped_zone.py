"""Module ped_zone auto-generated docstring."""

import numpy as np

from robot_sf.common.types import Vec2D, Zone


def sample_zone(zone: Zone, num_samples: int) -> list[Vec2D]:
    """
    Generate a specified number of random sample points within a zone.

    The zone is defined as a triangle with vertices a, b, and c. The points are generated
    using uniform random sampling.

    Parameters
    ----------
    zone : Zone
        The zone within which to generate sample points. This is a tuple of three 2D
        vectors representing the vertices of the triangle.
    num_samples : int
        The number of sample points to generate.

    Returns
    -------
    List[Vec2D]
        A list of the generated sample points. Each point is a tuple of two numbers
        representing the x and y coordinates.
    """
    # Unpack the vertices of the triangle
    a, b, c = zone
    # Convert the vertices to numpy arrays for vectorized computation
    a, b, c = np.array(a), np.array(b), np.array(c)
    # Compute the vectors from vertex b to vertices a and c
    vec_ba, vec_bc = a - b, c - b
    # Generate random numbers for the relative width and height of the points
    rel_width = np.random.uniform(0, 1, (num_samples, 1))
    rel_height = np.random.uniform(0, 1, (num_samples, 1))
    # Compute the coordinates of the points
    points = b + rel_width * vec_ba + rel_height * vec_bc
    # Convert the points to a list of tuples and return it
    return [(x, y) for x, y in points]
