
from dataclasses import dataclass, field
from typing import List
import numpy as np
from robot_sf.nav.nav_types import Vec2D, Line2D, SvgRectangle


@dataclass
class Obstacle:
    """
    A class to represent an obstacle.

    Attributes
    ----------
    vertices : List[Vec2D]
        The vertices of the obstacle.
    lines : List[Line2D]
        The lines that make up the obstacle. This is calculated in the post-init method.
    vertices_np : np.ndarray
        The vertices of the obstacle as a numpy array. This is calculated in the post-init method.

    Methods
    -------
    __post_init__():
        Validates and processes the vertices to create the lines and vertices_np attributes.
    """

    vertices: List[Vec2D]
    lines: List[Line2D] = field(init=False)
    vertices_np: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Validates and processes the vertices to create the lines and vertices_np
        attributes.
        Raises a ValueError if no vertices are specified.
        """

        if not self.vertices:
            raise ValueError('No vertices specified for obstacle!')

        # Convert vertices to numpy array
        self.vertices_np = np.array(self.vertices)

        # Create edges from vertices
        edges = list(zip(self.vertices[:-1], self.vertices[1:])) \
            + [(self.vertices[-1], self.vertices[0])]

        # Remove fake lines that are just points
        edges = list(filter(lambda l: l[0] != l[1], edges))

        # Create lines from edges
        lines = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = lines

        if not self.vertices:
            print('WARNING: obstacle is just a single point that cannot collide!')


def obstacle_from_svgrectangle(svg_rectangle: SvgRectangle) -> Obstacle:
    """
    Creates an obstacle from an SVG rectangle.

    Parameters
    ----------
    svg_rectangle : SvgRectangle
        The SVG rectangle to create the obstacle from.

    Returns
    -------
    Obstacle
        The obstacle created from the SVG rectangle.
    """

    return Obstacle([
        (svg_rectangle.x, svg_rectangle.y),
        (svg_rectangle.x + svg_rectangle.width, svg_rectangle.y),
        (svg_rectangle.x + svg_rectangle.width, svg_rectangle.y + svg_rectangle.height),
        (svg_rectangle.x, svg_rectangle.y + svg_rectangle.height)
        ])
