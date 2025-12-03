"""TODO docstring. Document this module."""

from dataclasses import dataclass, field

import numpy as np

from robot_sf.common.types import Line2D, Vec2D
from robot_sf.nav.nav_types import SvgRectangle


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

    vertices: list[Vec2D]
    lines: list[Line2D] = field(init=False)
    vertices_np: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Validates and processes the vertices to create the lines and vertices_np
        attributes.
        Raises a ValueError if no vertices are specified.
        """

        if not self.vertices:
            raise ValueError("No vertices specified for obstacle!")

        # Normalize potential numpy array vertices (e.g., from path based obstacles) to tuples
        # to ensure downstream equality checks (edge[0] != edge[1]) operate on plain Python
        # tuples instead of broadcasting numpy arrays which leads to the ambiguous truth value
        # error: "The truth value of an array with more than one element is ambiguous".
        # This keeps the public API unchanged while hardening against SVG path derived
        # obstacle vertices provided as np.ndarray rows.
        self.vertices = [tuple(v) if not isinstance(v, tuple) else v for v in self.vertices]

        # Convert vertices to numpy array
        self.vertices_np = np.array(self.vertices)

        # Create edges from vertices
        edges = [
            *list(zip(self.vertices[:-1], self.vertices[1:], strict=False)),
            (self.vertices[-1], self.vertices[0]),
        ]

        # Remove fake lines that are just points
        edges = list(filter(lambda edge: edge[0] != edge[1], edges))

        # Create lines from edges
        lines = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = lines

        if not self.vertices:
            pass


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

    return Obstacle(
        [
            (svg_rectangle.x, svg_rectangle.y),
            (svg_rectangle.x + svg_rectangle.width, svg_rectangle.y),
            (svg_rectangle.x + svg_rectangle.width, svg_rectangle.y + svg_rectangle.height),
            (svg_rectangle.x, svg_rectangle.y + svg_rectangle.height),
        ],
    )
