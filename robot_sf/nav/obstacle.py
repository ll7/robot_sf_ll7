"""Obstacle representation and construction for navigation and collision detection.

This module provides the Obstacle dataclass for representing geometric obstacles
in the simulation environment. Obstacles are defined by vertices forming a closed
polygon and are automatically converted to line segments for efficient collision
detection and path planning.

Key features:
- Vertex-based obstacle definition with automatic edge generation
- Support for SVG rectangle conversion to obstacle polygons
- Numpy array integration for efficient geometric computations
- Automatic deduplication of degenerate edges (point-to-point)
- Normalization of vertex formats (tuple/array) for consistent equality checks

Typical usage:
    # Create obstacle from vertices
    obstacle = Obstacle([(0, 0), (10, 0), (10, 10), (0, 10)])

    # Create from SVG rectangle
    rect = SvgRectangle(x=5, y=5, width=20, height=15)
    obstacle = obstacle_from_svgrectangle(rect)
"""

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon

from robot_sf.common.types import Line2D, Vec2D
from robot_sf.nav.nav_types import SvgRectangle


@dataclass
class Obstacle:
    """Represents a geometric obstacle as a closed polygon.

    Obstacles are defined by a sequence of vertices forming a closed polygon boundary.
    The class automatically generates line segments between consecutive vertices and
    provides both list and numpy array representations for flexible downstream usage.

    Attributes:
        vertices: List of 2D coordinate tuples defining the obstacle boundary vertices.
            Vertices are automatically normalized to tuples to ensure consistent equality
            checks regardless of input format (tuple vs numpy array).
        lines: Line segments forming the obstacle edges, computed automatically from
            vertices. Degenerate edges (point-to-point) are filtered out. Each line is
            represented as (x1, x2, y1, y2).
        vertices_np: Vertices as a numpy array for efficient geometric operations.
            Shape is (n_vertices, 2).

    Raises:
        ValueError: If vertices list is empty during initialization.

    Example:
        >>> obstacle = Obstacle([(0, 0), (10, 0), (10, 10), (0, 10)])
        >>> len(obstacle.lines)  # 4 edges for a square
        4
    """

    vertices: list[Vec2D]
    lines: list[Line2D] = field(init=False)
    vertices_np: np.ndarray = field(init=False)
    _polygon: Polygon | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate and process vertices to generate lines and numpy array representation.

        This method is automatically called after dataclass initialization. It performs:
        1. Validation that vertices list is non-empty
        2. Normalization of vertex format (converts numpy arrays to tuples)
        3. Conversion to numpy array for efficient operations
        4. Edge generation from consecutive vertex pairs
        5. Degenerate edge filtering (removes point-to-point edges)

        Raises:
            ValueError: If the vertices list is empty.

        Note:
            Vertex normalization ensures that equality checks (edge[0] != edge[1])
            operate on plain Python tuples instead of broadcasting numpy arrays,
            preventing "ambiguous truth value" errors from SVG path-derived vertices.
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

        self._polygon = Polygon(self.vertices) if len(self.vertices) >= 3 else None

    def contains_point(self, point: Vec2D) -> bool:
        """Return True if the point lies inside the obstacle polygon."""
        if self._polygon is None:
            return False
        return self._polygon.contains(Point(point))


def obstacle_from_svgrectangle(svg_rectangle: SvgRectangle) -> Obstacle:
    """Create a rectangular obstacle from an SVG rectangle specification.

    Converts an SVG rectangle (defined by x, y, width, height) into a closed
    polygon obstacle with four vertices corresponding to the rectangle corners.
    Vertices are ordered counter-clockwise starting from the top-left corner.

    Args:
        svg_rectangle: SVG rectangle specification containing position (x, y) and
            dimensions (width, height).

    Returns:
        Obstacle instance representing the rectangular boundary with four vertices
        and four edge line segments.

    Example:
        >>> rect = SvgRectangle(x=10, y=20, width=50, height=30)
        >>> obstacle = obstacle_from_svgrectangle(rect)
        >>> len(obstacle.vertices)
        4
    """

    return Obstacle(
        [
            (svg_rectangle.x, svg_rectangle.y),
            (svg_rectangle.x + svg_rectangle.width, svg_rectangle.y),
            (svg_rectangle.x + svg_rectangle.width, svg_rectangle.y + svg_rectangle.height),
            (svg_rectangle.x, svg_rectangle.y + svg_rectangle.height),
        ],
    )
