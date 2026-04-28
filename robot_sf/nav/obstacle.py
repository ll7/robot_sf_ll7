"""Obstacle representation and construction for navigation and collision detection.

This module provides the Obstacle dataclass for representing geometric obstacles
in the simulation environment. Obstacles retain a legacy representative vertex
ring for compatibility, but they can now also carry full Shapely polygon or
MultiPolygon geometry for compound members and holes. Line segments are
automatically derived for efficient collision detection and path planning.

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
from itertools import pairwise

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from robot_sf.common.types import Line2D, Vec2D
from robot_sf.nav.nav_types import SvgRectangle


@dataclass
class Obstacle:
    """Represents a geometric obstacle with optional compound polygon geometry.

    Obstacles retain a sequence of legacy representative vertices for compatibility,
    but the canonical obstacle shape may be a Polygon or MultiPolygon. The class
    automatically generates line segments from the full geometry and provides both
    list and numpy array representations for flexible downstream usage.

    Attributes:
        vertices: List of 2D coordinate tuples defining the obstacle boundary vertices.
            Vertices are automatically normalized to tuples to ensure consistent equality
            checks regardless of input format (tuple vs numpy array).
        geometry: Optional Shapely Polygon or MultiPolygon describing the canonical
            obstacle shape. When provided, vertices remain as a representative exterior
            ring for compatibility with legacy callers.
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
    geometry: Polygon | MultiPolygon | None = field(default=None, repr=False, compare=False)
    lines: list[Line2D] = field(init=False)
    vertices_np: np.ndarray = field(init=False)

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

        if self.geometry is not None:
            self.geometry = self._normalize_geometry(self.geometry)
            if self.geometry is not None and len(self.vertices) < 3:
                self.vertices = self._representative_vertices(self.geometry)
                self.vertices_np = np.array(self.vertices)

        if self.geometry is None and len(self.vertices) >= 3:
            self.geometry = Polygon(self.vertices)

        self.lines = (
            self._lines_from_geometry()
            if self.geometry is not None
            else self._lines_from_vertices()
        )

    def contains_point(self, point: Vec2D) -> bool:
        """Return True if the point lies inside the obstacle polygon."""
        if self.geometry is None or self.geometry.is_empty:
            return False
        return self.geometry.contains(Point(point))

    def iter_polygons(self) -> list[Polygon]:
        """Return the polygon components that make up this obstacle."""
        if self.geometry is None or self.geometry.is_empty:
            return []
        if isinstance(self.geometry, Polygon):
            return [self.geometry]
        if isinstance(self.geometry, MultiPolygon):
            return [poly for poly in self.geometry.geoms if not poly.is_empty]
        geoms = getattr(self.geometry, "geoms", None)
        if geoms is None:
            return []
        return [geom for geom in geoms if isinstance(geom, Polygon) and not geom.is_empty]

    @classmethod
    def from_geometry(
        cls,
        geometry: Polygon | MultiPolygon,
        *,
        representative_vertices: list[Vec2D] | None = None,
    ) -> "Obstacle":
        """Build an obstacle from a Shapely geometry while keeping legacy vertices.

        Returns:
            Obstacle: Compatibility wrapper around the supplied geometry.
        """
        vertices = representative_vertices
        if vertices is None:
            vertices = cls._representative_vertices(geometry)
        return cls(vertices=vertices, geometry=geometry)

    @staticmethod
    def _normalize_geometry(geometry: Polygon | MultiPolygon) -> Polygon | MultiPolygon | None:
        if geometry.is_empty:
            return None
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry
        return None

    @staticmethod
    def _representative_vertices(geometry: Polygon | MultiPolygon) -> list[Vec2D]:
        polygons: list[Polygon]
        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = [poly for poly in geometry.geoms if not poly.is_empty]
        else:
            polygons = []
        if not polygons:
            return []
        exterior = list(polygons[0].exterior.coords)[:-1]
        return [tuple(vertex) for vertex in exterior]

    @staticmethod
    def _ring_to_lines(ring) -> list[Line2D]:
        coords = [tuple(point) for point in ring.coords]
        if len(coords) < 2:
            return []
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 2:
            return []
        edges = [*pairwise(coords), (coords[-1], coords[0])]
        return [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges if p1 != p2]

    def _lines_from_vertices(self) -> list[Line2D]:
        edges = [*pairwise(self.vertices), (self.vertices[-1], self.vertices[0])]
        edges = list(filter(lambda edge: edge[0] != edge[1], edges))
        return [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]

    def _lines_from_geometry(self) -> list[Line2D]:
        lines: list[Line2D] = []
        for polygon in self.iter_polygons():
            lines.extend(self._ring_to_lines(polygon.exterior))
            for interior in polygon.interiors:
                lines.extend(self._ring_to_lines(interior))
        return lines


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
