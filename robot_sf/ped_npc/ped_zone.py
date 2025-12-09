"""Zone sampling utilities with optional obstacle avoidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_sf.common.types import Vec2D, Zone

try:  # Optional acceleration when shapely is available
    from shapely.geometry import Point as _ShapelyPoint
    from shapely.geometry import Polygon as _ShapelyPolygon

    _SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _SHAPELY_AVAILABLE = False
    _ShapelyPoint = None  # type: ignore[assignment]
    _ShapelyPolygon = None  # type: ignore[assignment]


def sample_zone(
    zone: Zone,
    num_samples: int,
    obstacle_polygons: list[list[Vec2D]] | None = None,
    max_attempts_per_point: int = 20,
) -> list[Vec2D]:
    """
    Generate random sample points within a triangular zone, avoiding obstacles when provided.

    Args:
        zone: Triangle zone vertices.
        num_samples: Number of points to sample.
        obstacle_polygons: Optional list of polygon vertex lists to reject points inside.
        max_attempts_per_point: Attempts before giving up per requested sample.

    Returns:
        list[Vec2D]: Sampled points that do not intersect obstacles.
    """
    obstacle_polygons = obstacle_polygons or []
    a, b, c = zone
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec_ba, vec_bc = a - b, c - b

    samples: list[Vec2D] = []
    attempts = 0
    max_attempts = max_attempts_per_point * max(num_samples, 1)
    while len(samples) < num_samples and attempts < max_attempts:
        rel_width = np.random.uniform(0, 1)
        rel_height = np.random.uniform(0, 1)
        point = b + rel_width * vec_ba + rel_height * vec_bc
        pt_tuple = (float(point[0]), float(point[1]))
        attempts += 1
        if obstacle_polygons and _point_in_any_obstacle(pt_tuple, obstacle_polygons):
            continue
        samples.append(pt_tuple)

    if len(samples) < num_samples:
        raise RuntimeError(
            f"Failed to sample {num_samples} points in zone without obstacle overlap "
            f"after {attempts} attempts.",
        )
    return samples


def _point_in_any_obstacle(point: Vec2D, obstacle_polygons: list[list[Vec2D]]) -> bool:
    """Return True if point lies inside any polygon."""
    if _SHAPELY_AVAILABLE:
        pt = _ShapelyPoint(point)
        for poly_vertices in obstacle_polygons:
            poly = _ShapelyPolygon(poly_vertices)
            if poly.contains(pt):
                return True
        return False

    return any(_point_in_polygon(point, poly) for poly in obstacle_polygons)


def _point_in_polygon(point: Vec2D, polygon: list[Vec2D]) -> bool:
    """Ray casting point-in-polygon check.

    Returns:
        bool: True when the point lies strictly inside the polygon.
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        intersects = ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
        if intersects:
            inside = not inside
    return inside
