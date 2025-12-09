"""Zone sampling utilities with optional obstacle avoidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Point as _ShapelyPoint
from shapely.geometry import Polygon as _ShapelyPolygon
from shapely.prepared import PreparedGeometry, prep

if TYPE_CHECKING:
    from robot_sf.common.types import Vec2D, Zone


def sample_zone(
    zone: Zone,
    num_samples: int,
    obstacle_polygons: list[list[Vec2D]] | list[PreparedGeometry] | None = None,
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
    prepared_polygons = prepare_obstacle_polygons(obstacle_polygons or [])
    a, b, c = zone
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec_ba, vec_bc = a - b, c - b

    samples: list[Vec2D] = []
    attempts = 0
    max_attempts = max_attempts_per_point * max(num_samples, 1)
    batch_size = max(num_samples * 2, 4)

    while len(samples) < num_samples and attempts < max_attempts:
        remaining = num_samples - len(samples)
        current_batch = max(batch_size, remaining)
        rel_width = np.random.uniform(0, 1, current_batch)
        rel_height = np.random.uniform(0, 1, current_batch)
        fold_mask = rel_width + rel_height > 1.0
        rel_width[fold_mask] = 1.0 - rel_width[fold_mask]
        rel_height[fold_mask] = 1.0 - rel_height[fold_mask]
        points = b + rel_width[:, None] * vec_ba + rel_height[:, None] * vec_bc
        candidates = [(float(x), float(y)) for x, y in points]
        attempts += current_batch

        if prepared_polygons:
            keep_mask = _points_outside_obstacles(candidates, prepared_polygons)
            filtered = [pt for pt, keep in zip(candidates, keep_mask, strict=False) if keep]
        else:
            filtered = candidates

        samples.extend(filtered)

    if len(samples) < num_samples:
        raise RuntimeError(
            f"Failed to sample {num_samples} points in zone without obstacle overlap "
            f"after {attempts} attempts.",
        )
    return samples[:num_samples]


def prepare_obstacle_polygons(
    obstacle_polygons: list[list[Vec2D]] | list[PreparedGeometry],
) -> list[PreparedGeometry]:
    """Normalize obstacles to prepared shapely geometries.

    Returns:
        list[PreparedGeometry]: Prepared polygons ready for fast containment checks.
    """
    prepared: list[PreparedGeometry] = []
    for poly in obstacle_polygons:
        if isinstance(poly, PreparedGeometry):
            prepared.append(poly)
        elif isinstance(poly, _ShapelyPolygon):
            prepared.append(prep(poly))
        else:
            prepared.append(prep(_ShapelyPolygon(poly)))
    return prepared


def _point_in_any_obstacle(point: Vec2D, obstacle_polygons: list[PreparedGeometry]) -> bool:
    """Return True if point lies inside any polygon."""
    pt = _ShapelyPoint(point)
    return any(poly.contains(pt) for poly in obstacle_polygons)


def _points_outside_obstacles(
    points: list[Vec2D], obstacle_polygons: list[PreparedGeometry]
) -> list[bool]:
    """Return mask marking points that do not intersect any obstacle."""
    if not obstacle_polygons:
        return [True] * len(points)
    mask = [True] * len(points)
    for idx, pt in enumerate(points):
        shp_pt = _ShapelyPoint(pt)
        if any(poly.contains(shp_pt) for poly in obstacle_polygons):
            mask[idx] = False
    return mask
