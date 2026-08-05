"""Utilities for sampling free-space positions across an entire map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely import contains_xy as _shp_contains_xy
from shapely.geometry import Polygon as _ShapelyPolygon
from shapely.prepared import PreparedGeometry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.common.types import Vec2D


def _as_shapely_polygons(obstacle_polygons: Iterable) -> list[_ShapelyPolygon]:
    """Normalize obstacle inputs to a list of Shapely Polygon geometries.

    Returns:
        List of Shapely Polygon geometries ready for vectorized containment checks.
    """
    polygons: list[_ShapelyPolygon] = []
    for poly in obstacle_polygons:
        if isinstance(poly, _ShapelyPolygon):
            polygons.append(poly)
        elif isinstance(poly, PreparedGeometry):
            polygons.append(poly.context)
        else:
            polygons.append(_ShapelyPolygon(poly))
    return polygons


def sample_free_points_in_bounds(
    bounds: tuple[float, float, float, float],
    num_samples: int,
    obstacle_polygons: Iterable | None = None,
    max_attempts_per_point: int = 50,
    rng: np.random.Generator | None = None,
) -> list[Vec2D]:
    """Sample points uniformly within map bounds while rejecting obstacle intersections.

    Args:
        bounds: Axis-aligned bounding box as (x_min, x_max, y_min, y_max).
        num_samples: Number of points to sample.
        obstacle_polygons: Optional list of polygons (vertex lists or Shapely) to avoid.
        max_attempts_per_point: Attempts per requested sample before giving up.
        rng: Optional seedable RNG. When None, uses NumPy's legacy global random stream.

    Returns:
        List of sampled points as (x, y) tuples outside obstacles.

    Raises:
        RuntimeError: If sampling fails to produce the requested number of points.
    """
    random_uniform = np.random.uniform if rng is None else rng.uniform

    x_min, x_max, y_min, y_max = bounds
    shapely_polys = _as_shapely_polygons(list(obstacle_polygons or []))

    samples: list[Vec2D] = []
    attempts = 0
    max_attempts = max_attempts_per_point * max(num_samples, 1)
    batch_size = max(num_samples * 2, 4)

    while len(samples) < num_samples and attempts < max_attempts:
        remaining = num_samples - len(samples)
        current_batch = max(batch_size, remaining)
        xs = random_uniform(x_min, x_max, current_batch)
        ys = random_uniform(y_min, y_max, current_batch)
        attempts += current_batch

        if shapely_polys:
            # Vectorized point-in-polygon via Shapely 2.x contains_xy (issue #6493)
            inside_any = np.zeros(current_batch, dtype=bool)
            for poly in shapely_polys:
                inside_any |= _shp_contains_xy(poly, xs, ys)
            free_mask = ~inside_any
            free_xs = xs[free_mask]
            free_ys = ys[free_mask]
            filtered = list(zip(free_xs.tolist(), free_ys.tolist(), strict=False))
        else:
            filtered = list(zip(xs.tolist(), ys.tolist(), strict=False))

        samples.extend(filtered)

    if len(samples) < num_samples:
        raise RuntimeError(
            f"Failed to sample {num_samples} free-space point(s) within bounds "
            f"after {attempts} attempts.",
        )

    return samples[:num_samples]


__all__ = ["sample_free_points_in_bounds"]
