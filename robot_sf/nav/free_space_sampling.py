"""Utilities for sampling free-space positions across an entire map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Point as _ShapelyPoint

from robot_sf.ped_npc.ped_zone import prepare_obstacle_polygons

if TYPE_CHECKING:
    from collections.abc import Iterable

    from shapely.prepared import PreparedGeometry

    from robot_sf.common.types import Vec2D


def _point_outside_obstacles(point: Vec2D, prepared_obstacles: list[PreparedGeometry]) -> bool:
    """Return True when a point is not contained in any prepared obstacle polygon."""
    shp_pt = _ShapelyPoint(point)
    return not any(poly.contains(shp_pt) for poly in prepared_obstacles)


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
        obstacle_polygons: Optional list of polygons (vertex lists or prepared) to avoid.
        max_attempts_per_point: Attempts per requested sample before giving up.
        rng: Optional seedable RNG. When None, uses NumPy's legacy global random stream.

    Returns:
        List of sampled points as (x, y) tuples outside obstacles.

    Raises:
        RuntimeError: If sampling fails to produce the requested number of points.
    """
    random_uniform = np.random.uniform if rng is None else rng.uniform

    x_min, x_max, y_min, y_max = bounds
    prepared = prepare_obstacle_polygons(list(obstacle_polygons or []))

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
        candidates = list(zip(xs, ys, strict=False))

        if prepared:
            filtered = [
                (float(x), float(y))
                for x, y in candidates
                if _point_outside_obstacles((x, y), prepared)
            ]
        else:
            filtered = [(float(x), float(y)) for x, y in candidates]

        samples.extend(filtered)

    if len(samples) < num_samples:
        raise RuntimeError(
            f"Failed to sample {num_samples} free-space point(s) within bounds "
            f"after {attempts} attempts.",
        )

    return samples[:num_samples]


__all__ = ["sample_free_points_in_bounds"]
