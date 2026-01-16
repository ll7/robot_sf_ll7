"""Utilities for sampling obstacle geometry into point clouds for metrics."""

from __future__ import annotations

from math import ceil, hypot
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.nav.obstacle import Obstacle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.common.types import Line2D


def _iter_line_segments(
    obstacles: Iterable[Obstacle | Line2D],
    bounds: Iterable[Line2D] | None = None,
) -> list[Line2D]:
    segments: list[Line2D] = []

    def _add_line(line: Line2D) -> None:
        try:
            (x1, y1), (x2, y2) = line
            segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
        except (TypeError, ValueError):
            return

    for obstacle in obstacles:
        if isinstance(obstacle, Obstacle):
            for line in obstacle.lines:
                if len(line) == 4:
                    x1, x2, y1, y2 = line
                    _add_line(((x1, y1), (x2, y2)))
                else:
                    _add_line(line)  # type: ignore[arg-type]
        else:
            _add_line(obstacle)

    if bounds is not None:
        for bound in bounds:
            _add_line(bound)

    return segments


def sample_obstacle_points(
    obstacles: Iterable[Obstacle | Line2D],
    bounds: Iterable[Line2D] | None = None,
    *,
    spacing: float = 0.5,
) -> np.ndarray:
    """Sample obstacles/bounds into a dense point cloud.

    Args:
        obstacles: Obstacle objects or line segments.
        bounds: Optional map bounds (line segments).
        spacing: Approximate spacing between consecutive points along a segment.

    Returns:
        (M, 2) array of obstacle points (float). Empty array when no geometry.
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    segments = _iter_line_segments(obstacles, bounds=bounds)
    if not segments:
        return np.zeros((0, 2), dtype=float)

    points: list[tuple[float, float]] = []
    for (x1, y1), (x2, y2) in segments:
        length = hypot(x2 - x1, y2 - y1)
        if length <= 1e-9:
            points.append((x1, y1))
            continue
        n = max(2, ceil(length / spacing) + 1)
        for i in range(n):
            t = i / (n - 1)
            points.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))

    return np.asarray(points, dtype=float)


__all__ = ["sample_obstacle_points"]
