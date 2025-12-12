"""Path smoothing utilities (Douglas-Peucker)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.common.types import Vec2D


def douglas_peucker(path: Sequence[Vec2D], epsilon: float) -> list[Vec2D]:
    """Simplify a polyline using the Douglas-Peucker algorithm.

    Args:
        path: Sequence of waypoints.
        epsilon: Tolerance; higher values yield fewer points.

    Returns:
        Simplified path preserving endpoints.
    """
    if len(path) <= 2:
        return list(path)
    if epsilon <= 0:
        return list(path)

    pts = np.array(path, dtype=float)

    def _recursive(start_idx: int, end_idx: int, keep: list[int]) -> None:
        segment = pts[end_idx] - pts[start_idx]
        seg_len = np.linalg.norm(segment)
        if seg_len == 0:
            return
        unit = segment / seg_len
        max_dist = -1.0
        max_idx = None
        for idx in range(start_idx + 1, end_idx):
            vector = pts[idx] - pts[start_idx]
            projection = np.dot(vector, unit)
            nearest = pts[start_idx] + projection * unit
            dist = np.linalg.norm(pts[idx] - nearest)
            if dist > max_dist:
                max_dist = dist
                max_idx = idx
        if max_dist > epsilon and max_idx is not None:
            keep.append(max_idx)
            _recursive(start_idx, max_idx, keep)
            _recursive(max_idx, end_idx, keep)

    keep_indices = [0, len(pts) - 1]
    _recursive(0, len(pts) - 1, keep_indices)
    keep_indices = sorted(set(keep_indices))
    return [tuple(pts[i]) for i in keep_indices]
