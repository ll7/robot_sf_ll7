"""High-performance Theta* variant with a faster line-of-sight check.

This module provides a drop-in replacement for python_motion_planning's ThetaStar
that accelerates the dominant hot path (line-of-sight collision checks) via a
minimal Bresenham traversal, optionally JIT-compiled with Numba.

Notes:
    - Supports 2D grids only (the common case for ClassicGlobalPlanner).
    - Output is functionally equivalent; expand order may differ.
    - Falls back to pure Python when Numba is unavailable.

HighPerformanceThetaStar speeds things up by attacking the main hot path: line-of-sight (LOS) collision checks. It replaces the upstream in_collision with a lightweight Bresenham traversal that:

- Avoids per-call NumPy allocations and conversions (no np.array, no zeros_like, minimal attribute lookups).
- Iterates on plain integers with a tiny inner loop instead of building temporary arrays.
- Binds this fast LOS function directly to the grid once, so every Theta* LOS check uses the optimized version.
- Optionally JIT-compiles the LOS routine with Numba when available, further reducing Python overhead.

Because Theta* calls LOS millions of times on large grids, removing these allocations and Python overhead yields a substantial runtime drop while keeping path optimality intact.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

from loguru import logger
from python_motion_planning.common import TYPES
from python_motion_planning.path_planner import ThetaStar

try:  # Optional accelerator
    from numba import njit
except ImportError:  # pragma: no cover - numba optional
    njit = None

if TYPE_CHECKING:
    from python_motion_planning.common import Grid


def _python_los_blocks(type_map, x0, y0, x1, y1, free_vals) -> bool:
    """Pure-Python Bresenham line traversal.

    Args:
        type_map: Numpy array backing the grid.
        x0: Start X index.
        y0: Start Y index.
        x1: End X index.
        y1: End Y index.
        free_vals: Set of cell values considered traversable.

    Returns:
        True when an obstacle is encountered along the segment, else False.
    """
    width, height = type_map.shape
    if not (0 <= x0 < width and 0 <= y0 < height):
        return True
    if not (0 <= x1 < width and 0 <= y1 < height):
        return True

    if type_map[x0, y0] not in free_vals:
        return True
    if type_map[x1, y1] not in free_vals:
        return True

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if type_map[x0, y0] not in free_vals:
            return True
        if x0 == x1 and y0 == y1:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


if njit:

    @njit(cache=True)
    def _numba_los_blocks(
        type_map, x0, y0, x1, y1, free_vals
    ) -> bool:  # pragma: no cover - exercised via python wrapper
        width, height = type_map.shape
        if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
            return True
        if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
            return True

        v0 = type_map[x0, y0]
        if v0 not in free_vals:
            return True
        v1 = type_map[x1, y1]
        if v1 not in free_vals:
            return True

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            v = type_map[x0, y0]
            if v not in free_vals:
                return True
            if x0 == x1 and y0 == y1:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy


def _bind_fast_in_collision(grid: Grid) -> None:
    """Attach a faster in_collision method to a Grid instance.

    The bound method uses either the Numba-accelerated LOS or a Python fallback.
    """
    free_vals = {TYPES.FREE, TYPES.START, TYPES.GOAL}
    if njit:

        def fast_in_collision(self, p1, p2):
            x0, y0 = p1
            x1, y1 = p2
            return _numba_los_blocks(self.type_map.array, x0, y0, x1, y1, free_vals)

    else:

        def fast_in_collision(self, p1, p2):
            x0, y0 = p1
            x1, y1 = p2
            return _python_los_blocks(self.type_map.array, x0, y0, x1, y1, free_vals)

    grid.in_collision = MethodType(fast_in_collision, grid)  # type: ignore[attr-defined]


class HighPerformanceThetaStar(ThetaStar):
    """Theta* variant with optimized line-of-sight collision checking.

    This class wraps the upstream Theta* but replaces the grid's in_collision with
    a faster implementation before delegating to the parent plan().
    """

    def plan(self):
        """Run planning with a fast in-collision check bound to the grid.

        Returns:
            tuple: (path, path_info) from the upstream Theta* implementation.
        """
        if getattr(self.map_, "dim", None) != 2:
            logger.warning(
                "HighPerformanceThetaStar only optimizes 2D grids; using base implementation."
            )
            return super().plan()

        if not getattr(self.map_, "__fast_collision_bound", False):
            _bind_fast_in_collision(self.map_)
            setattr(self.map_, "__fast_collision_bound", True)

        return super().plan()


__all__ = ["HighPerformanceThetaStar"]
