"""Direct branch coverage for the Theta* v2 line-of-sight fallback and planner binding.

Locks the pure-Python Bresenham line-of-sight bounds/free-cell/obstacle behavior,
the optional-accelerator Python fallback binding, the 2D-only guard, and the
one-time collision-check binding in :mod:`robot_sf.planner.theta_star_v2`.

These tests use tiny numpy grids and lightweight upstream fakes only; no maps or
benchmarks are loaded. See GitHub issue #6368.
"""

from __future__ import annotations

import types

import numpy as np
from loguru import logger
from python_motion_planning.common import TYPES
from python_motion_planning.path_planner import ThetaStar

import robot_sf.planner.theta_star_v2 as theta_module
from robot_sf.planner.theta_star_v2 import (
    HighPerformanceThetaStar,
    _bind_fast_in_collision,
    _python_los_blocks,
)

# Free cell values treated as traversable by the LOS helper (TYPES.FREE/START/GOAL).
FREE_VALS = np.array([TYPES.FREE, TYPES.START, TYPES.GOAL], dtype=np.int64)
# Any value absent from FREE_VALS is treated as an obstacle.
OBSTACLE = 1


def _zeros(shape=(5, 5)):
    """Return a fresh int8 grid filled with the FREE cell value."""
    return np.zeros(shape, dtype=np.int8)


class _LosGrid:
    """Minimal grid stand-in exposing type_map.array for the bound LOS method."""

    def __init__(self, array):
        """Wrap ``array`` so ``type_map.array`` mirrors the upstream Grid surface."""
        self.type_map = types.SimpleNamespace(array=array, shape=array.shape)


class _PlannerGrid(types.SimpleNamespace):
    """Grid stand-in exposing the minimal upstream Theta* map API for plan() tests."""

    def __init__(self, dim):
        """Store the grid dimension; upstream planning is faked per test."""
        super().__init__(dim=dim, type_map=None)

    def update_esdf(self):
        """No-op ESDF refresh placeholder; upstream planning is faked."""

    def get_neighbors(self, node):
        """Return no neighbors; upstream planning is faked."""
        return []

    def is_expandable(self, *args, **kwargs):
        """Treat all nodes as expandable; upstream planning is faked."""
        return True


def _fake_upstream_planner(monkeypatch):
    """Monkeypatch ``ThetaStar.plan`` with a counting fake and return its counter."""
    upstream = {"calls": 0}

    def fake_plan(self):
        upstream["calls"] += 1
        return [(0, 0)], {"expand": {}}

    monkeypatch.setattr(ThetaStar, "plan", fake_plan)
    return upstream


# ---------------------------------------------------------------------------
# Pure-Python Bresenham LOS boundary matrix
# ---------------------------------------------------------------------------


def test_python_los_out_of_bounds_start_returns_blocked():
    """A start endpoint outside the grid short-circuits to blocked (True)."""
    grid = _zeros()
    assert _python_los_blocks(grid, -1, 0, 2, 2, FREE_VALS) is True
    assert _python_los_blocks(grid, 0, -1, 2, 2, FREE_VALS) is True
    assert _python_los_blocks(grid, grid.shape[0], 0, 2, 2, FREE_VALS) is True


def test_python_los_out_of_bounds_end_returns_blocked():
    """An end endpoint outside the grid short-circuits to blocked (True)."""
    grid = _zeros()
    assert _python_los_blocks(grid, 0, 0, grid.shape[0], 2, FREE_VALS) is True
    assert _python_los_blocks(grid, 0, 0, 2, grid.shape[1], FREE_VALS) is True
    assert _python_los_blocks(grid, 0, 0, -1, 2, FREE_VALS) is True


def test_python_los_blocked_start_endpoint_returns_blocked():
    """A blocked start cell short-circuits to blocked before traversal."""
    grid = _zeros((3, 3))
    grid[0, 0] = OBSTACLE
    assert _python_los_blocks(grid, 0, 0, 2, 2, FREE_VALS) is True


def test_python_los_blocked_end_endpoint_returns_blocked():
    """A blocked end cell short-circuits to blocked before traversal."""
    grid = _zeros((3, 3))
    grid[2, 2] = OBSTACLE
    assert _python_los_blocks(grid, 0, 0, 2, 2, FREE_VALS) is True


def test_python_los_clear_horizontal_vertical_diagonal():
    """Clear horizontal, vertical, forward, and reverse diagonal lines are open."""
    grid = _zeros()
    assert _python_los_blocks(grid, 0, 2, 4, 2, FREE_VALS) is False  # horizontal (sy=-1)
    assert _python_los_blocks(grid, 2, 0, 2, 4, FREE_VALS) is False  # vertical (sx=-1)
    assert _python_los_blocks(grid, 0, 0, 4, 4, FREE_VALS) is False  # forward diagonal
    assert _python_los_blocks(grid, 4, 4, 0, 0, FREE_VALS) is False  # reverse diagonal


def test_python_los_obstacle_along_horizontal_vertical_diagonal():
    """Obstacles encountered mid-traversal block each stepping direction."""
    grid = _zeros()
    grid[2, 2] = OBSTACLE
    assert _python_los_blocks(grid, 0, 2, 4, 2, FREE_VALS) is True  # horizontal
    assert _python_los_blocks(grid, 2, 0, 2, 4, FREE_VALS) is True  # vertical
    assert _python_los_blocks(grid, 0, 0, 4, 4, FREE_VALS) is True  # diagonal


def test_python_los_treats_free_start_goal_as_traversable():
    """FREE, START, and GOAL cell values must all remain traversable end to end."""
    grid = np.full((3, 3), OBSTACLE, dtype=np.int8)
    grid[0, 0] = TYPES.FREE
    grid[1, 1] = TYPES.START
    grid[2, 2] = TYPES.GOAL
    assert _python_los_blocks(grid, 0, 0, 2, 2, FREE_VALS) is False
    assert _python_los_blocks(grid, 2, 2, 0, 0, FREE_VALS) is False


# ---------------------------------------------------------------------------
# Optional-accelerator Python fallback binding
# ---------------------------------------------------------------------------


def test_bind_fast_in_collision_python_fallback_routes_to_python_los(monkeypatch):
    """With the accelerator disabled, the bound check routes to the pure-Python LOS."""
    monkeypatch.setattr(theta_module, "njit", None)

    routed = []

    def spy(type_map, x0, y0, x1, y1, free_vals):
        routed.append((x0, y0, x1, y1))
        return _python_los_blocks(type_map, x0, y0, x1, y1, free_vals)

    monkeypatch.setattr(theta_module, "_python_los_blocks", spy)

    grid = _LosGrid(_zeros((3, 3)))
    _bind_fast_in_collision(grid)

    assert grid.in_collision((0, 0), (2, 2)) is False
    assert routed == [(0, 0, 2, 2)]
    grid.type_map.array[1, 1] = OBSTACLE
    assert grid.in_collision((0, 0), (2, 2)) is True


def test_bind_fast_in_collision_python_fallback_keeps_free_start_goal_clear(monkeypatch):
    """The Python fallback bound check treats FREE/START/GOAL cells as traversable."""
    monkeypatch.setattr(theta_module, "njit", None)

    grid = _LosGrid(np.full((3, 3), OBSTACLE, dtype=np.int8))
    grid.type_map.array[0, 0] = TYPES.FREE
    grid.type_map.array[1, 1] = TYPES.START
    grid.type_map.array[2, 2] = TYPES.GOAL
    _bind_fast_in_collision(grid)

    assert grid.in_collision((0, 0), (2, 2)) is False


# ---------------------------------------------------------------------------
# Planner dimension guard and one-time binding
# ---------------------------------------------------------------------------


def test_plan_skips_binding_and_delegates_for_non_2d(monkeypatch):
    """Non-2D grids skip fast binding, emit a warning, and still delegate upstream."""
    grid = _PlannerGrid(dim=3)
    binds = []
    monkeypatch.setattr(theta_module, "_bind_fast_in_collision", binds.append)
    upstream = _fake_upstream_planner(monkeypatch)
    planner = HighPerformanceThetaStar(map_=grid, start=(0, 0), goal=(1, 1))

    records = []
    sink = logger.add(records.append, level="WARNING")
    try:
        path, info = planner.plan()
    finally:
        logger.remove(sink)

    assert binds == []  # no fast collision binding for non-2D grids
    assert upstream["calls"] == 1  # upstream planning still delegated once
    assert path == [(0, 0)]
    assert info == {"expand": {}}
    assert any("only optimizes 2D grids" in message for message in records)


def test_plan_binds_collision_check_once_across_repeated_calls(monkeypatch):
    """A 2D grid binds collision checking once and delegates upstream on every call."""
    grid = _PlannerGrid(dim=2)
    binds = []
    monkeypatch.setattr(theta_module, "_bind_fast_in_collision", binds.append)
    upstream = _fake_upstream_planner(monkeypatch)
    planner = HighPerformanceThetaStar(map_=grid, start=(0, 0), goal=(1, 1))

    first = planner.plan()
    second = planner.plan()

    assert binds == [grid]  # collision check bound exactly once, on the first call
    assert upstream["calls"] == 2  # upstream planning delegated on every call
    assert getattr(grid, "__fast_collision_bound", False) is True
    assert first == second
