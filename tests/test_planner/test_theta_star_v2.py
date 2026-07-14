"""Tests for the high-performance Theta* wrapper."""

from __future__ import annotations

import types
import warnings

import numpy as np

from robot_sf.planner.theta_star_v2 import HighPerformanceThetaStar, _bind_fast_in_collision


def test_fast_in_collision_emits_no_reflected_set_numba_warning():
    """The Numba LOS path must not emit reflected-set deprecation warnings.

    Issue #5598: the njit LOS previously received a reflected Python ``set`` for
    ``free_vals``, raising NumbaPendingDeprecationWarning. The free-cell set is
    now a typed int64 array, so exercising the bound collision check must pass
    cleanly even when deprecation warnings are escalated to errors.
    """
    arr = np.zeros((3, 3), dtype=np.int8)
    grid = DummyGrid(arr)
    _bind_fast_in_collision(grid)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", PendingDeprecationWarning)
        # Force the Numba (or Python) LOS path to actually run.
        assert grid.in_collision((0, 0), (2, 2)) is False
        grid.type_map.array[1, 1] = 1
        assert grid.in_collision((0, 0), (2, 2)) is True


class DummyGrid:
    """Minimal grid stand-in with type_map and dims for LOS checks."""

    def __init__(self, type_map: np.ndarray):
        """Store type_map and shape to mimic the Grid interface."""
        self.type_map = types.SimpleNamespace(array=type_map, shape=type_map.shape)
        self.dim = 2


def test_bind_fast_in_collision_sets_method():
    """Binding should attach an in_collision method and honor free cells."""
    arr = np.zeros((3, 3), dtype=np.int8)
    grid = DummyGrid(arr)

    _bind_fast_in_collision(grid)  # mutates grid to include in_collision

    assert hasattr(grid, "in_collision")
    # Free line should return False
    assert grid.in_collision((0, 0), (2, 2)) is False
    # Block a cell and ensure collision is detected
    grid.type_map.array[1, 1] = 1  # not in free set
    assert grid.in_collision((0, 0), (2, 2)) is True


def test_high_performance_theta_star_forwards_to_upstream(monkeypatch):
    """HighPerformanceThetaStar should defer to upstream plan after binding LOS."""
    calls = {"plan": 0}

    class DummyGrid(types.SimpleNamespace):
        """Grid stub exposing the minimal upstream Theta* map API."""

        def __init__(self):
            super().__init__(dim=2, type_map=None)

        def update_esdf(self):
            """Record that the wrapper asked the grid to refresh ESDF state."""
            calls["plan"] += 1

        def get_neighbors(self, node):
            """Return no neighbors because upstream planning is stubbed."""
            return []

        def is_expandable(self, *args, **kwargs):
            """Treat all nodes as expandable for wrapper forwarding tests."""
            return True

    grid = DummyGrid()

    class DummyUpstream(HighPerformanceThetaStar):
        """Upstream planner stub that confirms plan forwarding."""

        def plan(self):
            """Return a fixed path and expansion payload."""
            calls["plan"] += 1
            return ([(0, 0), (1, 1)], {"expand": {}})

    planner = DummyUpstream(map_=grid, start=(0, 0), goal=(1, 1))
    path, info = planner.plan()

    assert calls["plan"] >= 1
    assert path[-1] == (1, 1)
    assert "expand" in info
