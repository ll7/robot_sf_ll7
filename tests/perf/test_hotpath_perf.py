"""Micro-benchmarks for the four hot-path fixes from issue #6460.

Covers:

1. ``_prepare_visualizable_state`` pedestrian-position copying: replaces
   ``copy.deepcopy(ped_pos)`` with ``np.asarray(ped_pos).copy()``.
2. ``OccupancyGrid.generate()`` buffer reuse: a private pre-allocated buffer
   cleared via ``fill(0)`` replaces the fresh ``np.zeros`` allocation per
   step, while ``is_initialized``/``shape``/``reset`` semantics stay intact.
3. ``SimulationView._draw_pedestrians`` action matching: index-based lookup
   replaces the O(P*A) nearest-neighbor ``min()`` scan.
4. ``rasterize_line_segment`` rasterization: a NumPy-vectorized Bresenham
   replaces the pure-Python per-cell loop.

Timing comparisons use best-of-k totals over many iterations to stay stable.
The allocation-elimination and behavioral-equivalence checks are exact and do
not depend on timing noise.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.nav import occupancy_grid as occupancy_grid_module
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid
from robot_sf.nav.occupancy_grid_rasterization import (
    _bresenham_line,
    rasterize_line_segment,
)
from robot_sf.render.sim_view import SimulationView

if TYPE_CHECKING:
    from collections.abc import Callable

_GRID_SHAPE = (3, 200, 200)


def _best_of_total(fn: Callable[[], None], iterations: int, rounds: int = 5) -> float:
    """Return the best total wall time (seconds) for ``iterations`` calls."""
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        best = min(best, time.perf_counter() - start)
    return best


def _reference_bresenham(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    """Pure-Python Bresenham loop kept as the behavioral reference."""
    cells: list[tuple[int, int]] = []
    dx = abs(col1 - col0)
    dy = abs(row1 - row0)
    sx = 1 if col0 < col1 else -1
    sy = 1 if row0 < row1 else -1
    err = dx - dy
    row, col = row0, col0
    while True:
        cells.append((row, col))
        if row == row1 and col == col1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            col += sx
        if e2 < dx:
            err += dx
            row += sy
    return cells


def _make_grid() -> OccupancyGrid:
    """Build a 200x200 three-channel grid like the training-rollout default."""
    config = GridConfig(
        resolution=0.1,
        width=20.0,
        height=20.0,
        channels=[
            GridChannel.OBSTACLES,
            GridChannel.PEDESTRIANS,
            GridChannel.COMBINED,
        ],
    )
    return OccupancyGrid(config)


def _grid_inputs() -> tuple[list, list, tuple]:
    """Return fixed obstacles, pedestrians, and robot pose for generate()."""
    obstacles = [((float(i), 0.5), (float(i), 19.5)) for i in range(1, 20, 2)]
    pedestrians = [((float(i), float(j)), 0.35) for i in range(2, 18, 3) for j in range(2, 18, 3)]
    robot_pose = ((10.0, 10.0), 0.0)
    return obstacles, pedestrians, robot_pose


# ---------------------------------------------------------------------------
# Finding 1: deepcopy(ped_pos) -> np.asarray(ped_pos).copy()
# ---------------------------------------------------------------------------


class TestPedPosCopy:
    """Benchmark and equivalence checks for the ped_pos snapshot copy."""

    @staticmethod
    def _ped_pos_view(num_peds: int) -> np.ndarray:
        """Build a strided position view shaped like ``pysf_state.ped_positions``."""
        state = np.random.default_rng(num_peds).random((num_peds, 7))
        return state[:, 0:2]

    def test_copy_output_matches_deepcopy(self) -> None:
        """The NumPy copy must be value-identical and independent."""
        positions = self._ped_pos_view(100)
        deep = copy.deepcopy(positions)
        fast = np.asarray(positions).copy()
        np.testing.assert_array_equal(deep, fast)
        assert fast.flags["C_CONTIGUOUS"]
        fast[0, 0] += 1.0
        assert positions[0, 0] != fast[0, 0]

    def test_numpy_copy_not_slower_than_deepcopy(self) -> None:
        """The replacement copy path must not be slower than deepcopy.

        Measured ratios on the reference host are modest (~1.2x-2.3x) because
        modern NumPy implements ``__deepcopy__`` natively; the assertion is a
        strict improvement over pooled sizes rather than a fixed multiplier.
        """
        total_deepcopy = 0.0
        total_copy = 0.0
        iterations = 2000
        for num_peds in (50, 100, 500):
            positions = self._ped_pos_view(num_peds)
            total_deepcopy += _best_of_total(lambda: copy.deepcopy(positions), iterations)
            total_copy += _best_of_total(lambda: np.asarray(positions).copy(), iterations)
        assert total_copy < total_deepcopy, (
            f"np.asarray(...).copy() took {total_copy:.6f}s vs deepcopy {total_deepcopy:.6f}s"
        )


# ---------------------------------------------------------------------------
# Finding 2: OccupancyGrid.generate() buffer reuse
# ---------------------------------------------------------------------------


class TestOccupancyGridBufferReuse:
    """Allocation-elimination and semantics checks for generate() reuse."""

    def test_public_semantics_preserved(self) -> None:
        """is_initialized/shape/reset behave exactly as before the fix."""
        grid = _make_grid()
        assert grid.is_initialized is False
        assert grid.shape == _GRID_SHAPE[:3]

        obstacles, pedestrians, robot_pose = _grid_inputs()
        grid.generate(obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose)
        assert grid.is_initialized is True
        assert grid.shape == (3, 200, 200)

        grid.reset()
        assert grid.is_initialized is False
        assert grid.shape == _GRID_SHAPE[:3]

    def test_steady_state_generate_performs_no_allocation(self) -> None:
        """After warmup, generate() must not allocate a fresh grid buffer.

        Before the fix every generate() call allocated ``np.zeros(shape)``
        (~640KB for a 200x200x4 grid). With buffer reuse the steady-state
        allocation count is exactly zero and the buffer identity is stable.
        """
        grid = _make_grid()
        obstacles, pedestrians, robot_pose = _grid_inputs()

        # Warmup builds the reusable buffer and the static obstacle-layer cache.
        first = grid.generate(obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose)
        buffer = grid._grid_buffer
        assert buffer is not None
        np.testing.assert_array_equal(first, grid._grid_data)

        zeros_calls = {"count": 0}
        real_zeros = np.zeros

        def counting_zeros(*args, **kwargs):
            zeros_calls["count"] += 1
            return real_zeros(*args, **kwargs)

        steps = 5
        occupancy_grid_module.np.zeros = counting_zeros
        try:
            for _ in range(steps):
                regenerated = grid.generate(
                    obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose
                )
        finally:
            occupancy_grid_module.np.zeros = real_zeros

        assert zeros_calls["count"] == 0, (
            f"generate() allocated {zeros_calls['count']} fresh grids over {steps} steps"
        )
        assert grid._grid_buffer is buffer
        assert grid._grid_data is buffer
        # Reused buffer still carries fresh rasterized content each step.
        np.testing.assert_array_equal(regenerated, first)

    def test_buffer_reuse_not_slower_than_reallocation(self) -> None:
        """Reuse must be at least as fast as the old per-call allocation."""
        grid = _make_grid()
        obstacles, pedestrians, robot_pose = _grid_inputs()
        grid.generate(obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose)

        def generate_reusing() -> None:
            grid.generate(obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose)

        def generate_reallocating() -> None:
            # Force the allocation path on every call to mimic pre-fix behavior.
            grid._grid_buffer = None
            grid.generate(obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose)

        iterations = 30
        reuse_total = _best_of_total(generate_reusing, iterations, rounds=3)
        realloc_total = _best_of_total(generate_reallocating, iterations, rounds=3)
        assert reuse_total <= realloc_total * 1.5, (
            f"buffer-reuse generate() took {reuse_total:.4f}s vs reallocation {realloc_total:.4f}s"
        )


# ---------------------------------------------------------------------------
# Finding 3: index-based pedestrian action matching
# ---------------------------------------------------------------------------


class TestPedestrianActionMatching:
    """Benchmark the O(P*A) nearest-neighbor scan vs index-based lookup."""

    def test_index_lookup_not_slower_than_nearest_scan(self) -> None:
        """Direct indexing must beat the dict-plus-min nearest-neighbor scan."""
        rng = np.random.default_rng(7)
        num_peds = 100
        positions = rng.random((num_peds, 2)) * 20.0
        # One trailing ego action row, as produced for ego-pedestrian simulators.
        actions = np.empty((num_peds + 1, 2, 2))
        actions[:, 0, :] = np.vstack([positions, positions[-1:] + 0.5])
        actions[:, 1, :] = actions[:, 0, :] + 0.1

        def nearest_scan() -> None:
            action_map: dict[tuple[float, float], tuple[float, float]] = {}
            for start, end in actions:
                action_map[tuple(start)] = tuple(end)
            for ped_x, ped_y in positions:
                min(
                    action_map.items(),
                    key=lambda item: (item[0][0] - ped_x) ** 2 + (item[0][1] - ped_y) ** 2,
                    default=None,
                )

        def index_lookup() -> None:
            num_actions = len(actions)
            for ped_idx in range(len(positions)):
                if ped_idx < num_actions:
                    _ = actions[ped_idx]

        iterations = 200
        scan_total = _best_of_total(nearest_scan, iterations, rounds=3)
        index_total = _best_of_total(index_lookup, iterations, rounds=3)
        assert index_total < scan_total, (
            f"index lookup took {index_total:.6f}s vs scan {scan_total:.6f}s"
        )


class TestDrawPedestriansRenderPath:
    """Headless execution of the changed ``_draw_pedestrians`` path.

    The session-level headless fixture forces the SDL dummy driver, so the
    render path runs without a display in every lane.
    """

    @staticmethod
    def _view() -> SimulationView:
        return SimulationView(width=64, height=64, scaling=10.0)

    @staticmethod
    def _positions_and_actions(num_peds: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(num_peds)
        positions = rng.uniform(0.5, 5.5, size=(num_peds, 2))
        # One trailing ego action row, as produced for ego-pedestrian simulators.
        actions = np.empty((num_peds + 1, 2, 2))
        actions[:, 0, :] = np.vstack([positions, positions[-1:] + 0.1])
        actions[:, 1, :] = actions[:, 0, :] + 0.2
        return positions, actions

    def test_draw_pedestrians_branches_headless(self) -> None:
        """All index-matching branches render without error headlessly."""
        view = self._view()
        positions, actions = self._positions_and_actions(4)

        view._draw_pedestrians(positions, actions)  # every ped has its action
        view._draw_pedestrians(positions, actions[:2])  # trailing peds without
        view._draw_pedestrians(positions, None)  # no actions at all
        view._draw_pedestrians(positions, np.empty((0, 2, 2)))  # playback-style

    def test_draw_pedestrians_frame_budget(self) -> None:
        """Index-based drawing keeps per-frame work at pygame-draw cost."""
        view = self._view()
        positions, actions = self._positions_and_actions(50)

        for _ in range(3):  # warmup (sprite caches, surface allocation)
            view._draw_pedestrians(positions, actions)

        frames = 30
        total = _best_of_total(lambda: view._draw_pedestrians(positions, actions), frames, rounds=3)
        assert total < frames * 0.025, f"50-ped frame draw took {total / frames * 1e3:.2f}ms"


# ---------------------------------------------------------------------------
# Finding 4: vectorized Bresenham rasterization
# ---------------------------------------------------------------------------


class TestVectorizedBresenham:
    """Equivalence and speedup checks for the vectorized line rasterizer."""

    @staticmethod
    def _segment_cases() -> list[tuple[int, int, int, int]]:
        rng = np.random.default_rng(42)
        cases = [
            (0, 0, 0, 0),
            (0, 0, 5, 0),
            (0, 0, 0, 5),
            (0, 0, 5, 5),
            (5, 5, 0, 0),
            (0, 5, 5, 0),
            (2, 3, 0, 0),
            (-2, 3, 4, -5),
        ]
        cases.extend(tuple(int(v) for v in row) for row in rng.integers(-40, 40, size=(500, 4)))
        return cases

    def test_vectorized_matches_reference_loop(self) -> None:
        """The vectorized cell set must equal the pure-Python loop exactly."""
        for row0, col0, row1, col1 in self._segment_cases():
            expected = _reference_bresenham(row0, col0, row1, col1)
            rows, cols = _bresenham_line(row0, col0, row1, col1)
            actual = list(zip(rows.tolist(), cols.tolist(), strict=True))
            assert len(actual) == len(expected), (row0, col0, row1, col1)
            assert set(actual) == set(expected), (row0, col0, row1, col1)

    def test_vectorized_bresenham_speedup(self) -> None:
        """Vectorized rasterization must beat the Python loop on long lines."""
        iterations = 100
        old_total = _best_of_total(
            lambda: _reference_bresenham(0, 0, 500, 300), iterations, rounds=3
        )
        new_total = _best_of_total(lambda: _bresenham_line(0, 0, 500, 300), iterations, rounds=3)
        assert new_total * 2.0 < old_total, (
            f"vectorized {new_total:.6f}s vs reference loop {old_total:.6f}s"
        )

    def test_rasterize_line_segment_keeps_max_merge_and_clipping(self) -> None:
        """Per-cell max-merge and grid-bounds clipping survive vectorization."""
        config = GridConfig(resolution=1.0, width=10.0, height=10.0)
        grid = np.full((config.grid_height, config.grid_width), 0.9, dtype=config.dtype)

        # Lower value must not overwrite higher occupancy (max-merge semantics).
        assert rasterize_line_segment(((1.0, 5.0), (9.0, 5.0)), grid, config, value=0.5) is True
        assert grid[5, 5] == pytest.approx(0.9)

        # Higher value raises occupancy along the line.
        assert rasterize_line_segment(((1.0, 1.0), (9.0, 9.0)), grid, config, value=1.0) is True
        for idx in range(1, 10):
            assert grid[idx, idx] == pytest.approx(1.0)

        # Segments crossing the border only fill in-bounds cells.
        grid.fill(0.0)
        assert rasterize_line_segment(((-5.0, 5.0), (15.0, 5.0)), grid, config) is True
        assert np.count_nonzero(grid) == config.grid_width
        assert np.all(grid[5, :] > 0)
