"""Comprehensive unit tests for line-segment rasterization and clipping.

Tests cover rasterize_line_segment and _clip_line_to_rect in the
robot_sf.nav.occupancy_grid_rasterization module, verifying segments fully
inside, crossing grid boundaries, fully outside, horizontal, vertical,
diagonal, and zero-length segments.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav import occupancy_grid_rasterization as rasterization
from robot_sf.nav.occupancy_grid import GridConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_10x10() -> tuple[np.ndarray, GridConfig]:
    """Return a 10x10 empty grid array and its config (1m resolution)."""
    config = GridConfig(resolution=1.0, width=10.0, height=10.0)
    grid = np.zeros((config.grid_height, config.grid_width), dtype=config.dtype)
    return grid, config


# ---------------------------------------------------------------------------
# _clip_line_to_rect – unit tests for the Liang-Barsky clipper
# ---------------------------------------------------------------------------


class TestClipLineToRect:
    """Liang-Barsky line clipping against rectangle [0,10] x [0,10]."""

    RECT = (0.0, 10.0, 0.0, 10.0)  # min_x, max_x, min_y, max_y

    def test_fully_inside(self) -> None:
        result = rasterization._clip_line_to_rect((2.0, 3.0), (8.0, 7.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((2.0, 3.0))
        assert end == pytest.approx((8.0, 7.0))

    def test_crosses_right_boundary(self) -> None:
        result = rasterization._clip_line_to_rect((5.0, 5.0), (15.0, 5.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((5.0, 5.0))
        assert end == pytest.approx((10.0, 5.0))

    def test_crosses_left_boundary(self) -> None:
        result = rasterization._clip_line_to_rect((-5.0, 5.0), (5.0, 5.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((0.0, 5.0))
        assert end == pytest.approx((5.0, 5.0))

    def test_crosses_top_boundary(self) -> None:
        result = rasterization._clip_line_to_rect((5.0, 15.0), (5.0, 5.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((5.0, 10.0))
        assert end == pytest.approx((5.0, 5.0))

    def test_crosses_bottom_boundary(self) -> None:
        result = rasterization._clip_line_to_rect((5.0, -5.0), (5.0, 5.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((5.0, 0.0))
        assert end == pytest.approx((5.0, 5.0))

    def test_crosses_two_boundaries(self) -> None:
        result = rasterization._clip_line_to_rect((-5.0, -5.0), (15.0, 15.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((0.0, 0.0))
        assert end == pytest.approx((10.0, 10.0))

    def test_fully_outside_left(self) -> None:
        result = rasterization._clip_line_to_rect((-10.0, 5.0), (-5.0, 5.0), *self.RECT)
        assert result is None

    def test_fully_outside_right(self) -> None:
        result = rasterization._clip_line_to_rect((15.0, 5.0), (20.0, 5.0), *self.RECT)
        assert result is None

    def test_fully_outside_below(self) -> None:
        result = rasterization._clip_line_to_rect((5.0, -10.0), (5.0, -5.0), *self.RECT)
        assert result is None

    def test_fully_outside_above(self) -> None:
        result = rasterization._clip_line_to_rect((5.0, 15.0), (5.0, 20.0), *self.RECT)
        assert result is None

    def test_parallel_and_outside_horizontal(self) -> None:
        result = rasterization._clip_line_to_rect((-5.0, 15.0), (15.0, 15.0), *self.RECT)
        assert result is None

    def test_parallel_and_outside_vertical(self) -> None:
        result = rasterization._clip_line_to_rect((15.0, -5.0), (15.0, 15.0), *self.RECT)
        assert result is None

    def test_horizontal_fully_inside(self) -> None:
        result = rasterization._clip_line_to_rect((1.0, 5.0), (9.0, 5.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((1.0, 5.0))
        assert end == pytest.approx((9.0, 5.0))

    def test_vertical_fully_inside(self) -> None:
        result = rasterization._clip_line_to_rect((5.0, 1.0), (5.0, 9.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((5.0, 1.0))
        assert end == pytest.approx((5.0, 9.0))

    def test_diagonal_fully_inside(self) -> None:
        result = rasterization._clip_line_to_rect((1.0, 1.0), (9.0, 9.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((1.0, 1.0))
        assert end == pytest.approx((9.0, 9.0))

    def test_zero_length_inside(self) -> None:
        result = rasterization._clip_line_to_rect((5.0, 5.0), (5.0, 5.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((5.0, 5.0))
        assert end == pytest.approx((5.0, 5.0))

    def test_diagonal_crosses_two_boundaries(self) -> None:
        result = rasterization._clip_line_to_rect((-2.0, 3.0), (12.0, 8.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start[0] == pytest.approx(0.0)
        assert end[0] == pytest.approx(10.0)

    def test_diagonal_crosses_corner(self) -> None:
        result = rasterization._clip_line_to_rect((-5.0, 15.0), (15.0, -5.0), *self.RECT)
        assert result is not None
        start, end = result
        assert start == pytest.approx((0.0, 10.0))
        assert end == pytest.approx((10.0, 0.0))

    def test_diagonal_fully_outside(self) -> None:
        result = rasterization._clip_line_to_rect((-10.0, -10.0), (-5.0, -5.0), *self.RECT)
        assert result is None


# ---------------------------------------------------------------------------
# rasterize_line_segment – integration tests
# ---------------------------------------------------------------------------


class TestRasterizeLineSegment:
    """End-to-end rasterization tests on a 10x10 grid (1m resolution)."""

    def test_fully_inside_diagonal(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((1.0, 1.0), (9.0, 9.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        occupied = np.argwhere(grid > 0)
        assert len(occupied) == 9
        for i in range(1, 10):
            assert grid[i, i] > 0

    def test_fully_outside(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((-10.0, -10.0), (-5.0, -5.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is False
        assert np.all(grid == 0)

    def test_crosses_left_boundary(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((-5.0, 5.0), (5.0, 5.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        occupied = np.argwhere(grid > 0)
        assert len(occupied) >= 5
        for col in range(6):
            assert grid[5, col] > 0

    def test_crosses_right_boundary(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((5.0, 5.0), (15.0, 5.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        occupied = np.argwhere(grid > 0)
        assert len(occupied) >= 5
        for col in range(5, 10):
            assert grid[5, col] > 0

    def test_horizontal_fully_inside(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((1.0, 5.0), (9.0, 5.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        for col in range(1, 10):
            assert grid[5, col] > 0

    def test_vertical_fully_inside(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((5.0, 1.0), (5.0, 9.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        for row in range(1, 10):
            assert grid[row, 5] > 0

    def test_zero_length_inside(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((5.0, 5.0), (5.0, 5.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        assert grid[5, 5] > 0

    def test_zero_length_outside(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((-5.0, -5.0), (-5.0, -5.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is False
        assert np.all(grid == 0)

    def test_diagonal_ascending(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((1.0, 1.0), (9.0, 9.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        for i in range(1, 10):
            assert grid[i, i] > 0

    def test_diagonal_descending(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((1.0, 9.0), (9.0, 1.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        assert grid[9, 1] > 0
        assert grid[1, 9] > 0

    def test_crosses_two_boundaries(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((-5.0, -5.0), (15.0, 15.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is True
        assert grid[0, 0] > 0
        assert grid[9, 9] > 0

    def test_outside_after_clipping_not_applied(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((-2.0, 5.0), (-1.0, 5.0))
        result = rasterization.rasterize_line_segment(line, grid, config)
        assert result is False

    def test_nonzero_grid_origin(self, grid_10x10) -> None:
        grid, config = grid_10x10
        line = ((11.0, 11.0), (19.0, 19.0))
        result = rasterization.rasterize_line_segment(
            line, grid, config, grid_origin_x=10.0, grid_origin_y=10.0
        )
        assert result is True
        for i in range(1, 10):
            assert grid[i, i] > 0

    def test_uses_max_operator(self, grid_10x10) -> None:
        grid, config = grid_10x10
        grid[:, :] = 0.5
        line = ((1.0, 1.0), (3.0, 3.0))
        result = rasterization.rasterize_line_segment(line, grid, config, value=1.0)
        assert result is True
        for i in range(1, 4):
            assert grid[i, i] == pytest.approx(1.0)

    def test_does_not_overwrite_larger_values(self, grid_10x10) -> None:
        grid, config = grid_10x10
        grid[5, 5] = 0.9
        line = ((1.0, 5.0), (9.0, 5.0))
        result = rasterization.rasterize_line_segment(line, grid, config, value=0.5)
        assert result is True
        assert grid[5, 5] == pytest.approx(0.9)
