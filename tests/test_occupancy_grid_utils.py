"""Coordinate-transform round-trip tests for occupancy_grid_utils.py.

Tests verify that coordinate transformations preserve invariants:
- world ↔ grid-indices round-trips return cell centers
- world ↔ ego round-trips are exact inverses
- bounds checking, clipping, and bounds queries behave correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.nav.occupancy_grid_utils import (
    clip_to_grid,
    ego_to_world,
    get_grid_bounds,
    grid_indices_to_world,
    is_within_grid,
    world_to_ego,
    world_to_grid_indices,
)

# ---------------------------------------------------------------------------
# World ↔ Grid round-trip invariants
# ---------------------------------------------------------------------------


class TestWorldGridRoundTrip:
    """Round-trip invariants: world ↔ grid indices conversions."""

    def test_cell_center_round_trip_default_origin(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        cells = [(0, 0), (50, 50), (0, 99), (99, 0), (99, 99), (25, 75), (33, 66)]
        for row, col in cells:
            wx, wy = grid_indices_to_world(row, col, config)
            r2, c2 = world_to_grid_indices(wx, wy, config)
            assert (row, col) == (r2, c2)

    def test_cell_center_round_trip_offset_origin(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        ox, oy = 10.0, 20.0
        cells = [(0, 0), (50, 50), (99, 99)]
        for row, col in cells:
            wx, wy = grid_indices_to_world(row, col, config, ox, oy)
            r2, c2 = world_to_grid_indices(wx, wy, config, ox, oy)
            assert (row, col) == (r2, c2)

    def test_arbitrary_world_point_lands_in_correct_cell(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        points = [(2.5, 3.7), (0.05, 0.05), (9.95, 9.95), (5.0, 5.0), (1.23, 4.56)]
        for wx, wy in points:
            row, col = world_to_grid_indices(wx, wy, config)
            cx, cy = grid_indices_to_world(row, col, config)
            half = config.resolution / 2.0
            assert abs(wx - cx) <= half + 1e-9
            assert abs(wy - cy) <= half + 1e-9

    def test_round_trip_with_offset_origin(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        origins = [(0.0, 0.0), (-5.0, -5.0), (10.0, 20.0), (100.0, 200.0)]
        for ox, oy in origins:
            for row, col in [(0, 0), (50, 50), (99, 99)]:
                wx, wy = grid_indices_to_world(row, col, config, ox, oy)
                r2, c2 = world_to_grid_indices(wx, wy, config, ox, oy)
                assert (row, col) == (r2, c2)

    def test_different_resolutions(self):
        for res in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            config = GridConfig(resolution=res, width=10.0, height=10.0)
            max_r = config.grid_height - 1
            max_c = config.grid_width - 1
            cells = [(0, 0), (max_r, max_c), (0, max_c), (max_r, 0)]
            if max_r >= 2 and max_c >= 2:
                mid_r, mid_c = max_r // 2, max_c // 2
                cells.append((mid_r, mid_c))
            for row, col in cells:
                wx, wy = grid_indices_to_world(row, col, config)
                r2, c2 = world_to_grid_indices(wx, wy, config)
                assert (row, col) == (r2, c2)

    def test_non_square_grid(self):
        config = GridConfig(resolution=0.5, width=5.0, height=10.0)
        for row, col in [(0, 0), (19, 9), (0, 9), (19, 0)]:
            wx, wy = grid_indices_to_world(row, col, config)
            r2, c2 = world_to_grid_indices(wx, wy, config)
            assert (row, col) == (r2, c2)


# ---------------------------------------------------------------------------
# World ↔ Ego round-trip invariants
# ---------------------------------------------------------------------------


class TestWorldEgoRoundTrip:
    """Round-trip invariants: world ↔ ego frame conversions."""

    def test_zero_pose_identity(self):
        pose = ((0.0, 0.0), 0.0)
        points = [(0.0, 0.0), (1.0, 2.0), (-3.0, 4.0), (10.0, -5.0)]
        for wx, wy in points:
            ex, ey = world_to_ego(wx, wy, pose)
            assert ex == pytest.approx(wx)
            assert ey == pytest.approx(wy)

    def test_round_trip_world_to_ego_to_world(self):
        poses = [
            ((0.0, 0.0), 0.0),
            ((5.0, 5.0), 0.0),
            ((0.0, 0.0), np.pi / 4),
            ((3.0, -2.0), np.pi / 2),
            ((-10.0, 10.0), -np.pi / 3),
            ((1.0, 2.0), 1.57),
            ((-5.0, 10.0), -np.pi),
            ((0.0, 0.0), 2 * np.pi),
            ((4.0, 4.0), -0.785398),
        ]
        points = [
            (0.0, 0.0),
            (1.0, 2.0),
            (-3.0, 4.0),
            (10.0, -5.0),
            (0.5, -0.5),
            (100.0, 100.0),
            (-50.0, 200.0),
        ]
        for pose in poses:
            for wx, wy in points:
                ex, ey = world_to_ego(wx, wy, pose)
                wx2, wy2 = ego_to_world(ex, ey, pose)
                assert wx2 == pytest.approx(wx, abs=1e-10)
                assert wy2 == pytest.approx(wy, abs=1e-10)

    def test_round_trip_ego_to_world_to_ego(self):
        poses = [
            ((0.0, 0.0), 0.0),
            ((5.0, 5.0), 0.0),
            ((3.0, -2.0), np.pi / 2),
            ((1.0, 2.0), np.pi / 4),
            ((-10.0, 10.0), -np.pi / 3),
        ]
        ego_points = [(0.0, 0.0), (1.0, 2.0), (-3.0, 4.0), (10.0, -5.0)]
        for pose in poses:
            for ex, ey in ego_points:
                wx, wy = ego_to_world(ex, ey, pose)
                ex2, ey2 = world_to_ego(wx, wy, pose)
                assert ex2 == pytest.approx(ex, abs=1e-10)
                assert ey2 == pytest.approx(ey, abs=1e-10)

    def test_rotation_preserves_distance(self):
        robot_x, robot_y = 5.0, 5.0
        pose = ((robot_x, robot_y), np.pi / 6)
        world_points = [
            (5.0, 5.0),
            (6.0, 5.0),
            (5.0, 7.0),
            (1.0, 2.0),
            (10.0, 10.0),
            (0.0, 0.0),
            (-3.0, 8.0),
        ]
        for wx, wy in world_points:
            ex, ey = world_to_ego(wx, wy, pose)
            world_dist = np.hypot(wx - robot_x, wy - robot_y)
            ego_dist = np.hypot(ex, ey)
            assert ego_dist == pytest.approx(world_dist, abs=1e-10)

    def test_rotation_preserves_distance_robot_at_origin(self):
        pose = ((0.0, 0.0), np.pi / 3)
        points = [(1.0, 0.0), (0.0, 2.0), (-3.0, 4.0), (5.0, -5.0)]
        for wx, wy in points:
            ex, ey = world_to_ego(wx, wy, pose)
            assert np.hypot(ex, ey) == pytest.approx(np.hypot(wx, wy), abs=1e-10)

    def test_negative_theta_round_trip(self):
        pose = ((2.0, 3.0), -2.5)
        for wx, wy in [(2.0, 3.0), (5.0, 7.0), (0.0, 0.0)]:
            ex, ey = world_to_ego(wx, wy, pose)
            wx2, wy2 = ego_to_world(ex, ey, pose)
            assert wx2 == pytest.approx(wx, abs=1e-10)
            assert wy2 == pytest.approx(wy, abs=1e-10)


# ---------------------------------------------------------------------------
# Grid bounds queries
# ---------------------------------------------------------------------------


class TestGridBounds:
    """Bounds queries: is_within_grid, clip_to_grid, get_grid_bounds."""

    def test_is_within_grid_inside(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        assert is_within_grid(0.0, 0.0, config) is True
        assert is_within_grid(10.0, 10.0, config) is True
        assert is_within_grid(5.0, 5.0, config) is True
        assert is_within_grid(0.0, 5.0, config) is True
        assert is_within_grid(10.0, 5.0, config) is True
        assert is_within_grid(5.0, 0.0, config) is True
        assert is_within_grid(5.0, 10.0, config) is True

    def test_is_within_grid_outside(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        assert is_within_grid(-0.1, 5.0, config) is False
        assert is_within_grid(5.0, -0.1, config) is False
        assert is_within_grid(10.1, 5.0, config) is False
        assert is_within_grid(5.0, 10.1, config) is False
        assert is_within_grid(-100.0, -100.0, config) is False
        assert is_within_grid(100.0, 100.0, config) is False

    def test_is_within_grid_offset_origin(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        ox, oy = 10.0, 20.0
        assert is_within_grid(10.0, 20.0, config, ox, oy) is True
        assert is_within_grid(20.0, 30.0, config, ox, oy) is True
        assert is_within_grid(9.9, 20.0, config, ox, oy) is False
        assert is_within_grid(10.0, 19.9, config, ox, oy) is False
        assert is_within_grid(20.1, 30.0, config, ox, oy) is False
        assert is_within_grid(20.0, 30.1, config, ox, oy) is False

    def test_clip_to_grid_inside_unchanged(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        assert clip_to_grid(5.0, 5.0, config) == (5.0, 5.0)
        assert clip_to_grid(0.0, 0.0, config) == (0.0, 0.0)
        assert clip_to_grid(10.0, 10.0, config) == (10.0, 10.0)

    def test_clip_to_grid_outside_clamped(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        assert clip_to_grid(-1.0, 5.0, config) == (0.0, 5.0)
        assert clip_to_grid(5.0, -1.0, config) == (5.0, 0.0)
        assert clip_to_grid(15.0, 5.0, config) == (10.0, 5.0)
        assert clip_to_grid(5.0, 15.0, config) == (5.0, 10.0)
        assert clip_to_grid(-100.0, -100.0, config) == (0.0, 0.0)
        assert clip_to_grid(100.0, 100.0, config) == (10.0, 10.0)

    def test_clip_to_grid_offset_origin(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        ox, oy = 10.0, 20.0
        cx, cy = clip_to_grid(5.0, 25.0, config, ox, oy)
        assert cx == 10.0 and cy == 25.0
        cx, cy = clip_to_grid(15.0, 15.0, config, ox, oy)
        assert cx == 15.0 and cy == 20.0
        cx, cy = clip_to_grid(100.0, 100.0, config, ox, oy)
        assert cx == 20.0 and cy == 30.0

    def test_clip_to_grid_is_within_grid_consistency(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        points = [(-5.0, -5.0), (15.0, 15.0), (5.0, 5.0), (-100.0, 100.0), (0.0, 0.0)]
        for wx, wy in points:
            cx, cy = clip_to_grid(wx, wy, config)
            assert is_within_grid(cx, cy, config)

    def test_get_grid_bounds_default_origin(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        assert get_grid_bounds(config) == (0.0, 10.0, 0.0, 10.0)

    def test_get_grid_bounds_offset_origin(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        assert get_grid_bounds(config, -5.0, -5.0) == (-5.0, 5.0, -5.0, 5.0)
        assert get_grid_bounds(config, 10.0, 20.0) == (10.0, 20.0, 20.0, 30.0)

    def test_get_grid_bounds_different_sizes(self):
        config = GridConfig(resolution=0.1, width=20.0, height=30.0)
        assert get_grid_bounds(config) == (0.0, 20.0, 0.0, 30.0)

    def test_get_grid_bounds_multiple_resolutions(self):
        for res in [0.05, 0.1, 0.5, 1.0]:
            config = GridConfig(resolution=res, width=10.0, height=20.0)
            assert get_grid_bounds(config) == (0.0, 10.0, 0.0, 20.0)


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestWorldGridEdgeCases:
    """Edge cases and boundary conditions for coordinate transforms."""

    def test_exact_boundary_points_bottom_left(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        row, col = world_to_grid_indices(0.0, 0.0, config)
        assert (row, col) == (0, 0)

    def test_exact_boundary_points_top_right(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        row, col = world_to_grid_indices(10.0, 10.0, config)
        assert (row, col) == (99, 99)

    def test_out_of_bounds_world_raises(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        with pytest.raises(ValueError):
            world_to_grid_indices(-0.1, 5.0, config)
        with pytest.raises(ValueError):
            world_to_grid_indices(5.0, -0.1, config)
        with pytest.raises(ValueError):
            world_to_grid_indices(10.1, 5.0, config)
        with pytest.raises(ValueError):
            world_to_grid_indices(5.0, 10.1, config)

    def test_grid_indices_out_of_bounds_raises(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        with pytest.raises(ValueError):
            grid_indices_to_world(-1, 0, config)
        with pytest.raises(ValueError):
            grid_indices_to_world(0, -1, config)
        with pytest.raises(ValueError):
            grid_indices_to_world(config.grid_height, 0, config)
        with pytest.raises(ValueError):
            grid_indices_to_world(0, config.grid_width, config)

    def test_grid_indices_last_valid_indices(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        last_r = config.grid_height - 1
        last_c = config.grid_width - 1
        wx, wy = grid_indices_to_world(last_r, last_c, config)
        r2, c2 = world_to_grid_indices(wx, wy, config)
        assert (r2, c2) == (last_r, last_c)

    def test_negative_origin_world_maps_correctly(self):
        config = GridConfig(resolution=0.5, width=10.0, height=10.0)
        ox, oy = -5.0, -5.0
        row, col = world_to_grid_indices(-5.0, -5.0, config, ox, oy)
        assert (row, col) == (0, 0)
        wx, wy = grid_indices_to_world(row, col, config, ox, oy)
        assert wx == pytest.approx(-4.75)
        assert wy == pytest.approx(-4.75)

    def test_partial_grid_coverage_round_trip(self):
        config = GridConfig(resolution=0.3, width=3.0, height=3.0)
        for row in range(config.grid_height):
            for col in range(config.grid_width):
                wx, wy = grid_indices_to_world(row, col, config)
                r2, c2 = world_to_grid_indices(wx, wy, config)
                assert (row, col) == (r2, c2)

    def test_world_point_near_bottom_left_corner(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        row, col = world_to_grid_indices(0.01, 0.01, config)
        assert (row, col) == (0, 0)

    def test_world_point_near_top_right_corner(self):
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        row, col = world_to_grid_indices(9.99, 9.99, config)
        assert (row, col) == (99, 99)


class TestEgoEdgeCases:
    """Edge cases for ego-frame transformations."""

    def test_point_at_robot_origin(self):
        pose = ((5.0, 5.0), 0.0)
        ex, ey = world_to_ego(5.0, 5.0, pose)
        assert ex == pytest.approx(0.0)
        assert ey == pytest.approx(0.0)

    def test_point_at_robot_origin_rotated(self):
        pose = ((5.0, 5.0), np.pi / 4)
        ex, ey = world_to_ego(5.0, 5.0, pose)
        assert ex == pytest.approx(0.0)
        assert ey == pytest.approx(0.0)

    def test_pure_translation_no_rotation(self):
        pose = ((3.0, 4.0), 0.0)
        ex, ey = world_to_ego(5.0, 7.0, pose)
        assert ex == pytest.approx(2.0)
        assert ey == pytest.approx(3.0)

    def test_pure_rotation_no_translation(self):
        pose = ((0.0, 0.0), np.pi / 2)
        ex, ey = world_to_ego(0.0, 1.0, pose)
        assert ex == pytest.approx(1.0, abs=1e-10)
        assert ey == pytest.approx(0.0, abs=1e-10)

    def test_full_turn_round_trip(self):
        pose = ((2.0, 3.0), 2 * np.pi)
        for wx, wy in [(2.0, 3.0), (5.0, 7.0), (0.0, 0.0)]:
            ex, ey = world_to_ego(wx, wy, pose)
            wx2, wy2 = ego_to_world(ex, ey, pose)
            assert wx2 == pytest.approx(wx, abs=1e-10)
            assert wy2 == pytest.approx(wy, abs=1e-10)
