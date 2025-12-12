"""Tests for motion planning adapter module.

Note: Full integration testing is performed via examples/advanced/27_motion_planning_adapter_test.py
which validates end-to-end functionality with real SVG maps and path planning.
"""

from __future__ import annotations

import pytest

from robot_sf.nav.motion_planning_adapter import MotionPlanningGridConfig, _world_to_grid


class TestMotionPlanningGridConfig:
    """Test suite for MotionPlanningGridConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = MotionPlanningGridConfig()
        assert cfg.cells_per_meter == 1.0
        assert cfg.add_boundary_obstacles is True
        assert cfg.inflate_radius_cells is None

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = MotionPlanningGridConfig(
            cells_per_meter=2.0, add_boundary_obstacles=False, inflate_radius_cells=3
        )
        assert cfg.cells_per_meter == 2.0
        assert cfg.add_boundary_obstacles is False
        assert cfg.inflate_radius_cells == 3

    def test_meters_per_cell_property(self):
        """Test the meters_per_cell computed property."""
        cfg = MotionPlanningGridConfig(cells_per_meter=2.0)
        assert cfg.meters_per_cell == 0.5

        cfg = MotionPlanningGridConfig(cells_per_meter=4.0)
        assert cfg.meters_per_cell == 0.25

    def test_meters_per_cell_inverse_relationship(self):
        """Test that meters_per_cell is correctly the inverse of cells_per_meter."""
        for cpm in [0.5, 1.0, 2.0, 5.0, 10.0]:
            cfg = MotionPlanningGridConfig(cells_per_meter=cpm)
            assert cfg.meters_per_cell == pytest.approx(1.0 / cpm)


class TestWorldToGrid:
    """Test suite for _world_to_grid coordinate conversion."""

    def test_basic_conversion(self):
        """Test basic coordinate scaling."""
        assert _world_to_grid(0.0, 1.0) == 0
        assert _world_to_grid(5.0, 1.0) == 5
        assert _world_to_grid(10.0, 1.0) == 10

    def test_fractional_coordinates(self):
        """Test conversion with fractional world coordinates."""
        assert _world_to_grid(2.3, 1.0) == 2
        assert _world_to_grid(2.7, 1.0) == 2
        assert _world_to_grid(2.99, 1.0) == 2
        assert _world_to_grid(3.0, 1.0) == 3

    def test_different_scales(self):
        """Test conversion with different cells_per_meter values."""
        assert _world_to_grid(5.0, 2.0) == 10
        assert _world_to_grid(5.0, 0.5) == 2
        assert _world_to_grid(5.0, 4.0) == 20

    def test_negative_coordinates(self):
        """Test conversion with negative coordinates (should floor towards negative)."""
        assert _world_to_grid(-1.0, 1.0) == -1
        assert _world_to_grid(-2.3, 1.0) == -3
        assert _world_to_grid(-0.5, 1.0) == -1

    def test_zero_coordinate(self):
        """Test conversion at origin."""
        assert _world_to_grid(0.0, 1.0) == 0
        assert _world_to_grid(0.0, 2.0) == 0
        assert _world_to_grid(0.0, 0.5) == 0
