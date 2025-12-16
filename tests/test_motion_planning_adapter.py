"""Tests for motion planning adapter module.

Note: Full integration testing is performed via examples/advanced/27_motion_planning_adapter_test.py
which validates end-to-end functionality with real SVG maps and path planning.
"""

from __future__ import annotations

import numpy as np
import pytest
from python_motion_planning.common import TYPES

from robot_sf.nav.motion_planning_adapter import (
    MotionPlanningGridConfig,
    _world_to_grid,
    map_definition_to_motion_planning_grid,
)
from robot_sf.nav.svg_map_parser import convert_map


def _make_obstacle_map(tmp_path):
    svg = tmp_path / "inflation.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="5" height="5">
  <rect inkscape:label="obstacle" x="2" y="2" width="1" height="1" />
  <rect inkscape:label="robot_spawn_zone" x="0.2" y="0.2" width="0.5" height="0.5" />
  <rect inkscape:label="robot_goal_zone" x="4.2" y="0.2" width="0.5" height="0.5" />
  <path inkscape:label="robot_route_0_0" d="M 0.2 0.2 L 4.7 0.2" />
</svg>
        """.strip()
    )
    return convert_map(str(svg))


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


def test_inflation_marks_cells(tmp_path):
    """Grid inflation should mark cells with TYPES.INFLATION."""
    map_def = _make_obstacle_map(tmp_path)

    grid_no_inflation = map_definition_to_motion_planning_grid(
        map_def,
        MotionPlanningGridConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=None,
            add_boundary_obstacles=False,
        ),
    )
    grid_inflated = map_definition_to_motion_planning_grid(
        map_def,
        MotionPlanningGridConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=1,
            add_boundary_obstacles=False,
        ),
    )

    type_map_no = np.asarray(grid_no_inflation.type_map.array)
    type_map_inflated = np.asarray(grid_inflated.type_map.array)

    assert np.count_nonzero(type_map_no == TYPES.OBSTACLE) > 0
    assert np.count_nonzero(type_map_no == TYPES.INFLATION) == 0
    assert np.count_nonzero(type_map_inflated == TYPES.INFLATION) > 0
