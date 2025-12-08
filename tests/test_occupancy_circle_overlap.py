"""Tests for circle rasterization with centers outside grid bounds.

This module specifically tests the edge case where a circle's center is outside
the grid bounds but the circle itself partially overlaps the grid. This catches
a logic error where circles were incorrectly skipped when their centers were
outside, even though they should have been partially rasterized.

Test Organization:
- Circles overlapping from each edge (left, right, top, bottom)
- Circles overlapping from corners
- Circles fully outside (should be skipped)
- Circles fully inside (baseline)
- Parametrized boundary detection tests
"""

import numpy as np
import pytest
from loguru import logger

from robot_sf.common.types import Circle2D
from robot_sf.nav.occupancy_grid import GridConfig, OccupancyGrid
from robot_sf.nav.occupancy_grid_rasterization import rasterize_circle


class TestCircleCenterOutsideButOverlapping:
    """Test circles with centers outside grid that partially overlap."""

    def test_right_edge_overlap(self):
        """Circle center outside right edge should still rasterize overlap."""
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

        # Circle at (10.5, 5.0) with radius 1.0
        # Grid x range: [0, 10], circle x range: [9.5, 11.5]
        # Should overlap from x=9.5 to x=10.0
        circle: Circle2D = ((10.5, 5.0), 1.0)
        rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

        occupied_cells = np.sum(grid > 0)
        logger.info(f"Right edge overlap: {occupied_cells} cells occupied")

        assert occupied_cells > 0, "Circle overlapping from right should occupy cells"
        assert occupied_cells < 80, "Partial overlap should not fill entire circle"

        # Check that rightmost column has occupied cells
        rightmost_column = grid[:, -1]
        assert np.any(rightmost_column > 0), "Rightmost column should be occupied"

    def test_left_edge_overlap(self):
        """Circle center outside left edge should still rasterize overlap."""
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

        # Circle at (-0.5, 5.0) with radius 1.0
        # Grid x range: [0, 10], circle x range: [-1.5, 0.5]
        # Should overlap from x=0.0 to x=0.5
        circle: Circle2D = ((-0.5, 5.0), 1.0)
        rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

        occupied_cells = np.sum(grid > 0)
        logger.info(f"Left edge overlap: {occupied_cells} cells occupied")

        assert occupied_cells > 0, "Circle overlapping from left should occupy cells"

        # Check that leftmost column has occupied cells
        leftmost_column = grid[:, 0]
        assert np.any(leftmost_column > 0), "Leftmost column should be occupied"

    def test_top_edge_overlap(self):
        """Circle center outside top edge should still rasterize overlap."""
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

        # Circle at (5.0, 10.5) with radius 1.0
        circle: Circle2D = ((5.0, 10.5), 1.0)
        rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

        occupied_cells = np.sum(grid > 0)
        logger.info(f"Top edge overlap: {occupied_cells} cells occupied")

        assert occupied_cells > 0, "Circle overlapping from top should occupy cells"

    def test_bottom_edge_overlap(self):
        """Circle center outside bottom edge should still rasterize overlap."""
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

        # Circle at (5.0, -0.5) with radius 1.0
        circle: Circle2D = ((5.0, -0.5), 1.0)
        rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

        occupied_cells = np.sum(grid > 0)
        logger.info(f"Bottom edge overlap: {occupied_cells} cells occupied")

        assert occupied_cells > 0, "Circle overlapping from bottom should occupy cells"


class TestCircleFullyOutsideGrid:
    """Test that circles fully outside grid are correctly skipped."""

    def test_far_right_no_overlap(self):
        """Circle far outside right should not occupy any cells."""
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

        # Circle at (15.0, 5.0) with radius 1.0 - fully outside
        circle: Circle2D = ((15.0, 5.0), 1.0)
        rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

        occupied_cells = np.sum(grid > 0)
        assert occupied_cells == 0, "Circle fully outside should not occupy cells"

    def test_far_left_no_overlap(self):
        """Circle far outside left should not occupy any cells."""
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

        circle: Circle2D = ((-5.0, 5.0), 1.0)
        rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

        assert np.sum(grid > 0) == 0, "Circle fully outside should not occupy cells"


class TestCircleCenterInsideGrid:
    """Baseline test for circles fully inside grid."""

    def test_center_inside_normal_rasterization(self):
        """Circle with center inside grid should rasterize normally."""
        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

        # Circle fully inside at (5.0, 5.0) with radius 0.5
        circle: Circle2D = ((5.0, 5.0), 0.5)
        rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

        occupied_cells = np.sum(grid > 0)
        logger.info(f"Center inside: {occupied_cells} cells occupied")

        # Should occupy approximately π * (0.5/0.1)² ≈ 78 cells
        assert occupied_cells > 60, "Circle inside grid should occupy cells"
        assert occupied_cells < 100, "Circle occupancy should be bounded"


@pytest.mark.parametrize(
    "center_x,center_y,radius,should_overlap,description",
    [
        # Overlapping cases
        (10.5, 5.0, 1.0, True, "Right edge overlap"),
        (-0.5, 5.0, 1.0, True, "Left edge overlap"),
        (5.0, 10.5, 1.0, True, "Top edge overlap"),
        (5.0, -0.5, 1.0, True, "Bottom edge overlap"),
        (10.7, 10.7, 1.0, True, "Top-right corner overlap"),
        (-0.7, -0.7, 1.0, True, "Bottom-left corner overlap"),
        # Non-overlapping cases
        (15.0, 5.0, 1.0, False, "Far right - no overlap"),
        (-5.0, 5.0, 1.0, False, "Far left - no overlap"),
        (5.0, 15.0, 1.0, False, "Far top - no overlap"),
        (5.0, -5.0, 1.0, False, "Far bottom - no overlap"),
        (15.0, 15.0, 1.0, False, "Far corner - no overlap"),
        # Edge case: circle barely touching
        (10.95, 5.0, 1.0, True, "Barely touching right edge"),
    ],
)
def test_circle_boundary_detection(
    center_x: float,
    center_y: float,
    radius: float,
    should_overlap: bool,
    description: str,
):
    """Parametrized test for various circle positions relative to grid boundary."""
    config = GridConfig(resolution=0.1, width=10.0, height=10.0)
    grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)

    circle: Circle2D = ((center_x, center_y), radius)
    rasterize_circle(circle, grid, config, grid_origin_x=0.0, grid_origin_y=0.0)

    occupied = np.sum(grid > 0)

    if should_overlap:
        assert occupied > 0, (
            f"{description}: Circle at ({center_x}, {center_y}) should overlap grid (got {occupied} cells)"
        )
    else:
        assert occupied == 0, (
            f"{description}: Circle at ({center_x}, {center_y}) should not overlap grid (got {occupied} cells)"
        )


class TestCircleOverlapWithOccupancyGrid:
    """Integration test using OccupancyGrid API."""

    def test_pedestrians_outside_grid_centers(self):
        """Pedestrians with centers outside should still appear in grid."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        # Pedestrians with centers outside but overlapping
        pedestrians = [
            (np.array([10.5, 5.0]), 0.8),  # Right edge
            (np.array([-0.5, 5.0]), 0.8),  # Left edge
            (np.array([5.0, 10.5]), 0.8),  # Top edge
            (np.array([5.0, -0.5]), 0.8),  # Bottom edge
        ]
        robot_pose = (np.array([5.0, 5.0]), 0.0)

        grid_data = grid.generate(obstacles=[], pedestrians=pedestrians, robot_pose=robot_pose)

        assert grid_data is not None
        ped_channel = grid_data[1]  # PEDESTRIANS channel
        occupied_cells = np.sum(ped_channel > 0)

        logger.info(
            f"Pedestrians outside centers integration test: {occupied_cells} cells occupied"
        )

        # All 4 pedestrians should contribute some cells
        assert occupied_cells > 0, "Pedestrians overlapping grid should occupy cells"
        # Each pedestrian should contribute, so expect reasonable number
        assert occupied_cells > 10, "Multiple overlapping pedestrians should occupy multiple cells"
