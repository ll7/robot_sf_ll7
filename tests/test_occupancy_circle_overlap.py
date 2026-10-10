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
from robot_sf.nav import occupancy_grid_utils
from robot_sf.nav.occupancy_grid import GridConfig, OccupancyGrid
from robot_sf.nav.occupancy_grid_rasterization import rasterize_circle, rasterize_circle_fast
from robot_sf.nav.occupancy_grid_utils import get_affected_cells


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

    def test_fast_rasterization_preserves_higher_values_and_unmasked_cells(self):
        """Fast circle rasterization should only raise masked cells below value."""
        config = GridConfig(resolution=1.0, width=7.0, height=7.0)
        grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
        grid[3, 3] = 0.9
        grid[0, 0] = 0.7

        rasterize_circle_fast(((3.0, 3.0), 1.5), grid, config, value=0.5)

        assert grid[3, 3] == 0.9, "Existing higher occupancy inside the mask should be preserved"
        assert grid[0, 0] == 0.7, "Cells outside the mask should not be changed"
        assert np.any(np.isclose(grid, 0.5)), "Masked lower cells should be raised to value"


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


def test_affected_cells_avoids_per_cell_world_conversion(monkeypatch):
    """Affected-cell bounds should not recompute world centers once per candidate cell."""
    config = GridConfig(resolution=0.05, width=20.0, height=20.0)

    def fail_per_cell_conversion(*_args, **_kwargs):
        raise AssertionError(
            "get_affected_cells should precompute cell bounds without per-cell conversion"
        )

    monkeypatch.setattr(occupancy_grid_utils, "grid_indices_to_world", fail_per_cell_conversion)

    inside_rows, inside_cols = get_affected_cells(10.0, 10.0, 1.25, config)
    outside_rows, outside_cols = get_affected_cells(20.75, 10.0, 1.0, config)

    assert len(inside_rows) > 0
    assert len(outside_rows) > 0
    assert all(
        0 <= row < config.grid_height and 0 <= col < config.grid_width
        for row, col in zip(inside_rows, inside_cols, strict=False)
    )
    assert all(
        0 <= row < config.grid_height and 0 <= col < config.grid_width
        for row, col in zip(outside_rows, outside_cols, strict=False)
    )
    assert any(col == config.grid_width - 1 for col in outside_cols)


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


@pytest.mark.parametrize(
    "center,radius,expected",
    [
        ((2.5, 2.5), 0.2, {(2, 2)}),
        ((2.5, 2.5), 1.0, {(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)}),
        ((2.5, 2.5), 1.5, {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}),
        # Centre on a cell boundary; both tangent cell centres are included.
        ((3.0, 2.5), 0.5, {(2, 2), (2, 3)}),
        ((2.25, 2.75), 0.4, {(2, 2)}),
        # Outside centre with one tangent centre inside the clipped grid.
        ((-0.25, 2.5), 0.75, {(2, 0)}),
    ],
)
@pytest.mark.parametrize("resolution,origin", [(1.0, (0.0, 0.0)), (0.25, (-4.0, 7.0))])
def test_fast_circle_exact_cell_centres(center, radius, expected, resolution, origin):
    """Hand-calculated masks sample centres, including tangency and clipping."""
    config = GridConfig(resolution=resolution, width=6 * resolution, height=6 * resolution)
    grid = np.zeros((6, 6), dtype=np.float32)
    world_center = tuple(origin[i] + center[i] * resolution for i in range(2))
    rasterize_circle_fast((world_center, radius * resolution), grid, config, *origin)
    assert {tuple(cell) for cell in np.argwhere(grid > 0)} == expected


def test_pedestrian_circle_centroid_and_polygon_agree():
    """A symmetric isolated pedestrian stays centred and matches polygon filling."""
    from robot_sf.nav.occupancy_grid_rasterization import (
        rasterize_pedestrians_array,
        rasterize_polygon,
    )

    config = GridConfig(resolution=0.25, width=3.0, height=3.0)
    circle_grid = np.zeros((12, 12), dtype=np.float32)
    polygon_grid = np.zeros_like(circle_grid)
    center = np.array([1.5, 1.5])
    radius = 0.6  # No sampled centre lies on the polygon/circle boundary.
    rasterize_pedestrians_array(center[None, :], np.array([radius]), circle_grid, config)
    angles = np.linspace(0.0, 2 * np.pi, 256, endpoint=False)
    vertices = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    rasterize_polygon([tuple(vertex) for vertex in vertices], polygon_grid, config)

    rows, cols = np.nonzero(circle_grid)
    centroid = (np.array([cols.mean(), rows.mean()]) + 0.5) * config.resolution
    np.testing.assert_allclose(centroid, center, atol=0.25 * config.resolution, rtol=0)
    np.testing.assert_array_equal(circle_grid, polygon_grid)


@pytest.mark.parametrize("array_input", [False, True])
def test_circle_channels_reach_observation_without_offset(array_input):
    """List/array pedestrians and robot centres survive grid generation unchanged."""
    from robot_sf.nav.occupancy_grid import GridChannel

    config = GridConfig(
        resolution=1.0, width=6.0, height=6.0, robot_radius=0.2, channels=list(GridChannel)
    )
    grid = OccupancyGrid(config)
    pedestrians = (np.array([[2.5, 2.5]]), np.array([0.2])) if array_input else [((2.5, 2.5), 0.2)]
    grid.generate(obstacles=[], pedestrians=pedestrians, robot_pose=((4.5, 4.5), 0.0))
    observation = grid.to_observation()
    for channel, expected in [
        (GridChannel.PEDESTRIANS, {(2, 2)}),
        (GridChannel.ROBOT, {(4, 4)}),
        (GridChannel.COMBINED, {(2, 2), (4, 4)}),
    ]:
        index = config.channels.index(channel)
        assert {tuple(cell) for cell in np.argwhere(observation[index] > 0)} == expected
    assert not np.any(grid.get_channel(GridChannel.OBSTACLES))
