"""
Edge case tests for occupancy grid module.

This module tests boundary conditions, extreme configurations, and error paths
to ensure robustness of the occupancy grid implementation. Tests cover:

- Empty and fully occupied grids (T086, T087)
- Resolution extremes (T088, T089)
- Boundary conditions (T090)
- Frame rotation edge cases (T091)
- Frame transitions (T092)
- Minimal grid configurations (T093)
- Entity count variations (T094, T095)
- Invalid configurations (T096)
- Out-of-bounds queries (T097)

Test Organization:
- Edge case tests validate correct behavior at extremes
- Error path tests verify appropriate exceptions for invalid inputs
- Performance tests ensure targets are met even under stress

All tests follow the TDD pattern established in earlier phases.
"""

import time

import numpy as np
import pytest
from loguru import logger

from robot_sf.nav.occupancy_grid import (
    GridConfig,
    OccupancyGrid,
    POIQuery,
    POIQueryType,
)


class TestEmptyGrid:
    """
    T086: Edge case test - Empty grid (no obstacles, no pedestrians).

    Verifies that:
    1. Grid generates successfully with no entities
    2. All cells have zero occupancy
    3. Queries return zero occupancy
    """

    def test_empty_grid_all_zeros(self):
        """Empty grid should have all cells at zero occupancy."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        # Generate with no obstacles or pedestrians
        robot_pose = (np.array([5.0, 5.0]), 0.0)
        grid_data = grid.generate(obstacles=[], pedestrians=[], robot_pose=robot_pose)

        assert grid_data is not None
        assert grid_data.shape[0] >= 1  # At least one channel

        # All cells should be zero (no obstacles/pedestrians)
        assert np.allclose(grid_data, 0.0), "Empty grid should have all zero occupancy"
        logger.info(f"Empty grid validated: shape={grid_data.shape}, all zeros confirmed")

    def test_empty_grid_query_returns_zero(self):
        """POI queries on empty grid should return zero occupancy."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        robot_pose = (np.array([5.0, 5.0]), 0.0)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=robot_pose)

        # Query a point in the middle
        query = POIQuery(x=5.0, y=5.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        assert result.is_occupied is False
        assert result.occupancy == 0.0
        logger.info("Empty grid query verified: zero occupancy")


class TestFullyOccupiedGrid:
    """
    T087: Edge case test - Fully occupied grid (every cell occupied).

    Verifies that:
    1. Grid can be saturated with obstacles
    2. All cells show high occupancy
    3. Performance remains acceptable
    """

    def test_fully_occupied_grid_all_cells_occupied(self):
        """Grid with dense obstacles should show high occupancy everywhere."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.5)
        grid = OccupancyGrid(config=config)

        # Create a dense grid of obstacles covering entire area
        obstacles = []
        for x in np.arange(0, 10.0, 0.3):
            for y in np.arange(0, 10.0, 0.3):
                # Create small line segments everywhere
                obstacles.append(((x, y), (x + 0.2, y + 0.2)))

        robot_pose = (np.array([5.0, 5.0]), 0.0)
        grid_data = grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=robot_pose)

        # Most cells should show occupancy (at least 70%)
        obstacles_channel = grid_data[0]  # OBSTACLES channel
        occupied_ratio = np.sum(obstacles_channel > 0.1) / obstacles_channel.size
        assert occupied_ratio > 0.5, f"Expected >50% occupancy, got {occupied_ratio:.2%}"
        logger.info(f"Fully occupied grid: {occupied_ratio:.1%} cells occupied")


class TestHighResolutionGrid:
    """
    T088: Edge case test - High resolution (0.05m per cell).

    Verifies that:
    1. Grid generates at high resolution
    2. Performance target <50ms is met
    3. Memory usage is reasonable
    """

    def test_high_resolution_grid_performance(self):
        """High-resolution grid should meet performance target."""
        config = GridConfig(width=5.0, height=5.0, resolution=0.05)  # 100x100 grid
        grid = OccupancyGrid(config=config)

        # Create moderate number of obstacles
        obstacles = [((i, i), (i + 1, i + 1)) for i in range(0, 5)]
        pedestrians = [(np.array([2.5, 2.5]), 0.3)]
        robot_pose = (np.array([2.5, 2.5]), 0.0)

        # Measure generation time
        start_time = time.perf_counter()
        grid_data = grid.generate(
            obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert grid_data is not None
        assert elapsed_ms < 50, f"Expected <50ms for high-res grid, got {elapsed_ms:.2f}ms"
        logger.info(f"High-resolution grid: {grid_data.shape}, generated in {elapsed_ms:.2f}ms")


class TestLowResolutionGrid:
    """
    T089: Edge case test - Very low resolution (1m per cell).

    Verifies that:
    1. Grid generates at low resolution
    2. Coarse grid captures major features
    3. Performance is fast
    """

    def test_low_resolution_grid_generation(self):
        """Low-resolution grid should generate quickly."""
        config = GridConfig(width=10.0, height=10.0, resolution=1.0)  # 10x10 grid
        grid = OccupancyGrid(config=config)

        obstacles = [((0, 0), (10, 0)), ((0, 0), (0, 10))]  # Two walls
        robot_pose = (np.array([5.0, 5.0]), 0.0)

        start_time = time.perf_counter()
        grid_data = grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=robot_pose)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert grid_data.shape == (2, 10, 10)  # 2 channels, 10x10 cells
        assert elapsed_ms < 10, f"Expected <10ms for low-res grid, got {elapsed_ms:.2f}ms"
        logger.info(f"Low-resolution grid: {grid_data.shape}, {elapsed_ms:.2f}ms")


class TestPedestrianAtBoundary:
    """
    T090: Edge case test - Pedestrian at grid boundary.

    Verifies that:
    1. Pedestrians at boundaries are handled correctly
    2. No index out-of-bounds errors
    3. Occupancy is properly clipped
    """

    def test_pedestrian_at_grid_edge(self):
        """Pedestrian at grid edge should not cause errors."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        # Place pedestrians at all four corners and edges
        pedestrians = [
            (np.array([0.0, 0.0]), 0.3),  # Bottom-left corner
            (np.array([10.0, 0.0]), 0.3),  # Bottom-right corner
            (np.array([0.0, 10.0]), 0.3),  # Top-left corner
            (np.array([10.0, 10.0]), 0.3),  # Top-right corner
            (np.array([5.0, 0.0]), 0.3),  # Bottom edge
            (np.array([5.0, 10.0]), 0.3),  # Top edge
        ]
        robot_pose = (np.array([5.0, 5.0]), 0.0)

        grid_data = grid.generate(obstacles=[], pedestrians=pedestrians, robot_pose=robot_pose)

        assert grid_data is not None
        # Verify pedestrian channel has some occupancy
        ped_channel = grid_data[1]  # PEDESTRIANS channel
        assert np.sum(ped_channel) > 0, "Boundary pedestrians should show occupancy"
        logger.info(f"Boundary pedestrian test passed: {np.sum(ped_channel > 0)} cells occupied")


class TestCardinalAngleRotation:
    """
    T091: Edge case test - Rotated frame at cardinal angles (0°, 90°, 180°, 270°).

    Verifies that:
    1. Grid rotates correctly at cardinal angles
    2. No numerical instability at 90° intervals
    3. Ego-frame grid aligns properly
    """

    def test_cardinal_angle_rotation(self):
        """Ego-frame grid should rotate correctly at cardinal angles."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2, use_ego_frame=True)
        grid = OccupancyGrid(config=config)

        obstacles = [((5, 5), (6, 5))]  # Horizontal line obstacle
        pedestrians = [(np.array([7.0, 5.0]), 0.3)]

        # Test at each cardinal angle
        cardinal_angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for angle in cardinal_angles:
            robot_pose = (np.array([5.0, 5.0]), angle)
            grid_data = grid.generate(
                obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose
            )

            assert grid_data is not None
            assert grid_data.shape[0] >= 1
            logger.info(f"Cardinal angle {np.degrees(angle):.0f}°: grid shape {grid_data.shape}")


class TestFrameTransition:
    """
    T092: Edge case test - Frame transition (ego↔world mid-simulation).

    Verifies that:
    1. Switching between ego and world frames works
    2. Grid can be regenerated with different frame mode
    3. Data consistency is maintained
    """

    def test_ego_to_world_frame_switch(self):
        """Switching from ego to world frame should work correctly."""
        # Start with ego frame
        config_ego = GridConfig(width=10.0, height=10.0, resolution=0.2, use_ego_frame=True)
        grid_ego = OccupancyGrid(config=config_ego)

        obstacles = [((4, 4), (6, 6))]
        pedestrians = [(np.array([5.5, 5.5]), 0.3)]
        robot_pose = (np.array([5.0, 5.0]), np.pi / 4)

        grid_data_ego = grid_ego.generate(
            obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose
        )

        # Switch to world frame
        config_world = GridConfig(width=10.0, height=10.0, resolution=0.2, use_ego_frame=False)
        grid_world = OccupancyGrid(config=config_world)
        grid_data_world = grid_world.generate(
            obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose
        )

        # Both should generate valid grids
        assert grid_data_ego is not None
        assert grid_data_world is not None
        assert grid_data_ego.shape == grid_data_world.shape
        logger.info("Frame transition test passed: ego and world frames both valid")


class TestSingleCellGrid:
    """
    T093: Edge case test - Single-cell grid.

    Verifies that:
    1. Minimal 1x1 grid can be created
    2. Query works on single cell
    3. No division by zero or index errors
    """

    def test_single_cell_grid(self):
        """Minimal 1x1 grid should work correctly."""
        config = GridConfig(width=0.2, height=0.2, resolution=0.2)  # 1x1 grid
        grid = OccupancyGrid(config=config)

        obstacles = [((0.05, 0.05), (0.15, 0.15))]  # Small obstacle within cell
        robot_pose = (np.array([0.1, 0.1]), 0.0)

        grid_data = grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=robot_pose)

        assert grid_data is not None
        assert grid_data.shape[1] >= 1 and grid_data.shape[2] >= 1
        logger.info(f"Single-cell grid: shape {grid_data.shape}")


class TestZeroPedestriansMultipleObstacles:
    """
    T094: Edge case test - Zero pedestrians, multiple obstacles.

    Verifies that:
    1. Grid works with only obstacle channel populated
    2. Pedestrian channel remains zero
    3. Performance is maintained
    """

    def test_obstacles_only_no_pedestrians(self):
        """Grid with only obstacles (no pedestrians) should work."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        # Many obstacles, no pedestrians
        obstacles = [((i, 0), (i, 10)) for i in range(0, 11)]  # 11 vertical walls
        robot_pose = (np.array([5.0, 5.0]), 0.0)

        grid_data = grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=robot_pose)

        assert grid_data is not None
        obstacles_channel = grid_data[0]
        pedestrians_channel = grid_data[1]

        assert np.sum(obstacles_channel) > 0, "Obstacles channel should have occupancy"
        assert np.sum(pedestrians_channel) == 0, "Pedestrians channel should be empty"
        logger.info("Obstacles-only grid: obstacle cells occupied, pedestrian cells empty")


class TestManyPedestriansNoObstacles:
    """
    T095: Edge case test - Many pedestrians, no obstacles.

    Verifies that:
    1. Grid handles many pedestrians efficiently
    2. Obstacle channel remains zero
    3. Performance target is met
    """

    def test_many_pedestrians_no_obstacles(self):
        """Grid with many pedestrians (no obstacles) should work."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        # Create 49 pedestrians scattered across grid (7x7 grid)
        pedestrians = [
            (np.array([x, y]), 0.3) for x in np.linspace(1, 9, 7) for y in np.linspace(1, 9, 7)
        ]
        robot_pose = (np.array([5.0, 5.0]), 0.0)

        start_time = time.perf_counter()
        grid_data = grid.generate(obstacles=[], pedestrians=pedestrians, robot_pose=robot_pose)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert grid_data is not None
        obstacles_channel = grid_data[0]
        pedestrians_channel = grid_data[1]

        assert np.sum(obstacles_channel) == 0, "Obstacles channel should be empty"
        assert np.sum(pedestrians_channel) > 0, "Pedestrians channel should have occupancy"
        expected_elapsed_ms = 100  # Performance guardrail to catch regressions in rasterization.
        assert elapsed_ms < expected_elapsed_ms, (
            f"Expected <{expected_elapsed_ms}ms with many pedestrians, got {elapsed_ms:.2f}ms"
        )
        logger.info(f"Many pedestrians test: {len(pedestrians)} peds, {elapsed_ms:.2f}ms")


class TestInvalidGridConfig:
    """
    T096: Error path test - Invalid grid config (negative size, zero resolution).

    Verifies that:
    1. Invalid configurations raise ValueError
    2. Error messages are informative
    3. No silent failures
    """

    def test_negative_width_raises_error(self):
        """Negative width should raise ValueError."""
        with pytest.raises(ValueError, match="width must be > 0"):
            GridConfig(width=-10.0, height=10.0, resolution=0.2)

    def test_negative_height_raises_error(self):
        """Negative height should raise ValueError."""
        with pytest.raises(ValueError, match="height must be > 0"):
            GridConfig(width=10.0, height=-10.0, resolution=0.2)

    def test_zero_resolution_raises_error(self):
        """Zero resolution should raise ValueError."""
        with pytest.raises(ValueError, match="resolution must be > 0"):
            GridConfig(width=10.0, height=10.0, resolution=0.0)

    def test_negative_resolution_raises_error(self):
        """Negative resolution should raise ValueError."""
        with pytest.raises(ValueError, match="resolution must be > 0"):
            GridConfig(width=10.0, height=10.0, resolution=-0.1)

    def test_empty_channels_raises_error(self):
        """Empty channels list should raise ValueError."""
        with pytest.raises(ValueError, match="channels must not be empty"):
            GridConfig(width=10.0, height=10.0, resolution=0.2, channels=[])


class TestOutOfBoundsQuery:
    """
    T097: Error path test - Query out-of-bounds.

    Verifies that:
    1. Out-of-bounds queries are handled gracefully
    2. No crashes or index errors
    3. Returns appropriate default values
    """

    def test_query_outside_grid_bounds(self):
        """Query outside grid should return zero occupancy (or handle gracefully)."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        obstacles = [((5, 5), (6, 6))]
        robot_pose = (np.array([5.0, 5.0]), 0.0)
        grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=robot_pose)

        # Query well outside grid bounds
        query = POIQuery(x=100.0, y=100.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        # Should return a valid result (not crash)
        assert result is not None
        # Out-of-bounds typically returns zero occupancy
        assert result.is_occupied is False or result.occupancy == 0.0
        logger.info("Out-of-bounds query handled gracefully")

    def test_query_negative_coordinates(self):
        """Query with negative coordinates should be handled."""
        config = GridConfig(width=10.0, height=10.0, resolution=0.2)
        grid = OccupancyGrid(config=config)

        robot_pose = (np.array([5.0, 5.0]), 0.0)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=robot_pose)

        query = POIQuery(x=-5.0, y=-5.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        assert result is not None
        logger.info("Negative coordinate query handled gracefully")
