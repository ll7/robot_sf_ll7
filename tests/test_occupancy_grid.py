"""Tests for Occupancy Grid Core Functionality (US1).

User Story 1 (US1): Grid Generation
Tests the creation of rasterized occupancy grids from obstacles and pedestrians.

Scope:
- Grid initialization with various configurations
- Obstacle rasterization (line segments)
- Pedestrian rasterization (circular objects)
- Multi-channel grid generation
- Ego-frame vs world-frame grids
- Grid shape and dtype validation

Success Criteria:
- T019: Grid generation O(N*M) performance (<5ms)
- T020: 100% test coverage of grid generation code
- T021: All fixtures produce valid grids
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid


class TestGridInitialization:
    """T001: Test grid initialization with various configurations."""

    def test_simple_grid_creation(self, simple_grid_config):
        """Test creating a simple grid."""
        grid = OccupancyGrid(config=simple_grid_config)

        assert grid.config == simple_grid_config
        assert not grid.is_initialized
        assert grid._grid_data is None

    def test_grid_shape_properties(self, simple_grid_config):
        """Test grid shape calculations."""
        grid = OccupancyGrid(config=simple_grid_config)

        assert grid.config.grid_width == 100  # 10.0m / 0.1m
        assert grid.config.grid_height == 100
        assert grid.config.num_channels == 2
        assert grid.shape == (2, 100, 100)

    def test_large_grid_creation(self, large_grid_config):
        """Test creating a larger grid."""
        grid = OccupancyGrid(config=large_grid_config)

        assert grid.config.grid_width == 200
        assert grid.config.grid_height == 200
        assert grid.config.num_channels == 3
        assert grid.shape == (3, 200, 200)

    def test_coarse_grid_creation(self, coarse_grid_config):
        """Test creating a coarse-resolution grid."""
        grid = OccupancyGrid(config=coarse_grid_config)

        assert grid.config.grid_width == 20
        assert grid.config.grid_height == 20

    def test_single_channel_grid(self, single_channel_config):
        """Test grid with single channel."""
        grid = OccupancyGrid(config=single_channel_config)

        assert grid.config.num_channels == 1
        assert grid.shape == (1, 100, 100)


class TestGridGeneration:
    """T002: Test grid generation from obstacles and pedestrians."""

    def test_basic_grid_generation(
        self, occupancy_grid, simple_obstacles, simple_pedestrians, robot_pose_center
    ):
        """Test basic grid generation with simple inputs."""
        grid = occupancy_grid
        grid_data = grid.generate(
            obstacles=simple_obstacles,
            pedestrians=simple_pedestrians,
            robot_pose=robot_pose_center,
        )

        assert grid.is_initialized
        assert grid_data is not None
        assert grid_data.shape == grid.shape
        assert grid_data.dtype == grid.config.dtype

    def test_grid_generation_returns_array(
        self, occupancy_grid, simple_obstacles, robot_pose_center
    ):
        """Test that generate() returns numpy array."""
        result = occupancy_grid.generate(
            obstacles=simple_obstacles,
            pedestrians=[],
            robot_pose=robot_pose_center,
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == occupancy_grid.config.dtype

    def test_empty_grid_generation(self, occupancy_grid, robot_pose_center):
        """Test grid generation with no obstacles or pedestrians."""
        grid_data = occupancy_grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=robot_pose_center,
        )

        assert grid_data.shape == occupancy_grid.shape
        # All cells should be unoccupied (0.0)
        # TODO: Verify after implementation

    def test_grid_generation_multiple_obstacles(
        self, occupancy_grid, complex_obstacles, robot_pose_center
    ):
        """Test grid generation with many obstacles."""
        grid_data = occupancy_grid.generate(
            obstacles=complex_obstacles,
            pedestrians=[],
            robot_pose=robot_pose_center,
        )

        assert grid_data.shape == occupancy_grid.shape

    def test_grid_generation_multiple_pedestrians(
        self, occupancy_grid, simple_obstacles, crowded_pedestrians, robot_pose_center
    ):
        """Test grid generation with many pedestrians."""
        grid_data = occupancy_grid.generate(
            obstacles=simple_obstacles,
            pedestrians=crowded_pedestrians,
            robot_pose=robot_pose_center,
        )

        assert grid_data.shape == occupancy_grid.shape


class TestGridChannels:
    """T003: Test multi-channel grid operations."""

    def test_get_single_channel(self, pre_generated_grid):
        """Test extracting a single channel."""
        obstacles_channel = pre_generated_grid.get_channel(GridChannel.OBSTACLES)

        assert obstacles_channel.shape == (100, 100)
        assert obstacles_channel.dtype == pre_generated_grid.config.dtype

    def test_get_all_channels(self, pre_generated_grid):
        """Test accessing all channels."""
        for channel in pre_generated_grid.config.channels:
            channel_data = pre_generated_grid.get_channel(channel)
            assert channel_data is not None
            assert channel_data.shape == (100, 100)

    def test_get_unavailable_channel_raises(self, single_channel_config):
        """Test that accessing unavailable channel raises error."""
        grid = OccupancyGrid(config=single_channel_config)
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=((5.0, 5.0), 0.0),
        )

        with pytest.raises(ValueError, match="not in grid"):
            grid.get_channel(GridChannel.PEDESTRIANS)


class TestGridBounds:
    """T004: Test grid bounds checking and coordinate validation."""

    def test_coordinate_bounds_checking(self):
        """Test that out-of-bounds coordinates raise errors.

        TODO: Implement after generate() has bounds validation.
        """
        # TODO (Phase 2): Add bounds checking to generate()
        pass

    def test_grid_origin_offset(self):
        """Test that grid origin offset works correctly.

        TODO: Implement after coordinate transform functions are used.
        """
        # TODO (Phase 2): Test world coordinate transforms
        pass


class TestGridDataTypes:
    """T005: Test grid data type handling."""

    def test_float32_grid(self):
        """Test grid with float32 dtype."""
        config = GridConfig(dtype=np.float32)
        grid = OccupancyGrid(config=config)
        grid_data = grid.generate([], [], ((5.0, 5.0), 0.0))

        assert grid_data.dtype == np.float32

    def test_uint8_grid(self):
        """Test grid with uint8 dtype (binary occupancy)."""
        config = GridConfig(dtype=np.uint8)
        grid = OccupancyGrid(config=config)
        grid_data = grid.generate([], [], ((5.0, 5.0), 0.0))

        assert grid_data.dtype == np.uint8

    def test_invalid_dtype_raises(self):
        """Test that invalid dtype raises error during initialization."""
        with pytest.raises(ValueError, match="dtype must be"):
            GridConfig(dtype=np.int32)


class TestGridReset:
    """T006: Test grid reset functionality."""

    def test_grid_reset_clears_data(self, pre_generated_grid):
        """Test that reset() clears grid data."""
        assert pre_generated_grid.is_initialized

        pre_generated_grid.reset()

        assert not pre_generated_grid.is_initialized
        assert pre_generated_grid._grid_data is None

    def test_grid_reset_allows_regeneration(
        self, occupancy_grid, simple_obstacles, robot_pose_center
    ):
        """Test that grid can be regenerated after reset."""
        # First generation
        occupancy_grid.generate(simple_obstacles, [], robot_pose_center)
        assert occupancy_grid.is_initialized

        # Reset
        occupancy_grid.reset()
        assert not occupancy_grid.is_initialized

        # Second generation
        occupancy_grid.generate(simple_obstacles, [], robot_pose_center)
        assert occupancy_grid.is_initialized


class TestGridRepresentation:
    """T007: Test string representation and introspection."""

    def test_grid_repr(self, occupancy_grid):
        """Test __repr__ output."""
        repr_str = repr(occupancy_grid)

        assert "OccupancyGrid" in repr_str
        assert "not initialized" in repr_str

    def test_grid_repr_after_generation(self, pre_generated_grid):
        """Test __repr__ after generation."""
        repr_str = repr(pre_generated_grid)

        assert "OccupancyGrid" in repr_str
        assert "initialized" in repr_str


# TODO (Phase 3): Add edge case tests
# - Very small grids (2x2, 1x1)
# - Very large grids (1000x1000)
# - Grids with floating-point coordinate misalignment
# - Grids with NaN or inf values
# - Memory usage validation for large grids
