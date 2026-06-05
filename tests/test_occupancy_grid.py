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

from robot_sf.nav import occupancy_grid_rasterization as rasterization
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid, POIQuery


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
        assert np.all(grid_data == 0.0)

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


class TestGridStaticObstacleLayerCache:
    """Regression tests for fixed-origin static obstacle-layer reuse."""

    def test_world_frame_reuses_static_obstacle_layer(self, monkeypatch):
        """Avoid rerasterizing unchanged static obstacles while preserving dynamic output."""
        config = GridConfig(
            resolution=0.1,
            width=6.0,
            height=6.0,
            channels=[
                GridChannel.OBSTACLES,
                GridChannel.PEDESTRIANS,
                GridChannel.COMBINED,
            ],
        )
        obstacles = [
            ((0.5, 0.5), (5.5, 0.5)),
            ((5.5, 0.5), (5.5, 5.5)),
            ((5.5, 5.5), (0.5, 5.5)),
            ((0.5, 5.5), (0.5, 0.5)),
        ]
        robot_pose = ((3.0, 3.0), 0.0)
        rasterize_calls = 0
        original_rasterize = rasterization.rasterize_obstacles

        def counting_rasterize(*args, **kwargs):
            """Count obstacle rasterization calls while preserving behavior."""
            nonlocal rasterize_calls
            rasterize_calls += 1
            return original_rasterize(*args, **kwargs)

        monkeypatch.setattr(rasterization, "rasterize_obstacles", counting_rasterize)

        grid = OccupancyGrid(config=config)
        first = grid.generate(
            obstacles=obstacles,
            pedestrians=[((2.0, 2.0), 0.25)],
            robot_pose=robot_pose,
        ).copy()
        assert rasterize_calls == 1

        second = grid.generate(
            obstacles=obstacles,
            pedestrians=[((4.0, 4.0), 0.25)],
            robot_pose=robot_pose,
        ).copy()
        assert rasterize_calls == 1

        expected = OccupancyGrid(config=config).generate(
            obstacles=obstacles,
            pedestrians=[((4.0, 4.0), 0.25)],
            robot_pose=robot_pose,
        )
        np.testing.assert_array_equal(second, expected)
        np.testing.assert_array_equal(second[0], first[0])
        assert not np.array_equal(second[1], first[1])
        np.testing.assert_array_equal(second[2], np.maximum(second[0], second[1]))

    def test_world_frame_cache_refreshes_when_obstacles_change(self, monkeypatch):
        """Obstacle input changes must refresh the cached static layer."""
        config = GridConfig(
            resolution=0.1,
            width=6.0,
            height=6.0,
            channels=[GridChannel.OBSTACLES],
        )
        obstacles = [((0.5, 0.5), (5.5, 0.5))]
        changed_obstacles = [*obstacles, ((0.5, 5.5), (5.5, 5.5))]
        robot_pose = ((3.0, 3.0), 0.0)
        rasterize_calls = 0
        original_rasterize = rasterization.rasterize_obstacles

        def counting_rasterize(*args, **kwargs):
            """Count obstacle rasterization calls while preserving behavior."""
            nonlocal rasterize_calls
            rasterize_calls += 1
            return original_rasterize(*args, **kwargs)

        monkeypatch.setattr(rasterization, "rasterize_obstacles", counting_rasterize)

        grid = OccupancyGrid(config=config)
        grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=robot_pose)
        grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=robot_pose)
        assert rasterize_calls == 1

        changed = grid.generate(
            obstacles=changed_obstacles,
            pedestrians=[],
            robot_pose=robot_pose,
        ).copy()
        assert rasterize_calls == 2

        expected = OccupancyGrid(config=config).generate(
            obstacles=changed_obstacles,
            pedestrians=[],
            robot_pose=robot_pose,
        )
        np.testing.assert_array_equal(changed, expected)

    def test_centered_world_frame_does_not_reuse_static_obstacle_layer(self, monkeypatch):
        """Moving-origin grids cannot reuse a fixed obstacle layer."""
        config = GridConfig(
            resolution=0.1,
            width=6.0,
            height=6.0,
            channels=[GridChannel.OBSTACLES],
            center_on_robot=True,
        )
        obstacles = [((0.5, 0.5), (5.5, 0.5))]
        rasterize_calls = 0
        original_rasterize = rasterization.rasterize_obstacles

        def counting_rasterize(*args, **kwargs):
            """Count obstacle rasterization calls while preserving behavior."""
            nonlocal rasterize_calls
            rasterize_calls += 1
            return original_rasterize(*args, **kwargs)

        monkeypatch.setattr(rasterization, "rasterize_obstacles", counting_rasterize)

        grid = OccupancyGrid(config=config)
        grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=((2.0, 2.0), 0.0))
        grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=((3.0, 3.0), 0.0))

        assert rasterize_calls == 2


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

    def test_point_query_clamps_out_of_bounds_coordinates_to_edge_cells(self):
        """Out-of-bounds point queries use the nearest valid grid cell."""
        config = GridConfig(
            resolution=1.0,
            width=3.0,
            height=3.0,
            channels=[GridChannel.OBSTACLES],
        )
        grid = OccupancyGrid(config=config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((0.0, 0.0), 0.0))
        grid._grid_data[0, 0, 0] = 0.25
        grid._grid_data[0, 2, 2] = 0.75

        low = grid.query(POIQuery(x=-10.0, y=-10.0))
        high = grid.query(POIQuery(x=99.0, y=99.0))

        assert low.num_cells == 1
        assert low.occupancy == pytest.approx(0.25)
        assert high.num_cells == 1
        assert high.occupancy == pytest.approx(0.75)

    def test_centered_world_frame_origin_offsets_query_coordinates(self):
        """World-frame grids centered on the robot apply the stored origin offset."""
        config = GridConfig(
            resolution=1.0,
            width=4.0,
            height=4.0,
            channels=[GridChannel.OBSTACLES],
            center_on_robot=True,
        )
        grid = OccupancyGrid(config=config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((10.0, 20.0), 0.0))
        grid._grid_data[0, 0, 0] = 0.2
        grid._grid_data[0, 3, 3] = 0.8

        metadata = grid.metadata()
        lower_left = grid.query(POIQuery(x=8.25, y=18.25))
        upper_right = grid.query(POIQuery(x=11.25, y=21.25))

        assert metadata["origin"] == pytest.approx((8.0, 18.0))
        assert lower_left.occupancy == pytest.approx(0.2)
        assert upper_right.occupancy == pytest.approx(0.8)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"resolution": np.nan}, "resolution must be finite"),
            ({"width": np.inf}, "width must be finite"),
            ({"height": np.inf}, "height must be finite"),
            ({"max_distance": np.nan}, "max_distance must be finite"),
            ({"robot_radius": np.inf}, "robot_radius must be finite"),
        ],
    )
    def test_grid_config_rejects_non_finite_geometry_values(self, kwargs, message):
        """Grid geometry must be finite so cell bounds stay well-defined."""
        with pytest.raises(ValueError, match=message):
            GridConfig(**kwargs)

    def test_generate_rejects_non_finite_robot_pose(self):
        """Robot pose validation fails before deriving grid origins from NaN values."""
        grid = OccupancyGrid(config=GridConfig())

        with pytest.raises(ValueError, match="robot_pose values must be finite"):
            grid.generate(obstacles=[], pedestrians=[], robot_pose=((np.nan, 0.0), 0.0))


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


class TestGridEdgeCases:
    """T008: Test edge-case grid dimensions and coordinate alignment."""

    def test_one_cell_grid_generation_and_query(self):
        """A 1x1 grid remains queryable from any clamped world coordinate."""
        config = GridConfig(
            resolution=1.0,
            width=1.0,
            height=1.0,
            channels=[GridChannel.OBSTACLES],
        )
        grid = OccupancyGrid(config=config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((0.0, 0.0), 0.0))
        grid._grid_data[0, 0, 0] = 1.0

        result = grid.query(POIQuery(x=100.0, y=-100.0))

        assert result.num_cells == 1
        assert result.occupancy == pytest.approx(1.0)

    def test_fractional_world_coordinates_stay_in_expected_cells(self):
        """Floating-point coordinate offsets map through floor-style cell indexing."""
        config = GridConfig(
            resolution=0.5,
            width=2.0,
            height=2.0,
            channels=[GridChannel.OBSTACLES],
        )
        grid = OccupancyGrid(config=config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((0.0, 0.0), 0.0))
        grid._grid_data[0, 1, 2] = 0.6

        result = grid.query(POIQuery(x=1.01, y=0.51))

        assert result.occupancy == pytest.approx(0.6)
