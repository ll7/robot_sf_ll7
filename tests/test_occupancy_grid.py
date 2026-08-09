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

from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger
from shapely.geometry import Polygon as _ShapelyPolygon

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


class TestPreparedGeometryCache:
    """Issue #2360: Shapely prepared obstacle geometries should be cached."""

    @staticmethod
    def _config(*, center_on_robot: bool = False) -> GridConfig:
        """Return a compact obstacle-only grid config for cache tests."""
        return GridConfig(
            resolution=0.1,
            width=6.0,
            height=6.0,
            channels=[GridChannel.OBSTACLES],
            center_on_robot=center_on_robot,
        )

    @staticmethod
    def _poly(offset: float = 0.0) -> _ShapelyPolygon:
        """Return a square obstacle polygon shifted by ``offset``."""
        return _ShapelyPolygon(
            [
                (1.0 + offset, 1.0 + offset),
                (4.0 + offset, 1.0 + offset),
                (4.0 + offset, 4.0 + offset),
                (1.0 + offset, 4.0 + offset),
            ]
        )

    @staticmethod
    def _count_prepare(monkeypatch):
        """Count calls to ``OccupancyGrid._prepare_obstacles``."""
        prepare_calls = 0
        original_prepare = OccupancyGrid._prepare_obstacles

        def counting_prepare(self_, polygons):
            nonlocal prepare_calls
            prepare_calls += 1
            return original_prepare(self_, polygons)

        monkeypatch.setattr(OccupancyGrid, "_prepare_obstacles", counting_prepare)
        return lambda: prepare_calls

    def test_world_frame_reuses_prepared_geometries(self, monkeypatch):
        """Same obstacle polygons across generate() calls must not re-prepare."""
        prepare_calls = self._count_prepare(monkeypatch)
        grid = OccupancyGrid(config=self._config())
        polygon = self._poly()
        robot_pose = ((3.0, 3.0), 0.0)

        grid.generate(
            obstacles=[], pedestrians=[], robot_pose=robot_pose, obstacle_polygons=[polygon]
        )
        grid.generate(
            obstacles=[], pedestrians=[], robot_pose=robot_pose, obstacle_polygons=[polygon]
        )

        assert prepare_calls() == 1

    def test_world_frame_refreshes_when_polygons_change(self, monkeypatch):
        """Different obstacle polygon input must re-trigger _prepare_obstacles."""
        prepare_calls = self._count_prepare(monkeypatch)
        grid = OccupancyGrid(config=self._config())
        robot_pose = ((3.0, 3.0), 0.0)
        grid.generate(
            obstacles=[], pedestrians=[], robot_pose=robot_pose, obstacle_polygons=[self._poly()]
        )
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=robot_pose,
            obstacle_polygons=[self._poly(offset=1.0)],
        )
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=robot_pose,
            obstacle_polygons=[self._poly(offset=1.0)],
        )

        assert prepare_calls() == 2

    def test_ego_frame_re_prepares_on_pose_change(self, monkeypatch):
        """Ego-frame transformed polygons depend on pose; must re-prepare."""
        prepare_calls = self._count_prepare(monkeypatch)
        grid = OccupancyGrid(config=self._config())
        polygon = self._poly()
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=((3.0, 3.0), 0.0),
            obstacle_polygons=[polygon],
            ego_frame=True,
        )
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=((4.0, 3.0), 0.0),
            obstacle_polygons=[polygon],
            ego_frame=True,
        )

        assert prepare_calls() == 2

    def test_ego_frame_reuses_when_pose_same(self, monkeypatch):
        """Same pose + same polygons in ego frame must not re-prepare."""
        prepare_calls = self._count_prepare(monkeypatch)
        grid = OccupancyGrid(config=self._config())
        polygon = self._poly()
        robot_pose = ((3.0, 3.0), 0.0)
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=robot_pose,
            obstacle_polygons=[polygon],
            ego_frame=True,
        )
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=robot_pose,
            obstacle_polygons=[polygon],
            ego_frame=True,
        )

        assert prepare_calls() == 1

    def test_center_on_robot_reuses_world_frame_prepared_geometries(self, monkeypatch):
        """Center-on-robot shifts grid origin but leaves query polygons in world frame."""
        prepare_calls = self._count_prepare(monkeypatch)
        grid = OccupancyGrid(config=self._config(center_on_robot=True))
        polygon = self._poly()
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=((3.0, 3.0), 0.0),
            obstacle_polygons=[polygon],
        )
        grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=((4.0, 4.0), 0.0),
            obstacle_polygons=[polygon],
        )

        assert prepare_calls() == 1


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


class TestGenerateStructuredLoggerPaths:
    """Cover the diagnostic logger paths in generate()'s sub-functions.

    These paths were migrated from f-string to structured Loguru style when
    generate() was decomposed (PR #6620). The validation guards and every
    rasterization channel are exercised so the migrated logger statements
    remain covered by the test suite.
    """

    def test_generate_rejects_non_list_obstacles(self):
        """Non-list obstacles log an error and raise TypeError."""
        grid = OccupancyGrid(config=GridConfig(channels=[GridChannel.OBSTACLES]))
        with pytest.raises(TypeError):
            grid.generate(
                obstacles="not-a-list",
                pedestrians=[],
                robot_pose=((0.0, 0.0), 0.0),
            )

    def test_generate_rejects_non_list_pedestrians(self):
        """Non-list pedestrians log an error and raise TypeError."""
        grid = OccupancyGrid(config=GridConfig(channels=[GridChannel.PEDESTRIANS]))
        with pytest.raises(TypeError):
            grid.generate(
                obstacles=[],
                pedestrians="not-a-list",
                robot_pose=((0.0, 0.0), 0.0),
            )

    def test_ego_frame_generate_rasterizes_every_channel(self):
        """Ego-frame generate exercises the obstacle, pedestrian, and robot channels."""
        config = GridConfig(
            resolution=0.5,
            width=4.0,
            height=4.0,
            channels=[
                GridChannel.OBSTACLES,
                GridChannel.PEDESTRIANS,
                GridChannel.ROBOT,
            ],
        )
        grid = OccupancyGrid(config=config)
        grid_data = grid.generate(
            obstacles=[((0.0, 0.0), (2.0, 2.0))],
            pedestrians=[((1.0, 1.0), 0.25)],
            robot_pose=((2.0, 2.0), 0.0),
            ego_frame=True,
        )

        assert grid_data.shape == grid.shape
        assert grid.is_initialized

    def test_generate_preserves_debug_message_format(self):
        """Generate diagnostics retain the prior f-string's rendered text."""
        config = GridConfig(
            resolution=1.0,
            width=4.0,
            height=6.0,
            channels=[GridChannel.OBSTACLES],
        )
        grid = OccupancyGrid(config=config)
        captured: list[str] = []
        handler_id = logger.add(
            lambda message: captured.append(message.record["message"]),
            level="DEBUG",
        )
        try:
            grid.generate(
                obstacles=[((0.0, 0.0), (1.0, 1.0))],
                pedestrians=[((2.0, 2.0), 0.25)],
                robot_pose=((1.25, 3.75), 0.0),
            )
        finally:
            logger.remove(handler_id)

        assert (
            "Generating grid: shape=(1, 6, 4), obstacles=1, pedestrians=1, "
            "ego_frame=False, origin=(0.00, 0.00)"
        ) in captured

    def test_array_rasterization_preserves_failure_log_messages(self, monkeypatch):
        """Array-rasterizer failure and summary diagnostics retain their prior text."""
        config = GridConfig(
            resolution=1.0,
            width=4.0,
            height=4.0,
            channels=[GridChannel.PEDESTRIANS],
        )
        grid_array = np.zeros((config.grid_height, config.grid_width), dtype=config.dtype)

        def fail_rasterization(*_args, **_kwargs):
            raise ValueError("forced failure")

        monkeypatch.setattr(rasterization, "rasterize_circle_fast", fail_rasterization)
        captured: list[str] = []
        handler_id = logger.add(
            lambda message: captured.append(message.record["message"]),
            level="DEBUG",
        )
        try:
            count = rasterization.rasterize_pedestrians_array(
                np.array([[1.0, 1.0], [2.0, 2.0]]),
                np.array([0.25, 0.5]),
                grid_array,
                config,
            )
        finally:
            logger.remove(handler_id)

        assert count == 0
        assert captured == [
            "Failed to rasterize pedestrian 0: forced failure",
            "Failed to rasterize pedestrian 1: forced failure",
            "Rasterized 0/2 pedestrians",
        ]


class TestVectorizedParity:
    """Parity tests for issue #6493 vectorization changes."""

    def test_ego_frame_vectorized_matches_scalar_world_to_ego(self):
        """Vectorized ego-frame grid matches scalar world_to_ego for obstacles and peds."""
        from robot_sf.nav.occupancy_grid_utils import world_to_ego

        config = GridConfig(
            resolution=0.2,
            width=10.0,
            height=10.0,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
        )
        obstacles = [
            ((1.0, 2.0), (3.0, 4.0)),
            ((5.0, 1.0), (7.0, 8.0)),
            ((0.5, 9.0), (9.5, 0.5)),
        ]
        pedestrians = [
            ((2.0, 3.0), 0.5),
            ((6.0, 7.0), 0.3),
            ((8.0, 1.0), 0.8),
        ]
        robot_pose = ((5.0, 5.0), np.pi / 4)

        # Generate with the vectorized path
        grid_vec = OccupancyGrid(config=config)
        grid_vec_data = grid_vec.generate(
            obstacles=obstacles,
            pedestrians=pedestrians,
            robot_pose=robot_pose,
            ego_frame=True,
        )

        # Manually compute scalar ego-frame transform and generate
        def scalar_to_ego(px, py):
            return world_to_ego(px, py, robot_pose)

        scalar_obstacles = [
            (scalar_to_ego(s[0], s[1]), scalar_to_ego(e[0], e[1])) for s, e in obstacles
        ]
        scalar_pedestrians = [(scalar_to_ego(c[0], c[1]), r) for c, r in pedestrians]

        grid_scalar = OccupancyGrid(config=config)
        grid_scalar._last_robot_pose = grid_vec._last_robot_pose
        grid_scalar._grid_origin = grid_vec._grid_origin
        grid_scalar._last_use_ego_frame = True
        grid_scalar._grid_data = np.zeros(
            (config.num_channels, config.grid_height, config.grid_width),
            dtype=config.dtype,
        )
        grid_scalar._rasterize_channels(
            scalar_obstacles,
            None,
            scalar_pedestrians,
            robot_pose,
            True,
            grid_vec._grid_origin[0],
            grid_vec._grid_origin[1],
        )

        np.testing.assert_allclose(
            grid_vec_data,
            grid_scalar._grid_data,
            atol=1e-6,
            err_msg="Vectorized ego-frame grid must match scalar world_to_ego grid",
        )

    def test_ego_frame_polygon_parity(self):
        """Vectorized ego-frame polygon transform matches scalar per-vertex transform."""
        from robot_sf.nav.occupancy_grid_utils import world_to_ego

        config = GridConfig(
            resolution=0.25,
            width=8.0,
            height=8.0,
            channels=[GridChannel.OBSTACLES],
        )
        polygon_verts = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
        robot_pose = ((4.0, 4.0), np.pi / 6)

        grid_vec = OccupancyGrid(config=config)
        grid_vec_data = grid_vec.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=robot_pose,
            ego_frame=True,
            obstacle_polygons=[polygon_verts],
        )

        # Scalar transform
        scalar_verts = [world_to_ego(x, y, robot_pose) for x, y in polygon_verts]
        grid_scalar = OccupancyGrid(config=config)
        grid_scalar._last_robot_pose = grid_vec._last_robot_pose
        grid_scalar._grid_origin = grid_vec._grid_origin
        grid_scalar._last_use_ego_frame = True
        grid_scalar._grid_data = np.zeros(
            (config.num_channels, config.grid_height, config.grid_width),
            dtype=config.dtype,
        )
        grid_scalar._rasterize_channels(
            [],
            [scalar_verts],
            [],
            robot_pose,
            True,
            grid_vec._grid_origin[0],
            grid_vec._grid_origin[1],
        )

        np.testing.assert_allclose(
            grid_vec_data,
            grid_scalar._grid_data,
            atol=1e-6,
            err_msg="Vectorized polygon ego-frame grid must match scalar transform",
        )

    def test_pedestrian_array_rasterization_matches_list_path(self):
        """rasterize_pedestrians_array produces identical grid to rasterize_pedestrians."""
        from robot_sf.nav.occupancy_grid_rasterization import (
            rasterize_pedestrians,
            rasterize_pedestrians_array,
        )

        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        pedestrians = [
            ((2.0, 3.0), 0.5),
            ((6.0, 7.0), 0.3),
            ((8.0, 1.0), 0.8),
            ((1.0, 9.0), 0.4),
        ]

        grid_list = np.zeros((config.grid_height, config.grid_width), dtype=config.dtype)
        rasterize_pedestrians(pedestrians, grid_list, config)

        positions = np.array([[c[0], c[1]] for c, r in pedestrians], dtype=float)
        radii = np.array([r for c, r in pedestrians], dtype=float)
        grid_array = np.zeros((config.grid_height, config.grid_width), dtype=config.dtype)
        rasterize_pedestrians_array(positions, radii, grid_array, config)

        np.testing.assert_array_equal(
            grid_list,
            grid_array,
            err_msg="Array pedestrian rasterization must match list-based path cell-for-cell",
        )

    def test_generate_accepts_array_pedestrians(self):
        """generate() with array pedestrians produces same grid as list pedestrians."""
        config = GridConfig(
            resolution=0.2,
            width=10.0,
            height=10.0,
            channels=[GridChannel.PEDESTRIANS],
        )
        pedestrians_list = [
            ((2.0, 3.0), 0.5),
            ((6.0, 7.0), 0.3),
            ((8.0, 1.0), 0.8),
        ]
        positions = np.array([[2.0, 3.0], [6.0, 7.0], [8.0, 1.0]])
        radii = np.array([0.5, 0.3, 0.8])
        robot_pose = ((5.0, 5.0), 0.0)

        grid_list = OccupancyGrid(config=config)
        data_list = grid_list.generate(
            obstacles=[], pedestrians=pedestrians_list, robot_pose=robot_pose
        )

        grid_arr = OccupancyGrid(config=config)
        data_arr = grid_arr.generate(
            obstacles=[], pedestrians=(positions, radii), robot_pose=robot_pose
        )

        np.testing.assert_array_equal(
            data_list,
            data_arr,
            err_msg="Array pedestrians in generate() must match list pedestrians",
        )

    def test_generate_rejects_mismatched_array_pedestrians(self):
        """Array positions and radii must describe the same pedestrian population."""
        config = GridConfig(channels=[GridChannel.PEDESTRIANS])
        grid = OccupancyGrid(config=config)

        with pytest.raises(ValueError, match="matching positions"):
            grid.generate(
                obstacles=[],
                pedestrians=(np.zeros((2, 2)), np.ones(1)),
                robot_pose=((0.0, 0.0), 0.0),
            )

    def test_get_affected_cells_returns_numpy_arrays(self):
        """get_affected_cells returns (row_indices, col_indices) NumPy arrays."""
        from robot_sf.nav.occupancy_grid_utils import get_affected_cells

        config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        rows, cols = get_affected_cells(5.0, 5.0, 0.3, config)
        assert isinstance(rows, np.ndarray)
        assert isinstance(cols, np.ndarray)
        assert rows.dtype == np.intp
        assert cols.dtype == np.intp
        assert len(rows) == len(cols)
        assert len(rows) == 36

    def test_get_affected_cells_returns_empty_for_degenerate_bounds(self):
        """Degenerate grid bounds should produce empty integer arrays."""
        from robot_sf.nav.occupancy_grid_utils import get_affected_cells

        config = SimpleNamespace(
            resolution=0.1,
            width=1.0,
            height=1.0,
            grid_width=10,
            grid_height=0,
        )
        rows, cols = get_affected_cells(0.5, 0.5, 0.3, config)

        assert rows.size == 0
        assert cols.size == 0
        assert rows.dtype == np.intp
        assert cols.dtype == np.intp

    def test_free_space_sampling_vectorized_rejects_obstacles(self):
        """Vectorized free-space sampling still rejects obstacle-intersecting points."""
        from shapely.geometry import Point, Polygon

        from robot_sf.nav.free_space_sampling import sample_free_points_in_bounds

        obstacle = Polygon([(2.0, 2.0), (2.0, 8.0), (8.0, 8.0), (8.0, 2.0)])
        rng = np.random.default_rng(42)
        result = sample_free_points_in_bounds(
            (0.0, 10.0, 0.0, 10.0), 20, obstacle_polygons=[obstacle], rng=rng
        )
        assert len(result) == 20
        for x, y in result:
            assert not obstacle.contains(Point(x, y))
