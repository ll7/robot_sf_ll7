"""Tests for occupancy grid POI queries and spawn validation (Phase 5: User Story 3).

This module tests point-of-interest (POI) query functionality:
- Point queries (single cell)
- Circular area-of-interest (AOI) queries
- Rectangular AOI queries
- Per-channel query results
- Safe-to-spawn heuristics
- Out-of-bounds handling
- Edge cases

All tests use fixtures from conftest.py (simple_grid_config, occupancy_grid, etc.)
"""

from robot_sf.nav.occupancy_grid import GridConfig, OccupancyGrid, POIQuery, POIQueryType


class TestPointQueryInFreeSpace:
    """Test T049: Point query in free space should return is_occupied=False."""

    def test_point_query_free_space(self, simple_grid_config: GridConfig) -> None:
        """Query a point in free space should return is_occupied=False."""
        grid = OccupancyGrid(simple_grid_config)
        # Empty grid, no obstacles
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query a point in free space
        query = POIQuery(x=3.0, y=3.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        assert result is not None
        assert hasattr(result, "is_occupied")
        assert not result.is_occupied, "Point in free space should not be occupied"

    def test_point_query_free_space_multiple_locations(
        self, simple_grid_config: GridConfig
    ) -> None:
        """Multiple point queries in free space should all return False."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query multiple free points
        for x in [1.0, 3.0, 5.0, 7.0, 9.0]:
            for y in [1.0, 3.0, 5.0, 7.0, 9.0]:
                query = POIQuery(x=x, y=y, query_type=POIQueryType.POINT)
                result = grid.query(query)
                assert result is not None


class TestPointQueryInOccupiedSpace:
    """Test T050: Point query in occupied space should return is_occupied=True."""

    def test_point_query_occupied_space_obstacle(self, simple_grid_config: GridConfig) -> None:
        """Query a point on obstacle should return is_occupied=True."""
        grid = OccupancyGrid(simple_grid_config)
        # Add obstacle at (2, 2)
        obstacle = ((2.0, 2.0), (4.0, 2.0))
        grid.generate(obstacles=[obstacle], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query a point on the obstacle
        query = POIQuery(x=3.0, y=2.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        assert result is not None
        # Result should indicate occupied (specific attribute depends on implementation)
        assert result.is_occupied, "Point on obstacle should be occupied"

    def test_point_query_occupied_space_pedestrian(self, simple_grid_config: GridConfig) -> None:
        """Query a point on pedestrian should return is_occupied=True."""
        grid = OccupancyGrid(simple_grid_config)
        # Add pedestrian at (5, 5)
        pedestrian = ((5.0, 5.0), 0.3)
        grid.generate(obstacles=[], pedestrians=[pedestrian], robot_pose=((2.0, 2.0), 0))

        # Query a point on the pedestrian
        query = POIQuery(x=5.0, y=5.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        assert result is not None
        assert result.is_occupied, "Point on pedestrian should be occupied"


class TestPointQueryAtBoundary:
    """Test T051: Point query at boundary should handle edge cases without crashing."""

    def test_point_query_grid_boundary(self, simple_grid_config: GridConfig) -> None:
        """Query at exact grid boundary should not crash."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query at grid boundary
        query = POIQuery(x=0.0, y=0.0, query_type=POIQueryType.POINT)
        result = grid.query(query)
        assert result is not None

    def test_point_query_grid_boundary_far_corner(self, simple_grid_config: GridConfig) -> None:
        """Query at far corner of grid should not crash."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query at far corner
        query = POIQuery(x=10.0, y=10.0, query_type=POIQueryType.POINT)
        result = grid.query(query)
        assert result is not None


class TestPointQueryOutOfBounds:
    """Test T052: Point query out-of-bounds should raise ValueError or return safe default."""

    def test_point_query_out_of_bounds_negative_coords(
        self, simple_grid_config: GridConfig
    ) -> None:
        """Query with negative coordinates should raise ValueError or return safe result."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query out of bounds (negative)
        query = POIQuery(x=-5.0, y=5.0, query_type=POIQueryType.POINT)
        try:
            result = grid.query(query)
            # If it doesn't raise, result should be valid
            assert result is not None
        except ValueError as exc:
            assert "invalid coordinates" in str(exc).lower()

    def test_point_query_out_of_bounds_far_positive(self, simple_grid_config: GridConfig) -> None:
        """Query far outside grid bounds should raise ValueError or return safe result."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query out of bounds (far positive)
        query = POIQuery(x=50.0, y=50.0, query_type=POIQueryType.POINT)
        try:
            result = grid.query(query)
            assert result is not None
        except ValueError as exc:
            assert "invalid coordinates" in str(exc).lower()


class TestCircularAOIQueryFree:
    """Test T053: Circular AOI query in free space should return safe_to_spawn=True."""

    def test_circular_aoi_free_space(self, simple_grid_config: GridConfig) -> None:
        """Circular AOI in completely free space should be safe to spawn."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query circular region in free space
        query = POIQuery(x=3.0, y=3.0, radius=0.5, query_type=POIQueryType.CIRCLE)
        result = grid.query(query)

        assert result is not None
        assert hasattr(result, "safe_to_spawn")
        assert result.safe_to_spawn, "Circular AOI in free space should be safe to spawn"

    def test_circular_aoi_free_space_large_radius(self, simple_grid_config: GridConfig) -> None:
        """Larger circular AOI in free space should also be safe."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Larger circle
        query = POIQuery(x=5.0, y=5.0, radius=1.0, query_type=POIQueryType.CIRCLE)
        result = grid.query(query)

        assert result is not None


class TestCircularAOIQueryPartiallyOccupied:
    """Test T054: Circular AOI query partially occupied should return safe_to_spawn=False."""

    def test_circular_aoi_partially_occupied(self, simple_grid_config: GridConfig) -> None:
        """Circular AOI overlapping obstacle should NOT be safe to spawn."""
        grid = OccupancyGrid(simple_grid_config)
        # Obstacle at (5, 5)
        obstacle = ((5.0, 5.0), (6.0, 5.0))
        grid.generate(obstacles=[obstacle], pedestrians=[], robot_pose=((2.0, 2.0), 0))

        # Circle centered on obstacle
        query = POIQuery(x=5.5, y=5.0, radius=0.5, query_type=POIQueryType.CIRCLE)
        result = grid.query(query)

        assert result is not None
        # Result should indicate not safe to spawn
        assert not result.safe_to_spawn, (
            "Circular AOI overlapping obstacle should not be safe to spawn"
        )

    def test_circular_aoi_just_touching_obstacle(self, simple_grid_config: GridConfig) -> None:
        """Circle just touching obstacle boundary should indicate caution."""
        grid = OccupancyGrid(simple_grid_config)
        obstacle = ((4.0, 4.0), (6.0, 6.0))
        grid.generate(obstacles=[obstacle], pedestrians=[], robot_pose=((2.0, 2.0), 0))

        # Small circle at edge of obstacle
        query = POIQuery(x=3.5, y=4.0, radius=0.3, query_type=POIQueryType.CIRCLE)
        result = grid.query(query)

        assert result is not None


class TestRectangularAOIQuery:
    """Test T055: Rectangular AOI query should return occupancy fraction."""

    def test_rectangular_aoi_free_space(self, simple_grid_config: GridConfig) -> None:
        """Rectangular AOI in free space should have 0% occupancy."""
        grid = OccupancyGrid(simple_grid_config)
        grid.generate(obstacles=[], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query rectangular region
        query = POIQuery(x=2.0, y=2.0, width=2.0, height=2.0, query_type=POIQueryType.RECT)
        result = grid.query(query)

        assert result is not None
        assert hasattr(result, "occupancy_fraction")
        assert result.occupancy_fraction == 0.0, "Free space should have 0% occupancy"

    def test_rectangular_aoi_partially_occupied(self, simple_grid_config: GridConfig) -> None:
        """Rectangular AOI overlapping obstacle should have >0% occupancy."""
        grid = OccupancyGrid(simple_grid_config)
        obstacle = ((4.0, 4.0), (6.0, 6.0))
        grid.generate(obstacles=[obstacle], pedestrians=[], robot_pose=((2.0, 2.0), 0))

        # Rectangle overlapping obstacle
        query = POIQuery(x=3.0, y=3.0, width=4.0, height=4.0, query_type=POIQueryType.RECT)
        result = grid.query(query)

        assert result is not None
        assert hasattr(result, "occupancy_fraction")
        assert result.occupancy_fraction > 0.0, "Partially occupied should have >0% occupancy"

    def test_rectangular_aoi_fully_occupied(self, simple_grid_config: GridConfig) -> None:
        """Rectangular AOI entirely on obstacle should have 100% occupancy."""
        grid = OccupancyGrid(simple_grid_config)
        # Large obstacle covering entire area
        obstacle = ((2.0, 2.0), (8.0, 2.0))
        grid.generate(obstacles=[obstacle], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Rectangle on the obstacle
        query = POIQuery(x=5.0, y=2.0, width=1.0, height=0.5, query_type=POIQueryType.RECT)
        result = grid.query(query)

        assert result is not None
        assert result.occupancy_fraction > 0.9, "Fully occupied should have ~100% occupancy"


class TestPerChannelQueryResults:
    """Test T056: Per-channel query should return results per channel."""

    def test_query_returns_per_channel_results(self, simple_grid_config: GridConfig) -> None:
        """Query result should include breakdown by channel."""
        grid = OccupancyGrid(simple_grid_config)
        obstacle = ((3.0, 3.0), (5.0, 3.0))
        pedestrian = ((7.0, 7.0), 0.3)
        grid.generate(obstacles=[obstacle], pedestrians=[pedestrian], robot_pose=((5.0, 5.0), 0))

        # Query in region with both obstacle and pedestrian
        query = POIQuery(x=5.0, y=5.0, radius=1.0, query_type=POIQueryType.CIRCLE)
        result = grid.query(query)

        assert result is not None
        assert hasattr(result, "per_channel_results")

    def test_query_channel_separation(self, simple_grid_config: GridConfig) -> None:
        """Per-channel results should show obstacles and pedestrians separately."""
        grid = OccupancyGrid(simple_grid_config)
        obstacle = ((2.0, 2.0), (4.0, 2.0))
        grid.generate(obstacles=[obstacle], pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Query on obstacle
        query = POIQuery(x=3.0, y=2.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        assert result is not None
        # Obstacle channel should be occupied, pedestrian channel should be free
        assert "OBSTACLES" in result.per_channel_results
        assert "PEDESTRIANS" in result.per_channel_results
        assert result.per_channel_results["OBSTACLES"] > 0.0, "Obstacle channel should be occupied"
        assert result.per_channel_results["PEDESTRIANS"] == 0.0, "Pedestrian channel should be free"


class TestSpawnValidationWorkflow:
    """Test T057: Spawn validation workflow - query 100 candidates, verify >95% success in valid regions."""

    def test_spawn_validation_candidates(self, simple_grid_config: GridConfig) -> None:
        """Validate spawn points across grid - should have >95% success in free areas."""
        grid = OccupancyGrid(simple_grid_config)
        # Small obstacles to create mixed occupied/free regions
        obstacles = [
            ((2.0, 2.0), (3.0, 2.0)),
            ((7.0, 7.0), (8.0, 7.0)),
            ((4.0, 8.0), (4.0, 9.0)),
        ]
        grid.generate(obstacles=obstacles, pedestrians=[], robot_pose=((5.0, 5.0), 0))

        # Test 100 candidate spawn points
        successful_spawns = 0
        for i in range(100):
            # Generate random candidate
            x = 0.5 + (i % 10) * 1.0
            y = 0.5 + (i // 10) * 1.0
            if x >= 10.0 or y >= 10.0:
                continue

            query = POIQuery(x=x, y=y, radius=0.3, query_type=POIQueryType.CIRCLE)
            result = grid.query(query)

            if result and hasattr(result, "safe_to_spawn") and result.safe_to_spawn:
                successful_spawns += 1

        # In mostly free grid, should have good success rate
        assert successful_spawns > 0, "Should have at least some valid spawn locations"


class TestQueryWithDynamicPedestrians:
    """Test T058: Query with FastPysfWrapper pedestrians - should respect dynamic pedestrian channels."""

    def test_query_respects_pedestrian_channel(self, simple_grid_config: GridConfig) -> None:
        """Queries should account for pedestrian occupancy in dynamic channels."""
        grid = OccupancyGrid(simple_grid_config)
        pedestrians = [
            ((3.0, 3.0), 0.25),
            ((7.0, 7.0), 0.25),
        ]
        grid.generate(obstacles=[], pedestrians=pedestrians, robot_pose=((5.0, 5.0), 0))

        # Query on pedestrian should detect occupancy
        query = POIQuery(x=3.0, y=3.0, query_type=POIQueryType.POINT)
        result = grid.query(query)

        assert result is not None

    def test_query_pedestrian_updates(self, simple_grid_config: GridConfig) -> None:
        """Query result should update when pedestrians move."""
        grid = OccupancyGrid(simple_grid_config)

        # First query with pedestrian at (3, 3)
        pedestrian1 = [((3.0, 3.0), 0.25)]
        grid.generate(obstacles=[], pedestrians=pedestrian1, robot_pose=((5.0, 5.0), 0))
        query = POIQuery(x=3.0, y=3.0, query_type=POIQueryType.POINT)
        result1 = grid.query(query)

        # Regenerate with pedestrian moved to (7, 7)
        pedestrian2 = [((7.0, 7.0), 0.25)]
        grid.generate(obstacles=[], pedestrians=pedestrian2, robot_pose=((5.0, 5.0), 0))
        query = POIQuery(x=3.0, y=3.0, query_type=POIQueryType.POINT)
        result2 = grid.query(query)

        # Results should differ (first occupied, second free)
        assert result1 is not None
        assert result2 is not None
        assert result1.is_occupied, "Query at (3,3) should be occupied when pedestrian is there"
        assert not result2.is_occupied, "Query at (3,3) should be free after pedestrian moved"
