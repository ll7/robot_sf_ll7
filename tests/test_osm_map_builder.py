"""Tests for OSM PBF importer and MapDefinition conversion.

Tests cover:
- PBF loading and parsing
- Tag filtering for driveable ways and obstacles
- UTM projection and buffering
- Polygon cleanup and validity
- End-to-end MapDefinition creation
- Backward compatibility with existing maps
"""

from pathlib import Path

import pytest

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.osm_map_builder import (
    OSMTagFilters,
    buffer_ways,
    cleanup_polygons,
    compute_obstacles,
    extract_obstacles,
    filter_driveable_ways,
    load_pbf,
    osm_to_map_definition,
    project_to_utm,
)


# Fixtures
@pytest.fixture
def pbf_fixture() -> str:
    """Path to test PBF file."""
    return "test_scenarios/osm_fixtures/sample_block.pbf"


@pytest.fixture
def tag_filters() -> OSMTagFilters:
    """Standard tag filter configuration."""
    return OSMTagFilters()


# PBF Loading Tests
class TestPBFLoading:
    """Test OSM PBF file loading and parsing."""

    def test_load_pbf_fixture_exists(self, pbf_fixture: str) -> None:
        """Verify test fixture is available."""
        assert Path(pbf_fixture).exists(), f"PBF fixture not found: {pbf_fixture}"

    def test_load_pbf_returns_geodataframe(self, pbf_fixture: str) -> None:
        """Test loading PBF returns valid GeoDataFrame."""
        gdf = load_pbf(pbf_fixture)
        assert gdf is not None
        assert len(gdf) > 0, "PBF loaded but empty"
        assert "geometry" in gdf.columns

    def test_load_pbf_file_not_found(self) -> None:
        """Test error handling for missing PBF file."""
        with pytest.raises(FileNotFoundError):
            load_pbf("nonexistent.pbf")


# Tag Filtering Tests
class TestTagFiltering:
    """Test semantic OSM tag filtering."""

    def test_filter_driveable_ways_returns_geometries(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test filtering driveable ways."""
        gdf = load_pbf(pbf_fixture)
        filtered = filter_driveable_ways(gdf, tag_filters)
        assert filtered is not None
        assert isinstance(filtered, type(gdf))

    def test_extract_obstacles_returns_geometries(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test extracting obstacles."""
        gdf = load_pbf(pbf_fixture)
        obstacles = extract_obstacles(gdf, tag_filters)
        assert obstacles is not None
        assert isinstance(obstacles, type(gdf))


# Projection Tests
class TestProjection:
    """Test UTM projection and coordinate transforms."""

    def test_project_to_utm_returns_tuple(self, pbf_fixture: str) -> None:
        """Test UTM projection returns (GeoDataFrame, zone_number)."""
        gdf = load_pbf(pbf_fixture)
        gdf_utm, zone = project_to_utm(gdf)
        assert gdf_utm is not None
        assert isinstance(zone, int)
        assert 1 <= zone <= 60
        # Round-trip projection should be nearly lossless in meters.
        round_trip = gdf_utm.to_crs(gdf.crs).to_crs(gdf_utm.crs)
        if not gdf_utm.empty:
            orig_centroid = gdf_utm.geometry.iloc[0].centroid
            round_centroid = round_trip.geometry.iloc[0].centroid
            assert orig_centroid.distance(round_centroid) < 0.1


# Geometry Processing Tests
class TestGeometryProcessing:
    """Test geometry buffering and cleanup."""

    def test_buffer_ways_returns_polygons(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test buffering line ways to polygons."""
        gdf = load_pbf(pbf_fixture)
        driveable = filter_driveable_ways(gdf, tag_filters)
        driveable_utm, _ = project_to_utm(driveable)

        buffered = buffer_ways(driveable_utm, half_width_m=1.5)
        assert len(buffered) > 0
        for poly in buffered:
            assert poly.is_valid
            assert not poly.is_empty

    def test_cleanup_polygons_returns_valid(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test polygon cleanup produces valid geometries."""
        gdf = load_pbf(pbf_fixture)
        driveable = filter_driveable_ways(gdf, tag_filters)
        driveable_utm, _ = project_to_utm(driveable)

        buffered = buffer_ways(driveable_utm, half_width_m=1.5)
        cleaned = cleanup_polygons(buffered)

        assert len(cleaned) > 0
        for poly in cleaned:
            assert poly.is_valid
            assert poly.area > 0.1


# Obstacle Derivation Tests
class TestObstacleDerivation:
    """Test obstacle computation via complement."""

    def test_compute_obstacles_returns_polygons(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test computing obstacles as spatial complement."""
        gdf = load_pbf(pbf_fixture)
        driveable = filter_driveable_ways(gdf, tag_filters)
        driveable_utm, _ = project_to_utm(driveable)

        buffered = buffer_ways(driveable_utm, half_width_m=1.5)
        cleaned = cleanup_polygons(buffered)

        from shapely.ops import unary_union

        walkable_union = unary_union(cleaned)

        bounds = driveable_utm.total_bounds
        obstacles = compute_obstacles(bounds, walkable_union)

        assert len(obstacles) > 0
        for obs in obstacles:
            assert obs.is_valid


# End-to-End Importer Tests
class TestEndToEndImporter:
    """Test full OSM to MapDefinition pipeline."""

    def test_osm_to_map_definition_creates_valid_map(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test end-to-end conversion produces valid MapDefinition."""
        map_def = osm_to_map_definition(pbf_fixture, tag_filters=tag_filters)

        assert isinstance(map_def, MapDefinition)
        assert map_def.width > 0
        assert map_def.height > 0
        assert len(map_def.obstacles) > 0

    def test_osm_to_map_definition_includes_allowed_areas(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test MapDefinition includes allowed_areas field."""
        map_def = osm_to_map_definition(pbf_fixture, tag_filters=tag_filters)

        assert hasattr(map_def, "allowed_areas")
        assert map_def.allowed_areas is not None
        assert len(map_def.allowed_areas) > 0

    def test_osm_to_map_definition_has_valid_bounds(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test MapDefinition has valid bounds as map edge line segments."""
        map_def = osm_to_map_definition(pbf_fixture, tag_filters=tag_filters)

        assert map_def.bounds is not None
        assert len(map_def.bounds) == 4  # 4 sides: bottom, right, top, left
        for line in map_def.bounds:
            assert len(line) == 4  # Each line is (x_start, x_end, y_start, y_end)
            assert line[0] <= line[1]
            assert line[2] <= line[3]


# Backward Compatibility Tests
class TestBackwardCompat:
    """Test backward compatibility with existing workflows."""

    def test_map_definition_allowed_areas_optional(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test MapDefinition works with allowed_areas=None (legacy)."""
        # Create a legacy MapDefinition (without allowed_areas)
        legacy_map = MapDefinition(
            width=100.0,
            height=100.0,
            obstacles=[],
            robot_spawn_zones=[],
            ped_spawn_zones=[],
            robot_goal_zones=[],
            ped_goal_zones=[],
            robot_routes=[],
            ped_crowded_zones=[],
            ped_routes=[],
            bounds=[
                ((0, 0), (100, 0)),
                ((100, 0), (100, 100)),
                ((100, 100), (0, 100)),
                ((0, 100), (0, 0)),
            ],
            allowed_areas=None,  # Explicitly None
        )

        assert legacy_map.allowed_areas is None
        assert hasattr(legacy_map, "allowed_areas")

    def test_osm_derived_map_is_valid_mapdefinition(
        self, pbf_fixture: str, tag_filters: OSMTagFilters
    ) -> None:
        """Test OSM-derived MapDefinition is compatible with existing code."""
        map_def = osm_to_map_definition(pbf_fixture, tag_filters=tag_filters)

        # Should have all required MapDefinition fields
        assert hasattr(map_def, "width")
        assert hasattr(map_def, "height")
        assert hasattr(map_def, "obstacles")
        assert hasattr(map_def, "bounds")
        assert hasattr(map_def, "allowed_areas")

        # Bounds should be list of Line2D (4 sides)
        assert len(map_def.bounds) == 4

        # Obstacles should be list of Obstacle objects
        assert len(map_def.obstacles) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
