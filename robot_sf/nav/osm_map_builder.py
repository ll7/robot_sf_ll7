"""OSM PBF to MapDefinition importer.

This module provides functionality to convert OpenStreetMap data from PBF files
into robot_sf MapDefinition objects. It handles:
- Semantic OSM tag filtering (driveable ways, obstacles)
- Coordinate projection to local UTM zones
- Geometry processing (buffering, cleanup, validation)
- Obstacle derivation via spatial complement

Key capabilities:
- Load OSM PBF files using OSMnx/GeoPandas
- Filter ways by tag semantics (not color-based SVG)
- Project to meter-based world coordinates
- Compute driveable and obstacle polygons
- Backward-compatible with existing MapDefinition

Design decisions:
- Uses OSMnx for MVP (PyOsmium as performance fallback)
- Meters-based UTM projection for reproducibility
- Optional allowed_areas field for explicit driveable zones
- Deterministic output (same PBF → same MapDefinition)
"""

import logging
from dataclasses import dataclass, field

import geopandas as gpd
from shapely.geometry import Polygon

from robot_sf.nav.map_config import MapDefinition

logger = logging.getLogger(__name__)


@dataclass
class OSMTagFilters:
    """Configuration for semantic OSM tag filtering.

    This dataclass defines which OSM tags correspond to driveable areas
    (pedestrian ways) and obstacles. Used to distinguish semantic features
    from color-based SVG parsing.

    Attributes:
        driveable_highways: List of OSM highway tags for pedestrian routes
        driveable_areas: List of OSM area tags (landuse, natural) for pedestrian zones
        obstacle_tags: List of (key, value) tuples for obstacles (buildings, water, etc.)
        excluded_tags: List of (key, value) tuples to exclude (steps, motorway, private access)
    """

    driveable_highways: list[str] = field(
        default_factory=lambda: [
            "footway",
            "path",
            "cycleway",
            "bridleway",
            "pedestrian",
            "track",
            "service",
        ]
    )
    """OSM highway tags that represent driveable/walkable ways."""

    driveable_areas: list[str] = field(
        default_factory=lambda: [
            "pedestrian",
            "footway",
            "residential",
            "service",
            "living_street",
        ]
    )
    """OSM landuse/natural tags representing driveable open areas."""

    obstacle_tags: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("building", "*"),
            ("natural", "water"),
            ("natural", "cliff"),
            ("natural", "tree"),
            ("waterway", "*"),
        ]
    )
    """OSM tags representing obstacles (buildings, water, cliffs, trees)."""

    excluded_tags: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("highway", "steps"),
            ("highway", "motorway"),
            ("highway", "motorway_link"),
            ("highway", "trunk"),
            ("highway", "trunk_link"),
            ("highway", "primary"),
            ("highway", "primary_link"),
            ("access", "private"),
            ("access", "no"),
        ]
    )
    """OSM tags to exclude from driveable areas (motorways, private access)."""


# Stub functions - to be implemented in Phase 1


def load_pbf(pbf_file: str, bbox: tuple | None = None) -> gpd.GeoDataFrame:
    """Load OSM PBF file and return GeoDataFrame.

    Args:
        pbf_file: Path to OSM PBF file
        bbox: Optional bounding box (minx, miny, maxx, maxy)

    Returns:
        GeoDataFrame with OSM features (ways and areas)

    Raises:
        FileNotFoundError: If pbf_file does not exist
        ValueError: If PBF is invalid or empty
    """
    # TODO: T006 implementation
    pass


def filter_driveable_ways(
    gdf: gpd.GeoDataFrame,
    tag_filters: OSMTagFilters = None,
) -> gpd.GeoDataFrame:
    """Filter GeoDataFrame for driveable highways.

    Args:
        gdf: Input GeoDataFrame with OSM features
        tag_filters: OSMTagFilters config (defaults to standard)

    Returns:
        Filtered GeoDataFrame containing only driveable ways
    """
    # TODO: T007 implementation
    pass


def extract_obstacles(
    gdf: gpd.GeoDataFrame,
    tag_filters: OSMTagFilters = None,
) -> gpd.GeoDataFrame:
    """Extract obstacle features (buildings, water, cliffs).

    Args:
        gdf: Input GeoDataFrame with OSM features
        tag_filters: OSMTagFilters config

    Returns:
        GeoDataFrame containing only obstacle features
    """
    # TODO: T008 implementation
    pass


def project_to_utm(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Project GeoDataFrame to local UTM zone.

    Auto-detects the appropriate UTM zone based on the data centroid
    and projects all geometries to meter-based coordinates.

    Args:
        gdf: Input GeoDataFrame (assumed in WGS84/EPSG:4326)

    Returns:
        Tuple of (projected_gdf, utm_zone_number)

    Raises:
        ValueError: If centroid cannot be determined
    """
    # TODO: T009 implementation
    pass


def buffer_ways(
    gdf: gpd.GeoDataFrame,
    half_width_m: float = 1.5,
) -> list[Polygon]:
    """Buffer line ways to polygons.

    Converts LineString geometries to Polygon buffers with specified
    width, using round cap and join styles for smooth results.

    Args:
        gdf: GeoDataFrame with (presumed) LineString geometries
        half_width_m: Half-width of buffer in meters

    Returns:
        List of buffered Polygon objects
    """
    # TODO: T010 implementation
    pass


def cleanup_polygons(polys: list[Polygon]) -> list[Polygon]:
    """Clean up polygons: repair self-intersections, simplify, validate.

    Args:
        polys: List of input polygons (may be invalid)

    Returns:
        List of cleaned, valid polygons
    """
    # TODO: T011 implementation
    pass


def compute_obstacles(
    bounds_box: tuple,
    walkable_union: Polygon,
) -> list[Polygon]:
    """Compute obstacles as spatial complement.

    Computes obstacles as the difference between the bounding box
    and the union of walkable areas. This ensures no gaps or overlaps.

    Args:
        bounds_box: (minx, miny, maxx, maxy) in meters
        walkable_union: Polygon representing all walkable areas

    Returns:
        List of obstacle Polygons
    """
    # TODO: T012 implementation
    pass


def osm_to_map_definition(
    pbf_file: str,
    bbox: tuple | None = None,
    line_buffer_m: float = 1.5,
    tag_filters: OSMTagFilters = None,
) -> MapDefinition:
    """Convert OSM PBF to MapDefinition.

    End-to-end conversion pipeline:
    1. Load PBF file
    2. Filter driveable ways and obstacles
    3. Project to local UTM
    4. Buffer ways and cleanup geometry
    5. Compute obstacles via complement
    6. Create MapDefinition with bounds, obstacles, allowed_areas

    Args:
        pbf_file: Path to OSM PBF file
        bbox: Optional bounding box filter
        line_buffer_m: Width of way buffers (default 1.5m)
        tag_filters: OSMTagFilters config (defaults to standard)

    Returns:
        MapDefinition object with bounds, obstacles, and allowed_areas

    Raises:
        FileNotFoundError: If PBF file does not exist
        ValueError: If PBF is empty or contains no driveable features
    """
    # TODO: T013 implementation
    pass
