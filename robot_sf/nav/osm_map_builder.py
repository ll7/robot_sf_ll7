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

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.affinity import translate
from shapely.errors import TopologicalError
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import triangulate, unary_union

from robot_sf.nav.map_config import MapDefinition, Obstacle


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


# Core importer functions


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
    pbf_path = Path(pbf_file)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_file}")

    try:
        # Load all relevant layers from PBF using GeoPandas
        # PBF files have multiple layers: lines, multipolygons, multilinestrings
        layers_to_load = ["lines", "multipolygons", "multilinestrings"]
        gdfs = []

        for layer in layers_to_load:
            try:
                gdf_layer = gpd.read_file(pbf_file, layer=layer, bbox=bbox)
                if not gdf_layer.empty:
                    gdfs.append(gdf_layer)
                    logger.info(f"Loaded {len(gdf_layer)} features from layer '{layer}'")
            except (OSError, ValueError) as e:
                logger.debug(f"Could not load layer '{layer}': {e}")
                continue

        if not gdfs:
            raise ValueError(f"PBF file contains no usable features: {pbf_file}")

        # Combine all layers, ensuring uniform structure
        gdf = pd.concat(gdfs, ignore_index=False).reset_index(drop=True)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        logger.info(f"Loaded {len(gdf)} total features from {pbf_file}")
        return gdf

    except (OSError, ValueError) as e:
        raise ValueError(f"Failed to load PBF file {pbf_file}: {e}") from e


def filter_driveable_ways(
    gdf: gpd.GeoDataFrame,
    tag_filters: OSMTagFilters | None = None,
) -> gpd.GeoDataFrame:
    """Filter GeoDataFrame for driveable highways.

    Args:
        gdf: Input GeoDataFrame with OSM features
        tag_filters: OSMTagFilters config (defaults to standard)

    Returns:
        Filtered GeoDataFrame containing only driveable ways
    """
    if tag_filters is None:
        tag_filters = OSMTagFilters()

    # Filter for highway features
    if "highway" not in gdf.index.names:
        # Check if highway is in columns
        if "highway" in gdf.columns:
            gdf = gdf[gdf["highway"].notna()]
        else:
            return gdf.iloc[0:0]  # Return empty GeoDataFrame
    else:
        gdf = gdf.loc[gdf.index.get_level_values("highway").notna()]

    # Check against tag filters
    mask = False
    for highway_tag in tag_filters.driveable_highways:
        if "highway" in gdf.columns:
            mask = mask | (gdf["highway"] == highway_tag)
        elif "highway" in gdf.index.names:
            mask = mask | (gdf.index.get_level_values("highway") == highway_tag)

    filtered = gdf[mask] if isinstance(mask, pd.Series) else gdf.iloc[0:0]
    logger.info(f"Filtered to {len(filtered)} driveable ways")
    return filtered


def extract_obstacles(
    gdf: gpd.GeoDataFrame,
    tag_filters: OSMTagFilters | None = None,
) -> gpd.GeoDataFrame:
    """Extract obstacle features (buildings, water, cliffs).

    Args:
        gdf: Input GeoDataFrame with OSM features
        tag_filters: OSMTagFilters config

    Returns:
        GeoDataFrame containing only obstacle features
    """
    if tag_filters is None:
        tag_filters = OSMTagFilters()

    obstacles_list = []

    for key, value in tag_filters.obstacle_tags:
        if key in gdf.columns or key in gdf.index.names:
            if key in gdf.columns:
                if value == "*":
                    mask = gdf[key].notna()
                else:
                    mask = gdf[key] == value
                obstacles_list.append(gdf[mask])
            else:
                # Check in index
                level_values = gdf.index.get_level_values(key) if key in gdf.index.names else None
                if level_values is not None:
                    if value == "*":
                        mask = level_values.notna()
                    else:
                        mask = level_values == value
                    obstacles_list.append(gdf[mask])

    if obstacles_list:
        result = gpd.GeoDataFrame(pd.concat(obstacles_list).drop_duplicates(subset=["geometry"]))
        logger.info(f"Extracted {len(result)} obstacle features")
        return result
    else:
        logger.warning("No obstacles found in OSM data")
        return gdf.iloc[0:0]  # Return empty GeoDataFrame


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
    if gdf.empty:
        raise ValueError("Cannot project empty GeoDataFrame")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # Get bounds and compute centroid
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    centroid_x = (bounds[0] + bounds[2]) / 2
    centroid_y = (bounds[1] + bounds[3]) / 2

    # Calculate UTM zone from longitude
    utm_zone = int((centroid_x + 180) / 6) + 1
    if not 1 <= utm_zone <= 60:
        logger.warning(f"Computed UTM zone {utm_zone} outside valid range [1, 60]; clamping")
        utm_zone = max(1, min(utm_zone, 60))

    hemisphere = "N" if centroid_y >= 0 else "S"
    utm_crs_prefix = "326" if hemisphere == "N" else "327"
    utm_crs = f"EPSG:{utm_crs_prefix}{utm_zone:02d}"

    # Project
    gdf_utm = gdf.to_crs(utm_crs)
    logger.info(f"Projected to UTM zone {utm_zone}{hemisphere} ({utm_crs})")

    return gdf_utm, utm_zone


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
    buffered = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        try:
            if isinstance(geom, LineString):
                buffered_geom = geom.buffer(
                    half_width_m,
                    cap_style="round",
                    join_style="round",
                )
                if isinstance(buffered_geom, Polygon):
                    buffered.append(buffered_geom)
                elif isinstance(buffered_geom, MultiPolygon):
                    buffered.extend(buffered_geom.geoms)
            elif isinstance(geom, Polygon):
                # Already a polygon, keep as-is but validate
                if geom.is_valid and not geom.is_empty:
                    buffered.append(geom)
        except (ValueError, TopologicalError) as e:
            logger.warning(f"Failed to buffer geometry: {e}")
            continue

    logger.info(f"Buffered {len(buffered)} ways")
    return buffered


def cleanup_polygons(polys: list[Polygon]) -> list[Polygon]:
    """Clean up polygons: repair self-intersections, simplify, validate.

    Args:
        polys: List of input polygons (may be invalid)

    Returns:
        List of cleaned, valid polygons
    """
    cleaned = []

    for poly in polys:
        if poly is None or poly.is_empty:
            continue

        try:
            # Repair self-intersections with buffer(0)
            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.is_empty:
                continue

            # Simplify with 0.1m tolerance
            poly = poly.simplify(0.1, preserve_topology=True)

            # Skip very small artifacts
            if poly.area > 0.1:
                if isinstance(poly, Polygon):
                    cleaned.append(poly)
                elif isinstance(poly, MultiPolygon):
                    cleaned.extend([p for p in poly.geoms if p.area > 0.1])
        except (ValueError, TopologicalError) as e:
            logger.warning(f"Failed to cleanup polygon: {e}")
            continue

    logger.info(f"Cleaned {len(cleaned)} polygons")
    return cleaned


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
    bounds_poly = box(*bounds_box)

    try:
        obstacles_union = bounds_poly.difference(walkable_union)
    except (ValueError, TopologicalError) as e:
        logger.warning(f"Failed to compute obstacle complement: {e}")
        return []

    if obstacles_union.is_empty:
        logger.info("No obstacles computed (walkable area covers entire bounds)")
        return []

    # Convert to list of polygons
    if isinstance(obstacles_union, Polygon):
        result = [obstacles_union]
    elif isinstance(obstacles_union, MultiPolygon):
        result = list(obstacles_union.geoms)
    else:
        result = []

    logger.info(f"Computed {len(result)} obstacle polygons from complement")
    return result


def _flatten_obstacle_polygons(obstacles: list[Polygon]) -> list[Polygon]:
    """Return obstacle polygons without interior holes.

    MapDefinition obstacles cannot represent holes; triangulate polygons with
    interiors and keep only triangles within the original geometry so walkable
    holes are not reintroduced as obstacles.
    """
    flattened: list[Polygon] = []
    for poly in obstacles:
        if poly.is_empty:
            continue
        if not poly.interiors:
            flattened.append(poly)
            continue
        try:
            triangles = [tri for tri in triangulate(poly) if tri.within(poly)]
        except (ValueError, TopologicalError) as exc:
            logger.warning(f"Failed to triangulate obstacle polygon: {exc}")
            flattened.append(poly)
            continue
        flattened.extend(triangles)
    return flattened


def osm_to_map_definition(
    pbf_file: str,
    bbox: tuple | None = None,
    line_buffer_m: float = 1.5,
    tag_filters: OSMTagFilters | None = None,
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
    if tag_filters is None:
        tag_filters = OSMTagFilters()

    # Load PBF
    logger.info(f"Starting OSM→MapDefinition conversion for {pbf_file}")
    gdf = load_pbf(pbf_file, bbox)

    # Filter & extract
    driveable_ways = filter_driveable_ways(gdf, tag_filters)
    if driveable_ways.empty:
        raise ValueError(f"No driveable ways found in {pbf_file}")

    # Project to UTM
    gdf_utm, _utm_zone = project_to_utm(gdf)
    driveable_ways_utm, _ = project_to_utm(driveable_ways)

    # Buffer & cleanup
    buffered = buffer_ways(driveable_ways_utm, line_buffer_m / 2)
    buffered = cleanup_polygons(buffered)

    if not buffered:
        raise ValueError("No valid driveable areas after buffering and cleanup")

    walkable_union = unary_union(buffered)
    if walkable_union.is_empty:
        raise ValueError("Walkable union is empty")

    # Ensure walkable_union is a Polygon
    if isinstance(walkable_union, LineString):
        walkable_union = walkable_union.buffer(line_buffer_m / 2)

    # Compute obstacles
    bounds = gdf_utm.total_bounds
    obstacles_polys = compute_obstacles(bounds, walkable_union)
    obstacles_polys = _flatten_obstacle_polygons(obstacles_polys)

    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    # Shift geometries into a local frame with (0, 0) at the map bounds.
    xoff = -minx
    yoff = -miny
    buffered_local = [translate(poly, xoff=xoff, yoff=yoff) for poly in buffered]
    obstacles_local = [translate(poly, xoff=xoff, yoff=yoff) for poly in obstacles_polys]

    # Create Obstacle objects
    obstacles_list = []
    for obs_poly in obstacles_local:
        vertices = list(obs_poly.exterior.coords)[:-1]  # Remove duplicate last point
        obstacles_list.append(Obstacle(vertices=vertices))

    logger.info(
        f"Created MapDefinition with {len(obstacles_list)} obstacles and {len(buffered)} allowed areas"
    )

    # Build MapDefinition
    # Create default spawn/goal zones from bounds
    # Rect is tuple[Vec2D, Vec2D, Vec2D] (corner1, corner2, corner3)
    corner1 = (0.0, 0.0)
    corner2 = (width, 0.0)
    corner3 = (width, height)
    default_spawn_zone = (corner1, corner2, corner3)

    # Build MapDefinition with all required fields
    return MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles_list,
        robot_spawn_zones=[default_spawn_zone],
        ped_spawn_zones=[default_spawn_zone],
        robot_goal_zones=[default_spawn_zone],
        ped_goal_zones=[default_spawn_zone],
        bounds=[
            (0.0, width, 0.0, 0.0),  # bottom
            (0.0, width, height, height),  # top
            (0.0, 0.0, 0.0, height),  # left
            (width, width, 0.0, height),  # right
        ],
        robot_routes=[],
        ped_routes=[],
        ped_crowded_zones=[],
        allowed_areas=buffered_local,  # Explicit walkable areas in local frame
    )
