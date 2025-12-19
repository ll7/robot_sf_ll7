Collecting workspace informationHere’s a concise evaluation and recommended approach tailored to this codebase.

Summary of options
- Vector (recommended): Parse OSM tags into walkable polygons and derive obstacles as the complement inside bounds. Integrates cleanly with `robot_sf.nav.map_config.MapDefinition` and downstream planners. Precise scaling via projection. New deps: GeoPandas/OSMnx/Shapely/PyProj.
- Raster: Build an occupancy mask directly from OSM features and use it with the grid adapters in `robot_sf.nav.motion_planning_adapter`. Robust for planning; less reusable for rendering and MapDefinition-driven flows.
- Strengthen current color-based SVG: Keep fast-pysf converter, add better polygon repairs and scale derivation. Still limited (depends on color; misses sidewalks and access semantics).

Recommendation
- Implement a vector pipeline that computes walkable areas from OSM semantics, then generates obstacles as the geometric complement within the map bounds. This feeds existing code that expects obstacle polygons, routes, zones, etc.

Key details
- Fetch OSM ways/areas by tags (exclude steps). Default line width $w=3\,\text{m}$; buffer radius $r=w/2$ so $r=1.5\,\text{m}$.
- Project geometries to a meter-based CRS (e.g., UTM) before buffering to ensure accurate widths.
- Walkable union = buffered “driveable lines” + “driveable areas” (highway=footway|path|cycleway; highway=pedestrian + area=yes; etc.).
- Obstacles = bounds polygon difference walkable union. Optionally union in building footprints and water features for realism when available.
- Clean geometries with Shapely (buffer(0), simplify) to avoid invalid polygons and self-intersections (addresses “Obstacles can sometimes not be handled as polygons correctly”).
- Produce [`robot_sf.nav.obstacle.Obstacle`](robot_sf/nav/obstacle.py) from polygon exteriors and assemble a [`robot_sf.nav.map_config.MapDefinition`](robot_sf/nav/map_config.py) with computed bounds. Spawn/goal zones may be synthesized or taken from annotations later.

Integration points
- New module for OSM import (programmatic API; no CLI): e.g., robot_sf/maps/osm_walkable_import.py.
- Optional: extend `robot_sf.maps.map_visualizer` to overlay walkable areas for sanity checks.
- Keep existing `robot_sf.nav.svg_map_parser.SvgMapConverter` unchanged; the OSM importer produces a MapDefinition directly.

Incremental plan
1) Create importer with minimal tag filters and buffering. Project to meters. Validate with a small bbox.  
2) Add geometric cleaning and complement-to-obstacles generation.  
3) Provide a demo script under examples/ to render and inspect results; reuse visualizer.  
4) Add tests for: valid MapDefinition, bounds correctness, non-empty obstacles, and invalid polygon repair.  
5) Document workflow in SVG_MAP_EDITOR.md (new section: “OSM-based walkable import”).

Proposed skeleton
````python
# ...existing code...
from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle

DEFAULT_LINE_WIDTH_M = 3.0  # driveable line width
DEFAULT_BUFFER_R_M = DEFAULT_LINE_WIDTH_M / 2  # r = w/2

def _to_obstacles(polys: Iterable[Polygon]) -> list[Obstacle]:
    obs: list[Obstacle] = []
    for poly in polys:
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)  # repair minor invalidities
            if poly.is_empty:
                continue
        # support MultiPolygon exploded earlier; here assume Polygon
        x, y = poly.exterior.xy
        vertices = list(zip(x, y))
        # drop duplicate closing point if present; Obstacle handles closing by lines
        if len(vertices) >= 2 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        obs.append(Obstacle(vertices))
    return obs

def osm_to_map_definition(
    *,
    place: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,  # (minx, miny, maxx, maxy) in lat/lon
    line_buffer_m: float = DEFAULT_BUFFER_R_M,
) -> MapDefinition:
    """
    Build a MapDefinition from OSM data by computing walkable areas and using
    the geometric complement as obstacles.

    Args:
        place: Named place query (e.g., "Augsburg, Germany").
        bbox: (minx, miny, maxx, maxy) geographical bbox in WGS84; used if place is None.
        line_buffer_m: Half-width buffer for driveable lines (meters).

    Returns:
        MapDefinition with obstacles derived from non-walkable areas inside bounds.
    """
    if place:
        gdf = ox.geocode_to_gdf(place)
        boundary_wgs84 = gdf.geometry.iloc[0]
    elif bbox:
        minx, miny, maxx, maxy = bbox
        boundary_wgs84 = Polygon(
            [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        )
    else:
        raise ValueError("Provide either 'place' or 'bbox'.")

    # Project boundary and query footprints in a local meter-based CRS
    boundary = gpd.GeoSeries([boundary_wgs84], crs="EPSG:4326").to_crs(ox.projection.project_gdf(
        gpd.GeoDataFrame(geometry=[boundary_wgs84], crs="EPSG:4326")
    ).crs).iloc[0]

    tags_lines = {
        "highway": [
            "footway", "path", "cycleway", "bridleway",
            # include shared paths; exclude steps later
        ],
    }
    tags_areas = {
        "highway": ["pedestrian", "footway", "path", "service", "residential", "unclassified"],
        "area": True,
    }

    # Lines (paths/footways), exclude steps
    lines = ox.geometries_from_polygon(boundary, tags_lines)
    if not lines.empty:
        lines = lines[~lines.get("highway", "").astype(str).str.contains("steps", na=False)]
        lines_proj = lines.to_crs(boundary.crs)
        line_geoms = lines_proj.geometry.buffer(line_buffer_m, cap_style=2)  # square caps for corridors
    else:
        line_geoms = gpd.GeoSeries([], crs=boundary.crs)

    # Polygons (explicit areas)
    areas = ox.geometries_from_polygon(boundary, tags_areas)
    area_geoms = areas.to_crs(boundary.crs).geometry if not areas.empty else gpd.GeoSeries([], crs=boundary.crs)

    walkable_union = unary_union([*line_geoms.geometry, *area_geoms])
    # Boundaries → polygon
    bounds_poly = Polygon(list(boundary.exterior.coords))

    # Obstacles = bounds minus walkable areas
    non_walkable = bounds_poly.difference(walkable_union)
    if isinstance(non_walkable, Polygon):
        obstacle_polys = [non_walkable]
    elif isinstance(non_walkable, MultiPolygon):
        obstacle_polys = list(non_walkable.geoms)
    else:
        obstacle_polys = []

    obstacles = _to_obstacles(obstacle_polys)

    # MapDefinition dimensions from bounds
    minx, miny, maxx, maxy = bounds_poly.bounds
    width = maxx - minx
    height = maxy - miny

    # Convert bounds to MapDefinition format
    bounds_lines = [
        (minx, maxx, miny, miny),  # bottom
        (minx, maxx, maxy, maxy),  # top
        (minx, minx, miny, maxy),  # left
        (maxx, maxx, miny, maxy),  # right
    ]

    # Minimal required zones (allow usage without manual annotations)
    center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
    minimal_zone = (
        (center_x - 1, center_y - 1),
        (center_x + 1, center_y - 1),
        (center_x + 1, center_y + 1),
    )

    return MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles,
        robot_spawn_zones=[minimal_zone],
        ped_spawn_zones=[],
        robot_goal_zones=[minimal_zone],
        bounds=bounds_lines,
        robot_routes=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[],
    )
# ...existing code...
````

Notes
- This complements the current color-based converter in map_osm_converter.py by using OSM semantics, fixing scale via projection, and repairing invalid polygons.
- Obstacles are derived from the non-walkable area, aligning with the desire to “define allowed areas” and make driveable space explicit.
- Use the visualizer in `robot_sf.maps.map_visualizer` to inspect outputs.
- For planners, you can rasterize the resulting MapDefinition with `robot_sf.nav.motion_planning_adapter.map_definition_to_motion_planning_grid`.

Testing
- Add unit tests validating MapDefinition shape and non-empty obstacles.
- Add a small bbox fixture and assert widths/heights > 0, and obstacle polygons are valid.

If you want to keep the SVG workflow, improve polygon validity checks in `robot_sf.nav.svg_map_parser.SvgMapConverter._process_obstacle_path` (already warns and validates with Shapely) and derive scale from OSM metadata rather than color-based constants.