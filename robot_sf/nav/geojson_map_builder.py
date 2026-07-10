# ruff: noqa: DOC201
"""Convert annotated OSM-derived GeoJSON into a runnable Robot SF map.

The converter keeps the repository's existing segment-map YAML contract: obstacle
polygons are emitted as vertices and become collision line segments when loaded.
GeoJSON uses WGS84 when no CRS is declared, then is projected to a local UTM
coordinate frame so all output positions and line-buffer widths are in metres.

OpenStreetMap tags identify walkable and obstacle geometry.  Scenario-specific
intent is deliberately explicit: GeoJSON features use ``robot_sf_role`` (or the
equivalent ``robot_sf:role``) for zones and routes.  This avoids guessing a robot
route or spawn position from public map geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas
import yaml
from loguru import logger
from shapely.affinity import translate
from shapely.errors import TopologicalError
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from robot_sf.nav.map_config import MapDefinition, serialize_map
from robot_sf.nav.osm_map_builder import OSMTagFilters, cleanup_polygons, compute_obstacles

if TYPE_CHECKING:
    from collections.abc import Iterable

_ROLE_KEYS = ("robot_sf_role", "robot_sf:role")
_ID_KEYS = ("robot_sf_id", "robot_sf:id", "id", "name")
_SPAWN_KEYS = ("robot_sf_spawn", "robot_sf:spawn")
_GOAL_KEYS = ("robot_sf_goal", "robot_sf:goal")
_ZONE_ROLES = (
    "robot_spawn",
    "robot_goal",
    "ped_spawn",
    "ped_goal",
    "ped_crowded",
)
_ROUTE_ROLES = ("robot_route", "ped_route")


def load_geojson(path: str | Path) -> geopandas.GeoDataFrame:
    """Load a non-empty GeoJSON file and assign the standard WGS84 CRS when absent."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"GeoJSON file not found: {source}")
    try:
        features = geopandas.read_file(source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read GeoJSON file {source}: {exc}") from exc
    if features.empty:
        raise ValueError(f"GeoJSON file contains no features: {source}")
    if features.crs is None:
        features = features.set_crs("EPSG:4326", allow_override=True)
    return features


def geojson_to_map_structure(
    path: str | Path,
    *,
    line_buffer_m: float = 1.5,
    tag_filters: OSMTagFilters | None = None,
) -> dict[str, Any]:
    """Convert annotated GeoJSON into the YAML structure consumed by ``serialize_map``.

    ``robot_sf_role`` values ``robot_spawn``, ``robot_goal``, ``ped_spawn``,
    ``ped_goal``, and ``ped_crowded`` declare polygon zones.  Each zone needs a
    stable ``robot_sf_id``.  ``robot_route`` and ``ped_route`` LineStrings use
    ``robot_sf_spawn`` and ``robot_sf_goal`` to name their endpoint zones.

    Raises:
        ValueError: When geometry is invalid, no walkable geometry is available,
            or the GeoJSON lacks the metadata required for a runnable robot map.
    """
    if line_buffer_m <= 0:
        raise ValueError("line_buffer_m must be greater than zero")
    filters = tag_filters or OSMTagFilters()
    projected = _project_to_local_metric(load_geojson(path))
    walkable_polys = _walkable_polygons(projected, filters, line_buffer_m)
    obstacle_polys = _obstacle_polygons(projected, filters, line_buffer_m)
    if obstacle_polys:
        try:
            walkable_geometry = unary_union(walkable_polys).difference(unary_union(obstacle_polys))
        except (ValueError, TopologicalError) as exc:
            raise ValueError(
                f"Could not subtract obstacle geometry from walkable areas: {exc}"
            ) from exc
        walkable_polys = cleanup_polygons(_polygon_parts(walkable_geometry))
    if not walkable_polys:
        raise ValueError("No valid walkable areas remain after obstacle exclusion")

    min_x, min_y, max_x, max_y = projected.total_bounds
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("GeoJSON bounds must have non-zero width and height")

    walkable_union = unary_union(walkable_polys)
    obstacle_segments = compute_obstacles((min_x, min_y, max_x, max_y), walkable_union)
    local_obstacles = [_local_polygon_vertices(poly, min_x, min_y) for poly in obstacle_segments]
    zones = _extract_zones(projected, min_x=min_x, min_y=min_y)
    _require_runnable_robot_zones(zones)
    routes = _extract_routes(projected, min_x=min_x, min_y=min_y)
    if not routes["robot_routes"]:
        raise ValueError(
            "No robot_route features found. Add an annotated LineString with "
            "robot_sf_spawn and robot_sf_goal metadata."
        )

    return {
        "x_margin": [0.0, float(max_x - min_x)],
        "y_margin": [0.0, float(max_y - min_y)],
        "obstacles": local_obstacles,
        "robot_spawn_zones": zones["robot_spawn"],
        "robot_goal_zones": zones["robot_goal"],
        "ped_spawn_zones": zones["ped_spawn"],
        "ped_goal_zones": zones["ped_goal"],
        "ped_crowded_zones": zones["ped_crowded"],
        "robot_routes": routes["robot_routes"],
        "ped_routes": routes["ped_routes"],
    }


def geojson_to_map_definition(
    path: str | Path,
    *,
    line_buffer_m: float = 1.5,
    tag_filters: OSMTagFilters | None = None,
) -> MapDefinition:
    """Convert annotated GeoJSON to the runtime ``MapDefinition`` representation."""
    return serialize_map(
        geojson_to_map_structure(path, line_buffer_m=line_buffer_m, tag_filters=tag_filters)
    )


def write_segment_map(
    source: str | Path,
    destination: str | Path,
    *,
    line_buffer_m: float = 1.5,
) -> Path:
    """Write a deterministic YAML segment-map converted from annotated GeoJSON."""
    target = Path(destination)
    if target.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Segment-map output must use a .yaml or .yml extension")
    target.parent.mkdir(parents=True, exist_ok=True)
    structure = geojson_to_map_structure(source, line_buffer_m=line_buffer_m)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(structure, stream, sort_keys=False)
    return target


def _project_to_local_metric(features: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Project all GeoJSON geometry into a single local UTM CRS measured in metres."""
    crs = features.estimate_utm_crs()
    if crs is None:
        raise ValueError("Could not determine a local UTM CRS from GeoJSON geometry")
    return features.to_crs(crs)


def _walkable_polygons(
    features: geopandas.GeoDataFrame,
    filters: OSMTagFilters,
    line_buffer_m: float,
) -> list[Polygon]:
    """Return valid polygons for explicit and OSM-tagged walkable geometry."""
    mask = _role_mask(features, "walkable") | _column_values_in(
        features, "highway", filters.driveable_highways
    )
    mask |= _column_values_in(features, "footway", {"sidewalk", "crossing"})
    mask |= _column_values_in(features, "area", filters.driveable_areas)
    polygons = cleanup_polygons(_buffer_geometries(features.loc[mask, "geometry"], line_buffer_m))
    if not polygons:
        raise ValueError(
            "No walkable geometry found. Add OSM highway/footway/area tags or "
            "robot_sf_role=walkable."
        )
    return polygons


def _obstacle_polygons(
    features: geopandas.GeoDataFrame,
    filters: OSMTagFilters,
    line_buffer_m: float,
) -> list[Polygon]:
    """Return valid polygons for explicit and OSM-tagged obstacle geometry."""
    mask = _role_mask(features, "obstacle")
    for key, value in filters.obstacle_tags:
        if key not in features.columns:
            continue
        values = features[key]
        mask |= values.notna() if value == "*" else values == value
    return cleanup_polygons(_buffer_geometries(features.loc[mask, "geometry"], line_buffer_m))


def _buffer_geometries(geometries: Iterable[Any], line_buffer_m: float) -> list[Polygon]:
    """Convert polygonal and line geometry to polygons without dropping multi-geometries."""
    polygons: list[Polygon] = []
    for geometry in geometries:
        for part in _geometry_parts(geometry):
            if isinstance(part, Polygon):
                polygons.append(part)
            elif isinstance(part, LineString):
                polygons.extend(_polygon_parts(part.buffer(line_buffer_m, cap_style="round")))
    return polygons


def _geometry_parts(geometry: Any) -> list[Any]:
    """Flatten supported Shapely collection types while ignoring empty geometry."""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, (Polygon, LineString)):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [part for child in geometry.geoms for part in _geometry_parts(child)]
    return []


def _polygon_parts(geometry: Any) -> list[Polygon]:
    """Extract non-empty polygons from a Shapely geometry or geometry collection."""
    return [part for part in _geometry_parts(geometry) if isinstance(part, Polygon)]


def _role_mask(features: geopandas.GeoDataFrame, role: str) -> Any:
    """Return a row mask for one documented Robot SF GeoJSON role."""
    mask = features.geometry.is_empty & False
    for key in _ROLE_KEYS:
        if key in features.columns:
            mask |= features[key].astype("string").str.lower() == role
    return mask


def _column_values_in(features: geopandas.GeoDataFrame, key: str, values: Iterable[str]) -> Any:
    """Return a row mask when a GeoJSON property contains one of ``values``."""
    if key not in features.columns:
        return features.geometry.is_empty & False
    return features[key].astype("string").str.lower().isin({value.lower() for value in values})


def _feature_value(feature: Any, keys: Iterable[str]) -> str | None:
    """Return the first non-empty metadata value from ``keys`` for a GeoJSON row."""
    for key in keys:
        value = feature.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _extract_zones(
    features: geopandas.GeoDataFrame,
    *,
    min_x: float,
    min_y: float,
) -> dict[str, list[list[tuple[float, float]]]]:
    """Extract ordered, rectangular scenario zones from explicitly annotated polygons."""
    zones: dict[str, list[tuple[str, list[tuple[float, float]]]]] = {
        role: [] for role in _ZONE_ROLES
    }
    seen_ids: set[str] = set()
    for _, feature in features.iterrows():
        role = _feature_value(feature, _ROLE_KEYS)
        if role not in _ZONE_ROLES:
            continue
        zone_id = _feature_value(feature, _ID_KEYS)
        if zone_id is None:
            raise ValueError(f"{role} feature requires a robot_sf_id")
        if zone_id in seen_ids:
            raise ValueError(f"Duplicate robot_sf_id for a zone: {zone_id!r}")
        polygons = _polygon_parts(feature.geometry)
        if len(polygons) != 1:
            raise ValueError(f"{role} zone {zone_id!r} must be one Polygon")
        rectangle = polygons[0].minimum_rotated_rectangle
        if not isinstance(rectangle, Polygon) or rectangle.area <= 0:
            raise ValueError(f"{role} zone {zone_id!r} has no usable area")
        coordinates = list(rectangle.exterior.coords)[:3]
        zones[role].append(
            (zone_id, [(float(x - min_x), float(y - min_y)) for x, y in coordinates])
        )
        seen_ids.add(zone_id)
    return {role: [zone for _, zone in sorted(entries)] for role, entries in zones.items()}


def _require_runnable_robot_zones(zones: dict[str, list[list[tuple[float, float]]]]) -> None:
    """Fail closed unless explicit robot start and goal zones are present."""
    missing = [role for role in ("robot_spawn", "robot_goal") if not zones[role]]
    if missing:
        raise ValueError(
            "GeoJSON is not a runnable robot scenario; add "
            + ", ".join(f"robot_sf_role={role}" for role in missing)
            + " polygon features."
        )


def _extract_routes(
    features: geopandas.GeoDataFrame,
    *,
    min_x: float,
    min_y: float,
) -> dict[str, list[dict[str, Any]]]:
    """Extract routes and resolve their named zone endpoints to stable output indexes."""
    zone_lookup = _zone_lookup(features)
    output: dict[str, list[dict[str, Any]]] = {"robot_routes": [], "ped_routes": []}
    for _, feature in features.iterrows():
        role = _feature_value(feature, _ROLE_KEYS)
        if role not in _ROUTE_ROLES:
            continue
        spawn_name = _feature_value(feature, _SPAWN_KEYS)
        goal_name = _feature_value(feature, _GOAL_KEYS)
        if spawn_name is None or goal_name is None:
            raise ValueError(f"{role} feature requires robot_sf_spawn and robot_sf_goal metadata")
        prefix = "robot" if role == "robot_route" else "ped"
        try:
            spawn_id = zone_lookup[f"{prefix}_spawn"][spawn_name]
            goal_id = zone_lookup[f"{prefix}_goal"][goal_name]
        except KeyError as exc:
            raise ValueError(
                f"{role} references an unknown {prefix} zone: {exc.args[0]!r}"
            ) from exc
        lines = [part for part in _geometry_parts(feature.geometry) if isinstance(part, LineString)]
        if len(lines) != 1 or len(lines[0].coords) < 2:
            raise ValueError(
                f"{role} {spawn_name!r}->{goal_name!r} must be one LineString with two points"
            )
        output[f"{prefix}_routes"].append(
            {
                "spawn_id": spawn_id,
                "goal_id": goal_id,
                "waypoints": [[float(x - min_x), float(y - min_y)] for x, y, *_ in lines[0].coords],
            }
        )
    return output


def _zone_lookup(features: geopandas.GeoDataFrame) -> dict[str, dict[str, int]]:
    """Map annotated zone names to the deterministic index used in emitted YAML."""
    records: dict[str, list[str]] = {role: [] for role in _ZONE_ROLES}
    for _, feature in features.iterrows():
        role = _feature_value(feature, _ROLE_KEYS)
        if role in records:
            zone_id = _feature_value(feature, _ID_KEYS)
            if zone_id is not None:
                records[role].append(zone_id)
    return {
        role: {zone_id: index for index, zone_id in enumerate(sorted(ids))}
        for role, ids in records.items()
    }


def _local_polygon_vertices(polygon: Polygon, min_x: float, min_y: float) -> list[list[float]]:
    """Return an obstacle polygon exterior as local-frame vertices without a repeated endpoint."""
    local = translate(polygon, xoff=-min_x, yoff=-min_y)
    return [[float(x), float(y)] for x, y in list(local.exterior.coords)[:-1]]


def build_parser() -> argparse.ArgumentParser:
    """Build the documented GeoJSON-to-segment-map command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Annotated OSM or GeoJSON FeatureCollection")
    parser.add_argument("output", type=Path, help="Destination Robot SF YAML map")
    parser.add_argument(
        "--line-buffer-m",
        type=float,
        default=1.5,
        help="Half-width in metres for tagged LineString paths and obstacles (default: 1.5)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the GeoJSON converter CLI and print the created map path."""
    args = build_parser().parse_args(argv)
    output = write_segment_map(args.input, args.output, line_buffer_m=args.line_buffer_m)
    logger.info("Wrote Robot SF segment map: {}", output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper script.
    raise SystemExit(main())
