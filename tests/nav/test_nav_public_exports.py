"""Focused export contract for the ``robot_sf.nav`` package.

Guards the reviewed ``__all__`` surfaces for issue #6798: every declared
export resolves to the pre-change object on its pre-change import path, no
declared name is missing, and private, stale, or unreviewed foreign names never leak
into the public surface. The ``robot_sf.nav`` facade ``__init__`` is
deliberately not exercised here (it is guarded by its own reviewed list);
this module covers the fifteen individual modules that gained ``__all__``.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

MODULE_NAMES = [
    "footprint_diagnostic",
    "geojson_map_builder",
    "geojson_map_provenance",
    "global_route",
    "map_config",
    "nav_types",
    "navigation",
    "obstacle",
    "occupancy",
    "occupancy_grid",
    "occupancy_grid_rasterization",
    "occupancy_grid_utils",
    "osm_map_builder",
    "svg_map_parser",
    "uncertainty_envelope",
]

FOOTPRINT_DIAGNOSTIC_ALL = [
    "FOOTPRINT_KIND_CIRCULAR",
    "FOOTPRINT_KIND_RECTANGULAR",
    "FOOTPRINT_ORIENTATION_SCHEMA_VERSION",
    "REQUIRED_SCENARIO_FAMILY_IDS",
    "CircularFootprint",
    "FootprintClearanceResult",
    "FootprintDiagnosticScenario",
    "FootprintModel",
    "FootprintOrientationConfigError",
    "RectangularFootprint",
    "build_diagnostic_report",
    "build_diagnostic_scenarios",
    "centerline_clearance_m",
    "footprint_aware_clearance_m",
    "load_footprint_orientation_config",
    "parse_diagnostic_parameters",
    "parse_footprints",
    "run_footprint_diagnostic",
    "validate_footprint_orientation_config",
]

GEOJSON_MAP_BUILDER_ALL = [
    "build_parser",
    "geojson_to_map_definition",
    "geojson_to_map_structure",
    "load_geojson",
    "main",
    "write_segment_map",
]

GEOJSON_MAP_PROVENANCE_ALL = ["validate_import_provenance"]

GLOBAL_ROUTE_ALL = ["GlobalRoute"]

MAP_CONFIG_ALL = [
    "GlobalRoute",
    "InfrastructureZone",
    "MapDefinition",
    "MapDefinitionPool",
    "Obstacle",
    "PedestrianWaitRule",
    "SinglePedestrianDefinition",
    "SocialGroupDefinition",
    "parse_social_group_definitions",
    "serialize_map",
]

NAV_TYPES_ALL = [
    "SUPPORTED_SEMANTIC_BOUNDARY_FLAGS",
    "SemanticBoundary",
    "SvgCircle",
    "SvgPath",
    "SvgRectangle",
]

NAVIGATION_ALL = [
    "NavigationSettings",
    "RouteNavigator",
    "get_prepared_obstacles",
    "sample_route",
]

OBSTACLE_ALL = [
    "Obstacle",
    "obstacle_from_svgrectangle",
]

OCCUPANCY_ALL = [
    "ContinuousOccupancy",
    "EgoPedContinuousOccupancy",
    "check_quality_of_map_point",
    "circle_collides_any",
    "circle_collides_any_lines",
    "is_circle_circle_intersection",
    "is_circle_line_intersection",
]

OCCUPANCY_GRID_ALL = [
    "OBSERVATION_CHANNEL_ORDER",
    "OCCUPANCY_FREE_THRESHOLD",
    "GridChannel",
    "GridConfig",
    "OccupancyGrid",
    "POIQuery",
    "POIQueryType",
    "POIResult",
    "RobotPoseRecord",
]

OCCUPANCY_GRID_RASTERIZATION_ALL = [
    "_bresenham_line",
    "rasterize_circle",
    "rasterize_circle_fast",
    "rasterize_line_segment",
    "rasterize_obstacles",
    "rasterize_pedestrians",
    "rasterize_pedestrians_array",
    "rasterize_polygon",
    "rasterize_robot",
]

OCCUPANCY_GRID_UTILS_ALL = [
    "clip_to_grid",
    "ego_to_world",
    "get_affected_cells",
    "get_grid_bounds",
    "grid_indices_to_world",
    "is_within_grid",
    "world_to_ego",
    "world_to_grid_indices",
]

OSM_MAP_BUILDER_ALL = [
    "OSMTagFilters",
    "buffer_ways",
    "cleanup_polygons",
    "compute_obstacles",
    "extract_obstacles",
    "filter_driveable_ways",
    "load_pbf",
    "osm_to_map_definition",
    "project_to_utm",
]

SVG_MAP_PARSER_ALL = [
    "SvgMapConverter",
    "convert_map",
    "load_svg_maps",
]

UNCERTAINTY_ENVELOPE_ALL = [
    "DEFAULT_ALPHA_MPS",
    "ENVELOPE_SCHEMA_VERSION",
    "ConformalInflationPolicy",
    "PedestrianUncertaintyEnvelope",
    "SpatialInflationPolicy",
    "effective_pedestrian_radius",
    "envelope_diagnostics",
    "envelope_from_position",
    "linear_inflation_policy",
]

MODULE_ALL = {
    "footprint_diagnostic": FOOTPRINT_DIAGNOSTIC_ALL,
    "geojson_map_builder": GEOJSON_MAP_BUILDER_ALL,
    "geojson_map_provenance": GEOJSON_MAP_PROVENANCE_ALL,
    "global_route": GLOBAL_ROUTE_ALL,
    "map_config": MAP_CONFIG_ALL,
    "nav_types": NAV_TYPES_ALL,
    "navigation": NAVIGATION_ALL,
    "obstacle": OBSTACLE_ALL,
    "occupancy": OCCUPANCY_ALL,
    "occupancy_grid": OCCUPANCY_GRID_ALL,
    "occupancy_grid_rasterization": OCCUPANCY_GRID_RASTERIZATION_ALL,
    "occupancy_grid_utils": OCCUPANCY_GRID_UTILS_ALL,
    "osm_map_builder": OSM_MAP_BUILDER_ALL,
    "svg_map_parser": SVG_MAP_PARSER_ALL,
    "uncertainty_envelope": UNCERTAINTY_ENVELOPE_ALL,
}

# Known private / unreviewed foreign module-level names that must stay out of the public
# surface. ``_bresenham_line`` is the reviewed exception: a public perf-test
# consumer imports it directly (see tests/perf/test_hotpath_perf.py).
UNEXPORTED_NAMES = {
    "footprint_diagnostic": ["_parse_footprint", "_validate_top_level_contract"],
    "geojson_map_builder": ["_ROLE_KEYS", "_extract_zones", "argparse"],
    "geojson_map_provenance": ["_SCHEMA_VERSION", "_require_mapping_fields", "hashlib"],
    "global_route": ["dist", "Rect", "Vec2D"],
    "map_config": ["_normalize_position", "os", "random", "logger"],
    "nav_types": ["dataclass", "Zone", "Vec2D"],
    "navigation": ["_PLANNER_RETRY_ATTEMPTS", "np", "logger", "PlanningError"],
    "obstacle": ["pairwise", "np", "Line2D"],
    "occupancy": ["_DEGENERATE_SEGMENT_EPS", "numba", "euclid_dist"],
    "occupancy_grid": ["_PYGAME_UNLOADED", "pygame", "rasterization", "grid_utils"],
    "occupancy_grid_rasterization": ["_clip_line_to_rect", "_points_in_polygon", "math"],
    "occupancy_grid_utils": ["_extract_pose", "TYPE_CHECKING"],
    "osm_map_builder": ["_require_maps_dependencies", "logger", "triangulate"],
    "svg_map_parser": ["_PathParseState", "_load_single_svg", "re", "ET"],
    "uncertainty_envelope": ["Callable", "TypeAlias", "np", "Protocol"],
}

REVIEWED_COMPATIBILITY_EXPORTS = {
    ("map_config", "GlobalRoute"): (
        "robot_sf.nav.global_route",
        "GlobalRoute",
    ),
    ("map_config", "Obstacle"): (
        "robot_sf.nav.obstacle",
        "Obstacle",
    ),
}


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_nav_module_declares_the_reviewed_export_surface(module_name: str) -> None:
    """Every nav module exports exactly its reviewed public surface."""
    module = importlib.import_module(f"robot_sf.nav.{module_name}")
    assert module.__all__ == MODULE_ALL[module_name]
    assert set(module.__all__) <= set(dir(module))


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_nav_module_all_names_resolve_on_pre_change_paths(module_name: str) -> None:
    """Every declared nav export resolves with its reviewed identity."""
    module = importlib.import_module(f"robot_sf.nav.{module_name}")
    expected_module = f"robot_sf.nav.{module_name}"

    for name in module.__all__:
        export = getattr(module, name)

        compatibility_origin = REVIEWED_COMPATIBILITY_EXPORTS.get((module_name, name))
        if compatibility_origin is not None:
            origin_module_name, origin_name = compatibility_origin
            origin_module = importlib.import_module(origin_module_name)

            assert export is getattr(origin_module, origin_name)
            assert export.__module__ == origin_module_name
            assert export.__qualname__ == origin_name
            continue

        if inspect.isclass(export) or inspect.isfunction(export):
            assert export.__module__ == expected_module
            assert export.__qualname__ == name
        else:
            assert export is not None


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_nav_module_keeps_private_and_unreviewed_foreign_names_unexported(
    module_name: str,
) -> None:
    """Private, stale, and unreviewed foreign names never leak into the public surface."""
    module = importlib.import_module(f"robot_sf.nav.{module_name}")
    declared = set(module.__all__)
    for name in UNEXPORTED_NAMES[module_name]:
        assert name not in declared
        assert name not in MODULE_ALL[module_name]
