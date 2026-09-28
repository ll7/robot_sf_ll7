"""Versioned goal-zone route margin checks for issue #9762."""

from __future__ import annotations

import copy
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from shapely.geometry import LineString, Point, Polygon

from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS
from robot_sf.nav.navigation import _goal_zone_polygon
from robot_sf.nav.svg_map_parser import _load_single_svg
from robot_sf.training.scenario_loader import load_scenarios_for_validation, resolve_map_id

REPO_ROOT = Path(__file__).parents[2]
RELEASE_007_MATRIX = REPO_ROOT / "configs/scenarios/classic_interactions_francis2023.yaml"
RELEASE_007_MATRIX_SHA256 = "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"
BASE_SUCCESSOR_MATRIX = (
    REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
)
SUCCESSOR_MATRIX = (
    REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3.yaml"
)
DOORWAY_SLICE_MANIFEST = REPO_ROOT / "configs/benchmarks/issue_9348_three_width_doorway_v1.yaml"
MIN_ROBOT_GOAL_MARGIN_M = DEFAULT_ROBOT_RADIUS
MARGIN_TOLERANCE_M = 1e-6

AFFECTED_SOURCE_MAPS = frozenset(
    {
        "maps/svg_maps/classic_bottleneck.svg",
        "maps/svg_maps/classic_crossing.svg",
        "maps/svg_maps/classic_bottleneck_high.svg",
        "maps/svg_maps/classic_bottleneck_medium.svg",
        "maps/svg_maps/classic_doorway.svg",
        "maps/svg_maps/classic_group_crossing.svg",
        "maps/svg_maps/classic_head_on_corridor.svg",
        "maps/svg_maps/classic_merging.svg",
        "maps/svg_maps/classic_overtaking.svg",
        "maps/svg_maps/classic_realworld_bottleneck.svg",
        "maps/svg_maps/classic_t_intersection.svg",
        "maps/svg_maps/classic_urban_crossing.svg",
        "maps/svg_maps/classic_station_platform.svg",
        "maps/svg_maps/francis2023/francis2023_blind_corner.svg",
        "maps/svg_maps/francis2023/francis2023_circular_crossing.svg",
        "maps/svg_maps/francis2023/francis2023_crowd_navigation.svg",
        "maps/svg_maps/francis2023/francis2023_down_path.svg",
        "maps/svg_maps/francis2023/francis2023_exiting_elevator.svg",
        "maps/svg_maps/francis2023/francis2023_exiting_room.svg",
        "maps/svg_maps/francis2023/francis2023_frontal_approach.svg",
        "maps/svg_maps/francis2023/francis2023_intersection_no_gesture.svg",
        "maps/svg_maps/francis2023/francis2023_join_group.svg",
        "maps/svg_maps/francis2023/francis2023_leave_group.svg",
        "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg",
        "maps/svg_maps/francis2023/francis2023_narrow_hallway.svg",
        "maps/svg_maps/francis2023/francis2023_parallel_traffic.svg",
        "maps/svg_maps/francis2023/francis2023_ped_obstruction.svg",
        "maps/svg_maps/francis2023/francis2023_ped_overtaking.svg",
        "maps/svg_maps/francis2023/francis2023_perpendicular_traffic.svg",
        "maps/svg_maps/francis2023/francis2023_robot_crowding.svg",
        "maps/svg_maps/francis2023/francis2023_robot_overtaking.svg",
    }
)
SUCCESSOR_MAPS = frozenset(
    f"maps/successor_svg_maps/issue_9762_{Path(source).stem}_goal_zone_entry_v2.svg"
    for source in AFFECTED_SOURCE_MAPS
)
SUCCESSOR_BASE_MAPS = {
    "maps/svg_maps/classic_station_platform.svg": (
        "maps/successor_svg_maps/classic_station_platform_v2.svg"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_matrix(path: Path) -> list[dict[str, object]]:
    report = load_scenarios_for_validation(path, base_dir=REPO_ROOT)
    assert report.load_error is None, f"{path.name}: {report.load_error}"
    assert not report.load_issues, f"{path.name}: {report.load_issues}"
    assert not report.entry_issues, f"{path.name}: {report.entry_issues}"
    return [dict(row) for row in report.scenarios]


def _map_path(scenario: dict[str, object]) -> Path:
    map_id = scenario.get("map_id")
    if isinstance(map_id, str) and map_id:
        return resolve_map_id(map_id, source=SUCCESSOR_MATRIX).resolve()
    map_file = scenario.get("map_file")
    assert isinstance(map_file, str) and map_file, f"missing map for {scenario.get('name')}"
    path = Path(map_file)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _without_robot_route_data(path: Path) -> bytes:
    """Normalize only route path data so map identity diffs stay explainable."""
    root = ET.parse(path).getroot()
    normalized = 0
    for node in root.iter():
        labels = [
            value for key, value in node.attrib.items() if key.endswith("}label") or key == "label"
        ]
        if any(re.fullmatch(r"robot_route_\d+_\d+", label) for label in labels):
            assert "d" in node.attrib, f"robot route in {path} lacks path data"
            node.attrib["d"] = "<ROBOT_ROUTE_DATA>"
            normalized += 1
    assert normalized > 0, f"no robot routes found in {path}"
    return ET.tostring(root, encoding="utf-8")


def test_successor_matrix_preserves_roster_and_historical_map_bytes() -> None:
    """The release successor changes only map identity; the #9348 baseline stays frozen."""
    release_rows = _load_matrix(RELEASE_007_MATRIX)
    base_successor_rows = _load_matrix(BASE_SUCCESSOR_MATRIX)
    successor_rows = _load_matrix(SUCCESSOR_MATRIX)
    assert _sha256(RELEASE_007_MATRIX) == RELEASE_007_MATRIX_SHA256, (
        "0.0.7 scenario matrix bytes changed"
    )
    release_names = [row.get("name") for row in release_rows]
    base_successor_names = [row.get("name") for row in base_successor_rows]
    successor_names = [row.get("name") for row in successor_rows]
    assert len(release_names) == len(set(release_names)), (
        "0.0.7 release matrix has duplicate scenarios"
    )
    assert base_successor_names == release_names, (
        "base successor matrix changed the 0.0.7 scenario roster or order"
    )
    assert successor_names == release_names, (
        "successor matrix changed release scenario roster or order"
    )

    def without_map_identity(row: dict[str, object]) -> dict[str, object]:
        normalized = copy.deepcopy(row)
        normalized.pop("map_file", None)
        normalized.pop("map_id", None)
        return normalized

    assert [without_map_identity(row) for row in successor_rows] == [
        without_map_identity(row) for row in base_successor_rows
    ], "successor matrix changed a non-map scenario condition"

    registry = yaml.safe_load((REPO_ROOT / "maps/registry.yaml").read_text(encoding="utf-8"))
    registry_hashes = {record["path"]: record["source_sha256"] for record in registry["maps"]}
    for source in AFFECTED_SOURCE_MAPS:
        source_path = REPO_ROOT / source
        registry_path = source.removeprefix("maps/")
        assert registry_hashes.get(registry_path) == _sha256(source_path), (
            f"historical source map is not byte-identical to its pinned registry entry: {source}"
        )

    doorway_manifest = yaml.safe_load(DOORWAY_SLICE_MANIFEST.read_text(encoding="utf-8"))
    doorway = doorway_manifest["base_scenario"]
    historical_doorway = REPO_ROOT / doorway["map_path"]
    assert doorway["map_path"] == "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg"
    assert _sha256(historical_doorway) == doorway["map_sha256"], (
        "issue #9348 historical 2 m doorway map bytes changed"
    )
    assert doorway["map_path"] not in SUCCESSOR_MAPS


def test_successor_maps_change_only_robot_route_and_keep_all_release_rows_reachable() -> None:
    """Every affected 0.0.8 route has a full robot-radius goal-zone margin."""
    rows = _load_matrix(SUCCESSOR_MATRIX)
    represented_successors: set[str] = set()
    offenders: list[str] = []
    checked_routes = 0
    radius_checked_routes = 0

    for index, scenario in enumerate(rows):
        name = scenario.get("name")
        assert isinstance(name, str) and name, f"successor row {index} has no scenario identity"
        map_path = _map_path(scenario)
        relative_map = map_path.relative_to(REPO_ROOT).as_posix()
        definitions = _load_single_svg(map_path, strict=True)
        map_def = definitions.get(map_path.stem)
        if map_def is None or len(definitions) != 1:
            offenders.append(f"{name}: parser expected one map definition in {relative_map}")
            continue
        routes = map_def.robot_routes or []
        if not routes:
            offenders.append(f"{name}: no robot routes in {relative_map}")
            continue
        for route_index, route in enumerate(routes):
            checked_routes += 1
            route_id = f"{name} route {route_index}"
            if not route.waypoints or not route.goal_zone:
                offenders.append(
                    f"{route_id}: route lacks waypoints or a goal zone in {relative_map}"
                )
                continue
            point = Point(route.waypoints[-1])
            polygon = _goal_zone_polygon(route.goal_zone)
            if not polygon.covers(point):
                offenders.append(
                    f"{route_id}: final waypoint is outside goal zone in {relative_map}"
                )
                continue
            if relative_map in SUCCESSOR_MAPS:
                represented_successors.add(relative_map)
            radius_checked_routes += 1
            margin = polygon.boundary.distance(point)
            if margin + MARGIN_TOLERANCE_M < MIN_ROBOT_GOAL_MARGIN_M:
                offenders.append(
                    f"{route_id}: goal-zone margin {margin:.6f} m is below "
                    f"{MIN_ROBOT_GOAL_MARGIN_M:.3f} m in {relative_map}"
                )

    assert checked_routes > 0, "successor matrix resolved to zero goal-zone routes"
    assert radius_checked_routes > 0, "successor matrix resolved to zero radius-sensitive routes"
    assert represented_successors == SUCCESSOR_MAPS, (
        f"successor map coverage differs: missing={sorted(SUCCESSOR_MAPS - represented_successors)}, "
        f"unexpected={sorted(represented_successors - SUCCESSOR_MAPS)}"
    )
    assert not offenders, (
        f"{len(offenders)} goal-zone route finding(s) under goal_zone_entry_v1:\n"
        + "\n".join(offenders)
    )


def test_successor_geometry_diff_is_limited_to_robot_route_data() -> None:
    """Versioned route repairs retain geometry and the certified interaction route."""
    for source in AFFECTED_SOURCE_MAPS:
        original_path = REPO_ROOT / SUCCESSOR_BASE_MAPS.get(source, source)
        successor_path = (
            REPO_ROOT
            / "maps/successor_svg_maps"
            / (f"issue_9762_{Path(source).stem}_goal_zone_entry_v2.svg")
        )
        assert successor_path.is_file(), f"missing successor map for {source}"
        assert _without_robot_route_data(original_path) == _without_robot_route_data(
            successor_path
        ), (
            f"successor map changed non-route SVG data: {source} -> "
            f"{successor_path.relative_to(REPO_ROOT)}"
        )

        original_def = _load_single_svg(original_path, strict=True)[original_path.stem]
        successor_def = _load_single_svg(successor_path, strict=True)[successor_path.stem]
        original_routes = original_def.robot_routes or []
        successor_routes = successor_def.robot_routes or []
        assert len(original_routes) == len(successor_routes) == 1, (
            f"expected one robot route in {source} and {successor_path}"
        )
        old_waypoints = list(original_routes[0].waypoints)
        new_waypoints = list(successor_routes[0].waypoints)
        if source == "maps/svg_maps/classic_crossing.svg":
            assert new_waypoints[:-1] == old_waypoints, (
                "classic crossing successor must append a goal-zone segment without changing "
                "the certified interaction route"
            )
            assert new_waypoints[-1] == (34.0, 34.0)
            added_segment = LineString([old_waypoints[-1], new_waypoints[-1]])
            obstacle_polygons = [
                Polygon(obstacle.vertices)
                for obstacle in successor_def.obstacles
                if getattr(obstacle, "vertices", None)
            ]
            assert obstacle_polygons, "classic crossing successor needs parsed obstacles"
            assert not any(added_segment.intersects(obstacle) for obstacle in obstacle_polygons)
            min_clearance = min(added_segment.distance(obstacle) for obstacle in obstacle_polygons)
            assert min_clearance + MARGIN_TOLERANCE_M >= DEFAULT_ROBOT_RADIUS
        elif source == "maps/svg_maps/classic_station_platform.svg":
            assert new_waypoints[:-1] == old_waypoints[:-1], (
                "station-platform successor must retain the #9725 spawn-corrected route "
                "approach before extending its endpoint"
            )
            assert old_waypoints[-1] == (77.0, 22.5)
            assert new_waypoints[-1] == (78.0, 22.5)
            added_segment = LineString([old_waypoints[-1], new_waypoints[-1]])
            obstacle_polygons = [
                Polygon(obstacle.vertices)
                for obstacle in successor_def.obstacles
                if getattr(obstacle, "vertices", None)
            ]
            assert obstacle_polygons, "station-platform successor needs parsed obstacles"
            assert not any(added_segment.intersects(obstacle) for obstacle in obstacle_polygons)
            min_clearance = min(added_segment.distance(obstacle) for obstacle in obstacle_polygons)
            assert min_clearance + MARGIN_TOLERANCE_M >= DEFAULT_ROBOT_RADIUS
        elif source == "maps/svg_maps/classic_realworld_bottleneck.svg":
            assert new_waypoints[:-1] == old_waypoints, (
                "realworld bottleneck successor must preserve the original route"
            )
            assert old_waypoints[-1] == (53.0, 15.0)
            assert new_waypoints[-1] == (54.0, 15.0)
            added_segment = LineString([old_waypoints[-1], new_waypoints[-1]])
            obstacle_polygons = [
                Polygon(obstacle.vertices)
                for obstacle in successor_def.obstacles
                if getattr(obstacle, "vertices", None)
            ]
            assert obstacle_polygons, "realworld bottleneck successor needs parsed obstacles"
            assert not any(added_segment.intersects(obstacle) for obstacle in obstacle_polygons)
            min_clearance = min(added_segment.distance(obstacle) for obstacle in obstacle_polygons)
            assert min_clearance + MARGIN_TOLERANCE_M >= DEFAULT_ROBOT_RADIUS
        elif source == "maps/svg_maps/classic_urban_crossing.svg":
            assert new_waypoints[:-1] == old_waypoints, (
                "urban crossing successor must preserve the original route"
            )
            assert old_waypoints[-1] == (39.0, 25.0)
            assert new_waypoints[-1] == (40.0, 25.0)
            added_segment = LineString([old_waypoints[-1], new_waypoints[-1]])
            obstacle_polygons = [
                Polygon(obstacle.vertices)
                for obstacle in successor_def.obstacles
                if getattr(obstacle, "vertices", None)
            ]
            assert obstacle_polygons, "urban crossing successor needs parsed obstacles"
            assert not any(added_segment.intersects(obstacle) for obstacle in obstacle_polygons)
            min_clearance = min(added_segment.distance(obstacle) for obstacle in obstacle_polygons)
            assert min_clearance + MARGIN_TOLERANCE_M >= DEFAULT_ROBOT_RADIUS
        else:
            assert len(new_waypoints) == len(old_waypoints)
            assert new_waypoints[:-1] == old_waypoints[:-1], (
                f"successor changed route approach before endpoint: {source}"
            )
