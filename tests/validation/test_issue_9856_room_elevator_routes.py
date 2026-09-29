"""Versioned room/elevator setup correction guards for issue #9856."""

from __future__ import annotations

import copy
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from shapely.geometry import LineString, Polygon

from robot_sf.nav.navigation import _goal_zone_polygon
from robot_sf.nav.svg_map_parser import _load_single_svg
from robot_sf.training.scenario_loader import load_scenarios_for_validation

ROOT = Path(__file__).parents[2]
BASE_MATRIX = ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3.yaml"
MATRIX = (
    ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_route_width_v4.yaml"
)
SOURCE_AND_SUCCESSOR = {
    "entering_room": (
        "maps/svg_maps/francis2023/francis2023_entering_room.svg",
        "3d18c06ee133949b14e4274c548b19b61f10ed518c105b5a94f660b53376c945",
        "1506b9ddbee141c7256c92802cb6513d3add4cf5a270a9bad3dd197ec7a69166",
        [(7.0, 10.0), (14.5, 9.0), (21.0, 10.0)],
    ),
    "exiting_room": (
        "maps/successor_svg_maps/issue_9762_francis2023_exiting_room_goal_zone_entry_v2.svg",
        "3e84da29278394078098cf393e7e6fe54914b7365aa99621468b3ade4ffd2fbf",
        "cfd25fa09fd2a315f84f39408e57c0ec59bff14ea86d6020a07b93d75a3e434c",
        [(18.0, 10.0), (14.5, 9.0), (3.0, 10.0)],
    ),
    "entering_elevator": (
        "maps/svg_maps/francis2023/francis2023_entering_elevator.svg",
        "fa52e7b55bae9ae8a1fdfc0c13cd3e18e43d2c005cb325fe9a978fa1c39f6c10",
        "1b486a22cbd1ee1677732bd9d3086892e83d7c6e86f6ad380db921c621597563",
        [(7.0, 10.0), (12.5, 9.5), (15.0, 10.0)],
    ),
    "exiting_elevator": (
        "maps/successor_svg_maps/issue_9762_francis2023_exiting_elevator_goal_zone_entry_v2.svg",
        "10ad90b9ee7dac2d84ec0ec8017575b9ab5b2549d249e215ec88b64d88db9019",
        "4d176eca64ea8528f5eabd9df4e0314a4a3b0b7ba615736933cd3a5697d3eb1d",
        [(12.5, 9.5), (3.0, 10.0)],
    ),
}
ELEVATOR_OBSTACLES = (
    ("12", "6.6", "6.4", "1"),
    ("12", "12.4", "6.4", "1"),
    ("17.4", "6.6", "1", "6.8"),
    ("12", "6.6", "1", "1"),
    ("12", "12.4", "1", "1"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, object]]:
    result = load_scenarios_for_validation(path, base_dir=ROOT)
    assert result.load_error is None
    assert not result.load_issues
    assert not result.entry_issues
    return [dict(row) for row in result.scenarios]


def _normalized_svg(path: Path, *, allow_elevator_shell: bool) -> bytes:
    root = ET.parse(path).getroot()
    obstacles = []
    for element in root.iter():
        label = next(
            (value for key, value in element.attrib.items() if key.endswith("}label")), None
        )
        if label == "robot_route_0_0":
            element.set("d", "<ROBOT_ROUTE_DATA>")
        if label == "obstacle" and element.tag.endswith("rect"):
            obstacles.append(element)
    assert len(obstacles) >= 9
    if allow_elevator_shell:
        for element in obstacles[4:9]:
            for key in ("x", "y", "width", "height"):
                element.set(key, "<ELEVATOR_SHELL>")
    return ET.tostring(root, encoding="utf-8")


def test_successor_matrix_changes_only_four_map_files_and_preserves_historical_bytes() -> None:
    """The 48-identity release slice and every non-map row field stay fixed."""
    assert _sha256(MATRIX) == "8a8aa9adb2b098f039a3e92a4849e16eb8d3665e928211f08e0c689584c9f389"
    assert _sha256(ROOT / "configs/scenarios/classic_interactions_francis2023.yaml") == (
        "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"
    )
    assert _sha256(ROOT / "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg") == (
        "7538ed173d462a5107afc1a1e43b5b2e6d2bc5c9604035cdec9a551e20a8b15e"
    )
    before, after = _rows(BASE_MATRIX), _rows(MATRIX)
    assert len(before) == len(after) == 48
    assert [row["name"] for row in before] == [row["name"] for row in after]
    changed_names = set()
    changed_map_names = set()
    for old, new in zip(before, after, strict=True):
        old_rest, new_rest = copy.deepcopy(old), copy.deepcopy(new)
        old_rest.pop("map_file", None)
        new_rest.pop("map_file", None)
        old_rest.pop("map_id", None)
        new_rest.pop("map_id", None)
        assert old_rest == new_rest
        map_changes = {key for key in ("map_file", "map_id") if old.get(key) != new.get(key)}
        if map_changes:
            # Include map_id-only changes so they cannot hide outside the four successors.
            changed_map_names.add(str(new["name"]))
        if old["map_file"] != new["map_file"]:
            changed_names.add(str(new["name"]))
    assert changed_names == {f"francis2023_{name}" for name in SOURCE_AND_SUCCESSOR}
    assert changed_map_names == changed_names


def test_successors_preserve_interactions_and_clear_robot_route() -> None:
    """All route segments clear the radius plus preflight margin in parsed geometry."""
    for name, (
        source,
        source_sha,
        successor_sha,
        expected_waypoints,
    ) in SOURCE_AND_SUCCESSOR.items():
        source_path = ROOT / source
        successor_path = (
            ROOT / f"maps/successor_svg_maps/issue_9856_francis2023_{name}_route_width_v1.svg"
        )
        assert _sha256(source_path) == source_sha
        assert _sha256(successor_path) == successor_sha
        elevator = "elevator" in name
        assert _normalized_svg(source_path, allow_elevator_shell=elevator) == _normalized_svg(
            successor_path, allow_elevator_shell=elevator
        )

        definition = _load_single_svg(successor_path, strict=True)[successor_path.stem]
        route = definition.robot_routes[0]
        assert route.waypoints == expected_waypoints
        polygons = [Polygon(obstacle.vertices) for obstacle in definition.obstacles]
        segment_clearances = [
            min(LineString([start, end]).distance(obstacle) for obstacle in polygons)
            for start, end in zip(route.waypoints, route.waypoints[1:], strict=False)
        ]
        assert segment_clearances and min(segment_clearances) >= 1.1
        if name == "entering_elevator":
            # The sampled final target can be anywhere inside this unchanged goal zone.
            assert (
                min(_goal_zone_polygon(route.goal_zone).distance(obstacle) for obstacle in polygons)
                >= 1.1
            )

        if elevator:
            root = ET.parse(successor_path).getroot()
            obstacle_rects = [
                node
                for node in root.iter()
                if node.tag.endswith("rect")
                and any(
                    key.endswith("}label") and value == "obstacle"
                    for key, value in node.attrib.items()
                )
            ]
            assert (
                tuple(
                    tuple(rect.get(key) for key in ("x", "y", "width", "height"))
                    for rect in obstacle_rects[4:9]
                )
                == ELEVATOR_OBSTACLES
            )
