"""Integration tests for SVG parsing and planner compatibility (US2)."""
# ruff: noqa: D103

from pathlib import Path

from robot_sf.nav.navigation import RouteNavigator
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner
from robot_sf.planner.global_planner import PlannerConfig

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "test_maps"


def test_parse_poi_circles(tmp_path):
    svg = tmp_path / "poi_map.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="5" height="5">
  <rect inkscape:label="robot_spawn_zone" x="0.5" y="0.5" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone" x="3.5" y="3.5" width="1" height="1" />
  <circle class="poi" id="poi_mid" cx="2.5" cy="2.5" r="0.2" inkscape:label="midpoint" />
  <path inkscape:label="robot_route_0_0" d="M 0.5 0.5 L 4 4" />
</svg>
        """.strip()
    )

    map_def = convert_map(str(svg))

    assert len(map_def.poi_positions) == 1
    assert map_def.poi_labels == {"poi_mid": "midpoint"}
    assert map_def.get_poi_by_label("midpoint") == map_def.poi_positions[0]


def test_route_navigator_accepts_planner_path():
    map_def = convert_map(str(FIXTURE_ROOT / "simple_corridor.svg"))
    planner = GlobalPlanner(map_def, PlannerConfig())

    start = (1.0, 4.5)
    goal = (18.0, 4.5)
    path = planner.plan(start, goal)

    navigator = RouteNavigator(path)
    assert navigator.current_waypoint == path[0]
    navigator.update_position(goal)
    assert navigator.reached_destination


def test_planner_runs_on_example_fixtures():
    map_def = convert_map(str(FIXTURE_ROOT / "complex_warehouse.svg"))
    planner = GlobalPlanner(map_def, PlannerConfig(fallback_on_failure=True))

    start = (1.5, 8.0)
    goal = (22.0, 12.0)
    path = planner.plan(start, goal)

    assert path[0] == start
    assert path[-1] == goal


def test_maps_without_pois_remain_backward_compatible():
    map_def = convert_map(str(FIXTURE_ROOT / "simple_corridor.svg"))

    assert map_def.poi_positions == []
    assert map_def.poi_labels == {}
