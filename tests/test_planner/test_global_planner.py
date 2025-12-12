"""Tests for the basic global planner path generation (US1)."""
# ruff: noqa: D103

from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig, PlanningFailedError

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "test_maps"


def _load_map(name: str):
    map_path = FIXTURE_ROOT / name
    return convert_map(str(map_path))


def _inflate_obstacles(map_def, margin: float) -> list[Polygon]:
    return [Polygon(obs.vertices).buffer(margin) for obs in map_def.obstacles]


def test_basic_path_generation_avoids_obstacles():
    map_def = _load_map("simple_corridor.svg")
    config = PlannerConfig(robot_radius=0.4, min_safe_clearance=0.3, fallback_on_failure=False)
    planner = GlobalPlanner(map_def, config=config)

    start = (1.2, 4.5)
    goal = (18.0, 4.5)

    path = planner.plan(start, goal)

    assert path[0] == start
    assert path[-1] == goal
    assert len(path) >= 2

    # Verify clearance from obstacles using inflated polygons
    inflated = _inflate_obstacles(map_def, config.robot_radius + config.min_safe_clearance)
    for waypoint in path:
        pt = Point(waypoint)
        assert all(not poly.contains(pt) for poly in inflated)


def test_clearance_respected_in_narrow_passage():
    map_def = _load_map("narrow_passage.svg")
    config = PlannerConfig(robot_radius=0.4, min_safe_clearance=0.2, fallback_on_failure=False)
    planner = GlobalPlanner(map_def, config=config)

    start = (1.0, 3.5)
    goal = (11.0, 3.5)

    with pytest.raises(PlanningFailedError):
        planner.plan(start, goal)


def test_returns_straight_line_on_empty_map(tmp_path):
    svg = tmp_path / "empty.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="10" height="5">
  <rect inkscape:label="robot_spawn_zone" x="0.5" y="2" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone" x="8.5" y="2" width="1" height="1" />
  <path inkscape:label="robot_route_0_0" d="M 0.5 2.5 L 9 2.5" />
</svg>
        """.strip()
    )
    map_def = convert_map(str(svg))
    planner = GlobalPlanner(map_def)

    start = (1.0, 2.5)
    goal = (9.0, 2.5)

    path = planner.plan(start, goal)

    assert path == [start, goal]


def test_raises_when_no_path_exists():
    map_def = _load_map("no_path.svg")
    config = PlannerConfig(fallback_on_failure=False)
    planner = GlobalPlanner(map_def, config=config)

    start = (1.0, 3.5)
    goal = (11.0, 3.5)

    with pytest.raises(PlanningFailedError):
        planner.plan(start, goal)


def test_plan_respects_via_pois(tmp_path):
    svg = tmp_path / "via.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="10" height="5">
  <rect inkscape:label="robot_spawn_zone" x="0.5" y="2" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone" x="8.5" y="2" width="1" height="1" />
  <path inkscape:label="robot_route_0_0" d="M 0.5 2.5 L 9 2.5" />
  <circle class="poi" id="poi_mid" cx="5" cy="2.5" r="0.2" inkscape:label="mid" />
</svg>
        """.strip()
    )
    map_def = convert_map(str(svg))
    planner = GlobalPlanner(map_def)

    start = (1.0, 2.5)
    goal = (9.0, 2.5)
    path = planner.plan(start, goal, via_pois=["poi_mid"])

    assert path[0] == start
    assert path[-1] == goal
    assert map_def.poi_positions[0] in path


def test_plan_allows_dynamic_start_not_in_spawn_zone():
    map_def = _load_map("simple_corridor.svg")
    planner = GlobalPlanner(map_def)

    start = (2.5, 2.5)  # outside spawn zone but inside map
    goal = (18.0, 4.5)

    path = planner.plan(start, goal)

    assert path[0] == start
    assert path[-1] == goal


def test_invalidate_cache_clears_graph():
    map_def = _load_map("simple_corridor.svg")
    planner = GlobalPlanner(map_def)

    planner.plan((1.0, 4.5), (18.0, 4.5))
    assert planner._graph is not None  # type: ignore[attr-defined]

    planner.invalidate_cache()
    assert planner._graph is None  # type: ignore[attr-defined]


def test_smoothing_keeps_endpoints_and_reduces_points():
    map_def = _load_map("simple_corridor.svg")
    config = PlannerConfig(enable_smoothing=True, smoothing_epsilon=0.5)
    planner = GlobalPlanner(map_def, config=config)

    start = (1.0, 4.5)
    goal = (18.0, 4.5)
    path = planner.plan(start, goal)

    assert path[0] == start
    assert path[-1] == goal
    assert len(path) <= 5  # simplified compared to unsmoothed path
