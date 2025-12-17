"""Unit tests for ClassicGlobalPlanner."""

from __future__ import annotations

import types

import pytest
from python_motion_planning.common import TYPES

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig, PlanningError


def _make_basic_map(tmp_path):
    svg = tmp_path / "basic_planner.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="6" height="3">
  <rect inkscape:label="robot_spawn_zone" x="0.2" y="0.2" width="0.5" height="0.5" />
  <rect inkscape:label="robot_goal_zone" x="4.8" y="0.2" width="0.5" height="0.5" />
  <path inkscape:label="robot_route_0_0" d="M 0.2 0.2 L 5.0 0.2" />
</svg>
        """.strip()
    )
    return convert_map(str(svg))


def _make_map_with_obstacle(tmp_path):
    svg = tmp_path / "blocked.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="3" height="3">
  <rect inkscape:label="obstacle" x="1" y="1" width="1" height="1" />
  <rect inkscape:label="robot_spawn_zone" x="0.2" y="0.2" width="0.5" height="0.5" />
  <rect inkscape:label="robot_goal_zone" x="2.2" y="0.2" width="0.5" height="0.5" />
  <path inkscape:label="robot_route_0_0" d="M 0.2 0.2 L 2.7 0.2" />
</svg>
        """.strip()
    )
    return convert_map(str(svg))


class SequenceRng:
    """Deterministic RNG returning a preset sequence."""

    def __init__(self, values):
        """Store a looping sequence of integers to replay via randrange."""
        self._values = list(values)
        self._idx = 0

    def randrange(self, upper):
        """Return the next value modulo the given upper bound."""
        value = self._values[self._idx % len(self._values)]
        self._idx += 1
        return value % upper


def test_scale_path_info_scales_length_and_sets_inflation(tmp_path):
    """Length scales by meters-per-cell and inflation is annotated."""
    map_def = _make_basic_map(tmp_path)
    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=2.0,
            inflate_radius_cells=1,
            add_boundary_obstacles=False,
        ),
    )
    grid = planner.grid
    raw_info = {"length": 4.0, "cost": 7}

    scaled = planner._scale_path_info(raw_info, grid, inflation=1)

    assert scaled["length"] == pytest.approx(2.0)  # length scales by meters_per_cell
    assert scaled["inflation_cells"] == 1
    assert scaled["cost"] == 7


def test_visualize_path_calls_fill_expands(monkeypatch, tmp_path):
    """visualize_path should overlay expands and pass the filled grid to renderer."""
    map_def = _make_basic_map(tmp_path)
    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=0,
            add_boundary_obstacles=False,
        ),
    )

    class DummyGrid:
        def __init__(self):
            self.expands = None

        def fill_expands(self, expands):
            self.expands = expands

    dummy_grid = DummyGrid()
    planner._grid = dummy_grid

    captured = {}

    def fake_render_path(grid, path, **kwargs):
        captured["grid"] = grid
        captured["path"] = path

    monkeypatch.setattr("robot_sf.planner.classic_global_planner.render_path", fake_render_path)

    path_world = [(0.5, 0.5), (1.5, 0.5)]
    expands = {"node": types.SimpleNamespace()}

    planner.visualize_path(path_world=path_world, path_info={"expand": expands}, show_expands=True)

    assert captured["grid"] is not None
    assert captured["grid"].expands == expands
    assert captured["path"] == [(0, 0), (1, 0)]


def test_plan_returns_expand_metadata(tmp_path):
    """plan should return expand metadata and scaled length."""
    map_def = _make_basic_map(tmp_path)
    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=0,
            add_boundary_obstacles=False,
        ),
    )

    start = (0.5, 0.5)
    goal = (4.5, 0.5)

    path, info = planner.plan(start, goal)

    expected_start = planner._grid_to_world(*planner._world_to_grid(*start))
    expected_goal = planner._grid_to_world(*planner._world_to_grid(*goal))

    assert path[0] == expected_start
    assert path[-1] == expected_goal
    assert info is not None
    assert info.get("expand")
    assert info.get("inflation_cells") == 0
    assert info.get("length", 0) == pytest.approx(expected_goal[0] - expected_start[0])


def test_plan_accepts_algorithm_override(monkeypatch, tmp_path):
    """plan should honor per-call algorithm override."""
    map_def = _make_basic_map(tmp_path)
    calls = {"a": 0, "theta": 0}

    class DummyAStar:
        def __init__(self, map_, start, goal):
            calls["a"] += 1
            self.start = start
            self.goal = goal

        def plan(self):
            return [self.start, self.goal], {"expand": {}}

    class DummyThetaStar:
        def __init__(self, map_, start, goal):
            calls["theta"] += 1
            self.start = start
            self.goal = goal

        def plan(self):
            return [self.start, self.goal], {"expand": {}}

    monkeypatch.setattr("robot_sf.planner.classic_global_planner.AStar", DummyAStar)
    monkeypatch.setattr("robot_sf.planner.classic_global_planner.ThetaStar", DummyThetaStar)

    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            algorithm="theta_star",
            cells_per_meter=1.0,
            inflate_radius_cells=0,
            add_boundary_obstacles=False,
        ),
    )

    path, info = planner.plan((0.5, 0.5), (1.5, 0.5), algorithm="a_star")

    assert calls["a"] == 1
    assert calls["theta"] == 0
    assert path[-1] == planner._grid_to_world(*planner._world_to_grid(1.5, 0.5))
    assert info is not None


def test_validate_point_rejects_obstacle(tmp_path):
    """validate_point should raise when the world point is occupied."""
    svg = tmp_path / "blocked.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="3" height="3">
  <rect inkscape:label="obstacle" x="1" y="1" width="1" height="1" />
  <rect inkscape:label="robot_spawn_zone" x="0.2" y="0.2" width="0.5" height="0.5" />
  <rect inkscape:label="robot_goal_zone" x="2.2" y="0.2" width="0.5" height="0.5" />
  <path inkscape:label="robot_route_0_0" d="M 0.2 0.2 L 2.7 0.2" />
</svg>
        """.strip()
    )
    map_def = convert_map(str(svg))
    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=0,
            add_boundary_obstacles=False,
        ),
    )

    with pytest.raises(PlanningError):
        planner.validate_point((1.1, 1.1))


def test_validate_point_allows_free_cell(tmp_path):
    """validate_point should return grid indices for free world points."""
    map_def = _make_basic_map(tmp_path)
    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=0,
            add_boundary_obstacles=False,
        ),
    )

    gx, gy = planner.validate_point((0.4, 0.4))
    assert (gx, gy) == planner._world_to_grid(0.4, 0.4)


def test_random_valid_point_on_grid_skips_invalid_cells(tmp_path):
    """random_valid_point_on_grid should reject obstacles and return a free world point."""
    map_def = _make_map_with_obstacle(tmp_path)
    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=0,
            add_boundary_obstacles=False,
        ),
    )
    rng = SequenceRng([1, 1, 0, 0])  # obstacle first, then a free cell

    point = planner.random_valid_point_on_grid(rng=rng, max_attempts=3)

    assert point == planner._grid_to_world(0, 0)
    cell_value = planner.grid.type_map[0][0]
    assert cell_value not in (TYPES.OBSTACLE, TYPES.INFLATION)


def test_plan_random_path_is_reproducible_with_seed(tmp_path):
    """plan_random_path should produce deterministic start/goal and path for the same seed."""
    map_def = _make_basic_map(tmp_path)
    planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=0,
            add_boundary_obstacles=False,
            algorithm="theta_star_v2",
        ),
    )

    path1, info1, start1, goal1 = planner.plan_random_path(seed=7, max_attempts=5)
    path2, info2, start2, goal2 = planner.plan_random_path(seed=7, max_attempts=5)

    assert start1 == start2
    assert goal1 == goal2
    assert path1 == path2
    assert path1[0] == planner._grid_to_world(*planner._world_to_grid(*start1))
    assert path1[-1] == planner._grid_to_world(*planner._world_to_grid(*goal1))
    assert info1 is not None
    assert info2 is not None
