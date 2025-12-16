"""Unit tests for ClassicGlobalPlanner."""

from __future__ import annotations

import types

import pytest

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig


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
