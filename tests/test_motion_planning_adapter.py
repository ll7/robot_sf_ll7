"""Tests for motion_planning_adapter helpers and visualization outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from python_motion_planning.common import TYPES

from robot_sf.common.types import Rect
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.motion_planning_adapter import (
    ClassicPlanVisualizer,
    MotionPlanningGridConfig,
    count_obstacle_cells,
    get_obstacle_statistics,
    map_definition_to_motion_planning_grid,
    set_start_goal_on_grid,
    visualize_grid,
    visualize_path,
)
from robot_sf.nav.obstacle import Obstacle


def _map_def_with_obstacle() -> MapDefinition:
    width = 4.0
    height = 4.0
    spawn_zone: Rect = ((0.5, 0.5), (1.0, 0.5), (0.5, 1.0))
    goal_zone: Rect = ((3.0, 3.0), (3.5, 3.0), (3.0, 3.5))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    obstacle = Obstacle([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)])
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.75, 0.75), (3.25, 3.25)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[obstacle],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def test_map_definition_to_motion_planning_grid_marks_obstacles() -> None:
    """Verify rasterization marks obstacle cells and preserves scale metadata."""
    grid = map_definition_to_motion_planning_grid(
        _map_def_with_obstacle(),
        MotionPlanningGridConfig(cells_per_meter=1.0, inflate_radius_cells=None),
    )
    assert grid.cells_per_meter == 1.0
    obstacle_count = count_obstacle_cells(grid)
    assert isinstance(obstacle_count, (int, np.integer))
    stats = get_obstacle_statistics(grid)
    assert stats["total_cells"] > 0


def test_set_start_goal_on_grid_marks_cells() -> None:
    """Ensure start and goal markers are written into the grid."""
    grid = map_definition_to_motion_planning_grid(_map_def_with_obstacle())
    set_start_goal_on_grid(grid, (1, 1), (2, 2))
    assert grid.type_map[1][1] == TYPES.START
    assert grid.type_map[2][2] == TYPES.GOAL


def test_visualize_grid_and_path_write_files(tmp_path: Path) -> None:
    """Check visualization helpers emit PNG files for offline inspection."""
    grid = map_definition_to_motion_planning_grid(_map_def_with_obstacle())
    grid_path = tmp_path / "grid.png"
    path_path = tmp_path / "path.png"
    visualize_grid(grid, grid_path, title="grid")
    visualize_path(grid, [(0, 0), (1, 1)], path_path, title="path")
    assert grid_path.exists()
    assert path_path.exists()


def test_visualize_path_handles_empty_path(tmp_path: Path) -> None:
    """Confirm empty paths still render the grid without errors."""
    grid = map_definition_to_motion_planning_grid(_map_def_with_obstacle())
    out_path = tmp_path / "empty_path.png"
    visualize_path(grid, [], out_path, title="empty")
    assert out_path.exists()


def test_visualization_helpers_forward_output_dpi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure configured output DPI is forwarded to matplotlib savefig calls."""
    grid = map_definition_to_motion_planning_grid(_map_def_with_obstacle())
    calls: list[int | None] = []
    original = Figure.savefig

    def _savefig_spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("dpi"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _savefig_spy)
    visualize_grid(grid, tmp_path / "grid_dpi.png", output_dpi=450)
    visualize_path(grid, [(0, 0), (1, 1)], tmp_path / "path_dpi.png", output_dpi=320)
    assert calls == [450, 320]


@pytest.mark.parametrize("invalid_dpi", [0, -1])
@pytest.mark.parametrize("renderer", [visualize_grid, visualize_path])
def test_visualization_helpers_reject_non_positive_dpi(
    tmp_path: Path,
    invalid_dpi: int,
    renderer,
) -> None:
    """Reject invalid non-positive output DPI values early for both renderers."""
    grid = map_definition_to_motion_planning_grid(_map_def_with_obstacle())
    output_path = tmp_path / f"invalid_{renderer.__name__}_{invalid_dpi}.png"
    with pytest.raises(ValueError, match="output_dpi must be in range"):
        if renderer is visualize_grid:
            renderer(grid, output_path, output_dpi=invalid_dpi)
        else:
            renderer(grid, [(0, 0), (1, 1)], output_path, output_dpi=invalid_dpi)


def test_visualization_helpers_reject_excessive_dpi(tmp_path: Path) -> None:
    """Reject unreasonably high output DPI values to avoid resource exhaustion."""
    grid = map_definition_to_motion_planning_grid(_map_def_with_obstacle())
    try:
        visualize_grid(grid, tmp_path / "too_large.png", output_dpi=100_000)
    except ValueError as exc:
        assert "output_dpi must be in range" in str(exc)
    else:
        msg = "Expected ValueError for output_dpi=100000"
        raise AssertionError(msg)


@pytest.mark.parametrize("invalid_dpi", [0.9, float("inf"), float("nan")])
def test_visualization_helpers_reject_invalid_float_dpi(tmp_path: Path, invalid_dpi: float) -> None:
    """Reject float edge cases that previously produced misleading conversion behavior."""
    grid = map_definition_to_motion_planning_grid(_map_def_with_obstacle())
    with pytest.raises(ValueError):
        visualize_grid(grid, tmp_path / "invalid_float.png", output_dpi=invalid_dpi)


def test_classic_visualizer_resolves_scale_from_grid() -> None:
    """Validate meters-per-cell resolution inference from grid metadata."""
    grid = map_definition_to_motion_planning_grid(
        _map_def_with_obstacle(), MotionPlanningGridConfig(cells_per_meter=2.0)
    )
    vis = ClassicPlanVisualizer("test")
    try:
        assert vis._resolve_meters_per_cell(grid, None) == 0.5
    finally:
        vis.close()
