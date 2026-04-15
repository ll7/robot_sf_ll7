"""Tests for map_visualizer rendering helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon as ShapelyPolygon

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.common.types import Rect
from robot_sf.maps import map_visualizer
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle


def _map_def() -> MapDefinition:
    width = 6.0
    height = 4.0
    spawn_zone: Rect = ((0.5, 0.5), (1.0, 0.5), (1.0, 1.0))
    goal_zone: Rect = ((5.0, 3.0), (5.5, 3.0), (5.5, 3.5))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    obstacle = Obstacle([(2.0, 1.0), (3.0, 1.0), (3.0, 2.0), (2.0, 2.0)])
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.75, 0.75), (5.25, 3.25)],
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


def test_compute_figsize_clamps_values() -> None:
    """Verify figure sizing clamps to configured min/max bounds."""
    assert map_visualizer._compute_figsize(0.0, 0.0) == (10, 8)
    fig_w, fig_h = map_visualizer._compute_figsize(200.0, 200.0)
    assert fig_w <= 20.0
    assert fig_h <= 20.0


def test_zone_to_polygon_expands_triangles() -> None:
    """Ensure 3-point rectangles are expanded into four corners."""
    zone = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    polygon = map_visualizer._zone_to_polygon(zone)
    assert len(polygon) == 4


def test_visualize_map_definition_writes_file(tmp_path: Path) -> None:
    """Confirm visualization writes PNG output for inspection."""
    out_path = tmp_path / "map.png"
    map_visualizer.visualize_map_definition(_map_def(), out_path, title="map")
    assert out_path.exists()


def test_render_map_definition_draws_polygon_holes_as_compound_path() -> None:
    """Obstacle with a hole should be rendered as a single compound PathPatch.

    The compound path has two MOVETO codes — one for the exterior ring and one
    for the interior ring — so holes are cut out without overdrawing the canvas.
    """
    obstacle = Obstacle.from_geometry(
        ShapelyPolygon(
            [(1.0, 1.0), (5.0, 1.0), (5.0, 3.0), (1.0, 3.0)],
            holes=[[(2.0, 1.5), (4.0, 1.5), (4.0, 2.5), (2.0, 2.5)]],
        )
    )
    map_def = _map_def()
    map_def.obstacles = [obstacle]

    fig, ax = plt.subplots()

    map_visualizer.render_map_definition(
        map_def,
        ax,
        show_routes=False,
        show_pois=False,
        show_zone_labels=False,
    )

    path_patches = [p for p in ax.patches if isinstance(p, PathPatch)]
    assert len(path_patches) >= 1, "Expected at least one PathPatch for the obstacle"
    compound_path = path_patches[0].get_path()
    moveto_count = sum(1 for code in compound_path.codes if code == MplPath.MOVETO)
    assert moveto_count == 2, (
        f"Expected compound path with 2 MOVETO codes (exterior + interior), got {moveto_count}"
    )
    plt.close(fig)
