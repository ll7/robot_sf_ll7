"""Regression tests for self-intersecting SVG obstacle path handling."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import Polygon

from robot_sf.nav.svg_map_parser import SvgMapConverter


def _get_path_by_id(converter: SvgMapConverter, path_id: str):
    """Return parsed SvgPath with matching SVG id."""
    for path in converter.path_info:
        if path.id == path_id:
            return path
    raise AssertionError(f"Expected path id={path_id!r} in SVG map")


def test_self_intersecting_obstacle_paths_are_repaired() -> None:
    """Known self-intersecting obstacle paths should become valid polygons."""
    converter = SvgMapConverter(
        str(Path("maps/obstacle_svg_maps/uni_campus_1350_obstacles_lake_traverse.svg"))
    )

    for path_id in ("path1948", "path1951"):
        svg_path = _get_path_by_id(converter, path_id)
        assert svg_path.label == "obstacle"

        obstacle = converter._process_obstacle_path(svg_path)
        polygon = Polygon(obstacle.vertices)
        assert polygon.is_valid, f"Obstacle {path_id} should be repaired to a valid polygon"
        assert polygon.area > 0.0
