"""Regression tests for self-intersecting SVG obstacle path handling."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from loguru import logger
from shapely.geometry import GeometryCollection, Polygon

from robot_sf.nav import svg_map_parser
from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.svg_map_parser import SvgMapConverter


def _get_path_by_id(converter: SvgMapConverter, path_id: str):
    """Return parsed SvgPath with matching SVG id."""
    for path in converter.path_info:
        if path.id == path_id:
            return path
    raise AssertionError(f"Expected path id={path_id!r} in SVG map")


def test_self_intersecting_obstacle_paths_are_repaired() -> None:
    """Known self-intersecting obstacle paths should become valid polygons."""
    repo_root = Path(__file__).resolve().parents[1]
    svg_fixture = (
        repo_root / "maps" / "obstacle_svg_maps" / "uni_campus_with_lake_as_obstacle_and_routes.svg"
    )
    converter = SvgMapConverter(str(svg_fixture))

    for path_id in ("path3", "path1948", "path1951"):
        svg_path = _get_path_by_id(converter, path_id)
        assert svg_path.label == "obstacle"

        obstacle = converter._process_obstacle_path(svg_path)
        assert obstacle.geometry is not None, f"Obstacle {path_id} should keep repaired geometry"
        assert obstacle.geometry.is_valid, (
            f"Obstacle {path_id} should be repaired to valid geometry"
        )
        assert obstacle.geometry.area > 0.0


def test_self_intersecting_obstacle_warnings_include_svg_filename() -> None:
    """Obstacle repair warnings should name the SVG map file that triggered them."""
    svg_map_parser._LOGGED_OBSTACLE_PATH_EVENTS.clear()
    repo_root = Path(__file__).resolve().parents[1]
    svg_fixture = (
        repo_root / "maps" / "obstacle_svg_maps" / "uni_campus_with_lake_as_obstacle_and_routes.svg"
    )

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        SvgMapConverter(str(svg_fixture))
    finally:
        logger.remove(sink_id)

    assert any(svg_fixture.name in message for message in messages)
    assert any("invalid polygon" in message and svg_fixture.name in message for message in messages)


def test_self_intersecting_obstacle_repair_logs_once_per_path() -> None:
    """Repeated reparsing should not spam the same obstacle repair warning/info lines."""
    svg_map_parser._LOGGED_OBSTACLE_PATH_EVENTS.clear()
    repo_root = Path(__file__).resolve().parents[1]
    svg_fixture = (
        repo_root / "maps" / "obstacle_svg_maps" / "uni_campus_with_lake_as_obstacle_and_routes.svg"
    )
    converter = SvgMapConverter(str(svg_fixture))
    svg_path = _get_path_by_id(converter, "path3")
    svg_map_parser._LOGGED_OBSTACLE_PATH_EVENTS.clear()

    messages: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: messages.append((message.record["level"].name, message.record["message"])),
        level="INFO",
    )
    try:
        converter._process_obstacle_path(svg_path)
        converter._process_obstacle_path(svg_path)
    finally:
        logger.remove(sink_id)
        svg_map_parser._LOGGED_OBSTACLE_PATH_EVENTS.clear()

    invalid_warnings = [
        message
        for level, message in messages
        if level == "WARNING" and "invalid polygon" in message
    ]
    repair_infos = [
        message
        for level, message in messages
        if level == "INFO" and "repaired invalid obstacle path" in message
    ]
    assert len(invalid_warnings) == 1
    assert len(repair_infos) == 1


def test_obstacle_log_dedupe_distinguishes_same_filename_in_different_dirs(
    tmp_path: Path,
) -> None:
    """Obstacle log dedupe should key by full path while keeping filename-only output."""
    svg_map_parser._LOGGED_OBSTACLE_PATH_EVENTS.clear()
    first_svg = tmp_path / "first" / "same.svg"
    second_svg = tmp_path / "second" / "same.svg"
    first_svg.parent.mkdir()
    second_svg.parent.mkdir()
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        for svg_path in (first_svg, second_svg, first_svg):
            svg_map_parser._log_obstacle_path_event_once(
                "invalid_polygon",
                svg_path,
                "obstacle_13",
                "warning",
                "SVG file '{svg}' obstacle path id={pid} produced invalid polygon",
            )
    finally:
        logger.remove(sink_id)
        svg_map_parser._LOGGED_OBSTACLE_PATH_EVENTS.clear()

    assert messages == [
        "SVG file 'same.svg' obstacle path id=obstacle_13 produced invalid polygon",
        "SVG file 'same.svg' obstacle path id=obstacle_13 produced invalid polygon",
    ]


def test_compound_obstacle_paths_preserve_detached_members(tmp_path: Path) -> None:
    """Compound SVG obstacle paths should keep all detached polygon members."""
    svg_file = tmp_path / "compound_obstacles.svg"
    svg_file.write_text(
        dedent(
            """
            <svg xmlns="http://www.w3.org/2000/svg"
                 xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
                 width="12" height="4" viewBox="0 0 12 4">
              <path id="compound_obstacle"
                    inkscape:label="obstacle"
                    d="M 0 0 L 4 0 L 4 4 L 0 4 Z M 6 0 L 10 0 L 10 4 L 6 4 Z"/>
            </svg>
            """
        ).strip(),
        encoding="utf-8",
    )

    converter = SvgMapConverter(str(svg_file))
    svg_path = _get_path_by_id(converter, "compound_obstacle")
    obstacle = converter._process_obstacle_path(svg_path)

    polygons = obstacle.iter_polygons()
    assert len(polygons) == 2
    assert all(poly.is_valid and poly.area > 0.0 for poly in polygons)
    assert obstacle.contains_point((1.0, 1.0))
    assert obstacle.contains_point((7.0, 1.0))
    assert not obstacle.contains_point((5.0, 1.0))


def test_obstacle_geometry_adapter_promotes_short_vertex_lists() -> None:
    """Geometry-backed obstacles should normalize short legacy vertex lists."""
    geometry = Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 0.0)])
    obstacle = Obstacle(vertices=[(0.0, 0.0)], geometry=geometry)

    assert obstacle.geometry is not None
    assert obstacle.geometry.is_valid
    assert len(obstacle.vertices) == 3
    assert obstacle.contains_point((1.0, 0.5))
    assert obstacle.iter_polygons() == [geometry]


def test_polygon_members_handles_deep_nested_geometry_collection() -> None:
    """Nested geometry collections should be flattened without recursion errors."""
    geometry = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)])
    for _ in range(1500):
        geometry = GeometryCollection([geometry])

    members = SvgMapConverter._polygon_members(geometry)
    assert len(members) == 1
    assert members[0].is_valid
    assert members[0].area > 0.0
