"""Unit tests for SVG inspection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.maps.verification.svg_inspection import inspect_svg

if TYPE_CHECKING:
    from pathlib import Path


def _write_svg(path: Path, body: str) -> Path:
    """Write a minimal SVG wrapper with provided body content.

    Args:
        path: Destination SVG path.
        body: Inner SVG XML content.

    Returns:
        Path: The written SVG path.
    """

    path.write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg"\n'
            '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n'
            '     width="12" height="12">\n'
            f"{body}\n"
            "</svg>\n"
        ),
        encoding="utf-8",
    )
    return path


def test_inspect_svg_detects_ped_route_only_mode(tmp_path: Path) -> None:
    """Route-only pedestrian maps should be flagged as explicit route-only mode."""
    svg_path = _write_svg(
        tmp_path / "route_only.svg",
        """
  <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
  <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="10" y="10" width="1" height="1" />
  <path id="ped_path" inkscape:label="ped_route_0_0" d="M 2 2 L 9 9" />
""".strip(),
    )

    report = inspect_svg(svg_path)

    assert report.ped_route_only_mode is True
    assert report.capability_metadata.ped_route_only_mode is True
    assert report.capability_metadata.explicit_ped_spawn_zones == 0
    assert report.capability_metadata.synthetic_ped_spawn_zones == 1
    assert report.capability_metadata.synthetic_ped_goal_zones == 1
    assert report.capability_metadata.has_pedestrian_runtime_routes is True
    assert report.capability_metadata.has_explicit_pedestrian_runtime_zones is False
    assert "PED_ROUTE_ONLY_MODE" in report.capability_metadata.parser_limitation_codes
    assert any(f.code == "PED_ROUTE_ONLY_MODE" for f in report.findings)


def test_inspect_svg_reports_explicit_runtime_capabilities(tmp_path: Path) -> None:
    """Explicit zones, routes, obstacles, and single-ped markers should be exposed."""
    svg_path = _write_svg(
        tmp_path / "explicit_runtime.svg",
        """
  <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
  <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="10" y="10" width="1" height="1" />
  <rect id="ped_spawn_zone_0" inkscape:label="ped_spawn_zone_0" x="2" y="2" width="1" height="1" />
  <rect id="ped_goal_zone_0" inkscape:label="ped_goal_zone_0" x="9" y="9" width="1" height="1" />
  <rect id="wall" inkscape:label="obstacle" x="4" y="4" width="1" height="1" />
  <circle id="ped_a_start" inkscape:label="single_ped_a_start" cx="3" cy="3" r="0.2" />
  <circle id="ped_a_goal" inkscape:label="single_ped_a_goal" cx="8" cy="8" r="0.2" />
  <path id="robot_path" inkscape:label="robot_route_0_0" d="M 1 1 L 10 10" />
  <path id="ped_path" inkscape:label="ped_route_0_0" d="M 2 2 L 9 9" />
""".strip(),
    )

    metadata = inspect_svg(svg_path).capability_metadata

    assert metadata.schema_version == "parser-capability-metadata.v1"
    assert metadata.explicit_robot_spawn_zones == 1
    assert metadata.explicit_robot_goal_zones == 1
    assert metadata.explicit_ped_spawn_zones == 1
    assert metadata.explicit_ped_goal_zones == 1
    assert metadata.robot_routes == 1
    assert metadata.ped_routes == 1
    assert metadata.obstacle_count == 1
    assert metadata.single_ped_start_markers == 1
    assert metadata.single_ped_goal_markers == 1
    assert metadata.single_pedestrian_definitions == 1
    assert metadata.has_explicit_robot_runtime_zones is True
    assert metadata.has_explicit_pedestrian_runtime_zones is True
    assert metadata.has_obstacles is True


def test_inspect_svg_reports_obstacle_only_template_capabilities(tmp_path: Path) -> None:
    """Obstacle-only/template-style maps should not look runtime-runnable."""
    svg_path = _write_svg(
        tmp_path / "obstacle_only.svg",
        """
  <rect id="wall" inkscape:label="obstacle" x="4" y="4" width="2" height="2" />
""".strip(),
    )

    metadata = inspect_svg(svg_path).capability_metadata

    assert metadata.obstacle_count == 1
    assert metadata.has_obstacles is True
    assert metadata.has_robot_runtime_routes is False
    assert metadata.has_pedestrian_runtime_routes is False
    assert metadata.has_explicit_robot_runtime_zones is False
    assert metadata.has_explicit_pedestrian_runtime_zones is False


def test_inspect_svg_preserves_unindexed_route_identity(tmp_path: Path) -> None:
    """Unindexed route labels must retain source id/label and command linkage."""
    svg_path = _write_svg(
        tmp_path / "unindexed_route.svg",
        """
  <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
  <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="10" y="10" width="1" height="1" />
  <path id="ped_path" inkscape:label="ped_route" d="M 2 2 L 9 2 L 9 9" />
""".strip(),
    )

    report = inspect_svg(svg_path)

    ped_route = next(route for route in report.routes if route.kind == "ped")
    assert ped_route.label == "ped_route"
    assert ped_route.path_id == "ped_path"
    assert ped_route.waypoint_count == 3
    assert set(ped_route.commands) >= {"M", "L"}


def test_inspect_svg_flags_risky_route_commands(tmp_path: Path) -> None:
    """Horizontal/vertical and relative commands should be reported as parser-risky."""
    svg_path = _write_svg(
        tmp_path / "risky_commands.svg",
        """
  <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
  <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="10" y="10" width="1" height="1" />
  <rect id="ped_spawn_zone_0" inkscape:label="ped_spawn_zone_0" x="2" y="2" width="1" height="1" />
  <rect id="ped_goal_zone_0" inkscape:label="ped_goal_zone_0" x="9" y="9" width="1" height="1" />
  <path id="ped_path" inkscape:label="ped_route_0_0" d="m 2 2 H 9 V 9" />
""".strip(),
    )

    report = inspect_svg(svg_path)

    codes = {f.code for f in report.findings}
    assert "ROUTE_USES_RISKY_PATH_COMMANDS" in codes
    assert "ROUTE_USES_RELATIVE_COMMANDS" in codes


def test_inspect_svg_flags_route_obstacle_crossing(tmp_path: Path) -> None:
    """Routes crossing obstacle interiors should raise an error-level finding."""
    svg_path = _write_svg(
        tmp_path / "crossing.svg",
        """
  <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
  <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="10" y="10" width="1" height="1" />
  <rect id="ped_spawn_zone_0" inkscape:label="ped_spawn_zone_0" x="2" y="2" width="1" height="1" />
  <rect id="ped_goal_zone_0" inkscape:label="ped_goal_zone_0" x="9" y="9" width="1" height="1" />
  <rect id="wall" inkscape:label="obstacle" x="4" y="4" width="2" height="2" />
  <path id="ped_path" inkscape:label="ped_route_0_0" d="M 2 2 L 9 9" />
""".strip(),
    )

    report = inspect_svg(svg_path)

    crossing = [f for f in report.findings if f.code == "ROUTE_CROSSES_OBSTACLE_INTERIOR"]
    assert crossing, "Expected obstacle-interior crossing finding for route."
    assert all(f.severity == "error" for f in crossing)
