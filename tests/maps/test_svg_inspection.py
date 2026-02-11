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
    assert any(f.code == "PED_ROUTE_ONLY_MODE" for f in report.findings)


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
