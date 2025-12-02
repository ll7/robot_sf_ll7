"""Structural validation tests for classical interaction SVG maps.

These tests ensure each new classical interaction SVG map:
  * Parses without exception using SvgMapConverter
  * Produces a MapDefinition object
  * Contains at least one robot spawn zone and goal zone (MapDefinition logs error otherwise)
  * Has width/height > 0
  * Contains obstacles (except explicitly documented minimal cases)
  * Ensures any route labels with indices reference existing spawn/goal zone indices
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.svg_map_parser import SvgMapConverter

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "maps" / "svg_maps"

CLASSIC_PREFIX = "classic_"


def _classic_svg_files():
    """Classic svg files.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return sorted(p for p in SVG_DIR.glob(f"{CLASSIC_PREFIX}*.svg") if p.is_file())


@pytest.mark.parametrize("svg_path", _classic_svg_files())
def test_classic_svg_parse(svg_path: Path):
    """Test classic svg parse.

    Args:
        svg_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    converter = SvgMapConverter(str(svg_path))
    md = converter.get_map_definition()
    assert isinstance(md, MapDefinition), "Converter did not produce MapDefinition"
    assert md.width > 0 and md.height > 0, "Invalid map dimensions"
    # robot spawn & goal zones (parser logs errors; we enforce here)
    assert md.robot_spawn_zones, "No robot spawn zones found"
    assert md.robot_goal_zones, "No robot goal zones found"
    # obstacles check (allow empty only for overtaking narrow lane case? keep required for now)
    assert md.obstacles, "Expected at least one obstacle to define geometry"


@pytest.mark.parametrize("svg_path", _classic_svg_files())
def test_classic_svg_route_indices(svg_path: Path):
    """If routes encode indices (e.g., ped_route_0_1) ensure indices within bounds."""
    converter = SvgMapConverter(str(svg_path))
    md = converter.get_map_definition()
    max_spawn_robot = len(md.robot_spawn_zones) - 1 if md.robot_spawn_zones else -1
    max_goal_robot = len(md.robot_goal_zones) - 1 if md.robot_goal_zones else -1

    route_label_pattern = re.compile(r"(ped_route|robot_route)_(\d+)_(\d+)")
    # Access protected attributes path_info for labels (test-only introspection)
    for path in converter.path_info:
        if not path.label:
            continue
        m = route_label_pattern.fullmatch(path.label)
        if not m:
            continue
        kind, spawn_str, goal_str = m.groups()
        spawn_i, goal_i = int(spawn_str), int(goal_str)
        if kind == "robot_route":
            assert 0 <= spawn_i <= max_spawn_robot, (
                f"robot_route spawn index {spawn_i} out of range"
            )
            assert 0 <= goal_i <= max_goal_robot, f"robot_route goal index {goal_i} out of range"
        # ped_route indices would relate to ped zones; we just ensure non-negative
        if kind == "ped_route":
            assert spawn_i >= 0 and goal_i >= 0, "ped_route indices must be non-negative"
