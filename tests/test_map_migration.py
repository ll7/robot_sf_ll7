"""Tests for the SVG-only map cutover and loader behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map

MAPS_ROOT = Path(__file__).resolve().parents[1] / "robot_sf" / "maps"
UNI_CAMPUS_SVG = MAPS_ROOT / "uni_campus_big.svg"


def test_repository_map_assets_are_svg_only() -> None:
    """The repository should not ship legacy JSON map assets."""
    json_maps = sorted(MAPS_ROOT.glob("*.json"))
    assert not json_maps, f"Legacy map assets remain: {json_maps}"


def test_svg_maps_are_loaded_and_json_files_are_ignored(tmp_path: Path) -> None:
    """Map folders should load SVG maps and ignore legacy JSON files."""
    (tmp_path / "alpha.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alpha.svg").write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="20" height="10" viewBox="0 0 20 10">
  <rect inkscape:label="robot_spawn_zone" x="1" y="1" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone" x="17" y="1" width="1" height="1" />
  <path inkscape:label="robot_route_0_0" d="M 1 1 L 10 1 L 18 1" />
</svg>
        """.strip(),
        encoding="utf-8",
    )

    pool = MapDefinitionPool(maps_folder=str(tmp_path))

    assert list(pool.map_defs) == ["alpha"]
    assert pool.map_defs["alpha"].width == pytest.approx(20.0)
    assert pool.map_defs["alpha"].height == pytest.approx(10.0)


def test_json_only_map_folder_is_rejected(tmp_path: Path) -> None:
    """Folders without SVG maps should fail closed."""
    (tmp_path / "legacy.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        MapDefinitionPool(maps_folder=str(tmp_path))


def test_migrated_svg_map_loads() -> None:
    """The migrated campus map should still parse successfully as SVG."""
    map_def = convert_map(str(UNI_CAMPUS_SVG))

    assert map_def is not None
    assert map_def.robot_routes
    assert map_def.ped_routes
