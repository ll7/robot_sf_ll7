"""Tests for the SVG-only map cutover and loader behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.nav.map_config import MapDefinitionPool, serialize_map
from robot_sf.nav.svg_map_parser import convert_map

MAPS_ROOT = Path(__file__).resolve().parents[1] / "robot_sf" / "maps"
UNI_CAMPUS_SVG = MAPS_ROOT / "uni_campus_big.svg"


def test_repository_map_assets_are_svg_only() -> None:
    """The repository should not ship legacy JSON map assets."""
    json_maps = sorted(MAPS_ROOT.glob("*.json"))
    assert not json_maps, f"Legacy map assets remain: {json_maps}"


def test_svg_maps_are_loaded_and_json_files_are_ignored(tmp_path: Path) -> None:
    """Map folders should load SVG maps and ignore legacy JSON files.

    Also checks that a single SVG robot_route is automatically mirrored so the
    pool always exposes both directions, matching serialize_map semantics.
    """
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
    alpha = pool.map_defs["alpha"]

    assert list(pool.map_defs) == ["alpha"]
    assert alpha.width == pytest.approx(20.0)
    assert alpha.height == pytest.approx(10.0)
    # The SVG has one route (0→0); the symmetric-route normalisation must not
    # add a duplicate because spawn_id == goal_id means the reverse is the same
    # pair.  If the SVG had a 0→1 route, a 1→0 reverse would be added.
    route_pairs = {(r.spawn_id, r.goal_id) for r in alpha.robot_routes}
    assert (0, 0) in route_pairs


def test_json_only_map_folder_is_rejected(tmp_path: Path) -> None:
    """Folders without SVG maps should fail closed."""
    (tmp_path / "legacy.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        MapDefinitionPool(maps_folder=str(tmp_path))


def test_serialize_map_handles_core_map_contract() -> None:
    """Synthetic map data should still round-trip through the legacy serializer."""
    map_structure = {
        "x_margin": [0, 10],
        "y_margin": [0, 10],
        "obstacles": [
            [[1, 1], [3, 1], [3, 3], [1, 3]],
        ],
        "robot_spawn_zones": [[[0, 0], [1, 0], [1, 1]]],
        "robot_goal_zones": [[[8, 8], [9, 8], [9, 9]]],
        "robot_routes": [
            {"spawn_id": 0, "goal_id": 0, "waypoints": [[0.5, 0.5], [5.0, 0.5], [8.5, 8.5]]}
        ],
        "ped_spawn_zones": [[[2, 2], [3, 2], [3, 3]]],
        "ped_goal_zones": [[[6, 6], [7, 6], [7, 7]]],
        "ped_crowded_zones": [[[4, 4], [5, 4], [5, 5]]],
        "ped_routes": [
            {"spawn_id": 0, "goal_id": 0, "waypoints": [[2.5, 2.5], [4.0, 4.0], [6.5, 6.5]]}
        ],
        "single_pedestrians": [
            {
                "id": "ped1",
                "start": [1, 1],
                "goal": [2, 2],
                "wait_at": [{"waypoint_index": 0, "wait_s": 1.5, "note": "pause"}],
            }
        ],
    }

    map_def = serialize_map(map_structure)

    assert map_def.width == pytest.approx(10.0)
    assert map_def.height == pytest.approx(10.0)
    assert len(map_def.obstacles) == 1
    assert len(map_def.robot_routes) == 2
    assert len(map_def.ped_routes) == 1
    assert len(map_def.single_pedestrians) == 1


def test_asymmetric_svg_routes_are_mirrored(tmp_path: Path) -> None:
    """SVG maps with asymmetric routes get the missing reverse added by the loader."""
    (tmp_path / "asym.svg").write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="20" height="10" viewBox="0 0 20 10">
  <rect inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
  <rect inkscape:label="robot_spawn_zone_1" x="17" y="1" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone_0" x="1" y="7" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone_1" x="17" y="7" width="1" height="1" />
  <path inkscape:label="robot_route_0_1" d="M 1.5 1.5 L 10 5 L 17.5 7.5" />
</svg>
        """.strip(),
        encoding="utf-8",
    )

    pool = MapDefinitionPool(maps_folder=str(tmp_path))
    asym = pool.map_defs["asym"]

    route_pairs = {(r.spawn_id, r.goal_id) for r in asym.robot_routes}
    # SVG provides 0→1; loader must add the symmetric 1→0 reverse.
    assert (0, 1) in route_pairs
    assert (1, 0) in route_pairs
    assert len(asym.robot_routes) == 2


def test_migrated_svg_map_loads() -> None:
    """The migrated campus map should still parse successfully as SVG."""
    map_def = convert_map(str(UNI_CAMPUS_SVG))

    assert map_def is not None
    assert map_def.robot_routes
    assert map_def.ped_routes
    assert all(
        obstacle.geometry is not None and obstacle.geometry.is_valid
        for obstacle in map_def.obstacles
    )
