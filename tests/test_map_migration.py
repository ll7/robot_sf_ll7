"""Tests for the JSON-to-SVG map migration path and loader fallback behavior."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from robot_sf.nav.map_config import MapDefinitionPool, serialize_map
from robot_sf.nav.svg_map_parser import convert_map

FIXTURE_ROOT = Path(__file__).resolve().parents[1]
UNI_CAMPUS_JSON = FIXTURE_ROOT / "robot_sf" / "maps" / "uni_campus_big.json"
UNI_CAMPUS_SVG = FIXTURE_ROOT / "robot_sf" / "maps" / "uni_campus_big.svg"


def _round_point(point: tuple[float, float]) -> tuple[float, float]:
    return (round(point[0], 3), round(point[1], 3))


def _collapse_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    collapsed: list[tuple[float, float]] = []
    for point in points:
        if not collapsed or point != collapsed[-1]:
            collapsed.append(point)
    if len(collapsed) > 1 and collapsed[0] == collapsed[-1]:
        collapsed.pop()
    return collapsed


def _cyclic_signature(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    collapsed = _collapse_points([_round_point(point) for point in points])
    if not collapsed:
        return ()

    def _rotations(sequence: list[tuple[float, float]]) -> list[tuple[tuple[float, float], ...]]:
        return [tuple(sequence[index:] + sequence[:index]) for index in range(len(sequence))]

    candidates = []
    for sequence in (collapsed, list(reversed(collapsed))):
        candidates.append(min(_rotations(sequence)))
    return min(candidates)


def _bbox_signature(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return tuple(round(value, 3) for value in (min(xs), min(ys), max(xs), max(ys)))


def _route_signature(route) -> tuple[int, int, tuple[tuple[float, float], ...]]:
    waypoints = _collapse_points([_round_point(point) for point in route.waypoints])
    return route.spawn_id, route.goal_id, tuple(waypoints)


def _obstacle_area(obstacle) -> float:
    points = _collapse_points([_round_point(point) for point in obstacle.vertices])
    if len(points) < 3:
        return 0.0
    return Polygon(points).area


def _zone_signature(zone) -> tuple[float, float, float, float]:
    return _bbox_signature(list(zone))


def _single_ped_signature(
    ped,
) -> tuple[str, tuple[float, float], tuple[float, float] | None, tuple[tuple[float, float], ...]]:
    start = _round_point(ped.start)
    goal = _round_point(ped.goal) if ped.goal is not None else None
    trajectory = tuple(_collapse_points([_round_point(point) for point in ped.trajectory or []]))
    return ped.id, start, goal, trajectory


def _assert_semantic_parity(legacy_map, svg_map) -> None:
    assert legacy_map.width == pytest.approx(svg_map.width)
    assert legacy_map.height == pytest.approx(svg_map.height)
    assert len(legacy_map.obstacles) == len(svg_map.obstacles)
    legacy_obstacle_area = sum(_obstacle_area(obstacle) for obstacle in legacy_map.obstacles)
    svg_obstacle_area = sum(_obstacle_area(obstacle) for obstacle in svg_map.obstacles)
    assert legacy_obstacle_area == pytest.approx(svg_obstacle_area, abs=1e-2)
    for attr in [
        "robot_spawn_zones",
        "robot_goal_zones",
        "ped_spawn_zones",
        "ped_goal_zones",
        "ped_crowded_zones",
    ]:
        assert Counter(_zone_signature(zone) for zone in getattr(legacy_map, attr)) == Counter(
            _zone_signature(zone) for zone in getattr(svg_map, attr)
        )
    assert Counter(_route_signature(route) for route in legacy_map.robot_routes) == Counter(
        _route_signature(route) for route in svg_map.robot_routes
    )
    assert Counter(_route_signature(route) for route in legacy_map.ped_routes) == Counter(
        _route_signature(route) for route in svg_map.ped_routes
    )
    assert Counter(_single_ped_signature(ped) for ped in legacy_map.single_pedestrians) == Counter(
        _single_ped_signature(ped) for ped in svg_map.single_pedestrians
    )


def test_migrated_svg_map_matches_legacy_json_semantics() -> None:
    """The migrated SVG should match the legacy JSON map semantically."""
    legacy_map = serialize_map(json.loads(UNI_CAMPUS_JSON.read_text(encoding="utf-8")))
    svg_map = convert_map(str(UNI_CAMPUS_SVG))

    assert svg_map is not None
    _assert_semantic_parity(legacy_map, svg_map)


def test_map_definition_pool_prefers_svg_and_keeps_deterministic_order(tmp_path: Path) -> None:
    """SVG files should win over JSON, while folder order stays deterministic."""
    alpha_json = {
        "x_margin": [0, 10],
        "y_margin": [0, 10],
        "obstacles": [],
        "robot_spawn_zones": [[[1, 1], [2, 1], [2, 2]]],
        "robot_goal_zones": [[[7, 1], [8, 1], [8, 2]]],
        "robot_routes": [{"spawn_id": 0, "goal_id": 0, "waypoints": [[1, 1], [5, 1], [8, 1]]}],
        "ped_spawn_zones": [],
        "ped_goal_zones": [],
        "ped_crowded_zones": [],
        "ped_routes": [],
    }
    beta_json = {
        "x_margin": [0, 30],
        "y_margin": [0, 10],
        "obstacles": [],
        "robot_spawn_zones": [[[1, 1], [2, 1], [2, 2]]],
        "robot_goal_zones": [[[27, 1], [28, 1], [28, 2]]],
        "robot_routes": [{"spawn_id": 0, "goal_id": 0, "waypoints": [[1, 1], [15, 1], [28, 1]]}],
        "ped_spawn_zones": [],
        "ped_goal_zones": [],
        "ped_crowded_zones": [],
        "ped_routes": [],
    }

    (tmp_path / "alpha.json").write_text(json.dumps(alpha_json), encoding="utf-8")
    (tmp_path / "beta.json").write_text(json.dumps(beta_json), encoding="utf-8")
    (tmp_path / "beta.svg").write_text(
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

    assert list(pool.map_defs) == ["alpha", "beta"]
    assert pool.map_defs["alpha"].width == pytest.approx(10.0)
    assert pool.map_defs["beta"].width == pytest.approx(20.0)
    assert pool.map_defs["beta"].height == pytest.approx(10.0)


def test_map_definition_pool_falls_back_to_json_when_svg_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed SVG load should preserve legacy JSON behavior when possible."""
    gamma_json = {
        "x_margin": [0, 33],
        "y_margin": [0, 11],
        "obstacles": [],
        "robot_spawn_zones": [[[1, 1], [2, 1], [2, 2]]],
        "robot_goal_zones": [[[30, 1], [31, 1], [31, 2]]],
        "robot_routes": [{"spawn_id": 0, "goal_id": 0, "waypoints": [[1, 1], [16, 1], [31, 1]]}],
        "ped_spawn_zones": [],
        "ped_goal_zones": [],
        "ped_crowded_zones": [],
        "ped_routes": [],
    }

    (tmp_path / "gamma.json").write_text(json.dumps(gamma_json), encoding="utf-8")
    (tmp_path / "gamma.svg").write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="44" height="11" viewBox="0 0 44 11">
  <rect inkscape:label="robot_spawn_zone" x="1" y="1" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone" x="40" y="1" width="1" height="1" />
  <path inkscape:label="robot_route_0_0" d="M 1 1 L 22 1 L 41 1" />
</svg>
        """.strip(),
        encoding="utf-8",
    )

    from robot_sf.nav.svg_map_parser import convert_map as real_convert_map

    def fake_convert_map(svg_path: str):
        if svg_path.endswith("gamma.svg"):
            return None
        return real_convert_map(svg_path)

    monkeypatch.setattr("robot_sf.nav.svg_map_parser.convert_map", fake_convert_map)

    pool = MapDefinitionPool(maps_folder=str(tmp_path))

    assert list(pool.map_defs) == ["gamma"]
    assert pool.map_defs["gamma"].width == pytest.approx(33.0)
    assert pool.map_defs["gamma"].height == pytest.approx(11.0)
