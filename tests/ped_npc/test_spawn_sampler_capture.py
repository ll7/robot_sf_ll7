"""Tests for the opt-in spawn-sampler capture object (issue #9312).

Proves the capture records sampler decisions and route assignments on a fixed
seed, serializes to JSON-safe mappings, and leaves the default path unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.navigation import get_prepared_obstacles
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.ped_npc.ped_population import (
    PedSpawnConfig,
    _sample_points_near_anchor,
    _sample_scatter_point,
    populate_simulation,
    sample_route,
)
from robot_sf.ped_npc.ped_zone import prepare_obstacle_polygons
from robot_sf.ped_npc.spawn_capture import SpawnSamplerCapture

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIC_CROSSING_SVG = _REPO_ROOT / "maps" / "svg_maps" / "classic_crossing.svg"


def _crossing_map():
    """Load classic_crossing map definition, or skip if unavailable."""
    import pytest

    if not _CLASSIC_CROSSING_SVG.exists():
        pytest.skip("classic_crossing.svg not available")
    md = SvgMapConverter(str(_CLASSIC_CROSSING_SVG)).get_map_definition()
    if not md.ped_routes:
        pytest.skip("classic_crossing has no pedestrian routes")
    return md


def _spawn_config() -> PedSpawnConfig:
    """Deterministic spread-distribution spawn config for capture tests."""
    return PedSpawnConfig(
        peds_per_area_m2=0.08,
        max_group_members=3,
        initial_speed=0.5,
        route_spawn_distribution="spread",
        route_spawn_seed=123,
    )


def _populate(md, config, capture: SpawnSamplerCapture | None):
    """Run populate_simulation with explicit capture routing."""
    return populate_simulation(
        0.5,
        config,
        md.ped_routes,
        md.ped_crowded_zones,
        obstacle_polygons=get_prepared_obstacles(md),
        sampler_capture=capture,
    )


def test_capture_records_assignments_and_decisions_on_fixed_seed() -> None:
    """A fixed-seed run records assignments, samples, and JSON-safe mappings."""
    md = _crossing_map()
    capture = SpawnSamplerCapture()
    pysf_state, _groups, _behaviors = _populate(md, _spawn_config(), capture)

    assert pysf_state.num_peds > 0
    assert capture.accepted_samples > 0
    assert capture.assigned_routes, "route groups must record their assignments"
    for record in capture.assigned_routes:
        assert record.waypoints, "each assignment must carry waypoints"
        assert record.initial_section >= 0
    mapping = capture.to_mapping()
    json.dumps(mapping)  # must be JSON-serializable for trace embedding
    assert mapping["accepted_samples"] == capture.accepted_samples
    assert len(mapping["assigned_routes"]) == len(capture.assigned_routes)


def test_capture_is_deterministic_for_fixed_seed() -> None:
    """Two fixed-seed runs produce identical capture payloads."""
    md = _crossing_map()
    first = SpawnSamplerCapture()
    _populate(md, _spawn_config(), first)
    second = SpawnSamplerCapture()
    _populate(md, _spawn_config(), second)
    assert first.to_mapping() == second.to_mapping()


def test_capture_disabled_by_default_leaves_population_unchanged() -> None:
    """Omitting the capture object preserves the exact default population."""
    md = _crossing_map()
    capture = SpawnSamplerCapture()
    captured_state, _, _ = _populate(md, _spawn_config(), capture)
    default_state, _, _ = _populate(md, _spawn_config(), None)
    assert np.array_equal(captured_state.pysf_states(), default_state.pysf_states()), (
        "capture must not change spawned state"
    )
    fresh = SpawnSamplerCapture()
    assert fresh.to_mapping()["assigned_routes"] == []
    assert fresh.route_anchor_attempts == 0


def _covering_obstacle() -> list:
    """Return prepared obstacles covering a wide area around the origin."""
    square = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
    return prepare_obstacle_polygons([square])


def test_route_point_obstacle_rejections_recorded() -> None:
    """Every obstacle-rejected route-point draw is counted deterministically."""
    capture = SpawnSamplerCapture()
    samples = _sample_points_near_anchor(
        (0.0, 0.0),
        3,
        1.0,
        np.random.default_rng(7),
        _covering_obstacle(),
        capture=capture,
    )
    assert samples == []
    assert capture.obstacle_rejections == 3 * 50
    assert capture.accepted_samples == 0


def test_route_anchor_failure_records_attempts() -> None:
    """An unplaceable route records exactly max_anchor_attempts plus one failure."""
    route = GlobalRoute(
        spawn_id=0,
        goal_id=1,
        waypoints=[(0.0, 0.0), (0.1, 0.0)],
        spawn_zone=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        goal_zone=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    capture = SpawnSamplerCapture()
    with pytest.raises(RuntimeError, match="Failed to sample"):
        sample_route(
            route,
            3,
            1.0,
            obstacle_polygons=_covering_obstacle(),
            rng=np.random.default_rng(11),
            capture=capture,
        )
    assert capture.route_anchor_attempts == 5
    assert capture.route_anchor_failures == 1


def test_scatter_separation_and_exclusion_rejections_recorded() -> None:
    """Scatter separation and exclusion rejections are counted by branch."""
    tiny_zone = ((0.0, 0.0), (0.01, 0.0), (0.0, 0.01))
    separated = SpawnSamplerCapture()
    with pytest.raises(RuntimeError, match="Failed to scatter-spawn"):
        _sample_scatter_point(
            tiny_zone,
            np.random.default_rng(13),
            [],
            [(0.0, 0.0)],
            0.4,
            max_attempts=4,
            capture=separated,
        )
    assert separated.separation_rejections == 4
    assert separated.obstacle_rejections == 0
    assert separated.accepted_samples == 0

    excluded = SpawnSamplerCapture()
    with pytest.raises(RuntimeError, match="Failed to scatter-spawn"):
        _sample_scatter_point(
            tiny_zone,
            np.random.default_rng(17),
            _covering_obstacle(),
            [],
            0.4,
            max_attempts=4,
            capture=excluded,
        )
    assert excluded.obstacle_rejections == 4
    assert excluded.separation_rejections == 0

    accepted = SpawnSamplerCapture()
    point = _sample_scatter_point(
        tiny_zone, np.random.default_rng(19), [], [], 0.4, capture=accepted
    )
    assert accepted.accepted_samples == 1
    assert point is not None
