"""Tests for the opt-in spawn-sampler capture object (issue #9312).

Proves the capture records sampler decisions and route assignments on a fixed
seed, serializes to JSON-safe mappings, and leaves the default path unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_sf.nav.navigation import get_prepared_obstacles
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.ped_npc.ped_population import (
    PedSpawnConfig,
    populate_simulation,
)
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
