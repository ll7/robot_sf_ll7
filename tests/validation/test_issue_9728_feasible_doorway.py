"""The #9728 successor is feasible without changing the historical doorway."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from robot_sf.benchmark.narrow_doorway_geometry_family import generate_variant_map
from robot_sf.benchmark.narrow_doorway_radius_audit import derive_doorway_geometry
from robot_sf.benchmark.spawn_preflight import (
    DEFAULT_CLEARANCE_MARGIN_M,
    DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_RESPAWN_WINDOW_STEPS,
    _check_release_scenario,
)
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_MAP = ROOT / "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg"
H400_MANIFEST = ROOT / "configs/benchmarks/issue_9348_three_width_doorway_v1.yaml"
SOURCE_MAP = (
    ROOT / "maps/successor_svg_maps/issue_9762_francis2023_narrow_doorway_goal_zone_entry_v2.svg"
)
SUCCESSOR_MAP = (
    ROOT / "maps/successor_svg_maps/issue_9728_francis2023_narrow_doorway_feasible_3p60_v1.svg"
)
BASE_MATRIX = ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3.yaml"
SUCCESSOR_MATRIX = (
    ROOT
    / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3_feasible_doorway_v1.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_versioned_map_matches_width_generator_and_preserves_preregistered_inputs(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated.svg"
    generate_variant_map(
        SOURCE_MAP,
        gap_width_m=3.6,
        constriction_depth_m=1.0,
        output_path=generated,
    )
    assert generated.read_bytes() == SUCCESSOR_MAP.read_bytes()
    assert _sha256(HISTORICAL_MAP) == (
        "7538ed173d462a5107afc1a1e43b5b2e6d2bc5c9604035cdec9a551e20a8b15e"
    )
    assert _sha256(H400_MANIFEST) == (
        "de828b9132fff186a537dff2427681c9117f149fc8f9aab1156a754abd25b906"
    )


def test_successor_changes_one_map_without_changing_48_scenario_identities() -> None:
    baseline = [dict(row) for row in load_scenarios(BASE_MATRIX)]
    successor = [dict(row) for row in load_scenarios(SUCCESSOR_MATRIX)]
    assert len(baseline) == len(successor) == 48
    assert [row["name"] for row in baseline] == [row["name"] for row in successor]
    changed = []
    for old, new in zip(baseline, successor, strict=True):
        if old != new:
            changed.append(old["name"])
            assert new["map_id"] is None
            assert new["map_file"].endswith(SUCCESSOR_MAP.name)
            assert {key: value for key, value in old.items() if key != "map_file"} == {
                key: value for key, value in new.items() if key != "map_file"
            }
    assert changed == ["francis2023_narrow_doorway"]

    doorway = next(row for row in successor if row["name"] == "francis2023_narrow_doorway")
    geometry = derive_doorway_geometry(SUCCESSOR_MATRIX, doorway)
    assert geometry.gap_width_m == pytest.approx(3.6)
    assert geometry.gap_lower_edge_m == pytest.approx(3.2)
    assert geometry.gap_upper_edge_m == pytest.approx(6.8)
    assert geometry.route_waypoints == ((7.0, 5.0), (27.0, 5.0))
    assert geometry.route_min_center_distance_m == pytest.approx(1.8)
    assert geometry.route_min_center_distance_m >= 1.0 + DEFAULT_CLEARANCE_MARGIN_M


def test_dev_seed_1001_ordered_route_and_stationary_reset_pass_setup_preflight() -> None:
    doorway = next(
        row
        for row in load_scenarios(SUCCESSOR_MATRIX)
        if row["name"] == "francis2023_narrow_doorway"
    )
    result = _check_release_scenario(
        (
            dict(doorway),
            str(SUCCESSOR_MATRIX),
            (1001,),
            DEFAULT_CLEARANCE_MARGIN_M,
            DEFAULT_RESPAWN_WINDOW_STEPS,
            DEFAULT_GRID_RESOLUTION_M,
            False,
        )
    )
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["overall_status"] == "valid", row
    for check in ("reset_clearance", "footprint_reachability", "passage_width", "respawn_safety"):
        assert row[check]["status"] == "pass", row
    assert row["footprint_reachability"]["route_waypoint_count"] == 3
    assert row["footprint_reachability"]["first_blocked_segment_index"] is None
    assert row["passage_width"]["minimum_opening_width_estimate_m"] >= 2.2
