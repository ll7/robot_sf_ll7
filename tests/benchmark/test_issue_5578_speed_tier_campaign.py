"""Unit tests for issue #5578 / #6101 speed-tier campaign compilation and preflight."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.issue_5578_speed_tier_synthesis import parse_cell
from scripts.benchmark.run_issue_5578_speed_tier_campaign import (
    EXPECTED_PLANNERS,
    EXPECTED_SCENARIOS,
    EXPECTED_SEEDS,
    EXPECTED_TIERS,
    TIER_ACTUATION_ENVELOPES,
    build_campaign_manifest,
    run_preflight_campaign,
    run_preflight_episode,
    validate_manifest,
)
from scripts.validation.check_issue_5578_robot_speed_tier_preregistration import (
    load_preregistration,
)


def test_manifest_compilation_contains_2160_cells_and_no_duplicates() -> None:
    """Verify that campaign manifest materializes exactly 2,160 unique registered cells."""
    manifest = build_campaign_manifest()

    assert len(manifest) == 2160

    seen = set()
    scenarios = set()
    tiers = set()
    planners = set()
    seeds = set()
    for cell in manifest:
        key = (cell["scenario_id"], cell["speed_tier_id"], cell["planner_id"], cell["seed"])
        assert key not in seen, f"Duplicate cell in manifest: {key}"
        seen.add(key)
        scenarios.add(cell["scenario_id"])
        tiers.add(cell["speed_tier_id"])
        planners.add(cell["planner_id"])
        seeds.add(cell["seed"])
        assert 111 <= cell["seed"] <= 140

    assert len(scenarios) == len(EXPECTED_SCENARIOS)
    assert len(tiers) == len(EXPECTED_TIERS)
    assert len(planners) == len(EXPECTED_PLANNERS)
    assert seeds == set(EXPECTED_SEEDS)


def test_check_only_mode_runs_without_side_effects(tmp_path: Path) -> None:
    """Verify check-only manifest validation completes without side effects."""
    out_file = tmp_path / "test_manifest.json"
    manifest = build_campaign_manifest()

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    validate_manifest(manifest)
    assert out_file.is_file()


def test_preflight_rejects_registered_seeds_111_to_140() -> None:
    """Verify preflight rejects registered seeds 111-140."""
    with pytest.raises(ValueError, match="Preflight seed 111 overlaps with registered seed range"):
        run_preflight_campaign(preflight_seeds=[111, 112])

    with pytest.raises(ValueError, match="Preflight seed 140 overlaps with registered seed range"):
        run_preflight_campaign(preflight_seeds=[140])

    with pytest.raises(ValueError, match="Preflight cannot be run on registered seed 125"):
        run_preflight_episode(
            scenario_id="classic_head_on_corridor_medium",
            speed_tier_id="cap_4_0",
            speed_cap_m_s=4.0,
            planner_id="goal_seek",
            seed=125,
        )


def test_disjoint_seed_preflight_activates_intervention() -> None:
    """Verify disjoint-seed preflight activates the speed-cap intervention."""
    report = run_preflight_campaign(
        preflight_seeds=[901, 902],
        scenarios=["classic_head_on_corridor_medium"],
        planners=["goal_seek"],
        horizon_steps=30,
    )

    assert report["activation_gate_passed"] is True
    assert report["disjoint_seeds"] == [901, 902]

    activations = report["tier_activations"]
    assert activations["cap_2_0_nominal"]["activated"] is True
    assert activations["cap_3_0"]["activated"] is True
    assert activations["cap_4_0"]["activated"] is True


def test_actuation_envelopes_match_preregistration() -> None:
    """Verify solved 4.0 m/s actuation envelopes match amended preregistration."""
    prereg = load_preregistration()
    tier_specs = {t["tier_id"]: t for t in prereg["robot_speed_axis"]["tiers"]}

    for tier_id, spec in tier_specs.items():
        local_env = TIER_ACTUATION_ENVELOPES[tier_id]
        assert local_env["max_forward_accel_m_s2"] == spec["max_accel_m_s2"]
        assert local_env["max_braking_decel_m_s2"] == spec["max_decel_m_s2"]
        assert local_env["peak_forward_speed_m_s"] == spec["cap_m_s"]
        assert local_env["stopping_distance_envelope_m"] == spec["stopping_distance_envelope_m"]

    # Verify cap_4_0 micromobility tier values from #6100
    cap_4_0 = TIER_ACTUATION_ENVELOPES["cap_4_0"]
    assert cap_4_0["peak_forward_speed_m_s"] == 4.0
    assert cap_4_0["max_forward_accel_m_s2"] == 2.0
    assert cap_4_0["max_braking_decel_m_s2"] == 4.0
    assert cap_4_0["stopping_distance_envelope_m"] == 2.0


def test_cell_summaries_are_compatible_with_synthesis() -> None:
    """Verify episode cell output structures are directly parsable by synthesis."""
    row_nominal = run_preflight_episode(
        scenario_id="classic_head_on_corridor_medium",
        speed_tier_id="cap_2_0_nominal",
        speed_cap_m_s=2.0,
        planner_id="scenario_adaptive_hybrid_orca_v2_collision_guard",
        seed=901,
        horizon_steps=10,
    )

    cell = parse_cell(row_nominal)
    assert cell.scenario_id == "classic_head_on_corridor_medium"
    assert cell.speed_tier_id == "cap_2_0_nominal"
    assert cell.planner_id == "scenario_adaptive_hybrid_orca_v2_collision_guard"
    assert cell.speed_cap_m_s == 2.0
