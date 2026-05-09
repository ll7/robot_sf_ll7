"""Tests for the confirmation_v1 scenario matrix."""

from __future__ import annotations

from pathlib import Path

from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
CONFIRMATION_MATRIX = ROOT / "configs" / "scenarios" / "confirmation_v1.yaml"


def test_confirmation_v1_matrix_loads_expected_distinct_scenarios() -> None:
    """Confirmation v1 should expose the curated eight-scenario robustness set."""
    scenarios = load_scenarios(CONFIRMATION_MATRIX, base_dir=CONFIRMATION_MATRIX)
    names = [scenario["name"] for scenario in scenarios]
    assert names == [
        "empty_map_8_directions_east",
        "corner_90_turn",
        "u_trap_local_minimum",
        "single_obstacle_circle",
        "narrow_passage",
        "symmetry_ambiguous_choice",
        "single_ped_crossing_orthogonal",
        "head_on_interaction",
    ]
    assert len({scenario["metadata"]["primary_capability"] for scenario in scenarios}) >= 5
    assert all(len(scenario.get("seeds", [])) >= 3 for scenario in scenarios)


def test_confirmation_v1_matrix_has_design_rationale_metadata() -> None:
    """Each confirmation scenario should carry provenance and interpretation metadata."""
    scenarios = load_scenarios(CONFIRMATION_MATRIX, base_dir=CONFIRMATION_MATRIX)
    for scenario in scenarios:
        metadata = scenario["metadata"]
        assert metadata.get("purpose")
        assert metadata.get("expected_behavior")
        assert metadata.get("expected_pass_criteria")
        assert metadata.get("primary_capability")
        assert metadata.get("target_failure_mode")
