"""Focused contract tests for the matched-budget comparison packet (#6921).

These tests verify the versioned comparison packet is well-formed: required
fields are present, the budget mapping is non-vacuous, no residual bounds or
benchmark claim boundary are relaxed, exclusions are complete, and the
domain-approval gate is present.

This is a diagnostic-only specification slice: it makes no benchmark, metric,
planner-ranking, safety, or paper-facing claim and runs no campaign.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.ped_npc.residual_adversary import ResidualAdversaryConfig
from robot_sf.ped_npc.residual_search import ResidualSearchConfig

PACKET_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "adversarial"
    / "issue_6921_matched_compute_packet.yaml"
)


@pytest.fixture()
def packet() -> dict:
    """Load the checked-in comparison packet."""
    return yaml.safe_load(PACKET_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Schema and required fields
# ---------------------------------------------------------------------------


def test_packet_file_exists() -> None:
    """The config file must be present on disk."""
    assert PACKET_PATH.exists(), f"Packet not found: {PACKET_PATH}"


def test_packet_schema_version(packet: dict) -> None:
    """Schema version must be the frozen v1 identifier."""
    assert packet["schema_version"] == "matched_compute_packet.v1"


def test_packet_issue_number(packet: dict) -> None:
    """Issue number must be 6921."""
    assert packet["issue"] == 6921


def test_packet_parent_issue(packet: dict) -> None:
    """Parent issue must be 4360."""
    assert packet["parent_issue"] == 4360


def test_packet_status(packet: dict) -> None:
    """Status must be diagnostic_only_packet."""
    assert packet["status"] == "diagnostic_only_packet"


def test_packet_has_arms(packet: dict) -> None:
    """Both arms must be defined."""
    arms = packet["arms"]
    assert "open_loop" in arms
    assert "reactive" in arms


def test_packet_has_scenario(packet: dict) -> None:
    """Scenario section must be present."""
    assert "scenario" in packet
    assert "template" in packet["scenario"]
    assert "scenario_ids" in packet["scenario"]


def test_packet_has_simulation(packet: dict) -> None:
    """Simulation section must be present."""
    assert "simulation" in packet
    assert "dt_s" in packet["simulation"]
    assert "total_sim_steps" in packet["simulation"]


def test_packet_has_budget(packet: dict) -> None:
    """Budget section must be present."""
    assert "budget" in packet
    assert "total_candidate_evaluations" in packet["budget"]


def test_packet_has_seeds(packet: dict) -> None:
    """Seeds section must be present."""
    assert "seeds" in packet
    assert "frozen_scenario_seeds" in packet["seeds"]
    assert "search_seed" in packet["seeds"]


def test_packet_has_bounds_identity(packet: dict) -> None:
    """Bounds identity section must be present."""
    assert "bounds_identity" in packet
    assert "residual_bounds" in packet["bounds_identity"]
    assert "objective_proxy" in packet["bounds_identity"]


def test_packet_has_provenance(packet: dict) -> None:
    """Provenance section must be present."""
    assert "provenance" in packet
    assert "parent_issue" in packet["provenance"]


def test_packet_has_exclusions(packet: dict) -> None:
    """Explicit exclusions section must be present."""
    assert "explicit_exclusions" in packet


def test_packet_has_gate(packet: dict) -> None:
    """Domain-approval gate section must be present."""
    assert "gate" in packet
    assert packet["gate"]["domain_approval_required"] is True


def test_packet_has_validation_command(packet: dict) -> None:
    """Validation command must be present."""
    assert "validation_command" in packet
    assert "pytest" in packet["validation_command"]


# ---------------------------------------------------------------------------
# Budget mapping non-vacuity
# ---------------------------------------------------------------------------


def test_budget_total_is_positive(packet: dict) -> None:
    """Total candidate evaluations must be > 0."""
    assert packet["budget"]["total_candidate_evaluations"] > 0


def test_budget_per_arm_is_positive(packet: dict) -> None:
    """Per-arm candidate evaluations must be > 0."""
    assert packet["budget"]["per_arm_candidate_evaluations"] > 0


def test_budget_per_arm_matches_grid_resolution(packet: dict) -> None:
    """Per-arm budget must equal grid_points_per_dim ** 2 for both arms."""
    grid_points = packet["arms"]["open_loop"]["residual_search"]["grid_points_per_dim"]
    expected = grid_points**2
    assert packet["budget"]["per_arm_candidate_evaluations"] == expected
    grid_points_reactive = packet["arms"]["reactive"]["residual_search"]["grid_points_per_dim"]
    assert packet["budget"]["per_arm_candidate_evaluations"] == grid_points_reactive**2


# ---------------------------------------------------------------------------
# Bounds identity — not relaxed
# ---------------------------------------------------------------------------


def test_residual_bounds_match_adversary_config(packet: dict) -> None:
    """Frozen residual bounds must match the upstream adversary config."""
    bounds = packet["bounds_identity"]["residual_bounds"]
    adversary_cfg_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "adversarial"
        / "issue_4360_residual_adversary.yaml"
    )
    adversary_payload = yaml.safe_load(adversary_cfg_path.read_text(encoding="utf-8"))
    upstream = adversary_payload["residual_adversary"]

    assert bounds["max_residual_accel_mps2"] == upstream["max_residual_accel_mps2"]
    assert bounds["max_jerk_mps3"] == upstream["max_jerk_mps3"]
    assert bounds["max_speed_delta_mps"] == upstream["max_speed_delta_mps"]
    assert (
        bounds["max_heading_change_per_macro_rad"] == upstream["max_heading_change_per_macro_rad"]
    )
    assert bounds["max_route_deviation_m"] == upstream["max_route_deviation_m"]
    assert bounds["min_separation_m"] == upstream["min_separation_m"]


def test_bounds_identity_objective_proxy(packet: dict) -> None:
    """Objective proxy must be the diagnostic magnitude proxy."""
    assert packet["bounds_identity"]["objective_proxy"] == "maximize_residual_magnitude"


def test_arms_share_identical_residual_bounds(packet: dict) -> None:
    """Both arms must use the same residual bounds."""
    open_loop_bounds = packet["arms"]["open_loop"]["residual_adversary"]
    reactive_bounds = packet["arms"]["reactive"]["residual_adversary"]
    for key in (
        "max_residual_accel_mps2",
        "max_jerk_mps3",
        "max_speed_delta_mps",
        "max_heading_change_per_macro_rad",
        "max_route_deviation_m",
        "min_separation_m",
    ):
        assert open_loop_bounds[key] == reactive_bounds[key], f"bound mismatch: {key}"


# ---------------------------------------------------------------------------
# Exclusion completeness
# ---------------------------------------------------------------------------


def test_exclusions_forbid_benchmark(packet: dict) -> None:
    """Benchmark execution must be explicitly forbidden."""
    exclusions = packet["explicit_exclusions"]
    assert exclusions["benchmark_execution"] == "forbidden_until_domain_approval"


def test_exclusions_forbid_slurm(packet: dict) -> None:
    """SLURM campaigns must be explicitly forbidden."""
    exclusions = packet["explicit_exclusions"]
    assert exclusions["slurm_campaign"] == "forbidden_until_domain_approval"


def test_exclusions_forbid_paper_claims(packet: dict) -> None:
    """Paper-facing claims must be explicitly forbidden."""
    exclusions = packet["explicit_exclusions"]
    assert exclusions["paper_facing_claims"] == "forbidden"


def test_exclusions_forbid_metric_claims(packet: dict) -> None:
    """Metric/ranking claims must be explicitly forbidden."""
    exclusions = packet["explicit_exclusions"]
    assert exclusions["metric_or_ranking_claims"] == "forbidden"


def test_exclusions_confirm_no_new_optimizer(packet: dict) -> None:
    """No new optimizer must be introduced."""
    exclusions = packet["explicit_exclusions"]
    assert exclusions["new_optimizer"] == "not_added"


def test_exclusions_confirm_no_new_planner(packet: dict) -> None:
    """No new planner integration must be introduced."""
    exclusions = packet["explicit_exclusions"]
    assert exclusions["new_planner_integration"] == "not_added"


def test_exclusions_confirm_fallback_exclusion(packet: dict) -> None:
    """Fallback/degraded evidence must not be treated as success."""
    exclusions = packet["explicit_exclusions"]
    assert exclusions["fallback_degraded_evidence"] == "not_treated_as_success"


# ---------------------------------------------------------------------------
# Domain-approval gate
# ---------------------------------------------------------------------------


def test_domain_approval_gate_requires_explicit_approval(packet: dict) -> None:
    """The gate must require explicit domain approval before any execution."""
    gate = packet["gate"]
    assert gate["domain_approval_required"] is True


# ---------------------------------------------------------------------------
# Config round-trip: packet fields must be parseable by upstream configs
# ---------------------------------------------------------------------------


def test_open_loop_residual_search_config_round_trip(packet: dict) -> None:
    """The open-loop arm's search config must parse as a valid ResidualSearchConfig."""
    rs = packet["arms"]["open_loop"]["residual_search"]
    config = ResidualSearchConfig(
        algorithm_name=rs["algorithm_name"],
        objective_proxy=rs["objective_proxy"],
        grid_points_per_dim=rs["grid_points_per_dim"],
        max_candidates=rs["max_candidates"],
    )
    assert config.algorithm_name == "finite_grid_search_v1"
    assert config.grid_points_per_dim == 3
    assert config.max_candidates == 9


def test_reactive_residual_search_config_round_trip(packet: dict) -> None:
    """The reactive arm's search config must parse as a valid ResidualSearchConfig."""
    rs = packet["arms"]["reactive"]["residual_search"]
    config = ResidualSearchConfig(
        algorithm_name=rs["algorithm_name"],
        objective_proxy=rs["objective_proxy"],
        grid_points_per_dim=rs["grid_points_per_dim"],
        max_candidates=rs["max_candidates"],
    )
    assert config.algorithm_name == "finite_grid_search_v1"
    assert config.grid_points_per_dim == 3
    assert config.max_candidates == 9


def test_residual_adversary_config_round_trip(packet: dict) -> None:
    """Both arms' residual adversary configs must parse as valid ResidualAdversaryConfig."""
    for arm_key in ("open_loop", "reactive"):
        ra = packet["arms"][arm_key]["residual_adversary"]
        config = ResidualAdversaryConfig(**ra)
        assert config.is_active is True
        assert config.macro_action_dt_s == 0.5
        assert config.max_residual_accel_mps2 == 1.5
        assert config.max_jerk_mps3 == 7.5


# ---------------------------------------------------------------------------
# Scenario frozen IDs
# ---------------------------------------------------------------------------


def test_scenario_ids_are_frozen(packet: dict) -> None:
    """Scenario IDs must be explicitly listed and non-empty."""
    ids = packet["scenario"]["scenario_ids"]
    assert isinstance(ids, list)
    assert len(ids) >= 1
    assert all(isinstance(s, str) for s in ids)


def test_frozen_scenario_seeds(packet: dict) -> None:
    """Scenario seeds must be frozen and non-empty."""
    seeds = packet["seeds"]["frozen_scenario_seeds"]
    assert isinstance(seeds, list)
    assert len(seeds) >= 1
    assert all(isinstance(s, int) for s in seeds)


def test_search_seed_is_frozen(packet: dict) -> None:
    """Search seed must be a frozen integer."""
    assert isinstance(packet["seeds"]["search_seed"], int)


def test_residual_adversary_seed_is_frozen(packet: dict) -> None:
    """Residual adversary seed must be a frozen integer."""
    assert isinstance(packet["seeds"]["residual_adversary_seed"], int)
