"""Focused contract tests for the matched-budget comparison packet (#6921).

These tests verify the versioned comparison packet is well-formed: required
fields are present, the budget arithmetic is consistent and derived from
simulation geometry, no residual bounds or benchmark claim boundary are
relaxed, exclusions are complete, provenance paths resolve, both arms'
runner bindings and trace schema are present, seeds are frozen and match
across arms, and the domain-approval gate is present.

This is a diagnostic-only specification slice: it makes no benchmark, metric,
planner-ranking, safety, or paper-facing claim and runs no campaign.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.config import SearchConfig, SearchSpaceConfig
from robot_sf.ped_npc.residual_adversary import ResidualAdversaryConfig
from robot_sf.ped_npc.residual_search import ResidualSearchConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = REPO_ROOT / "configs" / "adversarial" / "issue_6921_matched_compute_packet.yaml"


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
    """Schema version must identify the revised v2 preflight packet."""
    assert packet["schema_version"] == "matched_compute_packet.v2"
    assert packet["previous_schema_version"] == "matched_compute_packet.v1"


def test_packet_issue_number(packet: dict) -> None:
    """Issue number must be 6921."""
    assert packet["issue"] == 6921


def test_packet_parent_issue(packet: dict) -> None:
    """Parent issue must be 4360."""
    assert packet["parent_issue"] == 4360


def test_packet_status(packet: dict) -> None:
    """Status must remain diagnostic-only preflight."""
    assert packet["status"] == "diagnostic_only_preflight"
    assert packet["preflight_evidence_status"] == "diagnostic_only_preflight"


def test_packet_has_arms(packet: dict) -> None:
    """Both arms must be defined."""
    arms = packet["arms"]
    assert "open_loop" in arms
    assert "reactive" in arms


def test_packet_has_shared_trace_schema(packet: dict) -> None:
    """Both arms must publish the same trace schema identifier."""
    assert packet["packet"]["trace_schema"] == "matched_compute_trace.v1"
    for arm in packet["arms"].values():
        assert arm["runner_binding"]["trace_schema"] == "matched_compute_trace.v1"
    assert "evidence_status" in packet["budget"]["shared_trace_fields"]
    assert "simulator_steps_source" in packet["budget"]["shared_trace_fields"]


def test_packet_binds_native_runner_seams(packet: dict) -> None:
    """The packet must name the production seams, not the former stand-in."""
    open_loop = packet["arms"]["open_loop"]["runner_binding"]
    assert open_loop["runner"] == "robot_sf.adversarial.search.run_adversarial_search"
    assert (
        open_loop["production_evaluator"]
        == "robot_sf.adversarial.search.production_candidate_evaluator"
    )
    assert open_loop["policy"] == "social_force"
    assert open_loop["objective"] == "minimize_episode_min_robot_distance"
    assert open_loop["horizon_steps"] == packet["simulation"]["total_sim_steps"]
    assert open_loop["dt_s"] == packet["simulation"]["dt_s"]
    assert open_loop["budget"] == packet["budget"]["candidates_per_arm_per_episode"]
    assert open_loop["search_seed"] == 42
    assert open_loop["execution_mode"] == "native"

    reactive = packet["arms"]["reactive"]["runner_binding"]
    assert reactive["search_policy"] == "robot_sf.ped_npc.residual_search.FiniteGridSearchPolicy"
    assert reactive["controller"] == "robot_sf.ped_npc.residual_adversary.BoundedResidualAdversary"
    assert reactive["execution_mode"] == "native"


def test_packet_has_scenario(packet: dict) -> None:
    """Scenario section must reference the checked-in template."""
    assert "scenario" in packet
    assert "template" in packet["scenario"]
    assert "template_id" in packet["scenario"]
    assert packet["scenario"]["template_id"] == "crossing_ttc_template"
    assert "template_seed" in packet["scenario"]
    assert packet["scenario"]["template_seed"] == 123


def test_packet_has_simulation(packet: dict) -> None:
    """Simulation section must be present."""
    assert "simulation" in packet
    assert "dt_s" in packet["simulation"]
    assert "total_sim_steps" in packet["simulation"]
    assert "physics_steps_per_macro_action" in packet["simulation"]


def test_packet_has_budget(packet: dict) -> None:
    """Budget section must have explicit derived fields."""
    assert "budget" in packet
    budget = packet["budget"]
    assert "macro_actions_per_episode" in budget
    assert "candidates_per_macro_action_per_arm" in budget
    assert "candidates_per_arm_per_episode" in budget
    assert "total_candidates_all_arms_per_episode" in budget
    assert budget["open_loop_runner_budget_per_episode"] == 90
    assert budget["shared_trace_fields"]


def test_packet_has_seeds(packet: dict) -> None:
    """Seeds section must be present."""
    assert "seeds" in packet
    assert "frozen_scenario_seeds" in packet["seeds"]
    assert "search_seed" in packet["seeds"]
    assert "open_loop_search_seed" in packet["seeds"]
    assert "residual_adversary_seed" in packet["seeds"]


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


def test_packet_has_target_ped_idx_note(packet: dict) -> None:
    """A target_ped_idx_note must document the single-target choice."""
    assert "target_ped_idx_note" in packet
    assert "single-target" in packet["target_ped_idx_note"]


# ---------------------------------------------------------------------------
# Budget arithmetic — derived from simulation geometry
# ---------------------------------------------------------------------------


def test_budget_macro_actions_per_episode(packet: dict) -> None:
    """Macro-actions per episode must equal total_sim_steps / physics_steps_per_macro_action."""
    sim = packet["simulation"]
    budget = packet["budget"]
    expected = sim["total_sim_steps"] // sim["physics_steps_per_macro_action"]
    assert budget["macro_actions_per_episode"] == expected
    assert expected == 10


def test_budget_candidates_per_macro_action(packet: dict) -> None:
    """Reactive candidates per macro-action must equal grid_points_per_dim ** 2."""
    budget = packet["budget"]
    grid_pts = packet["arms"]["reactive"]["residual_search"]["grid_points_per_dim"]
    expected = grid_pts**2
    assert budget["candidates_per_macro_action_per_arm"] == expected
    assert expected == 9


def test_budget_per_arm_episode_total(packet: dict) -> None:
    """Per-arm episode total must equal macro_actions * candidates_per_macro_action."""
    budget = packet["budget"]
    expected = budget["macro_actions_per_episode"] * budget["candidates_per_macro_action_per_arm"]
    assert budget["candidates_per_arm_per_episode"] == expected
    assert expected == 90


def test_budget_all_arms_episode_total(packet: dict) -> None:
    """All-arms episode total must equal per_arm_episode_total * num_arms."""
    budget = packet["budget"]
    num_arms = len(packet["arms"])
    expected = budget["candidates_per_arm_per_episode"] * num_arms
    assert budget["total_candidates_all_arms_per_episode"] == expected
    assert expected == 180


def test_budget_non_vacuous(packet: dict) -> None:
    """Budget values must all be positive."""
    budget = packet["budget"]
    assert budget["macro_actions_per_episode"] > 0
    assert budget["candidates_per_macro_action_per_arm"] > 0
    assert budget["candidates_per_arm_per_episode"] > 0
    assert budget["total_candidates_all_arms_per_episode"] > 0


# ---------------------------------------------------------------------------
# Arm budget match — max_candidates equals per-macro-action budget
# ---------------------------------------------------------------------------


def test_reactive_max_candidates_match_macro_budget(packet: dict) -> None:
    """The reactive policy must evaluate the declared per-macro-action budget."""
    budget = packet["budget"]
    max_cand = packet["arms"]["reactive"]["residual_search"]["max_candidates"]
    assert max_cand == budget["candidates_per_macro_action_per_arm"]


def test_open_loop_runner_budget_matches_episode_budget(packet: dict) -> None:
    """The open-loop runner must bind its budget to the per-arm episode total."""
    assert (
        packet["budget"]["open_loop_runner_budget_per_episode"]
        == packet["budget"]["candidates_per_arm_per_episode"]
    )
    assert (
        packet["arms"]["open_loop"]["runner_binding"]["budget"]
        == packet["budget"]["open_loop_runner_budget_per_episode"]
    )


def test_arms_use_explicit_budget_contracts(packet: dict) -> None:
    """Each arm must bind its own native budget field explicitly."""
    assert packet["arms"]["open_loop"]["runner_binding"]["candidate_budget_field"] == "budget"
    assert (
        packet["arms"]["reactive"]["runner_binding"]["candidate_budget_field"] == "max_candidates"
    )
    assert packet["arms"]["reactive"]["residual_search"]["max_candidates"] == 9


# ---------------------------------------------------------------------------
# Seed fields — present in arms and match packet seeds
# ---------------------------------------------------------------------------


def test_arms_have_search_seed(packet: dict) -> None:
    """Both native seams must carry their search seed field."""
    assert packet["arms"]["open_loop"]["runner_binding"]["search_seed"] == 42
    assert packet["arms"]["reactive"]["residual_search"]["seed"] == 42


def test_arms_have_adversary_seed(packet: dict) -> None:
    """The reactive residual controller must carry its seed field."""
    ra = packet["arms"]["reactive"]["residual_adversary"]
    assert "seed" in ra


def test_arm_search_seeds_match_packet_seed(packet: dict) -> None:
    """Both native search seams must match the packet search seed."""
    packet_seed = packet["seeds"]["search_seed"]
    assert packet["arms"]["open_loop"]["runner_binding"]["search_seed"] == packet_seed
    assert packet["arms"]["reactive"]["residual_search"]["seed"] == packet_seed


def test_arm_adversary_seeds_match_packet_seed(packet: dict) -> None:
    """The reactive residual-adversary seed must match the packet seed."""
    packet_seed = packet["seeds"]["residual_adversary_seed"]
    arm_seed = packet["arms"]["reactive"]["residual_adversary"]["seed"]
    assert arm_seed == packet_seed


def test_search_and_adversary_seeds_are_equal(packet: dict) -> None:
    """Packet search_seed and residual_adversary_seed must be equal (42)."""
    assert packet["seeds"]["search_seed"] == packet["seeds"]["residual_adversary_seed"]


# ---------------------------------------------------------------------------
# Provenance and template path resolution
# ---------------------------------------------------------------------------


def test_provenance_scenario_template_resolves(packet: dict) -> None:
    """The provenance scenario_template path must exist on disk."""
    rel = packet["provenance"]["scenario_template"]
    path = REPO_ROOT / rel
    assert path.exists(), f"Provenance scenario_template not found: {path}"


def test_provenance_residual_search_config_resolves(packet: dict) -> None:
    """The provenance residual_search_config path must exist on disk."""
    rel = packet["provenance"]["residual_search_config"]
    path = REPO_ROOT / rel
    assert path.exists(), f"Provenance residual_search_config not found: {path}"


def test_provenance_residual_adversary_config_resolves(packet: dict) -> None:
    """The provenance residual_adversary_config path must exist on disk."""
    rel = packet["provenance"]["residual_adversary_config"]
    path = REPO_ROOT / rel
    assert path.exists(), f"Provenance residual_adversary_config not found: {path}"


def test_provenance_search_space_resolves(packet: dict) -> None:
    """The provenance search_space path must exist on disk."""
    rel = packet["provenance"]["search_space"]
    path = REPO_ROOT / rel
    assert path.exists(), f"Provenance search_space not found: {path}"


def test_provenance_dispatchable_inventory_resolves(packet: dict) -> None:
    """The provenance dispatchable_inventory path must exist on disk."""
    rel = packet["provenance"]["dispatchable_inventory"]
    path = REPO_ROOT / rel
    assert path.exists(), f"Provenance dispatchable_inventory not found: {path}"


def test_scenario_template_path_resolves(packet: dict) -> None:
    """The scenario.template path must exist on disk."""
    rel = packet["scenario"]["template"]
    path = REPO_ROOT / rel
    assert path.exists(), f"Scenario template not found: {path}"


def test_scenario_search_space_path_resolves(packet: dict) -> None:
    """The scenario.search_space path must exist on disk."""
    rel = packet["scenario"]["search_space"]
    path = REPO_ROOT / rel
    assert path.exists(), f"Scenario search_space not found: {path}"


def test_packet_search_space_freezes_template_seed(packet: dict) -> None:
    """The packet runner must sample the same frozen seed as its template."""
    search_space = SearchSpaceConfig.from_file(REPO_ROOT / packet["scenario"]["search_space"])
    template_seed = packet["scenario"]["template_seed"]
    assert search_space.scenario_seed.min == template_seed
    assert search_space.scenario_seed.max == template_seed


def test_packet_search_space_only_freezes_seed(packet: dict) -> None:
    """The packet-scoped space must preserve source bounds except for scenario seed."""
    frozen = yaml.safe_load(
        (REPO_ROOT / packet["scenario"]["search_space"]).read_text(encoding="utf-8")
    )
    source = yaml.safe_load(
        (REPO_ROOT / packet["scenario"]["source_search_space"]).read_text(encoding="utf-8")
    )
    frozen_variables = dict(frozen["variables"])
    source_variables = dict(source["variables"])
    assert frozen["source_space"] == packet["scenario"]["source_search_space"]
    assert frozen_variables.pop("scenario_seed") == {"min": 123, "max": 123}
    assert source_variables.pop("scenario_seed") == {"min": 100, "max": 999}
    assert frozen_variables == source_variables
    assert frozen["constraints"] == source["constraints"]


def test_scenario_template_contains_template_id(packet: dict) -> None:
    """The template YAML must define the named template_id used in the packet."""
    rel = packet["scenario"]["template"]
    template = yaml.safe_load((REPO_ROOT / rel).read_text(encoding="utf-8"))
    scenario_names = [s["name"] for s in template.get("scenarios", [])]
    assert packet["scenario"]["template_id"] in scenario_names, (
        f"template_id {packet['scenario']['template_id']!r} not in template "
        f"scenario names: {scenario_names}"
    )


def test_scenario_template_seed_matches(packet: dict) -> None:
    """The template YAML seed must match the packet template_seed."""
    rel = packet["scenario"]["template"]
    template = yaml.safe_load((REPO_ROOT / rel).read_text(encoding="utf-8"))
    template_seeds = template["scenarios"][0].get("seeds", [])
    assert packet["scenario"]["template_seed"] in template_seeds, (
        f"packet template_seed {packet['scenario']['template_seed']} "
        f"not in template seeds: {template_seeds}"
    )


# ---------------------------------------------------------------------------
# target_ped_idx — deliberate single-target choice
# ---------------------------------------------------------------------------


def test_target_ped_idx_is_single_target(packet: dict) -> None:
    """The reactive arm uses target_ped_idx [0], not the upstream all-target value."""
    tp = packet["arms"]["reactive"]["residual_adversary"]["target_ped_idx"]
    assert tp == [0]


def test_upstream_default_is_all_target() -> None:
    """The upstream adversary config uses -1 (all-target), confirming the
    packet's [0] is a deliberate divergence."""
    adv_cfg_path = REPO_ROOT / "configs" / "adversarial" / "issue_4360_residual_adversary.yaml"
    adv = yaml.safe_load(adv_cfg_path.read_text(encoding="utf-8"))
    assert adv["residual_adversary"]["target_ped_idx"] == -1


# ---------------------------------------------------------------------------
# Bounds identity — not relaxed
# ---------------------------------------------------------------------------


def test_residual_bounds_match_adversary_config(packet: dict) -> None:
    """Frozen residual bounds must match the upstream adversary config."""
    bounds = packet["bounds_identity"]["residual_bounds"]
    adversary_cfg_path = (
        REPO_ROOT / "configs" / "adversarial" / "issue_4360_residual_adversary.yaml"
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
    """Objective proxy must be the validated predicted-distance proxy."""
    assert packet["bounds_identity"]["objective_proxy"] == "minimize_predicted_robot_distance"


def test_arms_share_identical_residual_bounds(packet: dict) -> None:
    """The packet's frozen reactive residual bounds must match the source config."""
    bounds = packet["bounds_identity"]["residual_bounds"]
    reactive_bounds = packet["arms"]["reactive"]["residual_adversary"]
    for key in (
        "max_residual_accel_mps2",
        "max_jerk_mps3",
        "max_speed_delta_mps",
        "max_heading_change_per_macro_rad",
        "max_route_deviation_m",
        "min_separation_m",
    ):
        assert bounds[key] == reactive_bounds[key], f"bound mismatch: {key}"


def test_objective_projections_are_named_separately(packet: dict) -> None:
    """The reactive proxy and open-loop episode objective must not be conflated."""
    bounds = packet["bounds_identity"]
    assert bounds["objective_proxy"] == "minimize_predicted_robot_distance"
    assert bounds["open_loop_objective"] == "minimize_episode_min_robot_distance"
    assert "not assumed" in bounds["objective_comparability"]


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


def test_open_loop_objective_binding_is_explicit(packet: dict) -> None:
    """The open-loop runner must name its episode-record objective explicitly."""
    binding = packet["arms"]["open_loop"]["runner_binding"]
    assert binding["objective"] == "minimize_episode_min_robot_distance"
    assert binding["scenario_search_space"] == packet["scenario"]["search_space"]


def test_open_loop_binding_builds_valid_search_config(packet: dict) -> None:
    """The packet binding must construct the canonical search configuration."""
    binding = packet["arms"]["open_loop"]["runner_binding"]
    config = SearchConfig.from_files(
        policy=binding["policy"],
        scenario_template=REPO_ROOT / packet["scenario"]["template"],
        search_space=REPO_ROOT / binding["scenario_search_space"],
        objective=binding["objective"],
        output_dir=REPO_ROOT / "output" / "adversarial" / "matched_compute_test",
        budget=binding["budget"],
        seed=binding["search_seed"],
        horizon=binding["horizon_steps"],
        dt=binding["dt_s"],
    )
    config.validate()
    assert config.policy == "social_force"
    assert config.objective == "minimize_episode_min_robot_distance"
    assert config.budget == 90
    assert config.horizon == 50
    assert config.dt == 0.1
    assert config.search_space.scenario_seed.min == 123
    assert config.search_space.scenario_seed.max == 123


def test_reactive_residual_search_config_round_trip(packet: dict) -> None:
    """The reactive arm's search config must parse as a valid ResidualSearchConfig."""
    rs = packet["arms"]["reactive"]["residual_search"]
    config = ResidualSearchConfig(
        algorithm_name=rs["algorithm_name"],
        objective_proxy=rs["objective_proxy"],
        grid_points_per_dim=rs["grid_points_per_dim"],
        max_candidates=rs["max_candidates"],
        seed=rs["seed"],
    )
    assert config.algorithm_name == "finite_grid_search_v1"
    assert config.grid_points_per_dim == 3
    assert config.max_candidates == 9
    assert config.seed == 42


def test_residual_adversary_config_round_trip(packet: dict) -> None:
    """The reactive residual adversary config must parse without adaptation."""
    ra = packet["arms"]["reactive"]["residual_adversary"]
    config = ResidualAdversaryConfig(**ra)
    assert config.is_active is True
    assert config.macro_action_dt_s == 0.5
    assert config.max_residual_accel_mps2 == 1.5
    assert config.max_jerk_mps3 == 7.5
    assert config.seed == 42


# ---------------------------------------------------------------------------
# Scenario frozen identity
# ---------------------------------------------------------------------------


def test_scenario_template_id_is_frozen(packet: dict) -> None:
    """Scenario template_id must be the checked-in crossing_ttc_template."""
    assert packet["scenario"]["template_id"] == "crossing_ttc_template"


def test_scenario_template_seed_is_frozen(packet: dict) -> None:
    """Scenario template_seed must be 123 (the template's declared seed)."""
    assert packet["scenario"]["template_seed"] == 123


def test_frozen_scenario_seeds(packet: dict) -> None:
    """Scenario seeds must be frozen and match the checked-in template seed."""
    seeds = packet["seeds"]["frozen_scenario_seeds"]
    assert isinstance(seeds, list)
    assert len(seeds) >= 1
    assert all(isinstance(s, int) for s in seeds)
    assert seeds == [packet["scenario"]["template_seed"]]


def test_search_seed_is_frozen(packet: dict) -> None:
    """Search seed must be a frozen integer."""
    assert isinstance(packet["seeds"]["search_seed"], int)


def test_residual_adversary_seed_is_frozen(packet: dict) -> None:
    """Residual adversary seed must be a frozen integer."""
    assert isinstance(packet["seeds"]["residual_adversary_seed"], int)
