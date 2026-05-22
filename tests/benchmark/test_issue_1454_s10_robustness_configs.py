"""Contract tests for issue #1454 S10 robustness benchmark configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready_campaign import _load_campaign_scenarios, load_campaign_config

ROOT = Path(__file__).resolve().parents[2]
ISSUE_1353_STRESS = ROOT / "configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml"
STAGE_A = ROOT / "configs/benchmarks/issue_1454_s10_fixed_h100_broader_baselines.yaml"
STAGE_B = ROOT / "configs/benchmarks/issue_1454_s10_scenario_horizons_h500_broader_baselines.yaml"

S10_SEEDS = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a benchmark YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _planner_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return planner rows from a benchmark config."""
    return list(config["planners"])


def _planner_keys(config: dict[str, Any]) -> list[str]:
    """Return planner keys in config order."""
    return [str(planner["key"]) for planner in _planner_rows(config)]


def _resolved_seed_inventory(config_path: Path) -> list[int]:
    """Return the unique seeds resolved into the campaign scenarios."""
    cfg = load_campaign_config(config_path)
    return sorted(
        {int(seed) for scenario in _load_campaign_scenarios(cfg) for seed in scenario["seeds"]}
    )


def test_issue_1454_stage_configs_reuse_1353_broader_rows() -> None:
    """Stage A and B should preserve the #1353 broader planner comparison surface."""
    broader = _load_yaml(ISSUE_1353_STRESS)
    stage_a = _load_yaml(STAGE_A)
    stage_b = _load_yaml(STAGE_B)

    assert _planner_rows(stage_a) == _planner_rows(broader)
    assert _planner_rows(stage_b) == _planner_rows(broader)
    assert _planner_keys(stage_a) == _planner_keys(stage_b)


def test_issue_1454_stage_a_fixed_h100_s10_contract() -> None:
    """Stage A should only extend #1353 stress to the S10 fixed-h100 schedule."""
    cfg = load_campaign_config(STAGE_A)

    assert cfg.paper_facing is False
    assert cfg.paper_profile_version == "paper-matrix-v1"
    assert cfg.horizon == 100
    assert cfg.scenario_horizons_path is None
    assert cfg.seed_policy.mode == "seed-set"
    assert cfg.seed_policy.seed_set == "paper_eval_s10"
    assert cfg.stop_on_failure is False
    assert cfg.export_publication_bundle is False
    assert cfg.paper_interpretation_profile == "issue-1454-s10-fixed-h100-broader-robustness"
    assert _resolved_seed_inventory(STAGE_A) == S10_SEEDS


def test_issue_1454_stage_b_scenario_horizon_s10_contract() -> None:
    """Stage B should use the same S10 rows with the gated h500 horizon schedule."""
    cfg = load_campaign_config(STAGE_B)

    assert cfg.paper_facing is False
    assert cfg.paper_profile_version == "paper-matrix-v1"
    assert cfg.horizon is None
    assert (
        cfg.scenario_horizons_path
        == (ROOT / "configs/policy_search/scenario_horizons_h500.yaml").resolve()
    )
    assert all(planner.horizon_override is None for planner in cfg.planners)
    assert cfg.seed_policy.mode == "seed-set"
    assert cfg.seed_policy.seed_set == "paper_eval_s10"
    assert cfg.stop_on_failure is False
    assert cfg.export_publication_bundle is False
    assert (
        cfg.paper_interpretation_profile
        == "issue-1454-s10-scenario-horizons-h500-broader-robustness"
    )
    assert _resolved_seed_inventory(STAGE_B) == S10_SEEDS

    scenarios = _load_campaign_scenarios(cfg)
    horizon_meta = [
        scenario["metadata"]["scenario_horizon"]
        for scenario in scenarios
        if "scenario_horizon" in scenario.get("metadata", {})
    ]
    assert len(horizon_meta) == 48
    assert sum(1 for row in horizon_meta if row["status"] == "planner_blocked") == 3
    assert {scenario["simulation_config"]["max_episode_steps"] for scenario in scenarios} <= set(
        range(1, 601)
    )


def test_issue_1454_stage_configs_keep_primary_comparison_matched() -> None:
    """The primary fixed-vs-horizon verdict should be possible on identical planner keys."""
    stage_a = load_campaign_config(STAGE_A)
    stage_b = load_campaign_config(STAGE_B)

    assert [planner.key for planner in stage_a.planners] == [
        planner.key for planner in stage_b.planners
    ]
    assert [planner.algo for planner in stage_a.planners] == [
        planner.algo for planner in stage_b.planners
    ]
    assert stage_a.scenario_matrix_path == stage_b.scenario_matrix_path
    assert stage_a.comparability_mapping_path == stage_b.comparability_mapping_path
    assert (
        stage_a.route_clearance_certifications_path == stage_b.route_clearance_certifications_path
    )
