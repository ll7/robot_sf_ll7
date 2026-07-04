"""Static identity tests for issue #4365 h600 hybrid-vs-ORCA S30 config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.camera_ready._preflight import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._util import _hash_payload
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml"
)
SEED_SETS_PATH = REPO_ROOT / "configs/benchmarks/seed_sets_v1.yaml"

EXPECTED_SCENARIO_MATRIX_HASH = "c10df617a87c"
EXPECTED_S30_SCENARIO_SEED_HASH = "152eba3969a9"
EXPECTED_S30_SEEDS = list(range(111, 141))
EXPECTED_ROSTER = [
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "orca",
    "ppo",
]
EXPECTED_HYBRID_CONFIGS = {
    "scenario_adaptive_hybrid_orca_v1": (
        "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml"
    ),
    "scenario_adaptive_hybrid_orca_v2_collision_guard": (
        "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml"
    ),
    "hybrid_rule_v3_fast_progress_static_escape": (
        "configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml"
    ),
    "hybrid_rule_v3_fast_progress_static_escape_continuous": (
        "configs/policy_search/candidates/"
        "hybrid_rule_v3_fast_progress_static_escape_continuous.yaml"
    ),
}
EXPECTED_PPO_CONFIG = "configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_h600_hybrid_vs_orca_s30_config_identity() -> None:
    """Config pins h600, the predeclared S30 seed set, and the six-arm roster."""
    config = _load_yaml(CONFIG_PATH)
    seed_sets = _load_yaml(SEED_SETS_PATH)

    assert config["name"] == "paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30"
    assert config["paper_facing"] is False
    assert (
        config["paper_interpretation_profile"] == "diagnostic-h600-hybrid-vs-orca-s30-preregistered"
    )
    assert config["preregistration"]["expected_scenario_matrix_hash"] == (
        EXPECTED_SCENARIO_MATRIX_HASH
    )
    assert config["preregistration"]["expected_s30_scenario_seed_hash"] == (
        EXPECTED_S30_SCENARIO_SEED_HASH
    )
    assert config["scenario_matrix"] == "configs/scenarios/classic_interactions_francis2023.yaml"
    assert config["horizon"] == 600
    assert config["dt"] == pytest.approx(0.1)
    assert config["kinematics_matrix"] == ["differential_drive"]
    assert config["seed_policy"] == {
        "mode": "seed-set",
        "seed_set": "paper_eval_s30",
        "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
    }
    assert seed_sets["paper_eval_s30"] == EXPECTED_S30_SEEDS

    planners = config["planners"]
    assert [planner["key"] for planner in planners] == EXPECTED_ROSTER

    for planner in planners[:4]:
        key = planner["key"]
        assert planner == {
            "key": key,
            "algo": "hybrid_rule_local_planner",
            "planner_group": "experimental",
            "algo_config": EXPECTED_HYBRID_CONFIGS[key],
            "benchmark_profile": "experimental",
        }
        assert (REPO_ROOT / EXPECTED_HYBRID_CONFIGS[key]).is_file()

    assert planners[4] == {
        "key": "orca",
        "algo": "orca",
        "planner_group": "core",
        "benchmark_profile": "baseline-safe",
        "socnav_missing_prereq_policy": "fallback",
    }
    assert planners[5] == {
        "key": "ppo",
        "algo": "ppo",
        "planner_group": "experimental",
        "algo_config": EXPECTED_PPO_CONFIG,
        "benchmark_profile": "experimental",
        "adapter_impact_eval": True,
        "workers": 1,
    }
    assert (REPO_ROOT / EXPECTED_PPO_CONFIG).is_file()


def test_h600_hybrid_vs_orca_s30_loader_preserves_s30_expansion_hash() -> None:
    """Campaign loader resolves the predeclared S30 scenario-plus-seed payload."""
    cfg = load_campaign_config(CONFIG_PATH)

    assert cfg.name == "paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30"
    assert cfg.horizon == 600
    assert cfg.seed_policy.mode == "seed-set"
    assert cfg.seed_policy.seed_set == "paper_eval_s30"
    assert [planner.key for planner in cfg.planners] == EXPECTED_ROSTER

    scenarios = _load_campaign_scenarios(cfg)
    assert len(scenarios) == 48
    assert _hash_payload(scenarios) == EXPECTED_S30_SCENARIO_SEED_HASH
