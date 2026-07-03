"""Pre-registration contract for issue #4230 h600 hybrid roster."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    require_algorithm_allowed,
)
from robot_sf.benchmark.camera_ready._preflight import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._util import _hash_payload
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_roster.yaml"
BASE_H600_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon600_probe.yaml"
)
SEED_SETS_PATH = REPO_ROOT / "configs/benchmarks/seed_sets_v1.yaml"
EXPECTED_PLANNERS = [
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
]
EXPECTED_CANDIDATE_CONFIGS = {
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
SCIENTIFIC_SURFACE_FIELDS = (
    "scenario_matrix",
    "comparability_mapping",
    "amv_profile",
    "seed_policy",
    "snqi_weights",
    "snqi_baseline",
    "snqi_contract",
    "horizon",
    "dt",
    "record_forces",
    "resume",
    "stop_on_failure",
    "bootstrap_samples",
    "bootstrap_confidence",
    "bootstrap_seed",
    "kinematics_matrix",
    "export_publication_bundle",
    "include_videos_in_publication",
    "overwrite_publication_bundle",
)
DISALLOWED_ACCIDENTAL_BASELINES = {
    "goal",
    "social_force",
    "orca",
    "ppo",
    "prediction_planner",
    "socnav_sampling",
    "sacadrl",
    "gap_prediction",
}


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_h600_hybrid_roster_preserves_h600_surface_contract() -> None:
    """New roster must differ from the h600 probe only by identity and planners."""
    config = _load_yaml(CONFIG_PATH)
    base_config = _load_yaml(BASE_H600_CONFIG_PATH)

    assert config["name"] == "paper_experiment_matrix_v1_h600_hybrid_roster"
    for field in SCIENTIFIC_SURFACE_FIELDS:
        assert config[field] == base_config[field], field

    seed_sets = _load_yaml(SEED_SETS_PATH)
    assert seed_sets["eval"] == [111, 112, 113]
    assert config["paper_facing"] is False
    assert config["paper_interpretation_profile"] == "diagnostic-h600-hybrid-roster-preregistered"
    assert config["preregistration"]["expected_scenario_matrix_hash"] == "c10df617a87c"


def test_h600_hybrid_roster_uses_exact_verified_four_arm_roster() -> None:
    """Planner roster is the pre-registered h500 hybrid/scenario-adaptive top family."""
    config = _load_yaml(CONFIG_PATH)
    comparability = _load_yaml(REPO_ROOT / config["comparability_mapping"])
    planners = config["planners"]
    assert isinstance(planners, list)

    assert [planner["key"] for planner in planners] == EXPECTED_PLANNERS
    assert not (set(EXPECTED_PLANNERS) & DISALLOWED_ACCIDENTAL_BASELINES)
    assert not ({planner["key"] for planner in planners} & DISALLOWED_ACCIDENTAL_BASELINES)

    for planner in planners:
        key = planner["key"]
        assert planner == {
            "key": key,
            "algo": "hybrid_rule_local_planner",
            "planner_group": "experimental",
            "algo_config": EXPECTED_CANDIDATE_CONFIGS[key],
            "benchmark_profile": "experimental",
        }
        candidate_path = REPO_ROOT / EXPECTED_CANDIDATE_CONFIGS[key]
        assert candidate_path.is_file()
        candidate_payload = _load_yaml(candidate_path)
        assert candidate_payload["algo"] == "hybrid_rule_local_planner"
        assert candidate_payload["allow_testing_algorithms"] is True
        assert key in comparability["planner_key_mapping"]


def test_h600_hybrid_roster_loads_and_keeps_expected_hash() -> None:
    """Campaign loader accepts the config and resolves the h600 scenario surface."""
    cfg = load_campaign_config(CONFIG_PATH)

    assert cfg.name == "paper_experiment_matrix_v1_h600_hybrid_roster"
    assert cfg.horizon == 600
    assert cfg.dt == pytest.approx(0.1)
    assert cfg.seed_policy.mode == "seed-set"
    assert cfg.seed_policy.seed_set == "eval"
    assert [planner.key for planner in cfg.planners] == EXPECTED_PLANNERS

    scenarios = _load_campaign_scenarios(cfg)
    assert len(scenarios) == 48
    assert _hash_payload(scenarios) == "c10df617a87c"


def test_h600_hybrid_roster_keeps_hybrid_rule_explicit_opt_in() -> None:
    """The roster remains experimental and executable only through explicit candidate opt-in."""
    readiness = get_algorithm_readiness("hybrid_rule_local_planner")

    assert readiness is not None
    assert readiness.tier == "experimental"
    assert readiness.requires_explicit_opt_in is True
    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="hybrid_rule_local_planner",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )
    assert (
        require_algorithm_allowed(
            algo="hybrid_rule_local_planner",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )
        == readiness
    )
