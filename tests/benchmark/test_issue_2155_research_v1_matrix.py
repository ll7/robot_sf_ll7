"""Contract tests for issue #2155 research-v1 AMMV benchmark matrix.

This is a pre-execution contract.  No benchmark output, result comparison,
or performance claim is tested here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready_campaign import (
    _load_campaign_scenarios,
    load_campaign_config,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = ROOT / "configs/benchmarks/issue_2155_research_v1_ammv_matrix.yaml"
SCENARIO_MATRIX = ROOT / "configs/scenarios/issue_2155_research_v1_ammv.yaml"

# Expected scenario families and their member scenario names
EXPECTED_SCENARIO_FAMILIES: dict[str, set[str]] = {
    "bottleneck": {"classic_bottleneck_low", "classic_bottleneck_medium"},
    "doorway": {"classic_doorway_low", "classic_doorway_medium"},
    "head_on_corridor": {"classic_head_on_corridor_low", "classic_head_on_corridor_medium"},
    "overtaking": {"classic_overtaking_low"},
    "cross_trap": {"classic_cross_trap_low"},
    "t_intersection": {"classic_t_intersection_low"},
    "merging": {"classic_merging_low"},
    "group_crossing": {"classic_group_crossing_low"},
    "urban_crossing": {"classic_urban_crossing_medium"},
    "francis_basic": {
        "francis2023_frontal_approach",
        "francis2023_pedestrian_obstruction",
        "francis2023_blind_corner",
        "francis2023_narrow_hallway",
        "francis2023_narrow_doorway",
        "francis2023_down_path",
        "francis2023_intersection_wait",
        "francis2023_intersection_proceed",
    },
    "francis_overtaking": {
        "francis2023_robot_overtaking",
        "francis2023_pedestrian_overtaking",
        "francis2023_parallel_traffic",
    },
}

# Planner families: (key, algo, planner_group, expected_fields)
EXPECTED_PLANNER_FAMILIES: dict[str, dict[str, Any]] = {
    # Core baseline-safe family
    "goal": {
        "algo": "goal",
        "planner_group": "core",
        "benchmark_profile": "baseline-safe",
    },
    "social_force": {
        "algo": "social_force",
        "planner_group": "core",
        "benchmark_profile": "baseline-safe",
    },
    "orca": {
        "algo": "orca",
        "planner_group": "core",
        "benchmark_profile": "baseline-safe",
        "socnav_missing_prereq_policy": "fallback",
    },
    # Experimental broader baseline family
    "ppo": {
        "algo": "ppo",
        "planner_group": "experimental",
        "algo_config": "configs/baselines/ppo_15m_grid_socnav.yaml",
        "benchmark_profile": "experimental",
        "adapter_impact_eval": True,
    },
    "prediction_planner": {
        "algo": "prediction_planner",
        "planner_group": "experimental",
        "algo_config": "configs/algos/prediction_planner_camera_ready.yaml",
        "benchmark_profile": "experimental",
    },
    "socnav_sampling": {
        "algo": "socnav_sampling",
        "planner_group": "experimental",
        "benchmark_profile": "experimental",
        "socnav_missing_prereq_policy": "skip-with-warning",
    },
    "sacadrl": {
        "algo": "sacadrl",
        "planner_group": "experimental",
        "benchmark_profile": "experimental",
        "socnav_missing_prereq_policy": "fallback",
    },
    "socnav_bench": {
        "algo": "socnav_bench",
        "planner_group": "experimental",
        "benchmark_profile": "experimental",
        "socnav_missing_prereq_policy": "skip-with-warning",
    },
    # Hybrid rule family
    "hybrid_rule_v3_fast_progress_static_escape": {
        "algo": "hybrid_rule_local_planner",
        "planner_group": "experimental",
        "algo_config": "configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml",
        "benchmark_profile": "experimental",
    },
    "scenario_adaptive_hybrid_orca_v1": {
        "algo": "hybrid_rule_local_planner",
        "planner_group": "experimental",
        "algo_config": "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml",
        "benchmark_profile": "experimental",
    },
    # Predictive planner v2 family
    "prediction_planner_v2_full": {
        "algo": "prediction_planner",
        "planner_group": "experimental",
        "algo_config": "configs/algos/prediction_planner_camera_ready.yaml",
        "benchmark_profile": "experimental",
    },
    "prediction_planner_v2_xl_ego": {
        "algo": "prediction_planner",
        "planner_group": "experimental",
        "algo_config": "configs/algos/prediction_planner_camera_ready_xl_ego.yaml",
        "benchmark_profile": "experimental",
    },
}

# Ordered planner key groups for row ordering enforcement
CORE_KEYS = ["goal", "social_force", "orca"]
EXPERIMENTAL_BROADER_KEYS = [
    "ppo",
    "prediction_planner",
    "socnav_sampling",
    "sacadrl",
    "socnav_bench",
]
HYBRID_RULE_KEYS = [
    "hybrid_rule_v3_fast_progress_static_escape",
    "scenario_adaptive_hybrid_orca_v1",
]
PREDICTIVE_V2_KEYS = [
    "prediction_planner_v2_full",
    "prediction_planner_v2_xl_ego",
]
ALL_EXPECTED_PLANNER_KEYS = (
    CORE_KEYS + EXPERIMENTAL_BROADER_KEYS + HYBRID_RULE_KEYS + PREDICTIVE_V2_KEYS
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _planner_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config["planners"])


def _planner_keys(config: dict[str, Any]) -> list[str]:
    return [str(planner["key"]) for planner in _planner_rows(config)]


def _planner_rows_by_key(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(planner["key"]): planner for planner in _planner_rows(config)}


def _resolved_seed_inventory(config_path: Path) -> list[int]:
    cfg = load_campaign_config(config_path)
    return sorted(
        {int(seed) for scenario in _load_campaign_scenarios(cfg) for seed in scenario["seeds"]}
    )


# ── Scenario matrix tests ────────────────────────────────────────────────


def test_issue_2155_scenario_matrix_loads() -> None:
    """The research-v1 AMMV scenario matrix should load without error."""
    scenarios = _load_yaml(SCENARIO_MATRIX)
    assert scenarios is not None
    assert "includes" in scenarios
    assert "select_scenarios" in scenarios
    assert scenarios.get("schema_version") == "robot_sf.scenario_matrix.v1"


def test_issue_2155_scenario_families_are_compact() -> None:
    """Scenario families should match the compact AMMV contract exactly."""
    raw = _load_yaml(SCENARIO_MATRIX)
    selected = set(raw["select_scenarios"])
    expected_all = set().union(*EXPECTED_SCENARIO_FAMILIES.values())
    assert selected == expected_all, (
        f"Scenario selection mismatch.\n"
        f"  Extra:  {sorted(selected - expected_all)}\n"
        f"  Missing: {sorted(expected_all - selected)}"
    )


def test_issue_2155_scenario_families_have_no_duplicates() -> None:
    """Each scenario should appear in exactly one family."""
    seen: set[str] = set()
    for family_name, members in EXPECTED_SCENARIO_FAMILIES.items():
        duplicates = members & seen
        assert not duplicates, (
            f"Family '{family_name}' contains scenarios already in another family: {duplicates}"
        )
        seen |= members


def test_issue_2155_scenario_matrix_resolves() -> None:
    """The scenario matrix should resolve to valid scenario definitions."""
    cfg = load_campaign_config(BENCHMARK_CONFIG)
    scenarios = _load_campaign_scenarios(cfg)
    assert len(scenarios) >= len(set().union(*EXPECTED_SCENARIO_FAMILIES.values())), (
        f"Resolved {len(scenarios)} scenarios; expected at least "
        f"{len(set().union(*EXPECTED_SCENARIO_FAMILIES.values()))}"
    )
    scenario_names = {s["name"] for s in scenarios}
    expected_names = set().union(*EXPECTED_SCENARIO_FAMILIES.values())
    assert scenario_names == expected_names, (
        f"Resolved names differ from expected.\n"
        f"  Extra:  {sorted(scenario_names - expected_names)}\n"
        f"  Missing: {sorted(expected_names - scenario_names)}"
    )


# ── Benchmark config tests ──────────────────────────────────────────────


def test_issue_2155_benchmark_config_loads() -> None:
    """The benchmark config should load via the campaign config loader."""
    cfg = load_campaign_config(BENCHMARK_CONFIG)
    assert cfg.paper_facing is False
    assert cfg.paper_profile_version == "paper-matrix-v1"
    assert cfg.horizon == 100
    assert cfg.export_publication_bundle is False
    assert cfg.stop_on_failure is False


def test_issue_2155_benchmark_config_has_expected_planner_families() -> None:
    """All planner families should be present with correct metadata."""
    raw = _load_yaml(BENCHMARK_CONFIG)
    rows = _planner_rows_by_key(raw)

    assert set(rows.keys()) == set(ALL_EXPECTED_PLANNER_KEYS), (
        f"Planner key mismatch.\n"
        f"  Extra:  {sorted(set(rows.keys()) - set(ALL_EXPECTED_PLANNER_KEYS))}\n"
        f"  Missing: {sorted(set(ALL_EXPECTED_PLANNER_KEYS) - set(rows.keys()))}"
    )

    for key, expected in EXPECTED_PLANNER_FAMILIES.items():
        row = rows[key]
        row_clean = {k: v for k, v in row.items() if k != "key"}
        assert row_clean == expected, (
            f"Planner '{key}' row content mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {row_clean}"
        )


def test_issue_2155_planner_family_ordering() -> None:
    """Planner rows should be ordered by family: core, experimental, hybrid_rule, predictive_v2."""
    raw = _load_yaml(BENCHMARK_CONFIG)
    keys = _planner_keys(raw)
    assert keys == ALL_EXPECTED_PLANNER_KEYS, (
        f"Planner ordering mismatch.\n  Expected: {ALL_EXPECTED_PLANNER_KEYS}\n  Got:      {keys}"
    )


def test_issue_2155_planner_family_grouping() -> None:
    """Each planner should carry the correct planner_group."""
    raw = _load_yaml(BENCHMARK_CONFIG)
    rows = _planner_rows_by_key(raw)

    for key in CORE_KEYS:
        assert rows[key]["planner_group"] == "core", f"Core planner '{key}' has wrong group"
    for key in EXPERIMENTAL_BROADER_KEYS + HYBRID_RULE_KEYS + PREDICTIVE_V2_KEYS:
        assert rows[key]["planner_group"] == "experimental", (
            f"Experimental planner '{key}' has wrong group"
        )


def test_issue_2155_fallback_planners_have_prereq_policy() -> None:
    """Planners with fallback or skip prereqs should declare their policy."""
    raw = _load_yaml(BENCHMARK_CONFIG)
    rows = _planner_rows_by_key(raw)

    for key, row in rows.items():
        prereq = row.get("socnav_missing_prereq_policy", "none")
        if prereq in ("fallback", "skip-with-warning"):
            assert key in EXPECTED_PLANNER_FAMILIES, (
                f"Planner '{key}' with prereq policy '{prereq}' must be in expected families"
            )

    # Specific policies we expect
    assert rows["orca"]["socnav_missing_prereq_policy"] == "fallback"
    assert rows["sacadrl"]["socnav_missing_prereq_policy"] == "fallback"
    assert rows["socnav_sampling"]["socnav_missing_prereq_policy"] == "skip-with-warning"
    assert rows["socnav_bench"]["socnav_missing_prereq_policy"] == "skip-with-warning"


def test_issue_2155_benchmark_seed_policy() -> None:
    """Seed policy should use paper_eval_s5 (5 seeds)."""
    cfg = load_campaign_config(BENCHMARK_CONFIG)
    assert cfg.seed_policy.mode == "seed-set"
    assert cfg.seed_policy.seed_set == "paper_eval_s5"
    seeds = _resolved_seed_inventory(BENCHMARK_CONFIG)
    expected_seeds = [111, 112, 113, 114, 115]
    assert seeds == expected_seeds, f"Seeds mismatch: {seeds} != {expected_seeds}"


def test_issue_2155_benchmark_no_snqi_baseline() -> None:
    """Research-v1 matrix should not require SNQI baseline/weights."""
    raw = _load_yaml(BENCHMARK_CONFIG)
    assert "snqi_weights" not in raw, "snqi_weights should not be set for research-v1"
    assert "snqi_baseline" not in raw, "snqi_baseline should not be set for research-v1"
    snqi = raw.get("snqi_contract", {})
    assert snqi.get("enabled") is False, "SNQI contract should be disabled for research-v1"


def test_issue_2155_benchmark_artifact_policy() -> None:
    """Artifact policy should disable publication bundles."""
    raw = _load_yaml(BENCHMARK_CONFIG)
    assert raw.get("export_publication_bundle") is False
    assert raw.get("include_videos_in_publication") is False
    assert raw.get("overwrite_publication_bundle") is False


def test_issue_2155_benchmark_paper_facing() -> None:
    """The research-v1 matrix must not be marked as paper-facing."""
    cfg = load_campaign_config(BENCHMARK_CONFIG)
    assert cfg.paper_facing is False
    assert cfg.paper_interpretation_profile == "issue-2155-research-v1-ammv-matrix"


# ── Fail-closed semantics tests ──────────────────────────────────────────


def test_issue_2155_fallback_rows_excluded_from_success() -> None:
    """Planners with fallback or skip policies should be present but caveated.

    This test checks the contract declares fallback policies; campaign-level
    enforcement of the exclusion is the responsibility of the runner.
    """
    raw = _load_yaml(BENCHMARK_CONFIG)
    rows = _planner_rows_by_key(raw)
    fallback_or_skip = {
        key: row["socnav_missing_prereq_policy"]
        for key, row in rows.items()
        if row.get("socnav_missing_prereq_policy") in ("fallback", "skip-with-warning")
    }
    assert len(fallback_or_skip) >= 2, (
        f"Expected at least 2 fallback/skip planners, got {len(fallback_or_skip)}: "
        f"{fallback_or_skip}"
    )


def test_issue_2155_core_planners_anchor_campaign_success() -> None:
    """Native-core planners (goal, social_force) must have no fallback/skip policy.

    ORCA is an exception: it lives in the core family for historical alignment
    but declares socnav_missing_prereq_policy: fallback because it requires
    the optional rvo2 library.  Campaign-level success is still anchored on
    goal and social_force completing in native mode.
    """
    raw = _load_yaml(BENCHMARK_CONFIG)
    rows = _planner_rows_by_key(raw)
    for key in CORE_KEYS:
        prereq = rows[key].get("socnav_missing_prereq_policy")
        if key == "orca":
            continue
        prereq = rows[key].get("socnav_missing_prereq_policy")
        assert prereq is None or prereq == "none", (
            f"Native-core planner '{key}' must not have fallback/skip prereq policy, got '{prereq}'"
        )
