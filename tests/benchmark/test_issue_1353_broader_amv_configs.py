"""Contract tests for issue #1353 broader AMV preflight configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_NOMINAL = ROOT / "configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml"
PRIMARY_STRESS = ROOT / "configs/benchmarks/issue_1344_paired_stress_primary.yaml"
BROADER_NOMINAL = ROOT / "configs/benchmarks/issue_1353_paired_nominal_v1_broader_baselines.yaml"
BROADER_STRESS = ROOT / "configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml"

PROTOCOL_KEYS = {
    "paper_facing",
    "paper_profile_version",
    "scenario_matrix",
    "comparability_mapping",
    "route_clearance_certifications",
    "amv_profile",
    "seed_policy",
    "snqi_weights",
    "snqi_baseline",
    "snqi_contract",
    "workers",
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
    "repository_url",
    "release_tag",
    "doi",
}

EXPECTED_PLANNER_KEYS = {
    "goal",
    "social_force",
    "orca",
    "ppo",
    "prediction_planner",
    "socnav_sampling",
    "sacadrl",
    "socnav_bench",
}

EXPECTED_BROADER_ROWS = {
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
}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a benchmark YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _planner_keys(config: dict[str, Any]) -> set[str]:
    """Return planner keys from a benchmark config."""
    return {str(planner["key"]) for planner in config["planners"]}


def _planner_rows(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return planner rows keyed by planner key."""
    return {str(planner["key"]): planner for planner in config["planners"]}


def test_issue_1353_broader_configs_preserve_issue_1344_protocol() -> None:
    """The #1353 preflight configs should only expand planner rows."""
    for primary_path, broader_path in (
        (PRIMARY_NOMINAL, BROADER_NOMINAL),
        (PRIMARY_STRESS, BROADER_STRESS),
    ):
        primary = _load_yaml(primary_path)
        broader = _load_yaml(broader_path)

        for key in PROTOCOL_KEYS:
            assert broader[key] == primary[key], f"{broader_path.name} changed {key}"

        assert broader["paper_interpretation_profile"] == "issue-1353-broader-amv-preflight"


def test_issue_1353_broader_configs_keep_primary_rows_and_add_broader_rows() -> None:
    """The broader configs should include #1344 primary rows plus all broader baseline rows."""
    for primary_path, broader_path in (
        (PRIMARY_NOMINAL, BROADER_NOMINAL),
        (PRIMARY_STRESS, BROADER_STRESS),
    ):
        primary = _load_yaml(primary_path)
        broader = _load_yaml(broader_path)

        assert _planner_keys(primary).issubset(_planner_keys(broader))
        assert _planner_keys(broader) == EXPECTED_PLANNER_KEYS
        assert _planner_rows(broader) == {
            key: {"key": key, **expected_row} for key, expected_row in EXPECTED_BROADER_ROWS.items()
        }
