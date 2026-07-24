"""Contract tests for issue #6095 S10 ORCA/PPO discriminability calibration configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready_campaign import _load_campaign_scenarios, load_campaign_config

ROOT = Path(__file__).resolve().parents[2]
NOMINAL = ROOT / "configs/benchmarks/issue_6095_nominal_discriminability_v1.yaml"
STRESS = ROOT / "configs/benchmarks/issue_6095_stress_discriminability_v1.yaml"
PPO_ALGO = ROOT / "configs/baselines/ppo_15m_grid_socnav.yaml"
S10_SEEDS = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _planner_keys(config: dict[str, Any]) -> list[str]:
    return [str(planner["key"]) for planner in config["planners"]]


def _planner_algos(config: dict[str, Any]) -> list[str]:
    return [str(planner["algo"]) for planner in config["planners"]]


def _resolved_seed_inventory(config_path: Path) -> list[int]:
    cfg = load_campaign_config(config_path)
    return sorted(
        {int(seed) for scenario in _load_campaign_scenarios(cfg) for seed in scenario["seeds"]}
    )


class TestIssue6095NominalConfig:
    """Contract checks for the nominal discriminability config."""

    def test_planners_are_orca_and_ppo_only(self) -> None:
        cfg = _load_yaml(NOMINAL)
        assert _planner_keys(cfg) == ["orca", "ppo"]
        assert _planner_algos(cfg) == ["orca", "ppo"]

    def test_ppo_has_existing_algo_config(self) -> None:
        cfg = _load_yaml(NOMINAL)
        ppo = next(p for p in cfg["planners"] if p["key"] == "ppo")
        algo_path = ROOT / str(ppo["algo_config"])
        assert algo_path.is_file(), f"PPO algo_config not found: {algo_path}"
        assert algo_path.samefile(PPO_ALGO)

    def test_seed_policy_is_paper_eval_s10(self) -> None:
        cfg = _load_yaml(NOMINAL)
        assert cfg["seed_policy"]["mode"] == "seed-set"
        assert cfg["seed_policy"]["seed_set"] == "paper_eval_s10"

    def test_resolved_seeds_match_s10(self) -> None:
        assert _resolved_seed_inventory(NOMINAL) == S10_SEEDS

    def test_horizon_dt_kinematics(self) -> None:
        cfg = _load_yaml(NOMINAL)
        assert cfg["horizon"] == 100
        assert cfg["dt"] == 0.1
        assert cfg["kinematics_matrix"] == ["differential_drive"]

    def test_paper_facing_is_false(self) -> None:
        cfg = _load_yaml(NOMINAL)
        assert cfg["paper_facing"] is False

    def test_stop_on_failure_is_false(self) -> None:
        cfg = _load_yaml(NOMINAL)
        assert cfg["stop_on_failure"] is False

    def test_scenario_count(self) -> None:
        cfg = _load_yaml(NOMINAL)
        matrix_path = ROOT / str(cfg["scenario_matrix"])
        matrix = _load_yaml(matrix_path)
        select = matrix.get("select_scenarios", [])
        assert len(select) == 4


class TestIssue6095StressConfig:
    """Contract checks for the stress discriminability config."""

    def test_planners_are_orca_and_ppo_only(self) -> None:
        cfg = _load_yaml(STRESS)
        assert _planner_keys(cfg) == ["orca", "ppo"]
        assert _planner_algos(cfg) == ["orca", "ppo"]

    def test_ppo_has_existing_algo_config(self) -> None:
        cfg = _load_yaml(STRESS)
        ppo = next(p for p in cfg["planners"] if p["key"] == "ppo")
        algo_path = ROOT / str(ppo["algo_config"])
        assert algo_path.is_file(), f"PPO algo_config not found: {algo_path}"
        assert algo_path.samefile(PPO_ALGO)

    def test_seed_policy_is_paper_eval_s10(self) -> None:
        cfg = _load_yaml(STRESS)
        assert cfg["seed_policy"]["mode"] == "seed-set"
        assert cfg["seed_policy"]["seed_set"] == "paper_eval_s10"

    def test_resolved_seeds_match_s10(self) -> None:
        assert _resolved_seed_inventory(STRESS) == S10_SEEDS

    def test_horizon_dt_kinematics(self) -> None:
        cfg = _load_yaml(STRESS)
        assert cfg["horizon"] == 100
        assert cfg["dt"] == 0.1
        assert cfg["kinematics_matrix"] == ["differential_drive"]

    def test_paper_facing_is_false(self) -> None:
        cfg = _load_yaml(STRESS)
        assert cfg["paper_facing"] is False

    def test_stop_on_failure_is_false(self) -> None:
        cfg = _load_yaml(STRESS)
        assert cfg["stop_on_failure"] is False

    def test_stress_scenario_count(self) -> None:
        cfg = _load_yaml(STRESS)
        matrix_path = ROOT / str(cfg["scenario_matrix"])
        matrix = _load_yaml(matrix_path)
        includes = matrix.get("includes", [])
        assert len(includes) >= 2


class TestIssue6095CrossConfig:
    """Cross-config contract checks."""

    def test_both_configs_share_same_seed_policy(self) -> None:
        nominal = _load_yaml(NOMINAL)
        stress = _load_yaml(STRESS)
        assert nominal["seed_policy"] == stress["seed_policy"]

    def test_both_configs_share_same_planner_rows(self) -> None:
        nominal = _load_yaml(NOMINAL)
        stress = _load_yaml(STRESS)
        assert nominal["planners"] == stress["planners"]

    def test_both_configs_share_horizon_dt_kinematics(self) -> None:
        nominal = _load_yaml(NOMINAL)
        stress = _load_yaml(STRESS)
        assert nominal["horizon"] == stress["horizon"] == 100
        assert nominal["dt"] == stress["dt"] == 0.1
        assert nominal["kinematics_matrix"] == stress["kinematics_matrix"]

    def test_expected_row_count_nominal(self) -> None:
        cfg = load_campaign_config(NOMINAL)
        scenarios = _load_campaign_scenarios(cfg)
        assert len(scenarios) == 4
        seed_count = len(S10_SEEDS)
        planner_count = len(cfg.planners)
        assert seed_count == 10
        assert planner_count == 2
        total_rows = len(scenarios) * seed_count * planner_count
        assert total_rows == 80

    def test_expected_row_count_stress(self) -> None:
        cfg = load_campaign_config(STRESS)
        scenarios = _load_campaign_scenarios(cfg)
        assert len(scenarios) == 48
        seed_count = len(S10_SEEDS)
        planner_count = len(cfg.planners)
        assert seed_count == 10
        assert planner_count == 2
        total_rows = len(scenarios) * seed_count * planner_count
        assert total_rows == 960

    def test_different_scenario_matrices(self) -> None:
        nominal = _load_yaml(NOMINAL)
        stress = _load_yaml(STRESS)
        assert nominal["scenario_matrix"] != stress["scenario_matrix"]
