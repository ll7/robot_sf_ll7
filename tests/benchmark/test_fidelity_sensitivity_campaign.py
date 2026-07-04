"""Tests for the bounded issue #3207 actual campaign runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "research" / "fidelity_sensitivity_v1.yaml"


def _load_campaign_runner() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "benchmark" / "run_fidelity_sensitivity_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "fidelity_sensitivity_campaign_runner", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign runner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fidelity_sensitivity_campaign_runner"] = module
    spec.loader.exec_module(module)
    return module


campaign_runner = _load_campaign_runner()


def test_robot_env_imports_without_predictive_torch_dependency() -> None:
    """Default RobotEnv import must not require the optional predictive model stack."""
    from robot_sf.gym_env.robot_env import RobotEnv

    assert RobotEnv is not None


def test_variant_specs_bind_all_declared_fidelity_axes_to_runtime_effects() -> None:
    """The actual runner should execute real variant bindings, not dry-run payloads."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    variants = campaign_runner.load_variant_specs(config, include_all_variants=True)

    axes = {variant.axis for variant in variants if not variant.baseline}
    bindings = {variant.runtime_binding for variant in variants}

    assert {"integration_timestep", "social_force_speed_archetypes", "observation_noise"} <= axes
    assert "clearance_radius" in axes
    assert "sim_config.time_per_step_in_secs" in bindings
    assert "sim_config.archetype_composition" in bindings
    assert "planner_observation_noise" in bindings
    assert "unsupported" not in bindings


def test_timestep_variant_preserves_simulated_duration() -> None:
    """Changing integration timestep should not change the simulated time budget."""
    config = SimpleNamespace(
        sim_config=SimpleNamespace(
            time_per_step_in_secs=0.1,
            sim_time_in_secs=18.0,
            max_sim_steps=180,
        )
    )
    variant = campaign_runner.VariantSpec(
        axis="integration_timestep",
        key="integration_timestep__dt_0_05",
        source_key="dt_0_05",
        baseline=False,
        patch={"dt": 0.05},
        observation_noise={},
        runtime_binding="sim_config.time_per_step_in_secs",
    )

    campaign_runner.apply_variant(config, variant, seed=111)

    assert config.sim_config.time_per_step_in_secs == 0.05
    assert config.sim_config.sim_time_in_secs == 18.0


def test_timestep_episode_cap_preserves_horizon_duration() -> None:
    """The episode loop should compare equal simulated durations across dt variants."""
    step_count = 0

    def _fake_build_robot_config_from_scenario(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            sim_config=SimpleNamespace(
                max_sim_steps=180,
                sim_time_in_secs=18.0,
                time_per_step_in_secs=0.1,
                ped_radius=0.3,
            ),
            robot_config=SimpleNamespace(max_linear_speed=1.0, max_angular_speed=1.0),
        )

    class _FakeEnv:
        env_config = SimpleNamespace(
            sim_config=SimpleNamespace(time_per_step_in_secs=0.05, ped_radius=0.3)
        )

        def __init__(self) -> None:
            self.simulator = SimpleNamespace(
                robots=[
                    SimpleNamespace(
                        config=SimpleNamespace(radius=0.4),
                        pose=[None, 0.0],
                        current_speed=(0.0, 0.0),
                    )
                ],
                robot_pos=[[0.0, 0.0]],
                goal_pos=[[1.0, 0.0]],
                ped_pos=[],
                ped_vel=[],
            )

        def reset(self, *, seed: int) -> None:
            del seed

        def step(self, action):
            nonlocal step_count
            del action
            step_count += 1
            return None, 0.0, False, False, {"meta": {}}

        def close(self) -> None:
            pass

    original_build = campaign_runner.build_robot_config_from_scenario
    original_make_env = campaign_runner.make_robot_env
    try:
        campaign_runner.build_robot_config_from_scenario = _fake_build_robot_config_from_scenario
        campaign_runner.make_robot_env = lambda **kwargs: _FakeEnv()
        campaign_runner.run_episode(
            {"name": "unit"},
            scenario_path=Path("dummy.yaml"),
            variant=campaign_runner.VariantSpec(
                axis="integration_timestep",
                key="integration_timestep__dt_0_05",
                source_key="dt_0_05",
                baseline=False,
                patch={"dt": 0.05},
                observation_noise={},
                runtime_binding="sim_config.time_per_step_in_secs",
            ),
            planner_name="goal_seek",
            seed=111,
            horizon=180,
        )
    finally:
        campaign_runner.build_robot_config_from_scenario = original_build
        campaign_runner.make_robot_env = original_make_env

    assert step_count == 360


def test_surface_clearance_subtracts_robot_and_pedestrian_radii() -> None:
    """Clearance metrics should report surface margin rather than center distance."""
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            ped_pos=[[3.0, 0.0], [5.0, 0.0]],
            robot_pos=[[0.0, 0.0]],
            robots=[SimpleNamespace(config=SimpleNamespace(radius=0.4))],
        ),
        env_config=SimpleNamespace(sim_config=SimpleNamespace(ped_radius=0.3)),
    )

    clearances = campaign_runner._surface_clearances(env)

    assert clearances.tolist() == pytest.approx([2.3, 4.3])


def test_success_requires_route_success_without_collision() -> None:
    """Future evidence rows must not count colliding route completion as success."""

    def _fake_build_robot_config_from_scenario(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            sim_config=SimpleNamespace(
                max_sim_steps=5,
                sim_time_in_secs=0.5,
                time_per_step_in_secs=0.1,
                ped_radius=0.3,
            ),
            robot_config=SimpleNamespace(max_linear_speed=1.0, max_angular_speed=1.0),
        )

    class _FakeEnv:
        env_config = SimpleNamespace(
            sim_config=SimpleNamespace(time_per_step_in_secs=0.1, ped_radius=0.3)
        )

        def __init__(self) -> None:
            self.simulator = SimpleNamespace(
                robots=[
                    SimpleNamespace(
                        config=SimpleNamespace(radius=0.4),
                        pose=[None, 0.0],
                        current_speed=(0.0, 0.0),
                    )
                ],
                robot_pos=[[0.0, 0.0]],
                goal_pos=[[1.0, 0.0]],
                ped_pos=[[2.0, 0.0]],
                ped_vel=[[0.0, 0.0]],
            )

        def reset(self, *, seed: int) -> None:
            del seed

        def step(self, action):
            del action
            return (
                None,
                0.0,
                True,
                False,
                {
                    "collision": True,
                    "is_success": True,
                    "meta": {"distance_to_goal": 0.0},
                },
            )

        def close(self) -> None:
            pass

    original_build = campaign_runner.build_robot_config_from_scenario
    original_make_env = campaign_runner.make_robot_env
    try:
        campaign_runner.build_robot_config_from_scenario = _fake_build_robot_config_from_scenario
        campaign_runner.make_robot_env = lambda **kwargs: _FakeEnv()
        row = campaign_runner.run_episode(
            {"name": "unit"},
            scenario_path=Path("dummy.yaml"),
            variant=campaign_runner.VariantSpec(
                axis="baseline",
                key="baseline",
                source_key="nominal",
                baseline=True,
                patch={},
                observation_noise={},
                runtime_binding="baseline",
            ),
            planner_name="goal_seek",
            seed=111,
            horizon=5,
        )
    finally:
        campaign_runner.build_robot_config_from_scenario = original_build
        campaign_runner.make_robot_env = original_make_env

    assert row["route_success"] is True
    assert row["success"] is False
    assert row["metrics"]["success_rate"] == 0.0


def test_metrics_csv_uses_lf_line_endings(tmp_path: Path) -> None:
    """CSV evidence must satisfy git diff whitespace checks."""
    report = {
        "aggregates": {
            "baseline": {
                "goal_seek": {
                    "success_rate": 0.0,
                    "collision_rate": 0.0,
                    "min_clearance": 1.0,
                    "mean_clearance": 2.0,
                    "near_miss_rate": 0.0,
                    "comfort_exposure_mean": 0.0,
                    "time_to_goal_norm": 1.0,
                }
            }
        },
        "claim_boundary": "bounded",
        "date": "2026-06-20",
        "git_head": "abc1234",
        "git_worktree_dirty": True,
        "rank_stability": {
            "axes": [],
            "flipping_axes": [],
            "nominal_ranking": ["goal_seek"],
            "rank_stable": True,
        },
        "raw_rows_path": "output/example/episode_rows.jsonl",
        "result_caveats": ["unit_test"],
        "scenario_set": "configs/scenarios/sets/paper_cross_kinematics_v1.yaml",
        "scope": {
            "classification": "bounded_actual_slice",
            "episode_count": 1,
            "horizon": 1,
            "not_full_fixed_scope_reason": "unit test",
            "planners": ["goal_seek"],
            "seeds": [111],
        },
        "status": "actual_campaign_slice",
    }

    campaign_runner.write_outputs(
        rows=[],
        report=report,
        raw_root=tmp_path / "raw",
        evidence_dir=tmp_path / "evidence",
    )

    assert b"\r\n" not in (tmp_path / "evidence" / "planner_variant_metrics.csv").read_bytes()


def test_report_classifies_actual_slice_without_paper_or_sim_to_real_claim(tmp_path: Path) -> None:
    """Compact evidence should classify the bounded slice and preserve claim boundaries."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    variants = campaign_runner.load_variant_specs(config, include_all_variants=False)
    rows = []
    for variant in variants:
        for planner, success in (("goal_seek", 1.0), ("baseline_social_force", 0.0)):
            rows.append(
                {
                    "variant": variant.key,
                    "planner": planner,
                    "scenario_id": "classic_cross_trap_low",
                    "seed": 111,
                    "metrics": {
                        "success_rate": success,
                        "collision_rate": 0.0,
                        "min_clearance": 1.0 + success,
                        "mean_clearance": 2.0 + success,
                        "near_miss_rate": 0.0,
                        "comfort_exposure_mean": 0.0,
                        "time_to_goal_norm": 1.0,
                    },
                }
            )

    report = campaign_runner.build_report(
        config=config,
        rows=rows,
        variants=variants,
        scenario_set="configs/scenarios/sets/paper_cross_kinematics_v1.yaml",
        horizon=12,
        raw_rows_path=tmp_path / "episode_rows.jsonl",
        git_provenance={
            "git_head": "abc1234",
            "git_worktree_dirty": True,
            "git_status_short_at_generation": [" M example.py"],
        },
        date="2026-06-20",
    )

    assert report["status"] == "actual_campaign_slice"
    assert report["scope"]["classification"] == "bounded_actual_slice"
    assert report["git_worktree_dirty"] is True
    assert "not simulator-realism" in report["claim_boundary"]
    assert "sim-to-real" in report["claim_boundary"]
    assert report["rank_stability"]["rank_stable"] is True


def test_build_report_marks_zero_success_slice_rank_non_identifiable(tmp_path: Path) -> None:
    """All-zero success rates are metric drift evidence, not rank-stability evidence."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    variants = campaign_runner.load_variant_specs(config, include_all_variants=False)
    rows = []
    for variant in variants:
        for planner in ("goal_seek", "baseline_social_force"):
            rows.append(
                {
                    "variant": variant.key,
                    "planner": planner,
                    "scenario_id": "classic_cross_trap_low",
                    "seed": 111,
                    "metrics": {
                        "success_rate": 0.0,
                        "collision_rate": 1.0,
                        "min_clearance": 0.0,
                        "mean_clearance": 0.0,
                        "near_miss_rate": 1.0,
                        "comfort_exposure_mean": 1.0,
                        "time_to_goal_norm": 1.0,
                    },
                }
            )

    report = campaign_runner.build_report(
        config=config,
        rows=rows,
        variants=variants,
        scenario_set="configs/scenarios/sets/paper_cross_kinematics_v1.yaml",
        horizon=12,
        raw_rows_path=tmp_path / "episode_rows.jsonl",
        git_provenance={
            "git_head": "abc1234",
            "git_worktree_dirty": True,
            "git_status_short_at_generation": [" M example.py"],
        },
        date="2026-06-20",
    )

    rank_stability = report["rank_stability"]
    assert rank_stability["rank_identifiable"] is False
    assert rank_stability["rank_identifiability_reason"] == "primary_metric_zero_variance"
    assert rank_stability["rank_stable"] is None
    assert rank_stability["flipping_axes"] == []
    assert rank_stability["non_identifiable_axes"]
    assert "rank_non_identifiable_primary_metric_zero_variance" in report["result_caveats"]

    markdown = campaign_runner.format_markdown(report)
    first_axis = rank_stability["non_identifiable_axes"][0]
    assert "Rank evidence status: `non-identifiable`" in markdown
    assert (
        f"| `{first_axis}` | `non-identifiable: primary_metric_zero_variance` | NA | NA | NA |"
        in markdown
    )
    assert "primary_metric_zero_variance" in markdown


def test_fixed_scope_runner_binds_remaining_planner_prerequisites() -> None:
    """ORCA and hybrid cells bind once rvo2/opt-in gates are represented as satisfied."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    variant_index = campaign_runner.build_fixed_scope_variant_index(config)
    cells = [
        campaign_runner.FixedScopeRunCell(
            planner_group="orca",
            algorithm="orca",
            planner_available=True,
            planner_tier="baseline-ready",
            requires_explicit_opt_in=False,
            axis="integration_timestep",
            variant="dt_0_05",
            baseline_variant=False,
            seed=111,
            scenario_set="configs/benchmarks/paper_experiment_matrix_v1.yaml",
        ),
        campaign_runner.FixedScopeRunCell(
            planner_group="default_social_force",
            algorithm="social_force",
            planner_available=True,
            planner_tier="baseline-ready",
            requires_explicit_opt_in=False,
            axis="integration_timestep",
            variant="dt_0_05",
            baseline_variant=False,
            seed=111,
            scenario_set="configs/benchmarks/paper_experiment_matrix_v1.yaml",
        ),
        campaign_runner.FixedScopeRunCell(
            planner_group="hybrid_rule_v0_minimal",
            algorithm="hybrid_rule_local_planner",
            planner_available=True,
            planner_tier="experimental",
            requires_explicit_opt_in=False,
            axis="integration_timestep",
            variant="dt_0_05",
            baseline_variant=False,
            seed=111,
            scenario_set="configs/benchmarks/paper_experiment_matrix_v1.yaml",
        ),
    ]
    bindings = [campaign_runner.bind_fixed_scope_run_cell(cell, variant_index) for cell in cells]
    assert [binding.runner_bound for binding in bindings] == [True, True, True]
    assert [binding.planner_name for binding in bindings] == [
        "orca",
        "baseline_social_force",
        "hybrid_rule_v0_minimal",
    ]
