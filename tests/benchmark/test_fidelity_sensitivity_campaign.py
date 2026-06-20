"""Tests for the bounded issue #3207 actual campaign runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
