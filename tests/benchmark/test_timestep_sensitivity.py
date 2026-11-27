"""Regression coverage for the frozen development comparison."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark import timestep_sensitivity as study
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path


def test_expansion_preserves_the_release_roster_and_matched_development_grid(
    tmp_path: Path,
) -> None:
    """Changing a step size must preserve the scenario, seed and planner bindings."""
    study.main(["--output-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "timestep_manifest.json").read_text())
    root = get_repository_root()
    reference = yaml.safe_load(
        (
            root
            / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
        ).read_text()
    )
    expected_scenarios = {
        "classic_bottleneck_medium",
        "classic_doorway_medium",
        "classic_head_on_corridor_low",
        "classic_urban_crossing_medium",
        "francis2023_perpendicular_traffic",
    }
    assert len(manifest["campaigns"]) == 3
    for campaign, dt, steps in zip(
        manifest["campaigns"], (0.05, 0.1, 0.2), (1200, 600, 300), strict=True
    ):
        assert campaign["episode_count"] == 700
        assert campaign["analysis_only"]
        arms = campaign["arms"]
        assert {(arm["planner"], arm["scenario"]) for arm in arms} == {
            (planner["key"], scenario)
            for planner in reference["planners"]
            for scenario in expected_scenarios
        }
        assert len(arms) == 70
        for arm in arms:
            planner = next(p for p in reference["planners"] if p["key"] == arm["planner"])
            assert arm["algorithm"] == planner["algo"]
            assert arm["planner_config"] == planner.get("algo_config")
            assert arm["kinematics"] == "differential_drive"
            assert arm["seeds"] == list(range(1111, 1121))
            assert arm["simulation"] == {"interval_s": dt, "steps": steps, "horizon_s": 60.0}
            assert arm["controller_interval_s"] == arm["close_clearance_sample_interval_s"] == dt
            if arm["algorithm"] in {"prediction_planner", "predictive_mppi"}:
                assert arm["planner_timing"]["forecast"] == {
                    "interval_s": 0.1,
                    "steps": 8,
                    "horizon_s": 0.8,
                }
            if arm["algorithm"] == "socnav_sampling":
                assert arm["planner_timing"]["sampled_controls"] == {
                    "interval_s": dt,
                    "steps": {0.05: 40, 0.1: 20, 0.2: 10}[dt],
                    "horizon_s": 2.0,
                }


def test_witness_retains_native_steps_and_existing_runner_refusals(tmp_path: Path) -> None:
    """Make a native goal witness and retain a normal missing-planner error in the report."""
    cfg = load_campaign_config(get_repository_root() / study.CONFIG_PATTERN.format(ms=200))
    goal = next(planner for planner in cfg.planners if planner.key == "goal")
    unavailable = replace(goal, key="unavailable", algo_config_path=tmp_path / "missing.yaml")
    cfg = replace(cfg, planners=(goal, unavailable))
    results = study._run_witness(cfg, _load_campaign_scenarios(cfg), tmp_path)

    native, refused = results
    assert native["status"] == "qualified"
    assert native["execution_mode"] == "native"
    assert native["timestamps_s"] == [0.2, 0.4]
    assert native["seed"] == 1111
    assert refused["status"] == "refused"
    assert "Algorithm config file not found" in refused["reason"]


def test_expansion_identifies_an_omitted_campaign_step() -> None:
    """A campaign with no step size cannot report its timing contract."""
    cfg = load_campaign_config(get_repository_root() / study.CONFIG_PATTERN.format(ms=100))
    cfg = replace(cfg, dt=None)
    with pytest.raises(ValueError, match="requires an explicit campaign dt"):
        study.expand_timestep_campaign(cfg, _load_campaign_scenarios(cfg))
