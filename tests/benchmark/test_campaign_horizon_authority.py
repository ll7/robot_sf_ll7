"""Campaign episode budgets must reach the simulator without hidden scenario caps."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.benchmark.camera_ready.campaign import _prepare_campaign_planner_variant_run
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)


def test_real_release_0_0_8_template_has_600_step_simulator_budget():
    """The actual release matrix must give all 48 scenarios the declared H600 budget."""
    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    authored = {
        s["name"]: s["simulation_config"]["max_episode_steps"]
        for s in _load_campaign_scenarios(replace(cfg, horizon=None), repository_root=ROOT)
    }
    assert cfg.horizon == 600
    assert len(scenarios) == 48
    for scenario in scenarios:
        config = build_env_config(scenario, scenario_path=ROOT / "scoped_scenarios.json")
        assert config.sim_config.max_sim_steps == cfg.horizon, scenario["name"]
        assert scenario["simulation_config"]["max_episode_steps"] == cfg.horizon
        assert scenario["metadata"]["campaign_horizon"] == {
            "mode": "fixed",
            "horizon_steps": 600,
            "authored_max_episode_steps": authored[scenario["name"]],
        }


@pytest.mark.parametrize("override", [None, 500, 700])
def test_planner_budget_matches_scoped_scenarios_without_mutating_inputs(tmp_path, override):
    """Planner overrides reach both execution modes through their shared scoped scenario list."""
    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    original = deepcopy(scenarios)
    planner = replace(cfg.planners[0], horizon_override=override)
    context = SimpleNamespace(cfg=cfg, runs_dir=tmp_path, scenarios=scenarios)
    run = _prepare_campaign_planner_variant_run(
        context,
        planner=planner,
        kinematics="differential_drive",
        active_observation_mode="socnav_state",
        log_run=False,
    )
    expected = override if override is not None else cfg.horizon
    assert run.effective_horizon == expected
    for scenario in run.scoped_scenarios:
        assert scenario["simulation_config"]["max_episode_steps"] == expected, scenario["name"]
        assert scenario["metadata"]["campaign_horizon"]["horizon_steps"] == expected
        source = next(s for s in original if s["name"] == scenario["name"])
        assert (
            scenario["metadata"]["campaign_horizon"]["authored_max_episode_steps"]
            == (source["metadata"]["campaign_horizon"]["authored_max_episode_steps"])
        )
        config = build_env_config(scenario, scenario_path=ROOT / "scoped_scenarios.json")
        assert config.sim_config.max_sim_steps == expected
    assert scenarios == original


def test_absent_fixed_budget_preserves_real_scenario_limits():
    """Scenario-controlled campaigns retain authored limits when no fixed budget is requested."""
    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(replace(cfg, horizon=None), repository_root=ROOT)
    assert any(s["simulation_config"]["max_episode_steps"] == 400 for s in scenarios)
    assert any(s["simulation_config"]["max_episode_steps"] == 500 for s in scenarios)


@pytest.mark.parametrize("dt", [0.05, 0.2])
def test_campaign_budget_survives_runner_timestep_override(dt):
    """Converting a fixed budget to seconds must use the actual runner timestep."""
    from robot_sf.benchmark.map_runner.map_runner_episode import _resolve_episode_run_context

    cfg = load_campaign_config(TEMPLATE)
    scenario = _load_campaign_scenarios(cfg, repository_root=ROOT)[0]
    ctx = _resolve_episode_run_context(
        scenario=scenario,
        seed=103,
        horizon=cfg.horizon,
        dt=dt,
        algo="goal",
        scenario_path=ROOT / "scoped_scenarios.json",
        algo_config=None,
        algo_config_path=None,
        experimental_ped_impact=False,
        ped_impact_radius_m=2.0,
        ped_impact_window_steps=5,
        observation_mode=None,
        observation_level=None,
        benchmark_track=None,
        track_schema_version=None,
        observation_noise=None,
        tracking_precision=None,
        synthetic_actuation_profile=None,
        latency_stress_profile=None,
        safety_wrapper=None,
        cbf_safety_filter=None,
    )
    assert ctx.horizon_val == 600
    assert ctx.config.sim_config.max_sim_steps == 600
    assert ctx.config.sim_config.sim_time_in_secs == pytest.approx(600 * dt)
    assert ctx.scenario["simulation_config"]["max_episode_steps"] == 600
