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


def test_real_release_0_0_8_template_preserves_authored_budgets():
    """All 48 real release scenarios use the explicitly pinned authored schedule."""
    from collections import Counter
    from hashlib import sha256

    from scripts.tools.generate_authored_horizon_schedule import authored_schedule_bytes

    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    raw_cfg = replace(cfg, horizon=None, scenario_horizons_path=None, scenario_horizons_sha256=None)
    authored = {
        s["name"]: s["simulation_config"]["max_episode_steps"]
        for s in _load_campaign_scenarios(raw_cfg, repository_root=ROOT)
    }
    assert cfg.horizon is None
    assert cfg.scenario_horizons_path is not None
    schedule_bytes = cfg.scenario_horizons_path.read_bytes()
    assert sha256(schedule_bytes).hexdigest() == cfg.scenario_horizons_sha256
    assert authored_schedule_bytes(TEMPLATE, repository_root=ROOT) == schedule_bytes
    assert len(scenarios) == 48
    assert Counter(authored.values()) == {400: 25, 500: 13, 600: 8, 650: 1, 700: 1}
    for scenario in scenarios:
        expected = authored[scenario["name"]]
        config = build_env_config(scenario, scenario_path=ROOT / "scoped_scenarios.json")
        assert config.sim_config.max_sim_steps == expected, scenario["name"]
        assert scenario["simulation_config"]["max_episode_steps"] == expected
        assert scenario["metadata"]["scenario_horizon"]["authored_max_episode_steps"] == expected
        assert scenario["metadata"]["scenario_horizon"]["sha256"] == cfg.scenario_horizons_sha256
        assert "campaign_horizon" not in scenario["metadata"]


@pytest.mark.parametrize("arm_override", [False, True])
def test_undeclared_shorter_scenario_limit_is_refused(arm_override):
    """Neither campaign nor arm admission may silently extend authored limits."""
    cfg = load_campaign_config(TEMPLATE)
    cfg = replace(
        cfg,
        scenario_horizons_path=None,
        scenario_horizons_sha256=None,
        horizon=None if arm_override else 600,
    )
    if arm_override:
        cfg = replace(cfg, planners=(replace(cfg.planners[0], horizon_override=600),))
    with pytest.raises(ValueError, match="below fixed horizon.*declare scenario_horizons"):
        _load_campaign_scenarios(cfg, repository_root=ROOT)


def test_schedule_hash_drift_is_refused_after_config_load(tmp_path):
    """Pin enforcement happens again at scenario preparation, detecting changed sidecar bytes."""
    cfg = load_campaign_config(TEMPLATE)
    changed = tmp_path / "schedule.yaml"
    changed.write_bytes(cfg.scenario_horizons_path.read_bytes().replace(b": 500", b": 501", 1))
    with pytest.raises(ValueError, match="scenario_horizons_sha256"):
        _load_campaign_scenarios(replace(cfg, scenario_horizons_path=changed), repository_root=ROOT)


def test_planner_schedule_preserves_limits_without_mutating_inputs(tmp_path):
    """The shared prepared list used by both execution modes retains the full schedule."""
    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    original = deepcopy(scenarios)
    run = _prepare_campaign_planner_variant_run(
        SimpleNamespace(cfg=cfg, runs_dir=tmp_path, scenarios=scenarios),
        planner=cfg.planners[0],
        kinematics="differential_drive",
        active_observation_mode="socnav_state",
        log_run=False,
    )
    assert run.effective_horizon is None
    assert [s["simulation_config"] for s in run.scoped_scenarios] == [
        s["simulation_config"] for s in scenarios
    ]
    assert scenarios == original


@pytest.mark.parametrize("dt", [0.05, 0.2])
def test_scheduled_budget_survives_runner_timestep_override(dt):
    """Scheduled steps must convert to seconds after the effective timestep override."""
    from robot_sf.benchmark.map_runner.map_runner_episode import _resolve_episode_run_context

    cfg = load_campaign_config(TEMPLATE)
    scenario = _load_campaign_scenarios(cfg, repository_root=ROOT)[0]
    expected = scenario["simulation_config"]["max_episode_steps"]
    ctx = _resolve_episode_run_context(
        scenario=scenario,
        seed=103,
        horizon=0,
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
    assert ctx.horizon_val == expected
    assert ctx.config.sim_config.max_sim_steps == expected
    assert ctx.config.sim_config.sim_time_in_secs == pytest.approx(expected * dt)


@pytest.mark.parametrize(
    ("signal", "terminal_step", "expected"),
    [
        ("timeout", 500, "max_steps"),
        ("collision", 500, "collision"),
        ("success", 500, "success"),
        ("collision_and_success", 500, "collision"),
        ("early_timeout", 20, "terminated"),
        ("intentional", 20, "terminated"),
        ("intentional", 500, "terminated"),
    ],
)
def test_real_simulator_budget_timeout_and_terminal_controls(
    monkeypatch, signal, terminal_step, expected
):
    """Observe a real H500 RobotEnv timeout; injected info controls preserve terminal precedence."""
    import numpy as np

    import robot_sf.benchmark.map_runner.map_runner_episode as episode

    cfg = load_campaign_config(TEMPLATE)
    scenario = next(
        s
        for s in _load_campaign_scenarios(cfg, repository_root=ROOT)
        if s["name"] == "classic_bottleneck_low"
    )
    original = episode._step_collision_and_termination
    observed = []

    def observe_terminal(state, slc, *, step_idx, sim, **kwargs):
        if step_idx + 1 == terminal_step:
            if terminal_step == 500:
                # Require RobotEnv itself to terminate on timeout before changing any control.
                assert sim.terminated and not sim.truncated
                assert sim.info["meta"]["is_timesteps_exceeded"]
                assert not sim.info["meta"]["is_route_complete"]
                assert not episode.collision_event(sim.info)
            observed.append(step_idx + 1)
            if signal != "timeout":
                info = deepcopy(sim.info)
                info["meta"]["is_timesteps_exceeded"] = signal != "intentional"
                info["collision"] = signal in ("collision", "collision_and_success")
                info["meta"]["is_route_complete"] = signal in ("success", "collision_and_success")
                sim = replace(sim, info=info, terminated=True)
        return original(state, slc, step_idx=step_idx, sim=sim, **kwargs)

    def stationary_builder(algo, config, **kwargs):
        return lambda obs: np.zeros(2), {
            "algorithm": algo,
            "config": config,
            "diagnostic_policy": "stationary injected policy; not planner evidence",
        }

    monkeypatch.setattr(episode, "_step_collision_and_termination", observe_terminal)
    row = episode.run_map_episode(
        deepcopy(scenario),
        1001,
        horizon=0,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=ROOT / "scoped_scenarios.json",
        policy_builder=stationary_builder,
        record_simulation_step_trace=True,
    )
    assert observed == [terminal_step]
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    assert len(trace["steps"]) == terminal_step
    assert row["scenario_params"]["simulation_config"]["max_episode_steps"] == 500
    assert row["horizon"] == 500
    assert row["effective_budget_steps"] == 500
    assert row["termination_reason"] == expected
    assert row["outcome"]["timeout_event"] == (signal in ("timeout", "early_timeout"))


def test_schedule_refuses_fixed_arm_override(tmp_path):
    """An explicit schedule cannot be silently replaced by an arm's fixed horizon."""
    cfg = load_campaign_config(TEMPLATE)
    context = SimpleNamespace(
        cfg=cfg, runs_dir=tmp_path, scenarios=_load_campaign_scenarios(cfg, repository_root=ROOT)
    )
    with pytest.raises(ValueError, match="scenario_horizons cannot be combined"):
        _prepare_campaign_planner_variant_run(
            context,
            planner=replace(cfg.planners[0], horizon_override=700),
            kinematics="differential_drive",
            active_observation_mode="socnav_state",
            log_run=False,
        )
