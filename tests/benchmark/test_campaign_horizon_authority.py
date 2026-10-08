"""Native episode and large-grid controls for campaign budget enforcement."""

from copy import deepcopy
from dataclasses import replace

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from tests.benchmark.campaign_horizon_support import ROOT, TEMPLATE, _scheduled_context

# Per-identity authored oracle independently verified from YAML include/override bytes.


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
    # Label assertions precede new metadata, isolating the timeout-label regression.
    assert row["termination_reason"] == expected
    assert row["outcome"]["timeout_event"] == (signal in ("timeout", "early_timeout"))
    assert row["effective_budget_steps"] == 500
    assert row["scenario_params"]["run_horizon"] == 500


def test_all_scheduled_budgets_survive_rounding_sensitive_dt_grid(monkeypatch):
    """All 48 runner budgets agree with simulator limits on the 181-value dt grid.

    The previous 0.05/0.2 controls miss round-trip division just above an integer.
    This observes production context binding, without planner or environment steps.
    """
    import robot_sf.benchmark.map_runner.map_runner_episode as episode

    # Config construction (including SVG parsing) precedes the dt override and
    # does not depend on it. Build once per scenario, then clone the unbound config
    # so dt assignment and horizon binding still run independently at every point.
    build_env_config = episode._build_env_config
    configs = {}

    def cached_build_env_config(scenario, *, scenario_path):
        key = (scenario["name"], scenario_path)
        if key not in configs:
            configs[key] = build_env_config(scenario, scenario_path=scenario_path)
        return deepcopy(configs[key])

    monkeypatch.setattr(episode, "_build_env_config", cached_build_env_config)
    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    assert len(scenarios) == 48
    assert len({scenario["name"] for scenario in scenarios}) == len(scenarios)
    for scenario in scenarios:
        budget = scenario["simulation_config"]["max_episode_steps"]
        for millis in range(20, 201):
            dt = millis / 1000
            ctx = _scheduled_context(scenario, dt)
            assert ctx.horizon_val == budget
            assert ctx.config.sim_config.max_sim_steps == budget, (scenario["name"], dt, budget)


@pytest.mark.parametrize(
    ("name", "dt", "budget"),
    [
        ("francis2023_narrow_doorway", 0.052, 400),
        ("classic_bottleneck_low", 0.102, 500),
        ("classic_station_platform_medium", 0.118, 650),
        ("classic_realworld_double_bottleneck_high", 0.098, 700),
    ],
)
def test_rounding_sensitive_real_simulator_timeout(monkeypatch, name, dt, budget):
    """Observe the genuine simulator timeout before runner normalization, at dev seed 1001."""
    import numpy as np

    import robot_sf.benchmark.map_runner.map_runner_episode as episode

    cfg = load_campaign_config(TEMPLATE)
    scenario = next(
        s for s in _load_campaign_scenarios(cfg, repository_root=ROOT) if s["name"] == name
    )
    original = episode._step_collision_and_termination
    observed = []

    def observe(state, slc, *, step_idx, sim, **kwargs):
        if step_idx + 1 == budget:
            observed.append(step_idx + 1)
            assert sim.info["meta"]["max_sim_steps"] == budget
            assert sim.info["meta"]["is_timesteps_exceeded"]
            assert sim.terminated and not sim.truncated
            assert not sim.info["meta"]["is_route_complete"]
            assert not episode.collision_event(sim.info)
        return original(state, slc, step_idx=step_idx, sim=sim, **kwargs)

    def stationary(algo, config, **kwargs):
        return lambda obs: np.zeros(2), {"algorithm": algo, "config": config}

    monkeypatch.setattr(episode, "_step_collision_and_termination", observe)
    row = episode.run_map_episode(
        deepcopy(scenario),
        1001,
        horizon=0,
        dt=dt,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=ROOT / "scoped_scenarios.json",
        policy_builder=stationary,
    )
    assert observed == [budget]
    assert row["steps"] == row["horizon"] == row["effective_budget_steps"] == budget
    assert row["scenario_params"]["run_horizon"] == budget
    assert row["termination_reason"] == "max_steps"
    assert row["outcome"]["timeout_event"]


@pytest.mark.parametrize(
    ("config_name", "name", "algo", "seed", "steps", "reason", "avg_speed", "failure_to_progress"),
    [
        (
            "benchmark_data_2026_08",
            "francis2023_narrow_doorway",
            "orca",
            1001,
            400,
            "terminated",
            # Main 6fd6d463c (#10009): forward-axis ORCA projection.
            0.24059849301207314,
            232.0,
        ),
        (
            "benchmark_data_2026_08",
            "francis2023_narrow_doorway",
            "orca",
            1002,
            400,
            "terminated",
            # Main 6fd6d463c (#10009): forward-axis ORCA projection.
            0.23017620618943224,
            # Main 977378cbd (#10014): freeze terminal goal for metric-v2 scoring.
            241.0,
        ),
        (
            "runtime_smoke_v0_3",
            "francis2023_blind_corner",
            "goal",
            1001,
            263,
            "collision",
            0.94991463368817,
            0.0,
        ),
        (
            "runtime_smoke_v0_3",
            "francis2023_blind_corner",
            "goal",
            1002,
            262,
            "collision",
            0.9490333775512672,
            0.0,
        ),
        (
            "benchmark_data_2026_08",
            "classic_realworld_double_bottleneck_high",
            "social_force",
            1001,
            600,
            "max_steps",
            # Main 6fd6d463c (#10009): observe dt=0.1 instead of relaxation tau=0.5.
            0.21944634296423818,
            308.0,
        ),
    ],
)
def test_historical_runner_cap_matches_main_oracle(  # noqa: PLR0913
    config_name, name, algo, seed, steps, reason, avg_speed, failure_to_progress, monkeypatch
):
    """Native dev-seed oracle; changed planner values bisected through main 3a7a46a9."""
    import numpy as np

    import robot_sf.benchmark.map_runner.map_runner_episode as episode
    from robot_sf.benchmark.map_runner.map_runner import _build_policy
    from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode

    captured = {}
    teardown = episode._teardown_step_loop
    compute_metrics = episode._compute_post_loop_metrics

    def capture_targets(env, *args, **kwargs):
        captured["legacy_goal"] = np.array(env.simulator.goal_pos[0], copy=True)
        return teardown(env, *args, **kwargs)

    def capture_trajectory(**kwargs):
        captured.update(kwargs)
        return compute_metrics(**kwargs)

    monkeypatch.setattr(episode, "_teardown_step_loop", capture_targets)
    monkeypatch.setattr(episode, "_compute_post_loop_metrics", capture_trajectory)
    cfg = load_campaign_config(
        ROOT / "configs/benchmarks" / f"paper_experiment_matrix_v2_h600_s30_{config_name}.yaml"
    )
    scenario = next(
        s for s in _load_campaign_scenarios(cfg, repository_root=ROOT) if s["name"] == name
    )
    scenario["seeds"] = [seed]
    planner = next(p for p in cfg.planners if p.algo == algo)
    row = run_map_episode(
        scenario,
        seed,
        horizon=600,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo=algo,
        algo_config_path=planner.algo_config_path,
        scenario_path=ROOT / "scoped_scenarios.json",
        policy_builder=_build_policy,
    )
    main_episode_ids = {
        ("francis2023_narrow_doorway", 1001): "francis2023_narrow_doorway--1001--dd063c0bd8131283",
        ("francis2023_narrow_doorway", 1002): "francis2023_narrow_doorway--1002--ec38a96935d88b5e",
        ("francis2023_blind_corner", 1001): "francis2023_blind_corner--1001--553660886afe757a",
        ("francis2023_blind_corner", 1002): "francis2023_blind_corner--1002--1e9a60737c43cf9b",
        (
            "classic_realworld_double_bottleneck_high",
            1001,
        ): "classic_realworld_double_bottleneck_high--1001--6df9b00227cecdfe",
    }
    assert row["episode_id"] == main_episode_ids[(name, seed)]
    assert row["config_hash"] == main_episode_ids[(name, seed)].rsplit("--", 1)[1]
    assert row["steps"] == steps
    assert row["termination_reason"] == reason
    assert row["metrics"]["avg_speed"] == pytest.approx(avg_speed, rel=1e-12)
    if (
        row["metric_schema_version"] == "robot-sf-metrics.v2"
        and name == "classic_realworld_double_bottleneck_high"
    ):
        # #10014 freezes the final target; v1 used the current intermediate waypoint.
        # The same recorded trajectory must still reproduce the literal v1 oracle.
        positions = np.asarray(captured["robot_positions"])
        window = int(np.ceil(5.0 / 0.1))

        def count_for_goal(goal):
            distances = np.linalg.norm(positions - goal, axis=1)
            return sum(
                distances[i] - distances[i + window - 1] < 0.1
                for i in range(len(positions) - window + 1)
            )

        assert count_for_goal(captured["legacy_goal"]) == failure_to_progress
        assert not np.array_equal(captured["goal_vec"], captured["legacy_goal"])
        assert row["metrics"]["failure_to_progress"] == count_for_goal(captured["goal_vec"])
    else:
        assert row["metrics"]["failure_to_progress"] == failure_to_progress
    assert row["horizon"] == row["scenario_params"]["run_horizon"] == 600
    authored = scenario["simulation_config"]["max_episode_steps"]
    assert row["effective_budget_steps"] == min(authored, 600)
    assert row["metadata"]["scenario_horizon"] == {
        "policy": "legacy_runner_cap",
        "authored_max_episode_steps": authored,
        "runner_horizon": 600,
        "applied_max_episode_steps": min(authored, 600),
    }


def test_historical_authored_600_timeout_keeps_main_terminated_label():
    """Main labels the simulator's authored H600 stop terminated, even at the runner cap."""
    import numpy as np

    from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode

    cfg = load_campaign_config(
        ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"
    )
    scenario = next(
        s
        for s in _load_campaign_scenarios(cfg, repository_root=ROOT)
        if s["name"] == "classic_cross_trap_low"
    )
    scenario["seeds"] = [1001]

    def stationary(algo, config, **kwargs):
        return (lambda obs: np.zeros(2)), {"algorithm": algo, "config": config}

    row = run_map_episode(
        scenario,
        1001,
        horizon=600,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=ROOT / "scoped_scenarios.json",
        policy_builder=stationary,
    )
    assert row["steps"] == 600
    assert row["outcome"] == {
        "route_complete": False,
        "collision_event": False,
        "timeout_event": True,
    }
    assert row["termination_reason"] == "terminated"
