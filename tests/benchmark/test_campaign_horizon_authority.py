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

# Per-identity authored oracle independently verified from YAML include/override bytes.
AUTHORED_BUDGETS = {
    "classic_bottleneck_high": 500,
    "classic_bottleneck_low": 500,
    "classic_bottleneck_medium": 500,
    "classic_cross_trap_high": 600,
    "classic_cross_trap_low": 600,
    "classic_cross_trap_medium": 600,
    "classic_doorway_high": 500,
    "classic_doorway_low": 500,
    "classic_doorway_medium": 500,
    "classic_group_crossing_high": 500,
    "classic_group_crossing_low": 500,
    "classic_group_crossing_medium": 500,
    "classic_head_on_corridor_low": 500,
    "classic_head_on_corridor_medium": 500,
    "classic_merging_low": 600,
    "classic_merging_medium": 600,
    "classic_overtaking_low": 600,
    "classic_overtaking_medium": 600,
    "classic_realworld_double_bottleneck_high": 700,
    "classic_station_platform_medium": 650,
    "classic_t_intersection_low": 500,
    "classic_t_intersection_medium": 500,
    "classic_urban_crossing_medium": 600,
    "francis2023_accompanying_peer": 400,
    "francis2023_blind_corner": 400,
    "francis2023_circular_crossing": 400,
    "francis2023_crowd_navigation": 400,
    "francis2023_down_path": 400,
    "francis2023_entering_elevator": 400,
    "francis2023_entering_room": 400,
    "francis2023_exiting_elevator": 400,
    "francis2023_exiting_room": 400,
    "francis2023_following_human": 400,
    "francis2023_frontal_approach": 400,
    "francis2023_intersection_no_gesture": 400,
    "francis2023_intersection_proceed": 400,
    "francis2023_intersection_wait": 400,
    "francis2023_join_group": 400,
    "francis2023_leading_human": 400,
    "francis2023_leave_group": 400,
    "francis2023_narrow_doorway": 400,
    "francis2023_narrow_hallway": 400,
    "francis2023_parallel_traffic": 400,
    "francis2023_pedestrian_obstruction": 400,
    "francis2023_pedestrian_overtaking": 400,
    "francis2023_perpendicular_traffic": 400,
    "francis2023_robot_crowding": 400,
    "francis2023_robot_overtaking": 400,
}


def test_real_release_0_0_8_template_preserves_authored_budgets():
    """All 48 real release scenarios use the explicitly pinned authored schedule."""
    from collections import Counter
    from hashlib import sha256

    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    authored = AUTHORED_BUDGETS
    assert {s["name"] for s in scenarios} == set(authored)
    assert cfg.horizon is None
    assert cfg.scenario_horizons_path is not None
    schedule_bytes = cfg.scenario_horizons_path.read_bytes()
    assert len(scenarios) == 48
    assert Counter(authored.values()) == {400: 25, 500: 13, 600: 8, 650: 1, 700: 1}
    for scenario in scenarios:
        expected = authored[scenario["name"]]
        config = build_env_config(scenario, scenario_path=ROOT / "scoped_scenarios.json")
        assert config.sim_config.max_sim_steps == expected, scenario["name"]
        assert scenario["simulation_config"]["max_episode_steps"] == expected
    # Budget preservation above is independent of the new pin/provenance API.
    # A missing field on base is a provenance red, not a budget-regression red.
    assert sha256(schedule_bytes).hexdigest() == cfg.scenario_horizons_sha256
    for scenario in scenarios:
        expected = authored[scenario["name"]]
        assert scenario["metadata"]["scenario_horizon"]["authored_max_episode_steps"] == expected
        assert scenario["metadata"]["scenario_horizon"]["sha256"] == cfg.scenario_horizons_sha256
        assert "campaign_horizon" not in scenario["metadata"]

    # Generator parity is a separate preservation check, after the independent budget oracle.
    from scripts.tools.generate_authored_horizon_schedule import authored_schedule_bytes

    assert authored_schedule_bytes(TEMPLATE, repository_root=ROOT) == schedule_bytes


@pytest.mark.parametrize("arm_override", [False, True])
def test_undeclared_shorter_scenario_limit_is_refused(tmp_path, arm_override):
    """Neither campaign nor arm admission may silently extend authored limits."""
    import yaml

    raw = yaml.safe_load(TEMPLATE.read_text())
    for field in (
        "scenario_matrix",
        "comparability_mapping",
        "route_clearance_certifications",
        "snqi_weights",
        "snqi_baseline",
    ):
        if field in raw:
            raw[field] = str(ROOT / raw[field])
    raw.pop("scenario_horizons", None)
    raw.pop("scenario_horizons_sha256", None)
    raw["horizon"] = None if arm_override else 600
    raw["seed_policy"] = {"mode": "fixed-list", "seeds": [1001]}
    raw["planners"] = [{"key": "goal", "algo": "goal", "planner_group": "core"}]
    if arm_override:
        raw["planners"][0]["horizon"] = 600
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_campaign_config(path, repository_root=ROOT)
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
    """Preservation control: the shared prepared list retains the schedule and input bytes.

    This does not execute either process mode or isolate a new base regression.
    """
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
    cfg = load_campaign_config(TEMPLATE)
    scenario = _load_campaign_scenarios(cfg, repository_root=ROOT)[0]
    expected = scenario["simulation_config"]["max_episode_steps"]
    ctx = _scheduled_context(scenario, dt)
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
    # Label assertions precede new metadata, isolating the timeout-label regression.
    assert row["termination_reason"] == expected
    assert row["outcome"]["timeout_event"] == (signal in ("timeout", "early_timeout"))
    assert row["effective_budget_steps"] == 500
    assert row["scenario_params"]["run_horizon"] == 500


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


def _scheduled_context(scenario, dt):
    """Resolve the production runner context without executing an episode."""
    from robot_sf.benchmark.map_runner.map_runner_episode import _resolve_episode_run_context

    return _resolve_episode_run_context(
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


def test_all_scheduled_budgets_survive_rounding_sensitive_dt_grid():
    """All 48 runner budgets agree with simulator limits on the 181-value dt grid.

    The previous 0.05/0.2 controls miss round-trip division just above an integer.
    This observes production context binding, without planner or environment steps.
    """
    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    assert len(scenarios) == 48
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


def test_scheduled_identity_records_resolved_run_horizon():
    """Existing run_horizon consumers receive every scheduled budget, even with no fixed horizon."""
    from robot_sf.benchmark.map_runner.map_runner_identity import scenario_identity_payload

    cfg = load_campaign_config(TEMPLATE)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    for scenario in scenarios:
        payload = scenario_identity_payload(
            scenario, algo="goal", algo_config={}, horizon=0, dt=0.1, record_forces=False
        )
        assert payload.get("run_horizon") == scenario["simulation_config"]["max_episode_steps"]


@pytest.mark.parametrize("duration", [10.001, 10.05, 10.099, 10.1])
def test_duration_only_simulators_keep_ceiling_semantics(duration):
    """A genuine fractional duration still admits the next whole step (preservation control)."""
    from math import ceil

    from robot_sf.robot.robot_state import RobotState
    from robot_sf.sim.sim_config import SimulationSettings

    settings = SimulationSettings(sim_time_in_secs=duration, time_per_step_in_secs=0.1)
    state = RobotState(None, None, None, 0.1, duration)
    assert settings.max_sim_steps == state.max_sim_steps == ceil(duration / 0.1)


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
            0.24086784179045043,
            230.0,
        ),
        (
            "benchmark_data_2026_08",
            "francis2023_narrow_doorway",
            "orca",
            1002,
            400,
            "terminated",
            0.23080397754459608,
            262.0,
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
            1.452342043961671,
            253.0,
        ),
    ],
)
def test_historical_runner_cap_matches_main_oracle(
    config_name, name, algo, seed, steps, reason, avg_speed, failure_to_progress
):
    """Literal oracle captured from main 93ba0d75 with native planners on dev seeds."""
    from robot_sf.benchmark.map_runner.map_runner import _build_policy
    from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode

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


@pytest.mark.parametrize("version", [None, "0.0.8", "0.0.9", "0.1.0", "bad"])
def test_legacy_horizon_policy_requires_historical_version(tmp_path, version):
    """The opt-in cannot authorize a current or unidentified config."""
    import yaml

    raw = yaml.safe_load(TEMPLATE.read_text())
    raw.pop("scenario_horizons")
    raw.pop("scenario_horizons_sha256")
    for field in (
        "scenario_matrix",
        "comparability_mapping",
        "route_clearance_certifications",
        "snqi_weights",
        "snqi_baseline",
    ):
        if field in raw:
            raw[field] = str(ROOT / raw[field])
    raw.update(horizon_policy="legacy_runner_cap", horizon=600)
    raw["seed_policy"] = {"mode": "fixed-list", "seeds": [1001]}
    raw["planners"] = [{"key": "goal", "algo": "goal", "planner_group": "core"}]
    if version is not None:
        raw["protocol_version"] = version
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="legacy_runner_cap.*0.0.7"):
        load_campaign_config(path, repository_root=ROOT)


def test_legacy_provenance_preserves_historical_episode_identity():
    """New accounting does not rename a published historical H600 input."""
    from robot_sf.benchmark.camera_ready._config import _apply_fixed_campaign_horizon
    from robot_sf.benchmark.map_runner.map_runner_identity import scenario_identity_payload

    authored = {
        "name": "historical",
        "seeds": [1001],
        "simulation_config": {"max_episode_steps": 400},
        "metadata": {"study": "historical"},
    }
    resolved = _apply_fixed_campaign_horizon(
        [authored],
        horizon=600,
        horizon_policy="legacy_runner_cap",
        protocol_version="0.0.7",
    )[0]
    options = {"algo": "goal", "algo_config": {}, "horizon": 600, "dt": 0.1, "record_forces": False}
    assert scenario_identity_payload(resolved, **options) == scenario_identity_payload(
        authored, **options
    )


def test_legacy_mode_is_fenced_again_at_planner_preparation(tmp_path):
    """A caller replacing the parsed config cannot bypass the current-version fence."""
    cfg = load_campaign_config(
        ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"
    )
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    changed = replace(
        cfg, horizon_policy="legacy_runner_cap", protocol_version="0.0.8", horizon=600
    )
    with pytest.raises(ValueError, match="legacy_runner_cap.*0.0.7"):
        _prepare_campaign_planner_variant_run(
            SimpleNamespace(cfg=changed, runs_dir=tmp_path, scenarios=scenarios),
            planner=cfg.planners[0],
            kinematics="differential_drive",
            active_observation_mode="socnav_state",
            log_run=False,
        )


@pytest.mark.parametrize("reserved", ["campaign_horizon", "scenario_horizon"])
def test_matrix_override_cannot_plant_reserved_horizon_metadata(tmp_path, reserved):
    """An input override cannot masquerade as trusted admission provenance."""
    import yaml

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "include": [
                    str(
                        ROOT
                        / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
                    )
                ],
                "scenario_overrides": {"metadata": {reserved: {"authored_max_episode_steps": 600}}},
            }
        )
    )
    cfg = replace(load_campaign_config(TEMPLATE), scenario_matrix_path=matrix)
    with pytest.raises(ValueError, match="reserved admission keys"):
        _load_campaign_scenarios(cfg, repository_root=ROOT)


def test_passed_horizon_cannot_override_scheduled_400(monkeypatch):
    """A passed H600 cannot extend a schedule-bound H400 scenario."""
    import robot_sf.benchmark.map_runner.map_runner_episode as episode

    cfg = load_campaign_config(TEMPLATE)
    scenario = next(
        s
        for s in _load_campaign_scenarios(cfg, repository_root=ROOT)
        if s["name"] == "francis2023_blind_corner"
    )
    original = episode._resolve_episode_run_context

    def conflicting_horizon(**kwargs):
        return original(**{**kwargs, "horizon": 600})

    monkeypatch.setattr(episode, "_resolve_episode_run_context", conflicting_horizon)
    with pytest.raises(ValueError, match="passed horizon differs from bound scenario budget"):
        _scheduled_context(scenario, 0.1)


def test_registry_admission_depends_on_exact_bytes(tmp_path):
    """Renaming preserves admission; modifying identical-name YAML does not."""
    source = (
        ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"
    )
    (tmp_path / "configs").symlink_to(ROOT / "configs", target_is_directory=True)
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    cfg = load_campaign_config(path, repository_root=ROOT)
    assert cfg.protocol_version == "0.0.7"
    assert cfg.horizon_policy == "legacy_runner_cap"
    path.write_bytes(source.read_bytes() + b"\n# changed content\n")
    cfg = load_campaign_config(path, repository_root=ROOT)
    with pytest.raises(ValueError, match="authored limit.*below fixed horizon"):
        _load_campaign_scenarios(cfg, repository_root=ROOT)


def test_published_2026_08_manifest_is_valid_in_production():
    """Published immutable pins work through production admission, without injection."""
    from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest

    manifest = load_release_manifest(
        ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
    )
    result = validate_release_manifest(manifest)
    assert result["status"] == "valid", result["problems"]
    assert result["problems"] == []


def test_retired_legacy_extension_policy_has_no_alias():
    """The extension policy cannot remain available under its former name."""
    from robot_sf.benchmark.camera_ready._config import _validate_horizon_policy

    with pytest.raises(ValueError, match="Unknown horizon_policy"):
        _validate_horizon_policy("legacy_fixed_extends_authored", "0.0.7")


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
