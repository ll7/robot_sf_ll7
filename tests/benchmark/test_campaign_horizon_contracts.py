"""Fast authored-budget and historical-identity contracts without episode execution."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.benchmark.camera_ready.campaign import _prepare_campaign_planner_variant_run
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from tests.benchmark.campaign_horizon_support import (
    AUTHORED_BUDGETS,
    ROOT,
    TEMPLATE,
    _scheduled_context,
)


@pytest.mark.parametrize("protocol", ["0.0.8", "0.0.9"])
def test_schedule_below_authored_limit_is_refused(tmp_path, protocol):
    """A pinned schedule cannot silently truncate an authored current-protocol budget."""
    from robot_sf.benchmark.camera_ready._config import _apply_scenario_horizon_schedule
    from robot_sf.evidence.writers import write_text

    scenario = {
        "name": "dev_authored_h600",
        "seeds": [1001],
        "simulation_config": {"max_episode_steps": 600},
    }
    original = deepcopy(scenario)
    schedule = tmp_path / "schedule.yaml"
    write_text(
        schedule,
        "# AI-GENERATED / NEEDS-REVIEW: synthetic dev-seed schedule\n"
        "scenarios:\n  dev_authored_h600:\n"
        "    recommended_horizon_steps: 400\n    status: authored\n",
    )
    with pytest.raises(
        ValueError, match="dev_authored_h600.*authored limit 600.*scheduled horizon 400"
    ):
        _apply_scenario_horizon_schedule(
            [scenario], schedule_path=schedule, protocol_version=protocol
        )
    assert scenario == original
    # Historical schedules retain main's original truncation and provenance behavior.
    historical = _apply_scenario_horizon_schedule(
        [scenario], schedule_path=schedule, protocol_version="0.0.7"
    )[0]
    assert historical["simulation_config"]["max_episode_steps"] == 400
    assert "authored_max_episode_steps" not in historical["metadata"]["scenario_horizon"]


def test_d084_selected_release_schedules_overtaking_at_600():
    """Resolve the real D-083 campaign: D-084's overtaking budget survives runner binding."""
    cfg = load_campaign_config(
        ROOT
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"
    )
    scenario = next(
        s
        for s in _load_campaign_scenarios(cfg, repository_root=ROOT)
        if s["name"] == "francis2023_pedestrian_overtaking"
    )
    assert scenario["simulation_config"]["max_episode_steps"] == 600
    assert scenario["metadata"]["scenario_horizon"]["recommended_horizon_steps"] == 600
    context = _scheduled_context(scenario, cfg.dt)
    assert context.horizon_val == context.config.sim_config.max_sim_steps == 600


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
    assert Counter(authored.values()) == {400: 24, 500: 13, 600: 9, 650: 1, 700: 1}
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
    assert cfg.protocol_version is None
    assert cfg.horizon_policy is None
    # Losing registered policy provenance does not opt an unidentified config
    # into the new 0.0.8 simulator binding; ordinary main compatibility remains.
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    assert (
        next(s for s in scenarios if s["name"] == "francis2023_narrow_doorway")[
            "simulation_config"
        ]["max_episode_steps"]
        == 400
    )


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


@pytest.mark.parametrize("consumer", ["state", "critic"])
def test_integer_budget_reaches_state_and_critic_without_duration_roundtrip(monkeypatch, consumer):
    """A real authored H400 at dt=.052 must not turn into ceil(400.00000000000006)."""
    import numpy as np
    from gymnasium import spaces

    from robot_sf.gym_env import robot_env
    from robot_sf.robot.robot_state import RobotState
    from robot_sf.training.scenario_loader import load_scenarios_for_validation

    matrix = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    loaded = load_scenarios_for_validation(matrix, base_dir=ROOT)
    scenario = next(s for s in loaded.scenarios if s["name"] == "francis2023_blind_corner")
    assert scenario["simulation_config"]["max_episode_steps"] == 400
    config = build_env_config(dict(scenario), scenario_path=ROOT / "scoped_scenarios.json")
    config.sim_config.time_per_step_in_secs = 0.052
    config.sim_config.sim_time_in_secs = 400 * 0.052
    config.sim_config.episode_step_limit = 400
    if consumer == "state":
        state = RobotState(None, None, None, 0.052, 400 * 0.052)
        state.episode_step_limit = config.sim_config.episode_step_limit
        assert state.max_sim_steps == 400
    else:
        # Exercise the real consumer without constructing or stepping an environment.
        env = robot_env.RobotEnv.__new__(robot_env.RobotEnv)
        env._asymmetric_critic_enabled = True
        env._critic_privileged_state_key = "critic_state"
        env.observation_space = spaces.Dict({"position": spaces.Box(-1, 1, (2,), np.float32)})
        observed = []
        original = robot_env._asymmetric_critic_state_spec

        def observe(space, **kwargs):
            observed.append(kwargs["max_sim_steps"])
            return original(space, **kwargs)

        monkeypatch.setattr(robot_env, "_asymmetric_critic_state_spec", observe)
        env._apply_asymmetric_critic_observation_space(config)
        assert observed == [400]
        assert "critic_state" in env.observation_space.spaces


def test_budget_timeout_normalization_preserves_terminal_precedence():
    """A terminated budget timeout is max_steps; collisions, success and early exits win."""
    from inspect import signature

    from robot_sf.benchmark.termination_reason import resolve_termination_reason

    flags = {
        "terminated": True,
        "truncated": False,
        "success": False,
        "collision": False,
    }
    # Call the older signature too, so fail-on-base observes its wrong label, not TypeError.
    if "timeout" in signature(resolve_termination_reason).parameters:
        flags["timeout"] = True
    assert resolve_termination_reason(**flags, reached_max_steps=True) == "max_steps"
    assert resolve_termination_reason(**flags, reached_max_steps=False) == "terminated"
    assert resolve_termination_reason(**{**flags, "collision": True}, reached_max_steps=True) == (
        "collision"
    )
    assert resolve_termination_reason(**{**flags, "success": True}, reached_max_steps=True) == (
        "success"
    )
