"""Real planner and builder regressions from the hybrid default review."""

import gzip
import hashlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner import _build_policy
from robot_sf.benchmark.map_runner.map_runner_batch_plan import build_worker_fixed_params
from robot_sf.benchmark.map_runner.map_runner_env import (
    apply_policy_env_observation_overrides,
    build_env_config,
)
from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
from robot_sf.benchmark.map_runner.map_runner_worker import execute_map_job
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    _resolve_policy_search_candidate_runtime,
)
from robot_sf.benchmark.release_parameter_freeze import UnfrozenReleaseParametersError
from robot_sf.common.hybrid_defaults import defaults_for_source
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_policy_search_step_diagnostics import _json_ready
from tests.planner.test_hybrid_rule_local_planner import _obs

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"


def _validity_planner(enabled=True):
    return HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(
            {
                "planner_variant": "hybrid_rule_v4_clearance_braking",
                "physical_static_exclusion_enabled": False,
                "goal_next_validity_enabled": enabled,
                "route_guide_enabled": True,
            }
        )
    )


@pytest.mark.parametrize("completion", ("unbound", "radius", "zone"))
def test_terminal_goal_at_022_m_keeps_tracking_before_environment_success(completion):
    """An invalid successor cannot park the robot outside the actual terminal criterion."""
    planner = _validity_planner()
    observation = _obs(robot=(3.78, 4.0), goal=(4.0, 4.0))
    observation["goal"]["next"] = np.array([0.0, 0.0])
    observation["goal"]["next_valid"] = np.array([0.0])
    if completion != "unbound":
        extra = (
            {
                "completion_policy": "goal_zone_entry_v1",
                "goal_zone": ((3.8, 3.8), (4.2, 3.8), (4.2, 4.2)),
            }
            if completion == "zone"
            else {}
        )
        navigator = RouteNavigator(
            waypoints=[(3.0, 4.0), (4.0, 4.0)],
            waypoint_id=1,
            proximity_threshold=0.2,
            pos=(3.78, 4.0),
            **extra,
        )
        assert navigator.reached_destination is False
        planner.bind_env(SimpleNamespace(simulator=SimpleNamespace(robot_navs=[navigator])))
    linear, _ = planner.plan(observation)
    assert planner.last_decision()["planner_mode"] != "GOAL_STOP"
    assert linear > 0.0
    assert planner._route_guide.config.goal_tolerance == planner.config.goal_tolerance
    if completion == "unbound":
        observation["robot"]["position"] = np.array([4.0, 4.0])
    else:
        navigator.pos = (3.81, 4.0)
        assert navigator.reached_destination is True
        observation["robot"]["position"] = np.array([3.81, 4.0])
    assert planner.plan(observation) == (0.0, 0.0)
    assert planner.last_decision()["planner_mode"] == "GOAL_STOP"
    legacy = _validity_planner(enabled=False)
    old_observation = _obs(robot=(3.78, 4.0), goal=(4.0, 4.0))
    old_observation["goal"]["next"] = np.array([4.0, 4.0])
    assert legacy.plan(old_observation) == (0.0, 0.0)
    assert legacy.last_decision()["planner_mode"] == "GOAL_STOP"


@pytest.mark.parametrize("flattened", (False, True))
def test_enabled_validity_rejects_a_missing_sensor_field(flattened):
    """Both supported observation forms fail clearly instead of inventing validity."""
    observation = (
        {"robot_position": [0.0, 0.0], "goal_current": [4.0, 0.0], "goal_next": [5.0, 0.0]}
        if flattened
        else _obs(goal=(4.0, 0.0))
    )
    observation.get("goal", {}).pop("next_valid", None)
    with pytest.raises(ValueError, match="goal_next_validity_enabled requires.*next_valid"):
        _validity_planner().plan(observation)


def test_new_env_and_planner_share_defaults_on_registered_release_scenario():
    """A scenario asset alone cannot silently select legacy sensor defaults for new inputs."""
    scenario = load_scenarios(MATRIX)[0]
    env = build_env_config(scenario, scenario_path=MATRIX)
    planner = build_hybrid_rule_local_planner_config({})
    assert env.include_goal_next_valid is planner.goal_next_validity_enabled is True


def test_released_002_ppo_constructor_uses_legacy_sensor_defaults():
    """The recorded 0.0.2 PPO source keeps its missing sensor field off."""
    with defaults_for_source(ROOT / "configs/baselines/ppo_15m_grid_socnav.yaml"):
        assert RobotSimulationConfig().include_goal_next_valid is False


def test_worker_preserves_absent_algorithm_config_for_release_default_selection():
    """Batch serialization distinguishes an absent algorithm input from explicit inline input."""
    params = build_worker_fixed_params(
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        raw_policy_cfg={},
        algo_config_path=None,
        scenario_path=MATRIX,
        adapter_impact_eval=False,
        experimental_ped_impact=False,
        ped_impact_radius_m=1.0,
        ped_impact_window_steps=5,
        noise_spec={"enabled": False},
        tracking_precision_spec={"enabled": False},
        batch_observation_mode=None,
        observation_level=None,
        benchmark_track=None,
        track_schema_version=None,
        actuation_profile_metadata=None,
        latency_profile_metadata=None,
        latency_stress_metrics=None,
        safety_wrapper=None,
        record_planner_decision_trace=False,
        record_simulation_step_trace=False,
    )
    assert params["algo_config"] is None
    record = execute_map_job(
        (load_scenarios(MATRIX)[0], 1001, params),
        run_map_episode=partial(run_map_episode, policy_builder=_build_policy),
    )
    assert record["algorithm_metadata"]["hybrid_default_policy"]["default_set"] == "legacy-0.0.8"


def test_every_release_arm_keeps_full_base_environment_and_mapping_dumps():
    """Every catalogued arm retains full typed values, including absent algorithm inputs."""
    inventory = json.loads(
        (ROOT / "docs/validation/hybrid_defaults/released_arm_inventory.json").read_text()
    )
    baseline = json.loads(
        gzip.decompress(
            (ROOT / "tests/fixtures/hybrid_defaults/base_released_arms.json.gz").read_bytes()
        )
    )
    observed = set()
    for arm in inventory["arms"]:
        expected = baseline["arms"][arm["id"]]
        matrix = ROOT / arm["scenario_matrix"]
        scenario = load_scenarios(matrix)[0]
        assert scenario["name"] == expected["scenario"]
        source = arm["algo_config_path"] or arm["scenario_matrix"]
        with defaults_for_source(ROOT / source):
            constructor = json.dumps(_json_ready(asdict(RobotSimulationConfig())), sort_keys=True)
            assert constructor.replace(str(ROOT), "<repo>") == json.dumps(
                baseline["constructor"], sort_keys=True
            )
            env = build_env_config(scenario, scenario_path=matrix)
            if expected["runtime_status"] == "blocked-unfrozen-source":
                with pytest.raises(UnfrozenReleaseParametersError):
                    _resolve_policy_search_candidate_runtime(
                        default_algo=arm["algo"],
                        algo_config_path=arm["algo_config_path"],
                        scenario=scenario,
                        config_root=ROOT,
                    )
            else:
                _, raw = _resolve_policy_search_candidate_runtime(
                    default_algo=arm["algo"],
                    algo_config_path=arm["algo_config_path"],
                    scenario=scenario,
                    config_root=ROOT,
                )
                canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
                assert (
                    hashlib.sha256(canonical.encode()).hexdigest()
                    == expected["effective_mapping_sha256"]
                )
                apply_policy_env_observation_overrides(env, raw)
            dump = json.dumps(_json_ready(asdict(env)), sort_keys=True).replace(str(ROOT), "<repo>")
            assert dump == json.dumps(expected["environment"], sort_keys=True), arm["id"]
        observed.add(arm["id"])
    assert observed == set(baseline["arms"])


def test_unknown_configless_hybrid_on_released_assets_uses_current_defaults():
    """New algorithm inputs cannot inherit another released arm's missing-field policy."""
    scenario = load_scenarios(MATRIX)[0]
    record = run_map_episode(
        scenario,
        1001,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="hybrid_rule_local_planner",
        scenario_path=MATRIX,
        policy_builder=_build_policy,
    )
    assert record["algorithm_metadata"]["hybrid_default_policy"] == {"default_set": "current"}
    params = build_worker_fixed_params(
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="hybrid_rule_local_planner",
        raw_policy_cfg={},
        algo_config_path=None,
        scenario_path=MATRIX,
        adapter_impact_eval=False,
        experimental_ped_impact=False,
        ped_impact_radius_m=1.0,
        ped_impact_window_steps=5,
        noise_spec={"enabled": False},
        tracking_precision_spec={"enabled": False},
        batch_observation_mode=None,
        observation_level=None,
        benchmark_track=None,
        track_schema_version=None,
        actuation_profile_metadata=None,
        latency_profile_metadata=None,
        latency_stress_metrics=None,
        safety_wrapper=None,
        record_planner_decision_trace=False,
        record_simulation_step_trace=False,
    )
    dispatched = execute_map_job(
        (scenario, 1001, params),
        run_map_episode=partial(run_map_episode, policy_builder=_build_policy),
    )
    assert params["algo_config"] is None
    assert dispatched["algorithm_metadata"]["hybrid_default_policy"] == {"default_set": "current"}
