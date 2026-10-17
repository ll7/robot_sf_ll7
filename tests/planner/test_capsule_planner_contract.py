"""Capsule planner admission must fail before dependencies or fallback dispatch."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.map_runner.map_runner import build_map_policy
from robot_sf.benchmark.map_runner.map_runner_episode import (
    _prepare_policy_and_observation_contract,
)
from robot_sf.gym_env.base_env import attach_planner_to_map
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.adaptive_proxemic_selector import AdaptiveProxemicSelectorAdapter
from robot_sf.planner.crowdnav_height import CrowdNavHeightAdapter
from robot_sf.planner.group_avoidance import TangentSubgoalGroupAvoidanceAdapter
from robot_sf.planner.guarded_ppo import GuardedPPOAdapter
from robot_sf.planner.hybrid_global_rl import HybridGlobalRLLocalAdapter
from robot_sf.planner.hybrid_rule_local_planner import HybridRuleLocalPlannerAdapter
from robot_sf.planner.planner_selector_v2_diagnostic import PlannerSelectorV2DiagnosticAdapter
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter
from robot_sf.planner.recurrent_ppo_learned_adapter import RecurrentPPOLearnedLocalPolicyAdapter
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter
from robot_sf.planner.socnav_base import SamplingPlannerAdapter
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings


def robot_settings():
    """Create a real selected body; do not change its conservative radius."""
    return BicycleDriveSettings(radius=0.64, capsule={"length": 1.183, "width": 0.478})


@pytest.mark.parametrize(
    "algorithm",
    [
        "goal",
        "social_force",
        "orca",
        "ppo",
        "socnav_sampling",
        "sacadrl",
        "hybrid_rule_local_planner",
        "guarded_ppo",
        "predictive_mppi",
        "risk_dwa",
        "prediction_planner",
        "hrvo",
        "dwa",
        "mppi_social",
        "sicnav",
        "dr_mpc",
        "native_command",
        "unverified_new_planner",
    ],
)
def test_public_builder_refuses_capsule_before_optional_dependency_dispatch(monkeypatch, algorithm):
    """Every planner, including unknown keys, must reject instead of loading or falling back."""
    from robot_sf.benchmark.map_runner import map_runner

    def forbidden_dispatch(*args, **kwargs):
        raise AssertionError("planner dispatch reached before rejecting unsupported capsule")

    monkeypatch.setattr(
        map_runner._policy_builder_registry, "build_registered_policy", forbidden_dispatch
    )
    monkeypatch.setattr(map_runner, "build_native_command_policy", forbidden_dispatch)
    with pytest.raises(ValueError, match="does not support the selected capsule/bicycle robot"):
        build_map_policy(
            algorithm, {}, robot_kinematics="bicycle_drive", robot_config=robot_settings()
        )


def test_public_builder_cannot_ignore_nested_authored_body():
    """Existing robot_config-in-policy inputs cannot silently build a disc goal seeker."""
    with pytest.raises(ValueError, match="capsule/bicycle"):
        build_map_policy("goal", {"robot_config": {"capsule": {"length": 1.183, "width": 0.478}}})


@pytest.mark.parametrize(
    "adapter_type",
    [
        SamplingPlannerAdapter,
        ORCAPlannerAdapter,
        PredictionPlannerAdapter,
        RiskDWAPlannerAdapter,
        PredictiveMPPIAdapter,
        HybridRuleLocalPlannerAdapter,
        GuardedPPOAdapter,
        RecurrentPPOLearnedLocalPolicyAdapter,
        CrowdNavHeightAdapter,
        HybridGlobalRLLocalAdapter,
        AdaptiveProxemicSelectorAdapter,
        TangentSubgoalGroupAvoidanceAdapter,
        PlannerSelectorV2DiagnosticAdapter,
    ],
)
@pytest.mark.parametrize("binding", ["authored", "live"])
def test_direct_adapter_binding_refuses_capsule_before_state_changes(adapter_type, binding):
    """No constructor/dependency is needed to prove the real bind method rejects first."""
    settings = robot_settings()
    env = (
        SimpleNamespace(config=SimpleNamespace(robot_config=settings))
        if binding == "authored"
        else SimpleNamespace(simulator=SimpleNamespace(robots=[BicycleDriveRobot(settings)]))
    )
    # Call the actual public method, before optional checkpoint/runtime construction.
    adapter = (
        adapter_type()
        if adapter_type
        in (
            SamplingPlannerAdapter,
            ORCAPlannerAdapter,
            RiskDWAPlannerAdapter,
            HybridRuleLocalPlannerAdapter,
        )
        else object.__new__(adapter_type)
    )
    before = vars(adapter).copy()
    with pytest.raises(ValueError, match="capsule/bicycle"):
        adapter.bind_env(env)
    assert vars(adapter) == before


@pytest.mark.parametrize("already_attached", [False, True])
def test_global_routes_refuse_capsule_without_partial_map_mutation(already_attached):
    """Neither grid nor visibility routes establish SE(2) capsule/bicycle feasibility."""
    map_def = SimpleNamespace(_use_planner=already_attached)
    config = SimpleNamespace(robot_config=robot_settings(), use_planner=not already_attached)
    before = vars(map_def).copy()
    with pytest.raises(ValueError, match="capsule/bicycle"):
        attach_planner_to_map(map_def, config)
    assert vars(map_def) == before


def test_episode_pipeline_rejects_before_checkpoint_or_policy_loading(monkeypatch):
    """Check actual resolved settings even when a custom builder omits body metadata."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    def forbidden_load(*args, **kwargs):
        raise AssertionError("checkpoint resolved before capsule rejection")

    monkeypatch.setattr(
        map_runner_episode, "resolve_learned_checkpoint_observation_contract", forbidden_load
    )
    with pytest.raises(ValueError, match="capsule/bicycle"):
        _prepare_policy_and_observation_contract(
            scenario={"name": "capsule-diagnostic"},
            algo="ppo",
            policy_cfg={"model_path": str(Path("missing-checkpoint.zip"))},
            config=RobotSimulationConfig(robot_config=robot_settings()),
            observation_mode=None,
            observation_level=None,
            robot_kinematics="bicycle_drive",
            robot_command_mode="vx_vy",
            adapter_impact_eval=False,
            benchmark_track=None,
            track_schema_version=None,
            actuation_profile=None,
            policy_builder=forbidden_load,
        )


def test_explicit_disc_proxy_still_builds_and_binds_real_planner():
    """Refusal must not accidentally disable existing covering-disc diagnostics."""
    from math import tan

    from robot_sf.benchmark.planner_command_contract import validate_planner_contract

    settings = BicycleDriveSettings(radius=0.64)
    bicycle_config = {
        "bicycle_max_curvature": tan(0.78),
        "bicycle_max_velocity": 3.0,
        "bicycle_max_angular_speed": 3.0 * tan(0.78),
    }
    policy, meta = build_map_policy(
        "goal", bicycle_config, robot_kinematics="bicycle_drive", robot_config=settings
    )
    assert callable(policy)
    assert meta["algorithm"] == "goal"
    validate_planner_contract(
        algo="goal", robot_kinematics="bicycle_drive", algo_config={}, robot_config=settings
    )
    adapter = SamplingPlannerAdapter()
    adapter.bind_env(SimpleNamespace(config=SimpleNamespace(robot_config=settings)))


def test_public_contract_rejects_capsule_before_metadata_lookup():
    """Unknown algorithms cannot obscure the missing physical support with another error."""
    from robot_sf.benchmark.planner_command_contract import validate_planner_contract

    with pytest.raises(ValueError, match="capsule/bicycle"):
        validate_planner_contract(
            algo="unverified_new_planner",
            robot_kinematics="bicycle_drive",
            algo_config={},
            robot_config=robot_settings(),
        )


def test_embedded_capsule_cannot_be_overridden_with_disc_argument():
    """Contradictory caller metadata cannot erase an explicitly selected body."""
    with pytest.raises(ValueError, match="capsule/bicycle"):
        build_map_policy(
            "goal",
            {"robot_config": {"capsule": {"length": 1.183, "width": 0.478}}},
            robot_config=BicycleDriveSettings(radius=0.64),
        )


def test_batch_preflight_rejects_capsule_before_checkpoint_resolution(monkeypatch):
    """Batch dispatch must refuse an unsupported slot rather than silently skipping it."""
    from robot_sf.benchmark.map_runner import map_runner

    def forbidden_load(*args, **kwargs):
        raise AssertionError("batch checkpoint resolved before capsule rejection")

    monkeypatch.setattr(
        map_runner, "resolve_learned_checkpoint_observation_contract", forbidden_load
    )
    ctx = SimpleNamespace(
        filtered=[
            {
                "name": "capsule-diagnostic",
                "robot_config": {
                    "type": "bicycle_drive",
                    "capsule": {"length": 1.183, "width": 0.478},
                },
            }
        ],
        algo="goal",
        algo_config_path=None,
        raw_policy_cfg={},
        batch_observation_mode=None,
        observation_level=None,
    )
    with pytest.raises(ValueError, match="capsule/bicycle"):
        map_runner._validate_per_scenario_compatibility(ctx)


@pytest.mark.parametrize("binding", ["authored", "live"])
def test_direct_ppo_binding_refuses_capsule_before_observation_state_changes(monkeypatch, binding):
    """The standalone PPO entry point must reject before accepting an observation space."""
    from gymnasium.spaces import Box

    from robot_sf.baselines.ppo import PPOPlanner

    settings = SimpleNamespace(capsule={"length": 1.183, "width": 0.478})
    env = SimpleNamespace(observation_space=Box(-1.0, 1.0, (2,)))
    if binding == "authored":
        env.config = SimpleNamespace(robot_config=settings)
    else:
        env.simulator = SimpleNamespace(robots=[SimpleNamespace(config=settings)])
    planner = object.__new__(PPOPlanner)
    planner._runtime_observation_space = None
    monkeypatch.setattr(planner, "_validate_runtime_observation_space", lambda: None)
    before = vars(planner).copy()
    with pytest.raises(ValueError, match="capsule/bicycle"):
        planner.bind_env(env)
    assert vars(planner) == before
