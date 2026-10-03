"""Safety and intent regressions through the actual bicycle command path."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_episode import (
    _step_convert_and_execute,
    _step_safety_filters,
)
from robot_sf.benchmark.safety.safety_wrapper_runtime import (
    SafetyWrapperRuntimeConfig,
    make_deadlock_recovery_monitor,
)
from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.planner.kinematics_model import BicycleDriveKinematicsModel
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings


def model(**kwargs):
    """A physical cone with independently specified curvature 0.2 /m."""
    return BicycleDriveKinematicsModel(
        max_velocity=2.0, max_angular_speed=0.4, max_curvature=0.2, **kwargs
    )


def test_creep_is_disabled_by_default():
    """A yaw-only request cannot silently request forward motion."""
    assert model().project((0.0, 0.5)) == (0.0, 0.0)


@pytest.mark.parametrize("omega", [1.1e-6, -1.1e-6, 0.016, -0.016])
def test_opt_in_creep_rejects_noise(omega):
    """Even explicitly enabled creep must ignore sub-degree-per-second yaw."""
    assert model(creep_speed=0.1).project((0.0, omega)) == (0.0, 0.0)


def test_small_reverse_request_never_becomes_forward():
    """Speed priority preserves a negative request, including near standstill."""
    assert model(creep_speed=0.1, allow_backwards=True).project((-0.0005, 0.2)) == (
        -0.0005,
        0.0001,
    )


@pytest.mark.parametrize("creep_speed", [0.0, 0.0001])
def test_creep_never_reduces_a_positive_request(creep_speed):
    """Disabling creep or selecting a lower creep speed preserves bounded requested speed."""
    assert model(creep_speed=creep_speed).project((0.0005, 0.2)) == (0.0005, 0.0001)


def test_bicycle_requires_physical_curvature():
    """Separate scalar caps cannot substitute for wheelbase and steering physics."""
    with pytest.raises(ValueError, match="max_curvature"):
        BicycleDriveKinematicsModel(max_velocity=2.0, max_angular_speed=0.4)


def test_explicit_zero_curvature_is_a_valid_straight_plant():
    """Zero steering is physical, whereas an unspecified curvature is ambiguous."""
    straight = BicycleDriveKinematicsModel(
        max_velocity=2.0, max_angular_speed=0.4, max_curvature=0.0
    )
    assert straight.project((1.0, 0.2)) == (1.0, 0.0)


@pytest.mark.parametrize("recovery,hard_stop", [(False, True), (True, True), (True, False)])
def test_hard_stop_remains_stopped_through_bicycle_conversion(recovery, hard_stop):
    """The real safety stage and final conversion preserve a veto with creep opted in."""
    robot = BicycleDriveRobot(BicycleDriveSettings(radius=0.64, wheelbase=0.90, max_steer=0.52))
    # Also works on the pre-fix settings, which have no constructor field for creep.
    robot.config.creep_speed = 0.1
    config = SimpleNamespace(
        robot_config=robot.config,
        sim_config=SimpleNamespace(time_per_step_in_secs=0.1, ped_radius=0.4),
    )
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            robots=[robot],
            robot_pos=[np.array([0.0, 0.0])],
            ped_pos=np.array([[1.24 if hard_stop else 1.44, 0.0]]),
        ),
        action_space=robot.action_space,
    )

    def step(action):
        robot.apply_action(tuple(action), 0.1)
        return {}, 0.0, False, False, {}

    env.step = step
    runtime = SafetyWrapperRuntimeConfig(
        enabled=True, arm_key="wrapper_on", deadlock_recovery_enabled=recovery
    )
    slc = SimpleNamespace(
        config=config,
        safety_wrapper_runtime=runtime,
        safety_wrapper_deadlock_monitor=make_deadlock_recovery_monitor(runtime),
        cbf_runtime=SimpleNamespace(enabled=False),
        active_harness=None,
        record_simulation_step_trace=False,
    )
    state = SimpleNamespace(
        previous_trace_ped_pos=None,
        safety_wrapper_trace=[],
        ammv_command_actions=[],
    )
    for index in range(40):
        command = _step_safety_filters(
            state,
            slc,
            policy_command=(1.0, 0.5) if hard_stop else (0.0, 0.0),
            step_is_native=False,
            env=env,
            step_idx=index,
        )
        assert state.safety_wrapper_trace[-1]["intervention"] == (
            "hard_stop" if hard_stop else "none"
        )
        _step_convert_and_execute(state, slc, policy_command=command, step_is_native=False, env=env)
        assert robot.state.velocity == 0.0, "creep overrode a hard-stop/yield veto"
    assert robot.pos == (0.0, 0.0)
    if recovery:
        assert any(r["deadlock_recovery"]["recovery_active"] for r in state.safety_wrapper_trace)


@pytest.mark.parametrize("creep_speed", [0.05, 0.1])
def test_one_step_opt_in_turn_does_not_persist(creep_speed):
    """A meaningful isolated turn moves at most one centimetre then stops immediately."""
    robot = BicycleDriveRobot(BicycleDriveSettings(wheelbase=1.0, max_steer=math.pi / 4))
    adapter = PlannerActionAdapter(robot, robot.action_space, 0.1, model(creep_speed=creep_speed))
    robot.apply_action(tuple(adapter.from_velocity_command((0.0, 0.5))), 0.1)
    assert robot.state.velocity == pytest.approx(creep_speed)
    for _ in range(40):
        robot.apply_action(tuple(adapter.from_velocity_command((0.0, 0.0))), 0.1)
    assert robot.state.velocity == 0.0
    # The float32 steering action makes the norm exceed .01 by ~2e-18m.
    assert math.hypot(*robot.pos) <= 0.01 + 1e-9


def test_robot_config_controls_creep_speed():
    """The opt-in speed comes from the plant, rather than an adapter constant."""
    robot = BicycleDriveRobot(BicycleDriveSettings(wheelbase=1.0, max_steer=math.pi / 4))
    robot.config.creep_speed = 0.07
    adapter = PlannerActionAdapter(robot, robot.action_space, 0.1)
    robot.apply_action(tuple(adapter.from_velocity_command((0.0, 0.5))), 0.1)
    assert robot.state.velocity == pytest.approx(0.07)


@pytest.mark.parametrize("creep_speed", [-0.1, math.nan])
def test_mutable_robot_settings_reject_invalid_creep(creep_speed):
    """Revalidating mutable settings must reject negative or nonfinite creep."""
    settings = BicycleDriveSettings()
    settings.creep_speed = creep_speed
    with pytest.raises(ValueError, match="creep_speed"):
        settings.__post_init__(settings.limited_reverse, settings.max_reverse_speed)


@pytest.mark.parametrize("creep_speed", [-0.1, math.nan])
def test_model_rejects_invalid_creep(creep_speed):
    """Invalid configured creep cannot silently corrupt physical projection."""
    with pytest.raises(ValueError, match="creep_speed"):
        model(creep_speed=creep_speed)


def test_meaningful_opt_in_turn_is_projected():
    """One degree/second is an intentional turn, with forward speed bounded at .1."""
    assert model(creep_speed=0.1).project((0.0, math.radians(1))) == pytest.approx(
        (0.1, math.radians(1))
    )


def test_guarded_ppo_uncertainty_stop_survives_policy_projection_with_creep():
    """Real episode binding, uncertainty shield and conversion keep a slow-down veto stopped."""
    from pathlib import Path

    import yaml

    from robot_sf.benchmark.map_runner.map_runner_episode import _resolve_episode_run_context
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_common import (
        _project_with_feasibility,
    )
    from robot_sf.planner.kinematics_model import resolve_benchmark_kinematics_model
    from robot_sf.training.scenario_loader import load_scenarios
    from tests.planner.test_guarded_ppo import _obs
    from tests.planner.test_guarded_ppo_uncertainty_fallback import _uncertainty_guard

    scenario_path = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
    scenario = next(
        s for s in load_scenarios(scenario_path) if s["name"] == "classic_t_intersection_medium"
    )
    scenario = dict(scenario, seeds=[1001])
    scenario.pop("seed_set", None)
    scenario["robot_config"] = yaml.safe_load(
        Path("configs/robots/t60_bicycle_45deg_v1.yaml").read_text()
    )["robot_config"]
    scenario["robot_config"]["creep_speed"] = 0.1
    context = _resolve_episode_run_context(
        scenario=scenario,
        seed=1001,
        horizon=600,
        dt=0.1,
        algo="guarded_ppo",
        scenario_path=scenario_path,
        algo_config={},
        algo_config_path=None,
        experimental_ped_impact=False,
        ped_impact_radius_m=1.0,
        ped_impact_window_steps=1,
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
    guard = _uncertainty_guard(mode="slow_down", extra={"uncertainty_slow_down_speed_m_s": 0.0})
    decision = guard.choose_command_decision(
        _obs(ped_positions=[(1.0, 0.0)], ped_velocities=[(0.0, 0.0)]), (0.6, 0.3)
    )
    assert decision.decision_label == "uncertainty_fallback_slow_down"
    assert decision.filtered_action == (0.0, 0.3)
    policy_model = resolve_benchmark_kinematics_model(
        robot_kinematics="bicycle_drive", command_limits=context.policy_cfg
    )
    command = _project_with_feasibility(
        model=policy_model, command=decision.filtered_action, meta={}
    )
    robot = BicycleDriveRobot(context.config.robot_config)
    adapter = PlannerActionAdapter(robot, robot.action_space, 0.1)
    for _ in range(10):
        robot.apply_action(tuple(adapter.from_velocity_command(command)), 0.1)
    assert robot.state.velocity == 0.0, "in-policy veto must not regain forward creep"
    assert robot.pos == pytest.approx((0.0, 0.0))
