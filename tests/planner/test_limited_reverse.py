"""Limited reverse regressions; see the PR's per-test test-value answers."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HYBRID_RULE_V4_CLEARANCE_BRAKING_VARIANT,
    HybridRuleCandidate,
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
)
from robot_sf.planner.kinematics_model import resolve_benchmark_kinematics_model
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter, SocNavPlannerConfig
from robot_sf.robot.actuation_envelope import actuation_envelope_from_drive_config
from robot_sf.robot.bicycle_drive import BicycleDriveSettings, BicycleDriveState, BicycleMotion
from robot_sf.robot.differential_drive import (
    DifferentialDriveMotion,
    DifferentialDriveSettings,
    DifferentialDriveState,
)
from tests.planner.test_socnav_orca_module import _observation, _with_occupancy_grid


def _drive(cls=DifferentialDriveSettings, cap=0.5):
    # Set public attributes so the base run reaches the behavioral assertion,
    # rather than failing solely because its constructor lacks new keywords.
    drive = cls()
    drive.limited_reverse = True
    drive.max_reverse_speed = cap
    return drive


def _env(drive):
    return SimpleNamespace(
        env_config=SimpleNamespace(robot_config=drive),
        simulator=SimpleNamespace(robots=[SimpleNamespace(config=drive)]),
    )


@pytest.mark.parametrize("cap", [0.3, 0.5])
@pytest.mark.parametrize(
    "cls,motion_cls,state_cls",
    [
        (DifferentialDriveSettings, DifferentialDriveMotion, DifferentialDriveState),
        (BicycleDriveSettings, BicycleMotion, BicycleDriveState),
    ],
)
def test_drive_reverse_cap_and_acceleration(cls, motion_cls, state_cls, cap):
    drive = _drive(cls, cap)
    motion = motion_cls(drive)
    state = state_cls(pose=((0.0, 0.0), 0.0))
    speeds = []
    for _ in range(10):
        motion.move(state, (-100.0, 0.0), 0.1)
        speeds.append(state.velocity[0] if isinstance(state.velocity, tuple) else state.velocity)
    assert speeds[0] == pytest.approx(-0.1)
    assert speeds[-1] == pytest.approx(-cap)
    assert state.pose[0][0] < 0.0
    motion.move(state, (100.0, 0.0), 0.1)
    speed = state.velocity[0] if isinstance(state.velocity, tuple) else state.velocity
    assert speed == pytest.approx(-cap + 0.1)


@pytest.mark.parametrize("kinematics", ["differential_drive", "bicycle_drive"])
def test_unicycle_command_projection_accepts_and_caps_reverse(kinematics):
    model = resolve_benchmark_kinematics_model(
        robot_kinematics=kinematics,
        command_limits={"limited_reverse": True, "max_reverse_speed": 0.3},
    )
    assert model.project((-2.0, 0.0)) == (-0.3, 0.0)
    assert model.is_feasible((-0.3, 0.0))
    assert not model.is_feasible((-0.31, 0.0))


def test_orca_reverse_projection_and_rear_pedestrian_clearance():
    adapter = ORCAPlannerAdapter(
        SocNavPlannerConfig(occupancy_heading_sweep=0.0, occupancy_lookahead=1.0)
    )
    adapter.bind_env(_env(_drive()))
    obs = _with_occupancy_grid(_observation())
    kwargs = {
        "velocity_world": np.array([-2.0, 0.0]),
        "robot_pos": np.zeros(2),
        "robot_heading": 0.0,
    }
    assert adapter._velocity_world_to_command(**kwargs, observation=obs) == (-0.5, 0.0)
    blocked = _with_occupancy_grid(_observation(pedestrians=[[-0.95, 0.0]]))
    assert adapter._velocity_world_to_command(**kwargs, observation=blocked)[0] == 0.0
    adapter.bind_env(_env(DifferentialDriveSettings()))
    assert adapter._velocity_world_to_command(**kwargs, observation=obs)[0] == 0.0


def test_hybrid_reverse_rollout_and_rear_braking_contact():
    planner = HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(
            planner_variant=HYBRID_RULE_V4_CLEARANCE_BRAKING_VARIANT,
            static_clearance_escape_enabled=True,
        )
    )
    planner.bind_env(_env(_drive(cap=0.3)))
    candidate = HybridRuleCandidate(-0.5, 0.0, "reverse_escape", ((0.4, -0.5, 0.0),))
    clipped = planner._clip_candidate(candidate, speed_cap=2.0)
    assert clipped.linear == -0.3
    assert clipped.rollout_sequence[0][1] == -0.3
    realized = planner._v4_realized_rollout_commands([(-0.3, 0.0)] * 4, current_speed=0.0, dt=0.1)
    assert np.array(realized)[:, 0] == pytest.approx([-0.1, -0.2, -0.3, -0.3])
    state = {
        "robot_pos": np.zeros(2),
        "heading": 0.0,
        "current_speed": -0.3,
        "ped_pos": np.array([[-1.43, 0.0]]),
        "ped_vel": np.zeros((1, 2)),
        "dt": 0.1,
    }
    assert planner._v4_braking_rejection(candidate=clipped, state=state, collision_radius=1.4)


@pytest.mark.parametrize("cls", [DifferentialDriveSettings, BicycleDriveSettings])
def test_reverse_plant_provenance_is_opt_in(cls):
    legacy = cls()
    drive = _drive(cls)
    assert asdict(drive) == asdict(legacy)
    old = actuation_envelope_from_drive_config(legacy)
    assert "reverse_drive" not in old
    new = actuation_envelope_from_drive_config(drive)
    assert new["reverse_drive"] == {
        "schema_version": "limited_reverse.v1",
        "max_reverse_speed_m_s": 0.5,
    }


def test_unified_scenario_config_and_environment_identity():
    from robot_sf.gym_env.robot_env import _stable_config_hash
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.training.scenario_loader import _apply_robot_overrides

    legacy = RobotSimulationConfig()
    enabled = RobotSimulationConfig()
    _apply_robot_overrides(enabled, {"limited_reverse": True})
    assert enabled.robot_config.min_linear_speed == -0.5
    assert _stable_config_hash(enabled) != _stable_config_hash(legacy)
    capped = RobotSimulationConfig()
    _apply_robot_overrides(
        capped, {"type": "bicycle_drive", "limited_reverse": True, "max_reverse_speed": 0.3}
    )
    assert capped.robot_config.min_velocity == -0.3
    assert asdict(enabled.robot_config) == asdict(legacy.robot_config)


@pytest.mark.parametrize("kind", ["differential", "bicycle"])
def test_velocity_action_adapter_keeps_reverse_through_acceleration(kind):
    from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
    from robot_sf.robot.bicycle_drive import BicycleDriveRobot
    from robot_sf.robot.differential_drive import DifferentialDriveRobot

    robot = (
        DifferentialDriveRobot(_drive(cap=0.3))
        if kind == "differential"
        else BicycleDriveRobot(_drive(BicycleDriveSettings, cap=0.3))
    )
    adapter = PlannerActionAdapter(robot, robot.action_space, 0.1)
    assert robot.observation_space.low[0] == pytest.approx(-0.3)
    for expected in [-0.1, -0.2, -0.3, -0.3]:
        action = adapter.from_velocity_command((-2.0, 0.0))
        robot.apply_action(action, 0.1)
        assert robot.current_speed[0] == pytest.approx(expected)


def test_policy_binding_preserves_live_reverse_commands():
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_common import build_adapter_policy

    adapter = ORCAPlannerAdapter(
        SocNavPlannerConfig(occupancy_heading_sweep=0.0, occupancy_lookahead=1.0)
    )
    policy, _ = build_adapter_policy(
        algo_key="orca",
        algo_config={},
        meta={},
        adapter=adapter,
        adapter_name="ORCAPlannerAdapter",
        robot_kinematics="differential_drive",
        normalized_robot_command_mode=None,
    )
    policy._planner_bind_env(_env(_drive(cap=0.3)))
    assert policy._kinematics_model.project((-1.0, 0.0)) == (-0.3, 0.0)
    policy._planner_bind_env(_env(DifferentialDriveSettings()))
    assert policy._kinematics_model.project((-1.0, 0.0)) == (0.0, 0.0)


@pytest.mark.parametrize("value", [0.0, -0.5, float("nan"), float("inf"), True])
def test_scenario_rejects_invalid_reverse_cap(value):
    from robot_sf.training.scenario_loader import _differential_robot_settings

    with pytest.raises(ValueError, match="max_reverse_speed"):
        _differential_robot_settings({"limited_reverse": True, "max_reverse_speed": value})


def test_socnav_runner_binding_keeps_reverse_in_the_executed_policy():
    from robot_sf.benchmark.map_runner.map_runner import _build_policy

    policy, _ = _build_policy("orca", {"occupancy_heading_sweep": 0.0, "occupancy_lookahead": 1.0})
    policy._planner_bind_env(_env(_drive(cap=0.3)))
    obs = _with_occupancy_grid(_observation(goal=(-5.0, 0.0)))
    assert policy(obs)[0] == pytest.approx(-0.3)
    policy._planner_bind_env(_env(DifferentialDriveSettings()))
    assert policy(obs)[0] == 0.0


def test_hybrid_static_escape_reverses_but_rejects_a_rear_wall():
    planner = HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(
            planner_variant=HYBRID_RULE_V4_CLEARANCE_BRAKING_VARIANT,
            static_clearance_escape_enabled=True,
            continuous_static_clearance_enabled=True,
        )
    )
    env = _env(_drive(cap=0.3))
    env.simulator.map_def = SimpleNamespace(width=20.0, height=20.0)
    env.simulator.get_obstacle_lines = lambda: np.array([[8.65, 0.0, 8.65, 20.0]])
    planner.bind_env(env)
    candidate = HybridRuleCandidate(-0.3, 0.0, "reverse_escape")
    assert planner._static_clearance_escape_allowed(
        candidate=candidate,
        initial_clearance=1.0,
        current_min_clearance=1.0,
        hard_static_clearance=1.2,
    )
    obs = _with_occupancy_grid(_observation())
    state = planner._extract_state(obs)
    state.update(
        robot_pos=np.array([10.0, 10.0]),
        current_speed=-0.3,
        goal=np.array([15.0, 10.0]),
        robot_radius=1.0,
    )
    evaluation = planner._evaluate_candidate(
        candidate=candidate, observation=obs, state=state, speed_cap=2.0, nearest_ped=float("inf")
    )
    assert not evaluation["accepted"]
    assert evaluation["reason"] == "static_collision"
