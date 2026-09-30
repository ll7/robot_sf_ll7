"""RV6 counterexamples on parsed release maps and native command execution.

Fixed wall/corner literals supply the geometric oracle. No episode is run.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner_actions import policy_command_to_env_action
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, build_guarded_ppo_config
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings
from tests.planner.test_fx3_static_recovery import ROOT, observation

CORNER_MAP = (
    ROOT / "maps/successor_svg_maps/issue_9762_classic_group_crossing_goal_zone_entry_v2.svg"
)
CORRIDOR_MAP = ROOT / "maps/svg_maps/atomic_corridor_test.svg"
OVERTAKING_MAP = (
    ROOT / "maps/successor_svg_maps/issue_9762_classic_overtaking_goal_zone_entry_v2.svg"
)


def release(arm):
    """Load actual release settings and the MPPI checkpoint without fallback."""
    config = yaml.safe_load((ROOT / f"configs/algos/{arm}_release_v0_0_8.yaml").read_text())
    if arm == "risk_dwa":
        return RiskDWAPlannerAdapter(build_risk_dwa_config(config))
    if arm == "guarded_ppo":
        return GuardedPPOAdapter(
            build_guarded_ppo_config(config),
            fallback_adapter=RiskDWAPlannerAdapter(
                build_risk_dwa_config(config["fallback_risk_dwa"])
            ),
        )
    adapter = PredictiveMPPIAdapter(build_predictive_mppi_config(config), allow_fallback=False)
    assert adapter._predictor._ensure_model() is not None
    return adapter


def mppi_costs(adapter, obs, commands, horizon=None):
    """Score identical sequences via the scalar and batch production paths."""
    future, mask, steps = adapter._predict_future(obs)
    sequences = np.asarray([adapter._constant_sequence(c, horizon or steps) for c in commands])
    kwargs = {
        "robot_pos": np.asarray(obs["robot"]["position"]),
        "heading": obs["robot"]["heading"][0],
        "goal": np.asarray(obs["goal"]["current"]),
        "future": future,
        "mask": mask,
        "observation": obs,
        "anchor_action": (0.0, 0.0),
    }
    scalar = np.asarray([adapter._sequence_rollout(s, **kwargs) for s in sequences])
    batch = adapter._batch_sequence_rollout(sequences, **kwargs)
    np.testing.assert_allclose(scalar, batch, rtol=1e-10, atol=1e-8)
    return scalar


@pytest.mark.parametrize("arm", ["risk_dwa", "predictive_mppi"])
def test_unbound_grid_rejects_between_sample_corner_penetration(arm):
    """Clear raster endpoints conceal a 0.5 mm penetrating corner chord."""
    adapter = release(arm)
    normal = np.array([-1.0, -1.0]) / np.sqrt(2)
    tangent = np.array([1.0, -1.0]) / np.sqrt(2)
    midpoint = np.array([23.0, 23.0]) + 0.9995 * normal
    start, end = midpoint - 0.059 * tangent, midpoint + 0.061 * tangent
    obs = observation(adapter, CORNER_MAP, start, -np.pi / 4, goal=start + 4 * tangent)
    map_def = convert_map(str(CORNER_MAP))
    lines, polygons = RobotEnv._normalize_obstacles_for_grid(map_def.obstacles, map_def.bounds)
    grid = OccupancyGrid(
        GridConfig(
            resolution=0.2,
            width=32,
            height=32,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
            use_ego_frame=False,
            center_on_robot=False,
        )
    )
    grid.generate(
        lines, [], (tuple(start), -np.pi / 4), ego_frame=False, obstacle_polygons=polygons
    )
    obs["occupancy_grid"] = grid.to_observation()
    obs["occupancy_grid_meta"] = grid.metadata_observation()
    obs["robot"]["speed"] = [1.2]
    # Analytic circular-body distance to the real SVG corner at the chord's
    # nearest point; these clear endpoints cannot prove continuous clearance.
    assert np.linalg.norm(midpoint - [23, 23]) - 1 == pytest.approx(-0.0005)
    assert np.linalg.norm(start - [23, 23]) > 1
    assert np.linalg.norm(end - [23, 23]) > 1
    assert adapter._exact_obstacle_clearance(end, previous=start) == pytest.approx(-0.0005)
    adapter._static_clearance = None  # Supported standalone grid-only caller.
    c0 = adapter._min_obstacle_clearance(start, observation=obs)
    c1 = adapter._min_obstacle_clearance(end, observation=obs)
    assert 0 < c0 < c1 < 0.3
    if arm == "risk_dwa":
        score = adapter._rollout_score(
            robot_pos=start,
            heading=-np.pi / 4,
            goal=np.asarray(obs["goal"]["current"]),
            command=(1.2, 0.0),
            ped_pos=np.empty((0, 2)),
            ped_vel=np.empty((0, 2)),
            observation=obs,
            current_speed=1.2,
        )
        assert score == -np.inf, "grid-only endpoint checks cannot authorize recovery"
    else:
        costs = mppi_costs(adapter, obs, [(1.2, 0.0), (0.0, 0.1)])
        assert np.all(costs >= adapter.config.invalid_sequence_cost), costs


def test_guard_unbound_grid_preserves_legacy_stop_for_penetrating_fallback():
    """A real default DWA fallback must not win an unbound static/progress tie."""
    guard = release("guarded_ppo")
    # The guard supports standalone command planners, including default DWA;
    # its own recovery boundary must hold independently of fallback settings.
    guard.fallback_adapter = RiskDWAPlannerAdapter()
    normal = np.array([-1.0, -1.0]) / np.sqrt(2)
    tangent = np.array([1.0, -1.0]) / np.sqrt(2)
    midpoint = np.array([23.0, 23.0]) + 0.9995 * normal
    start = midpoint - 0.059 * tangent
    obs = observation(guard, CORNER_MAP, start, -np.pi / 4, goal=start + 4 * tangent)
    map_def = convert_map(str(CORNER_MAP))
    lines, polygons = RobotEnv._normalize_obstacles_for_grid(map_def.obstacles, map_def.bounds)
    grid = OccupancyGrid(
        GridConfig(
            resolution=0.2,
            width=32,
            height=32,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
            use_ego_frame=False,
            center_on_robot=False,
        )
    )
    grid.generate(
        lines, [], (tuple(start), -np.pi / 4), ego_frame=False, obstacle_polygons=polygons
    )
    obs["occupancy_grid"] = grid.to_observation()
    obs["occupancy_grid_meta"] = grid.metadata_observation()
    drive = DifferentialDriveRobot(DifferentialDriveSettings(radius=1.0))
    # Reach the observed speed through native acceleration, keeping wheel
    # speeds consistent with velocity before executing the corner chord.
    for _ in range(12):
        drive.apply_action((1.0, 0.0), 0.1)
    drive.state.pose = (tuple(start), -np.pi / 4)
    obs["robot"]["speed"] = list(drive.current_speed)
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[drive]))
    config = SimpleNamespace(
        robot_config=drive.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    fallback = guard.fallback_adapter.plan(obs)
    assert fallback == pytest.approx((1.2, 0.0))
    action = policy_command_to_env_action(env=env, config=config, command=fallback)
    drive.apply_action(tuple(action), 0.1)
    end = np.asarray(drive.pos)
    np.testing.assert_allclose(end, start + 0.12 * tangent, rtol=0, atol=1e-12)
    # Independent circle/corner oracle: clear endpoints hide 0.5 mm penetration.
    assert np.linalg.norm(start - [23, 23]) > 1
    assert np.linalg.norm(end - [23, 23]) > 1
    assert np.linalg.norm(midpoint - [23, 23]) - 1 == pytest.approx(-0.0005)
    assert guard._exact_obstacle_clearance(end, previous=start) == pytest.approx(-0.0005)
    bound = guard.choose_command_decision(obs, (0.0, 0.0))
    assert bound.filtered_action == (0.0, 0.0)
    assert bound.decision_label == "stop_best_effort"

    # Use the public lifecycle to remove exact geometry from guard and fallback.
    guard.bind_env(SimpleNamespace())
    assert not guard._static_recovery_available()
    c0 = guard._min_obstacle_clearance(start, observation=obs)
    c1 = guard._min_obstacle_clearance(end, observation=obs)
    assert 0 < c0 < c1 < guard.config.hard_obstacle_clearance
    assert guard.fallback_adapter.plan(obs) == pytest.approx(fallback)
    fallback_eval = guard._evaluate_command(obs, fallback)
    stop_eval = guard._evaluate_command(obs, (0.0, 0.0))
    assert not fallback_eval["safe"]
    assert not stop_eval["safe"]
    assert fallback_eval["min_obs_clear"] > stop_eval["min_obs_clear"]
    # 0cf58853's strict pedestrian-clearance arbitration stops on this inf tie,
    # irrespective of the apparent static-clearance or progress improvement.
    assert fallback_eval["min_ped_clear"] == stop_eval["min_ped_clear"] == np.inf
    decision = guard.choose_command_decision(obs, (0.0, 0.0))
    assert decision.filtered_action == (0.0, 0.0), (
        "unbound endpoint-only grid checks cannot select a below-margin translation"
    )
    assert decision.decision_label == "stop_best_effort"
    assert decision.selected_evaluation == stop_eval
    assert guard.diagnostics()["recovery_command_count"] == 0


@pytest.mark.parametrize("clearance", [0.01, 0.1])
def test_mppi_moving_recovery_rejects_native_braking_into_wall(clearance):
    """At 0.3 m/s a zero-speed command still coasts 0.025 m in step one."""
    adapter = release("predictive_mppi")
    obs = observation(adapter, CORRIDOR_MAP, [10.0, 11.0 - clearance], np.pi / 2, goal=[14, 10])
    obs["robot"]["speed"] = [0.3]
    drive = DifferentialDriveRobot(DifferentialDriveSettings(radius=1.0))
    drive.reset_state((tuple(obs["robot"]["position"]), np.pi / 2))
    drive.state.velocity = (0.3, 0.0)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[drive]))
    config = SimpleNamespace(
        robot_config=drive.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    action = policy_command_to_env_action(env=env, config=config, command=(0.0, -1.0))
    drive.apply_action(tuple(action), 0.1)
    assert drive.current_speed == pytest.approx((0.2, -0.1))
    actual_clearance = 12 - drive.pos[1] - 1
    assert actual_clearance == pytest.approx(clearance - 0.024999921875, abs=1e-10)
    assert actual_clearance < clearance
    costs = mppi_costs(adapter, obs, [(0, -1), (0, 0), (0.05, -1)])
    assert np.all(costs >= adapter.config.invalid_sequence_cost), "actual coast decreases clearance"
    assert adapter.plan(obs) == (0.0, 0.0)
    diagnostics = adapter.diagnostics()
    assert diagnostics["no_admissible_command"]
    assert not diagnostics["recovery_command"]


@pytest.mark.parametrize("deceleration", [0.5, 2.0])
def test_mppi_recovery_matches_native_motion_and_full_braking_coast(monkeypatch, deceleration):
    """Moving away is feasible; honor bound braking authority, yaw and odometry."""
    adapter = release("predictive_mppi")
    obs = observation(adapter, CORRIDOR_MAP, [10.0, 10.9], -np.pi / 2, goal=[14, 10])
    obs["robot"]["speed"] = [0.3]
    obs["robot"]["angular_velocity"] = [0.4]
    drive = DifferentialDriveRobot(
        DifferentialDriveSettings(radius=1.0, max_linear_decel=deceleration)
    )
    drive.reset_state((tuple(obs["robot"]["position"]), -np.pi / 2))
    drive.state.velocity = (0.3, 0.4)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    map_def = convert_map(str(CORRIDOR_MAP))
    lines, polygons = RobotEnv._normalize_obstacles_for_grid(map_def.obstacles, map_def.bounds)
    env = SimpleNamespace(
        simulator=SimpleNamespace(robots=[drive]),
        _get_static_grid_obstacles=lambda: (lines, polygons),
    )
    adapter.bind_env(env)
    config = SimpleNamespace(
        robot_config=drive.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    command = (0.0, -1.0)
    action = policy_command_to_env_action(env=env, config=config, command=command)
    drive.apply_action(tuple(action), 0.1)
    first_position = np.asarray(drive.pos)
    while drive.current_speed[0] > 1e-9:
        action = policy_command_to_env_action(env=env, config=config, command=(0, 0))
        drive.apply_action(tuple(action), 0.1)
    stopped_position = np.asarray(drive.pos)
    queried = []
    original = adapter._exact_obstacle_clearance

    def record(point, *, previous=None):
        if previous is not None:
            queried.append(np.asarray(point).copy())
        return original(point, previous=previous)

    monkeypatch.setattr(adapter, "_exact_obstacle_clearance", record)
    costs = mppi_costs(adapter, obs, [command], horizon=1)
    assert costs[0] < adapter.config.invalid_sequence_cost
    assert any(np.allclose(p, first_position, rtol=0, atol=1e-10) for p in queried), (
        "recovery must score native drive motion from the actual speed and yaw rate"
    )
    assert any(np.allclose(p, stopped_position, rtol=0, atol=1e-10) for p in queried), (
        "a short optimizer horizon must still check the complete braking coast"
    )


@pytest.mark.parametrize(
    "clearance,velocity,violation",
    [
        (0.4, (0, 0), "pedestrian_clearance"),
        (0.65, (0, 0), "first_step_pedestrian_clearance"),
        (0.8, (0, 0), None),
        (1.5, (0, 3.0), "time_to_collision"),
    ],
)
def test_guard_pedestrian_present_preserves_previous_arbitration(clearance, velocity, violation):
    """Static recovery must not change pedestrian-present tie behavior or gates."""
    guard = release("guarded_ppo")
    position = np.array([19.519706, 9.788156])
    obs = observation(guard, OVERTAKING_MAP, position, 0.872014, goal=[19, 14])
    obs["pedestrians"].update(
        count=[1],
        positions=np.asarray([position + [0, -(1.4 + clearance)]]),
        # SocNav observations express velocity in the robot's ego frame.
        velocities=np.asarray([[np.sin(0.872014) * velocity[1], np.cos(0.872014) * velocity[1]]]),
    )
    proposed = (0.0, 0.0)
    decision = guard.choose_command_decision(obs, proposed)
    assert decision.filtered_action == (0.0, 0.0), "0cf58853 stops when pedestrian clearances tie"
    assert decision.decision_label == "stop_best_effort"
    evaluation = decision.selected_evaluation
    if violation:
        assert violation in guard._violated_constraints(evaluation)
    else:
        assert evaluation["min_ped_clear"] >= guard.config.hard_ped_clearance
        assert evaluation["first_ped_clear"] >= guard.config.first_step_ped_clearance
        assert not np.isfinite(evaluation["min_ttc"])
    assert guard.diagnostics()["recovery_command_count"] == 0
    # The empty-world positive control is covered by test_fx3_static_recovery;
    # also pin safe-PPO authority here when a pedestrian is present.
    safe_obs = observation(guard, CORRIDOR_MAP, [10.0, 10.0], goal=[14, 10])
    safe_obs["pedestrians"].update(
        count=[1], positions=np.array([[10.0, 6.0]]), velocities=np.zeros((1, 2))
    )
    safe = guard.choose_command_decision(safe_obs, proposed)
    assert safe.filtered_action == proposed
    assert safe.selected_evaluation["safe"]
