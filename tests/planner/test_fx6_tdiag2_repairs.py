"""TDIAG2 dev-seed 1001/1002 states on unchanged release SVGs; no episodes.

Numeric contact endpoints are retained trace bytes, independent of the planner.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.map_runner_actions import policy_command_to_env_action
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings
from tests.planner.test_fx3_static_recovery import ROOT, observation
from tests.planner.test_fx4_recovery_safety import mppi_costs, release

STATES = json.loads(Path(__file__).with_name("tdiag2_fx6_states.json").read_text())


def cases(kind):
    """Preserve scenario/seed identities in pytest node IDs."""
    return [pytest.param(r, id=f"{r['scenario']}-{r['seed']}") for r in STATES if r["kind"] == kind]


def recorded_observation(adapter, row):
    """Parse the exact retained map and match guard float32 observation poses."""
    map_path = ROOT / "maps" / row["map"]
    assert hashlib.sha256(map_path.read_bytes()).hexdigest() == row["map_sha256"]
    obs = observation(adapter, map_path, row["position"], row["heading"], goal=row["goal"])
    obs["robot"]["heading"] = [float(np.float32(row["heading"]))]
    obs["robot"]["speed"] = [row.get("speed", 0.0)]
    obs["robot"]["angular_velocity"] = [row.get("angular", 0.0)]
    return obs


@pytest.mark.parametrize("row", cases("contact"))
def test_guard_stop_forecasts_recorded_native_contact(row):
    """All ten late stop commands still coast into the retained wall/corner."""
    guard = release("guarded_ppo")
    obs = recorded_observation(guard, row)
    drive = DifferentialDriveRobot(DifferentialDriveSettings())
    drive.reset_state((tuple(row["position"]), row["heading"]))
    drive.state.velocity = (row["speed"], row["angular"])
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    drive.apply_action(tuple(-np.asarray(drive.current_speed) / 0.1), 0.1)
    np.testing.assert_allclose(drive.pos, row["post"], rtol=0, atol=2e-6)
    assert guard._exact_obstacle_clearance(np.asarray(drive.pos)) < 0
    evaluation = guard._evaluate_command(obs, (0.0, 0.0))
    assert evaluation["min_obs_clear"] < 0, "zero speed request cannot erase native braking coast"
    assert not evaluation["safe"]


@pytest.mark.parametrize("row", cases("dead_band"))
def test_mppi_dead_band_admits_margin_preserving_rotation(row):
    """Seven rest states retain the hard margin without impossible first-step padding."""
    adapter = release("predictive_mppi")
    obs = recorded_observation(adapter, row)
    clearance = adapter._min_obstacle_clearance(np.asarray(row["position"]), observation=obs)
    assert 0.30 < clearance < 0.35
    costs = mppi_costs(adapter, obs, [(0, -1), (0, 0), (0, 1)])
    assert np.all(costs < adapter.config.invalid_sequence_cost), (
        "dead band rejects even safe rotation"
    )
    # The first-step exception never permits loss of the 0.30 m hard margin.
    invalid = adapter._hard_constraint_cost(
        min_clear=np.inf,
        min_obs=0.299,
        first_clear=np.inf,
        first_obs=clearance,
        current_obs=clearance,
    )
    assert invalid is not None
    adapter.plan(obs)
    assert not adapter.diagnostics()["no_admissible_command"]


@pytest.mark.parametrize("row", cases("arbitration"))
def test_guard_selects_executable_forward_recovery_over_vetoed_ppo(row):
    """Three exact-target states must compare fallback with executable stop only."""
    guard = release("guarded_ppo")
    obs = recorded_observation(guard, row)
    fallback = guard.fallback_adapter.plan(obs)
    assert fallback == pytest.approx(row["fallback"])
    assert guard.fallback_adapter.diagnostics()["recovery_command"]
    proposal = tuple(row["proposal"])
    assert not guard._evaluate_command(obs, proposal)["safe"]
    decision = guard.choose_command_decision(obs, proposal)
    assert decision.filtered_action == pytest.approx(fallback), "vetoed PPO cannot defeat recovery"
    assert decision.decision_label == "fallback_best_effort"
    assert decision.selected_evaluation["min_obs_clear"] >= row["clearance"]
    assert decision.selected_evaluation["progress"] > 0
    assert decision.hard_constraint_violation
    assert not guard.diagnostics()["no_admissible_command"]


@pytest.mark.parametrize("command", [(2.0, -1.0), (0.1, -0.1), (0.0, 0.0)])
@pytest.mark.parametrize("deceleration", [0.5, 1.0, 2.0])
def test_guard_all_command_rollouts_follow_bound_native_drive(monkeypatch, command, deceleration):
    """Proposal, fallback and stop use live yaw, accel and braking at a TDIAG2 pose."""
    row = next(
        r for r in STATES if r["kind"] == "contact" and r["scenario"] == "francis2023_exiting_room"
    )
    guard = release("guarded_ppo")
    obs = recorded_observation(guard, row)
    settings = DifferentialDriveSettings(max_linear_accel=0.5, max_linear_decel=deceleration)
    drive = DifferentialDriveRobot(settings)
    drive.reset_state((tuple(row["position"]), obs["robot"]["heading"][0]))
    drive.state.velocity = (1.2, 0.4)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    obs["robot"]["angular_velocity"] = [0.4]
    map_def = convert_map(str(ROOT / "maps" / row["map"]))
    lines, polygons = RobotEnv._normalize_obstacles_for_grid(map_def.obstacles, map_def.bounds)
    env = SimpleNamespace(
        simulator=SimpleNamespace(robots=[drive]),
        _get_static_grid_obstacles=lambda: (lines, polygons),
    )
    guard.bind_env(env)
    config = SimpleNamespace(
        robot_config=settings, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    action = policy_command_to_env_action(env=env, config=config, command=command)
    drive.apply_action(tuple(action), 0.1)
    queried = []
    original = guard._exact_obstacle_clearance

    def capture(point, *, previous=None):
        if previous is not None:
            queried.append(np.asarray(point))
        return original(point, previous=previous)

    monkeypatch.setattr(guard, "_exact_obstacle_clearance", capture)
    guard._evaluate_command(obs, command)
    np.testing.assert_allclose(queried[0], drive.pos, rtol=0, atol=1e-12)


@pytest.mark.parametrize("arm", ["guarded_ppo", "predictive_mppi"])
def test_guard_and_mppi_nominal_stop_check_full_braking_tail(arm):
    """TDIAG2 bottleneck step 60 is margin-clear but lacks full stopping distance."""
    map_path = (
        ROOT
        / "maps/successor_svg_maps/issue_9762_classic_realworld_bottleneck_goal_zone_entry_v2.svg"
    )
    position = [15.02338230061961, 12.610681278120751]
    heading = -0.001220828853547573
    adapter = release(arm)
    obs = observation(adapter, map_path, position, heading, goal=[17, 15])
    obs["robot"]["speed"] = [2.0]
    obs["robot"]["angular_velocity"] = [-0.040542590618133506]
    assert adapter._min_obstacle_clearance(np.asarray(position), observation=obs) > 0.35
    if arm == "guarded_ppo":
        result = adapter._evaluate_command(obs, (0, 0))
        assert result["min_obs_clear"] < 0, "1.2 s horizon must include the 2 s stop coast"
        assert not result["safe"]
    else:
        costs = mppi_costs(adapter, obs, [(0, 0)])
        assert costs[0] >= adapter.config.invalid_sequence_cost
