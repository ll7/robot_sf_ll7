"""Deterministic production-path witnesses for planner adapter residuals."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_native_command import (
    NativeCommandStepError,
    _parse_response,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import policy_command_to_env_action
from robot_sf.benchmark.runner import _NativeCommandPolicy
from robot_sf.planner import socnav_sampling_v2 as sampling
from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, GuardedPPOConfig
from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter
from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


def _observation(*, speed=0.0, angular=0.0, pedestrian=None):
    """Build an unnormalised adapter observation without resetting a simulator."""
    positions = np.asarray([] if pedestrian is None else [pedestrian], dtype=float).reshape(-1, 2)
    return {
        "robot": {
            "position": np.zeros(2),
            "heading": np.zeros(1),
            "speed": np.array([speed]),
            "angular_velocity": np.array([angular]),
            "radius": np.array([1.0]),
        },
        "goal": {"current": np.array([10.0, 0.0])},
        "pedestrians": {
            "positions": positions,
            "velocities": np.zeros_like(positions),
            "count": np.array([len(positions)]),
            "radius": np.array([0.4]),
        },
        "sim": {"timestep": np.array([0.1])},
    }


def test_guard_checks_pedestrians_until_braking_finishes():
    """The reported .82 m horizon gap must include the unsafe .50 m stopping gap."""
    guard = GuardedPPOAdapter(
        GuardedPPOConfig(
            clearance_model="surface_v2", robot_radius_m=1.0, pedestrian_radius_m=0.4, min_ttc=0.0
        )
    )
    observation = _observation(speed=2.0, pedestrian=(3.9, 0.0))
    result = guard._evaluate_command(observation, (0.0, 0.0))
    assert result["min_ped_clear"] == pytest.approx(0.5)
    assert not result["safe"]


def test_legacy_guard_ttc_is_first_contact_not_closest_approach():
    """A near miss has infinite contact time even when closest approach is imminent."""
    guard = GuardedPPOAdapter(GuardedPPOConfig(rollout_dt=0.1, rollout_steps=1))
    result = guard._evaluate_command(_observation(pedestrian=(1.0, 2.0)), (1.0, 0.0))
    assert np.isinf(result["min_ttc"])


def test_guard_grid_fallback_keeps_pedestrians_out_of_static_clearance():
    """Unbound geometry must read the static channel rather than combined occupancy."""
    guard = GuardedPPOAdapter()
    observation = _observation()
    grid = np.zeros((4, 20, 20))
    grid[[1, 3], 10, 10] = 1.0
    meta = {
        "origin": [-1.0, -1.0],
        "resolution": [0.1],
        "size": [2.0, 2.0],
        "channel_indices": [0, 1, 2, 3],
    }
    assert np.isinf(
        guard._min_obstacle_clearance(np.zeros(2), observation, grid_payload=(grid, meta))
    )
    grid[0, 10, 10] = 1.0
    assert guard._min_obstacle_clearance(np.zeros(2), observation, grid_payload=(grid, meta)) == 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("raw_action", [False, True])
def test_sacadrl_nonfinite_scores_stop_and_record_fallback(monkeypatch, bad, raw_action):
    """Invalid model scores cannot select a full-speed action through argmax."""
    adapter = SACADRLPlannerAdapter()
    model = SimpleNamespace(
        actions=np.full((2, 2), bad) if raw_action else np.array([[1.0, -0.5], [0.5, 0.0]]),
        predict=lambda _: np.array([[0.1, 0.2]]) if raw_action else np.array([[bad, bad]]),
    )
    monkeypatch.setattr(adapter, "_ensure_model", lambda: model)
    monkeypatch.setattr(adapter, "_build_network_input", lambda _: (np.zeros(3), 1.0, 10.0))
    assert adapter.plan(_observation()) == (0.0, 0.0)
    provenance = adapter.diagnostics()["checkpoint_provenance"]
    assert provenance["fallback_triggered"] is True
    assert provenance["fallback_reason"] == "nonfinite_model_output"
    model.actions = np.array([[1.0, -0.5], [0.5, 0.0]])
    model.predict = lambda _: np.array([[0.1, 0.2]])
    assert adapter.plan(_observation()) == (0.5, 0.0)


@pytest.mark.parametrize("parser", ["map", "classic"])
@pytest.mark.parametrize(
    "payload",
    [
        '{"vx":0,"vy":1}',
        '{"v":1,"omega":0,"unexpected":1}',
        '{"v":1,"omega":0,"linear":2}',
    ],
)
def test_native_command_rejects_unknown_or_ambiguous_keys(parser, payload):
    """Both real parsers refuse holonomic aliases, unknown keys and conflicting pairs."""

    def parse(text):
        if parser == "map":
            return _parse_response(text)
        return _NativeCommandPolicy._parse_response(None, text)

    error = NativeCommandStepError if parser == "map" else ValueError
    with pytest.raises(error):
        parse(payload)
    finite = parse('{"v":0.5,"omega":0.25}')
    np.testing.assert_allclose(finite, [0.5, 0.25])


@pytest.mark.parametrize("rollout_dt", [0.1, 0.2])
def test_prediction_score_rolls_out_bound_drive_from_observed_velocity(monkeypatch, rollout_dt):
    """Scoring a command must use the same accelerated turning pose as the bound plant."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=rollout_dt))
    settings = DifferentialDriveSettings(max_linear_accel=0.4, max_angular_accel=0.3)
    drive = DifferentialDriveRobot(settings)
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[drive]))
    adapter.bind_env(env)
    config = SimpleNamespace(
        robot_config=drive.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    observation = _observation(speed=0.4, angular=0.2)
    drive.state.velocity = (0.4, 0.2)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    expected = []
    for _ in range(3):
        for _ in range(round(rollout_dt / 0.1)):
            action = policy_command_to_env_action(env=env, config=config, command=(1.5, 0.8))
            drive.apply_action(tuple(action), 0.1)
        expected.append(drive.pos)

    def check_progress(*args, robot_traj, **kwargs):
        np.testing.assert_allclose(robot_traj, expected, atol=1e-14, rtol=0.0)
        return 0.0

    monkeypatch.setattr(adapter, "_goal_progress", check_progress)
    adapter._score_action(
        observation=observation,
        future_peds=np.zeros((0, 3, 2)),
        mask=np.zeros(0),
        v=1.5,
        w=0.8,
        steps=3,
    )


def test_prediction_sequence_rollout_uses_measured_drive_state(monkeypatch):
    """Sequence search must share the accelerated wheel odometry used by one-action scoring."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=0.1))
    drive = DifferentialDriveRobot(DifferentialDriveSettings(max_angular_accel=0.3))
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[drive]))
    adapter.bind_env(env)
    config = SimpleNamespace(
        robot_config=drive.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    drive.state.velocity = (0.4, 0.2)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    sequence = [(1.5, 0.8), (0.0, -0.5)]
    expected = []
    for command in np.repeat(sequence, 2, axis=0):
        action = policy_command_to_env_action(env=env, config=config, command=tuple(command))
        drive.apply_action(tuple(action), 0.1)
        expected.append(drive.pos)
    real_rollout = adapter._rollout_robot_sequence

    def check_rollout(**kwargs):
        positions, headings = real_rollout(**kwargs)
        np.testing.assert_allclose(positions, expected, atol=1e-14, rtol=0.0)
        return positions, headings

    monkeypatch.setattr(adapter, "_rollout_robot_sequence", check_rollout)
    adapter._score_action_sequence(
        observation=_observation(speed=0.4, angular=0.2),
        future_peds=np.zeros((0, 4, 2)),
        mask=np.zeros(0),
        sequence=sequence,
        steps=4,
    )


@pytest.mark.parametrize("planner", ["guard", "sampler"])
def test_static_grid_fallback_refuses_an_unseparated_combined_channel(planner):
    """Combined-only occupancy cannot identify which cells are static obstacles."""
    observation = _observation()
    observation["occupancy_grid"] = np.ones((1, 20, 20))
    observation["occupancy_grid_meta"] = {
        "origin": [-1.0, -1.0],
        "resolution": [0.1],
        "size": [2.0, 2.0],
        "channel_indices": [-1, -1, -1, 0],
    }
    guard = GuardedPPOAdapter()
    with pytest.raises(ValueError, match="static obstacle channel"):
        if planner == "guard":
            guard._min_obstacle_clearance(np.zeros(2), observation)
        else:
            sampling._ObstacleClearance(guard, observation)
