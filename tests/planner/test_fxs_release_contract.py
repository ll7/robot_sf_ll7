"""B1/B2 release bytes and B4 native-drive forecast regressions (#10007)."""

from math import cos, pi, sin
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import _build_socnav_config
from robot_sf.planner.socnav_base import SamplingPlannerAdapter, SocNavPlannerConfig
from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter, _SACADRLModel
from robot_sf.planner.socnav_sampling_v2 import _rollout
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


def release_adapter():
    """Load the actual release config through the benchmark filtering path."""
    payload = yaml.safe_load(Path("configs/algos/socnav_release_v0_0_8.yaml").read_text())
    return SACADRLPlannerAdapter(_build_socnav_config(payload))


def crowd_observation():
    """Twenty shuffled pedestrians with real count, padding and release geometry."""
    positions = np.zeros((64, 2))
    positions[:20, 0] = [12, 3, 21, 7, 15, 2, 18, 5, 10, 20, 4, 16, 8, 19, 6, 14, 9, 17, 11, 13]
    return {
        "robot_position": [0.0, 0.0],
        "robot_heading": [0.0],
        "robot_radius": [1.0],
        "robot_speed": [1.0],
        "robot_angular_velocity": [-0.6],
        "goal_current": [30.0, 0.0],
        "goal_next": [30.0, 0.0],
        "pedestrians_positions": positions,
        "pedestrians_velocities": np.zeros((64, 2)),
        "pedestrians_count": [20],
        "pedestrians_radius": [0.4],
        "sim": {"timestep": [0.1]},
    }


def test_release_preferred_speed_reaches_checkpoint_host_input():
    """Pin 1 m/s: at 2 m/s the 1 m/s² braking lag defeats the checkpoint's stop action."""
    vec, preferred, _ = release_adapter()._build_network_input(crowd_observation())
    reason = (
        "release must prefer 1 m/s: 1 m/s² braking cannot execute the checkpoint's instant stop"
    )
    assert preferred == 1.0, reason
    assert vec[0, 3] == 1.0, reason


def test_release_observes_nineteen_nearest_agents_without_padding_in_sequence():
    """FullTestSuite's 19 slots must reach the checkpoint, excluding padded rows."""
    adapter = release_adapter()
    obs = crowd_observation()
    vec, _, _ = adapter._build_network_input(obs)
    assert vec[0, 0] == 19.0, "release sequence must include 19 agents, not 3"
    assert vec.shape == (1, 138)
    np.testing.assert_array_equal(vec[0, 5:].reshape(19, 7)[:, 0], np.arange(2, 21))
    obs["pedestrians_count"] = [5]
    vec, _, _ = adapter._build_network_input(obs)
    assert vec[0, 0] == 5
    assert not np.any(vec[0, 5 + 5 * 7 :])


def test_checkpoint_rejects_configured_agent_count_above_nineteen():
    """An oversized sequence must fail before inference instead of truncating agent slots."""
    adapter = release_adapter()
    adapter.config.sacadrl_max_other_agents = 20
    vec, _, _ = adapter._build_network_input(crowd_observation())
    assert vec[0, 0] == 20
    model = _SACADRLModel.__new__(_SACADRLModel)
    model._input_dim = 138  # Bundled checkpoint: five host/count features + 19 * 7.
    model._x = "X:0"
    model._softmax = "Softmax:0"
    calls = []
    model._sess = SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError, match="checkpoint capacity.*19 agent slots"):
        model.predict(vec)
    assert not calls, "oversized observations must be rejected before TensorFlow inference"


def test_rollout_first_step_matches_trapezoidal_drive_ramp():
    """From rest the first step covers 5 mm and turns 0.005 rad, not 0.1 rad."""
    points, distance = _rollout(np.zeros(2), 0.0, 0.0, pi / 2, 2.0, 0.1, 0.1, (1.0, 1.0, 1.0, 1.0))
    # Native wheel odometry averages endpoint velocities and midpoint headings.
    assert points[0, 0] == pytest.approx(0.005 * cos(0.0025), abs=1e-12)
    assert points[0, 1] == pytest.approx(0.005 * sin(0.0025), abs=1e-12)
    assert distance[0] == pytest.approx(0.005, abs=1e-12)


def test_sampling_forecast_uses_bound_angular_acceleration_and_observed_turn_rate(monkeypatch):
    """Planning with a counter-turn must seed the drive and use its bound limits."""
    from robot_sf.planner import socnav_sampling_v2 as module

    settings = DifferentialDriveSettings(
        max_angular_accel=0.25, max_angular_speed=0.8, wheel_radius=0.09, interaxis_length=0.7
    )
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(socnav_sampling_version="bounded_v2"))
    adapter.bind_env(SimpleNamespace(env_config=SimpleNamespace(robot_config=settings)))
    obs = crowd_observation()
    obs["pedestrians_count"] = [0]
    obs["goal_current"] = [0.0, 30.0]
    original = module._rollout
    checked = []

    def inspect(*args, **kwargs):
        points, distance = original(*args, **kwargs)
        start, heading, speed0, target, speed, _, dt, limits = args
        # Oracle crosses the consumer boundary by applying the command to the actual robot.
        robot = DifferentialDriveRobot(settings)
        robot.state.pose = (tuple(start), heading)
        robot.state.velocity = (speed0, -0.6)
        robot.state.wheel_speeds = robot.movement._resulting_wheel_speeds(robot.current_speed)
        error = np.arctan2(np.sin(target - heading), np.cos(target - heading))
        command = np.array([speed, np.clip(limits[0] * error, -0.8, 0.8)])
        robot.apply_action(tuple((command - robot.current_speed) / dt), dt)
        np.testing.assert_allclose(
            points[0],
            robot.pos,
            atol=1e-12,
            err_msg="forecast must start from measured omega and bound angular ramp",
        )
        checked.append(points[0])
        return points, distance

    monkeypatch.setattr(module, "_rollout", inspect)
    adapter.plan(obs)
    assert checked
