"""Exercise the contact handoff through the sampler's policy factory."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import build_map_policy
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveSettings

CONFIG = Path(__file__).resolve().parents[2] / "configs/algos/velocity_sampler_dev_r9.yaml"


def _approach(pedestrian_x, pedestrian_y=0.0, *, half_plane_weight=0.0):
    config = yaml.safe_load(CONFIG.read_bytes())
    config["sampling_path_distance"] = False
    config["velocity_sampler"]["orca_half_plane_weight"] = half_plane_weight
    policy: Any
    policy, _ = build_map_policy("velocity_sampler", config, robot_kinematics="differential_drive")
    adapter = policy._planner_adapter
    walls = [((-5.0, y), (15.0, y)) for y in (-3.6, 3.6)]
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: walls),
        )
    )
    adapter.reset(seed=1163)
    observation = {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [0.6, 0.0],
            "radius": [1.0],
        },
        "goal": {"current": [10.0, 0.0]},
        "pedestrians": {
            "positions": [[pedestrian_x, pedestrian_y]],
            "velocities": [[-0.3, 0.0]],
            "count": [1],
            "radius": [0.4],
        },
        "sim": {"timestep": [0.1]},
    }
    return adapter, observation


def test_moving_sampler_hands_off_before_contact(monkeypatch) -> None:
    """A moving robot without a clear forecast can use a reachable ORCA target."""
    calls = []

    def orca_plan(self, observation):
        calls.append(observation)
        return 0.0, 0.6

    monkeypatch.setattr(ORCAPlannerAdapter, "plan", orca_plan)
    adapter, observation = _approach(1.8)

    assert adapter.plan(observation) == pytest.approx((0.5, 0.1))
    assert len(calls) == 1
    assert adapter.diagnostics()["orca_recovery_steps"] == 1
    assert adapter.diagnostics()["orca_contact_steps"] == 1
    assert adapter.diagnostics()["orca_contact_requests"] == 1

    adapter.reset(seed=1163)
    assert adapter.diagnostics()["orca_contact_steps"] == 0
    assert adapter.diagnostics()["orca_contact_requests"] == 0


def test_contact_handoff_rejects_unsafe_orca_braking_path(monkeypatch) -> None:
    """The imminent-contact trigger still requires the ORCA path to clear."""
    calls = []

    def orca_plan(self, observation):
        calls.append(observation)
        return 2.0, 0.6

    monkeypatch.setattr(ORCAPlannerAdapter, "plan", orca_plan)
    adapter, observation = _approach(1.8)

    assert adapter.plan(observation) == pytest.approx((0.5, -0.1))
    assert len(calls) == 1
    assert adapter.diagnostics()["orca_contact_requests"] == 1
    assert adapter.diagnostics()["orca_contact_steps"] == 0


def test_clear_sample_keeps_the_moving_sampler_in_control(monkeypatch) -> None:
    """An available clear profile leaves control with the moving sampler."""

    def orca_plan(self, observation):
        pytest.fail("clear sampled motion should keep the sampler in control")

    monkeypatch.setattr(ORCAPlannerAdapter, "plan", orca_plan)
    adapter, observation = _approach(2.0, 2.2)

    adapter.plan(observation)
    assert adapter.diagnostics()["orca_contact_requests"] == 0


def test_contact_handoff_requires_a_new_prediction_at_next_step(monkeypatch) -> None:
    """A previous contact request alone cannot extend the handoff."""
    calls = []

    def orca_plan(self, observation):
        calls.append(observation)
        return 0.0, 0.6

    monkeypatch.setattr(ORCAPlannerAdapter, "plan", orca_plan)
    adapter, observation = _approach(1.8)

    assert adapter.plan(observation) == pytest.approx((0.5, 0.1))
    observation["robot"]["speed"] = [0.5, 0.0]
    observation["pedestrians"]["positions"] = [[2.4, 0.0]]
    observation["pedestrians"]["velocities"] = [[0.0, 0.0]]
    adapter.plan(observation)

    assert len(calls) == 1
    assert adapter.diagnostics()["orca_recovery_steps"] == 1


@pytest.mark.parametrize("algo", ["velocity_sampler", "dra_mppi"])
def test_candidate_factory_requires_explicit_opt_in(algo: str) -> None:
    """Direct factory users must opt in before constructing either candidate."""
    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        build_map_policy(algo, {"sampling_pedestrian_prediction": True})


def test_orca_half_plane_cost_penalizes_motion_toward_a_walker() -> None:
    """The cost follows the first swept velocity projected onto the ego plane."""
    adapter, observation = _approach(0.0, 2.4, half_plane_weight=1.0)
    observation["robot"]["heading"] = [np.pi / 2]
    observation["robot"]["speed"] = [0.0, 0.0]
    observation["pedestrians"]["velocities"] = [[0.0, 0.0]]
    adapter.plan(observation)
    cutoff = (2.4 - 1.4) / (2 * adapter._orca.config.orca_time_horizon)
    points = np.array(
        [[[0.0, 0.0], [0.0, distance * 0.1]] for distance in (0.0, cutoff, cutoff + 0.2)]
    )
    np.testing.assert_allclose(
        adapter._orca_half_plane_cost(
            points, np.array([0.0, 0.1]), np.array([[0.0, 2.4]]), np.zeros((1, 2))
        ),
        [0.0, 0.0, 0.2],
        atol=1e-12,
    )
