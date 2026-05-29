"""Tests for the stochastic random reference baseline policy."""

from __future__ import annotations

import numpy as np

from robot_sf.baselines.interface import Observation as SharedObservation
from robot_sf.baselines.random_policy import Observation, RandomPlanner


def _obs() -> Observation:
    """Build a minimal Observation payload for RandomPlanner unit tests."""
    return Observation(
        dt=0.1,
        robot={
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [1.0, 0.0],
            "radius": 0.3,
        },
        agents=[],
        obstacles=[],
    )


def test_random_planner_seed_reproducibility_velocity_mode() -> None:
    """Equal seeds should generate identical random action sequences."""
    planner_a = RandomPlanner({"mode": "velocity", "v_max": 1.0}, seed=123)
    planner_b = RandomPlanner({"mode": "velocity", "v_max": 1.0}, seed=123)

    seq_a = [planner_a.step(_obs()) for _ in range(5)]
    seq_b = [planner_b.step(_obs()) for _ in range(5)]

    assert seq_a == seq_b


def test_random_planner_velocity_actions_respect_speed_limit() -> None:
    """Velocity samples should stay within the configured maximum speed."""
    planner = RandomPlanner({"mode": "velocity", "v_max": 1.5}, seed=0)
    for _ in range(100):
        action = planner.step(_obs())
        speed = float(np.hypot(action["vx"], action["vy"]))
        assert speed <= 1.5 + 1e-6


def test_random_planner_unicycle_actions_respect_limits() -> None:
    """Unicycle samples should respect linear and angular limits."""
    planner = RandomPlanner({"mode": "unicycle", "v_max": 1.0, "omega_max": 0.7}, seed=0)
    for _ in range(100):
        action = planner.step(_obs())
        assert 0.0 <= float(action["v"]) <= 1.0 + 1e-6
        assert -0.7 - 1e-6 <= float(action["omega"]) <= 0.7 + 1e-6


def test_random_planner_uses_shared_observation_alias() -> None:
    """The random baseline should expose the shared baseline observation container."""
    assert Observation is SharedObservation


def test_random_planner_accepts_dict_observation_without_obstacles() -> None:
    """Dict observations should normalize through the shared observation container."""
    planner = RandomPlanner({"mode": "velocity", "v_max": 1.0}, seed=0)
    obs = {
        "dt": 0.1,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [1.0, 0.0],
            "radius": 0.3,
        },
        "agents": [],
    }

    action = planner.step(obs)

    assert set(action) == {"vx", "vy"}
