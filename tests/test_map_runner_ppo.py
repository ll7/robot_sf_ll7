"""Unit tests for PPO support in map_runner policy bridge."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark import map_runner


class _DummyPPOPlanner:
    """Test double for PPO planner integration in map runner."""

    def __init__(self, config, *, seed=None):
        self.config = dict(config)
        self.seed = seed
        self.closed = False

    def step(self, _obs):
        return self.config.get("test_action", {"v": 0.5, "omega": 0.0})

    def close(self):
        self.closed = True

    def get_metadata(self):
        return {"algorithm": "ppo", "status": "ok", "config": dict(self.config)}


def _sample_obs(heading: float = 0.0) -> dict:
    return {
        "dt": 0.1,
        "robot": {
            "position": np.array([1.0, 1.0], dtype=float),
            "velocity": np.array([0.0, 0.0], dtype=float),
            "heading": np.array([heading], dtype=float),
            "radius": np.array([0.3], dtype=float),
        },
        "goal": {"current": np.array([4.0, 1.0], dtype=float)},
        "pedestrians": {
            "positions": np.array([[2.0, 2.0]], dtype=float),
            "velocities": np.array([[0.0, 0.0]], dtype=float),
            "radius": np.array([0.35], dtype=float),
        },
    }


def test_build_policy_ppo_accepts_unicycle_action(monkeypatch):
    """Ensure PPO map policy accepts native unicycle outputs."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, meta = map_runner._build_policy(
        "ppo",
        {"test_action": {"v": 0.7, "omega": -0.2}},
    )

    action_v, action_w = policy(_sample_obs())
    assert action_v == pytest.approx(0.7)
    assert action_w == pytest.approx(-0.2)
    assert meta["algorithm"] == "ppo"
    assert meta["status"] == "ok"
    assert callable(getattr(policy, "_planner_close", None))


def test_build_policy_ppo_converts_velocity_to_unicycle(monkeypatch):
    """Ensure velocity-vector PPO outputs are converted to unicycle commands."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, _ = map_runner._build_policy(
        "ppo",
        {
            "test_action": {"vx": 1.0, "vy": 0.0},
            "v_max": 0.8,
            "omega_max": 0.5,
        },
    )

    action_v, action_w = policy(_sample_obs(heading=np.pi / 2))
    assert action_v == pytest.approx(0.8)
    assert action_w == pytest.approx(-0.5)


def test_build_policy_ppo_rejects_unknown_action_payload(monkeypatch):
    """Reject malformed PPO action payloads lacking known action keys."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, _ = map_runner._build_policy(
        "ppo",
        {"test_action": {"foo": 1.0}},
    )

    with pytest.raises(ValueError, match="Unsupported PPO action payload"):
        policy(_sample_obs())
