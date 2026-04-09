"""Unit tests for SAC support in map_runner policy bridge."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark import map_runner


class _DummySACPlanner:
    """Test double for SAC planner integration in map runner."""

    def __init__(self, config, *, seed=None):
        self.config = dict(config)
        self.seed = seed
        self.closed = False
        self.last_obs = None

    def step(self, obs):
        self.last_obs = obs
        return self.config.get("test_action", {"v": 0.5, "omega": 0.0})

    def close(self):
        self.closed = True

    def get_metadata(self):
        return {"algorithm": "sac", "status": "ok", "config": dict(self.config)}


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


def test_build_policy_sac_accepts_unicycle_action(monkeypatch):
    """SAC map policy should accept native unicycle outputs."""
    monkeypatch.setattr(map_runner, "SACPlanner", _DummySACPlanner)
    policy, meta = map_runner._build_policy(
        "sac",
        {"test_action": {"v": 0.6, "omega": -0.1}},
    )

    action_v, action_w = policy(_sample_obs())
    assert action_v == pytest.approx(0.6)
    assert action_w == pytest.approx(-0.1)
    assert meta["algorithm"] == "sac"
    assert callable(getattr(policy, "_planner_close", None))


def test_build_policy_sac_dict_mode_passes_raw_observation(monkeypatch):
    """Benchmark SAC checkpoints should receive flattened dict observations unchanged."""
    dummy = _DummySACPlanner({"obs_mode": "dict", "test_action": {"v": 0.3, "omega": 0.0}})
    monkeypatch.setattr(map_runner, "SACPlanner", lambda *_args, **_kwargs: dummy)
    policy, _ = map_runner._build_policy(
        "sac",
        {"obs_mode": "dict", "test_action": {"v": 0.3, "omega": 0.0}},
    )

    obs = _sample_obs()
    obs["robot_position"] = np.array([1.0, 1.0], dtype=float)
    obs["goal_current"] = np.array([4.0, 1.0], dtype=float)
    obs["pedestrians_positions"] = np.array([[2.0, 2.0]], dtype=float)
    policy(obs)
    assert dummy.last_obs is obs
