"""Tests for SocNavBench-inspired planner adapters."""

import numpy as np

from robot_sf.planner.socnav import SamplingPlannerAdapter, SocNavPlannerConfig, SocNavPlannerPolicy


def _make_obs(goal=(5.0, 0.0), heading=0.0):
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.array(goal, dtype=np.float32),
            "next": np.array([0.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.zeros((1, 2), dtype=np.float32),
            "radius": np.array([0.4], dtype=np.float32),
            "count": np.array([0.0], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def test_sampling_adapter_moves_toward_goal():
    """Adapter moves forward when aligned with goal."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0, angular_gain=2.0))
    obs = _make_obs(goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v > 0.0
    assert abs(w) < 1e-6  # aligned with heading


def test_sampling_adapter_stops_within_tolerance():
    """Adapter stops when within goal tolerance."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(goal_tolerance=0.5))
    obs = _make_obs(goal=(0.2, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v == 0.0
    assert w == 0.0


def test_sampling_adapter_turns_toward_goal():
    """Adapter turns left when goal is on the left."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0, angular_gain=1.0))
    obs = _make_obs(goal=(0.0, 5.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert w > 0.0  # needs to turn left
    assert v >= 0.0


def test_policy_wrapper_calls_adapter():
    """Policy delegates to the underlying adapter."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0))
    policy = SocNavPlannerPolicy(adapter)
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    v, _w = policy.act(obs)
    assert v >= 0.0
