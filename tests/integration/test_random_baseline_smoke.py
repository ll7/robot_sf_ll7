"""TODO docstring. Document this module."""

from __future__ import annotations

from robot_sf.baselines import get_baseline


def _obs():
    """TODO docstring. Document this function."""
    return {
        "dt": 0.1,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [5.0, 0.0],
            "radius": 0.35,
        },
        "agents": [],
        "obstacles": [],
    }


def test_random_velocity_mode_step():
    """TODO docstring. Document this function."""
    Random = get_baseline("random")
    policy = Random({"mode": "velocity", "v_max": 1.5}, seed=123)
    act = policy.step(_obs())
    assert set(act.keys()) == {"vx", "vy"}
    assert abs(act["vx"]) <= 1.5 + 1e-6
    assert abs(act["vy"]) <= 1.5 + 1e-6


def test_random_unicycle_mode_step():
    """TODO docstring. Document this function."""
    Random = get_baseline("random")
    policy = Random({"mode": "unicycle", "v_max": 1.0, "omega_max": 2.0}, seed=42)
    act = policy.step(_obs())
    assert set(act.keys()) == {"v", "omega"}
    assert 0.0 <= act["v"] <= 1.0 + 1e-6
    assert -2.0 - 1e-6 <= act["omega"] <= 2.0 + 1e-6
