"""Regression tests for SACADRL adapter robustness on malformed pedestrian payloads."""

from __future__ import annotations

import math

from robot_sf.planner.socnav import SACADRLPlannerAdapter, SocNavPlannerConfig


def _base_observation() -> dict:
    """Create a minimal SocNav-style observation for planner tests."""
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "radius": [0.3],
        },
        "goal": {
            "current": [2.0, 0.0],
        },
        "pedestrians": {},
        "sim": {"timestep": [0.1]},
    }


def test_sacadrl_heuristic_handles_empty_pedestrian_payload(monkeypatch) -> None:
    """Planner should not crash when pedestrian arrays/count are missing or empty."""
    adapter = SACADRLPlannerAdapter(config=SocNavPlannerConfig(), allow_fallback=True)
    monkeypatch.setattr(adapter, "_ensure_model", lambda: None)

    obs = _base_observation()
    obs["pedestrians"] = {
        "positions": [],
        "velocities": [],
        "count": [],
        "radius": [],
    }

    linear, angular = adapter.plan(obs)
    assert math.isfinite(linear)
    assert math.isfinite(angular)


def test_sacadrl_heuristic_handles_malformed_pedestrian_arrays(monkeypatch) -> None:
    """Planner should sanitize malformed pedestrian vectors instead of indexing out of bounds."""
    adapter = SACADRLPlannerAdapter(config=SocNavPlannerConfig(), allow_fallback=True)
    monkeypatch.setattr(adapter, "_ensure_model", lambda: None)

    obs = _base_observation()
    obs["pedestrians"] = {
        "positions": [1.0, 2.0, 3.0],  # odd-length flat array
        "velocities": [0.1, 0.0, 0.2],
        "count": [5],
        "radius": [0.35],
    }

    linear, angular = adapter.plan(obs)
    assert math.isfinite(linear)
    assert math.isfinite(angular)
