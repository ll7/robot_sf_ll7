"""Hand-derived action bounds for the actual 0.0.8 SocNav release bindings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.planner.socnav import SocNavPlannerConfig
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter

ROOT = Path(__file__).parents[2]
TEMPLATE = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)


def _release_config(arm: str) -> SocNavPlannerConfig:
    campaign = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    row = next(row for row in campaign["planners"] if row["key"] == arm)
    algo_path = row.get("algo_config")
    config = yaml.safe_load((ROOT / algo_path).read_text(encoding="utf-8")) if algo_path else {}
    return SocNavPlannerConfig(**config)


def test_orca_release_command_clips_three_metre_preference_to_drive_bound(monkeypatch) -> None:
    """A 3 m/s forward world velocity must be emitted as 2 m/s, not 3 m/s."""
    adapter = ORCAPlannerAdapter(config=_release_config("orca"), allow_fallback=True)
    monkeypatch.setattr(adapter, "plan_velocity_world", lambda _obs: np.array([3.0, 0.0]))
    observation = {
        "robot": {"position": np.array([0.0, 0.0]), "heading": np.array([0.0])},
        "goal": {"current": np.array([5.0, 0.0])},
        "pedestrians": {"positions": np.zeros((0, 2)), "velocities": np.zeros((0, 2))},
    }
    linear, angular = adapter.plan(observation)
    assert linear == pytest.approx(2.0)
    assert angular == pytest.approx(0.0)


def test_sacadrl_release_command_clips_speed_and_turn_to_drive_bounds(monkeypatch) -> None:
    """A 3 m/s, pi/6 per 0.1 s model action is bounded to (2, 1)."""
    adapter = SACADRLPlannerAdapter(config=_release_config("sacadrl"), allow_fallback=True)
    model = SimpleNamespace(
        actions=np.array([[1.0, np.pi / 6]]),
        predict=lambda _obs: np.array([[1.0]]),
    )
    monkeypatch.setattr(adapter, "_ensure_model", lambda: model)
    monkeypatch.setattr(adapter, "_build_network_input", lambda _obs: (np.zeros((1, 5)), 3.0, 5.0))
    observation = {
        "robot": {"position": np.zeros(2)},
        "goal": {"current": np.array([5.0, 0.0])},
        "sim": {"timestep": np.array([0.1])},
    }
    linear, angular = adapter.plan(observation)
    assert linear == pytest.approx(2.0)
    assert angular == pytest.approx(1.0)
