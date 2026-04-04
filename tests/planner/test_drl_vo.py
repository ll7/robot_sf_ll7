"""Tests for the DRL-VO baseline adapter and benchmark registration."""

import numpy as np
import pytest

from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.drl_vo import DrlVoPlanner, DrlVoPlannerConfig
from robot_sf.baselines.social_force import Observation
from robot_sf.benchmark.runner import _create_robot_policy


def test_drl_vo_is_registered_in_baselines() -> None:
    """DRL-VO should be discoverable through the baseline registry."""
    available = list_baselines()
    assert "drl_vo" in available
    planner_cls = get_baseline("drl_vo")
    assert planner_cls is DrlVoPlanner


def test_drl_vo_fallback_to_goal_action() -> None:
    """The DRL-VO adapter should fall back to goal-seeking when no model is available."""
    planner = DrlVoPlanner(DrlVoPlannerConfig(fallback_to_goal=True), seed=0)
    obs = Observation(
        dt=0.1,
        robot={"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0], "radius": 0.3},
        agents=[],
    )
    action = planner.step(obs)
    assert isinstance(action, dict)
    assert set(action.keys()) == {"vx", "vy"}
    assert action["vx"] > 0.0
    assert abs(action["vy"]) < 1e-6


def test_drl_vo_runner_integration_policy() -> None:
    """The benchmark runner should be able to instantiate and execute a DRL-VO policy."""
    policy_fn, metadata = _create_robot_policy("drl_vo", None, seed=1)
    assert callable(policy_fn)
    assert metadata["algorithm"] == "drl_vo"

    velocity = policy_fn(
        np.array([0.0, 0.0], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        np.array([2.0, 0.0], dtype=float),
        np.zeros((0, 2), dtype=float),
        0.1,
    )
    assert isinstance(velocity, np.ndarray)
    assert velocity.shape == (2,)
    assert velocity[0] >= 0.0


def test_drl_vo_missing_model_path_falls_back(tmp_path) -> None:
    """When the DRL-VO checkpoint is missing, the planner should fall back to goal motion."""
    planner = DrlVoPlanner(
        {
            "model_path": str(tmp_path / "missing_model.pt"),
            "fallback_to_goal": True,
        },
        seed=123,
    )
    obs = Observation(
        dt=0.1,
        robot={"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [0.2, 0.0], "radius": 0.3},
        agents=[],
    )
    action = planner.step(obs)
    assert action["vx"] > 0.0
    assert action["vy"] == pytest.approx(0.0)


def test_drl_vo_allows_benchmark_opt_in_flag() -> None:
    """Benchmark configs may include allow_testing_algorithms without breaking DRL-VO init."""
    planner = DrlVoPlanner(
        {
            "allow_testing_algorithms": True,
            "fallback_to_goal": True,
        },
        seed=42,
    )
    assert planner.config.fallback_to_goal is True
    assert planner.config.action_space == "velocity"
