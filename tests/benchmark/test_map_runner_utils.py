"""Tests for map_runner helper utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner import (
    _build_policy,
    _goal_policy,
    _parse_algo_config,
    _select_seeds,
    _stack_ped_positions,
    _suite_key,
    _validate_behavior_sanity,
    _vel_and_acc,
)


def test_parse_algo_config_validates_yaml(tmp_path: Path) -> None:
    """Ensure YAML config parsing handles missing, valid, and invalid files."""
    assert _parse_algo_config(None) == {}
    with pytest.raises(FileNotFoundError):
        _parse_algo_config(str(tmp_path / "missing.yaml"))

    cfg_path = tmp_path / "algo.yaml"
    cfg_path.write_text("max_speed: 2.0\n", encoding="utf-8")
    assert _parse_algo_config(str(cfg_path))["max_speed"] == 2.0

    list_path = tmp_path / "algo_list.yaml"
    list_path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(TypeError):
        _parse_algo_config(str(list_path))


def test_goal_policy_and_build_policy() -> None:
    """Validate goal policy behavior and metadata wiring."""
    obs_at_goal = {
        "robot": {"position": [1.0, 1.0], "heading": [0.0]},
        "goal": {"current": [1.0, 1.0]},
    }
    assert _goal_policy(obs_at_goal) == (0.0, 0.0)

    policy, meta = _build_policy("goal", {"max_speed": 1.5})
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [1.0, 0.0]},
    }
    linear, angular = policy(obs)
    assert meta["status"] == "ok"
    assert linear > 0.0
    assert abs(angular) <= 1.0


def test_suite_seed_selection_and_behavior_sanity() -> None:
    """Check suite key selection and behavior sanity validation."""
    assert _suite_key(Path("classic_interactions.yaml")) == "classic_interactions"
    assert _suite_key(Path("francis2023.yaml")) == "francis2023"
    assert _suite_key(Path("other.yaml")) == "default"

    suite_seeds = {"default": [7, 8], "classic_interactions": [1, 2]}
    assert _select_seeds({"seeds": [3, 4]}, suite_seeds=suite_seeds, suite_key="default") == [
        3,
        4,
    ]
    assert _select_seeds({}, suite_seeds=suite_seeds, suite_key="classic_interactions") == [1, 2]

    errors = _validate_behavior_sanity({"metadata": {"behavior": "wait"}, "single_pedestrians": []})
    assert errors


def test_velocity_and_ped_stack_helpers() -> None:
    """Ensure velocity/acceleration and trajectory stacking behave for small inputs."""
    positions = np.array([[0.0, 0.0]])
    vel, acc = _vel_and_acc(positions, dt=0.1)
    assert np.allclose(vel, 0.0)
    assert np.allclose(acc, 0.0)

    traj = [np.array([[0.0, 0.0]]), np.array([[1.0, 1.0], [2.0, 2.0]])]
    stacked = _stack_ped_positions(traj)
    assert stacked.shape == (2, 2, 2)
