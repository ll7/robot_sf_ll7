"""Compatibility tests for the pedestrian environment."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.gym_env._stub_robot_model import StubRobotModel
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.gym_env.pedestrian_env_refactored import RefactoredPedestrianEnv


def test_pedestrian_env_uses_stub_robot_model() -> None:
    """Inject a stub model when missing to preserve legacy default behavior."""
    env = PedestrianEnv(robot_model=None)
    try:
        assert isinstance(env.robot_model, StubRobotModel)
    finally:
        env.exit()


def test_pedestrian_env_handles_mismatched_model_action_space() -> None:
    """Fallback to null action to avoid crashes when model outputs mismatch env."""

    class _BadModel:
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        def predict(self, _obs, **_kwargs):
            return np.zeros(3, dtype=np.float32), None

    env = PedestrianEnv(robot_model=_BadModel())
    try:
        assert isinstance(env, RefactoredPedestrianEnv)
        assert env._robot_action_space_valid is False
        env.reset()
        action = env.action_space.sample()
        env.step(action)
    finally:
        env.exit()


def test_make_pedestrian_env_uses_stub_robot_model() -> None:
    """Factory should preserve stub model fallback for backwards compatibility."""
    env = make_pedestrian_env(robot_model=None)
    try:
        assert isinstance(env.robot_model, StubRobotModel)
    finally:
        env.exit()


def test_pedestrian_env_records_and_resets(tmp_path, monkeypatch) -> None:
    """Ensure recording/reset semantics stay stable for downstream tooling."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    env = PedestrianEnv(robot_model=None, recording_enabled=True)
    try:
        _obs, info = env.reset()
        assert "info" in info

        action = env.action_space.sample()
        _obs, _reward, _terminated, _truncated, step_info = env.step(action)
        assert {"step", "meta", "collision", "success", "is_success"} <= step_info.keys()
        assert env.recorded_states

        _obs, _info = env.reset()
        recordings_dir = get_artifact_category_path("recordings")
        assert list(recordings_dir.glob("*.pkl"))
        assert env.recorded_states == []
    finally:
        env.exit()
