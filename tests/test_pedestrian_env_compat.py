"""Compatibility tests for the pedestrian environment."""

from __future__ import annotations

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.gym_env._stub_robot_model import StubRobotModel
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.pedestrian_env import PedestrianEnv


def test_pedestrian_env_uses_stub_robot_model() -> None:
    """Ensure PedestrianEnv injects a stub model when robot_model is None."""
    env = PedestrianEnv(robot_model=None)
    try:
        assert isinstance(env.robot_model, StubRobotModel)
    finally:
        env.exit()


def test_make_pedestrian_env_uses_stub_robot_model() -> None:
    """Ensure factory injects a stub model when robot_model is None."""
    env = make_pedestrian_env(robot_model=None)
    try:
        assert isinstance(env.robot_model, StubRobotModel)
    finally:
        env.exit()


def test_pedestrian_env_records_and_resets(tmp_path, monkeypatch) -> None:
    """Verify recording and reset behavior matches legacy PedestrianEnv."""
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
