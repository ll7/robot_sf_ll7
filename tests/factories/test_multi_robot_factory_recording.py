"""Tests for multi-robot factory recording parameter forwarding."""

from __future__ import annotations

from types import SimpleNamespace

from robot_sf.gym_env.environment_factory import EnvironmentFactory, make_multi_robot_env


def test_make_multi_robot_env_forwards_recording_args() -> None:
    """`make_multi_robot_env` should forward recording kwargs to the internal factory."""
    captured: dict[str, object] = {}

    def _fake_create_multi_robot_env(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(reset=lambda *a, **k: None, step=lambda *a, **k: None)

    original = EnvironmentFactory.create_multi_robot_env
    EnvironmentFactory.create_multi_robot_env = staticmethod(_fake_create_multi_robot_env)
    try:
        _env = make_multi_robot_env(
            num_robots=3,
            recording_enabled=True,
            record_video=True,
            video_path="/tmp/multi_robot.mp4",
            video_fps=15.0,
        )
    finally:
        EnvironmentFactory.create_multi_robot_env = original

    assert captured["num_robots"] == 3
    assert captured["recording_enabled"] is True
    assert captured["record_video"] is True
    assert captured["video_path"] == "/tmp/multi_robot.mp4"
    assert captured["video_fps"] == 15.0
