#!/usr/bin/env python3
"""
Integration test for the image-based observation system.
This is a simple validation script to ensure the image system works end-to-end.
"""

from __future__ import annotations

import pytest

from robot_sf.gym_env.environment_factory import make_image_robot_env, make_robot_env
from robot_sf.gym_env.unified_config import ImageRobotConfig, RobotSimulationConfig
from robot_sf.sensor.image_sensor import ImageSensorSettings


def test_image_system_integration(monkeypatch: pytest.MonkeyPatch):
    """Test that the image observation system works end-to-end."""
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

    # Test 1: Environment with image observations disabled
    config_disabled = RobotSimulationConfig()
    env_disabled = make_robot_env(config=config_disabled, debug=False)
    obs, _info = env_disabled.reset()
    assert "image" not in obs, "Image should not be in observation space when disabled"
    env_disabled.exit()

    # Test 2: Environment with image observations enabled
    image_config = ImageSensorSettings(
        width=64,
        height=64,
        channels=3,
        normalize=True,
        grayscale=False,
    )
    config_enabled = ImageRobotConfig(image_config=image_config)
    env_enabled = make_image_robot_env(config=config_enabled, debug=True)

    obs, _info = env_enabled.reset()
    assert "image" in obs, "Image should be in observation space when enabled"

    image_obs = obs["image"]
    assert image_obs.shape == (64, 64, 3), f"Expected (64, 64, 3), got {image_obs.shape}"

    action = env_enabled.action_space.sample()
    obs, _reward, _done, _truncated, _info = env_enabled.step(action)
    assert "image" in obs, "Image should be in observation after step"

    env_enabled.exit()
