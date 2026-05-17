"""Compatibility tests for the pedestrian environment."""

from __future__ import annotations

import inspect
from functools import partial

import numpy as np
from gymnasium import spaces

import robot_sf.gym_env.pedestrian_env as pedestrian_env_module
from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.gym_env._stub_robot_model import StubRobotModel
from robot_sf.gym_env.env_config import PedEnvSettings as LegacyPedEnvSettings
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.pedestrian_env import PedestrianEnv, _reward_function_name
from robot_sf.gym_env.reward import simple_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig


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
        """Model stub with an incompatible action-space shape."""

        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        def predict(self, _obs, **_kwargs):
            """Return an action vector that does not match the environment contract."""
            return np.zeros(3, dtype=np.float32), None

    env = PedestrianEnv(robot_model=_BadModel())
    try:
        assert isinstance(env, PedestrianEnv)
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
        assert isinstance(env, PedestrianEnv)
        assert isinstance(env.robot_model, StubRobotModel)
    finally:
        env.exit()


def test_pedestrian_env_does_not_mutate_input_config_for_deprecated_force_flag() -> None:
    """Deprecated force override should be isolated to the constructed environment."""
    config = PedestrianSimulationConfig()

    env = PedestrianEnv(
        env_config=config,
        robot_model=None,
        peds_have_obstacle_forces=False,
    )
    try:
        assert config.peds_have_static_obstacle_forces is True
        assert env.config.peds_have_static_obstacle_forces is False
    finally:
        env.exit()


def test_pedestrian_env_adapts_legacy_config_to_unified_boundary() -> None:
    """Legacy PedEnvSettings should be explicitly adapted before base setup."""
    config = LegacyPedEnvSettings(spawn_near_robot=False)

    env = PedestrianEnv(env_config=config, robot_model=None)
    try:
        assert isinstance(env.config, PedestrianSimulationConfig)
        assert env.config is not config
        assert env.config.spawn_near_robot is False
    finally:
        env.exit()


def test_pedestrian_env_constructor_avoids_bare_any_config_cast() -> None:
    """PedestrianEnv should not hide legacy config typing with a bare Any cast."""
    source = inspect.getsource(pedestrian_env_module.PedestrianEnv.__init__)

    assert 'cast("Any", env_config)' not in source


def test_pedestrian_env_create_spaces_matches_base_signature() -> None:
    """PedestrianEnv._create_spaces should return action and observation spaces only."""
    env = PedestrianEnv(robot_model=None)
    try:
        created_spaces = env._create_spaces()

        assert len(created_spaces) == 2
        assert env.orig_obs_space is not None
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


def test_reward_function_name_handles_partial_wrapped_rewards() -> None:
    """Partial-wrapped reward functions should still log the underlying reward name."""
    reward_func = partial(simple_ped_reward, robot_coll_reward=1.0)

    assert _reward_function_name(reward_func) == "simple_ped_reward"
