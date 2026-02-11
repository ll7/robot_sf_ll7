"""Unit tests for RLlib-oriented observation/action wrappers."""

from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.training.rllib_env_wrappers import (
    DEFAULT_FLATTEN_KEYS,
    DriveStateRaysFlattenWrapper,
    SymmetricActionRescaleWrapper,
    wrap_for_dreamerv3,
)


class _DummyDictEnv(Env):
    """Simple environment exposing Dict observations and Box actions."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        """Initialize deterministic spaces for wrapper assertions."""
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "drive_state": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "rays": spaces.Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._obs = {
            "drive_state": np.array([0.25, -0.5], dtype=np.float32),
            "rays": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        """Return one deterministic observation."""
        super().reset(seed=seed)
        return dict(self._obs), {}

    def step(self, action):  # type: ignore[override]
        """Echo deterministic observation to keep wrapper tests focused."""
        return dict(self._obs), 0.0, False, False, {"action_seen": np.asarray(action)}


def test_drive_state_rays_flatten_wrapper_respects_key_order():
    """Wrapper should concatenate keys in the explicit order."""
    env = _DummyDictEnv()
    wrapped = DriveStateRaysFlattenWrapper(env, keys=("rays", "drive_state"))
    obs, _ = wrapped.reset()

    assert wrapped.observation_space.shape == (5,)
    assert obs.shape == (5,)
    np.testing.assert_allclose(obs, np.array([0.1, 0.2, 0.3, 0.25, -0.5], dtype=np.float32))


def test_symmetric_action_rescale_wrapper_maps_minus1_plus1_to_env_bounds():
    """Rescaled actions should map exactly to the original action-space bounds."""
    env = _DummyDictEnv()
    wrapped = SymmetricActionRescaleWrapper(env)

    assert wrapped.action_space.shape == env.action_space.shape
    np.testing.assert_allclose(wrapped.action(np.array([-1.0, 1.0], dtype=np.float32)), [0.0, 2.0])
    np.testing.assert_allclose(wrapped.action(np.array([1.0, -1.0], dtype=np.float32)), [2.0, -2.0])


def test_wrap_for_dreamerv3_applies_expected_wrapper_stack():
    """Combined wrapper helper should return flattened observations and normalized actions."""
    env = _DummyDictEnv()
    wrapped = wrap_for_dreamerv3(
        env,
        flatten_observation=True,
        flatten_keys=DEFAULT_FLATTEN_KEYS,
        normalize_actions=True,
    )
    obs, _ = wrapped.reset()
    assert obs.shape == (5,)
    np.testing.assert_allclose(obs, np.array([0.25, -0.5, 0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(
        wrapped.action(np.array([0.0, 0.0], dtype=np.float32)),
        np.array([1.0, 0.0], dtype=np.float32),
    )


def test_wrap_for_dreamerv3_reset_and_step_match_observation_contract():
    """Dreamer wrapper stack should emit observations accepted by its declared space."""
    config = RobotSimulationConfig()
    config.use_image_obs = False
    config.include_grid_in_observation = False
    env = make_robot_env(config=config, debug=False, recording_enabled=False)
    wrapped = wrap_for_dreamerv3(
        env,
        flatten_observation=False,
        flatten_keys=DEFAULT_FLATTEN_KEYS,
        normalize_actions=True,
    )
    try:
        reset_obs, _ = wrapped.reset()
        assert wrapped.observation_space.contains(reset_obs)
        assert np.asarray(reset_obs["drive_state"]).dtype == np.float32
        assert np.asarray(reset_obs["rays"]).dtype == np.float32

        step_obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
        assert wrapped.observation_space.contains(step_obs)
        assert np.asarray(step_obs["drive_state"]).dtype == np.float32
        assert np.asarray(step_obs["rays"]).dtype == np.float32
    finally:
        wrapped.close()
