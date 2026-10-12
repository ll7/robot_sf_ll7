"""Protect actor/value identity and companion normalization during PPO warm starts."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_sf.training.released_warm_start import load_warm_start


def test_loaded_actions_and_value_weights_equal_checkpoint(tmp_path):
    """An optimizer update must not be required to recover the source actor/critic."""
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    source = PPO("MlpPolicy", env, seed=100001, n_steps=8, batch_size=8, device="cpu")
    source.num_timesteps = 10000
    source._n_updates = 12
    path = tmp_path / "model.zip"
    source.save(path)
    fixed = np.array([0.6, 0.8, 0.25], dtype=np.float32)
    checkpoint_policy = PPO.load(path, device="cpu")
    expected, _ = checkpoint_policy.predict(fixed, deterministic=True)
    loaded, _ = load_warm_start(path, env=env, init_seed=100002)
    actual, _ = loaded.predict(fixed, deterministic=True)
    np.testing.assert_array_equal(actual, expected)
    for key, tensor in source.policy.state_dict().items():
        assert torch.equal(tensor, loaded.policy.state_dict()[key]), key
    assert loaded.num_timesteps == loaded._n_updates == 0
    assert not loaded.policy.optimizer.state
    env.close()


def test_normalization_companion_restored_and_affects_fixed_actions(tmp_path):
    """Nontrivial observation moments survive loading, independently of the ZIP."""
    env = VecNormalize(DummyVecEnv([lambda: gym.make("Pendulum-v1")]), norm_reward=False)
    env.obs_rms.mean[:] = [1.0, -2.0, 0.5]
    env.obs_rms.var[:] = [4.0, 9.0, 0.25]
    env.obs_rms.count = 17
    source = PPO("MlpPolicy", env, seed=100001, n_steps=8, batch_size=8, device="cpu")
    path, stats = tmp_path / "model.zip", tmp_path / "vecnormalize.pkl"
    source.save(path)
    env.save(stats)
    loaded, restored = load_warm_start(
        path, env=DummyVecEnv([lambda: gym.make("Pendulum-v1")]), normalization_path=stats
    )
    fixed = np.array([[0.6, 0.8, 0.25]], dtype=np.float32)
    np.testing.assert_array_equal(env.normalize_obs(fixed), restored.normalize_obs(fixed))
    checkpoint_policy = PPO.load(path, device="cpu")
    expected, _ = checkpoint_policy.predict(env.normalize_obs(fixed), deterministic=True)
    actual, _ = loaded.predict(restored.normalize_obs(fixed), deterministic=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored.obs_rms.mean, [1.0, -2.0, 0.5])
    assert restored.obs_rms.count == 17
    with pytest.raises(ValueError, match="requires an environment"):
        load_warm_start(path, normalization_path=stats)
    env.close()
    restored.close()


def test_action_bound_migration_is_explicit_and_preserves_raw_actor(tmp_path):
    """New clipping bounds require opt-in; they never reinterpret action coordinates."""
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    source = PPO("MlpPolicy", env, seed=100001, n_steps=8, batch_size=8, device="cpu")
    with torch.no_grad():
        source.policy.action_net.weight.zero_()
        source.policy.action_net.bias.fill_(1.5)
    path = tmp_path / "model.zip"
    source.save(path)
    target = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    migrated, _ = load_warm_start(path, target_action_space=target)
    fixed = np.array([0.6, 0.8, 0.25], dtype=np.float32)
    assert source.predict(fixed, deterministic=True)[0].item() == 1.5
    assert migrated.predict(fixed, deterministic=True)[0].item() == 1.0
    tensor, _ = migrated.policy.obs_to_tensor(fixed)
    with torch.no_grad():
        assert migrated.policy.get_distribution(tensor).mode().item() == 1.5
    for key, value in source.policy.state_dict().items():
        assert torch.equal(value, migrated.policy.state_dict()[key])
    with pytest.raises(ValueError, match="matching Box shape/dtype"):
        load_warm_start(
            path, target_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        )
    env.close()
