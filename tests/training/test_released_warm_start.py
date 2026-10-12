"""Protect actor/value identity and companion normalization during PPO warm starts."""

import hashlib
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_sf.models import get_registry_entry
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


def test_prepared_seed_survives_canonical_resume_round_trip(tmp_path):
    """Saving/reloading keeps requested optimizer RNGs distinct and reproducible."""
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    source = PPO("MlpPolicy", env, seed=100000, n_steps=8, batch_size=8, device="cpu")
    source_path = tmp_path / "source.zip"
    source.save(source_path)
    fixed = np.array([0.6, 0.8, 0.25], dtype=np.float32)
    stochastic = []
    for seed in (100001, 100002):
        prepared, _ = load_warm_start(source_path, init_seed=seed)
        path = tmp_path / f"init{seed}.zip"
        prepared.save(path)
        resumed = PPO.load(path, env=env, device="cpu")
        assert resumed.seed == seed
        first = torch.rand(8)
        action, _ = resumed.predict(fixed, deterministic=False)
        again = PPO.load(path, env=env, device="cpu")
        assert again.seed == seed
        assert torch.equal(torch.rand(8), first)
        np.testing.assert_array_equal(again.predict(fixed, deterministic=False)[0], action)
        np.testing.assert_array_equal(
            resumed.predict(fixed, deterministic=True)[0],
            source.predict(fixed, deterministic=True)[0],
        )
        stochastic.append(action)
    assert not np.array_equal(stochastic[0], stochastic[1])
    env.close()


def test_b1002_release_actions_values_and_three_seed_round_trips(tmp_path):
    """Opt-in real-release parity uses finite fixed dict observations and no rollouts."""
    source_path = os.environ.get("ROBOT_SF_B1002_PARITY_CHECKPOINT")
    if not source_path:
        pytest.skip("set ROBOT_SF_B1002_PARITY_CHECKPOINT for the pinned release check")
    path = Path(source_path)
    entry = get_registry_entry("ppo_release_robot_b1002_last_20261001")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["github_release"]["sha256"]
    assert path.stat().st_size == entry["github_release"]["size_bytes"]
    assert entry["action_semantics"] == "velocity_delta"
    assert entry["predictive_foresight_enabled"] is False
    source = PPO.load(path, device="cpu")
    assert source.n_envs == 18
    assert source.policy_kwargs["features_extractor_kwargs"]["grid_channels"] == [32, 64, 64]
    np.testing.assert_array_equal(source.action_space.low, [-2, -1])
    np.testing.assert_array_equal(source.action_space.high, [2, 1])
    fixed = {
        key: np.zeros(space.shape, dtype=space.dtype)
        for key, space in source.observation_space.spaces.items()
    }
    fixed.update(
        goal_current=np.array([6.0, 4.0], dtype=np.float32),
        goal_next=np.array([8.0, 4.0], dtype=np.float32),
        map_size=np.array([32.0, 32.0], dtype=np.float32),
        robot_position=np.array([4.0, 4.0], dtype=np.float32),
        robot_radius=np.array([0.3], dtype=np.float32),
        occupancy_grid_meta_resolution=np.array([0.2], dtype=np.float32),
        occupancy_grid_meta_size=np.array([32.0, 32.0], dtype=np.float32),
        occupancy_grid_meta_channel_indices=np.array([0, 1, 2, -1], dtype=np.int32),
        occupancy_grid_meta_center_on_robot=np.array([1.0], dtype=np.float32),
        occupancy_grid_meta_use_ego_frame=np.array([1.0], dtype=np.float32),
    )
    fixed["occupancy_grid"][0, 75:85, 87:90] = 1.0
    fixed["occupancy_grid"][2] = fixed["occupancy_grid"][0]
    assert source.observation_space.contains(fixed)
    expected_action, _ = source.predict(fixed, deterministic=True)
    tensor, _ = source.policy.obs_to_tensor(fixed)
    with torch.no_grad():
        expected_value = source.policy.predict_values(tensor)
    for seed in (100001, 100002, 100003):
        prepared, _ = load_warm_start(path, init_seed=seed)
        prepared_path = tmp_path / f"init{seed}.zip"
        prepared.save(prepared_path)
        del prepared
        reloaded = PPO.load(prepared_path, device="cpu")
        assert reloaded.seed == seed
        assert reloaded.observation_space == source.observation_space
        assert reloaded.action_space == source.action_space
        assert reloaded.num_timesteps == reloaded._n_updates == 0
        assert not reloaded.policy.optimizer.state
        np.testing.assert_array_equal(
            reloaded.predict(fixed, deterministic=True)[0], expected_action
        )
        with torch.no_grad():
            assert torch.equal(reloaded.policy.predict_values(tensor), expected_value)
        for key, value in source.policy.state_dict().items():
            assert torch.equal(value, reloaded.policy.state_dict()[key]), key
        del reloaded
