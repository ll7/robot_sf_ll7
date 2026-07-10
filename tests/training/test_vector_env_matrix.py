"""Vectorized training environment support matrix tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.threaded_vec_env import ThreadedVecEnv
from scripts.training import train_ppo

_SCENARIO_PATH = Path("configs/scenarios/single/planner_sanity_simple.yaml").resolve()


def _ppo_training_env_factory(seed: int):
    """Build a picklable PPO training env factory for vector-env smoke tests."""
    scenario = load_scenarios(_SCENARIO_PATH)[0]
    return train_ppo._make_training_env(
        seed,
        scenario=scenario,
        scenario_definitions=None,
        scenario_path=_SCENARIO_PATH,
        exclude_scenarios=(),
        suite_name="vector_env_matrix",
        algorithm_name="ppo_spawn_smoke",
        env_overrides={"sim_config.sim_time_in_secs": 0.2},
        env_factory_kwargs={},
        scenario_sampling={},
    )


def _assert_vec_env_reset_step_close(vec_env) -> None:
    """Run the shared reset/step/close smoke for a Stable-Baselines VecEnv."""
    try:
        obs = vec_env.reset()
        assert obs is not None

        actions = np.zeros(
            (vec_env.num_envs, *vec_env.action_space.shape),
            dtype=vec_env.action_space.dtype,
        )
        step_obs, rewards, dones, infos = vec_env.step(actions)

        assert step_obs is not None
        assert rewards.shape == (vec_env.num_envs,)
        assert dones.shape == (vec_env.num_envs,)
        assert len(infos) == vec_env.num_envs
    finally:
        vec_env.close()


def test_ppo_training_factory_supports_dummy_vec_env() -> None:
    """Primary PPO entry point should reset/step/close under DummyVecEnv."""
    env_fns = [_ppo_training_env_factory(100)]

    _assert_vec_env_reset_step_close(DummyVecEnv(env_fns))


def test_ppo_training_factory_supports_subproc_spawn_vec_env() -> None:
    """Primary PPO entry point should run actual spawn workers, not only pickle round-trips."""
    env_fns = [_ppo_training_env_factory(200), _ppo_training_env_factory(201)]

    _assert_vec_env_reset_step_close(SubprocVecEnv(env_fns, start_method="spawn"))


def test_ppo_training_factory_supports_threaded_vec_env() -> None:
    """Primary PPO environment factories should support in-process threaded rollouts."""
    env_fns = [_ppo_training_env_factory(300), _ppo_training_env_factory(301)]

    _assert_vec_env_reset_step_close(ThreadedVecEnv(env_fns))


def test_ppo_training_factory_supports_threaded_lidar_batch_vec_env() -> None:
    """Primary PPO rollouts should execute the opt-in coordinated LiDAR batch path."""
    env_fns = [_ppo_training_env_factory(400), _ppo_training_env_factory(401)]

    _assert_vec_env_reset_step_close(ThreadedVecEnv(env_fns, batch_lidar=True))
