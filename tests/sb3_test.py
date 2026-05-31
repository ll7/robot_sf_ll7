"""TODO docstring. Document this module."""

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3", reason="StableBaselines3 not installed")

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.robot_env import RobotEnv


def _make_robot_env() -> RobotEnv:
    """Build a default RobotEnv instance for SB3 vectorized wrappers."""

    return RobotEnv()


def test_can_load_model_snapshot(tmp_path: Path) -> None:
    """Train, save, and reload a PPO model snapshot to ensure compatibility."""
    model_path = tmp_path / "ppo_model"
    model_file = model_path.with_suffix(".zip")

    vec_env = make_vec_env(_make_robot_env, n_envs=1)
    policy_kwargs = {"features_extractor_class": DynamicsExtractor}
    inf_env = None
    try:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            gamma=0.95,
            learning_rate=3e-4,
            verbose=0,
        )
        model.save(model_path)
        assert model_file.exists()

        inf_env = RobotEnv()
        model2 = PPO.load(model_path, env=inf_env)

        obs, _info = inf_env.reset()
        action, _ = model2.predict(obs, deterministic=True)

        assert action.shape == inf_env.action_space.shape
    finally:
        vec_env.close()
        if inf_env is not None:
            inf_env.close()
