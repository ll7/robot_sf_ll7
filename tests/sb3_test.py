"""TODO docstring. Document this module."""

import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.robot_env import RobotEnv


def test_can_load_model_snapshot():
    """TODO docstring. Document this function."""
    MODEL_PATH = "./temp/ppo_model"
    MODEL_FILE = f"{MODEL_PATH}.zip"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if os.path.exists(MODEL_FILE) and os.path.isfile(MODEL_FILE):
        os.remove(MODEL_FILE)

    vec_env = make_vec_env(lambda: RobotEnv(), n_envs=1)
    policy_kwargs = {"features_extractor_class": DynamicsExtractor}
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
    model.save(MODEL_PATH)
    assert os.path.exists(MODEL_FILE)

    inf_env = RobotEnv()
    model2 = PPO.load(MODEL_PATH, env=inf_env)

    obs, _info = inf_env.reset()
    action, _ = model2.predict(obs, deterministic=True)

    assert action.shape == inf_env.action_space.shape

    if os.path.exists(MODEL_FILE) and os.path.isfile(MODEL_FILE):
        os.remove(MODEL_FILE)
        os.rmdir(os.path.dirname(MODEL_PATH))
