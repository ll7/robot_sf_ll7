"""Module training_a2c auto-generated docstring."""

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.gym_env.robot_env import RobotEnv


def training():
    """Training.

    Returns:
        Any: Auto-generated placeholder description.
    """
    env = make_vec_env(lambda: RobotEnv(), n_envs=50, vec_env_cls=SubprocVecEnv)
    model = A2C("MlpPolicy", env, tensorboard_log="./logs/a2c_logs/")
    model.learn(total_timesteps=50_000_000, progress_bar=True)
    model.save("./model/a2c_model")


if __name__ == "__main__":
    training()
