from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from robot_sf.robot_env import RobotEnv


def training():
    env = make_vec_env(lambda: RobotEnv(), n_envs=32, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, tensorboard_log="./logs/ppo_logs/", n_steps=2048)
    save_model = CheckpointCallback(1_000_000, "./model/backup", "ppo_model")
    model.learn(total_timesteps=50_000_000, progress_bar=True, callback=save_model)
    model.save("./model/ppo_model")


if __name__ == '__main__':
    training()
