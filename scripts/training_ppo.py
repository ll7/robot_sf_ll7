from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from robot_sf.robot_env import RobotEnv


def training():
    env = make_vec_env(lambda: RobotEnv(difficulty=2), n_envs=50, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, tensorboard_log="./logs/ppo_logs/")
    model.learn(total_timesteps=5_000_000, progress_bar=True)
    model.save("./model/ppo_model")


if __name__ == '__main__':
    training()
