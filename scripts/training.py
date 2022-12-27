from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.env_util import make_vec_env
from robot_sf.robot_env import RobotEnv


def training():
    env = make_vec_env(lambda: RobotEnv(difficulty=2), n_envs=50)
    model = A2C("MlpPolicy", env, tensorboard_log="./logs/a2c_logs/")
    model.learn(total_timesteps=50_000_000, progress_bar=True)
    model.save("./model/a2c_model")


if __name__ == '__main__':
    training()
