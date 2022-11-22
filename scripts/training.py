from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from robot_sf.robot_env import RobotEnv


def training():
    env = make_vec_env(lambda: RobotEnv(), n_envs=50)
    model = A2C("MlpPolicy", env)
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    model.save("./model/dqn_model")


if __name__ == '__main__':
    training()
