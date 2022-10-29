from stable_baselines3 import A2C
from robot_sf.robot_env import RobotEnv


def training():
    env = RobotEnv()
    model = A2C("MlpPolicy", env)
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    model.save("dqn_model")


if __name__ == '__main__':
    training()
