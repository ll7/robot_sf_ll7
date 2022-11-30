from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from robot_sf.robot_env import RobotEnv


def training():
    env = RobotEnv(debug=True)
    model = A2C.load("./model/dqn_model", env=env)

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()

        if done:
            obs = env.reset()
            env.render()


if __name__ == '__main__':
    training()
