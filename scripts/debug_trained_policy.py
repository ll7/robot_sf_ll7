"""TODO docstring. Document this module."""

from stable_baselines3 import PPO

from robot_sf.gym_env.robot_env import RobotEnv


def training():
    """TODO docstring. Document this function."""
    env = RobotEnv(debug=True)
    model = PPO.load("./model/ppo_model", env=env)

    obs = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()

        if done:
            obs = env.reset()
            env.render()
    env.exit()


if __name__ == "__main__":
    training()
