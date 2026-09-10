"""Legacy interactive debug runner for a locally trained PPO checkpoint.

Diagnostic only: loads ``./model/ppo_model`` through Stable-Baselines3 into a
rendered :class:`~robot_sf.gym_env.robot_env.RobotEnv` and steps up to 10,000
frames using the legacy four-value ``env.step`` contract. Requires a local
checkpoint file and a display; it produces no benchmark or evidence artifacts.
"""

from stable_baselines3 import PPO

from robot_sf.gym_env.robot_env import RobotEnv


def training():
    """Run the legacy 10,000-step PPO checkpoint rollout with rendering.

    Loads the local checkpoint, renders every step, and resets the environment
    when an episode terminates. Missing checkpoints, display access, or
    mismatched observation shapes surface as their original errors; the
    environment is closed after the loop completes.
    """
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
    env.close()


if __name__ == "__main__":
    training()
