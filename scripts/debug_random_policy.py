"""
Generate a random policy to test the environment
"""

from loguru import logger

from robot_sf.gym_env.robot_env import RobotEnv


def benchmark():
    """Benchmark.

    Returns:
        Any: Auto-generated placeholder description.
    """
    total_steps = 20000
    env = RobotEnv(debug=True)
    env.reset()
    env.render()

    print("start of simulation")

    episode = 0
    ep_rewards = 0
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, reward, done, _, _ = env.step(rand_action)
        env.render()
        ep_rewards += reward

        if done:
            episode += 1
            logger.info(f"end of episode {episode}, total rewards {ep_rewards}")
            ep_rewards = 0
            _ = env.reset()
    env.exit()

    print("end of simulation")


if __name__ == "__main__":
    benchmark()
