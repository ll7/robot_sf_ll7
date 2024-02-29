"""
Generate a random policy to test the environment
"""
from robot_sf.robot_env import RobotEnv


def benchmark():
    total_steps = 20000
    env = RobotEnv(debug=True)
    obs = env.reset()
    env.render()

    print('start of simulation')

    episode = 0
    ep_rewards = 0
    for step in range(total_steps):
        rand_action = env.action_space.sample()
        obs, reward, done, _ = env.step(rand_action)
        env.render()
        ep_rewards += reward

        if done:
            episode += 1
            print(f'end of episode {episode}, total rewards {ep_rewards}')
            ep_rewards = 0
            obs = env.reset()
    env.exit()

    print('end of simulation')


if __name__ == "__main__":
    benchmark()
