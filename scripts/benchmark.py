# import time
from scalene import scalene_profiler
from robot_sf.robot_env import RobotEnv


def benchmark():
    total_steps = 10000
    env = RobotEnv(difficulty=2)
    obs = env.reset()

    peds_sim = env.sim_env
    scalene_profiler.start()

    env.step(env.action_space.sample())
    env.reset()
    print('start of simulation')

    # start_time = time.perf_counter()

    episode = 0
    ep_rewards = 0
    for step in range(total_steps):
        rand_action = env.action_space.sample()
        obs, reward, done, _ = env.step(rand_action)
        ep_rewards += reward
        # print(f'step {step}, reward {reward} (peds: {peds_sim.peds.size()})')

        if done:
            episode += 1
            print(f'end of episode {episode}, total rewards {ep_rewards:.3f}, remaining steps {total_steps - step}')
            ep_rewards = 0
            obs = env.reset()

    print('end of simulation')
    # scalene_profiler.stop()

    # end_time = time.perf_counter()
    # print(f'benchmark took {end_time - start_time} seconds')


if __name__ == "__main__":
    benchmark()
