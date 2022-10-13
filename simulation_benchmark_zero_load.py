from cProfile import Profile
from pstats import Stats
from robot_sf.robot_env import RobotEnv


def benchmark():
    total_steps = 20000
    env = RobotEnv(difficulty=2)
    obs = env.reset()

    peds_sim = env.sim_env
    print('start of simulation')

    episode = 0
    ep_rewards = 0
    for step in range(total_steps):
        rand_action = env.action_space.sample()
        obs, reward, done, _ = env.step(rand_action)
        ep_rewards += reward
        # print(f'step {step}, reward {reward} (peds: {peds_sim.peds.size()})')

        if done:
            episode += 1
            print(f'end of episode {episode}, total rewards {ep_rewards}')
            ep_rewards = 0
            obs = env.reset()

    print('end of simulation')
 

def main():
    profiler = Profile()
    profiler.enable()
    benchmark()
    profiler.disable()

    stats = Stats(profiler).sort_stats('tottime')
    stats.print_stats(50)


if __name__ == "__main__":
    main()
