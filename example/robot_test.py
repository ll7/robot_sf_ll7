from robot_sf.robot_env import RobotEnv


def main():
    total_steps = 1000
    env = RobotEnv(difficulty=2)
    obs = env.reset()

    peds_sim = env.robot.map.peds_sim_env
    print(peds_sim.max_population_for_new_individual)
    print(peds_sim.max_population_for_new_group)

    episode = 0
    ep_rewards = 0
    for step in range(total_steps):
        rand_action = env.action_space.sample()
        obs, reward, done, _ = env.step(rand_action)
        ep_rewards += reward
        print(f'step {step}, reward {reward} (peds: {peds_sim.peds.size()})')

        if done:
            episode += 1
            print(f'end of episode {episode}, total rewards {ep_rewards}')
            obs = env.reset()
 

if __name__ == "__main__":
    main()
