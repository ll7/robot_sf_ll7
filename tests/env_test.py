from robot_sf.robot_env import RobotEnv


def test_can_create_env():
    env = RobotEnv()
    assert env is not None


def test_can_simulate_with_pedestrians():
    total_steps = 1000
    env = RobotEnv()
    env.reset()
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, _, done, _ = env.step(rand_action)
        if done:
            env.reset()
