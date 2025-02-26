from gymnasium import spaces
from stable_baselines3 import PPO

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.sensor.sensor_fusion import OBS_RAYS, OBS_DRIVE_STATE


def test_can_create_env():
    env = RobotEnv()
    assert env is not None


def test_can_return_valid_observation():
    env = RobotEnv()
    drive_state_spec: spaces.Box = env.observation_space[OBS_DRIVE_STATE]
    lidar_state_spec: spaces.Box = env.observation_space[OBS_RAYS]

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert OBS_DRIVE_STATE in obs and OBS_RAYS in obs
    assert drive_state_spec.shape == obs[OBS_DRIVE_STATE].shape
    assert lidar_state_spec.shape == obs[OBS_RAYS].shape


def test_can_simulate_with_pedestrians():
    total_steps = 1000
    env = RobotEnv()
    env.reset()
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(rand_action)
        done = terminated or truncated
        if done:
            env.reset()


def test_ego_ped_env():
    total_steps = 1000
    robot_model = PPO.load("./model/run_043", env=None)
    env = PedestrianEnv(robot_model=robot_model)
    assert env is not None
    env.reset()
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, _, done, _, _ = env.step(rand_action)
        if done:
            env.reset()


if __name__ == "__main__":
    # test_can_create_env()
    # test_can_return_valid_observation()
    # test_can_simulate_with_pedestrians()
    test_ego_ped_env()
