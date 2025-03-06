"""Simulate a hardcoded deterministic policy with four actions."""

import loguru
from stable_baselines3 import PPO

from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_ego.unicycle_drive import UnicycleAction
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings
from scripts.debug_ped_policy import extract_info

logger = loguru.logger
ACCELERATION: UnicycleAction = (1, 0)
LEFT_CURVE: UnicycleAction = (0, 0.5)
RIGHT_CURVE: UnicycleAction = (0, -0.5)
NO_OP = (0.0, 0.0)


def select_action(obs: dict) -> UnicycleAction:
    drive_state = obs["drive_state"][0]
    target_angle = drive_state[3]

    if drive_state[0] < 1:  # velocity
        return ACCELERATION
    elif abs(target_angle) > 0.1:
        if target_angle > 0:
            return LEFT_CURVE
        else:
            return RIGHT_CURVE
    else:
        return NO_OP


def make_env():
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    map_definition = convert_map("maps/svg_maps/debug_02.svg")
    robot_model = PPO.load("./model/run_043", env=None)

    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    env_config.sim_config.ped_density_by_difficulty = ped_densities
    env_config.sim_config.difficulty = difficulty
    return PedestrianEnv(
        env_config,
        robot_model=robot_model,
        debug=True,
    )


def run():
    env = make_env()

    obs = env.reset()
    ep_rewards = 0
    for _ in range(10000):
        if isinstance(obs, tuple):
            action = select_action(obs[0])
        else:
            action = select_action(obs)
        obs, reward, done, _, meta = env.step(action)
        ep_rewards += reward
        env.render()

        if done:
            logger.info(extract_info(meta, ep_rewards))
            ep_rewards = 0
            obs = env.reset()
            env.render()
    env.exit()


if __name__ == "__main__":
    run()
