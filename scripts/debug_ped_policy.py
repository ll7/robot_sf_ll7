import os
from pathlib import Path

import loguru
from stable_baselines3 import PPO

from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def make_env():
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    map_definition = convert_map("maps/svg_maps/debug_03.svg")
    robot_model = PPO.load("./model/run_043", env=None)

    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty,
            ped_density_by_difficulty=ped_densities,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    return PedestrianEnv(env_config, robot_model=robot_model, debug=True, recording_enabled=False)


def get_file():
    """Get the latest model file."""

    filename = max(
        os.listdir("model_ped"),
        key=lambda x: os.path.getctime(os.path.join("model_ped", x)),
    )
    return Path("model_ped", filename)


def run():
    env = make_env()
    filename = get_file()
    # filename = "./model_ped/ppo_2024-09-06_23-52-17.zip"
    logger.info(f"Loading pedestrian model from {filename}")

    model = PPO.load(filename, env=env)

    obs = env.reset()
    ep_rewards = 0

    for _ in range(10000):
        if isinstance(obs, tuple):  # Check env.reset()
            obs = obs[0]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, meta = env.step(action)
        ep_rewards += reward
        env.render()

        if done:
            logger.info(extract_info(meta, ep_rewards))
            ep_rewards = 0
            obs = env.reset()
            env.render()
    env.exit()


def extract_info(meta: dict, reward: float) -> str:
    meta = meta["meta"]
    eps_num = meta["episode"]
    steps = meta["step_of_episode"]
    done = [key for key, value in meta.items() if value is True]
    dis = meta["distance_to_robot"]
    return f"Episode: {eps_num}, Steps: {steps}, Done: {done}, Reward: {reward}, Distance: {dis}"


if __name__ == "__main__":
    run()
