"""Simulate the trained robot and a pedestrian with a random policy."""
import os
from pathlib import Path

from loguru import logger
from stable_baselines3 import PPO

from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.render.playback_recording import load_states_and_visualize
from robot_sf.nav.svg_map_parser import convert_map


logger.info("Simulate the trained robot and a pedestrian with a random policy.")


def test_simulation(map_definition: MapDefinition):

    logger.info("Creating the environment.")
    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True)
        )

    robot_model = PPO.load("./model/run_043", env=None)

    env = PedestrianEnv(env_config, robot_model=robot_model, debug=True, recording_enabled=True)

    obs, _ = env.reset()

    logger.info("Simulating the random policy.")
    for _ in range(10000):
        action_ped = env.action_space.sample()
        obs, _, done, _, _ = env.step(action_ped)
        env.render()

        if done:
            obs, _ = env.reset()
            env.render()

    env.reset()
    env.exit()


def get_file():
    """Get the latest recorded file."""

    filename = max(
        os.listdir('recordings'), key=lambda x: os.path.getctime(os.path.join('recordings', x)))
    return Path('recordings', filename)


def main():

    map_def = convert_map("maps/svg_maps/debug_06.svg")

    test_simulation(map_def)

    load_states_and_visualize(get_file())


if __name__ == "__main__":
    main()
