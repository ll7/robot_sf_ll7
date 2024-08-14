"""Simulate the trained robot and a pedestrian with a random policy."""
import os
from pathlib import Path

from loguru import logger
from stable_baselines3 import PPO

from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.render.playback_recording import load_states_and_visualize




logger.info("Simulate the trained robot and a pedestrian with a random policy.")

def test_simulation(map_definition: MapDefinition):

    logger.info("Creating the environment.")
    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True)
    )
    env = PedestrianEnv(env_config, debug=True, recording_enabled=True)

    model = PPO.load("./model/run_043", env=None)

    obs = env.reset()

    logger.info("Simulating the random policy.")
    for _ in range(10000):
        action_robot, _ = model.predict(obs[0], deterministic=True)
        action_ped = env.action_space[1].sample()
        obs, _, done, _ = env.step(action_robot, action_ped)
        env.render()

    env.reset()
    env.exit()

def convert_map(svg_file: str):
    """Create MapDefinition from svg file."""

    logger.info("Converting SVG map to MapDefinition object.")
    logger.info(f"SVG file: {svg_file}")

    converter = SvgMapConverter(svg_file)
    return converter.map_definition

def get_file():
    """Get the latest recorded file."""

    filename = max(
        os.listdir('recordings'), key=lambda x: os.path.getctime(os.path.join('recordings', x)))
    return Path('recordings', filename)



def main():

    map_def = convert_map("maps/svg_maps/04_small_mid_object.svg")

    test_simulation(map_def)

    load_states_and_visualize(get_file())


if __name__ == "__main__":
    main()
