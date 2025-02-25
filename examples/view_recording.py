"""Simulate a random policy with a map defined in SVG format and view the recording."""

import os
from pathlib import Path

from loguru import logger

from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.render.playback_recording import load_states_and_visualize


logger.info("Simulate a random policy with a map defined in SVG format.")


def test_simulation(map_definition: MapDefinition):
    """Test the simulation with a random policy."""

    logger.info("Creating the environment.")
    env_config = EnvSettings(map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}))
    env = RobotEnv(env_config, debug=True, recording_enabled=True)  # Activate recording

    env.reset()

    logger.info("Simulating the random policy.")
    for _ in range(1000):
        action = env.action_space.sample()
        env.step(action)
        env.render()

    env.reset()  # Save the recording
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
        os.listdir("recordings"), key=lambda x: os.path.getctime(os.path.join("recordings", x))
    )
    return Path("recordings", filename)


def main():
    """Simulate a random policy with a map defined in SVG format and view the recording."""

    # Create example recording
    map_def = convert_map("maps/svg_maps/02_simple_maps.svg")
    test_simulation(map_def)

    # Load the states from the file and view the recording
    load_states_and_visualize(get_file())


if __name__ == "__main__":
    main()
