"""Simulate a random policy with a map defined in SVG format."""

import numpy as np
from loguru import logger

from robot_sf.gym_env.env_config import EnvSettings, SimulationSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

logger.info("Simulate a random policy with a map defined in SVG format.")


def test_simulation(map_definition: MapDefinition):
    """Test the simulation with a random policy."""

    logger.info("Creating the environment.")
    env_config = EnvSettings(
        sim_config=SimulationSettings(
            stack_steps=1,
            difficulty=0,
            ped_density_by_difficulty=[0.06],
        ),
        robot_config=DifferentialDriveSettings(radius=1.0),
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
    )
    env = RobotEnv(env_config, debug=True)
    # env.observation_space, env.action_space = prepare_gym_spaces()
    # model = PPO.load("./model/run_023", env=env)

    def obs_adapter(orig_obs):
        drive_state = orig_obs[OBS_DRIVE_STATE]
        ray_state = orig_obs[OBS_RAYS]
        drive_state = drive_state[:, :-1]
        drive_state[:, 2] *= 10
        drive_state = np.squeeze(drive_state)
        ray_state = np.squeeze(ray_state)
        return np.concatenate((ray_state, drive_state), axis=0)

    obs, _ = env.reset()
    logger.info("Simulating the random policy.")
    for _ in range(10000):
        obs = obs_adapter(obs)
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        env.render()

        if done:
            obs, _ = env.reset()
            env.render()
    env.exit()


def main():
    """Simulate a random policy with a map defined in SVG format."""
    logger.info("Simulating a random policy with the map.")

    svg_file = "maps/svg_maps/debug_06.svg"

    logger.info("Converting SVG map to MapDefinition object.")
    logger.info(f"SVG file: {svg_file}")

    converter = SvgMapConverter(svg_file)
    map_definition = converter.map_definition

    test_simulation(map_definition)

    logger.info("MapDefinition object created.")


if __name__ == "__main__":
    main()
