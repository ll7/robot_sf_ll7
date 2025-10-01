"""Simulate the trained robot and a trained pedestrian."""

import loguru

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def make_env(map_name: str, robot_model: str):
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    map_definition = convert_map(map_name)
    robot_model = load_trained_policy(robot_model)

    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty,
            ped_density_by_difficulty=ped_densities,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    return PedestrianEnv(env_config, robot_model=robot_model, debug=True, recording_enabled=False)


def run(filename: str, map_name: str, robot_model: str):
    env = make_env(map_name, robot_model)

    logger.info(f"Loading pedestrian model from {filename}")

    model = load_trained_policy(filename)

    obs, _ = env.reset()

    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env.render()

        if done:
            obs, _ = env.reset()
            env.render()
    env.exit()


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/debug_06.svg"
    PED_MODEL = "./model_ped/ppo_ped_02.zip"
    ROBOT_MODEL = "./model/run_043"

    run(PED_MODEL, SVG_MAP, ROBOT_MODEL)
