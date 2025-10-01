"""Simulate the trained robot and a trained pedestrian - Updated to use factory pattern."""

import loguru

# New factory pattern imports
from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def make_env_new(map_name: str, robot_model_path: str):
    """Create environment using new factory pattern."""
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    map_definition = convert_map(map_name)
    robot_model = load_trained_policy(robot_model_path)

    # Use new unified configuration
    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty,
            ped_density_by_difficulty=ped_densities,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )

    # Use factory pattern for cleaner creation
    return make_pedestrian_env(
        config=config,
        robot_model=robot_model,
        debug=True,
        recording_enabled=False,
    )


def make_env_old(map_name: str, robot_model_path: str):
    """Legacy environment creation - still works for backward compatibility."""
    from robot_sf.gym_env.env_config import PedEnvSettings
    from robot_sf.gym_env.pedestrian_env import PedestrianEnv

    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    map_definition = convert_map(map_name)
    robot_model = load_trained_policy(robot_model_path)

    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty,
            ped_density_by_difficulty=ped_densities,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    return PedestrianEnv(env_config, robot_model=robot_model, debug=True, recording_enabled=False)


def run(filename: str, map_name: str, robot_model: str, use_new_pattern: bool = True):
    """Run the simulation with either new or old environment creation pattern."""
    if use_new_pattern:
        logger.info("Using new factory pattern for environment creation")
        env = make_env_new(map_name, robot_model)
    else:
        logger.info("Using legacy environment creation (backward compatibility)")
        env = make_env_old(map_name, robot_model)

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

    # Demonstrate both new and old patterns work
    logger.info("=== Testing New Factory Pattern ===")
    try:
        run(PED_MODEL, SVG_MAP, ROBOT_MODEL, use_new_pattern=True)
    except Exception as e:
        logger.warning(f"New pattern failed: {e}")
        logger.info("=== Falling back to Legacy Pattern ===")
        run(PED_MODEL, SVG_MAP, ROBOT_MODEL, use_new_pattern=False)
