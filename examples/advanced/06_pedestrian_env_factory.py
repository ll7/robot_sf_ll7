"""Compare factory and legacy pedestrian environment creation.

Usage:
    uv run python examples/advanced/06_pedestrian_env_factory.py

Prerequisites:
    - maps/svg_maps/debug_06.svg
    - model_ped/ppo_ped_02.zip (required)
    - model/run_043.zip (optional; falls back to StubRobotModel if missing)

Expected Output:
    - Console logs showing factory and legacy environment runs for the same assets.

Limitations:
    - Opens a pygame window; use headless environment variables on CI.

References:
    - docs/dev_guide.md#pedestrian-environments
"""

from pathlib import Path

import loguru

# New factory pattern imports
from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env._stub_robot_model import StubRobotModel
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def _load_robot_model_or_stub(robot_model_path: str):
    """Load a robot model or fall back to the stub when missing."""
    if not Path(robot_model_path).exists():
        logger.warning(
            f"Robot model not found at {robot_model_path}; using StubRobotModel for demo."
        )
        return StubRobotModel()
    return load_trained_policy(robot_model_path)


def make_env_new(map_name: str, robot_model_path: str):
    """Create environment using new factory pattern."""
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    map_definition = convert_map(map_name)
    robot_model = _load_robot_model_or_stub(robot_model_path)

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
    robot_model = _load_robot_model_or_stub(robot_model_path)

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
    if not Path(filename).exists():
        raise FileNotFoundError(
            "Pedestrian model file not found: "
            f"{filename}. Download or place the pre-trained PPO model at this path. "
            "See docs/dev/issues/classic-interactions-ppo/ for guidance."
        )
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
    ROBOT_MODEL = "./model/run_043.zip"

    # Demonstrate both new and old patterns work
    logger.info("=== Testing New Factory Pattern ===")
    try:
        run(PED_MODEL, SVG_MAP, ROBOT_MODEL, use_new_pattern=True)
    except FileNotFoundError as e:
        logger.exception(str(e))
        raise SystemExit(1) from e
    except Exception as e:
        logger.warning(f"New pattern failed: {e}")
        logger.info("=== Falling back to Legacy Pattern ===")
        run(PED_MODEL, SVG_MAP, ROBOT_MODEL, use_new_pattern=False)
