"""Run ego pedestrian simulation with recording playback.

Usage:
    uv run python examples/advanced/11_ego_pedestrian_policy.py

Prerequisites:
    - maps/svg_maps/debug_06.svg
    - model/run_043

Expected Output:
    - Recording saved under `output/recordings/` and replayed via the playback viewer.

Limitations:
    - Requires Stable-Baselines3 and an interactive display for playback.

References:
    - docs/dev_guide.md#pedestrian-environments
"""

from pathlib import Path

from loguru import logger
from stable_baselines3 import PPO

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.render.playback_recording import load_states_and_visualize
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger.info("Running ego pedestrian simulation with random actions and recording playback.")


def test_simulation(map_definition: MapDefinition) -> None:
    """Run a short ego pedestrian simulation and render the playback loop."""
    logger.info("Creating the environment.")
    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.04]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )

    robot_model = PPO.load("./model/run_043", env=None)

    env = PedestrianEnv(
        env_config,
        robot_model=robot_model,
        debug=True,
        recording_enabled=True,
        peds_have_obstacle_forces=True,
    )
    try:
        _, _ = env.reset()

        logger.info("Simulating the random policy.")
        for _ in range(1000):
            action_ped = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action_ped)
            done = bool(terminated or truncated)
            env.render()

            if done:
                _, _ = env.reset()
                env.render()

        env.reset()
    finally:
        env.exit()


def get_file() -> Path:
    """Get the latest recorded file."""

    recordings_dir = get_artifact_category_path("recordings")
    if not recordings_dir.exists():
        raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")

    recordings = [path for path in recordings_dir.iterdir() if path.is_file()]
    if not recordings:
        raise FileNotFoundError(f"No recordings found in: {recordings_dir}")

    latest_file = max(recordings, key=lambda path: path.stat().st_mtime)
    return latest_file


def main() -> None:
    """Run ego pedestrian simulation and visualize the recorded playback.

    This function orchestrates the complete workflow:
    1. Loads the map from SVG
    2. Runs a simulation with the trained policy
    3. Loads and visualizes the recorded states
    """
    map_def = convert_map("maps/svg_maps/debug_06.svg")

    test_simulation(map_def)

    load_states_and_visualize(get_file())


if __name__ == "__main__":
    main()
