"""Record a simulation from an SVG map and replay it.

Usage:
    uv run python examples/advanced/15_view_recording.py

Prerequisites:
    - maps/svg_maps/02_simple_maps.svg

Expected Output:
    - Recording saved under `output/recordings/` and visualized via the playback tool.

Limitations:
    - Requires display access for rendering and playback visualization.

References:
    - docs/SIM_VIEW.md
"""

import os

from loguru import logger

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.render.playback_recording import load_states_and_visualize


def _fast_demo_enabled() -> bool:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    return os.environ.get("ROBOT_SF_FAST_DEMO", "0") == "1" or "PYTEST_CURRENT_TEST" in os.environ


def _step_budget(default: int) -> int:
    """TODO docstring. Document this function.

    Args:
        default: TODO docstring.

    Returns:
        TODO docstring.
    """
    override = os.environ.get("ROBOT_SF_EXAMPLES_MAX_STEPS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:  # pragma: no cover - defensive guard
            pass
    if _fast_demo_enabled():
        return min(default, 64)
    return default


logger.info("Recording a random policy rollout and replaying the results.")


def test_simulation(map_definition: MapDefinition):
    """Test the simulation with a random policy."""

    logger.info("Creating the environment.")
    env_config = EnvSettings(map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}))
    env = RobotEnv(env_config, debug=True, recording_enabled=True)  # Activate recording

    env.reset()

    step_budget = _step_budget(1000)
    logger.info("Simulating the random policy (steps=%s).", step_budget)
    for _ in range(step_budget):
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

    recordings_dir = get_artifact_category_path("recordings")
    latest_file = max(recordings_dir.iterdir(), key=lambda path: path.stat().st_ctime)
    return latest_file


def main():
    """Simulate a random policy with a map defined in SVG format and view the recording."""

    # Create example recording
    map_def = convert_map("maps/svg_maps/02_simple_maps.svg")
    test_simulation(map_def)

    # Load the states from the file and view the recording
    if _fast_demo_enabled():
        logger.info("Fast demo enabled: skipping playback visualization to keep runtime short.")
    else:
        load_states_and_visualize(get_file())


if __name__ == "__main__":
    main()
