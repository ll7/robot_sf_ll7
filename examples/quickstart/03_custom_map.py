"""Load an SVG map into Robot SF and simulate random navigation.

Usage:
    uv run python examples/quickstart/03_custom_map.py

Prerequisites:
    - maps/svg_maps/debug_06.svg (bundled sample map)

Expected Output:
    - Logs map conversion details and rollout progress on the custom map.
    - Keeps the simulation headless while sampling random actions.

Limitations:
    - Uses a random policy; episodes may end quickly on collisions.

References:
    - docs/dev_guide.md#quickstart
    - docs/SVG_MAP_EDITOR.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_sf.common.seed import set_global_seed
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.svg_map_parser import SvgMapConverter

STEP_COUNT = 30
SEED = 87234
MAP_ALIAS = "quickstart_svg"
MAP_PATH = Path(__file__).resolve().parents[2] / "maps/svg_maps/debug_06.svg"


def load_svg_map(path: Path) -> MapDefinition:
    """Parse the SVG file into a MapDefinition instance."""

    print(f"Loading SVG map from: {path}")
    converter = SvgMapConverter(str(path))
    return converter.map_definition


def build_config(map_definition: MapDefinition) -> RobotSimulationConfig:
    """Attach the provided map definition to a robot simulation config."""

    config = RobotSimulationConfig()
    config.map_pool = MapDefinitionPool(map_defs={MAP_ALIAS: map_definition})
    return config


def run_demo() -> None:
    """Execute a random rollout on the SVG-derived map."""

    set_global_seed(SEED)
    map_definition = load_svg_map(MAP_PATH)
    print(
        f"Loaded map '{MAP_ALIAS}' with width={map_definition.width:.2f} "
        f"height={map_definition.height:.2f}",
    )

    config = build_config(map_definition)
    env = make_robot_env(config=config, debug=False)

    total_reward = 0.0
    try:
        env.reset()
        print("Environment reset using the custom map.")
        print(f"Robot spawn zones: {len(map_definition.robot_spawn_zones)}")

        for step in range(1, STEP_COUNT + 1):
            action = env.action_space.sample()
            step_result = env.step(action)
            _, reward, done = _normalize_step(step_result)
            total_reward += reward

            if step == 1 or step % 5 == 0 or done:
                print(f"Step {step:02d}: reward={reward:.3f} done={done}")

            if done:
                print("Episode ended on the custom map; resetting.")
                env.reset()
    finally:
        env.exit()

    print(f"\nDemo complete. Total reward collected: {total_reward:.3f}")


def _normalize_step(step_result: tuple[Any, ...]) -> tuple[Any, float, bool]:
    """Support both Gym and Gymnasium step signatures."""

    if len(step_result) == 5:
        observation, reward, terminated, truncated, _ = step_result
        return observation, float(reward), bool(terminated or truncated)

    observation, reward, done, _ = step_result
    return observation, float(reward), bool(done)


if __name__ == "__main__":
    run_demo()
