"""Run social navigation SVG scenarios sequentially.

Purpose:
    Execute the classic quartet of SVG-based navigation scenarios to review
    path planning behaviours interactively or in headless mode.

Usage:
    uv run python examples/benchmarks/demo_social_nav_scenarios.py

Prerequisites:
    - `maps/svg_maps/static_humans.svg`
    - `maps/svg_maps/overtaking.svg`
    - `maps/svg_maps/crossing.svg`
    - `maps/svg_maps/door_passing.svg`

Expected Output:
    - Console logs per scenario with completion status
    - Optional rendering via `env.render()` in a window when display available

Limitations:
    - Rendering requires a display; set `SDL_VIDEODRIVER=dummy` for headless runs.
"""

import os
import time

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map

SVG_MAPS = [
    ("Static humans", "maps/svg_maps/static_humans.svg"),
    ("Overtaking", "maps/svg_maps/overtaking.svg"),
    ("Crossing", "maps/svg_maps/crossing.svg"),
    ("Door passing", "maps/svg_maps/door_passing.svg"),
]


def _fast_demo_enabled() -> bool:
    """Fast demo enabled.

    Returns:
        bool: Auto-generated placeholder description.
    """
    return os.environ.get("ROBOT_SF_FAST_DEMO", "0") == "1" or "PYTEST_CURRENT_TEST" in os.environ


def _step_budget(default: int) -> int:
    """Step budget.

    Args:
        default: Auto-generated placeholder description.

    Returns:
        int: Auto-generated placeholder description.
    """
    override = os.environ.get("ROBOT_SF_EXAMPLES_MAX_STEPS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:  # pragma: no cover
            pass
    if _fast_demo_enabled():
        return min(default, 64)
    return default


STEPS_PER_SCENARIO = _step_budget(200)


def run_svg_scenario(name: str, svg_path: str) -> None:
    """Run svg scenario.

    Args:
        name: Auto-generated placeholder description.
        svg_path: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    print(f"\n=== Scenario: {name} ===")
    map_def = convert_map(svg_path)
    pool = MapDefinitionPool(map_defs={name: map_def})
    config = RobotSimulationConfig()
    config.map_pool = pool
    env = make_robot_env(config=config, debug=True)
    env.reset()
    for step in range(STEPS_PER_SCENARIO):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break
    env.close()
    print(f"Completed scenario: {name}")
    if not _fast_demo_enabled():
        time.sleep(1)


def main() -> None:
    """Main.

    Returns:
        None: Auto-generated placeholder description.
    """
    for name, svg_path in SVG_MAPS:
        run_svg_scenario(name, svg_path)
    print("\nAll scenarios completed.")


if __name__ == "__main__":
    main()
