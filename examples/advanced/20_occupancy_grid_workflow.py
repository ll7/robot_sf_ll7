"""Advanced occupancy grid workflow: standalone grids, queries, and reward shaping.

What this demo covers:
1) Builds a standalone ego-frame grid with synthetic obstacles/pedestrians
2) Runs spawn-validation queries (point + circular regions) with per-channel results
3) Shows a simple occupancy-based reward shaping loop inside RobotEnv

All steps run headless and finish in a few seconds; no extra assets are required.
"""

from __future__ import annotations

import math
import os

from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import GridConfig, RobotSimulationConfig
from robot_sf.nav.occupancy_grid import (
    GridChannel,
    OccupancyGrid,
    POIQuery,
    POIQueryType,
)


def build_standalone_grid() -> OccupancyGrid:
    """Create and query a grid without spinning up the full environment."""
    grid_config = GridConfig(
        width=6.0,
        height=6.0,
        resolution=0.2,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
        use_ego_frame=True,
    )
    grid = OccupancyGrid(config=grid_config)

    obstacles = [((1.0, 1.0), (5.0, 1.0)), ((1.0, 4.5), (5.0, 4.5))]
    pedestrians = [((2.5, 3.0), 0.35), ((3.5, 2.0), 0.3)]
    robot_pose = ((3.0, 3.0), math.pi / 4)

    grid.generate(obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose)

    queries = [
        POIQuery(x=3.0, y=3.0, radius=0.5, query_type=POIQueryType.CIRCLE),
        POIQuery(x=1.2, y=1.0, query_type=POIQueryType.POINT),
    ]
    for query in queries:
        result = grid.query(query)
        logger.info(
            "Query %s @ (%.2f, %.2f): occupied=%s safe_to_spawn=%s per-channel=%s",
            query.query_type.value,
            query.x,
            query.y,
            result.is_occupied,
            result.safe_to_spawn,
            {ch.value: f"{val:.3f}" for ch, val in result.per_channel_results.items()},
        )

    _render_overlay(grid, robot_pose)
    return grid


def reward_shaping_demo() -> None:
    """Show how to derive a simple clearance penalty from the grid observation."""
    grid_config = GridConfig(
        width=8.0,
        height=8.0,
        resolution=0.25,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
    )
    config = RobotSimulationConfig(
        use_occupancy_grid=True,
        include_grid_in_observation=True,
        grid_config=grid_config,
    )
    env = make_robot_env(config=config, debug=False)

    obs, _info = env.reset(seed=7)
    shaped_return = 0.0

    for step_idx in range(8):
        grid_obs = obs["occupancy_grid"]
        clearance_penalty = float(grid_obs[-1].max())  # combined channel
        shaped_reward = -0.1 * clearance_penalty
        shaped_return += shaped_reward
        logger.info(
            "Step %02d: clearance_penalty=%.3f shaped_reward=%.3f",
            step_idx,
            clearance_penalty,
            shaped_reward,
        )

        action = env.action_space.sample()
        obs, _reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=step_idx + 20)

    env.close()
    logger.info("Toy shaped return after 8 steps: %.3f", shaped_return)


def _render_overlay(grid: OccupancyGrid, robot_pose) -> None:
    """Headless visualization of the grid overlay."""
    try:
        import pygame
    except ImportError:
        logger.warning("Skipping overlay rendering; pygame not installed.")
        return

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    scale = 4
    surface = pygame.Surface((grid.shape[2] * scale, grid.shape[1] * scale), pygame.SRCALPHA)
    grid.render_pygame(surface, robot_pose=robot_pose, scale=scale, alpha=160)
    logger.info("Rendered headless grid overlay surface at %s", surface.get_size())
    pygame.quit()


def main() -> None:
    """Run standalone grid build and reward shaping demos."""
    build_standalone_grid()
    reward_shaping_demo()


if __name__ == "__main__":
    main()
