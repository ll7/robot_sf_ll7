"""Quickstart: enable occupancy grid observations and run a spawn-safety query.

This script keeps everything headless and finishes in a few seconds. It:
1) Configures a 3-channel grid (obstacles, pedestrians, combined)
2) Resets the environment and inspects the grid observation
3) Runs a simple circular query at the grid center to check spawn clearance
4) Steps a handful of times to demonstrate continuous grid updates
"""

from __future__ import annotations

from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import GridConfig, RobotSimulationConfig
from robot_sf.nav.occupancy_grid import GridChannel, POIQuery, POIQueryType


def main() -> None:
    """Demonstrate enabling grid observations and a center spawn query."""
    grid_config = GridConfig(
        width=8.0,
        height=8.0,
        resolution=0.2,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
        use_ego_frame=False,
    )
    config = RobotSimulationConfig(
        use_occupancy_grid=True,
        include_grid_in_observation=True,
        grid_config=grid_config,
    )

    env = make_robot_env(config=config, debug=False)

    obs, _info = env.reset(seed=42)
    grid_obs = obs["occupancy_grid"]
    logger.info(
        "Grid observation initialised: shape=%s dtype=%s range=[%.3f, %.3f]",
        grid_obs.shape,
        grid_obs.dtype,
        float(grid_obs.min()),
        float(grid_obs.max()),
    )

    grid = env.unwrapped.occupancy_grid
    query = POIQuery(
        x=grid_config.width / 2,
        y=grid_config.height / 2,
        radius=0.5,
        query_type=POIQueryType.CIRCLE,
    )
    result = grid.query(query)
    logger.info(
        "Center query -> occupied=%s safe_to_spawn=%s occupancy_fraction=%.3f",
        result.is_occupied,
        result.safe_to_spawn,
        result.occupancy_fraction,
    )

    for step_idx in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        grid_obs = obs["occupancy_grid"]
        logger.debug(
            "Step %02d: grid max=%.3f reward=%.3f",
            step_idx,
            float(grid_obs.max()),
            float(reward),
        )
        if terminated or truncated:
            env.reset(seed=step_idx + 1)

    env.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
