"""Demo: occupancy-based reward shaping for RL-friendly observations.

This script keeps things lightweight and headless. It:
1) Enables occupancy grid observations
2) Derives a simple clearance penalty from the combined channel
3) Runs a short rollout to show how to integrate the shaped term
"""

from __future__ import annotations

from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import GridConfig, RobotSimulationConfig
from robot_sf.nav.occupancy_grid import GridChannel


def compute_clearance_penalty(grid_obs) -> float:
    """Return a penalty based on max occupancy in the combined channel."""
    combined = grid_obs[-1]  # expect COMBINED as last channel
    return float(combined.max())


def main() -> None:
    """Run a short headless rollout with a clearance-based shaped reward."""
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
    obs, _info = env.reset(seed=123)

    shaped_return = 0.0
    for step_idx in range(20):
        grid_obs = obs["occupancy_grid"]
        clearance_penalty = compute_clearance_penalty(grid_obs)

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
            obs, _info = env.reset(seed=step_idx + 200)

    env.close()
    logger.info("Finished. Cumulative shaped reward: %.3f", shaped_return)


if __name__ == "__main__":
    main()
