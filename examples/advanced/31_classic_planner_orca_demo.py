#!/usr/bin/env python3
"""Demonstrate classic global + ORCA-style local planning on the lake map."""

from __future__ import annotations

from loguru import logger

from robot_sf.common import ensure_interactive_backend
from robot_sf.common.artifact_paths import get_artifact_category_path, get_repository_root
from robot_sf.common.logging import configure_logging
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner.classic_global_planner import ClassicPlannerConfig
from robot_sf.planner.classic_planner_adapter import (
    PlannerActionAdapter,
    attach_classic_global_planner,
)
from robot_sf.planner.socnav import make_orca_policy
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

PYGAME_WINDOW = True
if PYGAME_WINDOW:
    logger.info("Using interactive pygame backend for rendering")
else:
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")


def main(steps: int = 600, seed: int | None = 13) -> None:
    """Run a short ORCA-style rollout guided by the classic global planner."""
    configure_logging(verbose=True)
    ensure_interactive_backend()

    root = get_repository_root()
    map_path = (
        root / "maps" / "obstacle_svg_maps" / "uni_campus_with_lake_as_obstacle_and_routes.svg"
    )
    map_def = convert_map(str(map_path))
    attach_classic_global_planner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=2,
            algorithm="theta_star_v2",
        ),
    )
    map_pool = MapDefinitionPool(map_defs={"lake": map_def})

    see_everything = False
    if see_everything:
        # see everything
        grid_cfg = GridConfig(
            width=map_def.width,
            height=map_def.height,
            resolution=1.0,
            use_ego_frame=False,
            center_on_robot=False,
        )
    else:
        # see robot area
        grid_cfg = GridConfig(
            width=20.0,
            height=20.0,
            resolution=1.0,
            use_ego_frame=False,
            center_on_robot=True,
        )
    env_config = RobotSimulationConfig(
        map_pool=map_pool,
        observation_mode=ObservationMode.SOCNAV_STRUCT,
        use_occupancy_grid=True,
        include_grid_in_observation=True,
        grid_config=grid_cfg,
        show_occupancy_grid=True,
        sim_config=SimulationSettings(
            stack_steps=1,
            difficulty=0,
            ped_density_by_difficulty=[0.03],
            time_per_step_in_secs=0.2,
        ),
        robot_config=BicycleDriveSettings(
            radius=0.5,
            wheelbase=1.0,
            max_velocity=2.0,
            max_accel=1.2,
            max_steer=0.7,
            allow_backwards=False,
        ),
        use_planner=True,
    )
    env = make_robot_env(config=env_config, debug=True)
    if env.sim_ui:
        env.sim_ui.show_lidar = False

    policy = make_orca_policy()
    action_adapter = PlannerActionAdapter(
        robot=env.simulator.robots[0],
        action_space=env.action_space,
        time_step=env_config.sim_config.time_per_step_in_secs,
    )

    obs, _ = env.reset(seed=seed)
    output_dir = get_artifact_category_path("plots")
    logger.info(
        "Starting ORCA demo on %s with planner %s; outputs -> %s",
        map_path.name,
        env_config.use_planner,
        output_dir,
    )

    for step_idx in range(steps):
        linear, angular = policy.act(obs)
        action = action_adapter.from_velocity_command((linear, angular))
        obs, _reward, done, truncated, info = env.step(action)
        env.render()

        if done or truncated:
            logger.info(
                "Episode ended at step %d (done=%s, truncated=%s, meta=%s)",
                step_idx,
                done,
                truncated,
                {
                    "goal": info.get("is_robot_at_goal"),
                    "collided": info.get("is_obstacle_collision"),
                },
            )
            break

    env.close()
    logger.info("✓ Classic planner + ORCA demo complete.")


if __name__ == "__main__":
    main()
