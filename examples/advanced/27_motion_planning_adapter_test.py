#!/usr/bin/env python3
"""Test the motion planning adapter with a simple example.

This example demonstrates:
1. Loading an SVG map and converting it to a motion planning grid
2. Using helper functions to analyze and visualize the grid
3. Planning a path using ThetaStar algorithm
4. Visualizing the planned path
"""

from pathlib import Path

from loguru import logger
from python_motion_planning.common import TYPES, Visualizer
from python_motion_planning.path_planner import ThetaStar

from robot_sf.common import ensure_interactive_backend
from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.common.logging import configure_logging
from robot_sf.nav.motion_planning_adapter import (
    MotionPlanningGridConfig,
    get_obstacle_statistics,
    map_definition_to_motion_planning_grid,
    visualize_grid,
)
from robot_sf.nav.svg_map_parser import convert_map

MAP_PATH = Path("maps/svg_maps/classic_overtaking.svg")
"""svg map file to load for the test."""


def main() -> None:
    """Run the motion planning adapter test and path planning demo."""
    configure_logging(verbose=True)
    ensure_interactive_backend()

    # Load and convert map
    map_def = convert_map(str(MAP_PATH))
    logger.info(
        "Loaded map: {w}x{h} with {obs} obstacles",
        w=map_def.width,
        h=map_def.height,
        obs=len(map_def.obstacles),
    )

    cfg = MotionPlanningGridConfig(cells_per_meter=1.0, inflate_radius_cells=2)
    grid = map_definition_to_motion_planning_grid(map_def, config=cfg)
    logger.info(
        "Converted to python_motion_planning Grid with shape {shape} and inflation radius {r}",
        shape=grid.type_map.shape,
        r=cfg.inflate_radius_cells,
    )

    # Analyze grid using helper functions
    stats = get_obstacle_statistics(grid)
    logger.info(
        "Obstacle cells: {obs} ({pct:.2f}% of grid)",
        obs=stats["obstacle_count"],
        pct=stats["obstacle_percentage"],
    )

    # Visualize grid using helper function
    output_dir = get_artifact_category_path("plots")
    # Set to None or empty string to show interactively instead of saving
    visualize_grid(grid, output_dir / "motion_planning_adapter_grid.png", title="Map Visualizer")
    visualize_grid(grid, None, title="Map Visualizer (Interactive)")

    logger.info("✓ Adapter grid generation completed.")

    # Path planning demo
    logger.info("Starting path planning")

    start = (5, 22)
    goal = (55, 5)

    grid.type_map[start] = TYPES.START
    grid.type_map[goal] = TYPES.GOAL
    logger.debug(f"Start: {start}, Goal: {goal}")

    planner = ThetaStar(map_=grid, start=start, goal=goal)
    path, path_info = planner.plan()
    grid.fill_expands(path_info["expand"])  # for visualizing the expanded nodes

    # Visualize path
    vis_path = Visualizer("Path Visualizer")
    vis_path.plot_grid_map(grid)
    vis_path.plot_path(path, style="--", color="C4")
    vis_path.fig.savefig(output_dir / "motion_planning_adapter_path.png")
    logger.info(f"Saved path visualization to {output_dir / 'motion_planning_adapter_path.png'}")
    vis_path.close()

    logger.info("✓ Path planning completed.")


if __name__ == "__main__":
    main()
