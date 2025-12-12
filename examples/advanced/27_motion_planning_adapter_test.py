#!/usr/bin/env python3
"""Test the classic global planner with grid-based planning.

This example demonstrates:
1. Loading an SVG map
2. Planning a path using the ClassicGlobalPlanner (ThetaStar algorithm)
3. Visualizing the planned path on the grid
"""

from pathlib import Path

from loguru import logger

from robot_sf.common import ensure_interactive_backend
from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.common.logging import configure_logging
from robot_sf.nav.motion_planning_adapter import (
    get_obstacle_statistics,
)
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig

MAP_PATH = Path("maps/svg_maps/map3_1350_buildings_inkscape.svg")
"""svg map file to load for the test."""


def main() -> None:
    """Run the classic global planner demo."""
    configure_logging(verbose=True)
    ensure_interactive_backend()

    # Load map
    map_def = convert_map(str(MAP_PATH))
    logger.info(
        "Loaded map: {w}x{h} with {obs} obstacles",
        w=map_def.width,
        h=map_def.height,
        obs=len(map_def.obstacles),
    )

    # Create planner
    planner_config = ClassicPlannerConfig(
        cells_per_meter=2.0,
        inflate_radius_cells=10,
        algorithm="theta_star",
    )
    planner = ClassicGlobalPlanner(map_def, config=planner_config)
    logger.info("Created ClassicGlobalPlanner with ThetaStar algorithm")

    # Visualize grid
    output_dir = get_artifact_category_path("plots")
    grid = planner.grid  # Access grid to visualize
    stats = get_obstacle_statistics(grid)
    logger.info(
        "Planning grid: {shape}, {obs} obstacle cells ({pct:.2f}%)",
        shape=grid.type_map.shape,
        obs=stats["obstacle_count"],
        pct=stats["obstacle_percentage"],
    )
    planner.visualize_grid(output_dir / "classic_planner_grid.png", title="Planning Grid")

    # Plan path (world coordinates)
    start_world = (5.0, 5.0)
    goal_world = (40.0, 5.0)
    logger.info("Planning from {start} to {goal}", start=start_world, goal=goal_world)

    path_world, path_info = planner.plan(start_world, goal_world)

    if not path_world:
        logger.error("Planning failed!")
        return

    logger.info(
        "Found path with {n} waypoints (length≈{length:.2f} m)",
        n=len(path_world),
        length=path_info.get("length") if path_info else float("nan"),
    )

    planner.visualize_path(
        path_world,
        output_dir / "classic_planner_path.png",
        title="Classic Planner Path",
        path_style="--",
        path_color="C4",
        linewidth=2,
    )

    logger.info("✓ Path planning completed.")


if __name__ == "__main__":
    main()
