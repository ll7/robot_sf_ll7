#!/usr/bin/env python3
"""Test the motion planning adapter with a simple example."""

from pathlib import Path

import numpy as np
from loguru import logger
from python_motion_planning.common import TYPES, Visualizer

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.common.logging import configure_logging
from robot_sf.nav.motion_planning_adapter import (
    MotionPlanningGridConfig,
    map_definition_to_motion_planning_grid,
)
from robot_sf.nav.svg_map_parser import convert_map

MAP_PATH = Path("maps/svg_maps/example_map_with_obstacles.svg")
"""svg map file to load for the test."""


def main():
    configure_logging(verbose=True)

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

    type_map_np = np.asarray(grid.type_map)
    obstacle_cells = np.count_nonzero(type_map_np == TYPES.OBSTACLE)
    total_cells = type_map_np.size
    logger.info(
        "Obstacle cells: {obs} ({pct:.2f}% of grid)",
        obs=obstacle_cells,
        pct=obstacle_cells / total_cells * 100,
    )

    vis = Visualizer("Path Visualizer")
    vis.plot_grid_map(grid, equal=True)
    # save fig attribute from vis to output/plots/motion_planning_adapter_grid.png
    output_dir = get_artifact_category_path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    vis.fig.savefig(output_dir / "motion_planning_adapter_grid.png")
    logger.info(f"Saved grid visualization to {output_dir / 'motion_planning_adapter_grid.png'}")
    vis.show()
    vis.close()

    logger.info("✓ Adapter grid generation completed.")


if __name__ == "__main__":
    main()
