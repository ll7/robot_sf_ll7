#!/usr/bin/env python3
"""Comparison demo of VisibilityPlanner vs ClassicGlobalPlanner.

This example demonstrates the two different global planning approaches:

1. VisibilityPlanner (visibility graph-based):
   - Uses continuous vector representation
   - Constructs visibility graph from obstacle corners
   - Good for sparse environments with clear line-of-sight

2. ClassicGlobalPlanner (grid-based with ThetaStar):
   - Uses rasterized grid representation
   - Supports any-angle paths (not constrained to grid directions)
   - Better for dense environments or narrow passages
"""

from pathlib import Path

from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import (
    ClassicGlobalPlanner,
    ClassicPlannerConfig,
    PlannerConfig,
    VisibilityPlanner,
)

MAP_PATH = Path("maps/svg_maps/classic_overtaking.svg")


def main() -> None:
    """Compare the two planner approaches on the same map."""
    configure_logging(verbose=True)

    # Load map
    map_def = convert_map(str(MAP_PATH))
    logger.info(
        "Loaded map: {w}x{h} with {obs} obstacles",
        w=map_def.width,
        h=map_def.height,
        obs=len(map_def.obstacles),
    )

    # Define start and goal
    start = (5.0, 22.0)
    goal = (55.0, 5.0)

    # Test VisibilityPlanner
    logger.info("\n=== Testing VisibilityPlanner ===")
    vis_planner = VisibilityPlanner(
        map_def,
        PlannerConfig(
            robot_radius=0.4,
            min_safe_clearance=0.3,
            enable_smoothing=True,
        ),
    )

    try:
        vis_path = vis_planner.plan(start, goal)
        logger.info(
            "VisibilityPlanner: Found path with {n} waypoints",
            n=len(vis_path),
        )
        logger.debug("Path waypoints: {path}", path=vis_path[:5])  # First 5 points
    except Exception as e:
        logger.error(f"VisibilityPlanner failed: {e}")
        vis_path = []

    # Test ClassicGlobalPlanner
    logger.info("\n=== Testing ClassicGlobalPlanner ===")
    classic_planner = ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_cells=2,
            algorithm="theta_star",
        ),
    )

    try:
        classic_path = classic_planner.plan(start, goal)
        logger.info(
            "ClassicGlobalPlanner: Found path with {n} waypoints",
            n=len(classic_path),
        )
        logger.debug("Path waypoints: {path}", path=classic_path[:5])  # First 5 points
    except Exception as e:
        logger.error(f"ClassicGlobalPlanner failed: {e}")
        classic_path = []

    # Summary
    logger.info("\n=== Summary ===")
    logger.info("Start: {start}, Goal: {goal}", start=start, goal=goal)
    if vis_path:
        logger.info(
            "✓ VisibilityPlanner: {n} waypoints, path length ≈ {length:.1f}m",
            n=len(vis_path),
            length=sum(
                (
                    (vis_path[i + 1][0] - vis_path[i][0]) ** 2
                    + (vis_path[i + 1][1] - vis_path[i][1]) ** 2
                )
                ** 0.5
                for i in range(len(vis_path) - 1)
            ),
        )
    else:
        logger.warning("✗ VisibilityPlanner: No path found")

    if classic_path:
        logger.info(
            "✓ ClassicGlobalPlanner: {n} waypoints, path length ≈ {length:.1f}m",
            n=len(classic_path),
            length=sum(
                (
                    (classic_path[i + 1][0] - classic_path[i][0]) ** 2
                    + (classic_path[i + 1][1] - classic_path[i][1]) ** 2
                )
                ** 0.5
                for i in range(len(classic_path) - 1)
            ),
        )
    else:
        logger.warning("✗ ClassicGlobalPlanner: No path found")

    logger.info("\n✓ Planner comparison complete!")


if __name__ == "__main__":
    main()
