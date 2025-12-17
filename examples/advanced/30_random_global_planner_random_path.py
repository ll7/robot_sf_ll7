#!/usr/bin/env python3
"""Plan and visualize a random path on the uni campus map."""

from __future__ import annotations

from loguru import logger

from robot_sf.common import ensure_interactive_backend
from robot_sf.common.artifact_paths import get_artifact_category_path, get_repository_root
from robot_sf.common.logging import configure_logging
from robot_sf.nav.motion_planning_adapter import get_obstacle_statistics
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig

MAP_PATH = (
    get_repository_root()
    / "maps/obstacle_svg_maps/uni_campus_1350_obstacles_lake_traverse_with_simple_routes.svg"
)
"""SVG map file to load for the demo."""


def main() -> None:
    """Run the classic global planner with randomly sampled endpoints."""
    configure_logging(verbose=True)
    ensure_interactive_backend()

    map_def = convert_map(str(MAP_PATH))
    logger.info(
        "Loaded map: {w}x{h} with {obs} obstacles",
        w=map_def.width,
        h=map_def.height,
        obs=len(map_def.obstacles),
    )

    planner_config = ClassicPlannerConfig(
        cells_per_meter=1.0,
        inflate_radius_cells=3,
        algorithm="theta_star_v2",
    )
    planner = ClassicGlobalPlanner(map_def, config=planner_config)

    output_dir = get_artifact_category_path("plots")
    grid = planner.grid
    stats = get_obstacle_statistics(grid)
    logger.info(
        "Planning grid: {shape}, {obs} obstacle cells ({pct:.2f}%)",
        shape=grid.type_map.shape,
        obs=stats["obstacle_count"],
        pct=stats["obstacle_percentage"],
    )
    planner.visualize_grid(
        output_dir / "classic_planner_random_grid.png", title="Random Planner Grid"
    )

    seed = 69
    logger.info("Sampling random start and goal with seed={seed}", seed=seed)
    path_world, path_info, start, goal = planner.plan_random_path(
        seed=seed, algorithm="theta_star_v2"
    )

    if not path_world:
        logger.error("Planning failed for sampled points")
        return

    length = path_info.get("length") if isinstance(path_info, dict) else None
    length_str = f"{length:.2f} m" if isinstance(length, (int, float)) else "unknown"
    logger.info(
        "Random path: {start} -> {goal} with {n} waypoints (length≈{length})",
        start=start,
        goal=goal,
        n=len(path_world),
        length=length_str,
    )

    planner.visualize_path(
        path_world,
        output_dir / "classic_planner_random_path.png",
        title="Classic Planner Random Path",
        path_style="--",
        path_color="C3",
        linewidth=2.0,
        marker="x",
        path_info=path_info,
        show_expands=True,
    )

    logger.info("✓ Random path planning completed. Plots written to {output}", output=output_dir)


if __name__ == "__main__":
    main()
