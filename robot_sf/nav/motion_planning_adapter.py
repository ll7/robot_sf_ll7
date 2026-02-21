"""Compatibility shim for classic motion-planning helpers.

The classic grid conversion and visualization helpers now live in
`robot_sf.planner.classic_global_planner` so planner logic is centralized.
This module keeps the historical import path stable.
"""

from robot_sf.planner.classic_global_planner import (
    ClassicPlanVisualizer,
    MotionPlanningGridConfig,
    count_obstacle_cells,
    get_obstacle_statistics,
    map_definition_to_motion_planning_grid,
    set_start_goal_on_grid,
    visualize_grid,
    visualize_path,
)

__all__ = [
    "ClassicPlanVisualizer",
    "MotionPlanningGridConfig",
    "count_obstacle_cells",
    "get_obstacle_statistics",
    "map_definition_to_motion_planning_grid",
    "set_start_goal_on_grid",
    "visualize_grid",
    "visualize_path",
]
