"""Navigation utilities for robot_sf.

This package provides navigation-related functionality including:
- SVG map parsing and conversion
- Motion planning adapters for grid-based planners
- Global route management
- Obstacle definitions
"""

from robot_sf.nav.motion_planning_adapter import (
    MotionPlanningGridConfig,
    count_obstacle_cells,
    get_obstacle_statistics,
    map_definition_to_motion_planning_grid,
    visualize_grid,
)

__all__ = [
    "MotionPlanningGridConfig",
    "count_obstacle_cells",
    "get_obstacle_statistics",
    "map_definition_to_motion_planning_grid",
    "visualize_grid",
]
