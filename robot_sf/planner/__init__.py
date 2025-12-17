"""Global planner package for SVG-based waypoint generation.

This package provides two global planning approaches:

1. VisibilityPlanner: Visibility graph-based planner for continuous path planning
   - Uses pyvisgraph for visibility graph construction
   - Operates on vector-based obstacle representations
   - Suitable for sparse environments with clear line-of-sight paths

2. ClassicGlobalPlanner: Grid-based planner using python_motion_planning
   - Uses rasterized grid representation
   - Supports algorithms like ThetaStar, A*
   - Better for dense environments or narrow passages
"""

from robot_sf.planner.classic_global_planner import (
    ClassicGlobalPlanner,
    ClassicPlannerConfig,
    PlanningError,
)
from robot_sf.planner.classic_planner_adapter import (
    PlannerActionAdapter,
    attach_classic_global_planner,
)
from robot_sf.planner.poi_sampler import POISampler
from robot_sf.planner.visibility_planner import (
    PlannerConfig,
    PlanningFailedError,
    VisibilityPlanner,
)
from robot_sf.planner.visualization import plot_global_plan, plot_visibility_graph

# Backwards compatibility aliases
GlobalPlanner = VisibilityPlanner

__all__ = [
    "ClassicGlobalPlanner",
    "ClassicPlannerConfig",
    "GlobalPlanner",
    "POISampler",
    "PlannerActionAdapter",
    "PlannerConfig",
    "PlanningError",
    "PlanningFailedError",
    "VisibilityPlanner",
    "attach_classic_global_planner",
    "plot_global_plan",
    "plot_visibility_graph",
]
