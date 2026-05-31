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
from robot_sf.planner.learned_risk_surface import (
    LocalRiskSurface,
    LocalRiskSurfaceSpec,
    RiskSurfacePlannerAdapter,
    RiskSurfaceUnavailable,
    attach_risk_surface_to_observation,
    build_local_risk_surface_spec,
    deterministic_pedestrian_risk_surface,
)
from robot_sf.planner.poi_sampler import POISampler
from robot_sf.planner.policy_stack_v1 import PolicyStackV1Adapter, PolicyStackV1Config
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, RiskDWAPlannerConfig
from robot_sf.planner.teb_commitment import TEBCommitmentConfig, TEBCommitmentPlannerAdapter
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
    "LocalRiskSurface",
    "LocalRiskSurfaceSpec",
    "POISampler",
    "PlannerActionAdapter",
    "PlannerConfig",
    "PlanningError",
    "PlanningFailedError",
    "PolicyStackV1Adapter",
    "PolicyStackV1Config",
    "RiskDWAPlannerAdapter",
    "RiskDWAPlannerConfig",
    "RiskSurfacePlannerAdapter",
    "RiskSurfaceUnavailable",
    "TEBCommitmentConfig",
    "TEBCommitmentPlannerAdapter",
    "VisibilityPlanner",
    "attach_classic_global_planner",
    "attach_risk_surface_to_observation",
    "build_local_risk_surface_spec",
    "deterministic_pedestrian_risk_surface",
    "plot_global_plan",
    "plot_visibility_graph",
]
