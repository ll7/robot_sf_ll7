"""Navigation utilities for robot_sf.

This package provides navigation-related functionality including:
- SVG map parsing and conversion
- Motion planning adapters for grid-based planners
- Global route management
- Obstacle definitions
- Probabilistic pedestrian prediction types and protocol
- Pedestrian uncertainty-envelope abstraction for conservative clearance
- Deterministic biased route condition generator and fixture evaluation
"""

from robot_sf.nav.biased_route_generator import (
    BiasedRouteConfig,
    BiasedRouteResult,
    CanonicalFixtureTopology,
    build_corridor_fixture,
    build_crossing_fixture,
    build_doorway_fixture,
    evaluate_route_observability_sequence,
    generate_biased_route,
    generate_corridor_homotopy_routes,
    generate_doorway_homotopy_routes,
    rasterize_route_to_grid,
)
from robot_sf.nav.motion_planning_adapter import (
    MotionPlanningGridConfig,
    count_obstacle_cells,
    get_obstacle_statistics,
    map_definition_to_motion_planning_grid,
    visualize_grid,
)
from robot_sf.nav.predictive_types import (
    ProbabilisticPrediction,
    ProbabilisticPredictor,
    TrajectoryDistribution,
)
from robot_sf.nav.proxemic_costmap import (
    ProxemicCostmapConfig,
    build_proxemic_costmap_config,
    config_hash,
    proxemic_cost_at_points,
)
from robot_sf.nav.uncertainty_envelope import (
    DEFAULT_ALPHA_MPS,
    ENVELOPE_SCHEMA_VERSION,
    ConformalInflationPolicy,
    PedestrianUncertaintyEnvelope,
    SpatialInflationPolicy,
    effective_pedestrian_radius,
    envelope_diagnostics,
    envelope_from_position,
    linear_inflation_policy,
)

__all__ = [
    "DEFAULT_ALPHA_MPS",
    "ENVELOPE_SCHEMA_VERSION",
    "BiasedRouteConfig",
    "BiasedRouteResult",
    "CanonicalFixtureTopology",
    "ConformalInflationPolicy",
    "MotionPlanningGridConfig",
    "PedestrianUncertaintyEnvelope",
    "ProbabilisticPrediction",
    "ProbabilisticPredictor",
    "ProxemicCostmapConfig",
    "SpatialInflationPolicy",
    "TrajectoryDistribution",
    "build_corridor_fixture",
    "build_crossing_fixture",
    "build_doorway_fixture",
    "build_proxemic_costmap_config",
    "config_hash",
    "count_obstacle_cells",
    "effective_pedestrian_radius",
    "envelope_diagnostics",
    "envelope_from_position",
    "evaluate_route_observability_sequence",
    "generate_biased_route",
    "generate_corridor_homotopy_routes",
    "generate_doorway_homotopy_routes",
    "get_obstacle_statistics",
    "linear_inflation_policy",
    "map_definition_to_motion_planning_grid",
    "proxemic_cost_at_points",
    "rasterize_route_to_grid",
    "visualize_grid",
]
