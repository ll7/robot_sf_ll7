"""Navigation utilities for robot_sf.

This package provides navigation-related functionality including:
- SVG map parsing and conversion
- Motion planning adapters for grid-based planners
- Global route management
- Obstacle definitions
- Probabilistic pedestrian prediction types and protocol
- Pedestrian uncertainty-envelope abstraction for conservative clearance
"""

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
    "ConformalInflationPolicy",
    "MotionPlanningGridConfig",
    "PedestrianUncertaintyEnvelope",
    "ProbabilisticPrediction",
    "ProbabilisticPredictor",
    "SpatialInflationPolicy",
    "TrajectoryDistribution",
    "count_obstacle_cells",
    "effective_pedestrian_radius",
    "envelope_diagnostics",
    "envelope_from_position",
    "get_obstacle_statistics",
    "linear_inflation_policy",
    "map_definition_to_motion_planning_grid",
    "visualize_grid",
]
