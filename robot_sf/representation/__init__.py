"""Sensor-agnostic scenario representation adapters and projections."""

from robot_sf.representation.scenario_belief import (
    AgentProjectionDiff,
    BeliefSource,
    EntityBelief,
    Estimate2D,
    ProjectionDiff,
    ScenarioBelief,
    TrackedAgentMetadata,
    VisibilityState,
    compute_clear_tracking_metrics,
    compute_projection_diff,
    scenario_belief_from_simulator_oracle,
    scenario_belief_from_visibility_limited_simulator,
)

__all__ = [
    "AgentProjectionDiff",
    "BeliefSource",
    "EntityBelief",
    "Estimate2D",
    "ProjectionDiff",
    "ScenarioBelief",
    "TrackedAgentMetadata",
    "VisibilityState",
    "compute_clear_tracking_metrics",
    "compute_projection_diff",
    "scenario_belief_from_simulator_oracle",
    "scenario_belief_from_visibility_limited_simulator",
]
