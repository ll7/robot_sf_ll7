"""Action-conditioned online collision-risk API and baselines (issue #5444).

This package exposes a planner-agnostic contract for estimating the probability
of robot-pedestrian contact within a horizon, conditioned on a candidate robot
action, while keeping deterministic warnings, model probabilities, and formal
hard-guard authority explicitly separate.

Public surface:

- :class:`~robot_sf.research.collision_risk.estimators.RiskEstimatorConfig`
- :class:`~robot_sf.research.collision_risk.estimators.CandidateAction`
- :func:`~robot_sf.research.collision_risk.estimators.action_from_constant_velocity`
- :func:`~robot_sf.research.collision_risk.estimators.estimate_action_conditioned_risk`
- :class:`~robot_sf.research.collision_risk.schema.ActionConditionedRiskEstimate`

Status: API + baseline fixture evidence, not a calibrated benchmark risk claim.
Hard guards remain authoritative; no ``safe`` label is emitted.
"""

from __future__ import annotations

from robot_sf.research.collision_risk.estimators import (
    ESTIMATOR_ID,
    FORECAST_MODEL_ID,
    GEOMETRY_VERSION,
    CandidateAction,
    CollisionRiskInputError,
    RiskEstimatorConfig,
    action_from_constant_velocity,
    estimate_action_conditioned_risk,
)
from robot_sf.research.collision_risk.schema import (
    DETERMINISTIC_FIELD_LABEL,
    GUARD_AUTHORITY_NOTE,
    RISK_SCHEMA_VERSION,
    ActionConditionedRiskEstimate,
    DeterministicRiskFields,
    LatencySummary,
    PerActorContribution,
    RiskProvenance,
    RiskSchemaError,
    UncertaintyState,
    latency_summary_from_samples,
)

__all__ = [
    "DETERMINISTIC_FIELD_LABEL",
    "ESTIMATOR_ID",
    "FORECAST_MODEL_ID",
    "GEOMETRY_VERSION",
    "GUARD_AUTHORITY_NOTE",
    "RISK_SCHEMA_VERSION",
    "ActionConditionedRiskEstimate",
    "CandidateAction",
    "CollisionRiskInputError",
    "DeterministicRiskFields",
    "LatencySummary",
    "PerActorContribution",
    "RiskEstimatorConfig",
    "RiskProvenance",
    "RiskSchemaError",
    "UncertaintyState",
    "action_from_constant_velocity",
    "estimate_action_conditioned_risk",
    "latency_summary_from_samples",
]
