"""Prediction helpers for planner-facing pedestrian state estimates."""

from importlib import import_module
from typing import Any

from robot_sf.prediction.goal_belief_contract import (
    ACTOR_FORBIDDEN_KEYS,
    ACTOR_UNITS,
    GOAL_BELIEF_SCHEMA_VERSION,
    HISTORY_ORDER,
    ActorObservationStep,
    ActorSpeedCapStatus,
    CensoringState,
    CoordinateFrame,
    ForceEstimate2D,
    GoalBeliefMode,
    GoalBeliefObservation,
    GoalBeliefSource,
    GoalBeliefV1,
    GoalCandidateKind,
    GoalCandidateProbability,
    ObservationMask,
    stable_config_hash,
)
from robot_sf.prediction.goal_intention import (
    CandidateGoal,
    GoalCandidate,
    GoalCandidateAvailability,
    GoalCandidateRole,
    GoalCandidateSet,
    GoalIntentionPosterior,
    GoalPosteriorConfig,
    HeadingGoalPosteriorConfig,
    candidate_goals_from_points,
    planner_goal_posterior_channel,
    planner_goal_posterior_channel_from_beliefs,
    planner_goal_posterior_channel_from_state,
    planner_goal_posterior_channel_unavailable,
    planner_oracle_goal_posterior_channel_from_state,
    update_goal_posterior,
    update_heading_goal_posterior,
)
from robot_sf.prediction.oracle_transition_trace import (
    ORACLE_TRANSITION_TRACE_SCHEMA_VERSION,
    SIMULATOR_TIMING_ORDER,
    SIMULATOR_TIMING_PROVENANCE,
    ControllerMutationFlags,
    DynamicsParameters,
    ExactInverseReason,
    ForceComponentOperationKind,
    ForceComponentRecord,
    ForceComponents,
    ForceOperationKind,
    ForceStageResult,
    ForceTimeRobotState,
    GoalChangeKind,
    OracleTransitionTraceV1,
    RobotForceState,
    SpeedCap,
    SpeedCapStatus,
    TransitionBoundary,
    TransitionBoundaryKind,
    validate_simulator_timing_order,
)

_TRACKER_GOAL_BELIEF_EXPORTS = {
    "TRACKER_GOAL_BELIEF_ADAPTER_SCHEMA_VERSION": "robot_sf.prediction.tracker_goal_belief_adapter",
    "TRACKER_GOAL_BELIEF_BLOCKER": "robot_sf.prediction.tracker_goal_belief_adapter",
    "TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY": "robot_sf.prediction.tracker_goal_belief_adapter",
    "TrackerGoalBeliefAdapter": "robot_sf.prediction.tracker_goal_belief_adapter",
    "TrackerGoalBeliefAdapterConfig": "robot_sf.prediction.tracker_goal_belief_adapter",
    "TrackerGoalBeliefChannel": "robot_sf.prediction.tracker_goal_belief_adapter",
}


def __getattr__(name: str) -> Any:
    """Lazily expose the tracker adapter without creating a sensor import cycle.

    Returns:
        The requested tracker adapter class, constant, or configuration type.
    """
    try:
        module_name = _TRACKER_GOAL_BELIEF_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [  # noqa: F822 - names are resolved lazily by __getattr__
    "ACTOR_FORBIDDEN_KEYS",
    "ACTOR_UNITS",
    "GOAL_BELIEF_SCHEMA_VERSION",
    "HISTORY_ORDER",
    "ORACLE_TRANSITION_TRACE_SCHEMA_VERSION",
    "SIMULATOR_TIMING_ORDER",
    "SIMULATOR_TIMING_PROVENANCE",
    "TRACKER_GOAL_BELIEF_ADAPTER_SCHEMA_VERSION",
    "TRACKER_GOAL_BELIEF_BLOCKER",
    "TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY",
    "ActorObservationStep",
    "ActorSpeedCapStatus",
    "CandidateGoal",
    "CensoringState",
    "ControllerMutationFlags",
    "CoordinateFrame",
    "DynamicsParameters",
    "ExactInverseReason",
    "ForceComponentOperationKind",
    "ForceComponentRecord",
    "ForceComponents",
    "ForceEstimate2D",
    "ForceOperationKind",
    "ForceStageResult",
    "ForceTimeRobotState",
    "GoalBeliefMode",
    "GoalBeliefObservation",
    "GoalBeliefSource",
    "GoalBeliefV1",
    "GoalCandidate",
    "GoalCandidateAvailability",
    "GoalCandidateKind",
    "GoalCandidateProbability",
    "GoalCandidateRole",
    "GoalCandidateSet",
    "GoalChangeKind",
    "GoalIntentionPosterior",
    "GoalPosteriorConfig",
    "HeadingGoalPosteriorConfig",
    "ObservationMask",
    "OracleTransitionTraceV1",
    "RobotForceState",
    "SpeedCap",
    "SpeedCapStatus",
    "TrackerGoalBeliefAdapter",
    "TrackerGoalBeliefAdapterConfig",
    "TrackerGoalBeliefChannel",
    "TransitionBoundary",
    "TransitionBoundaryKind",
    "candidate_goals_from_points",
    "planner_goal_posterior_channel",
    "planner_goal_posterior_channel_from_beliefs",
    "planner_goal_posterior_channel_from_state",
    "planner_goal_posterior_channel_unavailable",
    "planner_oracle_goal_posterior_channel_from_state",
    "stable_config_hash",
    "update_goal_posterior",
    "update_heading_goal_posterior",
    "validate_simulator_timing_order",
]
