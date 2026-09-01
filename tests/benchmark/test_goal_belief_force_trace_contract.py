"""Oracle/actor separation tests for the goal-belief contract (issue #8063)."""

from __future__ import annotations

import copy

import pytest

from robot_sf.prediction.goal_belief_contract import (
    ActorObservationStep,
    ForceEstimate2D,
    GoalBeliefObservation,
    GoalBeliefV1,
    GoalCandidateKind,
    GoalCandidateProbability,
    stable_config_hash,
)
from robot_sf.prediction.oracle_transition_trace import (
    SIMULATOR_TIMING_ORDER,
    DynamicsParameters,
    ForceComponents,
    GoalChangeKind,
    OracleTransitionTraceV1,
    SpeedCap,
    SpeedCapStatus,
    TransitionBoundary,
    TransitionBoundaryKind,
    validate_simulator_timing_order,
)

CONFIG_HASH = stable_config_hash({"contract_test": True, "version": 1})


def _actor_belief() -> GoalBeliefV1:
    """Build one actor record from observation-only values."""
    observation = GoalBeliefObservation(
        track_id="track-1",
        timestamp_s=0.0,
        step_index=0,
        config_hash=CONFIG_HASH,
        history_steps=(
            ActorObservationStep(
                timestamp_s=0.0,
                step_index=0,
                position_xy=(0.0, 0.0),
                velocity_xy=(1.0, 0.0),
            ),
        ),
        force_estimate=ForceEstimate2D(
            mean_xy=(0.1, 0.0),
            covariance_xy=((0.2, 0.0), (0.0, 0.2)),
        ),
        desired_velocity_xy=(1.0, 0.0),
        candidate_probabilities=(
            GoalCandidateProbability("candidate-a", GoalCandidateKind.ACTIVE_WAYPOINT, 0.8),
        ),
        unknown_candidate_probability=0.2,
        track_confidence=0.8,
    )
    return GoalBeliefV1.from_observation(observation)


def _trace(
    *, goal_before: tuple[float, float] = (0.0, 0.0), goal_after: tuple[float, float] = (2.0, 0.0)
) -> OracleTransitionTraceV1:
    """Build a synthetic trace with an explicit waypoint advance."""
    return OracleTransitionTraceV1(
        episode_id="episode-1",
        transition_id="episode-1:t0",
        transition_step_index=0,
        simulator_pedestrian_id="pysf-0",
        actor_track_id="track-1",
        backend="pysocialforce",
        pre_behavior=TransitionBoundary(
            boundary=TransitionBoundaryKind.PRE_BEHAVIOR,
            timestamp_s=0.0,
            step_index=0,
            position_xy=(0.0, 0.0),
            velocity_xy=(1.0, 0.0),
            active_goal_xy=goal_before,
            route_waypoint_index=0,
            goal_threshold_reached=True,
        ),
        post_behavior_pre_force=TransitionBoundary(
            boundary=TransitionBoundaryKind.POST_BEHAVIOR_PRE_FORCE,
            timestamp_s=0.0,
            step_index=0,
            position_xy=(0.0, 0.0),
            velocity_xy=(1.0, 0.0),
            active_goal_xy=goal_after,
            route_waypoint_index=1,
            goal_threshold_reached=False,
        ),
        post_integration=TransitionBoundary(
            boundary=TransitionBoundaryKind.POST_INTEGRATION,
            timestamp_s=0.1,
            step_index=1,
            position_xy=(0.1, 0.0),
            velocity_xy=(1.0, 0.0),
            active_goal_xy=goal_after,
            route_waypoint_index=1,
            goal_threshold_reached=False,
        ),
        force_components=ForceComponents(
            social_force_xy=(0.1, 0.0),
            goal_force_xy=(0.2, 0.0),
            total_force_xy=(0.3, 0.0),
        ),
        dynamics=DynamicsParameters(
            preferred_speed_mps=1.0,
            relaxation_time_s=0.5,
            desired_force_factor=2.0,
            goal_threshold_m=0.2,
            goal_threshold_reached=True,
        ),
        speed_cap=SpeedCap(SpeedCapStatus.APPLIED, max_speed_mps=1.2),
        goal_change_kind=GoalChangeKind.WAYPOINT_ADVANCE,
    )


def test_oracle_trace_round_trip_is_deterministic_and_timed() -> None:
    """The oracle trace records typed boundaries and canonical bytes."""
    trace = _trace()
    payload = trace.to_dict()

    assert payload["timing"]["order"] == list(SIMULATOR_TIMING_ORDER)
    assert payload["pre_behavior"]["active_goal_xy"] == [0.0, 0.0]
    assert payload["post_behavior_pre_force"]["active_goal_xy"] == [2.0, 0.0]
    assert OracleTransitionTraceV1.from_dict(copy.deepcopy(payload)).to_json() == trace.to_json()
    assert trace.content_digest == OracleTransitionTraceV1.from_dict(payload).content_digest


def test_randomized_oracle_goal_cannot_change_actor_bytes() -> None:
    """Two oracle truths paired with one observation produce identical actor serialization."""
    actor = _actor_belief()
    first = _trace(goal_before=(0.0, 0.0), goal_after=(2.0, 0.0))
    second = _trace(goal_before=(10.0, 5.0), goal_after=(-4.0, 8.0))

    assert actor.to_json() == _actor_belief().to_json()
    assert first.content_digest != second.content_digest
    assert "goal_before_behavior" not in actor.to_json()
    assert "goal_after_behavior" not in actor.to_json()


def test_timing_order_rejects_shifted_or_incomplete_records() -> None:
    """A one-step shift or omitted force stage cannot pass the timing contract."""
    with pytest.raises(ValueError, match="timing order mismatch"):
        validate_simulator_timing_order(SIMULATOR_TIMING_ORDER[:-1])

    shifted = _trace().to_dict()
    shifted["post_integration"]["step_index"] = 2
    with pytest.raises(ValueError, match="post_integration must be the next transition step"):
        OracleTransitionTraceV1.from_dict(shifted)


def test_reset_starts_a_new_episode_without_reusing_actor_linkage() -> None:
    """A fresh episode can omit the prior actor linkage instead of carrying stale state."""
    trace = _trace()
    reset_trace = copy.deepcopy(trace.to_dict())
    reset_trace["episode_id"] = "episode-2"
    reset_trace["transition_id"] = "episode-2:t0"
    reset_trace["actor_track_id"] = None

    parsed = OracleTransitionTraceV1.from_dict(reset_trace)
    assert parsed.episode_id == "episode-2"
    assert parsed.actor_track_id is None
