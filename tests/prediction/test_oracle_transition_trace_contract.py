"""Oracle/actor separation tests for the goal-belief contract (issue #8063)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import NoReturn

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
    ControllerMutationFlags,
    DynamicsParameters,
    ExactInverseReason,
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

CONFIG_HASH = stable_config_hash({"contract_test": True, "version": 1})


class _PrivilegedAccessTrap:
    """Descriptor that fails if a narrowed actor view exposes oracle state."""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    def __get__(self, _instance: object, _owner: type[object]) -> NoReturn:
        raise AssertionError(f"actor producer accessed privileged field {self.field_name}")


class _PhysicalActorView:
    """Test-only actor view with no oracle storage and trapping oracle names."""

    __slots__ = ("observation",)

    goal_before_behavior = _PrivilegedAccessTrap("goal_before_behavior")
    goal_after_behavior = _PrivilegedAccessTrap("goal_after_behavior")
    route_truth = _PrivilegedAccessTrap("route_truth")
    waypoint_truth = _PrivilegedAccessTrap("waypoint_truth")
    simulator_pedestrian_id = _PrivilegedAccessTrap("simulator_pedestrian_id")
    oracle_view = _PrivilegedAccessTrap("oracle_view")

    def __init__(self, observation: GoalBeliefObservation) -> None:
        self.observation = observation


@dataclass(frozen=True, slots=True)
class _SyntheticTransition:
    """One synthetic transition split into physically separate actor/oracle views."""

    actor_view: _PhysicalActorView
    oracle_view: OracleTransitionTraceV1


class _SyntheticLinkageTracker:
    """Test-only tracker proving reset generations own actor/oracle linkage."""

    def __init__(self) -> None:
        self.epoch_id = "epoch-1"
        self._links: dict[tuple[str, str], str] = {}

    def link(self, track_id: str, simulator_pedestrian_id: str) -> tuple[str, str]:
        """Register and return the current-generation linkage key."""
        key = (self.epoch_id, track_id)
        self._links[key] = simulator_pedestrian_id
        return key

    def reset(self) -> None:
        """Start a new tracking generation and discard old links."""
        self._links.clear()
        self.epoch_id = "epoch-2"

    def resolve(self, key: tuple[str, str]) -> str:
        """Resolve only a linkage owned by the current tracking generation."""
        if key[0] != self.epoch_id:
            raise ValueError("stale actor linkage belongs to a prior tracking epoch")
        try:
            return self._links[key]
        except KeyError as exc:
            raise ValueError("actor linkage is unavailable in the current tracking epoch") from exc


def _actor_observation() -> GoalBeliefObservation:
    """Build one narrowed actor observation without any privileged transition values."""
    return GoalBeliefObservation(
        track_id="track-1",
        tracking_epoch_id="epoch-1",
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


def _actor_belief() -> GoalBeliefV1:
    """Build one actor record from observation-only values."""
    return GoalBeliefV1.from_observation(_actor_observation())


def _trace(
    *,
    goal_before: tuple[float, float] = (0.0, 0.0),
    goal_after: tuple[float, float] = (2.0, 0.0),
    simulator_pedestrian_id: str = "pysf-0",
    actor_track_id: str | None = "track-1",
    actor_tracking_epoch_id: str | None = "epoch-1",
) -> OracleTransitionTraceV1:
    """Build a synthetic trace with an explicit waypoint advance."""
    return OracleTransitionTraceV1(
        episode_id="episode-1",
        transition_id="episode-1:t0",
        transition_step_index=0,
        simulator_pedestrian_id=simulator_pedestrian_id,
        actor_track_id=actor_track_id,
        actor_tracking_epoch_id=actor_tracking_epoch_id,
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
            force_time_robot_state=ForceTimeRobotState(
                (RobotForceState(robot_index=0, position_xy=(5.0, 5.0), heading_rad=0.0),)
            ),
            mutation_flags=ControllerMutationFlags(goal_redirected=True),
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
            registry_total_force_xy=(0.3, 0.0),
            final_pre_cap_force_xy=(0.3, 0.0),
            uncapped_velocity_xy=(1.03, 0.0),
            applied_velocity_xy=(1.0, 0.0),
        ),
        dynamics=DynamicsParameters(
            preferred_speed_mps=1.0,
            relaxation_time_s=0.5,
            desired_force_factor=2.0,
            goal_threshold_m=0.2,
            goal_threshold_reached=True,
        ),
        speed_cap=SpeedCap(
            SpeedCapStatus.APPLIED,
            max_speed_mps=1.0,
            uncapped_speed_mps=1.03,
            applied_speed_mps=1.0,
        ),
        goal_change_kind=GoalChangeKind.WAYPOINT_ADVANCE,
    )


def _synthetic_transition(
    *,
    goal_before: tuple[float, float],
    goal_after: tuple[float, float],
    simulator_pedestrian_id: str,
) -> _SyntheticTransition:
    """Build one transition with separate actor and oracle views."""
    return _SyntheticTransition(
        actor_view=_PhysicalActorView(_actor_observation()),
        oracle_view=_trace(
            goal_before=goal_before,
            goal_after=goal_after,
            simulator_pedestrian_id=simulator_pedestrian_id,
        ),
    )


def _produce_actor_belief(actor_view: _PhysicalActorView) -> GoalBeliefV1:
    """Run the same actor producer using only the narrowed actor view."""
    return GoalBeliefV1.from_observation(actor_view.observation)


def test_oracle_trace_round_trip_is_deterministic_and_timed() -> None:
    """The oracle trace records typed boundaries and canonical bytes."""
    trace = _trace()
    payload = trace.to_dict()

    assert payload["timing"]["order"] == list(SIMULATOR_TIMING_ORDER)
    assert payload["pre_behavior"]["active_goal_xy"] == [0.0, 0.0]
    assert payload["post_behavior_pre_force"]["active_goal_xy"] == [2.0, 0.0]
    assert payload["post_behavior_pre_force"]["force_time_robot_state"]["robot_states"]
    assert payload["force_components"]["registry_total_force_xy"] == [0.3, 0.0]
    assert payload["force_components"]["residual_operation"]["operation_kind"] == "not_applied"
    assert payload["force_components"]["model_variant_operation"]["operation_kind"] == "not_applied"
    assert payload["force_components"]["final_pre_cap_force_xy"] == [0.3, 0.0]
    assert payload["force_components"]["uncapped_velocity_xy"] == [1.03, 0.0]
    assert payload["force_components"]["applied_velocity_xy"] == [1.0, 0.0]
    assert payload["exact_inverse_eligible"] is False
    assert payload["exact_inverse_reasons"] == ["force_stage_uninstrumented"]
    assert payload["speed_cap"]["speed_cap_active"] is True
    assert OracleTransitionTraceV1.from_dict(copy.deepcopy(payload)).to_json() == trace.to_json()
    assert trace.content_digest == OracleTransitionTraceV1.from_dict(payload).content_digest


def test_randomized_oracle_goal_cannot_change_actor_bytes() -> None:
    """One actor producer cannot read randomized oracle fields from the same fixture."""
    first = _synthetic_transition(
        goal_before=(0.0, 0.0),
        goal_after=(2.0, 0.0),
        simulator_pedestrian_id="pysf-0",
    )
    second = _synthetic_transition(
        goal_before=(10.0, 5.0),
        goal_after=(-4.0, 8.0),
        simulator_pedestrian_id="pysf-17",
    )

    first_actor = _produce_actor_belief(first.actor_view)
    second_actor = _produce_actor_belief(second.actor_view)

    assert first_actor.to_json() == second_actor.to_json()
    assert first.oracle_view.content_digest != second.oracle_view.content_digest
    assert "goal_before_behavior" not in first_actor.to_json()
    assert "goal_after_behavior" not in first_actor.to_json()

    with pytest.raises(AssertionError, match="privileged field goal_after_behavior"):
        _ = first.actor_view.goal_after_behavior


def test_timing_order_rejects_shifted_or_incomplete_records() -> None:
    """A one-step shift or omitted force stage cannot pass the timing contract."""
    with pytest.raises(ValueError, match="timing order mismatch"):
        validate_simulator_timing_order(SIMULATOR_TIMING_ORDER[:-1])

    shifted = _trace().to_dict()
    shifted["post_integration"]["step_index"] = 2
    with pytest.raises(ValueError, match="post_integration must be the next transition step"):
        OracleTransitionTraceV1.from_dict(shifted)


def test_reset_starts_a_new_tracking_epoch_without_reusing_actor_linkage() -> None:
    """A reused track ID cannot resolve a prior episode's simulator linkage."""
    tracker = _SyntheticLinkageTracker()
    old_key = tracker.link("track-1", "pysf-0")
    old_trace = _trace(actor_tracking_epoch_id=old_key[0])
    assert (old_trace.actor_tracking_epoch_id, old_trace.actor_track_id) == old_key
    assert tracker.resolve(old_key) == "pysf-0"

    tracker.reset()
    new_observation = replace(_actor_observation(), tracking_epoch_id=tracker.epoch_id)
    new_belief = GoalBeliefV1.from_observation(new_observation)
    new_key = tracker.link(new_belief.track_id, "pysf-1")
    new_trace = _trace(
        simulator_pedestrian_id="pysf-1",
        actor_tracking_epoch_id=new_key[0],
    )

    with pytest.raises(ValueError, match="stale actor linkage"):
        tracker.resolve(old_key)
    assert (new_trace.actor_tracking_epoch_id, new_trace.actor_track_id) == new_key
    assert tracker.resolve(new_key) == "pysf-1"
    assert new_belief.tracking_epoch_id != old_key[0]

    reset_trace = copy.deepcopy(old_trace.to_dict())
    reset_trace["episode_id"] = "episode-2"
    reset_trace["transition_id"] = "episode-2:t0"
    reset_trace["actor_track_id"] = None
    reset_trace["actor_tracking_epoch_id"] = None
    parsed = OracleTransitionTraceV1.from_dict(reset_trace)
    assert parsed.episode_id == "episode-2"
    assert parsed.actor_track_id is None
    assert parsed.actor_tracking_epoch_id is None


def test_oracle_linkage_requires_a_tracking_epoch_pair() -> None:
    """A bare actor track ID cannot be serialized as oracle linkage."""
    with pytest.raises(ValueError, match="provided as a pair"):
        _trace(actor_tracking_epoch_id=None)


def test_unmodeled_controller_mutation_requires_an_inverse_reason() -> None:
    """Hold, respawn, and population mutations fail closed without explicit modeling."""
    trace = _trace()
    with pytest.raises(ValueError, match="each unmodeled controller mutation"):
        OracleTransitionTraceV1.from_dict(
            {
                **trace.to_dict(),
                "post_behavior_pre_force": {
                    **trace.to_dict()["post_behavior_pre_force"],
                    "mutation_flags": {
                        **trace.to_dict()["post_behavior_pre_force"]["mutation_flags"],
                        "hold_velocity_reset": True,
                    },
                },
            }
        )

    eligible = replace(
        trace,
        exact_inverse_eligible=True,
        exact_inverse_reasons=(),
    )
    assert eligible.exact_inverse_eligible is True
    assert eligible.exact_inverse_reasons == ()
    assert ExactInverseReason.FORCE_STAGE_UNINSTRUMENTED not in eligible.exact_inverse_reasons


def test_force_stage_requires_a_result_for_every_applied_operation() -> None:
    """Additive, replacement, and transformed stages cannot omit their recorded output."""
    with pytest.raises(ValueError, match="require delta_force_xy and result_force_xy"):
        ForceStageResult(
            operation_kind=ForceOperationKind.ADDITIVE,
            operation="residual",
            delta_force_xy=(0.1, 0.0),
        )
    with pytest.raises(ValueError, match="require result_force_xy only"):
        ForceStageResult(
            operation_kind=ForceOperationKind.REPLACEMENT,
            operation="replace",
        )
    with pytest.raises(ValueError, match="require result_force_xy only"):
        ForceStageResult(
            operation_kind=ForceOperationKind.TRANSFORMED,
            operation="transform",
        )


def test_residual_replacement_and_transform_are_folded_into_final_force() -> None:
    """Residual replacement and transform stages validate their own post-stage outputs."""
    replacement = ForceComponents(
        registry_total_force_xy=(1.0, 2.0),
        residual_operation=ForceStageResult(
            operation_kind=ForceOperationKind.REPLACEMENT,
            operation="replace_residual",
            result_force_xy=(4.0, 5.0),
        ),
        final_pre_cap_force_xy=(4.0, 5.0),
    )
    transformed = ForceComponents(
        registry_total_force_xy=(1.0, 2.0),
        residual_operation=ForceStageResult(
            operation_kind=ForceOperationKind.TRANSFORMED,
            operation="transform_residual",
            result_force_xy=(-2.0, 3.0),
        ),
        final_pre_cap_force_xy=(-2.0, 3.0),
    )

    assert replacement.final_pre_cap_force_xy == (4.0, 5.0)
    assert transformed.final_pre_cap_force_xy == (-2.0, 3.0)


def test_force_stage_composition_uses_declared_residual_then_model_order() -> None:
    """The model stage consumes the residual stage's recorded result, not the registry total."""
    components = ForceComponents(
        registry_total_force_xy=(1.0, 1.0),
        residual_operation=ForceStageResult(
            operation_kind=ForceOperationKind.REPLACEMENT,
            operation="replace_residual",
            result_force_xy=(2.0, 3.0),
        ),
        model_variant_operation=ForceStageResult(
            operation_kind=ForceOperationKind.ADDITIVE,
            operation="add_model_residual",
            delta_force_xy=(0.5, 0.5),
            result_force_xy=(2.5, 3.5),
        ),
        final_pre_cap_force_xy=(2.5, 3.5),
    )

    assert components.final_pre_cap_force_xy == (2.5, 3.5)


def test_exact_inverse_rejects_an_incomplete_applied_stage_payload() -> None:
    """An exact-inverse payload cannot bypass the applied-stage result requirement."""
    payload = _trace().to_dict()
    payload["exact_inverse_eligible"] = True
    payload["exact_inverse_reasons"] = []
    payload["force_components"]["residual_operation"] = {
        "operation_kind": "replacement",
        "operation": "replace_residual",
        "delta_force_xy": None,
        "result_force_xy": None,
    }

    with pytest.raises(ValueError, match="require result_force_xy only"):
        OracleTransitionTraceV1.from_dict(payload)


def test_exact_inverse_rejects_unknown_force_stage_operations() -> None:
    """An unknown stage remains ineligible even when all aggregate fields are present."""
    trace = _trace()
    components = replace(
        trace.force_components,
        model_variant_operation=ForceStageResult(
            operation_kind=ForceOperationKind.UNKNOWN,
            operation="opaque_model_stage",
        ),
    )

    with pytest.raises(ValueError, match="known force-stage operations"):
        replace(
            trace,
            force_components=components,
            exact_inverse_eligible=True,
            exact_inverse_reasons=(),
        )
