"""Contract tests for the two-phase maneuver commitment state machine."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from robot_sf.nav.global_route import RouteGeometry
from robot_sf.nav.predictive_types import MultimodalPrediction
from robot_sf.planner.maneuver_candidates import (
    ManeuverCandidate,
    ManeuverId,
    ManeuverPortfolioConfig,
    generate_maneuver_candidates,
)
from robot_sf.planner.maneuver_commitment import (
    CommitmentObservation,
    ManeuverCommitmentConfig,
    ManeuverCommitmentManager,
    ManeuverState,
    TransitionReason,
)
from robot_sf.planner.multimodal_trajectory_arbitrator import (
    CandidateEvaluation,
    arbitrate_multimodal_trajectories,
)
from robot_sf.planner.scenario_belief_adapter import (
    IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION,
    BeliefAwarePlannerInput,
    IdentitySafeProjectionDiagnostics,
)
from robot_sf.research.collision_risk import RiskEstimatorConfig
from robot_sf.robot.dynamics import RobotDynamicsState

DT_S = 0.1
HORIZON = 20


def _route(angle: float = 0.0) -> RouteGeometry:
    direction = np.array([math.cos(angle), math.sin(angle)])
    return RouteGeometry((tuple(direction * 0.0), tuple(direction * 20.0)))


def _candidates(angle: float = 0.0) -> tuple[ManeuverCandidate, ...]:
    state = RobotDynamicsState(
        x=0.0,
        y=0.0,
        heading=angle,
        linear_speed=0.8,
        angular_speed=0.0,
    )
    return tuple(
        generate_maneuver_candidates(
            _route(angle),
            state,
            config=ManeuverPortfolioConfig(dt_s=DT_S, horizon_steps=HORIZON),
            static_geometry=(),
            timestamp_s=0.0,
        )
    )


def _candidate_for(
    candidates: tuple[ManeuverCandidate, ...], state: ManeuverState
) -> ManeuverCandidate:
    maneuver = {
        ManeuverState.FOLLOW: ManeuverId.ROUTE_FOLLOW,
        ManeuverState.PASS_LEFT: ManeuverId.PASS_LEFT,
        ManeuverState.PASS_RIGHT: ManeuverId.PASS_RIGHT,
        ManeuverState.YIELD: ManeuverId.YIELD_CREEP,
        ManeuverState.STOP: ManeuverId.CONTROLLED_STOP,
    }[state]
    return next(candidate for candidate in candidates if candidate.maneuver is maneuver)


def _evaluation(candidate: ManeuverCandidate, values: dict[str, object] | None = None):
    values = values or {}
    hard_feasible = bool(values.get("hard_feasible", True))
    braking_feasible = bool(values.get("braking_feasible", True))
    risk_limit_ok = bool(values.get("risk_limit_ok", True))
    risk_bucket = int(values.get("risk_bucket", 0))
    progress = float(values.get("route_progress_m", 0.0))
    liveness = float(values.get("liveness_cost", 0.0))
    comfort = float(values.get("comfort_cost", 0.0))
    switch_cost = float(values.get("switch_cost", 0.0))
    conservative_risk = float(values.get("conservative_dynamic_risk", risk_bucket / 10.0))
    raw_risk = float(values.get("raw_dynamic_risk", conservative_risk))
    rejection = None if hard_feasible and risk_limit_ok else "fixture rejected"
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        maneuver=candidate.maneuver,
        hard_feasible=hard_feasible,
        hard_reasons=() if hard_feasible else ("fixture hard failure",),
        raw_dynamic_risk=raw_risk,
        conservative_dynamic_risk=conservative_risk,
        max_track_risk=raw_risk,
        risk_bucket=risk_bucket,
        critical_track_id=None,
        critical_mode_id=None,
        critical_step=None,
        route_progress_m=progress,
        liveness_cost=liveness,
        comfort_cost=comfort,
        switch_cost=switch_cost,
        decision_key=(
            0 if hard_feasible else 1,
            0 if braking_feasible else 1,
            0 if risk_limit_ok else 1,
            risk_bucket,
            0.0,
            0.0,
            conservative_risk,
            raw_risk,
            raw_risk,
            -progress,
            liveness,
            comfort,
            switch_cost,
            candidate.candidate_id,
        ),
        braking_feasible=braking_feasible,
        risk_limit_ok=risk_limit_ok,
        rejection_reason=rejection,
    )


def _evaluations(
    candidates: tuple[ManeuverCandidate, ...],
    *,
    by_candidate: dict[str, dict[str, object]] | None = None,
    by_state: dict[ManeuverState, dict[str, object]] | None = None,
) -> tuple[CandidateEvaluation, ...]:
    by_candidate = by_candidate or {}
    by_state = by_state or {}
    return tuple(
        _evaluation(
            candidate,
            {
                **by_state.get(_state_for(candidate), {}),
                **by_candidate.get(candidate.candidate_id, {}),
            },
        )
        for candidate in candidates
    )


def _state_for(candidate: ManeuverCandidate) -> ManeuverState:
    return {
        ManeuverId.ROUTE_FOLLOW: ManeuverState.FOLLOW,
        ManeuverId.PASS_LEFT: ManeuverState.PASS_LEFT,
        ManeuverId.PASS_RIGHT: ManeuverState.PASS_RIGHT,
        ManeuverId.YIELD_CREEP: ManeuverState.YIELD,
        ManeuverId.CONTROLLED_STOP: ManeuverState.STOP,
    }[candidate.maneuver]


def _context(
    manager: ManeuverCommitmentManager,
    candidates: tuple[ManeuverCandidate, ...],
    evaluations: tuple[CandidateEvaluation, ...],
    step: int,
    observation: CommitmentObservation | None = None,
):
    return manager.context_for_candidates(
        step=step,
        candidates=candidates,
        evaluations_without_switch_cost=evaluations,
        observation=observation,
    )


def _select(
    manager: ManeuverCommitmentManager,
    context,
    candidate: ManeuverCandidate | None,
    evaluations: tuple[CandidateEvaluation, ...],
):
    allowed = set(context.allowed_candidate_ids)
    return manager.observe_selection(
        step=context.step,
        selected_candidate_id=candidate.candidate_id if candidate else None,
        evaluations=tuple(item for item in evaluations if item.candidate_id in allowed),
        context=context,
    )


def _rank_candidates(
    candidates: tuple[ManeuverCandidate, ...],
    evaluations: tuple[CandidateEvaluation, ...],
    allowed_candidate_ids: tuple[str, ...] | None = None,
) -> ManeuverCandidate | None:
    """Apply #8062's deterministic decision-key ordering to fixture evaluations."""
    allowed = set(allowed_candidate_ids) if allowed_candidate_ids is not None else None
    eligible = [
        evaluation
        for evaluation in evaluations
        if evaluation.eligible and (allowed is None or evaluation.candidate_id in allowed)
    ]
    if not eligible:
        return None
    selected = min(eligible, key=lambda evaluation: evaluation.decision_key)
    return next(
        candidate for candidate in candidates if candidate.candidate_id == selected.candidate_id
    )


def _apply_switch_costs(
    evaluations: tuple[CandidateEvaluation, ...], switch_costs: dict[str, float]
) -> tuple[CandidateEvaluation, ...]:
    """Represent #8062's final decision key after commitment costs are supplied."""
    return tuple(
        replace(
            evaluation,
            switch_cost=switch_costs[evaluation.candidate_id],
            decision_key=(
                *evaluation.decision_key[:-2],
                switch_costs[evaluation.candidate_id],
                evaluation.candidate_id,
            ),
        )
        for evaluation in evaluations
    )


def _prime(
    state: ManeuverState,
    *,
    config: ManeuverCommitmentConfig | None = None,
    observation: CommitmentObservation | None = None,
):
    candidates = _candidates()
    manager = ManeuverCommitmentManager(config)
    evaluations = _evaluations(candidates)
    context = _context(manager, candidates, evaluations, 0, observation)
    transition = _select(manager, context, _candidate_for(candidates, state), evaluations)
    assert transition.transition_reason == TransitionReason.INITIAL_SELECTION.value
    return manager, candidates


def test_first_valid_selection_initializes_expected_semantic_state() -> None:
    manager, _ = _prime(ManeuverState.PASS_LEFT)
    assert manager.state.active is ManeuverState.PASS_LEFT
    assert manager.state.activated_step == 0
    assert manager.state.transition_count == 1


def test_repeated_same_maneuver_does_not_increment_transition_count() -> None:
    manager, candidates = _prime(ManeuverState.FOLLOW)
    evaluations = _evaluations(candidates)
    context = _context(manager, candidates, evaluations, 1)
    transition = _select(
        manager, context, _candidate_for(candidates, ManeuverState.FOLLOW), evaluations
    )
    assert not transition.transition_performed
    assert transition.transition_reason == TransitionReason.SAME_MANEUVER.value
    assert manager.state.transition_count == 1


def test_alternating_marginal_pass_scores_hold_during_dwell_and_margin() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 3, ManeuverState.PASS_RIGHT: 3},
        utility_hysteresis_margin=0.05,
    )
    manager, candidates = _prime(ManeuverState.PASS_LEFT, config=config)
    left = _candidate_for(candidates, ManeuverState.PASS_LEFT)
    right = _candidate_for(candidates, ManeuverState.PASS_RIGHT)
    for step, preferred in ((1, "right"), (2, "left"), (3, "right")):
        values = _evaluations(
            candidates,
            by_candidate={
                (right if preferred == "right" else left).candidate_id: {"route_progress_m": 1.01},
                (left if preferred == "right" else right).candidate_id: {"route_progress_m": 1.0},
            },
        )
        context = _context(manager, candidates, values, step)
        assert context.allowed_candidate_ids == tuple(
            sorted(
                candidate.candidate_id
                for candidate in candidates
                if _state_for(candidate) in {ManeuverState.PASS_LEFT, ManeuverState.STOP}
            )
        )
        _select(manager, context, left, values)
    assert manager.state.active is ManeuverState.PASS_LEFT
    assert manager.state.transition_count == 1


def test_opposite_pass_side_wins_after_dwell_only_when_margin_is_exceeded() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 2},
        utility_hysteresis_margin=0.05,
    )
    manager, candidates = _prime(ManeuverState.PASS_LEFT, config=config)
    values = _evaluations(
        candidates,
        by_state={
            ManeuverState.PASS_LEFT: {"route_progress_m": 1.0},
            ManeuverState.PASS_RIGHT: {"route_progress_m": 2.0},
        },
    )
    context = _context(manager, candidates, values, 2)
    assert context.release_reason == TransitionReason.NORMAL_ADVANTAGE.value
    assert set(context.allowed_candidate_ids) == {
        candidate.candidate_id for candidate in candidates
    }
    transition = _select(
        manager, context, _candidate_for(candidates, ManeuverState.PASS_RIGHT), values
    )
    assert transition.transition_reason == TransitionReason.NORMAL_ADVANTAGE.value
    assert manager.state.active is ManeuverState.PASS_RIGHT


def test_candidate_identity_change_inside_same_maneuver_is_not_full_switch() -> None:
    manager, candidates = _prime(ManeuverState.FOLLOW)
    follow = [c for c in candidates if c.maneuver is ManeuverId.ROUTE_FOLLOW]
    assert len(follow) >= 2
    evaluations = _evaluations(candidates)
    context = _context(manager, candidates, evaluations, 1)
    assert context.switch_costs[follow[1].candidate_id] > 0.0
    transition = _select(manager, context, follow[1], evaluations)
    assert not transition.transition_performed
    assert manager.state.active is ManeuverState.FOLLOW
    assert manager.state.transition_count == 1


def test_active_hard_infeasibility_releases_before_dwell_expiry() -> None:
    manager, candidates = _prime(
        ManeuverState.PASS_LEFT,
        config=ManeuverCommitmentConfig(min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 10}),
    )
    values = _evaluations(candidates, by_state={ManeuverState.PASS_LEFT: {"hard_feasible": False}})
    context = _context(manager, candidates, values, 1)
    assert context.release_reason == TransitionReason.ACTIVE_HARD_INFEASIBLE.value
    assert set(context.allowed_candidate_ids) == {
        candidate.candidate_id for candidate in candidates
    }


def test_strictly_safer_risk_class_releases_before_dwell_expiry() -> None:
    manager, candidates = _prime(
        ManeuverState.FOLLOW,
        config=ManeuverCommitmentConfig(min_dwell_steps_by_state={ManeuverState.FOLLOW: 10}),
    )
    values = _evaluations(
        candidates,
        by_state={
            ManeuverState.FOLLOW: {"risk_bucket": 2},
            ManeuverState.PASS_LEFT: {"risk_bucket": 1},
        },
    )
    context = _context(manager, candidates, values, 1)
    assert context.release_reason == TransitionReason.SAFER_RISK_CLASS.value
    assert set(context.allowed_candidate_ids) == {
        candidate.candidate_id for candidate in candidates
    }


def test_controlled_stop_remains_allowed_under_active_commitment() -> None:
    manager, candidates = _prime(ManeuverState.PASS_LEFT)
    values = _evaluations(candidates)
    context = _context(manager, candidates, values, 1)
    stop = _candidate_for(candidates, ManeuverState.STOP)
    assert stop.candidate_id in context.allowed_candidate_ids
    assert context.switch_costs[stop.candidate_id] == 0.0


def test_controlled_stop_selection_during_normal_hold_has_override_reason() -> None:
    manager, candidates = _prime(ManeuverState.PASS_LEFT)
    evaluations = _evaluations(
        candidates,
        by_state={
            ManeuverState.PASS_LEFT: {"route_progress_m": 1.0},
            ManeuverState.STOP: {"route_progress_m": 1.01},
        },
    )
    context = _context(manager, candidates, evaluations, 1)
    assert context.release_reason is None
    selected = _rank_candidates(candidates, evaluations, context.allowed_candidate_ids)
    assert selected is _candidate_for(candidates, ManeuverState.STOP)

    transition = _select(manager, context, selected, evaluations)

    assert transition.transition_performed
    assert transition.transition_reason == TransitionReason.CONTROLLED_STOP_OVERRIDE.value
    assert transition.safety_release
    assert manager.state.active is ManeuverState.STOP


def test_candidate_set_rejects_missing_controlled_stop() -> None:
    manager, full_candidates = _prime(ManeuverState.PASS_LEFT)
    candidates = tuple(
        candidate
        for candidate in full_candidates
        if candidate.maneuver is not ManeuverId.CONTROLLED_STOP
    )
    with pytest.raises(ValueError, match="must include a CONTROLLED_STOP"):
        _context(
            manager,
            candidates,
            _evaluations(candidates),
            1,
        )


@pytest.mark.parametrize(
    "invalidity",
    [
        {"hard_feasible": False},
        {"risk_limit_ok": False},
    ],
    ids=("hard-infeasible", "risk-limit-violating"),
)
def test_observe_selection_rejects_ineligible_selected_evaluation(
    invalidity: dict[str, object],
) -> None:
    manager = ManeuverCommitmentManager()
    candidates = _candidates()
    selected = _candidate_for(candidates, ManeuverState.PASS_LEFT)
    evaluations = _evaluations(candidates, by_candidate={selected.candidate_id: invalidity})
    context = _context(manager, candidates, evaluations, 0)
    before = manager.snapshot()

    with pytest.raises(ValueError, match="selected candidate is ineligible"):
        manager.observe_selection(
            step=0,
            selected_candidate_id=selected.candidate_id,
            evaluations=tuple(
                evaluation
                for evaluation in evaluations
                if evaluation.candidate_id in context.allowed_candidate_ids
            ),
            context=context,
        )

    assert manager.snapshot() == before


def test_regression_trace_suppresses_oscillation_rejoins_and_stops_immediately() -> None:
    candidates = _candidates()
    raw_trace = []
    for preferred in (
        ManeuverState.PASS_LEFT,
        ManeuverState.YIELD,
        ManeuverState.PASS_RIGHT,
        ManeuverState.YIELD,
        ManeuverState.PASS_LEFT,
    ):
        evaluations = _evaluations(
            candidates,
            by_state={preferred: {"route_progress_m": 2.0}},
        )
        selected = _rank_candidates(candidates, evaluations)
        assert selected is not None
        raw_trace.append(_state_for(selected))
    assert raw_trace == [
        ManeuverState.PASS_LEFT,
        ManeuverState.YIELD,
        ManeuverState.PASS_RIGHT,
        ManeuverState.YIELD,
        ManeuverState.PASS_LEFT,
    ]

    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 10},
        utility_hysteresis_margin=0.05,
        pass_complete_route_margin_m=0.25,
    )
    manager = ManeuverCommitmentManager(config)
    initial_observation = CommitmentObservation(route_progress_m=0.0, interaction_id="track-a")
    initial_evaluations = _evaluations(
        candidates,
        by_state={ManeuverState.PASS_LEFT: {"route_progress_m": 1.0}},
    )
    initial_context = _context(manager, candidates, initial_evaluations, 0, initial_observation)
    initial_selection = _rank_candidates(candidates, initial_evaluations)
    assert initial_selection is not None
    assert _state_for(initial_selection) is ManeuverState.PASS_LEFT
    _select(manager, initial_context, initial_selection, initial_evaluations)
    committed_trace = [manager.state.active]

    for step, preferred in ((1, ManeuverState.YIELD), (2, ManeuverState.PASS_RIGHT)):
        evaluations = _evaluations(
            candidates,
            by_state={
                ManeuverState.PASS_LEFT: {"route_progress_m": 1.0},
                preferred: {"route_progress_m": 1.01},
            },
        )
        context = _context(manager, candidates, evaluations, step)
        assert context.allowed_candidate_ids == tuple(
            sorted(
                candidate.candidate_id
                for candidate in candidates
                if _state_for(candidate) in {ManeuverState.PASS_LEFT, ManeuverState.STOP}
            )
        )
        final_evaluations = _apply_switch_costs(evaluations, dict(context.switch_costs))
        selected = _rank_candidates(candidates, final_evaluations, context.allowed_candidate_ids)
        assert selected is not None
        assert _state_for(selected) is ManeuverState.PASS_LEFT
        _select(manager, context, selected, final_evaluations)
        committed_trace.append(manager.state.active)

    rejoin_observation = CommitmentObservation(
        route_progress_m=0.3,
        lateral_offset_m=0.1,
        interaction_id="track-a",
    )
    rejoin_evaluations = _evaluations(
        candidates,
        by_state={
            ManeuverState.PASS_LEFT: {"route_progress_m": 1.0},
            ManeuverState.FOLLOW: {"route_progress_m": 1.01},
        },
    )
    rejoin_context = _context(manager, candidates, rejoin_evaluations, 3, rejoin_observation)
    assert rejoin_context.release_reason == TransitionReason.PASS_COMPLETE.value
    assert len(rejoin_context.allowed_candidate_ids) == len(candidates)
    committed_trace.append("rejoin")
    final_rejoin_evaluations = _apply_switch_costs(
        rejoin_evaluations, dict(rejoin_context.switch_costs)
    )
    rejoin_selection = _rank_candidates(
        candidates, final_rejoin_evaluations, rejoin_context.allowed_candidate_ids
    )
    assert rejoin_selection is not None
    assert _state_for(rejoin_selection) is ManeuverState.FOLLOW
    _select(manager, rejoin_context, rejoin_selection, final_rejoin_evaluations)
    committed_trace.append(manager.state.active)

    unsafe_motion_states = {
        state: {"hard_feasible": False}
        for state in (
            ManeuverState.FOLLOW,
            ManeuverState.PASS_LEFT,
            ManeuverState.PASS_RIGHT,
            ManeuverState.YIELD,
        )
    }
    safety_evaluations = _evaluations(
        candidates,
        by_state=unsafe_motion_states,
    )
    safety_context = _context(manager, candidates, safety_evaluations, 4)
    assert safety_context.release_reason == TransitionReason.ACTIVE_HARD_INFEASIBLE.value
    final_safety_evaluations = _apply_switch_costs(
        safety_evaluations, dict(safety_context.switch_costs)
    )
    safety_selection = _rank_candidates(
        candidates, final_safety_evaluations, safety_context.allowed_candidate_ids
    )
    assert safety_selection is not None
    assert _state_for(safety_selection) is ManeuverState.STOP
    stop_transition = _select(manager, safety_context, safety_selection, final_safety_evaluations)
    assert stop_transition.safety_release
    assert stop_transition.transition_reason == TransitionReason.ACTIVE_HARD_INFEASIBLE.value
    assert manager.state.active is ManeuverState.STOP
    committed_trace.append(manager.state.active)

    assert committed_trace == [
        ManeuverState.PASS_LEFT,
        ManeuverState.PASS_LEFT,
        ManeuverState.PASS_LEFT,
        "rejoin",
        ManeuverState.FOLLOW,
        ManeuverState.STOP,
    ]


def test_safety_override_releases_and_records_correct_reason() -> None:
    manager, candidates = _prime(ManeuverState.FOLLOW)
    stop = _candidate_for(candidates, ManeuverState.STOP)
    values = _evaluations(candidates)
    observation = CommitmentObservation(safety_override=True)
    context = _context(manager, candidates, values, 1, observation)
    assert len(context.allowed_candidate_ids) == len(candidates)
    transition = _select(manager, context, stop, values)
    assert transition.transition_reason == TransitionReason.SAFETY_OVERRIDE.value
    assert transition.safety_release


def test_pass_rejoin_releases_deterministically() -> None:
    manager, candidates = _prime(
        ManeuverState.PASS_LEFT,
        config=ManeuverCommitmentConfig(min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 10}),
        observation=CommitmentObservation(route_progress_m=0.0, interaction_id="track-a"),
    )
    values = _evaluations(candidates)
    observation = CommitmentObservation(
        route_progress_m=1.0,
        lateral_offset_m=0.1,
        interaction_id="track-a",
    )
    context = _context(manager, candidates, values, 1, observation)
    assert context.release_reason == TransitionReason.PASS_COMPLETE.value
    assert len(context.allowed_candidate_ids) == len(candidates)


def test_pass_interaction_clear_requires_persistence_window() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 10},
        clear_persistence_steps=2,
        pass_complete_route_margin_m=100.0,
    )
    manager, candidates = _prime(
        ManeuverState.PASS_LEFT,
        config=config,
        observation=CommitmentObservation(interaction_id="track-a"),
    )
    values = _evaluations(candidates)
    for step in (1, 2):
        context = _context(
            manager,
            candidates,
            values,
            step,
            CommitmentObservation(interaction_id="track-a", interaction_present=False),
        )
        if step == 1:
            assert context.release_reason is None
            _select(manager, context, _candidate_for(candidates, ManeuverState.PASS_LEFT), values)
        else:
            assert context.release_reason == TransitionReason.INTERACTION_CLEAR.value


def test_yield_to_follow_waits_for_clear_persistence() -> None:
    config = ManeuverCommitmentConfig(clear_persistence_steps=2)
    manager, candidates = _prime(ManeuverState.YIELD, config=config)
    values = _evaluations(candidates)
    first = _context(manager, candidates, values, 1)
    assert first.release_reason is None
    _select(manager, first, _candidate_for(candidates, ManeuverState.YIELD), values)
    second = _context(manager, candidates, values, 2)
    assert second.release_reason == TransitionReason.CLEAR_PERSISTENCE.value


def test_stop_to_follow_requires_clear_persistence_and_avoids_one_step_chatter() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.STOP: 1},
        clear_persistence_steps=2,
    )
    manager, candidates = _prime(ManeuverState.STOP, config=config)
    values = _evaluations(candidates)
    first = _context(manager, candidates, values, 1)
    assert first.release_reason is None
    _select(manager, first, _candidate_for(candidates, ManeuverState.STOP), values)
    second = _context(manager, candidates, values, 2)
    assert second.release_reason == TransitionReason.CLEAR_PERSISTENCE.value


def test_stop_maximum_duration_cannot_bypass_interaction_clear_persistence() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.STOP: 0},
        max_duration_steps_by_state={ManeuverState.STOP: 1},
        clear_persistence_steps=2,
    )
    manager, candidates = _prime(
        ManeuverState.STOP,
        config=config,
        observation=CommitmentObservation(interaction_id="track-a", interaction_present=True),
    )
    interacting_values = _evaluations(
        candidates,
        by_state={
            state: {"risk_bucket": 1}
            for state in (
                ManeuverState.FOLLOW,
                ManeuverState.PASS_LEFT,
                ManeuverState.PASS_RIGHT,
                ManeuverState.YIELD,
            )
        },
    )

    # The moving candidates remain in a worse safety class while the
    # interaction persists. The STOP timer has already expired, but it must
    # not open the portfolio or permit ROUTE_FOLLOW to escape the stop.
    for step in (1, 2):
        context = _context(
            manager,
            candidates,
            interacting_values,
            step,
            CommitmentObservation(interaction_id="track-a", interaction_present=True),
        )
        assert context.release_reason is None
        assert set(context.allowed_candidate_ids) == {
            candidate.candidate_id
            for candidate in candidates
            if _state_for(candidate) is ManeuverState.STOP
        }
        _select(
            manager, context, _candidate_for(candidates, ManeuverState.STOP), interacting_values
        )

    # Once a moving candidate is safe for the configured clear window, the
    # contract-approved path releases STOP and permits ROUTE_FOLLOW.
    clear_values = _evaluations(candidates)
    for step in (3, 4):
        context = _context(
            manager,
            candidates,
            clear_values,
            step,
            CommitmentObservation(interaction_id="track-a", interaction_present=False),
        )
        if step == 3:
            assert context.release_reason is None
            _select(manager, context, _candidate_for(candidates, ManeuverState.STOP), clear_values)
        else:
            assert context.release_reason == TransitionReason.CLEAR_PERSISTENCE.value
            _select(
                manager, context, _candidate_for(candidates, ManeuverState.FOLLOW), clear_values
            )
    assert manager.state.active is ManeuverState.FOLLOW


def test_stop_explicit_infeasibility_can_release_before_duration() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.STOP: 0},
        max_duration_steps_by_state={ManeuverState.STOP: 0},
        clear_persistence_steps=3,
    )
    manager, candidates = _prime(ManeuverState.STOP, config=config)
    values = _evaluations(candidates, by_state={ManeuverState.STOP: {"hard_feasible": False}})
    context = _context(manager, candidates, values, 1)
    assert context.release_reason == TransitionReason.ACTIVE_HARD_INFEASIBLE.value
    assert len(context.allowed_candidate_ids) == len(candidates)


def test_maximum_duration_releases_only_to_reevaluation() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 0},
        max_duration_steps_by_state={ManeuverState.PASS_LEFT: 2},
        pass_complete_route_margin_m=100.0,
    )
    manager, candidates = _prime(ManeuverState.PASS_LEFT, config=config)
    context = _context(manager, candidates, _evaluations(candidates), 2)
    assert context.release_reason == TransitionReason.MAXIMUM_DURATION.value
    assert len(context.allowed_candidate_ids) == len(candidates)


def test_stagnation_uses_measured_route_progress_only() -> None:
    config = ManeuverCommitmentConfig(stagnation_steps=2)
    manager, candidates = _prime(
        ManeuverState.FOLLOW,
        config=config,
        observation=CommitmentObservation(route_progress_m=0.0),
    )
    values = _evaluations(candidates)
    first = _context(manager, candidates, values, 1, CommitmentObservation(route_progress_m=0.0))
    assert first.projected_no_progress_steps == 1
    _select(manager, first, _candidate_for(candidates, ManeuverState.FOLLOW), values)
    second = _context(manager, candidates, values, 2, CommitmentObservation(route_progress_m=0.0))
    assert second.release_reason == TransitionReason.STAGNATION_REEVALUATION.value
    assert second.projected_no_progress_steps == 2
    _select(manager, second, _candidate_for(candidates, ManeuverState.FOLLOW), values)
    third = _context(manager, candidates, values, 3)
    assert third.projected_no_progress_steps == 2


def test_zero_step_thresholds_have_defined_immediate_diagnostic_semantics() -> None:
    clear_manager, candidates = _prime(
        ManeuverState.YIELD,
        config=ManeuverCommitmentConfig(
            min_dwell_steps_by_state={ManeuverState.YIELD: 0},
            clear_persistence_steps=0,
        ),
    )
    clear_context = _context(clear_manager, candidates, _evaluations(candidates), 1)
    assert clear_context.release_reason == TransitionReason.CLEAR_PERSISTENCE.value
    assert len(clear_context.allowed_candidate_ids) == len(candidates)

    stagnation_manager, candidates = _prime(
        ManeuverState.FOLLOW,
        config=ManeuverCommitmentConfig(stagnation_steps=0),
    )
    stagnation_context = _context(
        stagnation_manager,
        candidates,
        _evaluations(candidates),
        1,
        CommitmentObservation(route_progress_m=1.0),
    )
    assert stagnation_context.release_reason == TransitionReason.STAGNATION_REEVALUATION.value
    assert len(stagnation_context.allowed_candidate_ids) == len(candidates)

    transition_manager = ManeuverCommitmentManager(
        ManeuverCommitmentConfig(max_state_transitions_per_episode=0)
    )
    transition_candidates = _candidates()
    transition_values = _evaluations(transition_candidates)
    transition_context = _context(transition_manager, transition_candidates, transition_values, 0)
    transition = _select(
        transition_manager,
        transition_context,
        _candidate_for(transition_candidates, ManeuverState.FOLLOW),
        transition_values,
    )
    assert transition.transition_performed
    assert transition.transition_limit_reached
    assert transition_manager.state.active is ManeuverState.FOLLOW


def test_route_fingerprint_change_releases_and_resets_interaction() -> None:
    manager, candidates = _prime(
        ManeuverState.PASS_LEFT,
        observation=CommitmentObservation(route_fingerprint="route-a", interaction_id="track-a"),
    )
    context = _context(
        manager,
        candidates,
        _evaluations(candidates),
        1,
        CommitmentObservation(route_fingerprint="route-b", interaction_id="track-a"),
    )
    assert context.release_reason == TransitionReason.ROUTE_CHANGED.value
    transition = _select(
        manager, context, _candidate_for(candidates, ManeuverState.FOLLOW), _evaluations(candidates)
    )
    assert transition.transition_reason == TransitionReason.ROUTE_CHANGED.value
    assert manager.state.route_fingerprint == "route-b"
    assert manager.state.stable_interaction_id is None


def test_route_change_resets_stagnation_and_old_interaction_when_pass_is_reselected() -> None:
    config = ManeuverCommitmentConfig(pass_complete_route_margin_m=10.0)
    manager, candidates = _prime(
        ManeuverState.PASS_LEFT,
        config=config,
        observation=CommitmentObservation(
            route_fingerprint="route-a",
            route_progress_m=0.0,
            interaction_id="track-a",
        ),
    )
    values = _evaluations(candidates)
    first = _context(
        manager,
        candidates,
        values,
        1,
        CommitmentObservation(
            route_fingerprint="route-a",
            route_progress_m=0.0,
            interaction_id="track-a",
        ),
    )
    assert first.projected_no_progress_steps == 1
    _select(manager, first, _candidate_for(candidates, ManeuverState.PASS_LEFT), values)

    changed_route = _context(
        manager,
        candidates,
        values,
        2,
        CommitmentObservation(route_fingerprint="route-b", route_progress_m=0.01),
    )
    assert changed_route.release_reason == TransitionReason.ROUTE_CHANGED.value
    assert changed_route.projected_no_progress_steps == 0
    transition = _select(
        manager,
        changed_route,
        _candidate_for(candidates, ManeuverState.PASS_LEFT),
        values,
    )

    assert transition.transition_reason == TransitionReason.ROUTE_CHANGED.value
    assert manager.state.active is ManeuverState.PASS_LEFT
    assert manager.state.route_fingerprint == "route-b"
    assert manager.state.stable_interaction_id is None
    assert manager.state.last_route_progress_m == 0.01
    assert manager.state.no_progress_steps == 0


def test_unsupported_recovery_is_rejected_explicitly() -> None:
    candidates = _candidates()
    unsupported = replace(candidates[0])
    object.__setattr__(unsupported, "maneuver", "recovery")
    with pytest.raises(ValueError, match=TransitionReason.UNSUPPORTED_RECOVERY.value):
        ManeuverCommitmentManager().context_for_candidates(
            step=0,
            candidates=(unsupported,),
            evaluations_without_switch_cost=(),
        )


def test_candidate_set_requires_one_aligned_timestamp_and_time_grid() -> None:
    candidates = _candidates()
    shifted = replace(
        candidates[0],
        metadata={**dict(candidates[0].metadata), "timestamp_s": 0.1},
    )
    mismatched = (shifted, *candidates[1:])
    with pytest.raises(ValueError, match="time grids and timestamps must match"):
        _context(
            ManeuverCommitmentManager(),
            mismatched,
            _evaluations(mismatched),
            0,
        )


def test_configuration_rejects_boolean_counts_and_non_config_manager_input() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        ManeuverCommitmentConfig(max_state_transitions_per_episode=True)
    with pytest.raises(TypeError, match="ManeuverCommitmentConfig"):
        ManeuverCommitmentManager(False)  # type: ignore[arg-type]


def test_duplicate_same_step_observation_is_rejected_by_policy() -> None:
    manager, candidates = _prime(ManeuverState.FOLLOW)
    evaluations = _evaluations(candidates)
    context = _context(manager, candidates, evaluations, 1)
    _select(manager, context, _candidate_for(candidates, ManeuverState.FOLLOW), evaluations)
    with pytest.raises(ValueError, match="duplicate or out of order"):
        manager.observe_selection(
            step=1,
            selected_candidate_id=_candidate_for(candidates, ManeuverState.FOLLOW).candidate_id,
            evaluations=evaluations,
            context=context,
        )


def test_out_of_order_step_is_rejected() -> None:
    manager, candidates = _prime(ManeuverState.FOLLOW)
    with pytest.raises(ValueError, match="duplicate or out of order"):
        _context(manager, candidates, _evaluations(candidates), 0)


def test_reset_returns_to_deterministic_empty_state() -> None:
    manager, _ = _prime(ManeuverState.PASS_RIGHT)
    manager.reset(seed=123)
    snapshot = manager.snapshot()
    fresh = ManeuverCommitmentManager(manager.config)
    assert snapshot == fresh.snapshot()
    assert snapshot["active_state"] is None
    assert snapshot["transition_count"] == 0


def test_candidate_permutation_does_not_change_context_or_allowed_set() -> None:
    config = ManeuverCommitmentConfig()
    manager_a, candidates = _prime(ManeuverState.PASS_LEFT, config=config)
    manager_b, _ = _prime(ManeuverState.PASS_LEFT, config=config)
    evaluations = _evaluations(candidates)
    context_a = _context(manager_a, candidates, evaluations, 1)
    context_b = _context(manager_b, tuple(reversed(candidates)), tuple(reversed(evaluations)), 1)
    assert context_a.to_dict() == context_b.to_dict()


def test_context_calculation_does_not_change_observable_manager_state() -> None:
    manager, candidates = _prime(ManeuverState.PASS_LEFT)
    evaluations = _evaluations(candidates)
    before = manager.snapshot()
    context = _context(manager, candidates, evaluations, 1)
    assert manager.snapshot() == before
    assert context.previous_transition_count == manager.state.transition_count


def test_observation_requires_complete_final_arbitration_evaluations() -> None:
    manager, candidates = _prime(ManeuverState.PASS_LEFT)
    evaluations = _evaluations(candidates)
    context = _context(manager, candidates, evaluations, 1)
    allowed = set(context.allowed_candidate_ids)
    filtered = tuple(item for item in evaluations if item.candidate_id in allowed)
    with pytest.raises(ValueError, match="every allowed candidate exactly once"):
        manager.observe_selection(
            step=1,
            selected_candidate_id=_candidate_for(candidates, ManeuverState.PASS_LEFT).candidate_id,
            evaluations=filtered[:-1],
            context=context,
        )
    _select(manager, context, _candidate_for(candidates, ManeuverState.PASS_LEFT), evaluations)


def test_symmetric_pass_score_cases_select_each_favored_side() -> None:
    config = ManeuverCommitmentConfig(
        min_dwell_steps_by_state={ManeuverState.PASS_LEFT: 0, ManeuverState.PASS_RIGHT: 0},
        utility_hysteresis_margin=0.01,
    )
    decisions = []
    for favored, opposed in (
        (ManeuverState.PASS_LEFT, ManeuverState.PASS_RIGHT),
        (ManeuverState.PASS_RIGHT, ManeuverState.PASS_LEFT),
    ):
        manager, candidates = _prime(ManeuverState.FOLLOW, config=config)
        values = _evaluations(
            candidates,
            by_state={favored: {"route_progress_m": 2.0}, opposed: {"route_progress_m": 1.0}},
        )
        context = _context(manager, candidates, values, 1)
        transition = _select(manager, context, _candidate_for(candidates, favored), values)
        assert transition.transition_reason == TransitionReason.NORMAL_ADVANTAGE.value
        decisions.append(manager.state.active)
    assert decisions == [ManeuverState.PASS_LEFT, ManeuverState.PASS_RIGHT]


def test_transition_cap_is_diagnostic_and_never_locks_out_a_safer_action() -> None:
    config = ManeuverCommitmentConfig(max_state_transitions_per_episode=1)
    manager, candidates = _prime(ManeuverState.FOLLOW, config=config)
    values = _evaluations(
        candidates,
        by_state={ManeuverState.FOLLOW: {"hard_feasible": False}},
    )
    context = _context(manager, candidates, values, 1)
    transition = _select(manager, context, _candidate_for(candidates, ManeuverState.STOP), values)
    assert transition.transition_performed
    assert transition.transition_limit_reached
    assert manager.state.active is ManeuverState.STOP


def _empty_belief(step: int) -> BeliefAwarePlannerInput:
    diagnostics = IdentitySafeProjectionDiagnostics(
        status="empty",
        planner_name="BeliefGuidedLocalPlanner",
        belief_step=step,
        visible_track_count=0,
        occluded_track_count=0,
        stale_track_count=0,
        retained_track_count=0,
        retired_track_count=0,
        dropped_track_count=0,
        per_reason_drop_count=(),
        fallback_reason=None,
        ordered_track_ids=(),
        retired_track_ids=(),
        identity_lifecycle_tokens=(),
    )
    return BeliefAwarePlannerInput(
        legacy_observation=None,
        tracks={},
        belief_step=step,
        schema_version=IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION,
        diagnostics=diagnostics,
    )


def test_real_candidate_and_arbitrator_two_phase_seam_runs_in_repository() -> None:
    candidates = _candidates()
    forecast = MultimodalPrediction(
        forecasts={},
        prediction_horizon=HORIZON * DT_S,
        prediction_dt=DT_S,
        timestamp=0.0,
        metadata={"step": 0},
    )
    belief = _empty_belief(0)
    risk_config = RiskEstimatorConfig(
        horizon_steps=HORIZON,
        dt_s=DT_S,
        n_samples=32,
        min_samples_for_estimate=1,
    )
    route_progress = {
        candidate.candidate_id: float(candidate.states[-1, 0]) for candidate in candidates
    }
    base = arbitrate_multimodal_trajectories(
        candidates,
        forecast,
        belief=belief,
        risk_config=risk_config,
        route_progress=route_progress,
    )
    manager = ManeuverCommitmentManager()
    context = manager.context_for_candidates(
        step=0,
        candidates=candidates,
        evaluations_without_switch_cost=base.evaluations,
    )
    filtered = tuple(
        candidate
        for candidate in candidates
        if candidate.candidate_id in context.allowed_candidate_ids
    )
    result = arbitrate_multimodal_trajectories(
        filtered,
        forecast,
        belief=belief,
        risk_config=risk_config,
        route_progress=route_progress,
        switch_costs=context.switch_costs,
    )
    transition = manager.observe_selection(
        step=0,
        selected_candidate_id=result.selected_candidate_id,
        evaluations=result.evaluations,
        context=context,
    )
    assert result.selected_candidate_id is not None
    assert transition.transition_performed
    assert transition.config_hash == context.config_hash
    json.dumps(context.to_dict(), allow_nan=False)
    json.dumps(transition.to_dict(), allow_nan=False)
