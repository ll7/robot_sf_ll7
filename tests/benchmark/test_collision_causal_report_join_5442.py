"""Tests for the issue #5442 -> #5441 collision-causal-report join.

These verify that a ``last_avoidable_replay.v1`` result produced by the frozen-state
counterfactual replay engine can be embedded into the additive ``collision_causal_report.v1``
contract without re-running the engine, and that the fail-closed failure semantics survive the
join: a nondeterministic baseline or an incomplete feasible-action set yields a fully abstaining
causal report (``unknown``), never ``unavoidable``; an ``already_unavoidable`` replay maps to a
non-abstaining report whose ``supported_actual_cause`` is ``False``; an ``avoidable`` replay maps
to a non-abstaining report that names the intervention model and records the preventing
interventions with ``supported_actual_cause`` true. ``normative_fault`` is always ``not_assessed``.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark import last_avoidable_fixtures as fx
from robot_sf.benchmark.collision_causal_report import (
    CausalJoinMetadata,
    CollisionCausalReportError,
    collide_causal_report_from_last_avoidable,
    validate_collision_causal_report,
)
from robot_sf.benchmark.last_avoidable_replay import (
    SUBSTITUTION_HOLD,
    ReplayConfig,
    locate_last_avoidable,
)

_METADATA = CausalJoinMetadata(
    intervention_model="frozen_state_brake_swap__replayed_pedestrians",
    mechanism_label="guard_or_handoff_domination",
    cause_location="candidate_scoring_selection",
    unsafe_control_action_class="action_not_provided",
    confidence_level="supported_hypothesis",
)


def _run_replay(scenario, *, determinism_replays: int = 20):
    """Run the replay engine over a fixture and return the report."""
    contact_step = fx.find_contact_step(scenario)
    assert contact_step is not None and contact_step >= 1, "baseline must collide"
    horizon = contact_step + 6
    config = ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=horizon,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=determinism_replays,
        action_set_id="decel_lattice",
        feasibility_filter="all_admissible_decel",
        collision_predicate="euclidean_distance<=collision_radius",
        pedestrian_response=scenario.pedestrian_response,
    )
    model = fx.KinematicCollisionModel(scenario)
    baseline = fx.maintain_baseline_actions(contact_step + horizon + 2)
    return locate_last_avoidable(model, baseline, config)


def _join(scenario, *, determinism_replays: int = 20, **kwargs):
    """Run a fixture and embed the replay result into a validated causal report."""
    replay = _run_replay(scenario, determinism_replays=determinism_replays)
    report = collide_causal_report_from_last_avoidable(
        report_id="ccr-join-test",
        case_id="fixture",
        replay=replay,
        metadata=_METADATA,
        **kwargs,
    )
    return validate_collision_causal_report(report)


def test_avoidable_replay_joins_to_supporting_causal_report() -> None:
    """An avoidable replay yields a non-abstaining report with supported actual cause."""
    report = _join(fx.preventable_late_braking_scenario())
    assert report["abstained"] is False
    assert report["causal_contribution"]["verdict"] == "avoidable"
    assert report["causal_contribution"]["supported_actual_cause"] is True
    assert report["causal_contribution"]["intervention_model"]
    # The replay's t_uca / t_inevitable carry through as available timestamps.
    ts = report["observed_reconstruction"]["critical_timestamps"]
    assert ts["t_uca"]["available"] is True and ts["t_uca"]["step"] is not None
    assert ts["t_inevitable"]["available"] is True and ts["t_inevitable"]["step"] is not None
    # The summary has no per-element canonical trace, so coverage cannot make
    # planner-internal reconstruction fields appear available.
    elements = report["observed_reconstruction"]["elements"]
    assert all(element["available"] is False for element in elements.values())
    assert all(key in report["missing_fields"] for key in elements)
    # Every preventing branch from the replay is recorded as a preventing intervention.
    interventions = report["causal_contribution"]["interventions"]
    assert interventions and all(i["prevented_contact"] is True for i in interventions)
    assert report["normative_fault"] == "not_assessed"


def test_already_unavoidable_replay_joins_to_non_abstaining_unavoidable() -> None:
    """An already-unavoidable replay maps to 'unavoidable' with no planner cause."""
    report = _join(fx.already_unavoidable_scenario())
    assert report["abstained"] is False
    assert report["causal_contribution"]["verdict"] == "unavoidable"
    # No admissible action prevented contact, so no actual cause is supported.
    assert report["causal_contribution"]["supported_actual_cause"] is False
    assert report["causal_contribution"]["intervention_model"] == ""
    assert report["causal_contribution"]["interventions"] == []
    ts = report["observed_reconstruction"]["critical_timestamps"]
    assert ts["t_uca"]["available"] is False
    assert ts["t_inevitable"]["available"] is True and ts["t_inevitable"]["step"] == 0
    mechanism = report["proximate_mechanism"]
    assert mechanism["mechanism_label"] == "unknown"
    assert mechanism["cause_location"] == "unknown_or_interacting"
    assert mechanism["unsafe_control_action_class"] == "unknown"
    assert report["confidence"]["level"] == "unknown"
    assert report["normative_fault"] == "not_assessed"


def test_nondeterministic_baseline_joins_to_abstaining_unknown() -> None:
    """A nondeterministic baseline replay joins to a fully abstaining causal report."""
    report = _join(fx.nondeterministic_baseline_scenario(), determinism_replays=20)
    assert report["abstained"] is True
    assert report["causal_contribution"]["verdict"] == "unknown"
    assert report["causal_contribution"]["supported_actual_cause"] is False
    assert report["data_source"]["replay_determinism"] == "nondeterministic"
    # The abstention must flag every planner-internal element and unavailable timestamp missing.
    assert "t_uca" in report["missing_fields"]
    for key in (
        "observations",
        "predictions",
        "generated_candidates",
        "selected_candidate",
        "guard_arbitration_result",
        "feasible_command",
        "applied_command",
        "actor_states",
        "geometry",
    ):
        assert key in report["missing_fields"]


def test_missing_feasible_action_joins_to_abstaining_unknown() -> None:
    """A coverage-gap replay abstains to unknown; never 'unavoidable'."""
    report = _join(fx.missing_feasible_action_scenario())
    assert report["abstained"] is True
    assert report["causal_contribution"]["verdict"] == "unknown"
    assert report["causal_contribution"]["verdict"] != "unavoidable"


def test_unsupported_replay_verdict_abstains_and_fails_closed() -> None:
    """A future replay verdict becomes an explicit unknown report."""
    from dataclasses import replace

    replay = _run_replay(fx.preventable_late_braking_scenario())
    unsupported = replace(replay, verdict="future_verdict", abstained=False)
    report = collide_causal_report_from_last_avoidable(
        report_id="unsupported",
        case_id="fixture",
        replay=unsupported,
        metadata=_METADATA,
    )
    assert report["abstained"] is True
    assert report["abstention_reason"] == "unsupported_replay_verdict"
    assert report["causal_contribution"]["verdict"] == "unknown"
    assert report["causal_contribution"]["supported_actual_cause"] is False
    assert report["proximate_mechanism"]["mechanism_label"] == "unknown"
    assert report["confidence"]["level"] == "unknown"
    validate_collision_causal_report(report)


def test_two_action_interaction_joins_as_avoidable() -> None:
    """A closed-loop two-body interaction replay joins as an avoidable report."""
    scenario = fx.two_action_interaction_scenario()
    report = _join(scenario)
    assert report["causal_contribution"]["verdict"] == "avoidable"
    assert report["causal_contribution"]["supported_actual_cause"] is True
    assert report["causal_contribution"]["pedestrian_response_assumption"] == "closed_loop"


def test_join_rejects_unknown_mechanism_label() -> None:
    """The join rejects a mechanism_label outside the shared taxonomy."""
    from dataclasses import replace

    replay = _run_replay(fx.preventable_late_braking_scenario())
    bad_metadata = replace(_METADATA, mechanism_label="made_up_label")
    with pytest.raises(CollisionCausalReportError):
        collide_causal_report_from_last_avoidable(
            report_id="x",
            case_id="x",
            replay=replay,
            metadata=bad_metadata,
        )


def test_join_rejects_empty_intervention_model_for_supported_cause() -> None:
    """A supported actual cause must name the intervention model."""
    from dataclasses import replace

    replay = _run_replay(fx.preventable_late_braking_scenario())
    bad_metadata = replace(_METADATA, intervention_model="")
    with pytest.raises(CollisionCausalReportError, match="intervention_model"):
        collide_causal_report_from_last_avoidable(
            report_id="x",
            case_id="x",
            replay=replay,
            metadata=bad_metadata,
        )
