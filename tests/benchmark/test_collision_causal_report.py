"""Tests for the ``collision_causal_report.v1`` fail-closed contract.

Covers the two acceptance fixtures required by issue #5441:

* one complete synthetic collision report that validates, and
* one intentionally incomplete report that fails closed by abstaining;

plus negative cases proving each cross-field rule rejects fabricated causal
claims. Run with::

    uv run pytest -q tests/benchmark -k 'collision and causal or causal_report'
"""

from __future__ import annotations

import copy

import pytest

from robot_sf.benchmark.collision_causal_report import (
    CollisionCausalReportError,
    abstained_collision_causal_report,
    reconcile_collision_causal_report,
    validate_collision_causal_report,
)
from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_CONFIDENCES,
    MECHANISM_LABELS,
)


def _complete_report() -> dict:
    """A fully reconstructed, intervention-supported avoidable-collision report.

    The synthetic incident: the planner had generated an emergency-brake
    candidate, did not select it, and a frozen-state branch that swapped in the
    brake prevented contact under a replayed-pedestrian model.
    """

    return {
        "schema_version": "collision_causal_report.v1",
        "report_id": "ccr-synthetic-complete-001",
        "case_id": "crossing_ttc_low__seed7__orca_residual",
        "normative_fault": "not_assessed",
        "data_source": {
            "source_kind": "synthetic_fixture",
            "provenance_uri": "fixture://issue-5441/complete",
            "software_commit": None,
            "replay_determinism": "deterministic",
        },
        "abstained": False,
        "abstention_reason": "",
        "observed_reconstruction": {
            "critical_timestamps": {
                "t_danger": {
                    "available": True,
                    "step": 40,
                    "time_s": 4.0,
                    "source": "critical_intervals.ttc_threshold_crossing",
                },
                "t_uca": {
                    "available": True,
                    "step": 42,
                    "time_s": 4.2,
                    "source": "synthetic_injected_fault",
                },
                "t_inevitable": {
                    "available": True,
                    "step": 47,
                    "time_s": 4.7,
                    "source": "synthetic_branch_search",
                },
                "t_contact": {
                    "available": True,
                    "step": 50,
                    "time_s": 5.0,
                    "source": "event_ledger.CollisionEventRecord",
                },
            },
            "elements": {
                "observations": {
                    "available": True,
                    "source": "mechanism_trace.v1.input_condition",
                    "detail": {"nearest_ped_range_m": 1.8},
                },
                "predictions": {
                    "available": True,
                    "source": "mechanism_trace.v1.risk_score",
                    "detail": {"risk_score": 0.72},
                },
                "generated_candidates": {
                    "available": True,
                    "source": "mechanism_trace.v1.command_source",
                    "detail": ["orca_residual", "emergency_brake"],
                },
                "selected_candidate": {
                    "available": True,
                    "source": "mechanism_trace.v1.selected_command",
                    "detail": {"vx": 0.9, "vy": 0.1},
                },
                "guard_arbitration_result": {
                    "available": True,
                    "source": "mechanism_trace.v1.classification",
                    "detail": "active-but-irrelevant",
                },
                "feasible_command": {
                    "available": True,
                    "source": "synthetic_actuation_model",
                    "detail": {"vx": 0.9, "vy": 0.1},
                },
                "applied_command": {
                    "available": True,
                    "source": "mechanism_trace.v1.selected_command",
                    "detail": {"vx": 0.9, "vy": 0.1},
                },
                "actor_states": {
                    "available": True,
                    "source": "critical_intervals.trace_arrays",
                    "detail": {"n_pedestrians": 3},
                },
                "geometry": {
                    "available": True,
                    "source": "collision_definition_inventory.clearance_regime",
                    "detail": {"radius_sum_m": 1.4},
                },
            },
        },
        "proximate_mechanism": {
            "mechanism_label": "guard_or_handoff_domination",
            "cause_location": "candidate_scoring_selection",
            "unsafe_control_action_class": "action_not_provided",
            "rationale": (
                "The emergency-brake candidate existed but was not selected; the "
                "residual command continued into a closing pedestrian."
            ),
        },
        "causal_contribution": {
            "verdict": "avoidable",
            "intervention_model": "frozen_state_brake_swap__replayed_pedestrians",
            "pedestrian_response_assumption": "replayed",
            "supported_actual_cause": True,
            "interventions": [
                {
                    "name": "emergency_brake_at_t_uca",
                    "admissible": True,
                    "prevented_contact": True,
                    "assumptions": ["replayed pedestrian motion", "fixed simulator rng"],
                },
                {
                    "name": "bounded_steer_left",
                    "admissible": True,
                    "prevented_contact": False,
                    "assumptions": ["replayed pedestrian motion"],
                },
            ],
        },
        "confidence": {
            "level": "supported_hypothesis",
            "rationale": "One deterministic branch prevented contact under a single named model.",
        },
        "assumptions": [
            "single replayed-pedestrian model; closed-loop response not yet evaluated",
        ],
        "missing_fields": [],
        "competing_explanations": [
            {
                "explanation": "pedestrian reciprocity mismatch",
                "ruled_out": False,
                "note": "Not separable without a closed-loop pedestrian branch.",
            }
        ],
    }


def test_complete_report_validates():
    """The complete synthetic fixture satisfies schema and every semantic rule."""
    report = _complete_report()
    assert reconcile_collision_causal_report(report) == []
    assert validate_collision_causal_report(report) == report


def test_complete_report_reuses_shared_vocabularies():
    """mechanism_label and confidence.level come from the shared taxonomy vocabularies."""
    report = _complete_report()
    assert report["proximate_mechanism"]["mechanism_label"] in MECHANISM_LABELS
    assert report["confidence"]["level"] in MECHANISM_CONFIDENCES


def test_incomplete_report_fails_closed_by_abstaining():
    """The intentionally incomplete fixture: planner internals are absent.

    A geometry-only reconstruction (only t_contact known) cannot assert a cause,
    so the analyser abstains rather than inventing selected/applied actions.
    """

    report = abstained_collision_causal_report(
        report_id="ccr-synthetic-incomplete-001",
        case_id="crossing_ttc_low__seed9__blackbox_planner",
        reason="planner decision trace absent; selected/applied action provenance unavailable",
        source_kind="replayed_episode",
    )
    # It is a *valid* report precisely because it abstains and marks everything
    # unavailable instead of fabricating fields.
    assert reconcile_collision_causal_report(report) == []
    assert report["abstained"] is True
    assert report["causal_contribution"]["verdict"] == "unknown"
    assert report["causal_contribution"]["supported_actual_cause"] is False
    # t_uca and t_inevitable are unsupported in v1 and must be declared missing.
    assert "t_uca" in report["missing_fields"]
    assert "selected_candidate" in report["missing_fields"]


def test_normative_fault_must_be_not_assessed():
    """A non-`not_assessed` normative_fault is rejected."""
    report = _complete_report()
    report["normative_fault"] = "robot_at_fault"
    violations = reconcile_collision_causal_report(report)
    # schema const rejects it first; either way the report is rejected.
    assert violations
    with pytest.raises(CollisionCausalReportError):
        validate_collision_causal_report(report)


def test_actual_cause_requires_prevention_intervention():
    """supported_actual_cause needs an intervention branch that prevented contact."""
    report = _complete_report()
    for intervention in report["causal_contribution"]["interventions"]:
        intervention["prevented_contact"] = False
    violations = reconcile_collision_causal_report(report)
    assert any("prevented_contact true" in v for v in violations)


def test_actual_cause_requires_named_intervention_model():
    """supported_actual_cause needs a named (non-blank) intervention_model."""
    report = _complete_report()
    report["causal_contribution"]["intervention_model"] = "   "
    violations = reconcile_collision_causal_report(report)
    assert any("named intervention_model" in v for v in violations)


def test_inevitable_before_uca_forbids_planner_cause():
    """When t_inevitable <= t_uca, a planner action cannot be the supported cause."""
    report = _complete_report()
    # Contact becomes unavoidable at or before the first unsafe control action.
    report["observed_reconstruction"]["critical_timestamps"]["t_inevitable"]["step"] = 42
    report["observed_reconstruction"]["critical_timestamps"]["t_inevitable"]["time_s"] = 4.2
    report["observed_reconstruction"]["critical_timestamps"]["t_uca"]["step"] = 42
    violations = reconcile_collision_causal_report(report)
    assert any("already unavoidable" in v for v in violations)


def test_unavailable_element_must_be_declared_missing():
    """An unavailable element must appear in missing_fields; declaring it repairs the report."""
    report = _complete_report()
    report["observed_reconstruction"]["elements"]["predictions"] = {
        "available": False,
        "source": None,
        "detail": None,
    }
    # Not declared in missing_fields -> rejected.
    violations = reconcile_collision_causal_report(report)
    assert any("must be listed in missing_fields" in v for v in violations)
    # Declaring it repairs the report.
    report["missing_fields"].append("predictions")
    assert reconcile_collision_causal_report(report) == []


def test_unavailable_timestamp_cannot_carry_inferred_value():
    """An unavailable timestamp may not carry an inferred step/time_s."""
    report = _complete_report()
    report["observed_reconstruction"]["critical_timestamps"]["t_uca"] = {
        "available": False,
        "step": 42,
        "time_s": 4.2,
        "source": None,
    }
    report["missing_fields"].append("t_uca")
    violations = reconcile_collision_causal_report(report)
    assert any("inferred step/time_s" in v for v in violations)


def test_unknown_mechanism_label_rejected():
    """A mechanism_label outside the shared taxonomy is rejected."""
    report = _complete_report()
    report["proximate_mechanism"]["mechanism_label"] = "made_up_label"
    violations = reconcile_collision_causal_report(report)
    assert any("MECHANISM_LABELS" in v for v in violations)


def test_abstained_report_cannot_claim_actual_cause():
    """Setting abstained=true while still asserting a cause is a contradiction."""
    report = copy.deepcopy(_complete_report())
    report["abstained"] = True
    report["abstention_reason"] = "nondeterministic replay"
    # verdict/supported_actual_cause still assert a cause -> contradiction.
    violations = reconcile_collision_causal_report(report)
    assert any("verdict to 'unknown'" in v for v in violations)
    assert any("cannot set supported_actual_cause" in v for v in violations)
