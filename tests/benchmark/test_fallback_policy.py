"""Tests for canonical benchmark fallback and availability policy helpers."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest

from robot_sf.benchmark.fallback_policy import (
    availability_payload,
    benchmark_run_exit_code,
    campaign_exit_code,
    campaign_status_axes_payload,
    runtime_fallback_or_degraded_marker,
    summarize_benchmark_availability,
    summarize_campaign_outcome,
    summarize_campaign_status_axes,
)


@pytest.fixture
def guarded_runtime(request) -> dict[str, object]:
    """Build the real shield serialization and runner hook for a clear policy command.

    Returns:
        Runtime metadata matching the issue9718 diagnostic's nonintervened command.
    """
    from robot_sf.benchmark.map_runner.map_runner import _attach_guard_decision_stats
    from robot_sf.planner.safety_shield import ShieldDecision

    action = (2.0, -0.045798975974321365)
    label = getattr(request, "param", "ppo_clear")
    intervened = label not in {"ppo_clear", "ppo_safe", "goal_reached"}
    filtered_action = (0.0, 0.0) if intervened else action
    decision = ShieldDecision(
        proposed_action=action,
        filtered_action=filtered_action,
        decision_label=label,
        intervention_reason="proposed_action_clear_of_near_field",
        intervened=intervened,
        fallback_controller_state={
            "policy": "RiskDWAPlannerAdapter",
            "prior_available": False,
            "action_adaptation": {
                "mode": "guard_selected_command" if intervened else "direct_policy_command",
                "raw_policy_action": list(action),
                "adapted_action": list(filtered_action),
                "residual_clipped": False,
                "hard_guard_authoritative": True,
            },
        },
    )
    policy = Mock()
    policy._planner_stats.return_value = {
        "checkpoint_provenance": {"load_succeeded": True, "fallback_triggered": False}
    }
    _attach_guard_decision_stats(
        policy, {"shield_stats": {"last_decision": decision.to_metadata()}}
    )
    return policy._planner_stats()


def test_shield_dictionary_does_not_report_policy_fallback(guarded_runtime) -> None:
    """Typed diagnostic state must survive the real runner-to-availability path."""
    decision = guarded_runtime["last_decision"]
    assert decision["intervened"] is False
    assert decision["override_applied"] is False
    assert runtime_fallback_or_degraded_marker(guarded_runtime) is None
    summary = {
        "status": "ok",
        "written": 1,
        "total_jobs": 1,
        "failed_jobs": 0,
        "preflight": {"status": "ok"},
        "algorithm_metadata_contract": {
            "planner_kinematics": {"execution_mode": "mixed"},
            "planner_runtime": guarded_runtime,
        },
    }
    availability = summarize_benchmark_availability(summary)
    assert availability.execution_mode == "mixed"
    assert availability.benchmark_success is True
    assert benchmark_run_exit_code(summary) == 0


@pytest.fixture
def guarded_metadata(guarded_runtime) -> dict[str, object]:
    """Bind real shield telemetry to the established guarded composite identity.

    Returns:
        Algorithm metadata with the real serialized decision and positive decision counters.
    """
    label = guarded_runtime["last_decision"]["decision_label"]
    return {
        "algorithm": "ppo",
        "canonical_algorithm": "guarded_ppo",
        "planner_contract": {"planner_id": "guarded_ppo"},
        "planner_kinematics": {"execution_mode": "mixed"},
        "planner_runtime": guarded_runtime,
        "guard_stats": {label: 1},
        "shield_stats": {
            "decision_counts": {label: 1},
            "last_decision": guarded_runtime["last_decision"],
        },
    }


@pytest.mark.parametrize("guarded_runtime", ["fallback_safe"], indirect=True)
def test_declared_guarded_safe_command_remains_available(guarded_metadata) -> None:
    """The frozen composite planner's safe Risk-DWA intervention retains its contract."""
    from robot_sf.benchmark.release_acceptance import _status_markers

    runtime = guarded_metadata["planner_runtime"]
    decision = runtime["last_decision"]
    assert decision["intervened"] is True
    assert decision["fallback_controller_state"]["policy"] == "RiskDWAPlannerAdapter"
    assert runtime_fallback_or_degraded_marker(runtime) is None
    summary = {
        "status": "ok",
        "written": 1,
        "total_jobs": 1,
        "failed_jobs": 0,
        "preflight": {"status": "ok"},
        "algorithm_metadata_contract": guarded_metadata,
    }
    availability = summarize_benchmark_availability(summary)
    assert availability.execution_mode == "mixed"
    assert availability.availability_status == "available"
    assert availability.benchmark_success is True
    assert benchmark_run_exit_code(summary) == 0
    payload = {"status": "ok", "algorithm_metadata": guarded_metadata}
    assert _status_markers(payload, "row", expected_algorithm="guarded_ppo") == []
    assert _status_markers(payload, "row", expected_algorithm="goal")


@pytest.mark.parametrize(
    "guarded_runtime",
    ["fallback_best_effort", "uncertainty_fallback_configured"],
    indirect=True,
)
def test_guarded_genuine_fallback_counters_remain_forbidden(guarded_metadata) -> None:
    """Positive producer counters keep best-effort/uncertainty execution inadmissible."""
    from robot_sf.benchmark.release_acceptance import _status_markers

    label = guarded_metadata["planner_runtime"]["last_decision"]["decision_label"]
    payload = {"status": "ok", "algorithm_metadata": guarded_metadata}
    markers = _status_markers(payload, "row", expected_algorithm="guarded_ppo")
    assert any(f"guard_stats.{label}" in path and value == "1" for path, value in markers)


@pytest.mark.parametrize("guarded_runtime", ["fallback_safe"], indirect=True)
def test_guarded_safe_decision_does_not_hide_checkpoint_fallback(guarded_metadata) -> None:
    """A legitimate shield command cannot excuse a genuine failed checkpoint path."""
    from robot_sf.benchmark.release_acceptance import _status_markers

    runtime = guarded_metadata["planner_runtime"]
    runtime["checkpoint_provenance"]["fallback_triggered"] = True
    summary = {
        "status": "ok",
        "written": 1,
        "total_jobs": 1,
        "failed_jobs": 0,
        "preflight": {"status": "ok"},
        "algorithm_metadata_contract": guarded_metadata,
    }
    assert runtime_fallback_or_degraded_marker(runtime) == (
        "checkpoint_provenance.fallback_triggered",
        "true",
    )
    assert summarize_benchmark_availability(summary).benchmark_success is False
    assert benchmark_run_exit_code(summary) == 2
    payload = {"status": "ok", "algorithm_metadata": guarded_metadata}
    assert _status_markers(payload, "row", expected_algorithm="guarded_ppo")


@pytest.mark.parametrize("value", [None, [], "unused", 0, False, 1])
def test_shield_dictionary_rejects_wrong_type(value) -> None:
    """The declared dictionary cannot be substituted with a scalar or sequence."""
    assert runtime_fallback_or_degraded_marker({"fallback_controller_state": value}) == (
        "fallback_controller_state",
        "invalid",
    )


@pytest.mark.parametrize(
    ("marker", "value", "expected"),
    [
        ("fallback_triggered", True, "true"),
        ("degraded", True, "true"),
        ("status", "fallback", "fallback"),
        ("status", "degraded", "degraded"),
        ("fallback_count", 2, "2"),
        ("fallback_count", {}, "invalid"),
        ("fallback_count", float("nan"), "invalid"),
        ("fallback_used", "false", "invalid"),
    ],
)
def test_shield_dictionary_still_exposes_nested_failures(
    guarded_runtime, marker, value, expected
) -> None:
    """Recognizing the container must never skip genuine or malformed failure evidence."""
    payload = deepcopy(guarded_runtime)
    payload["last_decision"]["fallback_controller_state"]["events"] = [{marker: value}]
    assert runtime_fallback_or_degraded_marker(payload) == (
        f"last_decision.fallback_controller_state.events[0].{marker}",
        expected,
    )


def test_unknown_fallback_dictionary_is_still_invalid() -> None:
    """Only the documented typed field is a container; other counters stay strict."""
    assert runtime_fallback_or_degraded_marker({"unknown_fallback_counter": {}}) == (
        "unknown_fallback_counter",
        "invalid",
    )


def test_summarize_benchmark_availability_marks_fallback_as_not_available() -> None:
    """Fallback preflight must never count as benchmark-available."""
    summary = {
        "status": "ok",
        "written": 1,
        "total_jobs": 1,
        "failed_jobs": 0,
        "preflight": {"status": "fallback", "compatibility_reason": "missing optional dep"},
        "algorithm_metadata_contract": {"execution_mode": "adapter"},
    }

    availability = summarize_benchmark_availability(summary)

    assert availability.execution_mode == "adapter"
    assert availability.readiness_status == "fallback"
    assert availability.availability_status == "not_available"
    assert availability.benchmark_success is False
    assert availability.availability_reason == "missing optional dep"
    assert benchmark_run_exit_code(summary) == 2

    assert availability_payload(summary) == {
        "execution_mode": "adapter",
        "readiness_status": "fallback",
        "availability_status": "not_available",
        "benchmark_success": False,
        "availability_reason": "missing optional dep",
    }


def test_summarize_benchmark_availability_marks_partial_failure() -> None:
    """Planner runs with failed jobs must be non-success even if some episodes were written."""
    summary = {
        "status": "ok",
        "written": 2,
        "total_jobs": 3,
        "failed_jobs": 1,
        "preflight": {"status": "ok"},
        "algorithm_metadata_contract": {"execution_mode": "native"},
        "failures": [{"error": "worker crash"}],
    }

    availability = summarize_benchmark_availability(summary)

    assert availability.readiness_status == "native"
    assert availability.availability_status == "partial-failure"
    assert availability.benchmark_success is False
    assert availability.availability_reason == "worker crash"


def test_summarize_benchmark_availability_marks_latency_stress_as_not_available() -> None:
    """Latency-stress preflight rows should not count as benchmark evidence yet."""
    summary = {
        "status": "ok",
        "written": 1,
        "total_jobs": 1,
        "failed_jobs": 0,
        "preflight": {"status": "ok"},
        "algorithm_metadata_contract": {"execution_mode": "native"},
        "latency_stress_profile": {"name": "learned-policy-latency-stress-v0"},
        "latency_stress_metrics": {"held_action_ratio": "not_available"},
    }

    availability = summarize_benchmark_availability(summary)

    assert availability.availability_status == "not_available"
    assert availability.benchmark_success is False
    assert "runtime latency metrics" in str(availability.availability_reason)


def test_campaign_outcome_and_exit_code_distinguish_non_success_campaign_classes() -> None:
    """Campaign exit semantics should separate accepted unavailable rows from true failures."""
    success = summarize_campaign_outcome({"planner_rows": [{"status": "ok"}]})
    accepted_unavailable = summarize_campaign_outcome(
        {"planner_rows": [{"status": "ok"}, {"status": "not_available"}]}
    )
    unexpected_failure = summarize_campaign_outcome(
        {"planner_rows": [{"status": "ok"}, {"status": "partial-failure"}]}
    )

    assert success.status == "benchmark_success"
    assert success.exit_code == 0
    assert campaign_exit_code({"planner_rows": [{"status": "ok"}]}) == 0

    assert accepted_unavailable.status == "accepted_unavailable_only"
    assert accepted_unavailable.accepted_unavailable_runs == 1
    assert accepted_unavailable.unexpected_failed_runs == 0
    assert accepted_unavailable.exit_code == 3
    assert (
        campaign_exit_code({"planner_rows": [{"status": "ok"}, {"status": "not_available"}]}) == 3
    )

    assert unexpected_failure.status == "unexpected_failure"
    assert unexpected_failure.accepted_unavailable_runs == 0
    assert unexpected_failure.unexpected_failed_runs == 1
    assert unexpected_failure.exit_code == 2
    assert (
        campaign_exit_code({"planner_rows": [{"status": "ok"}, {"status": "partial-failure"}]}) == 2
    )

    assert campaign_exit_code({}) == 2
    assert campaign_exit_code(None) == 2


def test_campaign_outcome_ignores_partial_planner_rows_and_falls_back_to_runs() -> None:
    """Planner-row status summaries are trusted only when every row supplies a status."""
    outcome = summarize_campaign_outcome(
        {
            "planner_rows": [{"status": "ok"}, {}],
            "runs": [{"status": "partial-failure"}],
        }
    )

    assert outcome.status == "unexpected_failure"
    assert outcome.unexpected_failed_runs == 1
    assert outcome.exit_code == 2


def test_campaign_outcome_ignores_malformed_planner_rows_and_falls_back_to_runs() -> None:
    """Planner-row status summaries are ignored when any row is malformed."""
    outcome = summarize_campaign_outcome(
        {
            "planner_rows": [{"status": "ok"}, None],
            "runs": [{"status": "partial-failure"}],
        }
    )

    assert outcome.status == "unexpected_failure"
    assert outcome.unexpected_failed_runs == 1
    assert outcome.exit_code == 2


def test_campaign_exit_code_rejects_noncanonical_explicit_values() -> None:
    """Only canonical integer, non-bool campaign exit codes should bypass summarization."""
    success_payload = {"exit_code": True, "planner_rows": [{"status": "ok"}]}
    accepted_unavailable_payload = {
        "exit_code": 7,
        "planner_rows": [{"status": "ok"}, {"status": "not_available"}],
    }

    assert (
        campaign_exit_code({"exit_code": 0, "planner_rows": [{"status": "partial-failure"}]}) == 0
    )
    assert campaign_exit_code(success_payload) == 0
    assert campaign_exit_code(accepted_unavailable_payload) == 3


def test_campaign_outcome_accepts_all_ok_explicit_fields_without_rows() -> None:
    """Explicit campaign-success fields should preserve the benchmark-success class."""
    outcome = summarize_campaign_outcome(
        {
            "status": "benchmark_success",
            "benchmark_success": True,
            "status_reason": "all planner rows were benchmark-success",
            "successful_runs": 2,
            "accepted_unavailable_runs": 0,
            "unexpected_failed_runs": 0,
            "non_success_runs": 0,
            "total_runs": 2,
        }
    )

    assert outcome.status == "benchmark_success"
    assert outcome.benchmark_success is True
    assert outcome.successful_runs == 2
    assert outcome.exit_code == 0


def test_campaign_outcome_accepts_accepted_unavailable_explicit_fields_without_rows() -> None:
    """Explicit accepted-unavailable fields should keep the distinct nonzero code."""
    outcome = summarize_campaign_outcome(
        {
            "status": "accepted_unavailable_only",
            "benchmark_success": False,
            "status_reason": "campaign contains accepted unavailable/excluded rows and no unexpected failed rows",
            "successful_runs": 1,
            "accepted_unavailable_runs": 1,
            "unexpected_failed_runs": 0,
            "non_success_runs": 1,
            "total_runs": 2,
        }
    )

    assert outcome.status == "accepted_unavailable_only"
    assert outcome.benchmark_success is False
    assert outcome.accepted_unavailable_runs == 1
    assert outcome.unexpected_failed_runs == 0
    assert outcome.exit_code == 3


def test_campaign_status_axes_distinguish_partial_evidence_from_execution_completion() -> None:
    """Accepted unavailable rows should keep completed execution but partial evidence."""
    status_axes = summarize_campaign_status_axes(
        {
            "planner_rows": [
                {"status": "ok", "readiness_status": "native"},
                {"status": "not_available", "readiness_status": "fallback"},
            ]
        },
        expected_total_runs=2,
    )

    assert status_axes.campaign_execution_status == "completed"
    assert status_axes.evidence_status == "partial"
    assert status_axes.row_status_summary.successful_evidence_rows == 1
    assert status_axes.row_status_summary.accepted_unavailable_rows == 1
    assert status_axes.row_status_summary.unexpected_failed_rows == 0
    assert status_axes.row_status_summary.fallback_or_degraded_rows == 1
    assert campaign_status_axes_payload(
        {
            "planner_rows": [
                {"status": "ok", "readiness_status": "native"},
                {"status": "not_available", "readiness_status": "fallback"},
            ]
        },
        expected_total_runs=2,
    ) == {
        "campaign_execution_status": "completed",
        "evidence_status": "partial",
        "row_status_summary": {
            "successful_evidence_rows": 1,
            "accepted_unavailable_rows": 1,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 1,
        },
    }


def test_campaign_status_axes_mark_interrupted_unexpected_failure() -> None:
    """Unexpected failures with missing expected rows should surface interruption explicitly."""
    status_axes = summarize_campaign_status_axes(
        {"planner_rows": [{"status": "ok"}, {"status": "partial-failure"}]},
        expected_total_runs=3,
    )

    assert status_axes.campaign_execution_status == "interrupted"
    assert status_axes.evidence_status == "invalid"
    assert status_axes.row_status_summary.successful_evidence_rows == 1
    assert status_axes.row_status_summary.accepted_unavailable_rows == 0
    assert status_axes.row_status_summary.unexpected_failed_rows == 1


def test_campaign_outcome_explicit_counts_prefer_unexpected_failure_over_accepted_unavailable() -> (
    None
):
    """Unexpected failed rows must dominate mixed explicit non-success counters."""
    outcome = summarize_campaign_outcome(
        {
            "status": "accepted_unavailable_only",
            "benchmark_success": False,
            "successful_runs": 1,
            "accepted_unavailable_runs": 1,
            "unexpected_failed_runs": 1,
            "non_success_runs": 2,
            "total_runs": 3,
        }
    )

    assert outcome.status == "unexpected_failure"
    assert outcome.accepted_unavailable_runs == 1
    assert outcome.unexpected_failed_runs == 1
    assert outcome.non_success_runs == 2
    assert outcome.exit_code == 2


def test_campaign_outcome_unknown_explicit_status_fails_closed() -> None:
    """Unknown campaign status labels must not silently degrade into success."""
    outcome = summarize_campaign_outcome(
        {
            "status": "mystery_status",
            "benchmark_success": True,
            "successful_runs": 2,
            "non_success_runs": 0,
            "total_runs": 2,
        }
    )

    assert outcome.status == "malformed"
    assert outcome.benchmark_success is False
    assert (
        outcome.status_reason
        == "campaign result payload has unknown explicit status 'mystery_status'"
    )
    assert outcome.exit_code == 2


def test_summarize_benchmark_availability_fails_closed_for_malformed_payload() -> None:
    """Malformed summaries must fail closed instead of defaulting to benchmark success."""
    availability = summarize_benchmark_availability(None)

    assert availability.execution_mode == "unknown"
    assert availability.readiness_status == "degraded"
    assert availability.availability_status == "failed"
    assert availability.benchmark_success is False
    assert availability.availability_reason == "malformed_summary_payload"
    assert benchmark_run_exit_code(None) == 2


def test_summarize_benchmark_availability_surfaces_learned_contract_reason() -> None:
    """Learned-policy mismatch details should populate the availability reason."""
    summary = {
        "status": "ok",
        "written": 1,
        "total_jobs": 1,
        "failed_jobs": 0,
        "preflight": {
            "status": "fallback",
            "learned_policy_contract": {
                "critical_mismatches": ["obs_mode=image mismatch"],
                "warnings": ["determinism not guaranteed"],
            },
        },
    }

    availability = summarize_benchmark_availability(summary)

    assert availability.availability_status == "not_available"
    assert availability.availability_reason == (
        "obs_mode=image mismatch; determinism not guaranteed"
    )


def test_runtime_fallback_marker_is_key_aware_and_traverses_lists() -> None:
    """Structured runtime markers fail closed without matching descriptive text."""
    payload = {
        "description": "fallback is forbidden in benchmark evidence",
        "planner_runtime": {
            "status": "ok",
            "events": [
                {"message": "degraded wording is descriptive only"},
                {"fallback_count": 2},
            ],
        },
    }

    assert runtime_fallback_or_degraded_marker(payload) == (
        "planner_runtime.events[1].fallback_count",
        "2",
    )
    assert (
        runtime_fallback_or_degraded_marker(
            {"description": "fallback", "planner_runtime": {"status": "ok"}}
        )
        is None
    )


def test_runtime_fallback_marker_allows_empty_reason_with_explicit_false_sibling() -> None:
    """Unused fallback reasons may be empty only beside an explicit false flag."""
    for reason in (None, ""):
        assert (
            runtime_fallback_or_degraded_marker({"fallback_used": False, "fallback_reason": reason})
            is None
        )


def test_runtime_fallback_marker_allows_zero_canonical_proposal_status_counts() -> None:
    """Zero fallback/degraded proposal counters are valid no-event telemetry."""
    payload = {
        "proposal_status_counts": {
            "adapter": 1,
            "degraded": 0,
            "fallback": 0,
            "native": 1,
        }
    }

    assert runtime_fallback_or_degraded_marker(payload) is None


@pytest.mark.parametrize(
    ("value", "normalized"),
    [(1, "1"), (-1, "invalid"), (float("inf"), "invalid"), ("0", "invalid")],
)
def test_runtime_fallback_marker_rejects_nonzero_or_malformed_proposal_status_counts(
    value: object, normalized: str
) -> None:
    """Canonical proposal counters still fail closed on use or malformed telemetry."""
    payload = {"proposal_status_counts": {"fallback": value}}

    assert runtime_fallback_or_degraded_marker(payload) == (
        "proposal_status_counts.fallback",
        normalized,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"fallback_reason": None}, ("fallback_reason", "invalid")),
        (
            {"fallback_used": "false", "fallback_reason": None},
            ("fallback_used", "invalid"),
        ),
        ({"fallback_used": True, "fallback_reason": None}, ("fallback_used", "true")),
        (
            {"fallback_used": False, "fallback_reason": "unexpected runtime fallback"},
            ("fallback_reason", "invalid"),
        ),
        ({"fallback_used": False, "fallback_reason": " "}, ("fallback_reason", "invalid")),
        ({"fallback_used": False, "fallback_reason": 0}, ("fallback_reason", "invalid")),
        (
            {"fallback_used": False, "fallback_reason": None, "fallback_count": 1},
            ("fallback_count", "1"),
        ),
    ],
)
def test_runtime_fallback_marker_rejects_missing_invalid_or_positive_reason_state(
    payload: dict[str, object], expected: tuple[str, str]
) -> None:
    """Empty reasons never mask malformed flags, non-empty reasons, or positive counters."""
    assert runtime_fallback_or_degraded_marker(payload) == expected


@pytest.mark.parametrize("reason", [None, ""])
def test_runtime_fallback_marker_allows_nested_empty_reason_with_explicit_false_sibling(
    reason: object,
) -> None:
    """The empty-reason exception applies inside canonical runtime containers too."""
    assert (
        runtime_fallback_or_degraded_marker(
            {
                "planner_runtime": {
                    "foresight_prediction": {
                        "fallback_used": False,
                        "fallback_reason": reason,
                    }
                }
            }
        )
        is None
    )


def test_summarize_benchmark_availability_rejects_runtime_fallback_marker() -> None:
    """Runtime fallback metadata must prevent a successful benchmark classification."""
    summary = {
        "status": "ok",
        "written": 1,
        "total_jobs": 1,
        "failed_jobs": 0,
        "preflight": {"status": "ok"},
        "algorithm_metadata_contract": {
            "execution_mode": "native",
            "planner_runtime": {"fallback_triggered": True},
        },
    }

    availability = summarize_benchmark_availability(summary)

    assert availability.readiness_status == "fallback"
    assert availability.availability_status == "failed"
    assert availability.benchmark_success is False
    assert availability.availability_reason == (
        "planner runtime reported forbidden marker fallback_triggered=true"
    )
