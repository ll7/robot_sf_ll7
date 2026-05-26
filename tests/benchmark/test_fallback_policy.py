"""Tests for canonical benchmark fallback and availability policy helpers."""

from __future__ import annotations

from robot_sf.benchmark.fallback_policy import (
    availability_payload,
    benchmark_run_exit_code,
    campaign_exit_code,
    summarize_benchmark_availability,
    summarize_campaign_outcome,
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
