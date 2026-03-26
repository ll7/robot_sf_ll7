"""Tests for canonical benchmark fallback and availability policy helpers."""

from __future__ import annotations

from robot_sf.benchmark.fallback_policy import (
    benchmark_run_exit_code,
    campaign_exit_code,
    summarize_benchmark_availability,
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


def test_campaign_exit_code_uses_benchmark_success_flag() -> None:
    """Campaign CLI should exit non-zero for non-success benchmark results."""
    assert campaign_exit_code({"benchmark_success": True}) == 0
    assert campaign_exit_code({"benchmark_success": False}) == 2
