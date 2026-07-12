"""Tests for campaign arm_rollup (issue #5391)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.camera_ready._reporting import write_campaign_report
from robot_sf.benchmark.camera_ready._run_state import _build_arm_rollup

if TYPE_CHECKING:
    from pathlib import Path


def test_arm_rollup_ok_arms_produces_ok_rows() -> None:
    """Fully-green campaign produces ok rows with no error fields."""
    run_entries: list[dict[str, Any]] = [
        {
            "planner": {"key": "sfm", "algo": "sfm", "kinematics": "diff"},
            "status": "ok",
            "summary": {"written": 10, "failed_jobs": 0, "failures": []},
        },
        {
            "planner": {"key": "orca", "algo": "orca", "kinematics": "diff"},
            "status": "ok",
            "summary": {"written": 8, "failed_jobs": 0, "failures": []},
        },
    ]
    rollup = _build_arm_rollup(run_entries)
    assert len(rollup) == 2
    assert rollup[0]["planner_key"] == "sfm"
    assert rollup[0]["kinematics"] == "diff"
    assert rollup[0]["status"] == "ok"
    assert rollup[0]["episodes_written"] == 10
    assert rollup[0]["episodes_failed"] == 0
    assert "first_error" not in rollup[0]
    assert "distinct_error_count" not in rollup[0]
    assert rollup[1]["planner_key"] == "orca"
    assert rollup[1]["status"] == "ok"


def test_arm_rollup_failed_arm_names_first_error() -> None:
    """Failed arm rollup names the arm and its first error with distinct count."""
    run_entries: list[dict[str, Any]] = [
        {
            "planner": {"key": "sfm", "algo": "sfm", "kinematics": "diff"},
            "status": "ok",
            "summary": {"written": 10, "failed_jobs": 0, "failures": []},
        },
        {
            "planner": {"key": "orca", "algo": "orca", "kinematics": "ackermann"},
            "status": "failed",
            "summary": {
                "written": 0,
                "failed_jobs": 3,
                "failures": [
                    {"scenario_id": "s3", "seed": 2, "error": "ValueError('bad config')"},
                    {
                        "scenario_id": "s1",
                        "seed": 1,
                        "error": "RuntimeError('map resolution missing')",
                    },
                    {
                        "scenario_id": "s2",
                        "seed": 1,
                        "error": "RuntimeError('map resolution missing')",
                    },
                ],
            },
        },
    ]
    rollup = _build_arm_rollup(run_entries)
    assert len(rollup) == 2
    failed_arm = rollup[1]
    assert failed_arm["planner_key"] == "orca"
    assert failed_arm["kinematics"] == "ackermann"
    assert failed_arm["status"] == "failed"
    assert failed_arm["episodes_written"] == 0
    assert failed_arm["episodes_failed"] == 3
    assert "first_error" in failed_arm
    assert failed_arm["first_error"] == "ValueError('bad config')"
    assert failed_arm["distinct_error_count"] == 2
    assert len(failed_arm["first_error"]) <= 200


def test_arm_rollup_partial_failure_arm() -> None:
    """Partial-failure arm includes error fields from per-job failures."""
    run_entries: list[dict[str, Any]] = [
        {
            "planner": {"key": "hybrid", "algo": "hybrid", "kinematics": "diff"},
            "status": "partial-failure",
            "summary": {
                "written": 5,
                "failed_jobs": 2,
                "failures": [
                    {"scenario_id": "s1", "seed": 1, "error": "TimeoutError('episode timeout')"},
                    {"scenario_id": "s2", "seed": 1, "error": "AssertionError('invariant')"},
                ],
            },
        },
    ]
    rollup = _build_arm_rollup(run_entries)
    assert len(rollup) == 1
    arm = rollup[0]
    assert arm["status"] == "partial-failure"
    assert arm["episodes_written"] == 5
    assert arm["episodes_failed"] == 2
    assert "first_error" in arm
    assert arm["distinct_error_count"] == 2


def test_arm_rollup_exception_path_uses_summary_error() -> None:
    """Exception-during-batch arm falls back to summary-level error string."""
    run_entries: list[dict[str, Any]] = [
        {
            "planner": {"key": "broken", "algo": "custom", "kinematics": "diff"},
            "status": "failed",
            "summary": {
                "status": "failed",
                "error": "Planner 'broken' failed for kinematics 'diff': ImportError('no module')",
                "written": 0,
                "failed_jobs": 0,
                "failures": [],
            },
        },
    ]
    rollup = _build_arm_rollup(run_entries)
    arm = rollup[0]
    assert arm["status"] == "failed"
    assert "first_error" in arm
    assert "ImportError" in arm["first_error"]


def test_arm_rollup_none_optional_values_fail_closed_to_zero() -> None:
    """Null summary counters default safely and null errors stay absent."""
    run_entries: list[dict[str, Any]] = [
        {
            "planner": {"key": "broken", "algo": "custom", "kinematics": "diff"},
            "status": "failed",
            "summary": {
                "written": None,
                "episodes_total": 4,
                "failed_jobs": None,
                "failures": [],
                "error": None,
            },
        },
    ]

    arm = _build_arm_rollup(run_entries)[0]

    assert arm["episodes_written"] == 4
    assert arm["episodes_failed"] == 0
    assert "first_error" not in arm


def test_arm_rollup_not_available_arm_has_no_error() -> None:
    """Dependency-gated not_available arm has no error fields."""
    run_entries: list[dict[str, Any]] = [
        {
            "planner": {"key": "gated", "algo": "rl", "kinematics": "diff"},
            "status": "not_available",
            "summary": {"written": 0, "failed_jobs": 0, "failures": []},
        },
    ]
    rollup = _build_arm_rollup(run_entries)
    assert rollup[0]["status"] == "not_available"
    assert "first_error" not in rollup[0]


def test_arm_rollup_empty_run_entries() -> None:
    """Empty run entries produce an empty rollup list."""
    rollup = _build_arm_rollup([])
    assert rollup == []


def test_arm_rollup_appears_in_campaign_report(tmp_path: Path) -> None:
    """Campaign report includes arm rollup table with error columns when failures exist."""
    payload: dict[str, Any] = {
        "campaign": {"campaign_id": "test", "name": "test_campaign", "status": "ok"},
        "planner_rows": [],
        "arm_rollup": [
            {
                "planner_key": "sfm",
                "algo": "sfm",
                "kinematics": "diff",
                "status": "ok",
                "episodes_written": 10,
                "episodes_failed": 0,
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "kinematics": "ackermann",
                "status": "failed",
                "episodes_written": 0,
                "episodes_failed": 3,
                "first_error": "RuntimeError('map resolution')",
                "distinct_error_count": 1,
            },
        ],
        "warnings": [],
    }
    report_path = tmp_path / "campaign_report.md"
    write_campaign_report(report_path, payload)
    text = report_path.read_text(encoding="utf-8")
    assert "## Arm Rollup" in text
    assert "sfm" in text
    assert "orca" in text
    assert "RuntimeError" in text
    assert "distinct_errors" in text


def test_arm_rollup_report_all_ok_no_error_columns(tmp_path: Path) -> None:
    """All-ok rollup renders without error columns in the report."""
    payload: dict[str, Any] = {
        "campaign": {"campaign_id": "test", "name": "test_campaign", "status": "ok"},
        "planner_rows": [],
        "arm_rollup": [
            {
                "planner_key": "sfm",
                "algo": "sfm",
                "kinematics": "diff",
                "status": "ok",
                "episodes_written": 10,
                "episodes_failed": 0,
            },
        ],
        "warnings": [],
    }
    report_path = tmp_path / "campaign_report.md"
    write_campaign_report(report_path, payload)
    text = report_path.read_text(encoding="utf-8")
    assert "## Arm Rollup" in text
    assert "first_error" not in text
    assert "distinct_errors" not in text
