"""Focused tests for the issue #9431 release comparator."""

from __future__ import annotations

from scripts.analysis.compare_issue_9431_release import _execution_audit, _markdown


def test_execution_audit_admits_mixed_mode_and_outcome_failures() -> None:
    row = {
        "status": "collision",
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "mixed"},
            "fallback_used": False,
            "degraded": False,
            "unavailable": False,
        },
    }

    assert _execution_audit(row) == []


def test_execution_audit_rejects_fallback_and_runtime_error() -> None:
    row = {
        "status": "runtime_error",
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "native"},
            "fallback_used": True,
        },
    }

    assert _execution_audit(row) == ["fallback_used=true", "status=runtime_error"]


def test_markdown_records_exact_source_range_and_execution_modes() -> None:
    report = {
        "predecessor": {
            "archive_sha256": "old-archive",
            "source_commit": "old-source",
        },
        "successor": {
            "source_commit": "new-source",
            "bundle_sha256": "new-bundle",
        },
        "paired_rows": 0,
        "changed_episode_count": 0,
        "execution_audit": {"admitted": True},
        "arms": {},
        "seed_block_bootstrap": {},
    }

    rendered = _markdown(report)

    assert "`old-source..new-source`" in rendered
    assert "native, adapter, or mixed mode" in rendered
