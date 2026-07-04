"""Tests for issue #4017 constrained-RL readiness consolidation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.analysis.check_constrained_rl_readiness_issue_4017 import (
    BLOCKED_EXIT_CODE,
    assess_readiness,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_ready_report_maps_to_next_empirical_action(tmp_path: Path) -> None:
    """A complete diagnostic report is ready for the next empirical campaign."""
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")

    readiness = assess_readiness(report_path)

    assert readiness["status"] == "diagnostic_ready_for_empirical_campaign"
    assert readiness["ready_for_empirical_campaign"] is True
    assert readiness["ready_for_benchmark_claim"] is False
    assert readiness["blockers_remaining"] == []
    assert "paired constrained/unconstrained empirical campaign" in str(
        readiness["next_empirical_action"]
    )


def test_blocked_report_preserves_remaining_blockers(tmp_path: Path) -> None:
    """Missing runtime and absent multiplier updates block stronger handoff."""
    report = _report()
    report["status"] = "diagnostic_blocked"
    report["blockers"] = ["constrained trace has no completed episode multiplier records"]
    report["baseline"]["runtime_status"] = "missing_from_manifest"
    report["constraint_effect"]["multiplier_changed_constraints"] = []
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    readiness = assess_readiness(report_path)

    assert readiness["status"] == "diagnostic_blocked"
    assert readiness["ready_for_empirical_campaign"] is False
    assert "comparison report status is not diagnostic_ready" in readiness["blockers_remaining"]
    assert "baseline manifest does not record runtime_seconds" in readiness["blockers_remaining"]
    assert "no Lagrange multiplier update was observed" in readiness["blockers_remaining"]
    assert "Regenerate the matched CPU-smoke manifests" in readiness["next_empirical_action"]


def test_cli_exit_code_allows_explicit_blocked_diagnostic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI exits non-zero for blocked evidence unless explicitly allowed."""
    report = _report()
    report["fallback_or_degraded"] = True
    report_path = tmp_path / "comparison_report.json"
    output_path = tmp_path / "readiness.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert main(["--report", str(report_path), "--output", str(output_path)]) == BLOCKED_EXIT_CODE
    assert main(["--report", str(report_path), "--allow-blocked"]) == 0
    captured = capsys.readouterr()
    assert "diagnostic_blocked" in captured.out
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "diagnostic_blocked"


def _report() -> dict[str, object]:
    run_summary = {
        "dry_run": False,
        "runtime_status": "recorded",
        "fallback_or_degraded": False,
        "missing_fields": [],
    }
    return {
        "schema_version": "issue_4017.constrained_rl_diagnostic.v1",
        "issue": 4017,
        "evidence_tier": "diagnostic-only",
        "status": "diagnostic_ready",
        "blockers": [],
        "fallback_or_degraded": False,
        "baseline": dict(run_summary),
        "constrained": dict(run_summary),
        "constraint_effect": {
            "interpretation": "diagnostic_only",
            "matched_seed": True,
            "matched_total_timesteps": True,
            "benchmark_safety_claim": False,
            "budget_violation_constraints": ["collision_any"],
            "multiplier_changed_constraints": ["collision_any"],
        },
        "constraint_trace": {"record_count": 1},
    }
