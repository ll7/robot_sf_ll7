"""Regression tests for the end-to-end workflow factory acceptance harness."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.dev import workflow_factory_acceptance as harness


def test_acceptance_suite_runs_clean() -> None:
    """The complete factory acceptance suite runs clean in one call."""
    receipt = harness.run_acceptance_suite()

    assert receipt["schema"] == harness.RECEIPT_SCHEMA
    assert receipt["all_passed"] is True
    assert receipt["failed_count"] == 0
    assert receipt["passed_count"] == len(harness.ALL_SCENARIOS)
    assert receipt["total_scenarios"] == len(harness.ALL_SCENARIOS)
    assert receipt["failed_scenarios"] == []


@pytest.mark.parametrize("scenario_name", list(harness.ALL_SCENARIOS.keys()))
def test_each_scenario_executes_deterministically(scenario_name: str) -> None:
    """Each acceptance scenario executes and passes deterministically."""
    runner = harness.ALL_SCENARIOS[scenario_name]
    result = runner()

    assert result.passed is True
    assert result.failing_stage is None
    assert len(result.trace) > 0
    # Every step in the trace must have a non-empty owning helper, reason code, and next action
    for step in result.trace:
        assert step.stage
        assert step.state
        assert step.owning_helper
        assert step.reason_code
        assert step.expected_next_action


def test_failure_diagnostic_formatting() -> None:
    """A failed scenario emits the state, owning helper, reason code, and next action."""
    simulated_failure = harness.ScenarioResult(
        name="simulated_seam_regression",
        description="Simulated seam defect for trace validation",
        passed=False,
        failing_stage="admission",
        failing_state="claim_failed",
        owning_helper="scripts/dev/goal_issue_admission.py",
        reason_code="cannot_lock_ref_already_exists",
        expected_next_action="refresh_issue_queue",
        trace=[
            harness.TransitionStep(
                stage="issue",
                state="created",
                owning_helper="scripts/dev/issue_implementability.py",
                status="ok",
                reason_code="issue_created",
                expected_next_action="inspect_contract",
            ),
            harness.TransitionStep(
                stage="admission",
                state="claim_failed",
                owning_helper="scripts/dev/goal_issue_admission.py",
                status="failed",
                reason_code="cannot_lock_ref_already_exists",
                expected_next_action="refresh_issue_queue",
            ),
        ],
        error="Simulated network lock collision",
    )

    formatted = harness.format_failure_diagnostic(simulated_failure)

    assert "[SCENARIO FAILED] simulated_seam_regression" in formatted
    assert "Lifecycle Stage:      admission" in formatted
    assert "State:                claim_failed" in formatted
    assert "Owning Helper:        scripts/dev/goal_issue_admission.py" in formatted
    assert "Reason Code:          cannot_lock_ref_already_exists" in formatted
    assert "Expected Next Action: refresh_issue_queue" in formatted
    assert "Error Detail:         Simulated network lock collision" in formatted
    assert "1. [issue] state=created" in formatted
    assert "2. [admission] state=claim_failed" in formatted


def test_cli_json_mode(capsys: Any) -> None:
    """CLI runs in --json mode and outputs valid JSON matching the schema."""
    code = harness.main(["--json"])
    assert code == 0

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["schema"] == harness.RECEIPT_SCHEMA
    assert data["all_passed"] is True
    assert data["total_scenarios"] == 12


def test_cli_scenario_filter(capsys: Any) -> None:
    """CLI supports filtering by scenario name."""
    code = harness.main(["--scenario", "happy_path_e2e", "--verbose"])
    assert code == 0

    out = capsys.readouterr().out
    assert "[PASS] happy_path_e2e" in out
    assert "1. [issue] state=created -> ok" in out


def test_cli_unknown_scenario_fails(capsys: Any) -> None:
    """CLI fails closed with exit code 1 when given an unknown scenario."""
    code = harness.main(["--scenario", "unknown_scenario_xyz"])
    assert code == 1

    out = capsys.readouterr().out
    assert "Acceptance harness failed with 1 failure(s)." in out
