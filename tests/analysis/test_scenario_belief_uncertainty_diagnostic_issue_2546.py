"""Diagnostic acceptance tests for issue #2546 ScenarioBelief uncertainty stress.

These tests assert the bounded contract of the diagnostic runner:

1. every belief condition produces an uncertainty report,
2. the unsupported planner fails closed (never consumes uncertainty),
3. at least one uncertain condition changes a planner decision / failure
   predicate versus oracle, and
4. the report is labeled diagnostic-only / stress and carries a follow-up
   decision.

Claim boundary: diagnostic only / stress. No benchmark, safety, planner
performance, perception, or paper-facing claim.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "run_scenario_belief_uncertainty_diagnostic_issue_2546.py"
)

_spec = importlib.util.spec_from_file_location("_issue_2546_diag", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


@pytest.fixture(scope="module")
def report() -> dict:
    """Run the diagnostic once for the whole module."""
    return diag.run_diagnostic(seed=diag.DEFAULT_SEED)


def test_all_five_conditions_present(report: dict) -> None:
    """The diagnostic must cover all five declared belief conditions plus oracle baseline."""
    expected = {
        "oracle",
        "visibility_limited",
        "covariance_inflated",
        "class_probability",
        "existence_degraded",
    }
    assert set(report["conditions"]) == expected


def test_every_condition_produces_uncertainty_report(report: dict) -> None:
    """Each condition must emit a structured uncertainty report with the required fields."""
    required_top = {"schema_version", "ego", "agents"}
    required_agent = {
        "entity_id",
        "class_probabilities",
        "position_covariance_xy",
        "existence_probability",
        "visibility_state",
    }
    for name, result in report["conditions"].items():
        ur = result["uncertainty_report"]
        assert required_top <= set(ur), f"{name} report missing top-level fields"
        for agent_row in ur["agents"]:
            assert required_agent <= set(agent_row), f"{name} agent row missing fields"


def test_unsupported_planner_fails_closed_for_every_condition(report: dict) -> None:
    """The unsupported planner key must fail closed and never consume uncertainty."""
    for name, result in report["conditions"].items():
        unsupported = result["unsupported_planner"]
        assert unsupported["fail_closed"] is True, f"{name} unsupported did not fail closed"
        assert unsupported["uncertainty_consumed"] is False, f"{name} consumed uncertainty"
        assert unsupported["compatibility"]["reason"] == "unsupported_uncertainty_planner", (
            f"{name} wrong fail-closed reason"
        )
    assert report["unsupported_fail_closed_ok"] is True


def test_consuming_planner_reports_compatible(report: dict) -> None:
    """The stream_gap consuming planner must report a compatible uncertainty sidecar."""
    for name, result in report["conditions"].items():
        compat = result["consuming_planner"]["compatibility"]
        assert compat["status"] == "compatible", f"{name} consuming not compatible"
        assert compat["uncertainty_consumed"] is True, f"{name} did not consume uncertainty"


def test_at_least_one_behavior_difference_vs_oracle(report: dict) -> None:
    """At least one uncertain condition must change a failure predicate versus oracle."""
    assert report["any_behavior_difference"] is True
    nonempty = [name for name, diff in report["predicate_diffs_vs_oracle"].items() if diff]
    assert nonempty, "expected at least one condition with a predicate change vs oracle"


def test_specific_visibility_limited_difference(report: dict) -> None:
    """The visibility-limited condition must flip the planner off the oracle wait decision."""
    diff = report["predicate_diffs_vs_oracle"]["visibility_limited"]
    # Oracle waits for the corridor pedestrian; dropping it changes the linear command.
    assert "linear_speed" in diff
    assert diff["linear_speed"]["oracle"] != diff["linear_speed"]["condition"]


def test_uncertainty_gate_drops_corridor_agent_under_degraded_uncertainty(report: dict) -> None:
    """Covariance/class/existence degradation must trigger the stream_gap uncertainty gate."""
    for name in ("covariance_inflated", "class_probability", "existence_degraded"):
        gate = report["conditions"][name]["consuming_planner"]["uncertainty_gate"]
        assert gate["status"] == "applied"
        assert gate["dropped_count"] >= 1, f"{name} did not drop any agent via gate"


def test_oracle_gate_drops_nothing(report: dict) -> None:
    """The oracle condition must not drop any agent through the uncertainty gate."""
    gate = report["conditions"]["oracle"]["consuming_planner"]["uncertainty_gate"]
    assert gate["dropped_count"] == 0


def test_report_is_labeled_diagnostic_only(report: dict) -> None:
    """The report must carry honest diagnostic-only / stress labels, not benchmark claims."""
    assert report["claim_boundary"] == "diagnostic_only"
    assert report["evidence_tier"] == "stress"
    assert report["paper_grade"] is False
    assert report["not_benchmark_evidence"] is True


def test_follow_up_decision_is_valid_and_continue(report: dict) -> None:
    """The follow-up decision must be one of continue|revise|stop with a rationale."""
    fu = report["follow_up_decision"]
    assert fu["decision"] in {"continue", "revise", "stop"}
    assert fu["rationale"].strip()
    # Differences were found and unsupported consumption failed closed -> continue.
    assert fu["decision"] == "continue"


def test_report_is_json_serializable(report: dict) -> None:
    """The full report must serialize to JSON deterministically."""
    text = json.dumps(report, sort_keys=True)
    assert json.loads(text)["issue"] == diag.ISSUE


def test_determinism_across_runs() -> None:
    """Two runs at the same seed must produce identical decision-relevant output."""
    first = diag.run_diagnostic(seed=diag.DEFAULT_SEED)
    second = diag.run_diagnostic(seed=diag.DEFAULT_SEED)
    assert first["predicate_diffs_vs_oracle"] == second["predicate_diffs_vs_oracle"]
    assert first["follow_up_decision"] == second["follow_up_decision"]
