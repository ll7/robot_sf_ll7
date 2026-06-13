"""Tests for machine-checkable PR loop policy."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.dev.pr_loop_policy import (
    PolicyDecision,
    _review_state,
    classify_pr_state,
    evaluate_queue,
    format_text,
    main,
    recommend_action,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pr_loop_policy"


def _pr(
    number: int,
    *,
    overall: str = "success",
    labels: list[str] | None = None,
    draft: bool = False,
    head_sha: str = "abc123",
    expected_head_sha: str = "",
    status: str = "ok",
    artifacts: bool | None = None,
) -> dict[str, object]:
    """Build a compact PR snapshot dict for testing."""
    result: dict[str, object] = {
        "number": number,
        "status": status,
        "draft": draft,
        "head_sha": head_sha,
        "labels": labels or [],
        "checks": {
            "overall": overall,
            "total": 1,
            "by_conclusion": {overall: 1},
            "by_status": {"completed": 1} if overall != "pending" else {"in_progress": 1},
            "names": ["ci"],
        },
    }
    if expected_head_sha:
        result["expected_head_sha"] = expected_head_sha
    if artifacts is not None:
        result["artifacts"] = artifacts
    return result


def _snapshot(prs: list[dict[str, object]]) -> dict[str, object]:
    """Build a snapshot dict wrapping multiple PRs."""
    return {"schema": "pr_queue_snapshot.v1", "prs": prs}


# ---------------------------------------------------------------------------
# classify_pr_state
# ---------------------------------------------------------------------------


def test_classify_pending_ci() -> None:
    """Pending CI should classify as pending_ci."""
    pr = _pr(100, overall="pending")
    assert classify_pr_state(pr) == "pending_ci"


def test_classify_failed_ci() -> None:
    """Failed CI should classify as failed_ci."""
    pr = _pr(101, overall="failure")
    assert classify_pr_state(pr) == "failed_ci"


def test_classify_stale_worktree() -> None:
    """Head SHA mismatch should classify as stale_worktree."""
    pr = _pr(102, head_sha="new-sha", expected_head_sha="old-sha")
    assert classify_pr_state(pr) == "stale_worktree"


def test_classify_missing_artifacts() -> None:
    """Artifacts present but empty should classify as missing_artifacts."""
    pr = _pr(103, artifacts=False)
    assert classify_pr_state(pr) == "missing_artifacts"


def test_classify_ready_to_merge() -> None:
    """CI green + merge-ready label should classify as ready_to_merge."""
    pr = _pr(104, overall="success", labels=["merge-ready"])
    assert classify_pr_state(pr) == "ready_to_merge"


def test_classify_no_action_default() -> None:
    """No special conditions should classify as no_action."""
    pr = _pr(105)
    assert classify_pr_state(pr) == "no_action"


def test_classify_draft_is_no_action() -> None:
    """Draft PRs should always be no_action."""
    pr = _pr(106, draft=True, overall="pending")
    assert classify_pr_state(pr) == "no_action"


def test_classify_error_status_is_no_action() -> None:
    """Error status PRs should classify as no_action."""
    pr = {"number": 107, "status": "error", "error": "gh failed"}
    assert classify_pr_state(pr) == "no_action"


def test_classify_non_dict_input() -> None:
    """Non-dict input should classify as no_action."""
    assert classify_pr_state("not a dict") == "no_action"  # type: ignore[arg-type]
    assert classify_pr_state(None) == "no_action"  # type: ignore[arg-type]


def test_classify_success_without_merge_ready_is_no_action() -> None:
    """CI green but no merge-ready label should be no_action."""
    pr = _pr(108, overall="success", labels=["needs-review"])
    assert classify_pr_state(pr) == "no_action"


def test_classify_artifacts_true_not_missing() -> None:
    """Artifacts present and True should not be missing_artifacts."""
    pr = _pr(109, overall="success", artifacts=True)
    assert classify_pr_state(pr) == "no_action"


# ---------------------------------------------------------------------------
# recommend_action
# ---------------------------------------------------------------------------


def test_recommend_wait_ci_for_pending() -> None:
    """Pending CI should recommend wait_ci."""
    decision = recommend_action("pending_ci", pr_number=200, actions_remaining=3)
    assert decision.action == "wait_ci"
    assert decision.actions_remaining == 2


def test_recommend_inspect_for_failed_ci() -> None:
    """Failed CI should recommend inspect_failed_ci."""
    decision = recommend_action("failed_ci", pr_number=201, actions_remaining=3)
    assert decision.action == "inspect_failed_ci"


def test_recommend_verify_for_missing_artifacts() -> None:
    """Missing artifacts should recommend verify_artifacts."""
    decision = recommend_action("missing_artifacts", pr_number=202, actions_remaining=3)
    assert decision.action == "verify_artifacts"


def test_recommend_refresh_for_stale() -> None:
    """Stale worktree should recommend refresh_snapshot."""
    decision = recommend_action("stale_worktree", pr_number=203, actions_remaining=3)
    assert decision.action == "refresh_snapshot"


def test_recommend_mark_ready_for_ready() -> None:
    """Ready-to-merge should recommend mark_ready_candidate."""
    decision = recommend_action("ready_to_merge", pr_number=204, actions_remaining=3)
    assert decision.action == "mark_ready_candidate"


def test_recommend_no_action_for_default() -> None:
    """No-action state should recommend no_action."""
    decision = recommend_action("no_action", pr_number=205, actions_remaining=3)
    assert decision.action == "no_action"


def test_recommend_stop_when_budget_exhausted() -> None:
    """Zero remaining budget should recommend stop."""
    decision = recommend_action("pending_ci", pr_number=206, actions_remaining=0)
    assert decision.action == "stop"
    assert decision.actions_remaining == 0


def test_recommend_stop_negative_budget() -> None:
    """Negative budget should also recommend stop."""
    decision = recommend_action("pending_ci", pr_number=207, actions_remaining=-1)
    assert decision.action == "stop"
    assert decision.actions_remaining == 0


# ---------------------------------------------------------------------------
# evaluate_queue
# ---------------------------------------------------------------------------


def test_evaluate_queue_single_pr() -> None:
    """Single PR queue should produce one decision."""
    prs = [_pr(300, overall="pending")]
    result = evaluate_queue(prs, max_actions=3)
    assert result["schema"] == "pr_loop_policy.v1"
    assert result["max_actions"] == 3
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["action"] == "wait_ci"


def test_evaluate_queue_budget_enforced() -> None:
    """Budget should cap the number of actions produced before stop."""
    prs = [_pr(400, overall="pending") for _ in range(10)]
    result = evaluate_queue(prs, max_actions=2)
    assert result["actions_used"] == 2
    assert len(result["decisions"]) == 3
    assert result["decisions"][0]["action"] == "wait_ci"
    assert result["decisions"][1]["action"] == "wait_ci"
    assert result["decisions"][2]["action"] == "stop"


def test_evaluate_queue_stop_on_budget_exhaustion() -> None:
    """Budget=1 should process one PR then stop before the second."""
    prs = [_pr(500, overall="pending"), _pr(501, overall="failure")]
    result = evaluate_queue(prs, max_actions=1)
    assert result["actions_used"] == 1
    assert result["decisions"][0]["action"] == "wait_ci"
    assert result["decisions"][1]["action"] == "stop"


def test_evaluate_queue_empty() -> None:
    """Empty queue should produce zero decisions."""
    result = evaluate_queue([], max_actions=5)
    assert result["decisions"] == []
    assert result["actions_used"] == 0


def test_evaluate_queue_expected_sha_injection() -> None:
    """Expected head SHAs should be injected for staleness detection."""
    prs = [_pr(600, head_sha="new")]
    result = evaluate_queue(
        prs,
        max_actions=3,
        expected_head_shas={600: "old"},
    )
    assert result["decisions"][0]["state"] == "stale_worktree"
    assert result["decisions"][0]["action"] == "refresh_snapshot"


def test_evaluate_queue_artifact_presence_injection() -> None:
    """Artifact presence should be injected for missing-artifact detection."""
    prs = [_pr(700, overall="success", labels=["merge-ready"])]
    result = evaluate_queue(
        prs,
        max_actions=3,
        artifact_presence={700: False},
    )
    # Artifact check overrides ready_to_merge when artifacts are missing
    assert result["decisions"][0]["state"] == "missing_artifacts"


def test_evaluate_queue_mixed_states() -> None:
    """Mixed queue should produce correct per-PR decisions."""
    prs = [
        _pr(800, overall="pending"),
        _pr(801, overall="failure"),
        _pr(802, overall="success", labels=["merge-ready"]),
        _pr(803),
    ]
    result = evaluate_queue(prs, max_actions=10)
    actions = [d["action"] for d in result["decisions"]]
    assert actions == ["wait_ci", "inspect_failed_ci", "mark_ready_candidate", "no_action"]


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


def test_policy_decision_to_dict() -> None:
    """PolicyDecision should serialize deterministically."""
    d = PolicyDecision(
        pr=1,
        action="stop",
        state="pending_ci",
        flow_decision="stop",
        reason="budget",
        actions_remaining=0,
    )
    out = d.to_dict()
    assert out["pr"] == 1
    assert out["action"] == "stop"
    assert out["flow_decision"] == "stop"
    assert set(out.keys()) == {
        "pr",
        "action",
        "state",
        "flow_decision",
        "reason",
        "actions_remaining",
    }


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


def test_format_text_contains_actions() -> None:
    """Text output should list action, state, and flow for each decision."""
    result = {
        "max_actions": 5,
        "actions_used": 1,
        "decisions": [
            {
                "pr": 900,
                "action": "wait_ci",
                "state": "pending_ci",
                "flow_decision": "continue",
                "reason": "CI pending",
                "actions_remaining": 4,
            }
        ],
    }
    text = format_text(result)
    assert "PR #900" in text
    assert "wait_ci" in text
    assert "pending_ci" in text
    assert "flow=continue" in text


def test_format_text_empty_queue() -> None:
    """Text output should handle empty queue."""
    result = {"max_actions": 5, "actions_used": 0, "decisions": []}
    text = format_text(result)
    assert "actions_used: 0" in text


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


def test_main_stdin_json(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI should read snapshot from stdin and emit JSON."""
    snapshot = _snapshot([_pr(1000, overall="pending")])
    stdin = StringIO(json.dumps(snapshot))
    with patch("sys.stdin", stdin):
        rc = main(["--stdin", "--json"])
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decisions"][0]["action"] == "wait_ci"


def test_main_snapshot_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI should read snapshot from a file."""
    snapshot = _snapshot([_pr(1100, overall="failure")])
    snap_file = tmp_path / "queue.json"
    snap_file.write_text(json.dumps(snapshot))
    rc = main(["--snapshot", str(snap_file), "--json"])
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decisions"][0]["action"] == "inspect_failed_ci"


def test_main_missing_file() -> None:
    """CLI should fail on missing snapshot file."""
    rc = main(["--snapshot", "/nonexistent/file.json"])
    assert rc == 1


def test_main_invalid_json(tmp_path: Path) -> None:
    """CLI should fail on invalid JSON."""
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")
    rc = main(["--snapshot", str(bad)])
    assert rc == 1


def test_main_no_snapshot_arg() -> None:
    """CLI should fail when no snapshot source is provided."""
    rc = main(["--json"])
    assert rc == 1


def test_main_text_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI should emit human-readable text without --json."""
    snapshot = _snapshot([_pr(1200, overall="success", labels=["merge-ready"])])
    snap_file = tmp_path / "queue.json"
    snap_file.write_text(json.dumps(snapshot))
    rc = main(["--snapshot", str(snap_file)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "PR #1200" in text
    assert "mark_ready_candidate" in text


def test_main_max_actions_flag(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI --max-actions should control the loop budget."""
    snapshot = _snapshot([_pr(1300, overall="pending"), _pr(1301, overall="pending")])
    snap_file = tmp_path / "queue.json"
    snap_file.write_text(json.dumps(snapshot))
    rc = main(["--snapshot", str(snap_file), "--max-actions", "1", "--json"])
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["max_actions"] == 1
    assert output["actions_used"] == 1
    assert output["decisions"][0]["action"] == "wait_ci"
    assert output["decisions"][1]["action"] == "stop"


def test_main_expected_sha_pairs(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI --expected-sha should inject staleness detection."""
    snapshot = _snapshot([_pr(1400, head_sha="new")])
    snap_file = tmp_path / "queue.json"
    snap_file.write_text(json.dumps(snapshot))
    rc = main(["--snapshot", str(snap_file), "--expected-sha", "1400=old", "--json"])
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decisions"][0]["state"] == "stale_worktree"


def test_main_artifact_present_pairs(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI --artifact-present should inject artifact detection."""
    snapshot = _snapshot([_pr(1500, overall="success", labels=["merge-ready"])])
    snap_file = tmp_path / "queue.json"
    snap_file.write_text(json.dumps(snapshot))
    rc = main(["--snapshot", str(snap_file), "--artifact-present", "1500=false", "--json"])
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decisions"][0]["state"] == "missing_artifacts"


def test_main_cli_dry_run_no_gh(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI dry-run should never call gh, even with a realistic snapshot."""
    snapshot = _snapshot(
        [
            _pr(1600, overall="pending"),
            _pr(1601, overall="failure"),
            _pr(1602, overall="success", labels=["merge-ready"]),
        ]
    )
    snap_file = tmp_path / "queue.json"
    snap_file.write_text(json.dumps(snapshot))
    with patch("subprocess.run") as mock_run:
        rc = main(["--snapshot", str(snap_file), "--json"])
    mock_run.assert_not_called()
    assert rc == 0


def test_main_stdin_invalid_json() -> None:
    """CLI should fail on invalid stdin JSON."""
    stdin = StringIO("not json")
    with patch("sys.stdin", stdin):
        rc = main(["--stdin"])
    assert rc == 1


def test_main_non_array_snapshot(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI should fail when snapshot is not an array or does not contain prs."""
    snap_file = tmp_path / "bad.json"
    snap_file.write_text(json.dumps({"not_prs": []}))
    rc = main(["--snapshot", str(snap_file)])
    assert rc == 1


# ---------------------------------------------------------------------------
# _review_state
# ---------------------------------------------------------------------------


def test_review_state_scalar() -> None:
    """Scalar review_state field should be returned uppercased."""
    pr = {"number": 9001, "review_state": "approved"}
    assert _review_state(pr) == "APPROVED"


def test_review_state_scalar_review_field() -> None:
    """Fallback review field should be returned uppercased."""
    pr = {"number": 9002, "review": "changes_requested"}
    assert _review_state(pr) == "CHANGES_REQUESTED"


def test_review_state_dict_changes_requested() -> None:
    """reviews dict with CHANGES_REQUESTED > 0 should return CHANGES_REQUESTED."""
    pr = {"number": 9003, "reviews": {"CHANGES_REQUESTED": 1, "APPROVED": 2}}
    assert _review_state(pr) == "CHANGES_REQUESTED"


def test_review_state_dict_no_changes_requested() -> None:
    """reviews dict without CHANGES_REQUESTED should return first nonzero key."""
    pr = {"number": 9004, "reviews": {"APPROVED": 1, "COMMENTED": 3}}
    assert _review_state(pr) == "APPROVED"


def test_review_state_dict_all_zero() -> None:
    """reviews dict with all zero counts should fall through to scalar."""
    pr = {"number": 9005, "reviews": {"APPROVED": 0, "COMMENTED": 0}, "review": "commented"}
    assert _review_state(pr) == "COMMENTED"


def test_review_state_empty() -> None:
    """No review info should return empty string."""
    pr = {"number": 9006}
    assert _review_state(pr) == ""


def test_review_state_dict_takes_priority_over_scalar() -> None:
    """reviews dict should take priority over scalar review_state."""
    pr = {
        "number": 9007,
        "review_state": "approved",
        "reviews": {"CHANGES_REQUESTED": 1},
    }
    assert _review_state(pr) == "CHANGES_REQUESTED"


# ---------------------------------------------------------------------------
# flow_decision
# ---------------------------------------------------------------------------


def test_flow_decision_budget_exhausted() -> None:
    """Budget exhausted should always yield stop."""
    decision = recommend_action("pending_ci", pr_number=9100, actions_remaining=0)
    assert decision.flow_decision == "stop"


def test_flow_decision_pending_ci_continues() -> None:
    """Pending CI with budget should yield continue."""
    decision = recommend_action("pending_ci", pr_number=9101, actions_remaining=3)
    assert decision.flow_decision == "continue"


def test_flow_decision_ready_to_merge_continues() -> None:
    """Ready-to-merge with budget should yield continue."""
    decision = recommend_action("ready_to_merge", pr_number=9102, actions_remaining=3)
    assert decision.flow_decision == "continue"


def test_flow_decision_failed_ci_reroutes() -> None:
    """Failed CI should yield reroute."""
    decision = recommend_action("failed_ci", pr_number=9103, actions_remaining=3)
    assert decision.flow_decision == "reroute"


def test_flow_decision_missing_artifacts_reroutes() -> None:
    """Missing artifacts should yield reroute."""
    decision = recommend_action("missing_artifacts", pr_number=9104, actions_remaining=3)
    assert decision.flow_decision == "reroute"


def test_flow_decision_stale_worktree_reroutes() -> None:
    """Stale worktree should yield reroute."""
    decision = recommend_action("stale_worktree", pr_number=9105, actions_remaining=3)
    assert decision.flow_decision == "reroute"


def test_flow_decision_no_action_stops() -> None:
    """No-action state should yield stop."""
    decision = recommend_action("no_action", pr_number=9106, actions_remaining=3)
    assert decision.flow_decision == "stop"


def test_flow_decision_changes_requested_escalates_pending() -> None:
    """CHANGES_REQUESTED review on pending_ci should escalate, not continue."""
    decision = recommend_action(
        "pending_ci",
        pr_number=9107,
        actions_remaining=3,
        review_state="CHANGES_REQUESTED",
    )
    assert decision.flow_decision == "escalate"
    assert decision.action == "escalate"


def test_flow_decision_changes_requested_escalates_failed_ci() -> None:
    """CHANGES_REQUESTED review on failed_ci should escalate instead of reroute."""
    decision = recommend_action(
        "failed_ci",
        pr_number=9108,
        actions_remaining=3,
        review_state="CHANGES_REQUESTED",
    )
    assert decision.flow_decision == "escalate"
    assert decision.action == "escalate"


def test_flow_decision_changes_requested_escalates_missing_artifacts() -> None:
    """CHANGES_REQUESTED review on missing_artifacts should escalate."""
    decision = recommend_action(
        "missing_artifacts",
        pr_number=9109,
        actions_remaining=3,
        review_state="CHANGES_REQUESTED",
    )
    assert decision.flow_decision == "escalate"
    assert decision.action == "escalate"


def test_flow_decision_changes_requested_escalates_stale() -> None:
    """CHANGES_REQUESTED review on stale_worktree should escalate."""
    decision = recommend_action(
        "stale_worktree",
        pr_number=9110,
        actions_remaining=3,
        review_state="CHANGES_REQUESTED",
    )
    assert decision.flow_decision == "escalate"
    assert decision.action == "escalate"


# ---------------------------------------------------------------------------
# evaluate_queue with reviews
# ---------------------------------------------------------------------------


def test_evaluate_queue_reviews_dict_escalation() -> None:
    """PR with CHANGES_REQUESTED in reviews dict should produce escalate flow."""
    prs = [
        {
            "number": 9200,
            "status": "ok",
            "draft": False,
            "head_sha": "sha",
            "labels": [],
            "reviews": {"CHANGES_REQUESTED": 1},
            "checks": {"overall": "pending"},
        }
    ]
    result = evaluate_queue(prs, max_actions=3)
    d = result["decisions"][0]
    assert d["flow_decision"] == "escalate"
    assert d["action"] == "escalate"


def test_evaluate_queue_reviews_dict_approved_continue() -> None:
    """PR with only APPROVED in reviews dict should continue if state allows."""
    prs = [
        {
            "number": 9201,
            "status": "ok",
            "draft": False,
            "head_sha": "sha",
            "labels": ["merge-ready"],
            "reviews": {"APPROVED": 2},
            "checks": {"overall": "success"},
        }
    ]
    result = evaluate_queue(prs, max_actions=3)
    d = result["decisions"][0]
    assert d["flow_decision"] == "continue"
    assert d["action"] == "mark_ready_candidate"


# ---------------------------------------------------------------------------
# Fixture file test
# ---------------------------------------------------------------------------


def test_fixture_file_comprehensive_queue(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A comprehensive fixture file should exercise all states under budget."""
    fixture = FIXTURE_DIR / "comprehensive_queue.json"
    if not fixture.exists():
        pytest.skip("fixture file not present")
    dest = tmp_path / "fixture_copy.json"
    dest.write_text(fixture.read_text())
    rc = main(["--snapshot", str(dest), "--json"])
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "pr_loop_policy.v1"
    states_seen = {d["state"] for d in output["decisions"]}
    assert len(states_seen) >= 2
    flow_decisions = {d["flow_decision"] for d in output["decisions"]}
    assert flow_decisions.issubset({"stop", "continue", "reroute", "escalate"})
    pr2001 = next(d for d in output["decisions"] if d["pr"] == 2001)
    assert pr2001["flow_decision"] == "escalate"
    pr2002 = next(d for d in output["decisions"] if d["pr"] == 2002)
    assert pr2002["flow_decision"] == "escalate"
