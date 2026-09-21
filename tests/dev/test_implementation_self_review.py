"""Tests for exact-diff implementation self-review receipts (issue #9537).

All paths run offline with injected payloads and runners: no GitHub
access, no worktree mutation, no label writes. A receipt is handoff proof
only when it is exact-head and exact-diff bound, honestly reports executed
validation, and carries no unresolved blocking finding.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from typing import Any

import pytest

from scripts.dev import implementation_self_review as self_review

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
ISSUE_BODY = "## Objective\nFix the lifecycle drift.\n\n## Scope\nBounded.\n"
CONTRACT_DIGEST = hashlib.sha256(ISSUE_BODY.encode("utf-8")).hexdigest()
DIFF_TEXT = "diff --git a/scripts/dev/example.py b/scripts/dev/example.py\n"
DIFF_DIGEST = hashlib.sha256(DIFF_TEXT.encode("utf-8")).hexdigest()


def _checks(*, failed: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Build one passing verdict per required check id, failing the named ones."""
    return [
        {
            "id": check_id,
            "verdict": "fail" if check_id in failed else "pass",
            "evidence": f"reviewed {check_id}; no issue found"
            if check_id not in failed
            else f"reviewed {check_id}; defect confirmed",
        }
        for check_id in self_review.CHECK_IDS
    ]


def _declaration(**overrides: Any) -> dict[str, Any]:
    """Build one valid receipt declaration for offline tests."""
    payload: dict[str, Any] = {
        "repository": "ll7/robot_sf_ll7",
        "issue": 9537,
        "contract": {"source": "issue-body", "digest": CONTRACT_DIGEST},
        "delivery": {
            "base_ref": "origin/main",
            "base_sha": BASE_SHA,
            "head_sha": HEAD_SHA,
            "branch": "issue-9537-exact-diff-self-review",
            "worktree": "/tmp/issue-9537-exact-diff-self-review",
        },
        "diff": {
            "changed_paths": ["scripts/dev/example.py"],
            "stat": {"files": 1, "additions": 10, "deletions": 2},
            "diff_digest": DIFF_DIGEST,
        },
        "checks": _checks(),
        "validation": [
            {"command": "pytest tests/dev/test_example.py -q", "exit_code": 0, "result": "passed"}
        ],
        "findings": {"blocking": [], "non_blocking": ["follow-up: polish docs"]},
        "producer": {"identity": "implementation-agent"},
        "claim_boundary": "implementation-quality proof only; not review or approval",
    }
    payload.update(overrides)
    return payload


def _git_runner(
    *,
    head_sha: str = HEAD_SHA,
    base_sha: str = BASE_SHA,
    branch: str = "issue-9537-exact-diff-self-review",
    name_only: str = "scripts/dev/example.py\n",
    diff: str = DIFF_TEXT,
) -> Any:
    """Build an injectable Git runner with exact per-command outputs."""

    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["git", "rev-parse"]:
            if command[2] == "HEAD":
                return subprocess.CompletedProcess(command, 0, head_sha + "\n", "")
            return subprocess.CompletedProcess(command, 0, base_sha + "\n", "")
        if command[:2] == ["git", "branch"]:
            return subprocess.CompletedProcess(command, 0, branch + "\n", "")
        if command[:3] == ["git", "diff", "--name-only"]:
            return subprocess.CompletedProcess(command, 0, name_only, "")
        if command[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(command, 0, diff, "")
        raise AssertionError(f"unexpected git command: {command}")

    return _run


# Valid receipt and handoff.


def test_valid_receipt_builds_and_passes_handoff() -> None:
    """A complete honest receipt digests, validates, and authorizes handoff."""
    receipt = self_review.build_receipt(_declaration())

    assert receipt["schema"] == self_review.SCHEMA
    assert self_review.validate_receipt(receipt)["ok"] is True
    decision = self_review.handoff_decision(
        receipt,
        expected_issue=9537,
        expected_base_sha=BASE_SHA,
        expected_head_sha=HEAD_SHA,
    )

    assert decision == {"ok": True, "reasons": []}


def test_receipt_digest_binds_canonical_payload() -> None:
    """Any post-build mutation invalidates the recorded digest."""
    receipt = self_review.build_receipt(_declaration())
    tampered = dict(receipt)
    tampered["findings"] = {"blocking": [], "non_blocking": ["edited after build"]}

    result = self_review.validate_receipt(tampered)

    assert result["ok"] is False
    assert "receipt_digest does not match the canonical receipt payload" in result["errors"]


# Drift invalidation.


def test_moved_head_invalidates_handoff() -> None:
    """A head move after self-review refuses handoff without re-review."""
    receipt = self_review.build_receipt(_declaration())

    decision = self_review.handoff_decision(
        receipt,
        expected_issue=9537,
        expected_base_sha=BASE_SHA,
        expected_head_sha="c" * 40,
    )

    assert decision["ok"] is False
    assert any("expected head" in reason for reason in decision["reasons"])


def test_changed_diff_invalidates_verification() -> None:
    """A diff change after self-review fails worktree verification."""
    receipt = self_review.build_receipt(_declaration())
    runner = _git_runner(diff=DIFF_TEXT + "# concurrent edit\n")

    result = self_review.verify_receipt_against_git(
        receipt, worktree="/tmp/issue-9537-exact-diff-self-review", git_runner=runner
    )

    assert result["ok"] is False
    assert any("diff_digest" in error for error in result["errors"])


def test_changed_issue_body_invalidates_contract() -> None:
    """An issue-contract change invalidates the receipt binding."""
    receipt = self_review.build_receipt(_declaration())

    result = self_review.validate_receipt(receipt, issue_contract=ISSUE_BODY + "\nEdited.\n")

    assert result["ok"] is False
    assert "contract digest does not match the supplied issue body text" in result["errors"]


def test_changed_paths_mismatch_fails_verification() -> None:
    """An undeclared file in the live diff fails verification."""
    receipt = self_review.build_receipt(_declaration())
    runner = _git_runner()
    drifted = deepcopy(receipt)
    drifted["diff"] = dict(receipt["diff"])
    drifted["diff"]["changed_paths"] = ["scripts/dev/other.py"]

    result = self_review.verify_receipt_against_git(
        drifted, worktree="/tmp/issue-9537-exact-diff-self-review", git_runner=runner
    )

    assert result["ok"] is False
    assert any("changed_paths" in error for error in result["errors"])


# Honest validation.


def test_passing_claim_with_nonzero_exit_is_rejected() -> None:
    """Missing validation cannot be represented as passed."""
    declaration = _declaration(
        validation=[
            {"command": "pytest tests/dev/test_example.py -q", "exit_code": 2, "result": "passed"}
        ]
    )

    with pytest.raises(ValueError, match="non-zero exit"):
        self_review.build_receipt(declaration)


def test_empty_validation_is_rejected() -> None:
    """A receipt with no validation records carries no proof."""
    declaration = _declaration(validation=[])

    with pytest.raises(ValueError, match="non-empty list"):
        self_review.build_receipt(declaration)


def test_failed_validation_without_pass_blocks_handoff() -> None:
    """Only failed validation records cannot authorize handoff."""
    declaration = _declaration(
        validation=[
            {"command": "pytest tests/dev/test_example.py -q", "exit_code": 1, "result": "failed"}
        ]
    )

    with pytest.raises(ValueError, match="no validation record claims passed"):
        self_review.build_receipt(declaration)


# Findings discipline.


def test_blocking_finding_prevents_handoff() -> None:
    """An unresolved blocking finding returns the work to implementation."""
    declaration = _declaration(
        findings={"blocking": ["fail-open path in retry helper"], "non_blocking": []}
    )

    with pytest.raises(ValueError, match="blocking findings are unresolved"):
        self_review.build_receipt(declaration)


def test_failed_check_verdict_prevents_handoff() -> None:
    """A fail verdict on any check refuses handoff even without findings."""
    declaration = _declaration(checks=_checks(failed=("fail_closed",)))

    with pytest.raises(ValueError, match="fail verdict"):
        self_review.build_receipt(declaration)


def test_non_blocking_follow_up_permits_handoff() -> None:
    """Deferred follow-ups are recorded without blocking delivery."""
    receipt = self_review.build_receipt(
        _declaration(findings={"blocking": [], "non_blocking": ["follow-up #9600: polish docs"]})
    )
    decision = self_review.handoff_decision(
        receipt,
        expected_issue=9537,
        expected_base_sha=BASE_SHA,
        expected_head_sha=HEAD_SHA,
    )

    assert decision["ok"] is True


def test_missing_check_id_is_rejected() -> None:
    """A receipt skipping a required question cannot validate."""
    declaration = _declaration(checks=_checks()[:-1])

    with pytest.raises(ValueError, match="missing required ids"):
        self_review.build_receipt(declaration)


# Authority separation.


def test_self_review_grants_no_independent_review_authority() -> None:
    """Self-review structure carries no review verdict or approval field."""
    receipt = self_review.build_receipt(_declaration())

    assert "independent_verifier" not in receipt
    assert "approved" not in json.dumps(receipt)
    assert "review_verdict" not in receipt
    from scripts.dev import issue_completion_receipt as completion

    assert receipt["schema"] != completion.SCHEMA


def test_missing_receipt_refuses_handoff() -> None:
    """Autonomous handoff without a receipt is an explicit refusal, not a skip."""
    decision = self_review.handoff_decision(
        None,
        expected_issue=9537,
        expected_base_sha=BASE_SHA,
        expected_head_sha=HEAD_SHA,
    )

    assert decision == {
        "ok": False,
        "reasons": ["missing_receipt: no implementation self-review was produced"],
    }


def test_malformed_receipt_refuses_handoff() -> None:
    """A malformed receipt refuses with structured reasons."""
    decision = self_review.handoff_decision(
        {"schema": "wrong", "issue": "NaN"},
        expected_issue=9537,
        expected_base_sha=BASE_SHA,
        expected_head_sha=HEAD_SHA,
    )

    assert decision["ok"] is False
    assert len(decision["reasons"]) > 1


# Worktree verification success path.


def test_verify_receipt_against_matching_git_state() -> None:
    """Matching live Git state verifies the exact-head binding."""
    receipt = self_review.build_receipt(_declaration())
    runner = _git_runner()

    result = self_review.verify_receipt_against_git(
        receipt, worktree="/tmp/issue-9537-exact-diff-self-review", git_runner=runner
    )

    assert result == {
        "schema": self_review.SCHEMA,
        "ok": True,
        "errors": [],
        "head_sha": HEAD_SHA,
    }


def test_verify_fails_closed_on_git_failure() -> None:
    """Unreadable Git state never verifies."""
    receipt = self_review.build_receipt(_declaration())

    def _failing(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 128, "", "not a git repository")

    result = self_review.verify_receipt_against_git(
        receipt, worktree="/tmp/issue-9537-exact-diff-self-review", git_runner=_failing
    )

    assert result["ok"] is False
    assert any("git evidence failed" in error for error in result["errors"])


def test_collect_git_evidence_rejects_empty_diff() -> None:
    """A worktree with no diff cannot produce receipt evidence."""
    runner = _git_runner(name_only="", diff="")

    with pytest.raises(RuntimeError, match="no diff"):
        self_review.collect_git_evidence(
            worktree="/tmp/issue-9537-exact-diff-self-review", git_runner=runner
        )
