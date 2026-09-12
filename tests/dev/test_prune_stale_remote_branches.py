"""Tests for the stale remote branch sweep (issue #9087)."""

from __future__ import annotations

import pytest

from scripts.dev.prune_stale_remote_branches import (
    DELETE_CLAIM_CLOSED,
    DELETE_MERGED,
    KEEP_OPEN_CLAIM,
    KEEP_OPEN_PR,
    KEEP_PROBE_ERROR,
    KEEP_PROTECTED,
    KEEP_UNMERGED,
    apply_deletions,
    build_report,
    classify_heads,
    main,
    run_scan,
)


class FakeProbe:
    """In-memory probe stand-in with configurable failures."""

    def __init__(
        self,
        *,
        heads: dict[str, str],
        ancestors: dict[str, bool | None],
        open_pr_heads: set[str] | None,
        issue_states: dict[int, str | None],
        delete_ok: bool = True,
    ) -> None:
        """Store the head, ancestry, PR, and issue-state fixtures."""
        self.heads = dict(heads)
        self.ancestors = ancestors
        self.open_pr_heads_value = open_pr_heads
        self.issue_states = issue_states
        self.delete_ok = delete_ok
        self.deleted: list[str] = []

    def list_heads(self) -> dict[str, str]:
        return dict(self.heads)

    def is_ancestor(self, sha: str) -> bool | None:
        return self.ancestors.get(sha)

    def open_pr_heads(self) -> set[str] | None:
        return self.open_pr_heads_value

    def issue_state(self, number: int) -> str | None:
        return self.issue_states.get(number)

    def delete_head(self, ref: str) -> tuple[bool, str]:
        self.deleted.append(ref)
        if self.delete_ok:
            self.heads.pop(ref, None)
            return True, ""
        return False, "simulated deletion failure"


def _probe() -> FakeProbe:
    return FakeProbe(
        heads={
            "refs/heads/main": "sha-main",
            "refs/heads/release/0.0.5": "sha-release",
            "refs/heads/agent-claims/issue-1": "sha-claim-closed",
            "refs/heads/agent-claims/issue-2": "sha-claim-open",
            "refs/heads/agent-claims/issue-3": "sha-claim-unknown",
            "refs/heads/fix/merged": "sha-merged",
            "refs/heads/fix/merged-with-pr": "sha-merged-pr",
            "refs/heads/fix/unmerged": "sha-unmerged",
            "refs/heads/fix/unknown": "sha-unknown",
        },
        ancestors={
            "sha-main": True,
            "sha-release": True,
            "sha-merged": True,
            "sha-merged-pr": True,
            "sha-unmerged": False,
            "sha-unknown": None,
        },
        open_pr_heads={"fix/merged-with-pr"},
        issue_states={1: "closed", 2: "open", 3: None},
    )


def test_classify_heads_applies_every_reason_code() -> None:
    """Each head receives the documented reason code and action."""
    probe = _probe()
    rows = classify_heads(
        probe.list_heads(),
        is_ancestor=probe.is_ancestor,
        open_pr_heads=probe.open_pr_heads() or set(),
        issue_state=probe.issue_state,
    )
    reasons = {row.ref: row.reason for row in rows}
    actions = {row.ref: row.action for row in rows}

    assert reasons["refs/heads/main"] == KEEP_PROTECTED
    assert reasons["refs/heads/release/0.0.5"] == KEEP_PROTECTED
    assert reasons["refs/heads/agent-claims/issue-1"] == DELETE_CLAIM_CLOSED
    assert reasons["refs/heads/agent-claims/issue-2"] == KEEP_OPEN_CLAIM
    assert reasons["refs/heads/agent-claims/issue-3"] == KEEP_PROBE_ERROR
    assert reasons["refs/heads/fix/merged-with-pr"] == KEEP_OPEN_PR
    assert reasons["refs/heads/fix/merged"] == DELETE_MERGED
    assert reasons["refs/heads/fix/unmerged"] == KEEP_UNMERGED
    assert reasons["refs/heads/fix/unknown"] == KEEP_PROBE_ERROR
    assert actions["refs/heads/main"] == "keep"
    assert actions["refs/heads/fix/merged"] == "delete"
    assert actions["refs/heads/agent-claims/issue-1"] == "delete"


def test_build_report_is_deterministic_and_counts_reasons() -> None:
    """Candidates are sorted by ref and reason counts are stable."""
    probe = _probe()
    rows = classify_heads(
        probe.list_heads(),
        is_ancestor=probe.is_ancestor,
        open_pr_heads=probe.open_pr_heads() or set(),
        issue_state=probe.issue_state,
    )
    report = build_report(rows, repo="ll7/robot_sf_ll7", main_ref="origin/main")

    assert report["schema"] == "stale_remote_branch_report.v1"
    assert report["candidate_count"] == 2
    candidate_refs = [candidate["ref"] for candidate in report["candidates"]]
    assert candidate_refs == sorted(candidate_refs)
    assert set(candidate_refs) == {
        "refs/heads/agent-claims/issue-1",
        "refs/heads/fix/merged",
    }
    assert report["reason_counts"][KEEP_PROBE_ERROR] == 2
    assert report["head_count"] == 9


def test_apply_deletions_respects_limit_and_is_idempotent() -> None:
    """Apply deletes only candidates, honors the limit, and converges."""
    probe = _probe()
    first = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")
    applied = apply_deletions(probe, first, limit=1)

    assert applied["deleted_count"] == 1
    assert len(probe.deleted) == 1

    second = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")
    assert second["candidate_count"] == 1

    final = apply_deletions(probe, second, limit=10)
    assert final["deleted_count"] == 1
    third = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")
    assert third["candidate_count"] == 0


def test_apply_deletions_records_failures() -> None:
    """A failed deletion is reported and returns a failure count."""
    probe = _probe()
    probe.delete_ok = False
    report = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")

    applied = apply_deletions(probe, report, limit=10)

    assert applied["deleted_count"] == 0
    assert applied["failed_count"] == 2
    assert all(item["error"] for item in applied["deletions"])


def test_run_scan_fails_closed_without_open_pr_state() -> None:
    """An unreadable open-PR state refuses to classify (exit 2)."""
    probe = _probe()
    probe.open_pr_heads_value = None

    with pytest.raises(SystemExit) as excinfo:
        run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")

    assert excinfo.value.code == 2


def test_main_dry_run_and_apply_with_injected_probe(tmp_path) -> None:
    """main() wires scan, apply, and report output through the probe."""
    probe = _probe()
    report_path = tmp_path / "report.json"

    assert main(["--report", str(report_path)], probe=probe) == 0
    assert probe.deleted == []
    assert report_path.is_file()

    assert main(["--apply", "--limit", "5", "--report", str(report_path)], probe=probe) == 0
    assert probe.deleted == ["refs/heads/agent-claims/issue-1", "refs/heads/fix/merged"]


def test_main_returns_one_when_a_deletion_fails() -> None:
    """Deletion failures surface as a nonzero exit code."""
    probe = _probe()
    probe.delete_ok = False

    assert main(["--apply", "--limit", "5"], probe=probe) == 1
