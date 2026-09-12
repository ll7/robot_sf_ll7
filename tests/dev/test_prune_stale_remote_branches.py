"""Tests for the stale remote branch sweep (issue #9087)."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts.dev.prune_stale_remote_branches import (
    DELETE_CLAIM_CLOSED,
    DELETE_MERGED,
    KEEP_OPEN_CLAIM,
    KEEP_OPEN_PR,
    KEEP_PROBE_ERROR,
    KEEP_PROTECTED,
    KEEP_UNMERGED,
    GitGhProbe,
    apply_deletions,
    build_report,
    classify_heads,
    main,
    run_scan,
)

if TYPE_CHECKING:
    from pathlib import Path


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

    def delete_head(self, ref: str, expected_sha: str = "") -> tuple[bool, str]:
        current_sha = self.heads.get(ref)
        if current_sha is None:
            return False, f"ref not found: {ref}"
        if expected_sha and current_sha != expected_sha:
            return False, f"stale ref {ref}: expected {expected_sha}, found {current_sha}"
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


def test_apply_deletions_rejects_stale_tip_when_branch_advances() -> None:
    """Issue #9168: advancing A to B after scan leaves B untouched and reports stale tip."""
    probe = _probe()
    scan = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")
    candidate = next(c for c in scan["candidates"] if c["ref"] == "refs/heads/fix/merged")
    assert candidate["sha"] == "sha-merged"

    # Simulate another worker advancing the remote branch to sha-new-unmerged
    probe.heads["refs/heads/fix/merged"] = "sha-new-unmerged"

    applied = apply_deletions(probe, scan, limit=10)
    # The advanced ref must NOT be deleted
    assert "refs/heads/fix/merged" not in probe.deleted
    assert probe.heads["refs/heads/fix/merged"] == "sha-new-unmerged"

    # Deletion entry must record failure with stale tip error
    del_entry = next(d for d in applied["deletions"] if d["ref"] == "refs/heads/fix/merged")
    assert not del_entry["ok"]
    assert "stale" in del_entry["error"]


def test_apply_deletions_blocks_when_pr_opened_after_scan() -> None:
    """Issue #9168: a newly opened PR prevents deletion even when the tip is unchanged."""
    probe = _probe()
    scan = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")
    assert any(c["ref"] == "refs/heads/fix/merged" for c in scan["candidates"])

    # Simulate a PR opened for fix/merged before apply
    probe.open_pr_heads_value = {"fix/merged-with-pr", "fix/merged"}

    applied = apply_deletions(probe, scan, limit=10)
    assert "refs/heads/fix/merged" not in probe.deleted
    assert "refs/heads/fix/merged" in probe.heads

    del_entry = next(d for d in applied["deletions"] if d["ref"] == "refs/heads/fix/merged")
    assert not del_entry["ok"]
    assert "open PR detected" in del_entry["error"]


def test_apply_deletions_blocks_when_open_pr_refresh_unavailable() -> None:
    """Issue #9168: keep refs when open PR refresh fails."""
    probe = _probe()
    scan = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")

    probe.open_pr_heads_value = None

    applied = apply_deletions(probe, scan, limit=10)
    assert "refs/heads/fix/merged" not in probe.deleted

    del_entry = next(d for d in applied["deletions"] if d["ref"] == "refs/heads/fix/merged")
    assert not del_entry["ok"]
    assert "open PR refresh unavailable" in del_entry["error"]


def test_apply_deletions_blocks_when_claim_issue_reopened() -> None:
    """Issue #9168: a reopened claim issue prevents deletion even when tip is unchanged."""
    probe = _probe()
    scan = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")
    assert any(c["ref"] == "refs/heads/agent-claims/issue-1" for c in scan["candidates"])

    # Reopen issue 1
    probe.issue_states[1] = "open"

    applied = apply_deletions(probe, scan, limit=10)
    assert "refs/heads/agent-claims/issue-1" not in probe.deleted
    assert "refs/heads/agent-claims/issue-1" in probe.heads

    del_entry = next(
        d for d in applied["deletions"] if d["ref"] == "refs/heads/agent-claims/issue-1"
    )
    assert not del_entry["ok"]
    assert "claim issue 1 is open" in del_entry["error"]


def test_apply_deletions_blocks_when_claim_issue_refresh_unavailable() -> None:
    """Issue #9168: keep claim refs when issue state refresh fails."""
    probe = _probe()
    scan = run_scan(probe, repo="ll7/robot_sf_ll7", main_ref="origin/main")

    probe.issue_states[1] = None

    applied = apply_deletions(probe, scan, limit=10)
    assert "refs/heads/agent-claims/issue-1" not in probe.deleted

    del_entry = next(
        d for d in applied["deletions"] if d["ref"] == "refs/heads/agent-claims/issue-1"
    )
    assert not del_entry["ok"]
    assert "could not refresh state" in del_entry["error"]


def test_git_gh_probe_atomic_lease_with_local_bare_repo(tmp_path: Path) -> None:
    """Issue #9168: GitGhProbe.delete_head enforces atomic lease on a real git remote."""
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(work), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(work), "config", "user.name", "test"], check=True)
    (work / "file.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "-C", str(work), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(work), "commit", "-m", "initial"], check=True, capture_output=True
    )
    sha_a = subprocess.check_output(
        ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(["git", "-C", str(work), "remote", "add", "origin", str(bare)], check=True)
    subprocess.run(
        ["git", "-C", str(work), "push", "origin", "master:refs/heads/topic"],
        check=True,
        capture_output=True,
    )

    probe = GitGhProbe(repo="ll7/robot_sf_ll7", remote=str(bare), main_ref="master")

    # Mismatched expected_sha fails and leaves ref intact
    ok, error = probe.delete_head(
        "refs/heads/topic", expected_sha="0000000000000000000000000000000000000000"
    )
    assert not ok
    assert "stale info" in error or "rejected" in error
    heads = probe.list_heads()
    assert "refs/heads/topic" in heads
    assert heads["refs/heads/topic"] == sha_a

    # Matching expected_sha succeeds and deletes ref
    ok, error = probe.delete_head("refs/heads/topic", expected_sha=sha_a)
    assert ok
    assert error == ""
    heads_after = probe.list_heads()
    assert "refs/heads/topic" not in heads_after
