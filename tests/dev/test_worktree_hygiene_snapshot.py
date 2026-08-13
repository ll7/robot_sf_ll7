"""Tests for worktree hygiene snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.dev import worktree_hygiene_snapshot as snapshot

if TYPE_CHECKING:
    from pathlib import Path


def _result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return snapshot.subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _hygiene_row(
    *,
    path: str = "/repo/issue-7024",
    branch: str = "issue-7024-fix",
    current: bool = False,
    detached: bool = False,
    dirty_entries: int = 0,
    upstream: str | None = "origin/issue-7024-fix",
    ahead: int | None = 0,
    behind: int | None = 0,
) -> snapshot.WorktreeHygiene:
    return snapshot.WorktreeHygiene(
        path=path,
        branch=branch,
        head_sha="abc1234",
        is_current=current,
        is_detached=detached,
        dirty_entries=dirty_entries,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        issues=[],
    )


def _merged_evidence(**overrides: object) -> snapshot.RetirementEvidence:
    values: dict[str, object] = {"coverage_override": ("merged_pr", [])}
    values.update(overrides)
    return snapshot.RetirementEvidence(**values)


def test_parse_worktree_porcelain_handles_branch_and_detached() -> None:
    """Parse branch and detached rows from porcelain worktree output."""
    rows = snapshot._parse_worktree_porcelain(
        "\n".join(
            [
                "worktree /repo/main",
                "HEAD aaa",
                "branch refs/heads/main",
                "",
                "worktree /repo/detached",
                "HEAD bbb",
                "detached",
                "",
            ]
        )
    )

    assert rows == [
        {"path": "/repo/main", "head_sha": "aaa", "branch": "main"},
        {"path": "/repo/detached", "head_sha": "bbb", "detached": "true"},
    ]


def test_run_command_preserves_timeout_result(monkeypatch) -> None:
    """Timeouts remain bounded and reported as command failures."""

    def raise_timeout(*args, **kwargs):
        del args, kwargs
        raise snapshot.subprocess.TimeoutExpired(["git"], 30)

    monkeypatch.setattr(snapshot.subprocess, "run", raise_timeout)

    result = snapshot._run_command(["git"], timeout=30)

    assert result.returncode == 124
    assert result.stderr == "command timed out after 30 seconds"


def test_classify_issues_reports_dirty_missing_upstream_and_drift() -> None:
    """Report all hygiene issue classes represented by one row."""
    assert snapshot._classify_issues(
        branch="feature",
        is_detached=False,
        dirty_entries=2,
        upstream=None,
        ahead=1,
        behind=3,
    ) == ["dirty", "missing_upstream", "ahead", "behind"]


def test_build_snapshot_filters_and_counts(monkeypatch, tmp_path: Path) -> None:
    """Filter worktrees and aggregate issue counts in the snapshot."""
    main = tmp_path / "main"
    feature = tmp_path / "feature"
    main.mkdir()
    feature.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        f"worktree {main}",
                        "HEAD aaa",
                        "branch refs/heads/main",
                        "",
                        f"worktree {feature}",
                        "HEAD bbb",
                        "branch refs/heads/feature",
                        "",
                    ]
                )
            )
        if args == ["git", "status", "--porcelain"]:
            return _result(" M changed.py\n" if cwd == str(feature) else "")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            if cwd == str(feature):
                return _result("origin/feature\n")
            return _result("origin/main\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/feature"]:
            return _result("1\t2\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(filters=["feature"], worktree_limit=10)

    assert result.total_worktrees == 2
    assert result.included_worktrees == 1
    assert result.worktrees_truncated is False
    assert result.issue_counts == {"ahead": 1, "behind": 1, "dirty": 1}
    assert result.worktrees[0].branch == "feature"
    assert result.worktrees[0].dirty_entries == 1
    assert result.worktrees[0].ahead == 1
    assert result.worktrees[0].behind == 2


def test_repo_status_is_optional(monkeypatch, tmp_path: Path) -> None:
    """Include current checkout status only when requested."""
    main = tmp_path / "main"
    main.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {main}\nHEAD aaa\nbranch refs/heads/main\n")
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "status", "--short", "--branch"]:
            return _result("## main...origin/main\n M docs.md\n")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/main\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"]:
            return _result("0\t4\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(include_repo_status=True)

    assert result.repo_status is not None
    assert result.repo_status.branch_status == "## main...origin/main"
    assert result.repo_status.dirty_entries == 1
    assert result.repo_status.behind == 4


def test_missing_worktree_path_marks_status_failed(monkeypatch, tmp_path: Path) -> None:
    """Classify missing worktree paths as status failures."""
    main = tmp_path / "main"
    missing = tmp_path / "missing"
    main.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {missing}\nHEAD aaa\nbranch refs/heads/gone\n")
        if args == ["git", "status", "--porcelain"]:
            return _result(stderr="missing", returncode=127)
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result(stderr="missing", returncode=127)
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot()

    assert result.included_worktrees == 1
    assert result.issue_counts == {"missing_upstream": 1, "status_failed": 1}
    assert result.worktrees[0].dirty_entries == -1


def test_current_worktree_is_reported_when_truncated(monkeypatch, tmp_path: Path) -> None:
    """Preserve current worktree identity even when rows are truncated."""
    first = tmp_path / "first"
    current = tmp_path / "current"
    first.mkdir()
    current.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        f"worktree {first}",
                        "HEAD aaa",
                        "branch refs/heads/first",
                        "",
                        f"worktree {current}",
                        "HEAD bbb",
                        "branch refs/heads/current",
                    ]
                )
            )
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/first\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/first"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(current)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(worktree_limit=1)

    assert result.current_worktree == str(current)
    assert result.included_worktrees == 1
    assert result.worktrees_truncated is True


def test_current_worktree_ignores_malformed_rows(monkeypatch, tmp_path: Path) -> None:
    """Do not let rows without paths match the current checkout."""
    current = tmp_path / "current"
    current.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        "HEAD malformed",
                        "",
                        f"worktree {current}",
                        "HEAD aaa",
                        "branch refs/heads/current",
                    ]
                )
            )
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/current\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/current"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(current)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot()

    assert result.current_worktree == str(current)


def test_retirement_clean_merged_worktree_is_removeable() -> None:
    """A clean, covered branch without preservation evidence may be reported safe."""
    assessment = snapshot.assess_retirement(
        _hygiene_row(),
        _merged_evidence(),
    )

    assert assessment.decision == "removeable"
    assert assessment.coverage == "merged_pr"
    assert assessment.reasons == []


def test_retirement_merged_pr_must_cover_exact_worktree_head(monkeypatch) -> None:
    """A reused branch cannot provide merged coverage for a different commit."""
    monkeypatch.setattr(
        snapshot,
        "_run_command",
        lambda *args, **kwargs: _result(returncode=1),
    )
    assessment = snapshot.assess_retirement(
        _hygiene_row(),
        snapshot.RetirementEvidence(
            pull_requests=[
                {
                    "number": 7024,
                    "state": "MERGED",
                    "headRefName": "issue-7024-fix",
                    "headRefOid": "different-head",
                    "title": "Fix #7024",
                    "body": "",
                }
            ]
        ),
    )

    assert assessment.decision == "review"
    assert assessment.coverage == "unmerged"
    assert "branch_head_mismatch" in assessment.reasons
    assert "no_merged_coverage" in assessment.reasons


def test_retirement_merged_pr_with_exact_worktree_head_is_removeable() -> None:
    """A merged PR is authoritative only when its head matches the worktree head."""
    assessment = snapshot.assess_retirement(
        _hygiene_row(),
        snapshot.RetirementEvidence(
            pull_requests=[
                {
                    "number": 7024,
                    "state": "MERGED",
                    "headRefName": "issue-7024-fix",
                    "headRefOid": "abc1234",
                    "title": "Fix #7024",
                    "body": "",
                }
            ]
        ),
    )

    assert assessment.decision == "removeable"
    assert assessment.coverage == "merged_pr"


@pytest.mark.parametrize(
    ("row_kwargs", "decision", "reason"),
    [
        ({"dirty_entries": 1}, "preserve", "dirty"),
        ({"ahead": 1}, "preserve", "ahead_commits"),
        ({"branch": "main"}, "preserve", "protected_canonical_branch"),
        (
            {"detached": True, "branch": "", "upstream": None, "ahead": None},
            "preserve",
            "detached",
        ),
        ({"upstream": None, "ahead": None}, "preserve", "missing_upstream"),
        ({"dirty_entries": -1}, "review", "status_unavailable"),
    ],
)
def test_retirement_preserves_or_reviews_unavailable_local_state(
    row_kwargs: dict[str, object], decision: str, reason: str
) -> None:
    """Dirty, unpushed, detached, missing-upstream, and unknown status never pass."""
    assessment = snapshot.assess_retirement(
        _hygiene_row(**row_kwargs),
        _merged_evidence(),
    )

    assert assessment.decision == decision
    assert reason in assessment.reasons


def test_retirement_active_claim_is_preserved() -> None:
    """A matching remote issue claim is a hard preservation barrier."""
    assessment = snapshot.assess_retirement(
        _hygiene_row(),
        _merged_evidence(active_claims={7024: "abc1234"}),
    )

    assert assessment.decision == "preserve"
    assert assessment.active_claims == [7024]
    assert "active_issue_claim" in assessment.reasons


@pytest.mark.parametrize(
    ("artifact", "decision", "category"),
    [
        (
            snapshot.IgnoredArtifact(".pytest_cache", "cache", "known local cache root"),
            "removeable",
            "cache",
        ),
        (
            snapshot.IgnoredArtifact(
                "output/coverage", "disposable_output", "documented validation output"
            ),
            "removeable",
            "disposable_output",
        ),
        (
            snapshot.IgnoredArtifact("output/checkpoints", "durable_required", "durable evidence"),
            "preserve",
            "durable_required",
        ),
        (
            snapshot.IgnoredArtifact("output/unknown", "handoff_needed", "human classification"),
            "preserve",
            "handoff_needed",
        ),
    ],
)
def test_retirement_classifies_ignored_artifact_roots(
    artifact: snapshot.IgnoredArtifact, decision: str, category: str
) -> None:
    """Only explicitly allowlisted cache/output roots remain eligible."""
    assessment = snapshot.assess_retirement(
        _hygiene_row(),
        _merged_evidence(ignored_artifacts=[artifact]),
    )

    assert assessment.decision == decision
    assert assessment.ignored_artifacts[0].category == category


def test_retirement_tracked_durable_paths_are_preserved() -> None:
    """Tracked evidence/manifests are never emitted as safely removeable."""
    assessment = snapshot.assess_retirement(
        _hygiene_row(),
        _merged_evidence(tracked_durable_paths=["docs/context/evidence/result.json"]),
    )

    assert assessment.decision == "preserve"
    assert "tracked_durable_evidence_or_manifest" in assessment.reasons


def test_retirement_unknown_remote_state_is_review_blocker() -> None:
    """Unavailable PR/claim state cannot produce a safe-removal recommendation."""
    assessment = snapshot.assess_retirement(
        _hygiene_row(),
        snapshot.RetirementEvidence(
            pull_request_error="network unavailable",
            claims_error="claim ref unavailable",
        ),
    )

    assert assessment.decision == "review"
    assert "pull_request_state_unavailable" in assessment.reasons
    assert "active_claim_state_unavailable" in assessment.reasons


def test_build_retirement_plan_reports_remote_errors_without_removal(
    monkeypatch, tmp_path: Path
) -> None:
    """The report remains read-only and fail-closed when remote evidence is unavailable."""
    row = _hygiene_row(path=str(tmp_path / "worktree"))
    hygiene = snapshot.HygieneSnapshot(
        schema=snapshot.SCHEMA_VERSION,
        current_worktree=str(tmp_path / "main"),
        total_worktrees=1,
        included_worktrees=1,
        worktrees_truncated=False,
        filters=[],
        issue_counts={},
        repo_status=None,
        worktrees=[row],
        errors=[],
    )
    monkeypatch.setattr(snapshot, "_ignored_artifacts", lambda _path: ([], None))
    monkeypatch.setattr(snapshot, "_tracked_durable_paths", lambda _path: ([], None))

    plan = snapshot.build_retirement_plan(
        snapshot=hygiene,
        pull_requests=[],
        pull_request_error="PR API unavailable",
        active_claims={},
        claims_error="claim refs unavailable",
    )

    assert plan.removeable == []
    assert plan.review == [row.path]
    assert len(plan.errors) == 2


def test_build_retirement_plan_falls_back_to_branch_pr_query(monkeypatch, tmp_path: Path) -> None:
    """A truncated global PR inventory is replaced by exact branch evidence."""
    row = _hygiene_row(path=str(tmp_path / "worktree"))
    hygiene = snapshot.HygieneSnapshot(
        schema=snapshot.SCHEMA_VERSION,
        current_worktree=str(tmp_path / "main"),
        total_worktrees=1,
        included_worktrees=1,
        worktrees_truncated=False,
        filters=[],
        issue_counts={},
        repo_status=None,
        worktrees=[row],
        errors=[],
    )
    pr = {
        "number": 7024,
        "state": "MERGED",
        "headRefName": row.branch,
        "headRefOid": row.head_sha,
        "title": "Fix #7024",
        "body": "",
    }
    monkeypatch.setattr(
        snapshot,
        "_load_pull_request_rows",
        lambda _repo: ([], snapshot.PULL_REQUEST_INVENTORY_TRUNCATED),
    )
    monkeypatch.setattr(snapshot, "_query_head_pull_requests", lambda _repo, _branch: ([pr], None))
    monkeypatch.setattr(snapshot, "_load_active_claims", lambda _repo: ({}, None))
    monkeypatch.setattr(snapshot, "_ignored_artifacts", lambda _path: ([], None))
    monkeypatch.setattr(snapshot, "_tracked_durable_paths", lambda _path: ([], None))

    plan = snapshot.build_retirement_plan(snapshot=hygiene)

    assert plan.errors == []
    assert plan.removeable == [row.path]
