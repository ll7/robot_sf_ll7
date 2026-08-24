"""Tests for worktree hygiene snapshots."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dev import worktree_hygiene_snapshot as snapshot


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


def _lookup(status: str, reason: str | None = None):
    return snapshot.LookupState(status=status, reason=reason)


def _artifacts(classification: str = "none", status: str = "ok"):
    return [
        snapshot.ArtifactRootInspection(
            root="output",
            classification=classification,
            status=status,
        )
    ]


def _worktree_row(path: str, branch: str = "issue-1") -> snapshot.WorktreeHygiene:
    return snapshot.WorktreeHygiene(
        path=path,
        branch=branch,
        head_sha="aaa",
        is_current=False,
        is_detached=False,
        dirty_entries=0,
        upstream=f"origin/{branch}",
        ahead=0,
        behind=0,
    )


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


def test_run_command_caps_subprocess_timeout_at_deadline(monkeypatch, fake_subprocess) -> None:
    """A retirement deadline bounds one slow subprocess, not only the row loop."""
    fake_subprocess.set_default(_result())

    monkeypatch.setattr(snapshot.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(snapshot.subprocess, "run", fake_subprocess)

    result = snapshot._run_command(["git", "status"], timeout=30, deadline=101.5)

    assert result.returncode == 0
    assert fake_subprocess.last_kwargs is not None
    assert fake_subprocess.last_kwargs["timeout"] == pytest.approx(1.5)


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


def test_status_paths_preserves_modified_output_path() -> None:
    """Parse short-status rows without stripping the leading status-space."""
    assert snapshot._status_paths(
        " M output/fixtures/baseline.json\n"
        "?? output/benchmarks/new.jsonl\n"
        "!! output/model_cache/cache.bin\n"
    ) == (
        ["output/fixtures/baseline.json"],
        ["output/benchmarks/new.jsonl"],
        ["output/model_cache/cache.bin"],
    )


def test_build_snapshot_filters_and_counts(monkeypatch, tmp_path: Path, fake_subprocess) -> None:
    """Filter worktrees and aggregate issue counts in the snapshot."""
    main = tmp_path / "main"
    feature = tmp_path / "feature"
    main.mkdir()
    feature.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(
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
        ),
    )
    fake_subprocess.register(
        ["git", "status", "--porcelain"],
        lambda cmd, cwd=None, **kw: _result(" M changed.py\n" if cwd == str(feature) else ""),
    )
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        lambda cmd, cwd=None, **kw: _result(
            "origin/feature\n" if cwd == str(feature) else "origin/main\n"
        ),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/feature"],
        _result("1\t2\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"],
        _result("0\t0\n"),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(filters=["feature"], worktree_limit=10)

    assert result.total_worktrees == 2
    assert result.included_worktrees == 1
    assert result.worktrees_truncated is False
    assert result.issue_counts == {"ahead": 1, "behind": 1, "dirty": 1}
    assert result.worktrees[0].branch == "feature"
    assert result.worktrees[0].dirty_entries == 1
    assert result.worktrees[0].ahead == 1
    assert result.worktrees[0].behind == 2


def test_repo_status_is_optional(monkeypatch, tmp_path: Path, fake_subprocess) -> None:
    """Include current checkout status only when requested."""
    main = tmp_path / "main"
    main.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {main}\nHEAD aaa\nbranch refs/heads/main\n"),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))
    fake_subprocess.register(
        ["git", "status", "--short", "--branch"],
        _result("## main...origin/main\n M docs.md\n"),
    )
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result("origin/main\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"],
        _result("0\t4\n"),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(include_repo_status=True)

    assert result.repo_status is not None
    assert result.repo_status.branch_status == "## main...origin/main"
    assert result.repo_status.dirty_entries == 1
    assert result.repo_status.behind == 4


def test_missing_worktree_path_marks_status_failed(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Classify missing worktree paths as status failures."""
    main = tmp_path / "main"
    missing = tmp_path / "missing"
    main.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {missing}\nHEAD aaa\nbranch refs/heads/gone\n"),
    )
    fake_subprocess.register(
        ["git", "status", "--porcelain"],
        _result(stderr="missing", returncode=127),
    )
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result(stderr="missing", returncode=127),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot()

    assert result.included_worktrees == 1
    assert result.issue_counts == {"missing_upstream": 1, "status_failed": 1}
    assert result.worktrees[0].dirty_entries == -1


def test_current_worktree_is_reported_when_truncated(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Preserve current worktree identity even when rows are truncated."""
    first = tmp_path / "first"
    current = tmp_path / "current"
    first.mkdir()
    current.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(
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
        ),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result("origin/first\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/first"],
        _result("0\t0\n"),
    )

    monkeypatch.chdir(current)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(worktree_limit=1)

    assert result.current_worktree == str(current)
    assert result.included_worktrees == 1
    assert result.worktrees_truncated is True


def test_current_worktree_ignores_malformed_rows(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Do not let rows without paths match the current checkout."""
    current = tmp_path / "current"
    current.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(
            "\n".join(
                [
                    "HEAD malformed",
                    "",
                    f"worktree {current}",
                    "HEAD aaa",
                    "branch refs/heads/current",
                ]
            )
        ),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result("origin/current\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/current"],
        _result("0\t0\n"),
    )

    monkeypatch.chdir(current)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

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
    monkeypatch.setattr(snapshot, "_ignored_artifacts", lambda _path, **_kwargs: ([], None))
    monkeypatch.setattr(snapshot, "_tracked_durable_paths", lambda _path, **_kwargs: ([], None))

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


def test_build_retirement_plan_budget_marks_unprocessed_rows_review(
    monkeypatch, tmp_path: Path
) -> None:
    """A worktree budget preserves every row that was not fully inspected."""
    row_a = _hygiene_row(path=str(tmp_path / "worktree-a"), branch="feature-a")
    row_b = _hygiene_row(path=str(tmp_path / "worktree-b"), branch="feature-b")
    hygiene = snapshot.HygieneSnapshot(
        schema=snapshot.SCHEMA_VERSION,
        current_worktree=str(tmp_path / "main"),
        total_worktrees=2,
        included_worktrees=2,
        worktrees_truncated=False,
        filters=[],
        issue_counts={},
        repo_status=None,
        worktrees=[row_a, row_b],
        errors=[],
    )
    pr = {
        "number": 1,
        "state": "MERGED",
        "headRefName": row_a.branch,
        "headRefOid": row_a.head_sha,
        "title": "feature A",
        "body": "",
    }
    monkeypatch.setattr(snapshot, "_ignored_artifacts", lambda _path, **_kwargs: ([], None))
    monkeypatch.setattr(snapshot, "_tracked_durable_paths", lambda _path, **_kwargs: ([], None))

    plan = snapshot.build_retirement_plan(
        snapshot=hygiene,
        worktree_budget=1,
        time_budget_seconds=60,
        pull_requests=[pr],
        active_claims={},
    )

    assert plan.progress.terminal_status == snapshot.RETIREMENT_PLAN_INCOMPLETE
    assert plan.progress.total_worktrees == 2
    assert plan.progress.selected_worktrees == 2
    assert plan.progress.processed_worktrees == 1
    assert plan.progress.unprocessed_worktrees == 1
    assert plan.review == [row_b.path]
    assert plan.worktrees[-1].reasons == ["worktree budget exhausted"]
    assert any("retirement scan incomplete" in error for error in plan.errors)


def test_build_retirement_plan_caches_fallback_branch_lookups(monkeypatch, tmp_path: Path) -> None:
    """Fallback PR queries run once per unique branch, not once per worktree."""
    rows = [
        _hygiene_row(path=str(tmp_path / "a"), branch="feature-a"),
        _hygiene_row(path=str(tmp_path / "b"), branch="feature-b"),
        _hygiene_row(path=str(tmp_path / "c"), branch="feature-a"),
    ]
    hygiene = snapshot.HygieneSnapshot(
        schema=snapshot.SCHEMA_VERSION,
        current_worktree=str(tmp_path / "main"),
        total_worktrees=3,
        included_worktrees=3,
        worktrees_truncated=False,
        filters=[],
        issue_counts={},
        repo_status=None,
        worktrees=rows,
        errors=[],
    )
    calls: list[str] = []

    def fake_query_head(_repo: Path, branch: str, **_kwargs):
        calls.append(branch)
        return [
            {
                "number": 1,
                "state": "MERGED",
                "headRefName": branch,
                "headRefOid": "abc1234",
                "title": "feature",
                "body": "",
            }
        ], None

    monkeypatch.setattr(
        snapshot,
        "_load_pull_request_rows",
        lambda _repo, **_kwargs: ([], snapshot.PULL_REQUEST_INVENTORY_TRUNCATED),
    )
    monkeypatch.setattr(snapshot, "_query_head_pull_requests", fake_query_head)
    monkeypatch.setattr(snapshot, "_ignored_artifacts", lambda _path, **_kwargs: ([], None))
    monkeypatch.setattr(snapshot, "_tracked_durable_paths", lambda _path, **_kwargs: ([], None))

    plan = snapshot.build_retirement_plan(
        snapshot=hygiene,
        active_claims={},
        time_budget_seconds=60,
    )

    assert sorted(calls) == ["feature-a", "feature-b"]
    assert plan.progress.branch_lookup_calls == 2
    assert plan.progress.processed_worktrees == 3
    assert plan.progress.unprocessed_worktrees == 0
    assert len(plan.removeable) == 3


def test_include_all_retirement_plan_applies_budget_during_inventory(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """The all-worktree path stops before constructing rows beyond its budget."""
    worktree_paths = [tmp_path / name for name in ("a", "b", "c")]
    for path in worktree_paths:
        path.mkdir()
    rows = [(str(path), f"feature-{path.name}", f"head-{path.name}") for path in worktree_paths]
    porcelain = "\n\n".join(
        f"worktree {path}\nHEAD {head}\nbranch refs/heads/{branch}" for path, branch, head in rows
    )
    seen_cwds: list[str | None] = []

    fake_subprocess.register(["git", "worktree", "list", "--porcelain"], _result(porcelain))

    def handle_status(cmd, cwd=None, **kw):
        seen_cwds.append(cwd)
        if cwd == str(worktree_paths[2]):
            raise AssertionError("budgeted inventory inspected an unprocessed row")
        return _result("")

    def handle_upstream(cmd, cwd=None, **kw):
        seen_cwds.append(cwd)
        if cwd == str(worktree_paths[2]):
            raise AssertionError("budgeted inventory inspected an unprocessed row")
        return _result(f"origin/feature-{Path(cwd).name}\n")

    def handle_rev_list(cmd, cwd=None, **kw):
        seen_cwds.append(cwd)
        if cwd == str(worktree_paths[2]):
            raise AssertionError("budgeted inventory inspected an unprocessed row")
        return _result("0\t0\n")

    fake_subprocess.register(["git", "status", "--porcelain"], handle_status)
    fake_subprocess.register(["git", "rev-parse", "--abbrev-ref", "@{upstream}"], handle_upstream)
    fake_subprocess.register(["git", "rev-list"], handle_rev_list)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)
    monkeypatch.setattr(snapshot, "_ignored_artifacts", lambda _path, **_kwargs: ([], None))
    monkeypatch.setattr(snapshot, "_tracked_durable_paths", lambda _path, **_kwargs: ([], None))
    pull_requests = [
        {
            "number": index,
            "state": "MERGED",
            "headRefName": branch,
            "headRefOid": head,
            "title": "feature",
            "body": "",
        }
        for index, (_path, branch, head) in enumerate(rows, start=1)
    ]

    plan = snapshot.build_retirement_plan(
        include_all_worktrees=True,
        worktree_budget=2,
        time_budget_seconds=60,
        pull_requests=pull_requests,
        active_claims={},
    )

    assert plan.total_worktrees == 3
    assert plan.included_worktrees == 3
    assert plan.progress.processed_worktrees == 2
    assert plan.progress.unprocessed_worktrees == 1
    assert plan.review == [str(worktree_paths[2])]
    assert str(worktree_paths[2]) not in seen_cwds


def test_retirement_cli_accepts_scan_budgets() -> None:
    """Budget controls are explicit and machine-discoverable through the CLI parser."""
    args = snapshot._parse_args(
        [
            "--retirement-plan",
            "--include-all-worktrees",
            "--worktree-budget",
            "2",
            "--time-budget-seconds",
            "1.5",
        ]
    )

    assert args.retirement_plan is True
    assert args.include_all_worktrees is True
    assert args.worktree_budget == 2
    assert args.time_budget_seconds == 1.5


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
        lambda _repo, **_kwargs: ([], snapshot.PULL_REQUEST_INVENTORY_TRUNCATED),
    )
    monkeypatch.setattr(
        snapshot,
        "_query_head_pull_requests",
        lambda _repo, _branch, **_kwargs: ([pr], None),
    )
    monkeypatch.setattr(snapshot, "_load_active_claims", lambda _repo, **_kwargs: ({}, None))
    monkeypatch.setattr(snapshot, "_ignored_artifacts", lambda _path, **_kwargs: ([], None))
    monkeypatch.setattr(snapshot, "_tracked_durable_paths", lambda _path, **_kwargs: ([], None))

    plan = snapshot.build_retirement_plan(snapshot=hygiene)

    assert plan.errors == []
    assert plan.removeable == [row.path]


def test_retirement_plan_marks_clean_merged_worktree_removable(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """A clean, merged, unclaimed row with no durable artifacts is removable."""
    main = tmp_path / "main"
    done = tmp_path / "done"
    main.mkdir()
    done.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {done}\nHEAD aaa\nbranch refs/heads/done\n"),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result("origin/done\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/done"],
        _result("0\t0\n"),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REMOVABLE


def test_retirement_plan_preserves_dirty_worktree(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Dirty tracked or untracked status fails closed to preserve."""
    main = tmp_path / "main"
    dirty = tmp_path / "dirty"
    main.mkdir()
    dirty.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {dirty}\nHEAD aaa\nbranch refs/heads/dirty\n"),
    )
    fake_subprocess.register(
        ["git", "status", "--porcelain"], _result(" M changed.py\n?? new.txt\n")
    )
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result("origin/dirty\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/dirty"],
        _result("0\t0\n"),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_PRESERVE
    assert "tracked or untracked status is dirty" in result.worktrees[0].retirement.reasons


def test_retirement_plan_preserves_ahead_worktree(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Unpushed commits are a preservation risk even if other checks are green."""
    main = tmp_path / "main"
    ahead = tmp_path / "ahead"
    main.mkdir()
    ahead.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {ahead}\nHEAD aaa\nbranch refs/heads/ahead\n"),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result("origin/ahead\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/ahead"],
        _result("2\t0\n"),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_PRESERVE
    assert "worktree has commits ahead of upstream" in result.worktrees[0].retirement.reasons


def test_retirement_plan_reviews_detached_worktree(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Detached rows need human review because branch/upstream meaning is missing."""
    main = tmp_path / "main"
    detached = tmp_path / "detached"
    main.mkdir()
    detached.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {detached}\nHEAD aaa\ndetached\n"),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REVIEW
    assert "detached HEAD needs human review" in result.worktrees[0].retirement.reasons


def test_retirement_plan_reviews_missing_upstream(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Rows without upstreams are not safe-removal recommendations."""
    main = tmp_path / "main"
    missing = tmp_path / "missing"
    main.mkdir()
    missing.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {missing}\nHEAD aaa\nbranch refs/heads/missing\n"),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result(stderr="no upstream", returncode=128),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REVIEW
    assert "upstream is missing" in result.worktrees[0].retirement.reasons


def test_retirement_plan_reviews_unavailable_external_state(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Unavailable claim, merge, or artifact state prevents removable classification."""
    main = tmp_path / "main"
    uncertain = tmp_path / "uncertain"
    main.mkdir()
    uncertain.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(f"worktree {uncertain}\nHEAD aaa\nbranch refs/heads/uncertain\n"),
    )
    fake_subprocess.register(["git", "status", "--porcelain"], _result(""))
    fake_subprocess.register(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        _result("origin/uncertain\n"),
    )
    fake_subprocess.register(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/uncertain"],
        _result("0\t0\n"),
    )

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("unavailable"),
        merge_lookup=lambda row: _lookup("unavailable"),
        artifact_lookup=lambda row: _artifacts("unavailable", "unavailable"),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REVIEW
    assert "claim state unavailable" in result.worktrees[0].retirement.reasons
    assert "merge state unavailable" in result.worktrees[0].retirement.reasons
    assert "output artifact state unavailable" in result.worktrees[0].retirement.reasons


def test_output_root_inspection_preserves_durable_evidence(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Ignored evidence-like output paths are reported and classified as durable-required."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    row = snapshot.WorktreeHygiene(
        path=str(worktree),
        branch="issue-1",
        head_sha="aaa",
        is_current=False,
        is_detached=False,
        dirty_entries=0,
        upstream="origin/issue-1",
        ahead=0,
        behind=0,
    )

    fake_subprocess.register(
        ["git", "status", "--ignored", "--short", "-uall", "--", "output"],
        _result("!! output/benchmarks/run/episodes.jsonl\n"),
    )
    fake_subprocess.register(["git", "ls-files", "--", "output"], _result(""))

    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    artifacts = snapshot._inspect_output_root(row)

    assert artifacts == [
        snapshot.ArtifactRootInspection(
            root="output",
            classification="durable_required",
            status="ok",
            ignored_entries=1,
            sample_paths=["output/benchmarks/run/episodes.jsonl"],
            reason="output/ contains evidence-like paths",
        )
    ]


def test_output_root_inspection_allows_clean_baseline_tracked_output(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Clean output files already tracked on origin/main are not local preservation risk."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    row = _worktree_row(str(worktree))

    fake_subprocess.register(
        ["git", "status", "--ignored", "--short", "-uall", "--", "output"],
        _result(""),
    )
    fake_subprocess.register(
        ["git", "ls-files", "--", "output"],
        _result("output/fixtures/baseline.json\n"),
    )
    fake_subprocess.register(
        ["git", "ls-tree", "-r", "--name-only", "origin/main", "--", "output"],
        _result("output/fixtures/baseline.json\n"),
    )

    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    artifacts = snapshot._inspect_output_root(row)
    preserve, review, reasons = snapshot._artifact_retirement_risks(artifacts)

    assert artifacts == [
        snapshot.ArtifactRootInspection(
            root="output",
            classification="tracked_baseline",
            status="ok",
            tracked_entries=1,
            sample_paths=["output/fixtures/baseline.json"],
            reason="clean baseline-tracked files exist under output/",
        )
    ]
    assert preserve is False
    assert review is False
    assert reasons == []


def test_output_root_inspection_preserves_mixed_baseline_and_durable_output(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Baseline-tracked output does not mask ignored or untracked durable evidence."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    row = _worktree_row(str(worktree))

    fake_subprocess.register(
        ["git", "status", "--ignored", "--short", "-uall", "--", "output"],
        _result("!! output/benchmarks/run/episodes.jsonl\n"),
    )
    fake_subprocess.register(
        ["git", "ls-files", "--", "output"],
        _result("output/fixtures/baseline.json\n"),
    )
    fake_subprocess.register(
        ["git", "ls-tree", "-r", "--name-only", "origin/main", "--", "output"],
        _result("output/fixtures/baseline.json\n"),
    )

    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    artifacts = snapshot._inspect_output_root(row)
    preserve, review, reasons = snapshot._artifact_retirement_risks(artifacts)

    assert artifacts[0].classification == "durable_required"
    assert artifacts[0].ignored_entries == 1
    assert artifacts[0].tracked_entries == 1
    assert artifacts[0].sample_paths == [
        "output/benchmarks/run/episodes.jsonl",
        "output/fixtures/baseline.json",
    ]
    assert preserve is True
    assert review is False
    assert reasons == ["output has durable_required"]


def test_output_root_inspection_preserves_modified_tracked_output(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Modified tracked output files remain preservation evidence."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    row = _worktree_row(str(worktree))

    fake_subprocess.register(
        ["git", "status", "--ignored", "--short", "-uall", "--", "output"],
        _result(" M output/fixtures/baseline.json\n"),
    )
    fake_subprocess.register(
        ["git", "ls-files", "--", "output"],
        _result("output/fixtures/baseline.json\n"),
    )
    fake_subprocess.register(
        ["git", "ls-tree", "-r", "--name-only", "origin/main", "--", "output"],
        _result("output/fixtures/baseline.json\n"),
    )

    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    artifacts = snapshot._inspect_output_root(row)
    preserve, review, reasons = snapshot._artifact_retirement_risks(artifacts)

    assert artifacts[0].classification == "tracked_evidence"
    assert artifacts[0].reason == "tracked output/ files have local modifications"
    assert preserve is True
    assert review is False
    assert reasons == ["output has tracked_evidence"]


def test_output_root_inspection_preserves_branch_local_tracked_output(
    monkeypatch, tmp_path: Path, fake_subprocess
) -> None:
    """Tracked output files absent from origin/main remain preservation evidence."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    row = _worktree_row(str(worktree))

    fake_subprocess.register(
        ["git", "status", "--ignored", "--short", "-uall", "--", "output"],
        _result(""),
    )
    fake_subprocess.register(
        ["git", "ls-files", "--", "output"],
        _result("output/research_reports/local.json\n"),
    )
    fake_subprocess.register(
        ["git", "ls-tree", "-r", "--name-only", "origin/main", "--", "output"],
        _result(""),
    )

    monkeypatch.setattr(snapshot, "_run_command", fake_subprocess)

    artifacts = snapshot._inspect_output_root(row)
    preserve, review, reasons = snapshot._artifact_retirement_risks(artifacts)

    assert artifacts[0].classification == "tracked_evidence"
    assert artifacts[0].reason == "branch-local tracked files exist under output/"
    assert preserve is True
    assert review is False
    assert reasons == ["output has tracked_evidence"]
