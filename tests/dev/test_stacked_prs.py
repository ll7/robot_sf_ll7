"""Contract tests for the guarded stacked-PR coordinator (issue #7345)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from scripts.dev.stacked_prs import (
    _parse_expected_heads,
    _retarget_plan,
    _review_digest,
    build_stack_status,
    merge_cascade,
    retarget_stack,
    summarize_check_runs,
    sync_stack,
)


def _completed(
    args: list[str], *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Build a deterministic subprocess result for git fakes."""
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def _pr_payload(
    number: int,
    *,
    head_ref: str,
    head_sha: str,
    base_ref: str,
    base_sha: str,
    state: str = "open",
    merged: bool = False,
) -> dict[str, Any]:
    """Build the REST pull-request shape consumed by the coordinator."""
    return {
        "number": number,
        "title": f"feat: stack {number}",
        "body": "body",
        "state": state,
        "merged": merged,
        "draft": False,
        "mergeable": True,
        "mergeable_state": "clean",
        "head": {"ref": head_ref, "sha": head_sha},
        "base": {"ref": base_ref, "sha": base_sha},
        "labels": [{"name": "merge-ready"}],
        "requested_reviewers": [],
        "requested_teams": [],
    }


def _ready_entry(number: int, *, head_ref: str, head_sha: str, base_ref: str) -> dict[str, Any]:
    """Build a compact already-validated stack entry for cascade tests."""
    return {
        "pr": number,
        "head_ref": head_ref,
        "head_sha": head_sha,
        "base_ref": base_ref,
        "base_sha": "b" * 40,
        "merge_ready": True,
        "reasons": [],
        "checks": {"overall": "success"},
    }


def test_expected_head_parser_requires_unique_hex_guards() -> None:
    expected, error = _parse_expected_heads(["12=" + "a" * 40, "13=" + "b" * 12])

    assert error is None
    assert expected == {12: "a" * 40, 13: "b" * 12}

    duplicate, duplicate_error = _parse_expected_heads(["12=" + "a" * 40, "12=" + "b" * 40])
    malformed, malformed_error = _parse_expected_heads(["12=not-a-sha"])
    assert duplicate is None
    assert "duplicate" in (duplicate_error or "")
    assert malformed is None
    assert "7-40 hex" in (malformed_error or "")


def test_retarget_plan_maps_root_to_main_and_children_to_parent_branch() -> None:
    plan = _retarget_plan(
        [
            {"pr": 10, "head_ref": "stack-root", "head_sha": "a" * 40, "base_ref": "main"},
            {"pr": 11, "head_ref": "stack-child", "head_sha": "b" * 40, "base_ref": "main"},
            {"pr": 12, "head_ref": "stack-tip", "head_sha": "c" * 40, "base_ref": "stack-child"},
        ]
    )

    assert [item["desired_base_ref"] for item in plan] == ["main", "stack-root", "stack-child"]
    assert [item["change_required"] for item in plan] == [False, True, False]


def test_check_summary_drops_older_cancelled_run() -> None:
    summary = summarize_check_runs(
        [
            {
                "id": 1,
                "name": "CI",
                "status": "completed",
                "conclusion": "cancelled",
                "completed_at": "2026-08-17T10:00:00Z",
            },
            {
                "id": 2,
                "name": "CI",
                "status": "completed",
                "conclusion": "success",
                "completed_at": "2026-08-17T11:00:00Z",
            },
            {
                "id": 3,
                "name": "Lint",
                "status": "completed",
                "conclusion": "neutral",
                "completed_at": "2026-08-17T11:00:00Z",
            },
        ]
    )

    assert summary["overall"] == "success"
    assert summary["superseded_count"] == 1
    assert summary["failures"] == []


def test_check_summary_fails_closed_for_pending_and_failed_current_runs() -> None:
    pending = summarize_check_runs(
        [{"id": 1, "name": "CI", "status": "in_progress", "conclusion": None}]
    )
    failed = summarize_check_runs(
        [{"id": 1, "name": "CI", "status": "completed", "conclusion": "failure"}]
    )
    missing = summarize_check_runs([])

    assert pending["overall"] == "pending"
    assert failed["overall"] == "failure"
    assert missing["overall"] == "unknown"


def test_review_digest_changes_when_review_content_changes() -> None:
    review = {"id": 1, "state": "COMMENTED", "body": "first", "user": {"login": "owner"}}
    changed = {**review, "body": "second"}

    first = _review_digest([review], [], [])
    second = _review_digest([changed], [], [])

    assert first != second


def test_status_rejects_unknown_threads_without_remote_writes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Status must expose thread uncertainty instead of treating it as green."""
    main_sha = "m" * 40
    root_sha = "a" * 40
    payloads = {
        "repos/owner/repo/git/ref/heads/main": {"object": {"sha": main_sha}},
        "repos/owner/repo/pulls/1": _pr_payload(
            1,
            head_ref="root",
            head_sha=root_sha,
            base_ref="main",
            base_sha=main_sha,
        ),
        "repos/owner/repo/pulls/1/reviews?per_page=100": [],
        "repos/owner/repo/pulls/1/comments?per_page=100": [],
        "repos/owner/repo/issues/1/comments?per_page=100": [],
        f"repos/owner/repo/commits/{root_sha}/check-runs?per_page=100": [
            {"id": 1, "name": "CI", "status": "completed", "conclusion": "success"}
        ],
    }

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        assert method == "GET"
        return payloads[path], None

    monkeypatch.setattr(
        "scripts.dev.stacked_prs._fetch_review_data",
        lambda repo, number, api: (
            {
                "reviews": [],
                "review_comments": [],
                "conversation_comments": [],
                "review_digest": "digest",
                "review_states": {},
            },
            None,
        ),
    )
    result = build_stack_status(
        "owner/repo",
        [1],
        api=fake_api,
        thread_fetcher=lambda number: (None, "GraphQL unavailable"),
    )

    assert result["status"] == "ok"
    assert result["entries"][0]["merge_ready"] is False
    assert "review_threads_not_evaluated" in result["entries"][0]["reasons"]


def test_retarget_apply_refuses_any_head_mismatch_before_patch() -> None:
    payload = _pr_payload(
        1,
        head_ref="root",
        head_sha="a" * 40,
        base_ref="feature",
        base_sha="b" * 40,
    )
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_api(method: str, path: str, body: dict[str, Any] | None) -> tuple[Any, None]:
        calls.append((method, path, body))
        if method == "GET":
            return payload, None
        raise AssertionError("PATCH must not run after an exact-head mismatch")

    result = retarget_stack(
        "owner/repo",
        [1],
        expected_heads={1: "c" * 40},
        apply=True,
        api=fake_api,
    )

    assert result["status"] == "blocked"
    assert all(method == "GET" for method, _, _ in calls)


def test_sync_dry_run_is_clean_and_never_force_pushes(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_git(args: list[str], worktree: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args == ["status", "--porcelain"]:
            return _completed(args)
        if args == ["branch", "--show-current"]:
            return _completed(args, stdout="original\n")
        if args == ["worktree", "list", "--porcelain"]:
            return _completed(
                args, stdout=f"worktree {worktree.resolve()}\nbranch refs/heads/original\n\n"
            )
        raise AssertionError(f"dry-run should not execute mutation: {args}")

    result = sync_stack(
        ["root", "child"],
        worktree=tmp_path,
        git_runner=fake_git,
    )

    assert result["status"] == "dry_run"
    assert [command[-1] for command in result["commands"] if command[0] == "push"] == [
        "root",
        "child",
    ]
    assert all("--force" not in command for command in result["commands"])
    assert commands[-1] == ["worktree", "list", "--porcelain"]


def test_sync_apply_restores_original_branch(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    current_branch = "original"

    def fake_git(args: list[str], worktree: Path) -> subprocess.CompletedProcess[str]:
        nonlocal current_branch
        commands.append(args)
        if args == ["status", "--porcelain"]:
            return _completed(args)
        if args == ["branch", "--show-current"]:
            return _completed(args, stdout=f"{current_branch}\n")
        if args == ["worktree", "list", "--porcelain"]:
            return _completed(
                args, stdout=f"worktree {worktree.resolve()}\nbranch refs/heads/original\n\n"
            )
        if args[0] == "checkout":
            current_branch = args[1]
        return _completed(args)

    result = sync_stack(
        ["root", "child"],
        worktree=tmp_path,
        apply=True,
        git_runner=fake_git,
    )

    assert result["status"] == "applied"
    assert result["restored_original_branch"] is True
    assert commands[-1] == ["checkout", "original"]
    assert not any("--force" in command for command in commands)


def test_merge_cascade_requires_all_exact_heads_before_mutating(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    entries = [_ready_entry(1, head_ref="root", head_sha="a" * 40, base_ref="main")]
    snapshot = {
        "schema": "stacked_prs.v1",
        "status": "ok",
        "entries": entries,
        "all_merge_ready": True,
    }
    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_stack_status", lambda *args, **kwargs: snapshot
    )
    calls: list[tuple[str, str]] = []

    def fake_api(method: str, path: str, body: dict[str, Any] | None) -> tuple[Any, None]:
        calls.append((method, path))
        return {}, None

    result = merge_cascade("owner/repo", [1], apply=True, api=fake_api)

    assert result["status"] == "blocked"
    assert calls == []


def test_merge_cascade_squashes_root_then_explicitly_retargets_next(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root_sha = "a" * 40
    next_sha = "c" * 40
    parent_ref = "root-branch"
    snapshot = {
        "schema": "stacked_prs.v1",
        "status": "ok",
        "entries": [
            _ready_entry(1, head_ref=parent_ref, head_sha=root_sha, base_ref="main"),
            _ready_entry(2, head_ref="child-branch", head_sha=next_sha, base_ref=parent_ref),
        ],
        "all_merge_ready": True,
    }
    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_stack_status", lambda *args, **kwargs: snapshot
    )
    root_payload = _pr_payload(
        1,
        head_ref=parent_ref,
        head_sha=root_sha,
        base_ref="main",
        base_sha="b" * 40,
        state="closed",
        merged=True,
    )
    next_payload = _pr_payload(
        2,
        head_ref="child-branch",
        head_sha=next_sha,
        base_ref=parent_ref,
        base_sha=root_sha,
    )
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_api(method: str, path: str, body: dict[str, Any] | None) -> tuple[Any, None]:
        calls.append((method, path, body))
        if method == "PUT":
            return {"merged": True, "sha": "d" * 40}, None
        if method == "PATCH":
            next_payload["base"] = {"ref": "main", "sha": "e" * 40}
            return next_payload, None
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "e" * 40}}, None
        if "/pulls/1" in path:
            return root_payload, None
        if "/pulls/2" in path:
            return next_payload, None
        raise AssertionError(path)

    result = merge_cascade(
        "owner/repo",
        [1, 2],
        expected_heads={1: root_sha, 2: next_sha},
        apply=True,
        api=fake_api,
    )

    assert result["status"] == "merged_waiting_for_ci"
    assert result["base_advance"] == {"mode": "explicit", "base_ref": "main"}
    assert [call[0:2] for call in calls].count(("PUT", "repos/owner/repo/pulls/1/merge")) == 1
    merge_call = next(call for call in calls if call[0] == "PUT")
    assert merge_call[2] == {"sha": root_sha, "merge_method": "squash"}
    assert not any(call[0] == "PUT" and "/pulls/2/merge" in call[1] for call in calls)


def test_cli_json_status_error_is_machine_readable(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_stack_status",
        lambda repo, prs: {"schema": "stacked_prs.v1", "status": "error", "error": "fixture"},
    )
    from scripts.dev.stacked_prs import main

    assert main(["status", "--repo", "owner/repo", "--prs", "1", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "fixture"
