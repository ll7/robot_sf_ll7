"""Contract tests for the guarded stacked-PR coordinator (issue #7345)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from scripts.dev.pr_metadata import metadata_digest, metadata_trailer
from scripts.dev.stacked_prs import (
    _get_paginated_list,
    _parse_expected_heads,
    _retarget_plan,
    _review_digest,
    build_stack_status,
    merge_cascade,
    retarget_stack,
    summarize_check_runs,
    summarize_merge_queue_gate,
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
    metadata = "e" * 64
    clear_holds = {
        key: {"status": "clear", "reason_codes": [], "source": "fixture"}
        for key in (
            "merge",
            "dependency",
            "draft",
            "domain",
            "scientific_evidence",
            "legal_release",
            "security",
        )
    }
    return {
        "pr": number,
        "state": "open",
        "head_ref": head_ref,
        "head_sha": head_sha,
        "base_ref": base_ref,
        "base_sha": "b" * 40,
        "metadata_digest": metadata,
        "merge_ready": True,
        "reasons": [],
        "checks": {"overall": "success"},
        "required_checks": {
            "status": "success",
            "head_sha": head_sha,
            "checks": [
                {
                    "name": "CI",
                    "head_sha": head_sha,
                    "status": "completed",
                    "conclusion": "success",
                    "state": "success",
                }
            ],
            "reason_codes": [],
        },
        "implementation_review": {
            "status": "accepted",
            "carrier": {
                "identity": "independent-fixture",
                "kind": "static_report",
                "head_sha": head_sha,
                "metadata_digest": metadata,
                "evidence_digest": "f" * 64,
                "verdict": "accepted",
            },
            "reason_codes": [],
            "precedence": 2,
        },
        "review_threads": {"status": "resolved", "unresolved": 0},
        "requested_reviewer_count": 0,
        "requested_team_count": 0,
        "holds": clear_holds,
        "merge_queue_gate": {
            "status": "success",
            "name": "merge-queue-gate",
            "head_sha": head_sha,
            "exact_head": True,
        },
        "closing_discipline": {
            "status": "passed",
            "blockers": [],
            "head_sha": head_sha,
            "body_sha256": hashlib.sha256(b"body").hexdigest(),
            "sources": {
                "pull_request": "live_pr_snapshot",
                "commits": "paginated_pr_commits",
                "issues": "current_issue_metadata",
            },
        },
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


def test_merge_queue_gate_requires_newest_exact_head_success() -> None:
    head_sha = "a" * 40
    older = {
        "id": 1,
        "name": "merge-queue-gate",
        "status": "completed",
        "conclusion": "success",
        "completed_at": "2026-08-17T10:00:00Z",
        "head_sha": head_sha,
    }
    newer_pending = {
        **older,
        "id": 2,
        "status": "in_progress",
        "conclusion": None,
        "completed_at": None,
        "started_at": "2026-08-17T11:00:00Z",
    }

    assert summarize_merge_queue_gate([], head_sha=head_sha)["status"] == "missing"
    assert (
        summarize_merge_queue_gate([older, newer_pending], head_sha=head_sha)["status"] == "pending"
    )
    assert (
        summarize_merge_queue_gate([{**older, "head_sha": "b" * 40}], head_sha=head_sha)["status"]
        == "mismatch"
    )
    assert (
        summarize_merge_queue_gate(
            [{key: value for key, value in older.items() if key != "head_sha"}],
            head_sha=head_sha,
        )["status"]
        == "malformed"
    )
    assert summarize_merge_queue_gate([older], head_sha=head_sha)["status"] == "success"


def test_explicit_holds_and_withdrawn_review_carriers_fail_closed() -> None:
    from scripts.dev.pr_loop_policy import (
        current_explicit_merge_hold_reasons,
        has_current_accepted_gate_verdict,
    )

    head_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
    trailer = f"gate-verdict: accepted @ {head_sha}"
    assert current_explicit_merge_hold_reasons(
        {
            "labels": ["merge-ready: no"],
            "reviews": [
                {
                    "authorAssociation": "OWNER",
                    "state": "APPROVED",
                    "body": "domain-approval: pending",
                }
            ],
        }
    ) == ["domain-approval:pending", "merge-ready:no"]

    for state in ("DISMISSED", "CHANGES_REQUESTED", "PENDING"):
        assert not has_current_accepted_gate_verdict(
            {
                "reviews": [
                    {
                        "authorAssociation": "OWNER",
                        "state": state,
                        "body": trailer,
                    }
                ]
            },
            head_sha,
        )
    assert has_current_accepted_gate_verdict(
        {
            "reviews": [
                {
                    "authorAssociation": "OWNER",
                    "state": "APPROVED",
                    "body": trailer,
                }
            ]
        },
        head_sha,
    )


def test_status_positive_control_requires_current_merge_queue_gate(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    main_sha = "m" * 40
    head_sha = "a" * 40
    title = "feat: stack 1"
    body = "body"
    review_body = (
        f"gate-verdict: accepted @ {head_sha}\n{metadata_trailer(metadata_digest(title, body))}"
    )
    payloads = {
        "repos/owner/repo/git/ref/heads/main": {"object": {"sha": main_sha}},
        "repos/owner/repo/pulls/1": {
            **_pr_payload(
                1,
                head_ref="root",
                head_sha=head_sha,
                base_ref="main",
                base_sha=main_sha,
            ),
            "title": title,
            "body": body,
        },
        "repos/owner/repo/pulls/1/reviews?per_page=100": [
            {
                "author_association": "OWNER",
                "state": "APPROVED",
                "body": review_body,
            }
        ],
        "repos/owner/repo/pulls/1/comments?per_page=100": [],
        "repos/owner/repo/issues/1/comments?per_page=100": [],
        f"repos/owner/repo/commits/{head_sha}/check-runs?per_page=100": {
            "total_count": 2,
            "check_runs": [
                {
                    "id": 1,
                    "name": "CI",
                    "status": "completed",
                    "conclusion": "success",
                    "head_sha": head_sha,
                },
                {
                    "id": 2,
                    "name": "merge-queue-gate",
                    "status": "completed",
                    "conclusion": "success",
                    "head_sha": head_sha,
                    "completed_at": "2026-08-17T11:00:00Z",
                },
            ],
        },
    }

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        assert method == "GET"
        assert payload is None
        return payloads[path], None

    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_closing_discipline_evidence",
        lambda *args, **kwargs: {
            "status": "passed",
            "blockers": [],
            "head_sha": head_sha,
            "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "sources": {
                "pull_request": "live_pr_snapshot",
                "commits": "paginated_pr_commits",
                "issues": "current_issue_metadata",
            },
        },
    )

    result = build_stack_status(
        "owner/repo",
        [1],
        api=fake_api,
        thread_fetcher=lambda number: (True, None),
    )

    entry = result["entries"][0]
    assert entry["merge_queue_gate"]["status"] == "success"
    assert entry["explicit_holds"] == []
    assert entry["merge_ready"] is True


def test_review_digest_changes_when_review_content_changes() -> None:
    review = {"id": 1, "state": "COMMENTED", "body": "first", "user": {"login": "owner"}}
    changed = {**review, "body": "second"}

    first = _review_digest([review], [], [])
    second = _review_digest([changed], [], [])

    assert first != second


def test_paginated_list_reads_full_first_page_and_records_provenance() -> None:
    first_page = [{"id": index} for index in range(100)]
    calls: list[str] = []

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        assert method == "GET"
        assert payload is None
        calls.append(path)
        if len(calls) == 1:
            return first_page, None
        return [{"id": 100}], None

    rows, pagination, error = _get_paginated_list(
        "repos/owner/repo/pulls/1/reviews?per_page=100", api=fake_api
    )

    assert error is None
    assert rows is not None and len(rows) == 101
    assert pagination == {
        "pages_read": 2,
        "page_size": 100,
        "page_budget": 100,
        "row_count": 101,
        "truncated": False,
    }
    assert calls == [
        "repos/owner/repo/pulls/1/reviews?per_page=100",
        "repos/owner/repo/pulls/1/reviews?per_page=100&page=2",
    ]


def test_paginated_list_accepts_check_run_object_envelope() -> None:
    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        assert method == "GET"
        assert payload is None
        return {"total_count": 1, "check_runs": [{"id": 1, "name": "CI"}]}, None

    rows, pagination, error = _get_paginated_list(
        "repos/owner/repo/commits/sha/check-runs?per_page=100",
        api=fake_api,
        response_key="check_runs",
    )

    assert error is None
    assert rows == [{"id": 1, "name": "CI"}]
    assert pagination is not None and pagination["row_count"] == 1


def test_paginated_list_rejects_malformed_page() -> None:
    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        assert method == "GET"
        assert payload is None
        if path.endswith("page=2"):
            return {"not": "a list"}, None
        return [{"id": index} for index in range(100)], None

    rows, pagination, error = _get_paginated_list(
        "repos/owner/repo/pulls/1/comments?per_page=100", api=fake_api
    )

    assert rows is None
    assert pagination is None
    assert "was not a list" in (error or "")


def test_paginated_list_fails_closed_at_page_budget() -> None:
    calls: list[str] = []

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        assert method == "GET"
        assert payload is None
        calls.append(path)
        return [{"id": len(calls)} for _ in range(100)], None

    rows, pagination, error = _get_paginated_list(
        "repos/owner/repo/issues/1/comments?per_page=100",
        api=fake_api,
        page_budget=2,
    )

    assert rows is None
    assert pagination == {
        "pages_read": 2,
        "page_size": 100,
        "page_budget": 2,
        "row_count": 200,
        "truncated": True,
    }
    assert "may be truncated" in (error or "")
    assert len(calls) == 2


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
        f"repos/owner/repo/commits/{root_sha}/check-runs?per_page=100": {
            "total_count": 1,
            "check_runs": [{"id": 1, "name": "CI", "status": "completed", "conclusion": "success"}],
        },
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
                "pagination": {
                    "reviews": {},
                    "review_comments": {},
                    "conversation_comments": {},
                },
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_closing_discipline_evidence",
        lambda *args, **kwargs: {
            "status": "passed",
            "blockers": [],
            "head_sha": root_sha,
            "body_sha256": hashlib.sha256(b"body").hexdigest(),
            "sources": {
                "pull_request": "live_pr_snapshot",
                "commits": "paginated_pr_commits",
                "issues": "current_issue_metadata",
            },
        },
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


def test_merge_cascade_rejects_status_success_without_closing_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    entry = _ready_entry(1, head_ref="root", head_sha="a" * 40, base_ref="main")
    entry.pop("closing_discipline")
    snapshot = {
        "schema": "stacked_prs.v1",
        "status": "ok",
        "main": {"sha": "1" * 40},
        "entries": [entry],
        "all_merge_ready": True,
    }
    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_stack_status", lambda *args, **kwargs: snapshot
    )
    calls: list[tuple[str, str]] = []

    def fake_api(method: str, path: str, body: dict[str, Any] | None) -> tuple[Any, None]:
        calls.append((method, path))
        return {}, None

    result = merge_cascade(
        "owner/repo",
        [1],
        expected_heads={1: entry["head_sha"]},
        apply=True,
        api=fake_api,
    )

    assert result["status"] == "blocked"
    assert "closing_discipline_unavailable" in result["receipt"]["reason_codes"]
    assert calls == []


def test_merge_cascade_squashes_root_then_explicitly_retargets_next(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root_sha = "a" * 40
    next_sha = "c" * 40
    parent_ref = "root-branch"
    snapshot = {
        "schema": "stacked_prs.v1",
        "status": "ok",
        "main": {"sha": "1" * 40},
        "entries": [
            _ready_entry(1, head_ref=parent_ref, head_sha=root_sha, base_ref="main"),
            _ready_entry(2, head_ref="child-branch", head_sha=next_sha, base_ref=parent_ref),
        ],
        "all_merge_ready": True,
    }
    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_stack_status", lambda *args, **kwargs: snapshot
    )
    captured_receipt: dict[str, Any] = {}

    def fake_apply(receipt: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], None]:
        captured_receipt.update(receipt)
        return {
            "pr": receipt["pr_number"],
            "merge_commit_sha": "d" * 40,
            "receipt": receipt,
        }, None

    monkeypatch.setattr("scripts.dev.stacked_prs.apply_guarded_merge", fake_apply)
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
    assert captured_receipt["closing_discipline"]["status"] == "passed"
    assert not any(call[0] == "PUT" and "/merge" in call[1] for call in calls)


def test_cli_json_status_error_is_machine_readable(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "scripts.dev.stacked_prs.build_stack_status",
        lambda repo, prs: {"schema": "stacked_prs.v1", "status": "error", "error": "fixture"},
    )
    from scripts.dev.stacked_prs import main

    assert main(["status", "--repo", "owner/repo", "--prs", "1", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "fixture"


# ---------------------------------------------------------------------------
# Issue #7515: check-ancestry gate
# ---------------------------------------------------------------------------


def _ancestry_payload(
    pr_number: int,
    *,
    head_ref: str,
    head_sha: str,
    base_ref: str,
    body: str = "",
) -> dict[str, Any]:
    """Build a REST pull payload for check-ancestry fixture tests."""
    return {
        "number": pr_number,
        "title": f"feat: ancestry {pr_number}",
        "body": body,
        "state": "open",
        "merged": False,
        "draft": False,
        "head": {"ref": head_ref, "sha": head_sha},
        "base": {"ref": base_ref, "sha": "b" * 40},
    }


def _ancestry_git_runner(tmp_path: Path) -> Any:
    """Build a fake git runner with a synthetic contaminated-history layout.

    ``origin/main`` sits at ``b*40``; the child head ``c*40`` has a non-main
    ancestry whose merge base is ``a*40`` and whose commits include a foreign
    commit plus the branch's own commit.
    """

    def fake_git(args: list[str], worktree: Path) -> subprocess.CompletedProcess[str]:
        if args == ["fetch", "--no-tags", "origin", "main"]:
            return _completed(args)
        if args == ["rev-parse", "refs/remotes/origin/main"]:
            return _completed(args, stdout="b" * 40 + "\n")
        if args == ["merge-base", "refs/remotes/origin/main", "c" * 40]:
            return _completed(args, stdout="a" * 40 + "\n")
        if args == ["log", "--oneline", f"refs/remotes/origin/main..{'c' * 40}"]:
            return _completed(
                args, stdout=f"{'f' * 7} foreign work (#7308)\n{'e' * 7} intended work\n"
            )
        if args[:2] == ["diff", "--name-only"]:
            return _completed(args, stdout="robot_sf/foreign.py\nrobot_sf/own.py\n")
        raise AssertionError(f"unexpected git call: {args}")

    return fake_git


def test_check_ancestry_classifies_undeclared_contamination_blocked(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    """A PR with undeclared non-main ancestry fails closed (issue #7515)."""
    from scripts.dev.stacked_prs import check_ancestry

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        if path.endswith("/pulls/1"):
            return _ancestry_payload(
                1,
                head_ref="fix/child",
                head_sha="c" * 40,
                base_ref="main",
                body="no declaration",
            ), None
        raise AssertionError(f"unexpected api call: {path}")

    result = check_ancestry(
        "owner/repo",
        target="1",
        worktree=tmp_path,
        api=fake_api,
        git_runner=_ancestry_git_runner(tmp_path),
    )

    assert result["state"] == "undeclared_stack"
    assert result["status"] == "blocked"
    assert result["unexpected_commits"] == [
        f"{'f' * 7} foreign work (#7308)",
        f"{'e' * 7} intended work",
    ]
    assert result["unexpected_paths"] == ["robot_sf/foreign.py", "robot_sf/own.py"]
    assert "remediation_command" in result


def test_check_ancestry_classifies_clean_pr_ok(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A PR created from current main with only intended commits passes."""
    from scripts.dev.stacked_prs import check_ancestry

    def fake_git(args: list[str], worktree: Path) -> subprocess.CompletedProcess[str]:
        if args == ["fetch", "--no-tags", "origin", "main"]:
            return _completed(args)
        if args == ["rev-parse", "refs/remotes/origin/main"]:
            return _completed(args, stdout="b" * 40 + "\n")
        if args == ["merge-base", "refs/remotes/origin/main", "c" * 40]:
            return _completed(args, stdout="b" * 40 + "\n")
        if args == ["log", "--oneline", f"refs/remotes/origin/main..{'c' * 40}"]:
            return _completed(args, stdout=f"{'e' * 7} intended work\n")
        if args[:2] == ["diff", "--name-only"]:
            return _completed(args, stdout="robot_sf/own.py\n")
        raise AssertionError(f"unexpected git call: {args}")

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        if path.endswith("/pulls/2"):
            return _ancestry_payload(
                2,
                head_ref="fix/clean",
                head_sha="c" * 40,
                base_ref="main",
            ), None
        raise AssertionError(f"unexpected api call: {path}")

    result = check_ancestry(
        "owner/repo",
        target="2",
        worktree=tmp_path,
        api=fake_api,
        git_runner=fake_git,
    )

    assert result["state"] == "clean"
    assert result["status"] == "ok"


def test_check_ancestry_classifies_declared_stack(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A declared stack is stacked (never independently mergeable), not blocked."""
    from scripts.dev.stacked_prs import check_ancestry

    declaration = f"## Stack Declaration\nparent_pr: #7308\nparent_head: {'a' * 40}\n"
    calls: list[str] = []

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        calls.append(path)
        if path.endswith("/pulls/3"):
            return _ancestry_payload(
                3,
                head_ref="fix/child",
                head_sha="c" * 40,
                base_ref="main",
                body=declaration,
            ), None
        if path.endswith("/pulls/7308"):
            return _ancestry_payload(
                7308,
                head_ref="fix/parent",
                head_sha="a" * 40,
                base_ref="main",
            ), None
        raise AssertionError(f"unexpected api call: {path}")

    result = check_ancestry(
        "owner/repo",
        target="3",
        worktree=tmp_path,
        api=fake_api,
        git_runner=_ancestry_git_runner(tmp_path),
    )

    assert result["state"] == "stacked"
    assert result["status"] == "ok"
    assert result["classification"] == "stacked_not_independently_mergeable"
    assert result["mergeable"] is False
    assert any(path.endswith("/pulls/7308") for path in calls)


def test_check_ancestry_invalidates_closed_unmerged_parent(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A closed-unmerged declared parent fails closed (issue #7515)."""
    from scripts.dev.stacked_prs import check_ancestry

    declaration = f"## Stack Declaration\nparent_pr: #7308\nparent_head: {'a' * 40}\n"

    def fake_api(method: str, path: str, payload: dict[str, Any] | None) -> tuple[Any, None]:
        if path.endswith("/pulls/4"):
            return _ancestry_payload(
                4,
                head_ref="fix/child",
                head_sha="c" * 40,
                base_ref="main",
                body=declaration,
            ), None
        if path.endswith("/pulls/7308"):
            payload = _ancestry_payload(
                7308,
                head_ref="fix/parent",
                head_sha="a" * 40,
                base_ref="main",
                body="",
            )
            payload["state"] = "closed"
            payload["merged"] = False
            return payload, None
        raise AssertionError(f"unexpected api call: {path}")

    result = check_ancestry(
        "owner/repo",
        target="4",
        worktree=tmp_path,
        api=fake_api,
        git_runner=_ancestry_git_runner(tmp_path),
    )

    assert result["state"] == "parent_invalidated"
    assert result["status"] == "blocked"
