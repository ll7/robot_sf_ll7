"""Tests for the fail-closed pre-publication remote-state gate."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from scripts.dev import check_prepublication_state as gate


def _snapshot(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid state snapshot for pure decision tests."""
    payload: dict[str, Any] = {
        "schema": gate.SCHEMA,
        "kind": "snapshot",
        "captured_at_utc": "2026-08-12T10:00:00+00:00",
        "repo": "ll7/robot_sf_ll7",
        "issue": 6916,
        "issue_state": "OPEN",
        "issue_updated_at": "2026-08-12T10:00:00Z",
        "issue_closed_at": None,
        "closing_prs": [],
        "open_covering_prs": [],
        "remote": "origin",
        "base_ref": "main",
        "base_sha": "base-a",
        "branch": "feature/fresh-state",
        "remote_branch_sha": "branch-a",
        "local_head_sha": "head-a",
        "tree_state": "clean",
    }
    payload.update(overrides)
    return payload


def test_unchanged_state_is_ready_and_records_exact_shas() -> None:
    """A no-op remote refresh should permit publication."""
    baseline = _snapshot()

    result = gate.evaluate_state(baseline, _snapshot())

    assert result["decision"] == "ready"
    assert result["reason"] == "remote_state_unchanged"
    assert result["exact_shas"] == {
        "baseline": {
            "base_sha": "base-a",
            "remote_branch_sha": "branch-a",
            "local_head_sha": "head-a",
        },
        "current": {
            "base_sha": "base-a",
            "remote_branch_sha": "branch-a",
            "local_head_sha": "head-a",
        },
    }


def test_newly_merged_closing_pr_supersedes_issue() -> None:
    """A newly observed closing PR must stop duplicate publication."""
    baseline = _snapshot()
    current = _snapshot(
        closing_prs=[
            {
                "number": 7001,
                "title": "fix: deliver the claimed issue",
                "merged_at": "2026-08-12T10:02:00Z",
            }
        ]
    )

    result = gate.evaluate_state(baseline, current)

    assert result["decision"] == "superseded"
    assert result["reason"] == "merged_pr_closes_issue"
    assert result["new_closing_prs"][0]["number"] == 7001


def test_newly_opened_covering_pr_supersedes_issue() -> None:
    """A newly opened explicit covering PR must stop duplicate publication."""
    baseline = _snapshot()
    current = _snapshot(
        open_covering_prs=[
            {
                "number": 7002,
                "title": "fix: deliver the claimed issue",
                "created_at": "2026-08-12T10:02:00Z",
            }
        ]
    )

    result = gate.evaluate_state(baseline, current)

    assert result["decision"] == "superseded"
    assert result["reason"] == "open_pr_covers_issue"
    assert result["new_open_covering_prs"][0]["number"] == 7002


def test_closed_issue_supersedes_even_without_pr_search_match() -> None:
    """The issue state remains a safe supersession signal if PR search is sparse."""
    result = gate.evaluate_state(_snapshot(), _snapshot(issue_state="CLOSED"))

    assert result["decision"] == "superseded"
    assert result["reason"] == "issue_closed"


def test_base_movement_requires_refresh() -> None:
    """A moving base invalidates the pre-publication snapshot."""
    result = gate.evaluate_state(_snapshot(), _snapshot(base_sha="base-b"))

    assert result["decision"] == "refresh-required"
    assert result["reason"] == "base_changed"
    assert result["drift"]["base_sha"] == {"baseline": "base-a", "current": "base-b"}


def test_remote_branch_tip_movement_requires_refresh() -> None:
    """A changed remote branch tip is a publication race, even if main is stable."""
    result = gate.evaluate_state(_snapshot(), _snapshot(remote_branch_sha="branch-b"))

    assert result["decision"] == "refresh-required"
    assert result["reason"] == "remote_branch_changed"
    assert result["drift"]["remote_branch_sha"] == {
        "baseline": "branch-a",
        "current": "branch-b",
    }


def test_local_head_movement_requires_refresh() -> None:
    """A local commit after the baseline invalidates readiness evidence."""
    result = gate.evaluate_state(_snapshot(), _snapshot(local_head_sha="head-b"))

    assert result["decision"] == "refresh-required"
    assert result["reason"] == "local_head_changed"


def test_dirty_worktree_blocks_even_when_shas_are_unchanged() -> None:
    """A dirty worktree cannot support a trustworthy publication decision."""
    result = gate.evaluate_state(_snapshot(), _snapshot(tree_state="dirty"))

    assert result["decision"] == "blocked"
    assert result["reason"] == "dirty_worktree"


def test_unknown_current_issue_state_blocks() -> None:
    """An unexpected live issue state must fail closed."""
    result = gate.evaluate_state(_snapshot(), _snapshot(issue_state="LOCKED"))

    assert result["decision"] == "blocked"
    assert result["reason"] == "issue_state_unknown"


def test_unknown_baseline_issue_state_blocks() -> None:
    """An unexpected baseline issue state must not permit publication."""
    result = gate.evaluate_state(_snapshot(issue_state="LOCKED"), _snapshot())

    assert result["decision"] == "blocked"
    assert result["reason"] == "baseline_issue_state_unknown"


def test_snapshot_paths_disambiguate_sanitization_collisions() -> None:
    """Branches that sanitize to one filename receive distinct snapshot paths."""
    first = gate._default_snapshot_path("feature/fresh-state")
    second = gate._default_snapshot_path("feature-fresh-state")

    assert first != second
    assert first == gate._default_snapshot_path("feature/fresh-state")


def test_run_converts_timeout_to_gate_error(monkeypatch) -> None:
    """A stalled external command becomes a bounded fail-closed error."""

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    with pytest.raises(gate.GateError, match="timed out after 3s"):
        gate._run(["git", "fetch"], timeout=3)


def test_fetch_refs_uses_exact_remote_refs(monkeypatch) -> None:
    """Ref refresh must not rely on broad branch-pattern matching."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:2] == ["git", "ls-remote"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "branch-a\trefs/heads/feature/fresh-state\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate, "_run", fake_run)
    monkeypatch.setattr(gate, "_git_output", lambda *_: "base-a")

    assert gate._fetch_refs(remote="origin", base_ref="main", branch="feature/fresh-state") == (
        "base-a",
        "branch-a",
    )
    assert [
        "git",
        "fetch",
        "--no-tags",
        "origin",
        "refs/heads/main:refs/remotes/origin/main",
    ] in commands
    assert [
        "git",
        "ls-remote",
        "--heads",
        "origin",
        "refs/heads/feature/fresh-state",
    ] in commands


def test_fetch_refs_normalizes_remote_qualified_base_ref(monkeypatch) -> None:
    """A remote-qualified base ref must fetch the underlying branch name."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:2] == ["git", "ls-remote"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "branch-a\trefs/heads/feature/fresh-state\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate, "_run", fake_run)
    monkeypatch.setattr(gate, "_git_output", lambda *_: "base-a")

    assert gate._fetch_refs(
        remote="origin", base_ref="origin/main", branch="feature/fresh-state"
    ) == ("base-a", "branch-a")
    assert [
        "git",
        "fetch",
        "--no-tags",
        "origin",
        "refs/heads/main:refs/remotes/origin/main",
    ] in commands


def test_collect_live_state_records_explicit_closing_pr(monkeypatch) -> None:
    """Live collection extracts explicit references from merged and open PR data."""
    commands: list[list[str]] = []

    def fake_json(command: list[str]) -> Any:
        commands.append(command)
        if command[1] == "issue":
            return {"state": "OPEN", "updatedAt": "2026-08-12T10:00:00Z", "closedAt": None}
        if command[command.index("--state") + 1] == "open":
            return [
                {
                    "number": 7002,
                    "title": "fix: unrelated issue",
                    "body": "Closes otherorg/otherrepo#6916",
                    "createdAt": "2026-08-12T10:04:00Z",
                    "updatedAt": "2026-08-12T10:05:00Z",
                    "isDraft": True,
                    "headRefName": "fix/other",
                    "headRefOid": "head-other",
                    "baseRefName": "main",
                },
                {
                    "number": 7003,
                    "title": "fix: open covering PR",
                    "body": "Closes https://github.com/ll7/robot_sf_ll7/issues/6916",
                    "createdAt": "2026-08-12T10:06:00Z",
                    "updatedAt": "2026-08-12T10:07:00Z",
                    "isDraft": False,
                    "headRefName": "fix/6916-open",
                    "headRefOid": "head-open",
                    "baseRefName": "main",
                },
            ]
        return [
            {
                "number": 7000,
                "title": "fix unrelated issue",
                "body": "Closes otherorg/otherrepo#6916",
                "mergedAt": "2026-08-12T10:02:00Z",
                "mergeCommit": {"oid": "merge-a"},
                "headRefName": "fix/6916",
                "headRefOid": "head-fix",
                "baseRefName": "main",
            },
            {
                "number": 7001,
                "title": "fix: close the issue",
                "body": "Closes https://github.com/ll7/robot_sf_ll7/issues/6916",
                "mergedAt": "2026-08-12T10:03:00Z",
                "mergeCommit": {"oid": "merge-b"},
                "headRefName": "fix/6916-target",
                "headRefOid": "head-fix-target",
                "baseRefName": "main",
            },
        ]

    monkeypatch.setattr(gate, "_json_command", fake_json)
    monkeypatch.setattr(gate, "_fetch_refs", lambda **_: ("base-a", "branch-a"))
    monkeypatch.setattr(gate, "_git_output", lambda *_: "head-a")
    monkeypatch.setattr(gate, "_tree_state", lambda: "clean")

    result = gate.collect_live_state(
        repo="ll7/robot_sf_ll7",
        issue=6916,
        branch="feature/fresh-state",
        base_ref="origin/main",
    )

    assert result["base_ref"] == "main"
    assert result["closing_prs"] == [
        {
            "number": 7001,
            "title": "fix: close the issue",
            "merged_at": "2026-08-12T10:03:00Z",
            "merge_commit": {"oid": "merge-b"},
            "head_ref": "fix/6916-target",
            "head_sha": "head-fix-target",
            "base_ref": "main",
        }
    ]
    assert result["open_covering_prs"] == [
        {
            "number": 7003,
            "title": "fix: open covering PR",
            "created_at": "2026-08-12T10:06:00Z",
            "updated_at": "2026-08-12T10:07:00Z",
            "is_draft": False,
            "head_ref": "fix/6916-open",
            "head_sha": "head-open",
            "base_ref": "main",
        }
    ]
    assert result["remote_state_sources"] == {
        "issue": "graphql",
        "closing_prs": "graphql",
        "open_covering_prs": "graphql",
    }
    assert result["remote_state_fallbacks"] == {}
    assert [command[1] for command in commands] == ["issue", "pr", "pr"]


def test_collect_live_state_falls_back_to_rest_per_remote_field(monkeypatch) -> None:
    """GraphQL quota exhaustion should produce a mixed-source, auditable snapshot."""
    commands: list[list[str]] = []

    def fail_graphql(command: list[str]) -> Any:
        commands.append(command)
        raise gate.GateError("GraphQL: API rate limit already exceeded")

    monkeypatch.setattr(gate, "_json_command", fail_graphql)
    monkeypatch.setattr(
        gate,
        "_issue_state_rest",
        lambda **_: {
            "state": "OPEN",
            "updatedAt": "2026-08-13T07:00:00Z",
            "closedAt": None,
        },
    )
    monkeypatch.setattr(
        gate,
        "_closing_prs_rest",
        lambda **_: [
            {
                "number": 7033,
                "title": "fix: REST fallback (#7033)",
                "merged_at": "2026-08-13T07:01:00Z",
                "merge_commit": {"oid": "merge-rest"},
                "head_ref": "fix/rest",
                "head_sha": "head-rest",
                "base_ref": "main",
            }
        ],
    )
    monkeypatch.setattr(
        gate,
        "_open_covering_prs_rest",
        lambda **_: [
            {
                "number": 7034,
                "title": "fix: open REST fallback",
                "created_at": "2026-08-13T07:02:00Z",
            }
        ],
    )
    monkeypatch.setattr(gate, "_fetch_refs", lambda **_: ("base-a", "branch-a"))
    monkeypatch.setattr(gate, "_git_output", lambda *_: "head-a")
    monkeypatch.setattr(gate, "_tree_state", lambda: "clean")

    result = gate.collect_live_state(
        repo="ll7/robot_sf_ll7",
        issue=7033,
        branch="feature/rest-fallback",
    )

    assert result["issue_state"] == "OPEN"
    assert result["closing_prs"][0]["number"] == 7033
    assert result["open_covering_prs"][0]["number"] == 7034
    assert result["remote_state_sources"] == {
        "issue": "rest",
        "closing_prs": "rest",
        "open_covering_prs": "rest",
    }
    assert result["remote_state_fallbacks"] == {
        "issue": "GraphQL: API rate limit already exceeded",
        "closing_prs": "GraphQL: API rate limit already exceeded",
        "open_covering_prs": "GraphQL: API rate limit already exceeded",
    }
    assert [command[1] for command in commands] == ["issue", "pr", "pr"]


def test_collect_live_state_uses_configured_rest_page_budget(monkeypatch) -> None:
    """REST fallback discovery must use and record the caller's page budget."""
    observed: dict[str, int] = {}

    def fail_graphql(command: list[str]) -> Any:
        raise gate.GateError("GraphQL: API rate limit already exceeded")

    def closing_prs_rest(**kwargs: Any) -> list[dict[str, Any]]:
        observed["closing"] = kwargs["max_pages"]
        return []

    def open_covering_prs_rest(**kwargs: Any) -> list[dict[str, Any]]:
        observed["open"] = kwargs["max_pages"]
        return []

    monkeypatch.setattr(gate, "_json_command", fail_graphql)
    monkeypatch.setattr(
        gate,
        "_issue_state_rest",
        lambda **_: {"state": "OPEN", "updatedAt": "now", "closedAt": None},
    )
    monkeypatch.setattr(gate, "_closing_prs_rest", closing_prs_rest)
    monkeypatch.setattr(gate, "_open_covering_prs_rest", open_covering_prs_rest)
    monkeypatch.setattr(gate, "_fetch_refs", lambda **_: ("base-a", "branch-a"))
    monkeypatch.setattr(gate, "_git_output", lambda *_: "head-a")
    monkeypatch.setattr(gate, "_tree_state", lambda: "clean")

    result = gate.collect_live_state(
        repo="ll7/robot_sf_ll7",
        issue=7033,
        branch="feature/rest-fallback",
        max_pr_pages=40,
    )

    assert observed == {"closing": 40, "open": 40}
    assert result["rest_pr_page_budget"] == 40


def test_collect_live_state_does_not_mask_graphql_auth_failure(monkeypatch) -> None:
    """GraphQL authentication failures must remain fail-closed instead of using REST."""
    monkeypatch.setattr(
        gate,
        "_json_command",
        lambda command: (_ for _ in ()).throw(
            gate.GateError("GraphQL: Resource not accessible by integration")
        ),
    )
    monkeypatch.setattr(
        gate,
        "_issue_state_rest",
        lambda **_: (_ for _ in ()).throw(AssertionError("REST must not mask auth failure")),
    )

    with pytest.raises(gate.GateError, match="Resource not accessible"):
        gate.collect_live_state(
            repo="ll7/robot_sf_ll7",
            issue=7033,
            branch="feature/rest-fallback",
        )


def test_closing_prs_rest_normalizes_and_filters_rows(monkeypatch) -> None:
    """REST closing-PR discovery keeps only merged PRs that close this repository issue."""
    monkeypatch.setattr(
        gate,
        "fetch_closed_pr_rows",
        lambda **_: (
            [
                {
                    "number": 7033,
                    "title": "fix: REST fallback",
                    "body": "Closes https://github.com/ll7/robot_sf_ll7/issues/7033",
                    "merged_at": "2026-08-13T07:01:00Z",
                    "merge_commit_sha": "merge-rest",
                    "head": {"ref": "fix/rest", "sha": "head-rest"},
                    "base": {"ref": "main"},
                },
                {
                    "number": 7034,
                    "title": "fix unrelated issue",
                    "body": "Closes other/repo#7033",
                    "merged_at": "2026-08-13T07:02:00Z",
                },
                {"number": 7035, "title": "closed without merge", "merged_at": None},
            ],
            SimpleNamespace(truncated=False, row_count=3, pages_read=1, page_budget=20),
        ),
    )

    assert gate._closing_prs_rest(repo="ll7/robot_sf_ll7", issue=7033) == [
        {
            "number": 7033,
            "title": "fix: REST fallback",
            "merged_at": "2026-08-13T07:01:00Z",
            "merge_commit": {"oid": "merge-rest"},
            "head_ref": "fix/rest",
            "head_sha": "head-rest",
            "base_ref": "main",
        }
    ]


def test_closing_prs_rest_fails_closed_when_inventory_is_truncated(monkeypatch) -> None:
    """A bounded REST PR inventory cannot authorize publication when it is partial."""
    monkeypatch.setattr(
        gate,
        "fetch_closed_pr_rows",
        lambda **_: (
            [],
            SimpleNamespace(truncated=True, row_count=2000, pages_read=20, page_budget=20),
        ),
    )

    with pytest.raises(gate.GateError, match="inventory is truncated"):
        gate._closing_prs_rest(repo="ll7/robot_sf_ll7", issue=7033)


def test_open_covering_prs_rest_normalizes_and_filters_rows(monkeypatch) -> None:
    """REST open-PR discovery keeps only same-repository explicit references."""
    monkeypatch.setattr(
        gate,
        "fetch_open_pr_rows",
        lambda **_: (
            [
                {
                    "number": 7033,
                    "title": "fix: open REST fallback",
                    "body": "Closes https://github.com/ll7/robot_sf_ll7/issues/7033",
                    "created_at": "2026-08-13T07:01:00Z",
                    "updated_at": "2026-08-13T07:02:00Z",
                    "draft": True,
                    "head": {"ref": "fix/open-rest", "sha": "head-open-rest"},
                    "base": {"ref": "main"},
                },
                {
                    "number": 7034,
                    "title": "fix unrelated issue",
                    "body": "Closes other/repo#7033",
                    "created_at": "2026-08-13T07:03:00Z",
                },
            ],
            SimpleNamespace(truncated=False, row_count=2, pages_read=1, page_budget=20),
        ),
    )

    assert gate._open_covering_prs_rest(repo="ll7/robot_sf_ll7", issue=7033) == [
        {
            "number": 7033,
            "title": "fix: open REST fallback",
            "created_at": "2026-08-13T07:01:00Z",
            "updated_at": "2026-08-13T07:02:00Z",
            "is_draft": True,
            "head_ref": "fix/open-rest",
            "head_sha": "head-open-rest",
            "base_ref": "main",
        }
    ]


def test_open_covering_prs_rest_fails_closed_when_inventory_is_truncated(monkeypatch) -> None:
    """A partial open-PR inventory cannot authorize publication."""
    monkeypatch.setattr(
        gate,
        "fetch_open_pr_rows",
        lambda **_: (
            [],
            SimpleNamespace(truncated=True, row_count=2000, pages_read=20, page_budget=20),
        ),
    )

    with pytest.raises(gate.GateError, match="open-PR inventory is truncated"):
        gate._open_covering_prs_rest(repo="ll7/robot_sf_ll7", issue=7033)


def test_non_fast_forward_integration_fails_closed_without_reset(monkeypatch) -> None:
    """A merge conflict returns refresh-required and never resets the worktree."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["git", "merge", "--no-edit"]:
            return subprocess.CompletedProcess(command, 1, "", "CONFLICT (content)")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate, "_run", fake_run)
    monkeypatch.setattr(gate, "_tree_state", lambda: "clean")
    monkeypatch.setattr(gate, "_git_branch", lambda: "feature/fresh-state")
    monkeypatch.setattr(gate, "_git_head_sha", lambda: "head-a")
    monkeypatch.setattr(gate, "_git_sha", lambda ref: "head-remote")
    monkeypatch.setattr(gate, "_fetch_remote_branch", lambda **_: None)

    result = gate._integrate_targets(
        remote="origin",
        branch="feature/fresh-state",
        expected_local_head_sha="head-a",
        targets=[
            {
                "ref": "refs/remotes/origin/feature/fresh-state",
                "sha": "head-remote",
                "kind": "remote_branch",
            }
        ],
    )

    assert result["ok"] is False
    assert result["reason"] == "integration_conflict"
    assert result["merge_aborted"] is True
    assert ["git", "merge", "--abort"] in commands
    assert all(command[:2] not in (["git", "reset"], ["git", "clean"]) for command in commands)
    assert not any(
        command[:2] == ["git", "branch"]
        and any(argument in {"-d", "-D", "--delete"} for argument in command[2:])
        for command in commands
    )
    assert not any(
        command[:2] == ["git", "push"]
        and ("--delete" in command or any(argument.startswith(":") for argument in command[2:]))
        for command in commands
    )


def test_parser_exposes_capture_check_and_sync() -> None:
    """All three lifecycle actions remain discoverable from the CLI parser."""
    parser = gate._parser()

    capture = parser.parse_args(
        ["capture", "--repo", "o/r", "--issue", "1", "--max-pr-pages", "40"]
    )
    check = parser.parse_args(["check", "--snapshot-path", "state.json"])
    sync = parser.parse_args(["sync", "--snapshot-path", "state.json", "--integrate"])

    assert capture.command == "capture"
    assert capture.max_pr_pages == 40
    assert check.command == "check"
    assert check.max_pr_pages is None
    assert sync.integrate


def test_capture_help_documents_repository_formats(capsys) -> None:
    """Capture help must explain both explicit and local repository arguments."""
    parser = gate._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["capture", "--help"])

    help_text = capsys.readouterr().out
    assert "OWNER/REPO" in help_text
    assert "ll7/robot_sf_ll7" in help_text
    assert "local checkout path" in help_text


def test_local_repository_argument_resolves_github_remote(tmp_path, monkeypatch) -> None:
    """A local checkout path should become the normalized GitHub repository slug."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="git@github.com:ll7/robot_sf_ll7.git\n",
            stderr="",
        )

    monkeypatch.setattr(gate, "_run", fake_run)

    assert gate._normalize_repo_argument(str(checkout), remote="origin") == "ll7/robot_sf_ll7"
    assert commands == [["git", "-C", str(checkout.resolve()), "remote", "get-url", "origin"]]


def test_capture_cli_stores_normalized_local_repository(tmp_path, monkeypatch, capsys) -> None:
    """The documented local-path capture invocation must persist a repository slug."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    snapshot_path = tmp_path / "snapshot.json"

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        assert command == [
            "git",
            "-C",
            str(checkout.resolve()),
            "remote",
            "get-url",
            "origin",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="https://github.com/ll7/robot_sf_ll7.git\n",
            stderr="",
        )

    monkeypatch.setattr(gate, "_run", fake_run)
    monkeypatch.setattr(
        gate,
        "collect_live_state",
        lambda **kwargs: _snapshot(repo=kwargs["repo"], issue=kwargs["issue"]),
    )

    assert (
        gate.main(
            [
                "capture",
                "--repo",
                str(checkout),
                "--issue",
                "7206",
                "--branch",
                "issue-7206-prepublication-repo-format",
                "--snapshot-path",
                str(snapshot_path),
            ]
        )
        == 0
    )

    capsys.readouterr()
    assert json.loads(snapshot_path.read_text(encoding="utf-8"))["repo"] == "ll7/robot_sf_ll7"


def test_repository_argument_preserves_explicit_slug() -> None:
    """An explicit repository slug should not require local Git metadata."""
    assert gate._normalize_repo_argument("ll7/robot_sf_ll7", remote="origin") == (
        "ll7/robot_sf_ll7"
    )
    assert gate._normalize_repo_argument("github.example/ll7/robot_sf_ll7", remote="origin") == (
        "github.example/ll7/robot_sf_ll7"
    )


def test_local_repository_argument_fails_before_github_reads(tmp_path, monkeypatch) -> None:
    """A checkout without the requested remote must produce an actionable local error."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="fatal: No such remote 'origin'",
        )

    monkeypatch.setattr(gate, "_run", fake_run)

    with pytest.raises(gate.GateError, match="pass an explicit OWNER/REPO"):
        gate._normalize_repo_argument(str(checkout), remote="origin")


def test_parser_rejects_non_positive_rest_page_budget() -> None:
    """The CLI must reject a page budget that cannot make progress."""
    parser = gate._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["capture", "--repo", "o/r", "--issue", "1", "--max-pr-pages", "0"])


def test_page_budget_reuses_snapshot_value_unless_overridden() -> None:
    """Checks should preserve the capture budget while allowing an explicit refresh override."""
    assert gate._effective_page_budget(None, snapshot={"rest_pr_page_budget": 40}) == 40
    assert gate._effective_page_budget(60, snapshot={"rest_pr_page_budget": 40}) == 60

    with pytest.raises(gate.GateError, match="invalid REST PR page budget"):
        gate._effective_page_budget(None, snapshot={"rest_pr_page_budget": 0})


def test_check_cli_writes_ready_decision(tmp_path, monkeypatch, capsys) -> None:
    """The check command emits and persists a publication-permitting decision."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: _snapshot())

    assert gate.main(["check", "--snapshot-path", str(snapshot_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    decision_path = tmp_path / "state.decision.json"
    assert output["decision"] == "ready"
    assert json.loads(decision_path.read_text(encoding="utf-8"))["reason"] == (
        "remote_state_unchanged"
    )


def test_check_cli_writes_superseded_decision_for_closed_issue(
    tmp_path, monkeypatch, capsys
) -> None:
    """The check command persists supersession when the issue closes remotely."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: _snapshot(issue_state="CLOSED"))

    assert gate.main(["check", "--snapshot-path", str(snapshot_path)]) == 3

    output = json.loads(capsys.readouterr().out)
    decision_path = tmp_path / "state.decision.json"
    persisted = json.loads(decision_path.read_text(encoding="utf-8"))
    assert output["decision"] == "superseded"
    assert persisted["reason"] == "issue_closed"


def test_sync_failure_nests_integration_result(tmp_path, monkeypatch, capsys) -> None:
    """Integration failures retain the original drift reason and schema shape."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: _snapshot(base_sha="base-b"))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {
            "ok": False,
            "reason": "integration_conflict",
            "detail": "CONFLICT (content)",
            "merged": [],
            "merge_aborted": True,
        },
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 2

    output = json.loads(capsys.readouterr().out)
    persisted = json.loads((tmp_path / "state.decision.json").read_text(encoding="utf-8"))
    assert output["reason"] == "base_changed"
    assert output["integration"]["reason"] == "integration_conflict"
    assert "ok" not in output
    assert persisted == output


def test_sync_unpushed_branch_success_integrates_base(tmp_path, monkeypatch, capsys) -> None:
    """An unpushed branch integrates a moved base and records ready without self-comparison."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot(remote_branch_sha=None)), encoding="utf-8")
    states = iter(
        [
            _snapshot(base_sha="base-b", remote_branch_sha=None),
            _snapshot(base_sha="base-b", remote_branch_sha=None, local_head_sha="head-merge"),
        ]
    )
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: next(states))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {
            "ok": True,
            "merged": ["refs/remotes/origin/main"],
            "merged_shas": ["base-b"],
        },
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "ready"
    assert output["reason"] == "remote_state_integrated"
    assert output["comparison"] == "expected_post_integration"
    assert output["integrated"] == ["refs/remotes/origin/main"]
    persisted_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert persisted_snapshot["base_sha"] == "base-b"
    assert persisted_snapshot["local_head_sha"] == "head-merge"


def test_sync_pushed_branch_base_merge_requires_push_and_recapture(
    tmp_path, monkeypatch, capsys
) -> None:
    """A pushed branch whose head advances during base merge requires push and fresh capture."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    states = iter(
        [
            _snapshot(base_sha="base-b", remote_branch_sha="branch-a", local_head_sha="head-a"),
            _snapshot(base_sha="base-b", remote_branch_sha="branch-a", local_head_sha="head-merge"),
        ]
    )
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: next(states))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {
            "ok": True,
            "merged": ["refs/remotes/origin/main"],
            "merged_shas": ["base-b"],
        },
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "refresh-required"
    assert output["reason"] == "remote_branch_changed"
    assert output["comparison"] == "expected_post_integration"
    assert output["drift"]["remote_branch_sha"] == {
        "baseline": "head-merge",
        "current": "branch-a",
    }


def test_sync_pushed_branch_fast_forward_remote_branch_ready(tmp_path, monkeypatch, capsys) -> None:
    """A fast-forward merge of remote branch tracking ref is ready post-integration."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    states = iter(
        [
            _snapshot(remote_branch_sha="branch-b", local_head_sha="head-a"),
            _snapshot(remote_branch_sha="branch-b", local_head_sha="branch-b"),
        ]
    )
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: next(states))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {
            "ok": True,
            "merged": ["refs/remotes/origin/feature/fresh-state"],
            "merged_shas": ["branch-b"],
        },
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "ready"
    assert output["reason"] == "remote_state_integrated"
    assert output["comparison"] == "expected_post_integration"
    persisted_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert persisted_snapshot["local_head_sha"] == "branch-b"


def test_sync_rejects_local_head_drift_non_integrable(tmp_path, monkeypatch, capsys) -> None:
    """Local commits made after baseline capture cannot be integrated or self-baselined."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot(local_head_sha="head-a")), encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "collect_live_state",
        lambda **_: _snapshot(local_head_sha="head-b", base_sha="base-b"),
    )
    integrate_called = False

    def fake_integrate(**_):
        nonlocal integrate_called
        integrate_called = True
        return {"ok": True, "merged": []}

    monkeypatch.setattr(gate, "_integrate_targets", fake_integrate)

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert not integrate_called
    assert output["decision"] == "refresh-required"
    assert output["integration"]["reason"] == "local_head_drift_non_integrable"
    persisted_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert persisted_snapshot["local_head_sha"] == "head-a"


def test_sync_rejects_when_no_integrable_targets(tmp_path, monkeypatch, capsys) -> None:
    """When decision has no base or remote branch targets, sync fails closed."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: _snapshot())
    monkeypatch.setattr(
        gate,
        "evaluate_state",
        lambda baseline, current: gate._decision(
            baseline, current, decision="refresh-required", reason="unknown", extra={"drift": {}}
        ),
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["integration"]["reason"] == "no_integrable_targets"


def test_sync_detects_base_movement_race_during_integration(tmp_path, monkeypatch, capsys) -> None:
    """Base moving during merge execution fails closed with refresh-required."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot(remote_branch_sha=None)), encoding="utf-8")
    states = iter(
        [
            _snapshot(base_sha="base-b", remote_branch_sha=None),
            _snapshot(base_sha="base-c", remote_branch_sha=None, local_head_sha="head-merge"),
        ]
    )
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: next(states))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {"ok": True, "merged": ["refs/remotes/origin/main"], "merged_shas": ["base-b"]},
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "refresh-required"
    assert output["reason"] == "base_changed"
    assert output["drift"]["base_sha"] == {"baseline": "base-b", "current": "base-c"}


def test_sync_detects_remote_branch_movement_race_during_integration(
    tmp_path, monkeypatch, capsys
) -> None:
    """Remote branch moving during merge execution fails closed with refresh-required."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    states = iter(
        [
            _snapshot(remote_branch_sha="branch-b"),
            _snapshot(remote_branch_sha="branch-c", local_head_sha="branch-b"),
        ]
    )
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: next(states))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {
            "ok": True,
            "merged": ["refs/remotes/origin/feature/fresh-state"],
            "merged_shas": ["branch-b"],
        },
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "refresh-required"
    assert output["reason"] == "remote_branch_changed"
    assert output["drift"]["remote_branch_sha"] == {"baseline": "branch-b", "current": "branch-c"}


def test_sync_post_integration_fails_closed_on_dirty_tree(tmp_path, monkeypatch, capsys) -> None:
    """A dirty worktree after integration fails closed with blocked."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot(remote_branch_sha=None)), encoding="utf-8")
    states = iter(
        [
            _snapshot(base_sha="base-b", remote_branch_sha=None),
            _snapshot(
                base_sha="base-b",
                remote_branch_sha=None,
                local_head_sha="head-merge",
                tree_state="dirty",
            ),
        ]
    )
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: next(states))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {"ok": True, "merged": ["refs/remotes/origin/main"], "merged_shas": ["base-b"]},
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 4

    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "blocked"
    assert output["reason"] == "dirty_worktree"


def test_sync_post_integration_fails_closed_on_blocking_ancestry(
    tmp_path, monkeypatch, capsys
) -> None:
    """Blocking ancestry after integration fails closed with blocked."""
    snapshot_path = tmp_path / "state.json"
    snapshot_path.write_text(json.dumps(_snapshot(remote_branch_sha=None)), encoding="utf-8")
    states = iter(
        [
            _snapshot(base_sha="base-b", remote_branch_sha=None),
            _snapshot(
                base_sha="base-b",
                remote_branch_sha=None,
                local_head_sha="head-merge",
                ancestry={"state": "undeclared_stack"},
            ),
        ]
    )
    monkeypatch.setattr(gate, "collect_live_state", lambda **_: next(states))
    monkeypatch.setattr(
        gate,
        "_integrate_targets",
        lambda **_: {"ok": True, "merged": ["refs/remotes/origin/main"], "merged_shas": ["base-b"]},
    )

    assert gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"]) == 4

    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "blocked"
    assert output["reason"] == "undeclared_stack_ancestry"


def test_integrate_targets_validations(monkeypatch) -> None:
    """Unit tests verifying all pre-merge validation gates in _integrate_targets."""
    # 1. Dirty worktree
    monkeypatch.setattr(gate, "_tree_state", lambda: "dirty")
    res = gate._integrate_targets(
        remote="origin", branch="feat", expected_local_head_sha="h1", targets=[]
    )
    assert res == {"ok": False, "reason": "dirty_worktree"}

    # 2. Branch mismatch
    monkeypatch.setattr(gate, "_tree_state", lambda: "clean")
    monkeypatch.setattr(gate, "_git_branch", lambda: "other-branch")
    res = gate._integrate_targets(
        remote="origin", branch="feat", expected_local_head_sha="h1", targets=[]
    )
    assert res["reason"] == "branch_mismatch"

    # 3. Local head drift
    monkeypatch.setattr(gate, "_git_branch", lambda: "feat")
    monkeypatch.setattr(gate, "_git_head_sha", lambda: "h2")
    res = gate._integrate_targets(
        remote="origin", branch="feat", expected_local_head_sha="h1", targets=[]
    )
    assert res["reason"] == "local_head_drift"

    # 4. Target ref unresolvable
    monkeypatch.setattr(gate, "_git_head_sha", lambda: "h1")

    def fail_git_sha(ref):
        raise gate.GateError("ref not found")

    monkeypatch.setattr(gate, "_git_sha", fail_git_sha)
    res = gate._integrate_targets(
        remote="origin",
        branch="feat",
        expected_local_head_sha="h1",
        targets=[{"ref": "refs/remotes/origin/main", "sha": "b1"}],
    )
    assert res["reason"] == "target_ref_unresolvable"

    # 5. Target ref moved (SHA mismatch)
    monkeypatch.setattr(gate, "_git_sha", lambda ref: "b2")
    res = gate._integrate_targets(
        remote="origin",
        branch="feat",
        expected_local_head_sha="h1",
        targets=[{"ref": "refs/remotes/origin/main", "sha": "b1"}],
    )
    assert res["reason"] == "target_ref_moved"
    assert res["expected_sha"] == "b1"
    assert res["actual_sha"] == "b2"


# ---------------------------------------------------------------------------
# Issue #7515: undeclared/mismatched stack ancestry blocks pre-PR publication
# ---------------------------------------------------------------------------


def test_undeclared_stack_ancestry_blocks_pre_publication() -> None:
    """An undeclared non-main ancestry must block publication before PR creation."""
    result = gate.evaluate_state(_snapshot(), _snapshot(ancestry={"state": "undeclared_stack"}))

    assert result["decision"] == "blocked"
    assert result["reason"] == "undeclared_stack_ancestry"
    assert result["ancestry"]["state"] == "undeclared_stack"


def test_mismatched_declaration_blocks_pre_publication() -> None:
    """A mismatched stack declaration must fail closed before PR creation."""
    result = gate.evaluate_state(
        _snapshot(), _snapshot(ancestry={"state": "mismatched_declaration"})
    )

    assert result["decision"] == "blocked"
    assert result["reason"] == "undeclared_stack_ancestry"


def test_invalidated_parent_blocks_pre_publication() -> None:
    """A closed-unmerged/rewritten parent must fail closed before PR creation."""
    result = gate.evaluate_state(_snapshot(), _snapshot(ancestry={"state": "parent_invalidated"}))

    assert result["decision"] == "blocked"
    assert result["reason"] == "undeclared_stack_ancestry"


def test_declared_stack_does_not_block_publication() -> None:
    """A declared stack may be published, though it is never independently mergeable."""
    result = gate.evaluate_state(_snapshot(), _snapshot(ancestry={"state": "stacked"}))

    assert result["decision"] == "ready"
    assert result["reason"] == "remote_state_unchanged"


def test_clean_ancestry_does_not_block_publication() -> None:
    """A clean ancestry block leaves publication-ready evidence intact."""
    result = gate.evaluate_state(_snapshot(), _snapshot(ancestry={"state": "clean"}))

    assert result["decision"] == "ready"
    assert result["reason"] == "remote_state_unchanged"


def test_ancestry_block_without_state_is_not_a_blocker() -> None:
    """A malformed/empty ancestry block must not invent a blocking reason."""
    result = gate.evaluate_state(_snapshot(), _snapshot(ancestry={"error": "boom"}))

    assert result["decision"] == "ready"


def test_capture_parser_exposes_declaration_text() -> None:
    """The capture CLI accepts the canonical stack declaration text."""
    parser = gate._parser()

    captured = parser.parse_args(
        [
            "capture",
            "--repo",
            "o/r",
            "--issue",
            "1",
            "--declaration-text",
            "## Stack Declaration\nparent_pr: #2\nparent_head: " + "a" * 40,
        ]
    )

    assert captured.declaration_text is not None
    assert "parent_pr" in captured.declaration_text


def test_collect_live_state_records_ancestry_block(monkeypatch) -> None:
    """Live collection records the ancestry classification into the snapshot."""
    from scripts.dev.stack_ancestry import ancestry_state

    monkeypatch.setattr(
        gate,
        "collect_ancestry_facts",
        lambda **_: (
            {
                "main_tip_sha": "b" * 40,
                "merge_base_sha": "a" * 40,
                "commits": ["foreign work (#9999)", "intended work"],
                "changed_paths": ["foreign.py", "own.py"],
            },
            None,
        ),
    )
    monkeypatch.setattr(gate, "_git_output", lambda *_: "c" * 40)
    monkeypatch.setattr(gate, "_tree_state", lambda: "clean")
    monkeypatch.setattr(gate, "_fetch_refs", lambda **_: ("base-a", "branch-a"))
    monkeypatch.setattr(
        gate,
        "_json_command",
        lambda command: {"state": "OPEN", "updatedAt": "now", "closedAt": None},
    )
    monkeypatch.setattr(gate, "_closing_prs", lambda **_: [])
    monkeypatch.setattr(gate, "_open_covering_prs", lambda **_: [])

    snapshot = gate.collect_live_state(
        repo="ll7/robot_sf_ll7",
        issue=6916,
        branch="feature/fresh-state",
        declaration_text="## Stack Declaration\nparent_pr: #9999\nparent_head: " + "a" * 40,
    )

    assert snapshot["ancestry"]["state"] == "stacked"
    assert snapshot["ancestry"]["declared_parent"] == 9999
    assert snapshot["ancestry"]["unexpected_paths"] == ["foreign.py", "own.py"]
    assert ancestry_state is not None


def test_covering_issue_numbers_matches_ownership_formats_only() -> None:
    """_covering_issue_numbers recognizes ownership, title, and branch signals."""
    repo = "ll7/robot_sf_ll7"

    # Strong ownership verbs in the body
    assert 7448 in gate._covering_issue_numbers(body="Fixes #7448", repo=repo)
    assert 7448 in gate._covering_issue_numbers(body="Implements #7448", repo=repo)
    assert 7448 in gate._covering_issue_numbers(body="Addresses #7448", repo=repo)
    assert 7448 in gate._covering_issue_numbers(
        body="Closes https://github.com/ll7/robot_sf_ll7/issues/7448", repo=repo
    )

    # Context-only references must not block a dependent or stacked PR.
    assert 7448 not in gate._covering_issue_numbers(body="Refs #7448.", repo=repo)
    assert 7448 not in gate._covering_issue_numbers(body="Part of #7448", repo=repo)

    # Title conventions
    assert 7448 in gate._covering_issue_numbers(
        title="fix(dev): fail closed on non-live carriers (#7448)", repo=repo
    )
    assert 7448 in gate._covering_issue_numbers(
        title="fix(dev): fail closed on non-live carriers (issue-7448)", repo=repo
    )
    assert 3 not in gate._covering_issue_numbers(title="fix: parser behavior (3)", repo=repo)
    assert 7448 in gate._covering_issue_numbers(title="fix issue-7448 bug", repo=repo)

    # Branch conventions
    assert 7448 in gate._covering_issue_numbers(
        head_ref="fix/issue-7448-pr-head-truth-20260818", repo=repo
    )
    assert 7448 in gate._covering_issue_numbers(head_ref="repair/issue-7448", repo=repo)

    # Foreign repo exclusion
    assert 7448 not in gate._covering_issue_numbers(
        body="Closes otherorg/otherrepo#7448", repo=repo
    )


def test_fetch_claim_ref_detects_existing_and_missing(monkeypatch) -> None:
    """_fetch_claim_ref checks remote ref presence."""

    def fake_run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        if "agent-claims/issue-7474" in command[-1]:
            return subprocess.CompletedProcess(
                command, 0, "sha-7474\trefs/heads/agent-claims/issue-7474\n", ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate, "_run", fake_run)

    found = gate._fetch_claim_ref(remote="origin", issue=7474)
    assert found["exists"] is True
    assert found["sha"] == "sha-7474"

    missing = gate._fetch_claim_ref(remote="origin", issue=9999)
    assert missing["exists"] is False
    assert missing["sha"] is None


def test_fetch_claim_ref_lookup_failure_fails_closed(monkeypatch) -> None:
    """A remote/authentication failure must not look like an absent claim ref."""

    def fake_run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 128, "", "fatal: authentication failed")

    monkeypatch.setattr(gate, "_run", fake_run)

    with pytest.raises(gate.GateError, match="authentication failed"):
        gate._fetch_claim_ref(remote="origin", issue=7474)


def test_capture_cli_superseded_when_competing_open_pr_exists(
    tmp_path, monkeypatch, capsys
) -> None:
    """Capture must fail closed with superseded when an open PR on another branch exists."""
    snapshot_path = tmp_path / "snapshot.json"
    competing_pr = {
        "number": 7462,
        "title": "fix(workflow): reject fabricated PR head provenance",
        "head_ref": "fix/issue-7448-pr-head-truth-20260818",
        "head_sha": "head-competing",
        "base_ref": "main",
    }
    monkeypatch.setattr(
        gate,
        "collect_live_state",
        lambda **kwargs: _snapshot(
            repo=kwargs["repo"],
            issue=kwargs["issue"],
            branch="fix/issue-7448-gate-verdict-sha-20260818",
            open_covering_prs=[competing_pr],
        ),
    )

    exit_code = gate.main(
        [
            "capture",
            "--repo",
            "ll7/robot_sf_ll7",
            "--issue",
            "7448",
            "--branch",
            "fix/issue-7448-gate-verdict-sha-20260818",
            "--snapshot-path",
            str(snapshot_path),
        ]
    )

    assert exit_code == 3
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "superseded"
    assert output["reason"] == "open_pr_covers_issue"
    assert output["open_covering_prs"][0]["number"] == 7462


def test_capture_cli_ready_when_open_pr_is_same_branch(tmp_path, monkeypatch, capsys) -> None:
    """Capture permits re-checking and updating an existing PR on the same branch."""
    snapshot_path = tmp_path / "snapshot.json"
    own_pr = {
        "number": 7471,
        "title": "fix(dev): fail closed on non-live exact-head SHA carriers (#7448)",
        "head_ref": "fix/issue-7448-gate-verdict-sha-20260818",
        "head_sha": "head-own",
        "base_ref": "main",
    }
    monkeypatch.setattr(
        gate,
        "collect_live_state",
        lambda **kwargs: _snapshot(
            repo=kwargs["repo"],
            issue=kwargs["issue"],
            branch="fix/issue-7448-gate-verdict-sha-20260818",
            open_covering_prs=[own_pr],
        ),
    )

    exit_code = gate.main(
        [
            "capture",
            "--repo",
            "ll7/robot_sf_ll7",
            "--issue",
            "7448",
            "--branch",
            "fix/issue-7448-gate-verdict-sha-20260818",
            "--snapshot-path",
            str(snapshot_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "ready"
    assert output["reason"] == "baseline_captured"


def test_evaluate_state_superseded_on_competing_open_pr() -> None:
    """evaluate_state fails closed when a competing open PR exists on another branch."""
    baseline = _snapshot(branch="fix/my-branch")
    competing_pr = {
        "number": 7462,
        "title": "fix: competing PR",
        "head_ref": "fix/competing-branch",
        "head_sha": "head-competing",
        "base_ref": "main",
    }
    current = _snapshot(branch="fix/my-branch", open_covering_prs=[competing_pr])

    result = gate.evaluate_state(baseline, current)

    assert result["decision"] == "superseded"
    assert result["reason"] == "open_pr_covers_issue"
    assert result["open_covering_prs"][0]["number"] == 7462


def test_evaluate_state_ready_when_open_pr_is_same_branch() -> None:
    """evaluate_state allows same-branch PR updates without supersession."""
    baseline = _snapshot(branch="fix/my-branch")
    own_pr = {
        "number": 7471,
        "title": "fix: own PR",
        "head_ref": "fix/my-branch",
        "head_sha": "head-own",
        "base_ref": "main",
    }
    current = _snapshot(branch="fix/my-branch", open_covering_prs=[own_pr])

    result = gate.evaluate_state(baseline, current)

    assert result["decision"] == "ready"
    assert result["reason"] == "remote_state_unchanged"


# ---------------------------------------------------------------------------
# Real temporary Git repository end-to-end integration tests (issue #7829)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_git_repo(tmp_path: Path, monkeypatch):
    """Set up a real bare remote and a cloned local worker repository."""
    remote_dir = tmp_path / "remote.git"
    remote_dir.mkdir()
    subprocess.run(["git", "init", "--bare", "-b", "main"], cwd=remote_dir, check=True)

    worker_dir = tmp_path / "worker"
    subprocess.run(["git", "clone", str(remote_dir), str(worker_dir)], check=True)
    subprocess.run(["git", "config", "user.name", "Test Committer"], cwd=worker_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=worker_dir, check=True)

    (worker_dir / ".gitignore").write_text("*.json\noutput/\n", encoding="utf-8")
    (worker_dir / "base.txt").write_text("initial base content\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "base.txt"], cwd=worker_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial main commit"], cwd=worker_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=worker_dir, check=True)

    initial_main_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worker_dir, capture_output=True, text=True, check=True
    ).stdout.strip()

    # Monkeypatch GitHub CLI commands to return deterministic OPEN issue state with no PRs
    def fake_json_command(command: list[str]) -> Any:
        if "issue" in command and "view" in command:
            return {
                "state": "OPEN",
                "updatedAt": "2026-08-25T10:00:00Z",
                "closedAt": None,
            }
        if "pr" in command and "list" in command:
            return []
        return {}

    monkeypatch.setattr(gate, "_json_command", fake_json_command)
    monkeypatch.setattr(gate, "_closing_prs", lambda **_: [])
    monkeypatch.setattr(gate, "_open_covering_prs", lambda **_: [])
    monkeypatch.setattr(gate, "_fetch_claim_ref", lambda **_: {"claimed": False, "sha": None})

    return SimpleNamespace(remote=remote_dir, worker=worker_dir, initial_main_sha=initial_main_sha)


def test_real_git_unpushed_branch_sync_integrates_main_and_is_ready(
    real_git_repo, monkeypatch, capsys
) -> None:
    """An unpushed branch merges updated remote main and becomes ready with updated snapshot."""
    worker = real_git_repo.worker
    monkeypatch.chdir(worker)

    # Create feature branch with a local commit
    subprocess.run(["git", "checkout", "-b", "feature/unpushed-work"], cwd=worker, check=True)
    (worker / "feature.txt").write_text("feature content\n", encoding="utf-8")
    subprocess.run(["git", "add", "feature.txt"], cwd=worker, check=True)
    subprocess.run(["git", "commit", "-m", "feature commit"], cwd=worker, check=True)

    decl = f"## Stack Declaration\nparent_pr: 100\nparent_head: {real_git_repo.initial_main_sha}\n"
    snapshot_path = worker / "state.json"
    exit_code = gate.main(
        [
            "capture",
            "--repo",
            "ll7/robot_sf_ll7",
            "--issue",
            "7829",
            "--branch",
            "feature/unpushed-work",
            "--declaration-text",
            decl,
            "--snapshot-path",
            str(snapshot_path),
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    # Advance remote main by creating a commit in a temporary clone
    other_dir = worker.parent / "other"
    subprocess.run(["git", "clone", str(real_git_repo.remote), str(other_dir)], check=True)
    subprocess.run(["git", "config", "user.name", "Main Author"], cwd=other_dir, check=True)
    subprocess.run(["git", "config", "user.email", "main@example.com"], cwd=other_dir, check=True)
    (other_dir / "main_update.txt").write_text("new main work\n", encoding="utf-8")
    subprocess.run(["git", "add", "main_update.txt"], cwd=other_dir, check=True)
    subprocess.run(["git", "commit", "-m", "update main"], cwd=other_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=other_dir, check=True)

    # Check returns 2 (refresh-required)
    check_exit = gate.main(["check", "--snapshot-path", str(snapshot_path)])
    assert check_exit == 2
    capsys.readouterr()

    # Sync with --integrate performs clean merge of main and records ready
    sync_exit = gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"])
    assert sync_exit == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "ready"
    assert output["reason"] == "remote_state_integrated"
    assert output["comparison"] == "expected_post_integration"

    # Verify both feature.txt and main_update.txt exist in the working directory
    assert (worker / "feature.txt").exists()
    assert (worker / "main_update.txt").exists()


def test_real_git_pushed_branch_sync_advances_head_requires_push(
    real_git_repo, monkeypatch, capsys
) -> None:
    """A pushed branch whose head advances during base merge is not ready until pushed."""
    worker = real_git_repo.worker
    monkeypatch.chdir(worker)

    # Create feature branch and push to origin
    subprocess.run(["git", "checkout", "-b", "feature/pushed-work"], cwd=worker, check=True)
    (worker / "feature2.txt").write_text("feature 2 content\n", encoding="utf-8")
    subprocess.run(["git", "add", "feature2.txt"], cwd=worker, check=True)
    subprocess.run(["git", "commit", "-m", "feature 2 commit"], cwd=worker, check=True)
    subprocess.run(["git", "push", "-u", "origin", "feature/pushed-work"], cwd=worker, check=True)

    decl = f"## Stack Declaration\nparent_pr: 100\nparent_head: {real_git_repo.initial_main_sha}\n"
    snapshot_path = worker / "state.json"
    exit_code = gate.main(
        [
            "capture",
            "--repo",
            "ll7/robot_sf_ll7",
            "--issue",
            "7829",
            "--branch",
            "feature/pushed-work",
            "--declaration-text",
            decl,
            "--snapshot-path",
            str(snapshot_path),
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    # Advance remote main
    other_dir = worker.parent / "other2"
    subprocess.run(["git", "clone", str(real_git_repo.remote), str(other_dir)], check=True)
    subprocess.run(["git", "config", "user.name", "Main Author"], cwd=other_dir, check=True)
    subprocess.run(["git", "config", "user.email", "main@example.com"], cwd=other_dir, check=True)
    (other_dir / "main_update2.txt").write_text("main update 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "main_update2.txt"], cwd=other_dir, check=True)
    subprocess.run(["git", "commit", "-m", "update main 2"], cwd=other_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=other_dir, check=True)

    # Sync with --integrate merges main locally, but since remote_branch is out of sync, returns 2
    sync_exit = gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"])
    assert sync_exit == 2
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "refresh-required"
    assert output["reason"] == "remote_branch_changed"
    assert output["comparison"] == "expected_post_integration"

    # Push integrated head to origin and capture fresh snapshot -> check is ready!
    subprocess.run(["git", "push", "origin", "feature/pushed-work"], cwd=worker, check=True)
    recapture_exit = gate.main(
        [
            "capture",
            "--repo",
            "ll7/robot_sf_ll7",
            "--issue",
            "7829",
            "--branch",
            "feature/pushed-work",
            "--declaration-text",
            decl,
            "--snapshot-path",
            str(snapshot_path),
        ]
    )
    assert recapture_exit == 0
    capsys.readouterr()

    check_exit = gate.main(["check", "--snapshot-path", str(snapshot_path)])
    assert check_exit == 0


def test_real_git_local_commit_after_capture_fails_closed(
    real_git_repo, monkeypatch, capsys
) -> None:
    """A local commit made after capture fails closed and is rejected by sync --integrate."""
    worker = real_git_repo.worker
    monkeypatch.chdir(worker)

    subprocess.run(["git", "checkout", "-b", "feature/drifted-head"], cwd=worker, check=True)
    (worker / "f3.txt").write_text("f3 initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "f3.txt"], cwd=worker, check=True)
    subprocess.run(["git", "commit", "-m", "f3 initial commit"], cwd=worker, check=True)

    decl = f"## Stack Declaration\nparent_pr: 100\nparent_head: {real_git_repo.initial_main_sha}\n"
    snapshot_path = worker / "state.json"
    gate.main(
        [
            "capture",
            "--repo",
            "ll7/robot_sf_ll7",
            "--issue",
            "7829",
            "--branch",
            "feature/drifted-head",
            "--declaration-text",
            decl,
            "--snapshot-path",
            str(snapshot_path),
        ]
    )
    capsys.readouterr()

    # Local commit after capture
    (worker / "local_only.txt").write_text("local post capture\n", encoding="utf-8")
    subprocess.run(["git", "add", "local_only.txt"], cwd=worker, check=True)
    subprocess.run(["git", "commit", "-m", "post capture commit"], cwd=worker, check=True)

    # Advance remote main
    other_dir = worker.parent / "other3"
    subprocess.run(["git", "clone", str(real_git_repo.remote), str(other_dir)], check=True)
    (other_dir / "m3.txt").write_text("m3\n", encoding="utf-8")
    subprocess.run(["git", "add", "m3.txt"], cwd=other_dir, check=True)
    subprocess.run(["git", "commit", "-m", "m3 commit"], cwd=other_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=other_dir, check=True)

    # Sync --integrate refuses to merge into drifted local head
    sync_exit = gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"])
    assert sync_exit == 2
    output = json.loads(capsys.readouterr().out)
    assert output["integration"]["reason"] == "local_head_drift_non_integrable"


def test_real_git_merge_conflict_aborts_cleanly(real_git_repo, monkeypatch, capsys) -> None:
    """A merge conflict during sync --integrate aborts cleanly, leaving a clean tree."""
    worker = real_git_repo.worker
    monkeypatch.chdir(worker)

    subprocess.run(["git", "checkout", "-b", "feature/conflict-test"], cwd=worker, check=True)
    (worker / "base.txt").write_text("feature branch conflicting edit\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=worker, check=True)
    subprocess.run(["git", "commit", "-m", "feature conflicting edit"], cwd=worker, check=True)

    decl = f"## Stack Declaration\nparent_pr: 100\nparent_head: {real_git_repo.initial_main_sha}\n"
    snapshot_path = worker / "state.json"
    gate.main(
        [
            "capture",
            "--repo",
            "ll7/robot_sf_ll7",
            "--issue",
            "7829",
            "--branch",
            "feature/conflict-test",
            "--declaration-text",
            decl,
            "--snapshot-path",
            str(snapshot_path),
        ]
    )
    capsys.readouterr()

    # Create conflict on remote main
    other_dir = worker.parent / "other4"
    subprocess.run(["git", "clone", str(real_git_repo.remote), str(other_dir)], check=True)
    (other_dir / "base.txt").write_text("main branch conflicting edit\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=other_dir, check=True)
    subprocess.run(["git", "commit", "-m", "main conflicting edit"], cwd=other_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=other_dir, check=True)

    # Sync --integrate fails with conflict and aborts cleanly
    sync_exit = gate.main(["sync", "--snapshot-path", str(snapshot_path), "--integrate"])
    assert sync_exit == 2
    output = json.loads(capsys.readouterr().out)
    assert output["integration"]["reason"] == "integration_conflict"
    assert output["integration"]["merge_aborted"] is True

    # Verify worktree is clean
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=worker, capture_output=True, text=True, check=True
    )
    assert status.stdout.strip() == ""
