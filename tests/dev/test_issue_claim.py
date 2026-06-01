"""Tests for the cross-machine issue claim helper."""

from __future__ import annotations

import argparse

import pytest

from scripts.dev import issue_claim


def test_claim_ref_is_stable_per_issue() -> None:
    """Claim refs should be predictable so every PC contends for the same ref."""
    assert issue_claim.claim_ref(123) == "refs/heads/agent-claims/issue-123"
    assert issue_claim.short_claim_ref(123) == "agent-claims/issue-123"


def test_validate_issue_number_rejects_non_positive_values() -> None:
    """The CLI should reject invalid issue identifiers before building git refs."""
    with pytest.raises(argparse.ArgumentTypeError):
        issue_claim.validate_issue_number("0")
    with pytest.raises(argparse.ArgumentTypeError):
        issue_claim.validate_issue_number("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        issue_claim.validate_issue_number("abc")


def test_build_resolve_source_command_uses_requested_ref() -> None:
    """Acquire should resolve the requested source ref before creating a GitHub ref."""
    command = issue_claim.build_resolve_source_command(source_ref="origin/main")

    assert command == ["git", "rev-parse", "--verify", "origin/main^{commit}"]


def test_build_acquire_command_uses_github_create_ref_api() -> None:
    """Acquire should use GitHub create-ref so existing claims fail instead of fast-forwarding."""
    command = issue_claim.build_acquire_command(123, repo="ll7/robot_sf_ll7", sha="abc123")

    assert command == [
        "gh",
        "api",
        "-X",
        "POST",
        "repos/ll7/robot_sf_ll7/git/refs",
        "-f",
        "ref=refs/heads/agent-claims/issue-123",
        "-f",
        "sha=abc123",
    ]


def test_status_payload_for_unclaimed_issue() -> None:
    """Empty ls-remote output means no current issue claim."""
    result = issue_claim.CommandResult(
        command=("git", "ls-remote"),
        returncode=0,
        stdout="",
        stderr="",
    )

    payload = issue_claim._status_from_ls_remote(result, issue_number=123, remote="origin")

    assert payload["ok"] is True
    assert payload["claimed"] is False
    assert payload["claim_ref"] == "agent-claims/issue-123"
    assert payload["sha"] is None


def test_status_payload_for_claimed_issue() -> None:
    """A matching remote ref should produce a claimed status with its SHA."""
    result = issue_claim.CommandResult(
        command=("git", "ls-remote"),
        returncode=0,
        stdout="abc123\trefs/heads/agent-claims/issue-123\n",
        stderr="",
    )

    payload = issue_claim._status_from_ls_remote(result, issue_number=123, remote="origin")

    assert payload["ok"] is True
    assert payload["claimed"] is True
    assert payload["sha"] == "abc123"


def test_status_payload_ignores_non_exact_ls_remote_matches() -> None:
    """ls-remote may return suffix matches, so status should require the exact claim ref."""
    result = issue_claim.CommandResult(
        command=("git", "ls-remote"),
        returncode=0,
        stdout=(
            "abc123\trefs/heads/archive/agent-claims/issue-123\n"
            "def456\trefs/heads/agent-claims/issue-123-extra\n"
        ),
        stderr="",
    )

    payload = issue_claim._status_from_ls_remote(result, issue_number=123, remote="origin")

    assert payload["ok"] is True
    assert payload["claimed"] is False
    assert payload["sha"] is None


def test_status_payload_uses_exact_match_from_multiline_ls_remote() -> None:
    """When multiple refs are returned, the exact claim ref should determine the status."""
    result = issue_claim.CommandResult(
        command=("git", "ls-remote"),
        returncode=0,
        stdout=(
            "abc123\trefs/heads/archive/agent-claims/issue-123\n"
            "def456\trefs/heads/agent-claims/issue-123\n"
        ),
        stderr="",
    )

    payload = issue_claim._status_from_ls_remote(result, issue_number=123, remote="origin")

    assert payload["ok"] is True
    assert payload["claimed"] is True
    assert payload["sha"] == "def456"


def test_release_issue_succeeds_when_claim_ref_is_already_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release should be safe to retry after another process already deleted the ref."""
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> issue_claim.CommandResult:
        calls.append(command)
        return issue_claim.CommandResult(
            command=tuple(command),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(issue_claim, "_run", fake_run)

    payload = issue_claim.release_issue(123, remote="origin")

    assert payload["ok"] is True
    assert payload["claimed"] is False
    assert payload["stdout"] == "Ref does not exist, nothing to release."
    assert len(calls) == 1
    assert calls[0][0:2] == ["git", "ls-remote"]


def test_release_issue_deletes_existing_claim_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """Release should still delete the remote ref when status finds an existing claim."""
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> issue_claim.CommandResult:
        calls.append(command)
        if command[0:2] == ["git", "ls-remote"]:
            return issue_claim.CommandResult(
                command=tuple(command),
                returncode=0,
                stdout="abc123\trefs/heads/agent-claims/issue-123\n",
                stderr="",
            )
        return issue_claim.CommandResult(
            command=tuple(command),
            returncode=0,
            stdout="deleted\n",
            stderr="",
        )

    monkeypatch.setattr(issue_claim, "_run", fake_run)

    payload = issue_claim.release_issue(123, remote="origin")

    assert payload["ok"] is True
    assert payload["claimed"] is False
    assert payload["stdout"] == "deleted"
    assert calls[-1] == ["git", "push", "origin", ":refs/heads/agent-claims/issue-123"]


def test_main_returns_failure_when_acquire_push_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rejected GitHub create-ref call should make the acquire command fail closed."""
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> issue_claim.CommandResult:
        calls.append(command)
        if command[0:2] == ["git", "rev-parse"]:
            return issue_claim.CommandResult(
                command=tuple(command),
                returncode=0,
                stdout="abc123\n",
                stderr="",
            )
        return issue_claim.CommandResult(
            command=tuple(command),
            returncode=1,
            stdout="",
            stderr="remote ref already exists",
        )

    monkeypatch.setattr(issue_claim, "_run", fake_run)

    assert issue_claim.main(["acquire", "123"]) == 1
    assert calls[-1][0:4] == ["gh", "api", "-X", "POST"]
