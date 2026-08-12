"""Tests for the fail-closed pre-publication remote-state gate."""

from __future__ import annotations

import json
import subprocess
from typing import Any

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


def test_collect_live_state_records_explicit_closing_pr(monkeypatch) -> None:
    """Live collection extracts an explicit closing reference from merged PR data."""
    commands: list[list[str]] = []

    def fake_json(command: list[str]) -> Any:
        commands.append(command)
        if command[1] == "issue":
            return {"state": "OPEN", "updatedAt": "2026-08-12T10:00:00Z", "closedAt": None}
        return [
            {
                "number": 7001,
                "title": "fix: close the issue",
                "body": "Closes #6916",
                "mergedAt": "2026-08-12T10:02:00Z",
                "mergeCommit": {"oid": "merge-a"},
                "headRefName": "fix/6916",
                "headRefOid": "head-fix",
                "baseRefName": "main",
            }
        ]

    monkeypatch.setattr(gate, "_json_command", fake_json)
    monkeypatch.setattr(gate, "_fetch_refs", lambda **_: ("base-a", "branch-a"))
    monkeypatch.setattr(gate, "_git_output", lambda *_: "head-a")
    monkeypatch.setattr(gate, "_tree_state", lambda: "clean")

    result = gate.collect_live_state(
        repo="ll7/robot_sf_ll7",
        issue=6916,
        branch="feature/fresh-state",
    )

    assert result["closing_prs"] == [
        {
            "number": 7001,
            "title": "fix: close the issue",
            "merged_at": "2026-08-12T10:02:00Z",
            "merge_commit": {"oid": "merge-a"},
            "head_ref": "fix/6916",
            "head_sha": "head-fix",
            "base_ref": "main",
        }
    ]
    assert [command[1] for command in commands] == ["issue", "pr"]


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
    monkeypatch.setattr(gate, "_fetch_remote_branch", lambda **_: None)

    result = gate._integrate_targets(
        remote="origin",
        branch="feature/fresh-state",
        targets=["refs/remotes/origin/feature/fresh-state"],
    )

    assert result["ok"] is False
    assert result["reason"] == "integration_conflict"
    assert "reset" not in " ".join(" ".join(command) for command in commands)


def test_parser_exposes_capture_check_and_sync() -> None:
    """All three lifecycle actions remain discoverable from the CLI parser."""
    parser = gate._parser()

    assert parser.parse_args(["capture", "--repo", "o/r", "--issue", "1"]).command == "capture"
    assert parser.parse_args(["check", "--snapshot-path", "state.json"]).command == "check"
    assert parser.parse_args(["sync", "--snapshot-path", "state.json", "--integrate"]).integrate


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
