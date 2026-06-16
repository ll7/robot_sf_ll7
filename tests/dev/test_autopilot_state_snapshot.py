"""Tests for compact goal-autopilot state snapshots."""

from __future__ import annotations

import json
import subprocess

from scripts.dev import autopilot_state_snapshot as snapshot


def _result(command: list[str], *, stdout: str = "", stderr: str = "", returncode: int = 0):
    return snapshot.CommandResult(
        command=tuple(command),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_parse_worktree_porcelain_summarizes_linked_worktrees() -> None:
    """Worktree porcelain output should become compact branch/head rows."""
    rows = snapshot._parse_worktree_porcelain(
        "\n".join(
            [
                "worktree /repo/main",
                "HEAD abc123",
                "branch refs/heads/main",
                "",
                "worktree /repo/feature",
                "HEAD def456",
                "branch refs/heads/issue-1",
                "",
            ]
        )
    )

    assert rows == [
        {
            "path": "/repo/main",
            "head_sha": "abc123",
            "branch": "main",
            "bare": False,
            "detached": False,
        },
        {
            "path": "/repo/feature",
            "head_sha": "def456",
            "branch": "issue-1",
            "bare": False,
            "detached": False,
        },
    ]


def test_build_snapshot_includes_queue_claim_pr_and_worktree_state(monkeypatch) -> None:
    """A normal snapshot should include compact queue, claim, PR, and worktree state."""
    origin_main_sha = "6e55ea36affa82ea1b3c870c27f0133464295fd0"
    exact_results = {
        ("git", "branch", "--show-current"): _result(
            ["git", "branch", "--show-current"],
            stdout="issue-2671-compact-state-snapshots\n",
        ),
        ("git", "rev-parse", "HEAD"): _result(
            ["git", "rev-parse", "HEAD"],
            stdout=f"{origin_main_sha}\n",
        ),
        ("git", "rev-parse", "--verify", "origin/main^{commit}"): _result(
            ["git", "rev-parse", "--verify", "origin/main^{commit}"],
            stdout=f"{origin_main_sha}\n",
        ),
        ("git", "worktree", "list", "--porcelain"): _result(
            ["git", "worktree", "list", "--porcelain"],
            stdout=(
                "worktree /repo/main\n"
                f"HEAD {origin_main_sha}\n"
                "branch refs/heads/main\n\n"
                "worktree /repo/issue-2671\n"
                f"HEAD {origin_main_sha}\n"
                "branch refs/heads/issue-2671-compact-state-snapshots\n"
            ),
        ),
        ("git", "status", "--short", "--branch", "--untracked-files=no"): _result(
            ["git", "status", "--short", "--branch", "--untracked-files=no"],
            stdout="## issue-2671-compact-state-snapshots...origin/main\n",
        ),
    }

    def fake_run(command: list[str], *, timeout: int = 30):
        del timeout
        if result := exact_results.get(tuple(command)):
            return result
        if command[:3] == ["git", "ls-remote", "--heads"]:
            return _result(
                command,
                stdout=f"{origin_main_sha}\trefs/heads/agent-claims/issue-2671\n",
            )
        if command[:3] == ["gh", "issue", "list"]:
            return _result(
                command,
                stdout=json.dumps(
                    [
                        {
                            "number": 2671,
                            "title": "Reduce token burn",
                            "state": "OPEN",
                            "labels": [{"name": "enhancement"}],
                            "updatedAt": "2026-06-12T09:00:00Z",
                            "url": "https://example.test/issues/2671",
                        }
                    ]
                ),
            )
        if command[:3] == ["gh", "pr", "view"]:
            return _result(
                command,
                stdout=json.dumps(
                    {
                        "number": 2683,
                        "title": "compact CI monitor evidence",
                        "state": "OPEN",
                        "mergeable": "MERGEABLE",
                        "headRefName": "issue-2672",
                        "headRefOid": "abc123",
                        "statusCheckRollup": [
                            {"name": "ci", "status": "completed", "conclusion": "success"}
                        ],
                        "url": "https://example.test/pull/2683",
                    }
                ),
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(snapshot, "_run", fake_run)
    args = snapshot._build_parser().parse_args(
        [
            "--include-worktrees",
            "--claim-issue",
            "2671",
            "--issue-search",
            "is:issue is:open 2671",
            "--pr",
            "2683",
        ]
    )

    payload = snapshot.build_snapshot(args)

    assert payload["schema"] == "autopilot_state_snapshot.v1"
    assert payload["ok"] is True
    assert payload["freshness"]["route_evidence_only"] is True
    assert payload["git"]["branch"] == "issue-2671-compact-state-snapshots"
    assert payload["git"]["worktree_count"] == 2
    assert payload["git"]["worktrees_truncated"] is False
    assert payload["git"]["worktrees"][0]["branch"] == "issue-2671-compact-state-snapshots"
    assert payload["git"]["compact_status"]["full_untracked_inventory_omitted"] is True
    assert payload["controller_checkpoint"]["branch"] == "issue-2671-compact-state-snapshots"
    assert payload["controller_checkpoint"]["next_action"] == "continue_from_snapshot"
    assert payload["claims"] == [
        {
            "issue": 2671,
            "ok": True,
            "claimed": True,
            "claim_ref": "agent-claims/issue-2671",
            "sha": origin_main_sha,
            "stale_against_origin_main": False,
            "error": None,
        }
    ]
    assert payload["issues"][0]["labels"] == ["enhancement"]
    assert payload["prs"][0]["checks"]["overall"] == "success"
    assert payload["sources"]


def test_compact_status_omits_untracked_inventory_and_reports_generated_paths(
    monkeypatch, tmp_path
) -> None:
    """Status snapshots should avoid generated untracked tree dumps."""
    venv_file = tmp_path / ".venv"
    opencode_file = tmp_path / ".opencode"
    node_modules_dir = tmp_path / "node_modules"
    output_coverage_dir = tmp_path / "output" / "coverage"
    venv_file.touch()
    opencode_file.touch()
    node_modules_dir.mkdir()
    output_coverage_dir.mkdir(parents=True)

    def fake_run(command: list[str], *, timeout: int = 30):
        del timeout
        if command == ["git", "status", "--short", "--branch", "--untracked-files=no"]:
            return _result(command, stdout="## branch...origin/main\n M docs/dev_guide.md\n")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(snapshot, "_run", fake_run)

    status, source, error = snapshot.compact_status_snapshot()

    assert error is None
    assert source["name"] == "git.status_compact"
    assert status["tracked_or_staged_count"] == 1
    assert status["generated_paths_present"] == [".venv", ".opencode"]
    assert status["tracked_or_staged"] == [" M docs/dev_guide.md"]
    assert status["full_untracked_inventory_omitted"] is True


def test_claim_snapshot_marks_stale_claim_against_origin_main(monkeypatch) -> None:
    """Claim refs should expose when their source SHA is behind origin/main."""
    old_sha = "1111111111111111111111111111111111111111"
    new_sha = "2222222222222222222222222222222222222222"

    def fake_run(command: list[str], *, timeout: int = 30):
        del timeout
        return _result(command, stdout=f"{old_sha}\trefs/heads/agent-claims/issue-2671\n")

    monkeypatch.setattr(snapshot, "_run", fake_run)

    rows, sources, errors = snapshot.claim_snapshot(
        [2671], remote="origin", origin_main_sha=new_sha
    )

    assert errors == []
    assert sources[0]["name"] == "claim.issue_2671"
    assert rows[0]["claimed"] is True
    assert rows[0]["stale_against_origin_main"] is True


def test_claim_snapshot_reports_missing_state_errors(monkeypatch) -> None:
    """A failed claim lookup should stay compact but make the snapshot not-ok."""

    def fake_run(command: list[str], *, timeout: int = 30):
        del timeout
        return _result(command, stderr="network unavailable", returncode=1)

    monkeypatch.setattr(snapshot, "_run", fake_run)

    rows, _sources, errors = snapshot.claim_snapshot(
        [2671],
        remote="origin",
        origin_main_sha="6e55ea36affa82ea1b3c870c27f0133464295fd0",
    )

    assert errors == ["issue 2671: network unavailable"]
    assert rows[0]["ok"] is False
    assert rows[0]["claimed"] is None
    assert rows[0]["error"] == "network unavailable"


def test_run_converts_timeouts_to_compact_command_result(monkeypatch) -> None:
    """Hung git or gh commands should become compact snapshot errors, not tracebacks."""

    def fake_subprocess_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["gh"], timeout=5)

    monkeypatch.setattr(snapshot.subprocess, "run", fake_subprocess_run)

    result = snapshot._run(["gh", "issue", "list"], timeout=5)

    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == "command timed out after 5 seconds"


def test_checks_summary_ignores_malformed_rollup_entries() -> None:
    """Unexpected PR rollup entries should not crash compact PR snapshots."""
    summary = snapshot._checks_summary(
        [
            None,  # type: ignore[list-item]
            "bad",  # type: ignore[list-item]
            {"name": "ci", "status": "completed", "conclusion": "success"},
        ]
    )

    assert summary == {
        "overall": "success",
        "total": 1,
        "by_conclusion": {"success": 1},
        "by_status": {"completed": 1},
        "names": ["ci"],
    }
