"""Tests for compact goal-autopilot state snapshots."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

from scripts.dev import autopilot_state_snapshot as snapshot

if TYPE_CHECKING:
    from pathlib import Path


def _result(command: list[str], *, stdout: str = "", stderr: str = "", returncode: int = 0):
    return snapshot.CommandResult(
        command=tuple(command),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_json_compatibility_flag_is_accepted() -> None:
    """The documented --json flag remains compatible with the JSON-default helper."""
    args = snapshot._build_parser().parse_args(["--include-worktrees", "--json"])

    assert args.json is True
    assert args.include_worktrees is True


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


def test_build_snapshot_includes_queue_claim_pr_and_worktree_state(monkeypatch, tmp_path) -> None:
    """A normal snapshot should include compact queue, claim, PR, and worktree state."""
    monkeypatch.chdir(tmp_path)
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
    assert payload["controller_checkpoint"]["token_efficiency"] == {
        "parent_output_limit_lines": 200,
        "compact_first": True,
        "recommended_next_steps": list(snapshot.TOKEN_EFFICIENCY_ACTIONS),
    }
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
    assert payload["issues"][0]["admission"]["classification"] == "needs_ready_label"
    assert payload["issues"][0]["admission"]["claim_outcome"] == "not_checked"
    assert payload["prs"][0]["checks"]["overall"] == "success"
    assert payload["sources"]


def test_queue_issue_admission_uses_canonical_wrapper_for_ready_issue() -> None:
    """The canonical queue must expose the live check-only admission verdict."""
    result = {
        "schema": "goal_issue_admission.v1",
        "ok": True,
        "outcome": "ready_check_only",
        "write_attempted": False,
        "source_ref": "origin/main",
        "preflight": {
            "classification": "ready",
            "reasons": ["issue state and execution contract permit claim admission"],
            "ready": True,
            "write_allowed": True,
            "claim": {
                "ok": True,
                "claimed": False,
                "claim_ref": "agent-claims/issue-2672",
                "sha": None,
            },
        },
        "claim": None,
    }
    issue = {
        "number": 2672,
        "title": "ready issue",
        "state": "OPEN",
        "labels": [{"name": "state:ready"}],
    }

    with patch(
        "scripts.dev.autopilot_state_snapshot.goal_issue_admission.admit_issue",
        return_value=result,
    ) as admit:
        admission = snapshot._queue_issue_admission(issue)

    assert admission["outcome"] == "ready_check_only"
    assert admission["claim_outcome"] == "unclaimed"
    admit.assert_called_once_with(
        2672,
        repo="ll7/robot_sf_ll7",
        remote="origin",
        source_ref="origin/main",
        check_only=True,
    )


def test_compact_status_omits_untracked_inventory_and_reports_generated_paths(
    monkeypatch, tmp_path
) -> None:
    """Status snapshots should avoid generated untracked tree dumps and report generated roots."""
    venv_dir = tmp_path / ".venv"
    opencode_dir = tmp_path / ".opencode"
    node_modules_dir = tmp_path / "node_modules"
    output_coverage_dir = tmp_path / "output" / "coverage"
    pytest_cache_dir = tmp_path / ".pytest_cache"
    pycache_dir = tmp_path / "__pycache__"
    venv_dir.mkdir()
    opencode_dir.mkdir()
    node_modules_dir.mkdir()
    output_coverage_dir.mkdir(parents=True)
    pytest_cache_dir.mkdir()
    pycache_dir.mkdir()

    # Populate generated trees with child files so a directory-aware check still reports roots only.
    (venv_dir / "bin").mkdir()
    (venv_dir / "bin" / "python").touch()
    site_packages_dir = venv_dir / "lib" / "python3.11" / "site-packages"
    site_packages_dir.mkdir(parents=True)
    (site_packages_dir / "pkg.py").touch()
    opencode_node_modules = opencode_dir / "node_modules"
    opencode_node_modules.mkdir()
    (opencode_node_modules / "package.json").touch()
    left_pad_dir = opencode_node_modules / "left-pad"
    left_pad_dir.mkdir()
    (left_pad_dir / "index.js").touch()
    lodash_dir = node_modules_dir / "lodash"
    lodash_dir.mkdir()
    (lodash_dir / "index.js").touch()
    vite_bin_dir = node_modules_dir / "vite" / "bin"
    vite_bin_dir.mkdir(parents=True)
    (vite_bin_dir / "vite.js").touch()
    (output_coverage_dir / "coverage.xml").touch()
    htmlcov_dir = output_coverage_dir / "htmlcov"
    htmlcov_dir.mkdir()
    (htmlcov_dir / "index.html").touch()
    (pytest_cache_dir / "README.md").touch()
    (pycache_dir / "module.cpython-313.pyc").touch()

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
    assert sorted(status["generated_paths_present"]) == sorted(
        [".venv", ".opencode", "node_modules", "output", ".pytest_cache", "__pycache__"]
    )
    assert status["tracked_or_staged"] == [" M docs/dev_guide.md"]
    assert status["full_untracked_inventory_omitted"] is True
    # Child files of generated trees must not leak into the compact status payload.
    payload_text = json.dumps(status)
    assert "site-packages/pkg.py" not in payload_text
    assert "left-pad/index.js" not in payload_text
    assert "lodash/index.js" not in payload_text
    assert "vite.js" not in payload_text
    assert "htmlcov/index.html" not in payload_text
    assert "module.cpython-313.pyc" not in payload_text


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


def test_route_manifest_snapshot_exposes_failed_route_without_acceptance(tmp_path: Path) -> None:
    """Route handoffs should expose terminal/missing state while keeping acceptance unset."""
    manifest_path = tmp_path / "routing_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "routed_worker_manifest.v2",
                "route_evidence_only": True,
                "chosen_route": {"provider": "opencode-go"},
                "chosen_run_dir": str(tmp_path / "run-1"),
                "attempted_routes": [
                    {
                        "attempt_index": 0,
                        "route": {"provider": "opencode-go"},
                        "returncode": 124,
                        "failure_class": "timeout",
                        "terminal_state": "timeout",
                        "run_dir": str(tmp_path / "run-1"),
                        "scope_check": {
                            "ok": True,
                            "spill_detected": False,
                        },
                        "compact_artifacts": {
                            "result_json": {"present": False, "reason": "missing"},
                            "result_md": {"present": True},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = snapshot.route_manifest_snapshot(manifest_path)

    assert row["status"] == "ok"
    assert row["chosen_terminal_state"] == "timeout"
    assert row["chosen_failure_class"] == "timeout"
    assert row["chosen_missing_artifacts"] == ["result_json"]
    assert row["acceptance_state"] == "not_established"
    assert row["route_evidence_only"] is True
    assert len(row["failed_attempts"]) == 1
    assert len(row["incomplete_output_attempts"]) == 1
    assert row["aggregation"] == "inconclusive"
    assert row["next_action"] == "inspect_parent_diff_and_run_local_validation"


def test_route_manifest_snapshot_overrides_false_confirmed_aggregation(tmp_path: Path) -> None:
    """A reported confirmed aggregate must be downgraded when output evidence is empty."""
    manifest_path = tmp_path / "routing_manifest.json"
    compact_artifacts = {
        key: {"present": True, "path": f"run/{key}", "size_bytes": 10}
        for key in ("result_json", "result_md", "diffstat", "status", "validation")
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "routed_worker_manifest.v2",
                "route_evidence_only": True,
                "aggregation": "confirmed",
                "chosen_route": {"provider": "antigravity"},
                "chosen_run_dir": str(tmp_path / "run-1"),
                "attempted_routes": [
                    {
                        "attempt_index": 0,
                        "route": {"provider": "antigravity"},
                        "returncode": 0,
                        "failure_class": "none",
                        "terminal_state": "none",
                        "run_dir": str(tmp_path / "run-1"),
                        "stdout": "",
                        "stderr": "headless command permission denied",
                        "compact_artifacts": compact_artifacts,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = snapshot.route_manifest_snapshot(manifest_path)

    assert row["aggregation"] == "inconclusive"
    assert row["aggregation_reason"] == "reported_confirmed_without_usable_worker_output"
    assert row["chosen_output_contract"]["missing_evidence"] == [
        "worker_output_empty",
        "permission_denied",
    ]


def test_route_manifest_snapshot_preserves_valid_success_route_evidence(tmp_path: Path) -> None:
    """A complete v2 success remains route evidence without implying acceptance."""
    run_dir = tmp_path / "run-success"
    manifest_path = tmp_path / "routing_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "routed_worker_manifest.v2",
                "route_evidence_only": True,
                "chosen_route": {"provider": "local"},
                "chosen_run_dir": str(run_dir),
                "attempted_routes": [
                    {
                        "attempt_index": 0,
                        "route": {"provider": "local"},
                        "returncode": 0,
                        "failure_class": "none",
                        "terminal_state": "none",
                        "run_dir": str(run_dir),
                        "compact_artifacts": {
                            "result_json": {"present": True, "size_bytes": 10},
                            "result_md": {"present": True, "size_bytes": 10},
                            "validation": {"present": True, "size_bytes": 10},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = snapshot.route_manifest_snapshot(manifest_path)

    assert row["status"] == "ok"
    assert row["chosen_terminal_state"] == "none"
    assert row["failed_attempts"] == []
    assert row["aggregation"] == "confirmed"
    assert row["acceptance_state"] == "not_established"


def test_route_manifest_snapshot_rejects_missing_schema(tmp_path: Path) -> None:
    """A route manifest without the canonical schema must fail closed."""
    manifest_path = tmp_path / "routing_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "route_evidence_only": True,
                "attempted_routes": [],
            }
        ),
        encoding="utf-8",
    )

    row = snapshot.route_manifest_snapshot(manifest_path)

    assert row["status"] == "malformed"
    assert row["schema"] is None
    assert "schema" in row["error"]
    assert row["acceptance_state"] == "not_established"


def test_route_manifest_snapshot_marks_missing_terminal_state_unavailable(
    tmp_path: Path,
) -> None:
    """A v2 attempt without terminal state must not look like a successful route."""
    run_dir = tmp_path / "run-1"
    manifest_path = tmp_path / "routing_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "routed_worker_manifest.v2",
                "route_evidence_only": True,
                "chosen_route": {"provider": "opencode-go"},
                "chosen_run_dir": str(run_dir),
                "attempted_routes": [
                    {
                        "attempt_index": 0,
                        "route": {"provider": "opencode-go"},
                        "returncode": 0,
                        "failure_class": "none",
                        "run_dir": str(run_dir),
                        "compact_artifacts": {
                            "result_json": {"present": False, "reason": "missing"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = snapshot.route_manifest_snapshot(manifest_path)

    assert row["status"] == "unavailable"
    assert row["chosen_terminal_state"] == "unavailable"
    assert row["chosen_failure_class"] == "missing_terminal_state"
    assert row["chosen_missing_artifacts"] == ["result_json"]
    assert row["failed_attempts"][0]["terminal_state"] == "unavailable"
    assert row["acceptance_state"] == "not_established"
    assert row["next_action"] == "inspect_route_manifest_path_and_route_artifacts"


def test_route_manifest_snapshot_rejects_unsupported_terminal_state(tmp_path: Path) -> None:
    """Unknown terminal states must remain malformed rather than accepted evidence."""
    manifest_path = tmp_path / "routing_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "routed_worker_manifest.v2",
                "route_evidence_only": True,
                "attempted_routes": [
                    {
                        "terminal_state": "unexpected",
                        "returncode": 0,
                        "run_dir": str(tmp_path / "run-1"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = snapshot.route_manifest_snapshot(manifest_path)

    assert row["status"] == "malformed"
    assert "unsupported terminal_state" in row["error"]
    assert row["acceptance_state"] == "not_established"


def test_route_manifest_snapshot_rejects_malformed_attempt_record(tmp_path: Path) -> None:
    """Non-object attempt records must not be normalized into unavailable evidence."""
    manifest_path = tmp_path / "routing_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "routed_worker_manifest.v2",
                "route_evidence_only": True,
                "attempted_routes": ["not-an-attempt"],
            }
        ),
        encoding="utf-8",
    )

    row = snapshot.route_manifest_snapshot(manifest_path)

    assert row["status"] == "malformed"
    assert "malformed attempt records" in row["error"]
    assert row["acceptance_state"] == "not_established"


def test_route_manifest_snapshot_reports_unavailable_manifest(tmp_path: Path) -> None:
    """A missing route manifest must remain an explicit handoff error."""
    row = snapshot.route_manifest_snapshot(tmp_path / "missing-routing-manifest.json")

    assert row["status"] == "unavailable"
    assert row["acceptance_state"] == "not_established"
    assert "route manifest unavailable" in row["error"]


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
        "superseded": 0,
        "by_conclusion": {"success": 1},
        "by_status": {"completed": 1},
        "names": ["ci"],
    }


def test_checks_summary_suppresses_superseded_cancelled_rerun() -> None:
    """Verify that a newer successful rerun prevents a superseded cancellation from creating a false failure."""
    summary = snapshot._checks_summary(
        [
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-08-14T01:00:00Z",
            },
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "status": "completed",
                "conclusion": "success",
                "startedAt": "2026-08-14T01:05:00Z",
            },
        ]
    )

    assert summary["overall"] == "success"
    assert summary["total"] == 1
    assert summary["superseded"] == 1
    assert summary["by_conclusion"] == {"success": 1}
    assert summary["names"] == ["pr-body-contracts"]


def test_checks_summary_keeps_unreplaced_cancellation_fail_closed() -> None:
    """Verify that an unreplaced cancellation keeps the snapshot fail-closed."""
    summary = snapshot._checks_summary(
        [
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-08-14T01:00:00Z",
            }
        ]
    )

    assert summary["overall"] == "failure"
    assert summary["total"] == 1
    assert summary["superseded"] == 0
    assert summary["by_conclusion"] == {"cancelled": 1}
