"""Tests for the PR CI status checker (deterministic, no live GitHub)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.check_pr_ci_status import (
    _fetch_ci_status,
    _format_human,
    _rollup_conclusion,
    _rollup_name,
    _rollup_status,
    main,
)


def test_format_human_success() -> None:
    """Human-readable output should include PR number, state, and check summaries."""
    data = {
        "status": "ok",
        "pr": 42,
        "title": "Fix the thing",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "branch": "fix-thing",
        "head_sha": "abc123",
        "checks": {
            "total": 3,
            "overall": "failure",
            "by_conclusion": {"success": 2, "failure": 1},
            "by_status": {"completed": 3},
            "names": ["lint", "test", "build"],
        },
        "reviews": {"APPROVED": 2},
    }
    output = _format_human(data)
    assert "PR #42" in output
    assert "Fix the thing" in output
    assert "OPEN" in output
    assert "MERGEABLE" in output
    assert "abc123" in output
    assert "checks: failure" in output
    assert "success=2" in output
    assert "failure=1" in output
    assert "APPROVED=2" in output


def test_format_human_error() -> None:
    """Error status should produce an error message."""
    data = {"status": "error", "error": "HTTP 404: not found"}
    output = _format_human(data)
    assert "ERROR" in output
    assert "HTTP 404" in output


def test_format_human_no_reviews() -> None:
    """When no reviews are present, that section should be omitted."""
    data = {
        "status": "ok",
        "pr": 1,
        "title": "no reviews",
        "state": "OPEN",
        "mergeable": "UNKNOWN",
        "branch": "no-reviews",
        "checks": {
            "total": 1,
            "overall": "success",
            "by_conclusion": {"success": 1},
            "by_status": {"completed": 1},
            "names": ["ci"],
        },
        "reviews": {},
    }
    output = _format_human(data)
    assert "reviews:" not in output


def test_main_with_explicit_pr_and_failure_exit(
    capsys: pytest.CaptureFixture,
) -> None:
    """main() should exit non-zero when CI has failures."""
    mock_data = json.dumps(
        {
            "number": 42,
            "title": "broken PR",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "broken",
            "headRefOid": "abc123",
            "statusCheckRollup": [
                {"conclusion": "success", "status": "completed", "name": "lint"},
                {"conclusion": "failure", "status": "completed", "name": "test"},
            ],
            "reviews": [],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=mock_data,
            stderr="",
        )
        rc = main(["42"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "PR #42" in captured.out
    assert "failure=1" in captured.out
    assert "checks: failure" in captured.out
    assert "abc123" in captured.out


def test_main_with_startup_failure_exit(capsys: pytest.CaptureFixture) -> None:
    """GitHub startup_failure conclusions should be treated as failed CI."""
    mock_data = json.dumps(
        {
            "number": 43,
            "title": "workflow issue",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "workflow-issue",
            "statusCheckRollup": [
                {"conclusion": "startup_failure", "status": "completed", "name": "ci"},
            ],
            "reviews": [],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_data, stderr="")
        rc = main(["43"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "startup_failure=1" in captured.out


def test_main_with_explicit_pr_and_success(
    capsys: pytest.CaptureFixture,
) -> None:
    """main() should exit zero when all CI checks pass."""
    mock_data = json.dumps(
        {
            "number": 1,
            "title": "working PR",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "working",
            "statusCheckRollup": [
                {"conclusion": "success", "status": "completed", "name": "ci"},
            ],
            "reviews": [{"state": "APPROVED"}],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=mock_data,
            stderr="",
        )
        rc = main(["1"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "PR #1" in captured.out
    assert "checks: success" in captured.out


def test_main_treats_successful_legacy_status_as_success(
    capsys: pytest.CaptureFixture,
) -> None:
    """Legacy commit statuses expose state without conclusion/status fields."""
    mock_data = json.dumps(
        {
            "number": 2537,
            "title": "legacy status",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "legacy-status",
            "statusCheckRollup": [
                {"conclusion": "success", "status": "completed", "name": "ci"},
                {"state": "success", "name": "CodeRabbit"},
            ],
            "reviews": [{"state": "COMMENTED"}],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_data, stderr="")
        rc = main(["2537", "--json"])

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["checks"]["overall"] == "success"
    assert payload["checks"]["by_conclusion"] == {"success": 2}
    assert payload["checks"]["by_status"] == {"completed": 2}
    assert payload["checks"]["names"] == ["CodeRabbit", "ci"]


def test_main_preserves_pending_legacy_status(
    capsys: pytest.CaptureFixture,
) -> None:
    """Pending legacy commit statuses should still block success."""
    mock_data = json.dumps(
        {
            "number": 2,
            "title": "pending legacy",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "pending-legacy",
            "statusCheckRollup": [
                {"conclusion": "success", "status": "completed", "name": "ci"},
                {"state": "pending", "name": "external-status"},
            ],
            "reviews": [],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_data, stderr="")
        rc = main(["2", "--json"])

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["checks"]["overall"] == "pending"
    assert payload["checks"]["by_conclusion"] == {"success": 1, "pending": 1}
    assert payload["checks"]["by_status"] == {"completed": 1, "pending": 1}
    assert payload["checks"]["details"][1]["name"] == "external-status"
    assert payload["checks"]["details"][1]["status"] == "pending"


def test_main_treats_error_legacy_status_as_failure(
    capsys: pytest.CaptureFixture,
) -> None:
    """Legacy error states should fail CI preflight."""
    mock_data = json.dumps(
        {
            "number": 3,
            "title": "error legacy",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "error-legacy",
            "statusCheckRollup": [
                {"state": "error", "name": "external-status"},
            ],
            "reviews": [],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_data, stderr="")
        rc = main(["3"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "checks: failure" in captured.out
    assert "error=1" in captured.out


@pytest.mark.parametrize(
    ("check", "expected"),
    [
        ({"conclusion": "SUCCESS", "state": "failure"}, "success"),
        ({"conclusion": None, "state": "SUCCESS"}, "success"),
        ({"state": ""}, "pending"),
        ({}, "pending"),
    ],
)
def test_rollup_conclusion_prefers_conclusion_then_state(
    check: dict[str, object],
    expected: str,
) -> None:
    """Conclusion normalization should handle check-run and legacy-status entries."""
    assert _rollup_conclusion(check) == expected


@pytest.mark.parametrize(
    ("check", "expected"),
    [
        ({"status": "IN_PROGRESS", "state": "success"}, "in_progress"),
        ({"status": "", "state": "SUCCESS"}, "completed"),
        ({"state": "FAILURE"}, "completed"),
        ({"state": "ERROR"}, "completed"),
        ({"state": "PENDING"}, "pending"),
        ({}, "completed"),
    ],
)
def test_rollup_status_prefers_status_then_legacy_state(
    check: dict[str, object],
    expected: str,
) -> None:
    """Lifecycle status normalization should avoid double-counting legacy pending checks."""
    assert _rollup_status(check) == expected


def test_rollup_name_prefers_check_name_then_legacy_context() -> None:
    """Legacy status contexts should not render as unknown checks."""
    assert _rollup_name({"name": "ci", "context": "legacy"}) == "ci"
    assert _rollup_name({"context": "CodeRabbit"}) == "CodeRabbit"
    assert _rollup_name({}) == "unknown"


def test_main_json_output(capsys: pytest.CaptureFixture) -> None:
    """main() with --json should emit a JSON blob."""
    mock_data = json.dumps(
        {
            "number": 7,
            "title": "json test",
            "state": "OPEN",
            "mergeable": "UNKNOWN",
            "headRefName": "json-test",
            "statusCheckRollup": [],
            "reviews": [],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=mock_data,
            stderr="",
        )
        rc = main(["7", "--json"])

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["pr"] == 7
    assert payload["head_sha"] == ""
    assert payload["checks"]["overall"] == "pending"


def test_main_gh_error(capsys: pytest.CaptureFixture) -> None:
    """main() should report gh errors and exit non-zero."""
    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="HTTP 404: Not Found (PR not found)",
        )
        rc = main(["999999"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.out


def test_fetch_ci_status_malformed_json() -> None:
    """Malformed successful gh output should return an error payload."""
    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="{oops", stderr="")
        data = _fetch_ci_status("42")

    assert data["status"] == "error"
    assert "Failed to parse gh output as JSON" in data["error"]


def test_fetch_ci_status_non_object_json() -> None:
    """Non-object successful gh output should return an error payload."""
    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
        data = _fetch_ci_status("42")

    assert data["status"] == "error"
    assert data["error"] == "gh output is not a JSON object"


def test_main_with_backoff(capsys: pytest.CaptureFixture) -> None:
    """main() should pass --backoff to _fetch_ci_status."""
    mock_data = json.dumps(
        {
            "number": 3,
            "title": "backoff test",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "backoff",
            "statusCheckRollup": [],
            "reviews": [],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=mock_data,
            stderr="",
        )
        rc = main(["3", "--backoff", "0.0"])

    assert rc == 0


def test_format_human_lists_non_success_details() -> None:
    """Human output should distinguish queued/in-progress/failing checks."""
    data = {
        "status": "ok",
        "pr": 5,
        "title": "details",
        "state": "OPEN",
        "mergeable": "UNKNOWN",
        "branch": "details",
        "head_sha": "abc123",
        "checks": {
            "total": 3,
            "overall": "pending",
            "by_conclusion": {"success": 1, "pending": 2},
            "by_status": {"completed": 1, "queued": 1, "in_progress": 1},
            "names": ["done", "queued", "running"],
            "details": [
                {
                    "name": "done",
                    "status": "completed",
                    "conclusion": "success",
                    "details_url": "https://example.test/done",
                },
                {
                    "name": "queued",
                    "status": "queued",
                    "conclusion": "pending",
                    "details_url": "https://example.test/queued",
                },
                {
                    "name": "running",
                    "status": "in_progress",
                    "conclusion": "pending",
                    "details_url": "https://example.test/running",
                },
            ],
        },
        "reviews": {},
    }

    output = _format_human(data)

    assert "done:" not in output
    assert "queued: queued/pending" in output
    assert "running: in_progress/pending" in output
    assert "https://example.test/running" in output


def test_main_bounded_polling_times_out_pending(
    capsys: pytest.CaptureFixture,
) -> None:
    """Bounded polling should return a distinct code when checks never settle."""
    mock_data = json.dumps(
        {
            "number": 7,
            "title": "pending poll",
            "state": "OPEN",
            "mergeable": "UNKNOWN",
            "headRefName": "pending-poll",
            "statusCheckRollup": [
                {
                    "name": "ci",
                    "status": "queued",
                    "conclusion": "",
                    "detailsUrl": "https://example.test/ci",
                }
            ],
            "reviews": [],
        }
    )

    with (
        patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run,
        patch("scripts.dev.check_pr_ci_status.time.sleep") as mock_sleep,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_data, stderr="")
        rc = main(["7", "--poll-attempts", "2", "--poll-interval", "0.1"])

    assert rc == 2
    assert mock_run.call_count == 2
    mock_sleep.assert_called_once_with(0.1)
    captured = capsys.readouterr()
    assert "poll attempt 1/2" in captured.out
    assert "poll attempt 2/2" in captured.out
    assert "ci: queued/pending" in captured.out


def test_main_bounded_polling_stops_on_success(
    capsys: pytest.CaptureFixture,
) -> None:
    """Bounded polling should stop once pending checks pass."""
    pending_data = json.dumps(
        {
            "number": 8,
            "title": "eventual pass",
            "state": "OPEN",
            "mergeable": "UNKNOWN",
            "headRefName": "eventual-pass",
            "statusCheckRollup": [{"name": "ci", "status": "queued", "conclusion": ""}],
            "reviews": [],
        }
    )
    success_data = json.dumps(
        {
            "number": 8,
            "title": "eventual pass",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "eventual-pass",
            "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
            "reviews": [],
        }
    )

    with (
        patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run,
        patch("scripts.dev.check_pr_ci_status.time.sleep") as mock_sleep,
    ):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=pending_data, stderr=""),
            MagicMock(returncode=0, stdout=success_data, stderr=""),
        ]
        rc = main(["8", "--poll-attempts", "5", "--poll-interval", "0.1"])

    assert rc == 0
    assert mock_run.call_count == 2
    mock_sleep.assert_called_once_with(0.1)
    captured = capsys.readouterr()
    assert "poll attempt 2/5" in captured.out
    assert "checks: success" in captured.out


def test_main_bounded_polling_json_includes_monitor_metadata(
    capsys: pytest.CaptureFixture,
) -> None:
    """JSON polling should emit compact monitor metadata suitable for a delegation ledger."""
    pending_data = json.dumps(
        {
            "number": 9,
            "title": "metadata poll",
            "state": "OPEN",
            "mergeable": "UNKNOWN",
            "headRefName": "metadata-poll",
            "headRefOid": "abc123",
            "statusCheckRollup": [{"name": "ci", "status": "queued", "conclusion": ""}],
            "reviews": [],
        }
    )
    success_data = json.dumps(
        {
            "number": 9,
            "title": "metadata poll",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "metadata-poll",
            "headRefOid": "abc123",
            "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
            "reviews": [],
        }
    )

    with (
        patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run,
        patch("scripts.dev.check_pr_ci_status.time.sleep") as mock_sleep,
        patch("scripts.dev.check_pr_ci_status.time.time", return_value=1000.0),
    ):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=pending_data, stderr=""),
            MagicMock(returncode=0, stdout=success_data, stderr=""),
        ]
        rc = main(
            [
                "9",
                "--json",
                "--expected-head-sha",
                "abc123",
                "--poll-attempts",
                "5",
                "--poll-interval",
                "2.0",
            ]
        )

    assert rc == 0
    assert mock_run.call_count == 2
    mock_sleep.assert_called_once_with(2.0)
    payloads = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(payloads) == 2
    assert payloads[0]["monitor"] == {
        "route": "ci_wait_monitor",
        "expected_head_sha": "abc123",
        "head_sha_matches_expected": True,
        "poll_attempt": 1,
        "poll_attempts": 5,
        "poll_interval_seconds": 2.0,
        "wait_budget_seconds": 8.0,
        "deadline_epoch_seconds": 1008,
        "route_evidence_only": True,
    }
    assert payloads[1]["monitor"]["poll_attempt"] == 2
    assert payloads[1]["checks"]["overall"] == "success"


def test_main_expected_head_sha_mismatch_returns_json_error(
    capsys: pytest.CaptureFixture,
) -> None:
    """A stale PR head should fail closed before monitor output is trusted."""
    mock_data = json.dumps(
        {
            "number": 10,
            "title": "stale monitor",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "stale-monitor",
            "headRefOid": "actual",
            "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
            "reviews": [],
        }
    )

    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_data, stderr="")
        rc = main(["10", "--json", "--expected-head-sha", "expected"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["error"] == "PR head SHA changed while monitoring CI"
    assert payload["head_sha"] == "actual"
    assert payload["monitor"]["expected_head_sha"] == "expected"
    assert payload["monitor"]["head_sha_matches_expected"] is False
    assert payload["monitor"]["route_evidence_only"] is True


def test_main_gh_not_installed() -> None:
    """main() should exit non-zero when gh is not on PATH."""
    with patch(
        "scripts.dev.check_pr_ci_status.subprocess.run",
        side_effect=FileNotFoundError("gh not found"),
    ):
        rc = main(["42"])
    assert rc == 1


def test_main_gh_timeout() -> None:
    """main() should exit non-zero when gh times out."""
    with patch(
        "scripts.dev.check_pr_ci_status.subprocess.run",
        side_effect=subprocess.TimeoutExpired(["gh"], timeout=30),
    ):
        rc = main(["42"])
    assert rc == 1


def test_help_includes_worktree_safe_invocation(
    capsys: pytest.CaptureFixture,
) -> None:
    """--help should document the wrapper invocation used by agent workflows."""
    with pytest.raises(SystemExit):
        main(["--help"])

    captured = capsys.readouterr()
    assert "run_worktree_shared_venv.sh" in captured.out
    assert "uv run python scripts/dev/check_pr_ci_status.py" in captured.out
    assert "--expected-head-sha" in captured.out
    assert "no local .venv" in captured.out
    assert "UV_NO_SYNC" in captured.out


def test_direct_invocation_help_succeeds_without_pythonpath() -> None:
    """python scripts/dev/check_pr_ci_status.py --help should work without PYTHONPATH."""
    import os

    script = str(
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dev" / "check_pr_ci_status.py"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "run_worktree_shared_venv.sh" in result.stdout
    assert "--expected-head-sha" in result.stdout
