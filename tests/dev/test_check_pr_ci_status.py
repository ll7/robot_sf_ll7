"""Tests for the PR CI status checker (deterministic, no live GitHub)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.check_pr_ci_status import _format_human, main


def test_format_human_success() -> None:
    """Human-readable output should include PR number, state, and check summaries."""
    data = {
        "status": "ok",
        "pr": 42,
        "title": "Fix the thing",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "branch": "fix-thing",
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


def test_main_gh_not_installed() -> None:
    """main() should exit non-zero when gh is not on PATH."""
    with patch(
        "scripts.dev.check_pr_ci_status.subprocess.run",
        side_effect=FileNotFoundError("gh not found"),
    ):
        rc = main(["42"])
    assert rc == 1
