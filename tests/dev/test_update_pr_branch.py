"""Tests for the guarded REST-based PR branch updater."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.dev.update_pr_branch import main, update_pr_branch


def _gh_response(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Create a mock subprocess.CompletedProcess for gh calls."""
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_update_pr_branch_sends_guarded_rest_request() -> None:
    """The update request must carry the caller's expected head SHA."""
    with patch("scripts.dev.update_pr_branch._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps({"head": {"sha": "head_sha"}})),
            _gh_response(stdout=json.dumps({"message": "Update queued"})),
        ]
        result = update_pr_branch(
            "42",
            repo="owner/repo",
            expected_head_sha="head_sha",
        )

    assert result["status"] == "update_requested"
    assert result["updated"] is True
    request_args = mock_gh.call_args_list[1].args[0]
    assert request_args[:5] == [
        "api",
        "repos/owner/repo/pulls/42/update-branch",
        "--method",
        "PUT",
        "-f",
    ]
    assert request_args[5] == "expected_head_sha=head_sha"


def test_update_pr_branch_does_not_request_on_head_mismatch() -> None:
    """A changed PR head must fail closed before the mutating REST call."""
    with patch("scripts.dev.update_pr_branch._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps({"head": {"sha": "new_head_sha"}}))
        result = update_pr_branch(
            "42",
            repo="owner/repo",
            expected_head_sha="old_head_sha",
        )

    assert result["status"] == "head_mismatch"
    assert result["updated"] is False
    mock_gh.assert_called_once_with(["api", "repos/owner/repo/pulls/42"])


def test_main_json_reports_guard_failure(capsys) -> None:  # type: ignore[no-untyped-def]
    """The CLI exposes a machine-readable non-mutating guard failure."""
    with patch("scripts.dev.update_pr_branch._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps({"head": {"sha": "new_head_sha"}}))
        rc = main(
            [
                "42",
                "--repo",
                "owner/repo",
                "--expected-head-sha",
                "old_head_sha",
                "--json",
            ]
        )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "head_mismatch"
    assert payload["updated"] is False


def test_gate_worktree_missing_fails_closed_before_update(capsys) -> None:  # type: ignore[no-untyped-def]
    """A missing gate worktree must fail closed before any branch update request."""
    missing_health = {
        "exists": False,
        "classification": "missing",
        "cleanup_owner": "owner=auto-smart-routing; pr=#5819; gate=gate-5819",
        "lease_owner": "auto-smart-routing",
        "lease_pr_number": 5819,
        "lease_gate_id": "gate-5819",
    }
    with patch(
        "scripts.dev.update_pr_branch._verify_gate_worktree", return_value=missing_health
    ) as mock_verify:
        rc = main(
            [
                "5819",
                "--repo",
                "owner/repo",
                "--expected-head-sha",
                "head_sha",
                "--gate-worktree-path",
                "/abs/missing-wt",
                "--json",
            ]
        )

    assert rc == 1
    mock_verify.assert_called_once_with("/abs/missing-wt")
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "gate_worktree_missing"
    assert payload["updated"] is False
    assert "5819" in (payload["gate_worktree_health"]["cleanup_owner"] or "")


def test_gate_worktree_present_allows_update(capsys) -> None:  # type: ignore[no-untyped-def]
    """A present gate worktree lets the normal guarded update proceed."""
    present_health = {"exists": True, "classification": "healthy", "cleanup_owner": None}
    with (
        patch("scripts.dev.update_pr_branch._verify_gate_worktree", return_value=present_health),
        patch("scripts.dev.update_pr_branch._gh") as mock_gh,
    ):
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps({"head": {"sha": "head_sha"}})),
            _gh_response(stdout=json.dumps({"message": "queued"})),
        ]
        rc = main(
            [
                "5819",
                "--repo",
                "owner/repo",
                "--expected-head-sha",
                "head_sha",
                "--gate-worktree-path",
                "/abs/present-wt",
                "--json",
            ]
        )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "update_requested"
    assert payload["updated"] is True
