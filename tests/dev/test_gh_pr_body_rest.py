"""Offline tests for the REST-only PR body updater (issue #5221)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from scripts.dev.gh_pr_body_rest import main, update_pr_body

if TYPE_CHECKING:
    from pathlib import Path


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for ``gh api``."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def test_update_pr_body_patches_rest_endpoint_and_verifies_body(tmp_path: Path) -> None:
    """The helper must send JSON through PATCH and verify GitHub's returned body."""
    body_file = tmp_path / "body.md"
    body_file.write_text("## Summary\n\nUpdated body\n", encoding="utf-8")
    response = {
        "body": body_file.read_text(encoding="utf-8"),
        "html_url": "https://example/pr/5220",
    }
    with patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run:
        mock_run.return_value = _proc(stdout=json.dumps(response))
        result = update_pr_body(5220, body_file, repo="ll7/robot_sf_ll7")

    assert result == {
        "status": "ok",
        "number": 5220,
        "repo": "ll7/robot_sf_ll7",
        "url": "https://example/pr/5220",
    }
    assert mock_run.call_args.args[0] == [
        "gh",
        "api",
        "--method",
        "PATCH",
        "repos/ll7/robot_sf_ll7/pulls/5220",
        "--input",
        "-",
    ]
    assert json.loads(mock_run.call_args.kwargs["input"]) == {"body": response["body"]}


def test_update_pr_body_fails_closed_when_response_body_differs(tmp_path: Path) -> None:
    """A successful HTTP response is insufficient when it reports a different body."""
    body_file = tmp_path / "body.md"
    body_file.write_text("expected", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        mock_patch.return_value = _proc(stdout=json.dumps({"body": "different"}))
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "did not preserve" in result["error"]


def test_update_pr_body_fails_closed_for_missing_file(tmp_path: Path) -> None:
    """A missing body file must never issue a partial REST update."""
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        result = update_pr_body(5220, tmp_path / "missing.md")

    assert result["status"] == "error"
    assert "could not read body file" in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_fails_closed_on_api_error(tmp_path: Path) -> None:
    """Authentication and API failures must be visible to the caller."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        mock_patch.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "Bad credentials" in result["error"]


def test_cli_prints_compact_success_json(tmp_path: Path, capsys) -> None:
    """The command-line contract is a single machine-readable success result."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        mock_patch.return_value = _proc(
            stdout=json.dumps({"body": "body", "html_url": "https://example/pr/5220"})
        )
        rc = main(["5220", "--repo", "ll7/robot_sf_ll7", "--body-file", str(body_file)])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["url"] == "https://example/pr/5220"
