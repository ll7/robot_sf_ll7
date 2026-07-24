"""Offline tests for the REST-only label helper (issue #6266)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from scripts.dev.gh_pr_label_rest import add_label, main, remove_label


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for ``gh api``."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _mock_labels_payload(*names: str) -> str:
    """Build a JSON labels-array payload from label names."""
    return json.dumps([{"name": n} for n in names])


class TestAddLabel:
    """Tests for the add_label helper function."""

    def test_adds_label_via_rest_endpoint_and_verifies(self) -> None:
        """The helper must POST JSON labels[] and verify via re-read."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "cheap-lane"})),
                _proc(stdout=_mock_labels_payload("cheap-lane", "bug")),
            ]
            result = add_label(5220, "cheap-lane", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "cheap-lane",
            "action": "add",
            "repo": "ll7/robot_sf_ll7",
        }
        # First call: POST to add the label
        assert mock_run.call_args_list[0].args[0] == [
            "gh",
            "api",
            "--method",
            "POST",
            "repos/ll7/robot_sf_ll7/issues/5220/labels",
            "--input",
            "-",
        ]
        assert json.loads(mock_run.call_args_list[0].kwargs["input"]) == {"labels": ["cheap-lane"]}
        # Second call: GET to verify
        assert mock_run.call_args_list[1].args[0] == [
            "gh",
            "api",
            "repos/ll7/robot_sf_ll7/issues/5220/labels",
        ]

    def test_fails_closed_on_authentication_error(self) -> None:
        """Auth failures must be surfaced in the result."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            mock_post.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
            result = add_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "Bad credentials" in result["error"]

    def test_fails_closed_on_timeout(self) -> None:
        """A timeout must remain a structured error rather than escaping."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=30)
            result = add_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "timed out" in result["error"]
        assert "not verified" in result["error"]

    def test_fails_closed_when_post_write_verification_fails(self) -> None:
        """A successful POST is insufficient when the re-read lacks the label."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "cheap-lane"})),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            result = add_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "was not found in labels after add" in result["error"]

    def test_fails_closed_for_negative_number(self) -> None:
        """Zero or negative numbers must be rejected without network calls."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            result = add_label(0, "cheap-lane")

        assert result["status"] == "error"
        assert "must be positive" in result["error"]
        mock_post.assert_not_called()

    def test_fails_closed_for_empty_label(self) -> None:
        """Empty label strings must be rejected."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            result = add_label(5220, "")

        assert result["status"] == "error"
        assert "non-empty" in result["error"]
        mock_post.assert_not_called()


class TestRemoveLabel:
    """Tests for the remove_label helper function."""

    def test_removes_label_via_rest_endpoint_and_verifies(self) -> None:
        """The helper must DELETE the label endpoint and verify via re-read."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            result = remove_label(5220, "cheap-lane", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "cheap-lane",
            "action": "remove",
            "repo": "ll7/robot_sf_ll7",
        }
        # First call: DELETE the label
        assert mock_run.call_args_list[0].args[0] == [
            "gh",
            "api",
            "--method",
            "DELETE",
            "repos/ll7/robot_sf_ll7/issues/5220/labels/cheap-lane",
        ]
        # Second call: GET to verify
        assert mock_run.call_args_list[1].args[0] == [
            "gh",
            "api",
            "repos/ll7/robot_sf_ll7/issues/5220/labels",
        ]

    def test_fails_closed_on_authentication_error(self) -> None:
        """Auth failures must be surfaced in the result."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_del:
            mock_del.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
            result = remove_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "Bad credentials" in result["error"]

    def test_fails_closed_when_post_write_verification_fails(self) -> None:
        """A successful DELETE is insufficient when the re-read still has the label."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(stdout=_mock_labels_payload("cheap-lane", "bug")),
            ]
            result = remove_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "was still found" in result["error"]

    def test_fails_closed_for_negative_number(self) -> None:
        """Zero or negative numbers must be rejected without network calls."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_del:
            result = remove_label(0, "cheap-lane")

        assert result["status"] == "error"
        assert "must be positive" in result["error"]
        mock_del.assert_not_called()


class TestCli:
    """Tests for the CLI entry point."""

    def test_cli_add_prints_compact_success_json(self) -> None:
        """The command-line contract is a single machine-readable success result."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "cheap-lane"})),
                _proc(stdout=_mock_labels_payload("cheap-lane", "bug")),
            ]
            rc = main(["add", "5220", "--label", "cheap-lane", "--repo", "ll7/robot_sf_ll7"])

        assert rc == 0

    def test_cli_add_prints_error_json_to_stderr_on_failure(self, capsys) -> None:
        """A failed add must print JSON to stderr and exit 1."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            mock_post.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
            rc = main(["add", "5220", "--label", "cheap-lane"])

        captured = capsys.readouterr()
        assert rc == 1
        payload = json.loads(captured.err)
        assert payload["status"] == "error"
        assert "Bad credentials" in payload["error"]

    def test_cli_remove_prints_compact_success_json(self) -> None:
        """The CLI must also succeed for remove."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            rc = main(["remove", "5220", "--label", "cheap-lane", "--repo", "ll7/robot_sf_ll7"])

        assert rc == 0
