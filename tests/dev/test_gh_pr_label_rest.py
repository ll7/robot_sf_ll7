"""Offline tests for the REST-only label helper (issue #6266)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from scripts.dev.gh_pr_label_rest import (
    LABEL_PAGE_CEILING,
    LABEL_PAGE_SIZE,
    _get_label_names,
    add_label,
    get_label_names,
    main,
    remove_label,
)


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for ``gh api``."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _mock_labels_payload(*names: str) -> str:
    """Build a JSON labels-array payload from label names."""
    return json.dumps([{"name": n} for n in names])


def _page_path(page: int) -> str:
    return f"repos/ll7/robot_sf_ll7/issues/5220/labels?per_page=100&page={page}"


class TestLabelRead:
    """Tests for complete and strict paginated label reads."""

    def test_finds_label_on_second_page(self) -> None:
        page_one = [{"name": f"label-{index}"} for index in range(LABEL_PAGE_SIZE)]
        with patch("scripts.dev.gh_pr_label_rest._gh_api_get") as mock_get:
            mock_get.side_effect = [
                _proc(stdout=json.dumps(page_one)),
                _proc(stdout=_mock_labels_payload("target")),
            ]
            result = _get_label_names(5220, repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "labels": [f"label-{i}" for i in range(100)] + ["target"],
        }
        assert [call.args[0] for call in mock_get.call_args_list] == [
            _page_path(1),
            _page_path(2),
        ]

    def test_rejects_malformed_page_and_row(self) -> None:
        for payload in (
            {"name": "not-a-page"},
            [{"name": ""}],
            [{"name": None}],
            [{"name": 42}],
            ["not-a-row"],
        ):
            with patch(
                "scripts.dev.gh_pr_label_rest._gh_api_get",
                return_value=_proc(stdout=json.dumps(payload)),
            ):
                result = _get_label_names(5220)

            assert result["status"] == "error"
            assert "page 1" in result["error"]
            assert "labels" not in result

    def test_fails_closed_when_page_fetch_fails(self) -> None:
        page_one = [{"name": f"label-{index}"} for index in range(LABEL_PAGE_SIZE)]
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
        ) as mock_get:
            mock_get.side_effect = [
                _proc(stdout=json.dumps(page_one)),
                _proc(returncode=1, stderr="HTTP 503: unavailable"),
            ]
            result = _get_label_names(5220)

        assert result == {
            "status": "error",
            "error": "could not read labels page 2: HTTP 503: unavailable",
        }

    def test_fails_closed_at_page_ceiling(self) -> None:
        full_page = _mock_labels_payload(*[f"label-{index}" for index in range(LABEL_PAGE_SIZE)])
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
            return_value=_proc(stdout=full_page),
        ) as mock_get:
            result = _get_label_names(5220)

        assert result["status"] == "error"
        assert str(LABEL_PAGE_CEILING) in result["error"]
        assert "labels" not in result
        assert mock_get.call_count == LABEL_PAGE_CEILING

    def test_public_read_api_rejects_nonpositive_number(self) -> None:
        with patch("scripts.dev.gh_pr_label_rest._gh_api_get") as mock_get:
            result = get_label_names(0)

        assert result["status"] == "error"
        assert "must be positive" in result["error"]
        mock_get.assert_not_called()


class TestAddLabel:
    """Tests for the add_label helper function."""

    def test_merge_ready_requires_matching_open_head(self) -> None:
        """The merge-ready write performs the exact-head preflight first."""
        head_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
        base_sha = "b1c2d3e4f5061728394a5b6c7d8e9f0011121314"
        with (
            patch(
                "scripts.dev.gh_pr_label_rest.guard_pr_write",
                return_value={
                    "status": "ok",
                    "observed_head_sha": head_sha,
                    "observed_base_sha": base_sha,
                },
            ) as mock_guard,
            patch(
                "scripts.dev.gh_pr_label_rest.check_merge_ready_carriers",
                return_value={"status": "ok"},
            ) as mock_carriers,
            patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "merge-ready"})),
                _proc(stdout=_mock_labels_payload("merge-ready")),
            ]
            result = add_label(
                5220,
                "merge-ready",
                repo="ll7/robot_sf_ll7",
                expected_head_sha=head_sha,
                expected_base_sha=base_sha,
            )

        assert result["status"] == "ok"
        mock_guard.assert_called_once_with(
            5220,
            repo="ll7/robot_sf_ll7",
            expected_head_sha=head_sha,
            expected_base_sha=base_sha,
            operation="merge_ready_label",
        )
        mock_carriers.assert_called_once_with(
            5220,
            repo="ll7/robot_sf_ll7",
            live_head=head_sha,
            live_base=base_sha,
        )

    def test_merge_ready_withholds_write_when_carrier_gate_fails(self) -> None:
        """A stale carrier (e.g. pending domain review) blocks the label write."""
        head_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
        with (
            patch(
                "scripts.dev.gh_pr_label_rest.guard_pr_write",
                return_value={
                    "status": "ok",
                    "observed_head_sha": head_sha,
                    "observed_base_sha": "b1c2d3e4f5061728394a5b6c7d8e9f0011121314",
                },
            ),
            patch(
                "scripts.dev.gh_pr_label_rest.check_merge_ready_carriers",
                return_value={
                    "status": "error",
                    "error": "review comment carries stale-carrier sentinel(s)",
                },
            ),
            patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post,
        ):
            result = add_label(
                5220,
                "merge-ready",
                expected_head_sha=head_sha,
            )

        assert result["status"] == "error"
        assert "stale-carrier sentinel" in result["error"]
        mock_post.assert_not_called()

    def test_merge_ready_stale_state_skips_post(self) -> None:
        """A merged or moved PR must not receive a merge-ready label write."""
        stale = {
            "status": "review_skipped_stale_state",
            "reason": "pr_not_open",
            "observed_state": "MERGED",
        }
        with (
            patch("scripts.dev.gh_pr_label_rest.guard_pr_write", return_value=stale),
            patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post,
        ):
            result = add_label(
                5220,
                "merge-ready",
                expected_head_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9001020304",
            )

        assert result == stale
        mock_post.assert_not_called()

    def test_adds_label_via_rest_endpoint_and_verifies(self) -> None:
        """The helper must POST JSON labels[] and verify via re-read."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "state:running"})),
                _proc(stdout=_mock_labels_payload("state:running", "bug")),
            ]
            result = add_label(5220, "state:running", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "state:running",
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
        assert json.loads(mock_run.call_args_list[0].kwargs["input"]) == {
            "labels": ["state:running"]
        }
        # Second call: GET to verify
        assert mock_run.call_args_list[1].args[0] == [
            "gh",
            "api",
            _page_path(1),
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
            result = remove_label(5220, "state:running", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "state:running",
            "action": "remove",
            "repo": "ll7/robot_sf_ll7",
        }
        # First call: DELETE the label
        assert mock_run.call_args_list[0].args[0] == [
            "gh",
            "api",
            "--method",
            "DELETE",
            "repos/ll7/robot_sf_ll7/issues/5220/labels/state%3Arunning",
        ]
        # Second call: GET to verify
        assert mock_run.call_args_list[1].args[0] == [
            "gh",
            "api",
            _page_path(1),
        ]

    def test_fails_closed_on_authentication_error(self) -> None:
        """Auth failures must be surfaced in the result."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_del:
            mock_del.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
            result = remove_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "Bad credentials" in result["error"]

    def test_treats_concurrent_absent_label_delete_as_idempotent(self) -> None:
        """A verified already-absent label is a successful remove outcome."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(returncode=1, stderr="gh: Label does not exist (HTTP 404)"),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            result = remove_label(5220, "state:running", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "state:running",
            "action": "remove",
            "repo": "ll7/robot_sf_ll7",
            "idempotent": True,
        }

    def test_fails_closed_when_absent_delete_readback_still_has_label(self) -> None:
        """The narrow 404 is not success when authoritative readback disagrees."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(returncode=1, stderr="gh: Label does not exist (HTTP 404)"),
                _proc(stdout=_mock_labels_payload("state:running", "bug")),
            ]
            result = remove_label(5220, "state:running")

        assert result["status"] == "error"
        assert "was still found" in result["error"]

    def test_fails_closed_on_unrelated_not_found_error(self) -> None:
        """A generic 404 must not be mistaken for the absent-label race."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_del:
            mock_del.return_value = _proc(returncode=1, stderr="gh: Not Found (HTTP 404)")
            result = remove_label(5220, "state:running")

        assert result["status"] == "error"
        assert "Not Found" in result["error"]

    def test_fails_closed_when_absent_delete_readback_fails(self) -> None:
        """An idempotent response still requires a successful labels readback."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(returncode=1, stderr="gh: Label does not exist (HTTP 404)"),
                _proc(returncode=1, stderr="HTTP 401: Bad credentials"),
            ]
            result = remove_label(5220, "state:running")

        assert result["status"] == "error"
        assert "could not read labels" in result["error"]

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

    def test_cli_list_prints_compact_label_inventory(self, capsys) -> None:
        """The list command exposes the strict REST read for shell workflows."""
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
            return_value=_proc(stdout=_mock_labels_payload("state:ready", "bug")),
        ):
            rc = main(["list", "5220", "--repo", "ll7/robot_sf_ll7"])

        captured = capsys.readouterr()
        assert rc == 0
        assert json.loads(captured.out) == {
            "action": "list",
            "labels": ["state:ready", "bug"],
            "number": 5220,
            "repo": "ll7/robot_sf_ll7",
            "status": "ok",
        }

    def test_cli_list_prints_read_error_to_stderr(self, capsys) -> None:
        """A failed label read remains an observable nonzero CLI result."""
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
            return_value=_proc(returncode=1, stderr="HTTP 403: forbidden"),
        ):
            rc = main(["list", "5220"])

        captured = capsys.readouterr()
        assert rc == 1
        payload = json.loads(captured.err)
        assert payload["status"] == "error"
        assert "forbidden" in payload["error"]

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
