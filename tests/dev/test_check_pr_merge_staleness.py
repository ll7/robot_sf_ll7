"""Tests for the PR merge-staleness checker (deterministic, no live GitHub)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.check_pr_merge_staleness import (
    _detect_workflow_run_base_sha,
    check_merge_staleness,
    format_human,
    main,
)


def _gh_response(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Create a mock subprocess.CompletedProcess for gh calls."""
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


# ── check_merge_staleness ────────────────────────────────────────────────────


def test_stale_when_main_sha_differs() -> None:
    """Result should be stale when current main differs from the PR base."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        # _get_main_sha
        mock_gh.side_effect = [
            _gh_response(stdout="main_head_sha"),
        ]
        data = check_merge_staleness("42", base_sha="old_base_sha", repo="owner/repo")

    assert data["status"] == "ok"
    assert data["stale"] is True
    assert data["detection"] == "base_vs_main"
    assert data["pr"] == "42"
    assert data["base_sha"] == "old_base_sha"
    assert data["main_sha"] == "main_head_sha"


def test_fresh_when_base_matches_main() -> None:
    """Result should be fresh when the PR base matches current main."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="same_sha"),
        ]
        data = check_merge_staleness("100", base_sha="same_sha", repo="owner/repo")

    assert data["stale"] is False
    assert data["detection"] == "base_vs_main"


def test_error_when_main_sha_fails() -> None:
    """Error status should propagate when fetching main SHA fails."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="not found")
        data = check_merge_staleness("1", base_sha="abc", repo="owner/repo")

    assert data["status"] == "error"
    assert data["stale"] is None
    assert data["detection"] == "error"
    assert "Failed to fetch current main SHA" in data["error"]


def test_main_sha_empty_string_treated_as_error() -> None:
    """An empty main SHA response should report an error."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout="")
        data = check_merge_staleness("1", base_sha="abc", repo="owner/repo")

    assert data["status"] == "error"


# ── workflow-run base provenance detection ───────────────────────────────────


def test_detect_workflow_run_base_sha_uses_branch_and_pr_base_metadata() -> None:
    """Use the current PR branch and the run's recorded base SHA."""
    runs_payload = {
        "workflow_runs": [
            {
                "status": "in_progress",
                "head_sha": "head_sha",
                "pull_requests": [{"number": 42, "base": {"sha": "in_progress_base"}}],
            },
            {
                "status": "completed",
                "head_sha": "old_head_sha",
                "pull_requests": [{"number": 42, "base": {"sha": "old_base"}}],
            },
            {
                "status": "completed",
                "head_sha": "head_sha",
                "pull_requests": [{"number": 42, "base": {"sha": "ci_base_sha"}}],
            },
        ]
    }
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps({"headRefName": "feature/x", "headRefOid": "head_sha"})),
            _gh_response(stdout=json.dumps(runs_payload)),
        ]
        result = _detect_workflow_run_base_sha("owner/repo", "42")

    assert result == "ci_base_sha"
    actions_args = mock_gh.call_args_list[1].args[0]
    assert "branch=feature/x" in actions_args
    assert "event=pull_request" in actions_args


def test_detect_workflow_run_base_sha_skips_other_prs() -> None:
    """A run for another PR must not be treated as this PR's CI provenance."""
    runs_payload = {
        "workflow_runs": [
            {
                "status": "completed",
                "head_sha": "head_sha",
                "pull_requests": [{"number": 99, "base": {"sha": "other_base"}}],
            }
        ]
    }
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps({"headRefName": "feature/x", "headRefOid": "head_sha"})),
            _gh_response(stdout=json.dumps(runs_payload)),
        ]
        result = _detect_workflow_run_base_sha("owner/repo", "42")

    assert result is None


def test_detect_workflow_run_base_sha_returns_none_on_api_error() -> None:
    """Actions API failures leave the caller on the documented fallback path."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps({"headRefName": "feature/x", "headRefOid": "head_sha"})),
            _gh_response(returncode=1, stderr="API unavailable"),
        ]
        result = _detect_workflow_run_base_sha("owner/repo", "42")

    assert result is None


def test_workflow_run_base_sha_fresh() -> None:
    """When the CI-tested base matches main, the PR is fresh."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha") as detect,
    ):
        detect.return_value = "current_main"
        mock_gh.side_effect = [
            # _get_main_sha
            _gh_response(stdout="current_main"),
        ]
        data = check_merge_staleness("42", base_sha="old_base", repo="owner/repo")

    assert data["stale"] is False
    assert data["detection"] == "workflow_run_base_sha"
    assert data["ci_base_sha"] == "current_main"


def test_workflow_run_base_sha_stale() -> None:
    """When the CI-tested base differs from main, the PR is stale."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha") as detect,
    ):
        detect.return_value = "old_ci_base"
        mock_gh.side_effect = [_gh_response(stdout="new_main")]
        data = check_merge_staleness("42", base_sha="old_base", repo="owner/repo")

    assert data["stale"] is True
    assert data["detection"] == "workflow_run_base_sha"
    assert data["ci_base_sha"] == "old_ci_base"
    assert data["main_sha"] == "new_main"


def test_workflow_run_fallback_when_base_provenance_is_unavailable() -> None:
    """Should fall back to base-vs-main when workflow provenance is unavailable."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha") as detect,
    ):
        detect.return_value = None
        mock_gh.side_effect = [_gh_response(stdout="current_main")]
        data = check_merge_staleness("42", base_sha="current_main", repo="owner/repo")

    assert data["detection"] == "base_vs_main"


def test_fallback_when_detect_returns_none() -> None:
    """Should fall back to base-vs-main when merge SHA detection returns None."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha") as detect,
    ):
        detect.return_value = None
        mock_gh.return_value = _gh_response(stdout="main_sha")
        data = check_merge_staleness("42", base_sha="main_sha", repo="owner/repo")

    assert data["detection"] == "base_vs_main"
    assert data["stale"] is False


# ── format_human ─────────────────────────────────────────────────────────────


def test_format_human_stale() -> None:
    """Stale result should include STALE verdict and detection method."""
    data = {
        "status": "ok",
        "stale": True,
        "detection": "workflow_run_base_sha",
        "pr": "42",
        "base_sha": "old",
        "main_sha": "new",
        "ci_base_sha": "old",
    }
    output = format_human(data)
    assert "STALE" in output
    assert "PR #42" in output
    assert "workflow_run_base_sha" in output
    assert "ci_base_sha: old" in output


def test_format_human_fresh() -> None:
    """Fresh result should include FRESH verdict."""
    data = {
        "status": "ok",
        "stale": False,
        "detection": "base_vs_main",
        "pr": "1",
        "base_sha": "abc",
        "main_sha": "abc",
    }
    output = format_human(data)
    assert "FRESH" in output


def test_format_human_unknown() -> None:
    """Unknown staleness should include UNKNOWN verdict."""
    data = {
        "status": "ok",
        "stale": None,
        "detection": "error",
        "pr": "1",
        "base_sha": "abc",
        "main_sha": None,
    }
    output = format_human(data)
    assert "UNKNOWN" in output


def test_format_human_error() -> None:
    """Error status should print the error message."""
    data = {"status": "error", "error": "API down"}
    output = format_human(data)
    assert "ERROR" in output
    assert "API down" in output


def test_format_human_shows_warning() -> None:
    """Warning field should appear in the human output."""
    data = {
        "status": "ok",
        "stale": False,
        "detection": "base_vs_main",
        "pr": "5",
        "base_sha": "sha",
        "main_sha": "sha",
        "warning": "Cannot detect workflow-run base provenance",
    }
    output = format_human(data)
    assert "warning: Cannot detect workflow-run base provenance" in output


# ── main() CLI ───────────────────────────────────────────────────────────────


def test_main_exit_0_when_fresh(capsys: pytest.CaptureFixture) -> None:
    """main() should return 0 when the PR is not stale."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        mock_gh.side_effect = [
            # repo view
            _gh_response(stdout="ll7/robot_sf_ll7"),
            # pr view base sha
            _gh_response(stdout="same_sha"),
            # main sha
            _gh_response(stdout="same_sha"),
        ]
        rc = main(["42"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "FRESH" in captured.out


def test_main_exit_1_when_stale(capsys: pytest.CaptureFixture) -> None:
    """main() should return 1 when the PR is stale."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="ll7/robot_sf_ll7"),
            _gh_response(stdout="old_base"),
            _gh_response(stdout="new_main"),
        ]
        rc = main(["42"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "STALE" in captured.out


def test_main_exit_2_on_error(capsys: pytest.CaptureFixture) -> None:
    """main() should return 2 when the check itself errors."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="ll7/robot_sf_ll7"),
            _gh_response(stdout="abc"),
            _gh_response(returncode=1, stderr="network error"),
        ]
        rc = main(["1"])

    assert rc == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.out


def test_main_json_output(capsys: pytest.CaptureFixture) -> None:
    """--json should emit machine-readable JSON."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="ll7/robot_sf_ll7"),
            _gh_response(stdout="abc"),
            _gh_response(stdout="abc"),
        ]
        rc = main(["7", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pr"] == "7"
    assert payload["stale"] is False
    assert payload["status"] == "ok"


def test_main_repo_flag(capsys: pytest.CaptureFixture) -> None:
    """--repo should skip the repo detection step."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="sha"),
            _gh_response(stdout="sha"),
        ]
        rc = main(["1", "--repo", "ll7/robot_sf_ll7"])

    assert rc == 0
    # Should have made exactly 2 gh calls (pr view, main sha), not 3.
    assert mock_gh.call_count == 2


def test_main_pr_flag_alias(capsys: pytest.CaptureFixture) -> None:
    """--pr should work as a named alias for the positional PR number."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="sha"),
            _gh_response(stdout="sha"),
        ]
        rc = main(["--pr", "99", "--repo", "ll7/robot_sf_ll7"])

    assert rc == 0


def test_main_rejects_conflicting_pr_inputs(capsys: pytest.CaptureFixture) -> None:
    """Conflicting positional and --pr should fail before invoking gh."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        with pytest.raises(SystemExit) as excinfo:
            main(["1", "--pr", "2"])

    assert excinfo.value.code == 2
    mock_gh.assert_not_called()


def test_main_requires_pr_number(capsys: pytest.CaptureFixture) -> None:
    """No PR number should fail with exit 2."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        with pytest.raises(SystemExit) as excinfo:
            main([])

    assert excinfo.value.code == 2
    mock_gh.assert_not_called()


def test_main_gh_not_installed() -> None:
    """main() should exit 1 when gh is not on PATH."""
    with patch(
        "scripts.dev.check_pr_merge_staleness._gh",
        side_effect=FileNotFoundError,
    ):
        rc = main(["42", "--repo", "ll7/robot_sf_ll7"])
    assert rc == 1


def test_main_gh_timeout() -> None:
    """main() should exit 1 when gh times out."""
    with patch(
        "scripts.dev.check_pr_merge_staleness._gh",
        side_effect=subprocess.TimeoutExpired(["gh"], timeout=30),
    ):
        rc = main(["42", "--repo", "ll7/robot_sf_ll7"])
    assert rc == 1


def test_main_repo_detect_failure(capsys: pytest.CaptureFixture) -> None:
    """Repository detection failure should exit 1."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="auth error")
        rc = main(["42"])

    assert rc == 1
    assert "Failed to detect repository" in capsys.readouterr().err
