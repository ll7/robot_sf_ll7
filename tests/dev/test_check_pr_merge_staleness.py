"""Tests for the PR merge-staleness checker (deterministic, no live GitHub)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.check_pr_merge_staleness import (
    _detect_workflow_run_base_sha,
    _fetch_main_sha_with_retry,
    _fetch_pr_base_sha,
    _fetch_workflow_runs_with_retry,
    _gh_with_retry,
    _is_transient_gh_failure,
    _PermanentApiError,
    check_merge_staleness,
    format_human,
    main,
)


def _gh_response(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Create a mock subprocess.CompletedProcess for gh calls."""
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


# ── check_merge_staleness ────────────────────────────────────────────────────


def test_fetch_pr_base_sha_uses_rest_pull_endpoint() -> None:
    """Fetch the base SHA without the unsupported gh ``baseRefOid`` field."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.return_value = _gh_response(
            stdout=json.dumps({"base": {"sha": "base_sha"}, "number": 42})
        )
        base_sha, error = _fetch_pr_base_sha("42", repo="owner/repo")

    assert base_sha == "base_sha"
    assert error is None
    assert mock_gh.call_args.args[0] == ["api", "repos/owner/repo/pulls/42"]


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
    """Error status should propagate when fetching main SHA fails (permanent)."""
    with patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="not found")
        data = check_merge_staleness("1", base_sha="abc", repo="owner/repo")

    assert data["status"] == "error"
    assert data["stale"] is None
    assert data["detection"] == "error"
    assert "Failed to fetch" in data["error"] or "not found" in data["error"]


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
    head_args = mock_gh.call_args_list[0].args[0]
    assert head_args[:5] == ["pr", "view", "42", "--repo", "owner/repo"]
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
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep"),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps({"headRefName": "feature/x", "headRefOid": "head_sha"})),
            # Persistent transient failure exhausts the retry budget.
            _gh_response(returncode=1, stderr="API unavailable"),
            _gh_response(returncode=1, stderr="API unavailable"),
            _gh_response(returncode=1, stderr="API unavailable"),
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
            # REST pull request base SHA
            _gh_response(stdout=json.dumps({"base": {"sha": "same_sha"}})),
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
            _gh_response(stdout=json.dumps({"base": {"sha": "old_base"}})),
            _gh_response(stdout="new_main"),
        ]
        rc = main(["42"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "STALE" in captured.out


def test_main_exit_2_on_error(capsys: pytest.CaptureFixture) -> None:
    """main() should return 2 when the check itself errors (persistent transient)."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch(
            "scripts.dev.check_pr_merge_staleness._detect_workflow_run_base_sha", return_value=None
        ),
        patch("scripts.dev.check_pr_merge_staleness.time.sleep") as mock_sleep,
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="ll7/robot_sf_ll7"),
            _gh_response(stdout=json.dumps({"base": {"sha": "abc"}})),
            # Persistent transient failure: exhausts the retry budget.
            _gh_response(returncode=1, stderr="network error"),
            _gh_response(returncode=1, stderr="network error"),
            _gh_response(returncode=1, stderr="network error"),
            _gh_response(returncode=1, stderr="network error"),
        ]
        rc = main(["1"])

    assert rc == 2
    assert mock_sleep.call_count == 3
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
            _gh_response(stdout=json.dumps({"base": {"sha": "abc"}})),
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
            _gh_response(stdout=json.dumps({"base": {"sha": "sha"}})),
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
            _gh_response(stdout=json.dumps({"base": {"sha": "sha"}})),
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


# ── retry / backoff (issue #5958) ─────────────────────────────────────────────


def test_is_transient_gh_failure_classifies_transient() -> None:
    """429/5xx/timeout/connection failures are transient and worth retrying."""
    transient_cases = [
        _gh_response(returncode=1, stderr="HTTP 429: rate limit"),
        _gh_response(returncode=1, stderr="HTTP 503 Service Unavailable"),
        _gh_response(returncode=1, stderr="request timed out"),
        _gh_response(returncode=1, stderr="connection reset by peer"),
    ]
    for resp in transient_cases:
        assert _is_transient_gh_failure(resp) is True


def test_is_transient_gh_failure_classifies_permanent() -> None:
    """Auth / not-found failures are permanent and must not be retried."""
    permanent_cases = [
        _gh_response(returncode=1, stderr="HTTP 404 Not Found"),
        _gh_response(returncode=1, stderr="graphql: Could not resolve to a Repository"),
        _gh_response(returncode=1, stderr="HTTP 401 Unauthorized"),
    ]
    for resp in permanent_cases:
        assert _is_transient_gh_failure(resp) is False


def test_gh_with_retry_succeeds_after_transient_failure() -> None:
    """A transient failure followed by success returns the success without sleep."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep") as mock_sleep,
    ):
        mock_gh.side_effect = [
            _gh_response(returncode=1, stderr="HTTP 503"),
            _gh_response(stdout="main_head_sha"),
        ]
        result = _gh_with_retry(["api", "repos/o/r/git/refs/heads/main", "--jq", ".object.sha"])

    assert result.stdout.strip() == "main_head_sha"
    assert mock_sleep.call_count == 1


@pytest.mark.parametrize(
    ("stderr", "expected_kind"),
    [
        ("HTTP 404 Not Found", "not_found"),
        ("graphql: Could not resolve to a Repository", "permanent"),
        ("HTTP 403 Forbidden", "permanent"),
        ("HTTP 401 Unauthorized", "auth"),
    ],
)
def test_gh_with_retry_permanent_failure_raises_immediately(
    stderr: str, expected_kind: str
) -> None:
    """Permanent failures raise without any retry sleep."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep") as mock_sleep,
    ):
        mock_gh.return_value = _gh_response(returncode=1, stderr=stderr)
        with pytest.raises(_PermanentApiError) as excinfo:
            _gh_with_retry(["api", "repos/o/r/git/refs/heads/main"])

    assert excinfo.value.kind == expected_kind
    assert mock_gh.call_count == 1
    assert mock_sleep.call_count == 0


def test_gh_with_retry_persistent_transient_raises_unavailable() -> None:
    """Persistent transient failure exhausts the budget and raises unavailable."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep") as mock_sleep,
    ):
        mock_gh.return_value = _gh_response(returncode=1, stderr="HTTP 429 rate limit")
        with pytest.raises(_PermanentApiError) as excinfo:
            _gh_with_retry(["api", "repos/o/r/git/refs/heads/main"])

    assert excinfo.value.kind == "unavailable"
    # max_attempts=4 -> 3 sleeps before the final attempt fails.
    assert mock_sleep.call_count == 3


def test_gh_with_retry_backoff_grows() -> None:
    """Backoff should grow exponentially and be capped at the maximum."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep") as mock_sleep,
    ):
        mock_gh.return_value = _gh_response(returncode=1, stderr="HTTP 503")
        with pytest.raises(_PermanentApiError):
            _gh_with_retry(["api", "x"], max_attempts=4, backoff_base=1.0, max_backoff=8.0)

    sleeps = [c.args[0] for c in mock_sleep.call_args_list]
    assert sleeps == [1.0, 2.0, 4.0]


def test_fetch_main_sha_with_retry_returns_sha() -> None:
    """A transient hiccup resolved on retry still yields the main SHA."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep"),
    ):
        mock_gh.side_effect = [
            _gh_response(returncode=1, stderr="HTTP 503"),
            _gh_response(stdout="resolved_main_sha"),
        ]
        result = _fetch_main_sha_with_retry("owner/repo")

    assert result.sha == "resolved_main_sha"
    assert result.unavailable is False


def test_fetch_main_sha_with_retry_unavailable_on_persistent_failure() -> None:
    """Persistent transient failure returns an explicit unavailable result."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep"),
    ):
        mock_gh.return_value = _gh_response(returncode=1, stderr="HTTP 429 rate limit")
        result = _fetch_main_sha_with_retry("owner/repo")

    assert result.sha is None
    assert result.unavailable is True
    assert "unavailable" in (result.error or "")


def test_fetch_main_sha_with_retry_permanent_not_found() -> None:
    """A permanent not-found failure is reported without exhausting retries."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep") as mock_sleep,
    ):
        mock_gh.return_value = _gh_response(returncode=1, stderr="HTTP 404 Not Found")
        result = _fetch_main_sha_with_retry("owner/repo")

    assert result.sha is None
    assert result.unavailable is False
    assert mock_sleep.call_count == 0


def test_fetch_workflow_runs_with_retry_returns_none_on_persistent_failure() -> None:
    """Persistent transient failure returns None (fallback path), not a base SHA."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep"),
    ):
        mock_gh.return_value = _gh_response(returncode=1, stderr="HTTP 503")
        result = _fetch_workflow_runs_with_retry("owner/repo", "feature/x")

    assert result is None


def test_check_merge_staleness_error_on_unavailable_main_sha() -> None:
    """Persistent transient main-SHA failure must not infer freshness; error instead."""
    with (
        patch("scripts.dev.check_pr_merge_staleness._gh") as mock_gh,
        patch("scripts.dev.check_pr_merge_staleness.time.sleep"),
    ):
        mock_gh.return_value = _gh_response(returncode=1, stderr="HTTP 429 rate limit")
        data = check_merge_staleness("1", base_sha="abc", repo="owner/repo")

    assert data["status"] == "error"
    assert data["stale"] is None
    assert data["detection"] == "error"
    assert data["main_sha"] is None
    assert "persistent transient" in data["error"]
