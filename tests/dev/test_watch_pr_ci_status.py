"""Tests for drift-aware PR CI waiting."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.watch_pr_ci_status import (
    DEFAULT_BASELINE_SECONDS,
    DEFAULT_MULTIPLIER,
    fetch_recent_successful_ci_durations,
    main,
    wait_budget_seconds,
    watch_pr_ci_status,
)


def _status(overall: str, *, head_sha: str = "abc123") -> dict[str, object]:
    """Return a compact check status payload."""
    return {
        "status": "ok",
        "pr": 42,
        "head_sha": head_sha,
        "checks": {
            "overall": overall,
            "total": 1,
            "by_conclusion": {overall: 1},
            "by_status": {"completed": 1} if overall != "pending" else {"in_progress": 1},
            "names": ["ci"],
        },
    }


def test_default_wait_budget_uses_stable_baseline() -> None:
    """The default budget should match the documented 920s baseline with 30% headroom."""
    assert wait_budget_seconds(DEFAULT_BASELINE_SECONDS, DEFAULT_MULTIPLIER) == 1196


def test_success_does_not_fetch_recent_runtime_samples() -> None:
    """Normal successful waits should avoid refreshing CI timing evidence."""
    fetch_durations = MagicMock(return_value=[1000])

    result = watch_pr_ci_status(
        pr_number="42",
        expected_head_sha="abc123",
        fetch_status=MagicMock(return_value=_status("success")),
        fetch_durations=fetch_durations,
    )

    assert result.final_status == "success"
    assert result.drift_sample is None
    fetch_durations.assert_not_called()


def test_failure_returns_without_drift_sampling() -> None:
    """Failed CI is a terminal state, not runtime drift evidence."""
    fetch_durations = MagicMock(return_value=[1000])

    result = watch_pr_ci_status(
        pr_number="42",
        fetch_status=MagicMock(return_value=_status("failure")),
        fetch_durations=fetch_durations,
    )

    assert result.final_status == "failure"
    fetch_durations.assert_not_called()


def test_stale_head_sha_returns_error() -> None:
    """A branch update while waiting should invalidate the monitor result."""
    result = watch_pr_ci_status(
        pr_number="42",
        expected_head_sha="expected",
        fetch_status=MagicMock(return_value=_status("pending", head_sha="new-head")),
        fetch_durations=MagicMock(return_value=[1000]),
    )

    assert result.final_status == "error"
    assert "head SHA changed" in result.error


def test_pending_after_budget_collects_drift_sample() -> None:
    """Runtime sampling should happen only after the stable wait budget is exhausted."""
    fetch_durations = MagicMock(return_value=[800, 1000, 1200])

    result = watch_pr_ci_status(
        pr_number="42",
        baseline_seconds=0,
        poll_interval_seconds=1,
        fetch_status=MagicMock(return_value=_status("pending")),
        fetch_durations=fetch_durations,
    )

    assert result.final_status == "timeout"
    assert result.drift_sample is not None
    assert result.drift_sample.sample_count == 3
    assert result.drift_sample.median_seconds == 1000
    assert result.drift_sample.recommended_budget_seconds == 1300
    fetch_durations.assert_called_once()


def test_fetch_recent_successful_ci_durations_parses_gh_run_list() -> None:
    """The drift sampler should use run-list timestamps without inspecting every run."""
    payload = json.dumps(
        [
            {
                "databaseId": 1,
                "displayTitle": "ok",
                "status": "completed",
                "conclusion": "success",
                "createdAt": "2026-06-01T00:00:00Z",
                "updatedAt": "2026-06-01T00:10:00Z",
            },
            {
                "databaseId": 2,
                "displayTitle": "failed",
                "status": "completed",
                "conclusion": "failure",
                "createdAt": "2026-06-01T00:00:00Z",
                "updatedAt": "2026-06-01T00:20:00Z",
            },
        ]
    )
    with patch("scripts.dev.watch_pr_ci_status._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=payload, stderr="")

        durations = fetch_recent_successful_ci_durations(workflow="CI", limit=2)

    assert durations == [600]


def test_main_returns_timeout_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI should map pending-over-budget to exit 2."""
    with patch("scripts.dev.watch_pr_ci_status._fetch_ci_status") as fetch_status:
        fetch_status.return_value = _status("pending")
        with patch(
            "scripts.dev.watch_pr_ci_status.fetch_recent_successful_ci_durations"
        ) as samples:
            samples.return_value = [1000]
            rc = main(["42", "--baseline-seconds", "0", "--json"])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["final_status"] == "timeout"
