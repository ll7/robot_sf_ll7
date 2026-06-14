"""Tests for drift-aware PR CI waiting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.watch_pr_ci_status import (
    DEFAULT_BASELINE_SECONDS,
    DEFAULT_MULTIPLIER,
    _duration_seconds,
    _parse_timestamp,
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


def test_once_returns_pending_without_sleep_or_drift_sampling() -> None:
    """One-shot mode should snapshot current status without waiting silently."""
    fetch_durations = MagicMock(return_value=[1000])
    sleep = MagicMock()

    result = watch_pr_ci_status(
        pr_number="42",
        fetch_status=MagicMock(return_value=_status("pending")),
        fetch_durations=fetch_durations,
        sleep=sleep,
        once=True,
    )

    assert result.final_status == "pending"
    assert result.drift_sample is None
    fetch_durations.assert_not_called()
    sleep.assert_not_called()


def test_progress_json_is_emitted_to_stream() -> None:
    """Long waits can emit compact progress evidence without poll-only silence."""
    now = iter([0.0, 0.0, 1.0, 1.0])
    stream = MagicMock()

    result = watch_pr_ci_status(
        pr_number="42",
        baseline_seconds=0,
        fetch_status=MagicMock(return_value=_status("pending")),
        fetch_durations=MagicMock(return_value=[]),
        monotonic=lambda: next(now),
        sleep=MagicMock(),
        emit_progress_json_every=1,
        progress_stream=stream,
    )

    assert result.final_status == "timeout"
    assert stream.write.call_count > 0
    written = "".join(str(call.args[0]) for call in stream.write.call_args_list)
    assert "pr_ci_watch_progress.v1" in written
    assert '"status": "pending"' in written


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


def test_main_once_pending_returns_timeout_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI one-shot pending status should return exit 2 with parseable JSON."""
    with patch("scripts.dev.watch_pr_ci_status._fetch_ci_status") as fetch_status:
        fetch_status.return_value = _status("pending")
        rc = main(["42", "--once", "--json"])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["final_status"] == "pending"


def test_parse_timestamp_accepts_any_and_returns_none_for_invalid() -> None:
    """_parse_timestamp should handle Any input gracefully."""
    assert _parse_timestamp(None) is None
    assert _parse_timestamp("") is None
    assert _parse_timestamp(42) is None
    assert _parse_timestamp("not-a-timestamp") is None
    assert _parse_timestamp("2026-06-01T00:00:00Z") is not None


def test_duration_seconds_handles_timezone_mismatch() -> None:
    """_duration_seconds should return None on tz-aware/naive TypeError."""
    result = _duration_seconds("2026-06-01T00:00:00+00:00", "2026-06-01T00:10:00")
    assert result is None


def test_terminal_state_returns_immediate_error() -> None:
    """CLOSED/MERGED PR state should return error without polling."""
    fetch_status = MagicMock(
        return_value={
            "status": "ok",
            "state": "CLOSED",
            "pr": 42,
            "head_sha": "abc123",
            "checks": {},
        }
    )
    result = watch_pr_ci_status(
        pr_number="42",
        fetch_status=fetch_status,
        fetch_durations=MagicMock(),
    )
    assert result.final_status == "error"
    assert "CLOSED" in result.error or "closed" in result.error


def test_main_valueerror_on_invalid_args(capsys: pytest.CaptureFixture[str]) -> None:
    """main should catch ValueError for invalid --multiplier."""
    rc = main(["42", "--multiplier", "-1"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "multiplier" in captured.err


def test_poll_interval_alias_sets_same_dest() -> None:
    """--poll-interval should write to poll_interval_seconds like --poll-interval-seconds."""
    from scripts.dev.watch_pr_ci_status import _parse_args

    args_alias = _parse_args(["42", "--poll-interval", "45"])
    args_long = _parse_args(["42", "--poll-interval-seconds", "45"])
    assert args_alias.poll_interval_seconds == 45
    assert args_long.poll_interval_seconds == 45
    assert args_alias.poll_interval_seconds == args_long.poll_interval_seconds


def test_budget_seconds_override_sets_budget_overridden() -> None:
    """--budget-seconds should bypass the baseline*multiplier computation."""
    result = watch_pr_ci_status(
        pr_number="42",
        budget_override_seconds=300,
        baseline_seconds=920,
        multiplier=1.3,
        fetch_status=MagicMock(return_value=_status("success")),
        fetch_durations=MagicMock(return_value=[1000]),
    )
    assert result.budget_seconds == 300
    assert result.budget_overridden is True
    assert result.baseline_seconds == 920
    assert result.multiplier == 1.3


def test_budget_seconds_zero_causes_immediate_timeout() -> None:
    """--budget-seconds 0 should exhaust budget on first poll and return timeout."""
    result = watch_pr_ci_status(
        pr_number="42",
        budget_override_seconds=0,
        baseline_seconds=920,
        multiplier=1.3,
        fetch_status=MagicMock(return_value=_status("pending")),
        fetch_durations=MagicMock(return_value=[800]),
    )
    assert result.final_status == "timeout"
    assert result.budget_seconds == 0
    assert result.budget_overridden is True


def test_budget_overridden_false_by_default() -> None:
    """When --budget-seconds is not set, budget_overridden should be False."""
    result = watch_pr_ci_status(
        pr_number="42",
        baseline_seconds=920,
        multiplier=1.3,
        fetch_status=MagicMock(return_value=_status("success")),
        fetch_durations=MagicMock(return_value=[1000]),
    )
    assert result.budget_overridden is False
    assert result.budget_seconds == 1196


def test_main_budget_seconds_cli_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --budget-seconds should pass through to watch result."""
    with patch("scripts.dev.watch_pr_ci_status._fetch_ci_status") as fetch_status:
        fetch_status.return_value = _status("success")
        rc = main(["42", "--budget-seconds", "500", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget_seconds"] == 500
    assert payload["budget_overridden"] is True


def test_main_poll_interval_alias_cli_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --poll-interval should be accepted alongside --poll-interval-seconds."""
    with patch("scripts.dev.watch_pr_ci_status._fetch_ci_status") as fetch_status:
        fetch_status.return_value = _status("success")
        rc = main(["42", "--poll-interval", "90", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["poll_interval_seconds"] == 90


def test_help_includes_expected_head_sha_example(capsys: pytest.CaptureFixture[str]) -> None:
    """--help should contain the SHA-guarded long-poll example."""
    from scripts.dev.watch_pr_ci_status import _parse_args

    with pytest.raises(SystemExit):
        _parse_args(["--help"])

    captured = capsys.readouterr()
    assert "--expected-head-sha" in captured.out
    assert "uv run python scripts/dev/watch_pr_ci_status.py" in captured.out
    assert "--json" in captured.out
    assert "long-poll" in captured.out.lower() or "long poll" in captured.out.lower()


def test_repo_root_on_sys_path() -> None:
    """The script should insert the repo root into sys.path before importing check_pr_ci_status."""
    from pathlib import Path

    from scripts.dev.watch_pr_ci_status import _REPO_ROOT

    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    assert _REPO_ROOT == repo_root
    assert _REPO_ROOT in sys.path


def test_direct_invocation_help_succeeds() -> None:
    """python scripts/dev/watch_pr_ci_status.py --help should succeed without PYTHONPATH."""
    import os
    import subprocess
    import sys as _sys

    script = str(
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dev" / "watch_pr_ci_status.py"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [_sys.executable, script, "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "--expected-head-sha" in result.stdout


def test_direct_invocation_import_without_pythonpath() -> None:
    """The module should be importable when the repo root is not explicitly on PYTHONPATH."""
    import importlib
    import sys as _sys

    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    paths_to_remove = [p for p in _sys.path if p == repo_root]
    for p in paths_to_remove:
        _sys.path.remove(p)

    saved_modules = {}
    for name in ("scripts.dev.watch_pr_ci_status", "scripts.dev.check_pr_ci_status"):
        if name in _sys.modules:
            saved_modules[name] = _sys.modules.pop(name)

    try:
        mod = importlib.import_module("scripts.dev.watch_pr_ci_status")
        assert hasattr(mod, "_REPO_ROOT")
        assert mod._REPO_ROOT == repo_root
    finally:
        _sys.modules.update(saved_modules)
        for p in paths_to_remove:
            if p not in _sys.path:
                _sys.path.insert(0, p)
