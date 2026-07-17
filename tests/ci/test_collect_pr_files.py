"""Offline tests for the retrying PR changed-files collector (issue #5918).

These tests pin the two behaviors the issue requires:

* a transient GitHub API 502/503/504/429 (which arrives as an HTML body that
  cannot be JSON-parsed -- the original ``invalid character '<'`` failure) must
  be retried rather than red the job; and
* a *persistent* API failure must still fail closed with a single actionable
  reason that is clearly distinguishable from a body/contract validation failure.

No real ``gh`` subprocess or wall-clock sleep runs: the page fetcher and the
sleep seam are injected.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from scripts.ci.collect_pr_files import (
    PrFilesFetchError,
    _backoff_delay,
    fetch_all_pr_files,
    main,
    write_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for a single ``gh api`` page."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _entry(filename: str, status: str = "modified") -> dict[str, str]:
    """One files-endpoint entry as gh would return it."""
    return {"filename": filename, "status": status, "sha": "abc"}


def _scripted_runner(
    responses: dict[tuple[int, int], MagicMock | Exception],
) -> Callable[..., MagicMock]:
    """Return a run_page callable keyed by ``(page, attempt_within_page)``.

    ``attempt_within_page`` resets to 1 for each new page (the retry loop counts
    attempts per page). An ``Exception`` value is raised to model timeouts / OS
    errors; a ``MagicMock`` (CompletedProcess) is returned as-is.
    """
    seen: dict[int, int] = {}

    def run_page(repo, pr_number, page, per_page, *, timeout):
        attempt = seen.get(page, 0) + 1
        seen[page] = attempt
        value = responses[(page, attempt)]
        if isinstance(value, Exception):
            raise value
        return value

    return run_page


HTML_503 = (
    "<html><head><title>503 Server Error</title></head><body>Server Error for GH API</body></html>"
)


# ── retry-then-success (the core acceptance) ─────────────────────────────────


def test_transient_html_503_is_retried_then_succeeds() -> None:
    """The exact issue #5918 symptom (HTML 503 body) must be retried, not red."""
    good_page1 = json.dumps([_entry("a.py", "added"), _entry("b.md", "modified")])
    run_page = _scripted_runner(
        {
            (1, 1): _proc(stdout=HTML_503, returncode=0),  # HTML body, rc 0
            (1, 2): _proc(stdout=good_page1),  # success on retry
            (2, 1): _proc(stdout="[]"),  # empty page ends pagination
        }
    )
    sleeps: list[float] = []

    files = fetch_all_pr_files(
        "owner/repo", "123", run_page=run_page, sleep=sleeps.append, rng=lambda: 0.5
    )

    assert [e["filename"] for e in files] == ["a.py", "b.md"]
    # A backoff sleep happened between the failed attempt and the successful retry.
    assert sleeps == [pytest.approx(1.0 * 0.5)]  # base_delay(1.0) * 2**0 * jitter(0.5)


def test_nonzero_exit_is_retried_then_succeeds() -> None:
    """A non-zero gh exit (recognised HTTP error) is also retryable."""
    run_page = _scripted_runner(
        {
            (1, 1): _proc(returncode=1, stderr="HTTP 503: Service Unavailable"),
            (1, 2): _proc(stdout=json.dumps([_entry("x.py", "modified")])),
            (2, 1): _proc(stdout="[]"),
        }
    )
    files = fetch_all_pr_files(
        "owner/repo", "1", run_page=run_page, sleep=lambda _d: None, rng=lambda: 0.0
    )
    assert [e["filename"] for e in files] == ["x.py"]


def test_timeout_is_treated_as_transient_and_retried() -> None:
    """A per-page timeout is a transient failure that is retried, then can succeed."""
    run_page = _scripted_runner(
        {
            (1, 1): subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=30),
            (1, 2): _proc(stdout=json.dumps([_entry("ok.py", "modified")])),
            (2, 1): _proc(stdout="[]"),
        }
    )
    files = fetch_all_pr_files(
        "o/r", "7", run_page=run_page, sleep=lambda _d: None, rng=lambda: 0.0
    )
    assert [e["filename"] for e in files] == ["ok.py"]


# ── persistent failure fails closed (the core acceptance) ────────────────────


def test_persistent_html_failure_raises_with_actionable_diagnostic() -> None:
    """Exhausting retries on HTML error pages must fail closed with a clear reason."""
    run_page = _scripted_runner(
        {
            (1, 1): _proc(stdout=HTML_503),
            (1, 2): _proc(stdout=HTML_503),
            (1, 3): _proc(stdout=HTML_503),
        }
    )

    with pytest.raises(PrFilesFetchError) as exc_info:
        fetch_all_pr_files(
            "owner/repo",
            "5918",
            max_attempts=3,
            run_page=run_page,
            sleep=lambda _d: None,
            rng=lambda: 0.0,
        )

    message = str(exc_info.value)
    assert "repos/owner/repo/pulls/5918/files" in message
    assert "after 3 attempt(s)" in message
    assert "transient GitHub API outage" in message
    assert "NOT a PR body/contract validation failure" in message
    assert "non-JSON response" in message  # correlates with logs


def test_persistent_nonzero_exit_fails_closed() -> None:
    """A persistent non-zero exit also fails closed after retries are spent."""
    attempts = {"n": 0}

    def run_page(*args, **kwargs):
        attempts["n"] += 1
        return _proc(returncode=42, stderr="HTTP 502: Bad Gateway")

    with pytest.raises(PrFilesFetchError) as exc_info:
        fetch_all_pr_files(
            "o/r", "9", max_attempts=2, run_page=run_page, sleep=lambda _d: None, rng=lambda: 0.0
        )
    assert "gh api exited with code 42" in str(exc_info.value)
    assert attempts["n"] == 2  # did not exceed the per-page attempt budget


def test_no_sleep_after_final_attempt() -> None:
    """Backoff must not sleep after the last permitted attempt."""
    sleeps: list[float] = []
    run_page = _scripted_runner(
        {
            (1, 1): _proc(stdout=HTML_503),
            (1, 2): _proc(stdout=HTML_503),
            (1, 3): _proc(stdout=HTML_503),
        }
    )

    with pytest.raises(PrFilesFetchError):
        fetch_all_pr_files(
            "o/r", "1", max_attempts=3, run_page=run_page, sleep=sleeps.append, rng=lambda: 1.0
        )
    # 3 attempts => exactly 2 inter-attempt sleeps (none after the final attempt).
    assert len(sleeps) == 2


# ── pagination ───────────────────────────────────────────────────────────────


def test_pagination_collects_all_pages() -> None:
    """Full pages continue until a partial/empty page ends collection."""
    full_page = [_entry(f"a{i}.py", "modified") for i in range(100)]
    last_page = [_entry("b.py", "added"), _entry("c.py", "modified")]
    run_page = _scripted_runner(
        {
            (1, 1): _proc(stdout=json.dumps(full_page)),
            (2, 1): _proc(stdout=json.dumps(last_page)),
            (3, 1): _proc(stdout="[]"),
        }
    )
    files = fetch_all_pr_files(
        "o/r", "1", per_page=100, run_page=run_page, sleep=lambda _d: None, rng=lambda: 0.0
    )
    assert len(files) == 102
    assert files[-1]["filename"] == "c.py"


# ── outputs: changed / status TSV / added ────────────────────────────────────


def test_write_outputs_separates_changed_added_and_status(tmp_path: Path) -> None:
    """The three output files carry changed names, status TSV, and added names."""
    files = [
        _entry("new.py", "added"),
        _entry("touched.py", "modified"),
        _entry("gone.py", "removed"),
    ]
    changed = tmp_path / "changed.txt"
    status = tmp_path / "status.tsv"
    added = tmp_path / "added.txt"

    write_outputs(
        files,
        out_changed_files=str(changed),
        out_files_status=str(status),
        out_added_files=str(added),
    )

    assert changed.read_text(encoding="utf-8").splitlines() == ["new.py", "touched.py", "gone.py"]
    assert added.read_text(encoding="utf-8").splitlines() == ["new.py"]
    assert status.read_text(encoding="utf-8").splitlines() == [
        "added\tnew.py",
        "modified\ttouched.py",
        "removed\tgone.py",
    ]


# ── backoff math ─────────────────────────────────────────────────────────────


def test_backoff_delay_grows_and_is_jittered_and_capped() -> None:
    """Backoff cap doubles per attempt, is jittered in [0, cap], and capped."""
    max_rng = lambda: 1.0  # noqa: E731  delay == cap
    zero_rng = lambda: 0.0  # noqa: E731  delay == 0 (full jitter lower bound)
    assert _backoff_delay(1, base_delay=1.0, max_delay=30.0, rng=max_rng) == 1.0
    assert _backoff_delay(2, base_delay=1.0, max_delay=30.0, rng=max_rng) == 2.0
    assert _backoff_delay(3, base_delay=1.0, max_delay=30.0, rng=max_rng) == 4.0
    # Capped at max_delay rather than growing without bound.
    assert _backoff_delay(10, base_delay=1.0, max_delay=30.0, rng=max_rng) == 30.0
    # Full jitter: rng=0 => zero sleep.
    assert _backoff_delay(5, base_delay=1.0, max_delay=30.0, rng=zero_rng) == 0.0


# ── CLI entry point ──────────────────────────────────────────────────────────


def test_cli_writes_outputs_and_exits_zero(tmp_path: Path) -> None:
    """A successful collection writes the requested files and exits 0."""
    changed = tmp_path / "changed.txt"
    import scripts.ci.collect_pr_files as mod

    orig_run, orig_sleep = mod._run_gh_api_page, mod.time.sleep
    mod._run_gh_api_page = _scripted_runner(
        {  # type: ignore[assignment]
            (1, 1): _proc(stdout=json.dumps([_entry("z.py", "added")])),
            (2, 1): _proc(stdout="[]"),
        }
    )
    mod.time.sleep = lambda _d: None  # type: ignore[assignment,method-assign]
    try:
        rc = main(["--repo", "o/r", "--pr-number", "5", "--out-changed-files", str(changed)])
    finally:
        mod._run_gh_api_page, mod.time.sleep = orig_run, orig_sleep  # type: ignore[assignment,method-assign]

    assert rc == 0
    assert changed.read_text(encoding="utf-8").splitlines() == ["z.py"]


def test_cli_requires_at_least_one_output() -> None:
    """Without any --out-* the helper refuses to do work (fail-closed contract)."""
    rc = main(["--repo", "o/r", "--pr-number", "5"])
    assert rc == 2


def test_cli_fails_closed_on_persistent_failure(tmp_path: Path, capsys) -> None:
    """Persistent failure exits 1, the diagnostic goes to stderr, no partial file."""
    changed = tmp_path / "changed.txt"
    import scripts.ci.collect_pr_files as mod

    orig_run, orig_sleep = mod._run_gh_api_page, mod.time.sleep
    mod._run_gh_api_page = lambda *a, **k: _proc(stdout=HTML_503)  # type: ignore[assignment]
    mod.time.sleep = lambda _d: None  # type: ignore[assignment,method-assign]
    try:
        rc = main(["--repo", "o/r", "--pr-number", "5", "--out-changed-files", str(changed)])
    finally:
        mod._run_gh_api_page, mod.time.sleep = orig_run, orig_sleep  # type: ignore[assignment,method-assign]

    assert rc == 1
    err = capsys.readouterr().err
    assert "Failed to collect changed files" in err
    assert "NOT a PR body/contract validation failure" in err
    # No partial output file is written on failure.
    assert not changed.exists()
