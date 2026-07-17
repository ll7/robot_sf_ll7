#!/usr/bin/env python3
"""Collect a PR's changed files from the GitHub API with bounded retry/backoff.

Why this exists
---------------
Both the ``PR Contract Check`` and ``PR body contracts`` workflows used an inline
``gh api --paginate "repos/$REPO/pulls/$PR/files?per_page=100"`` call to gather
the changed-file list. When GitHub returns a transient HTML 502/503/504/429 page,
the CLI writes that HTML to stdout, the ``--jq`` filter fails with
``invalid character '<' looking for beginning of value``, and the whole job exits
red *before* its real checker ever runs (issue #5918). A transient GitHub API
blip must not look like a body/contract failure.

This helper is the shared retrying collector used by both workflows. It pages the
``repos/{owner}/{repo}/pulls/{number}/files`` endpoint one page at a time so each
page can be retried independently. HTTP 429/5xx responses, connection failures,
and non-JSON HTML error pages are retried with exponential delay + jitter;
permanent failures such as HTTP 401/403/404 fail immediately. Every failure
fails closed with a clear, actionable reason.

Contract
--------
- Exits ``0`` and writes the requested output files only after the *full* file
  list has been collected; a partial/failed run never leaves a misleading
  truncated file behind.
- Exits ``1`` and prints a single diagnostic to stderr on an immediate permanent
  failure or after transient retries are exhausted -- the workflows surface
  this verbatim.

Usage
-----
::

    uv run python scripts/ci/collect_pr_files.py \\
        --repo "$REPO" --pr-number "$PR_NUMBER" \\
        --out-changed-files "$RUNNER_TEMP/pr_changed_files.txt" \\
        --out-files-status "$RUNNER_TEMP/pr_files_status.tsv" \\
        --out-added-files "$RUNNER_TEMP/pr_added_files.txt"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_PER_PAGE = 100
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_TIMEOUT = 30

# Retry only failures that are genuinely transient: HTTP 429/5xx, timeouts or
# connection resets, and the HTML error-page response from issue #5918. In
# particular, authentication/authorization/not-found responses are permanent
# and must fail immediately rather than consuming the retry budget.

_CONNECTION_FAILURE_MARKERS = (
    "connection reset",
    "connection aborted",
    "connection refused",
    "unexpected eof",
    "tls handshake timeout",
    "i/o timeout",
)


class PrFilesFetchError(RuntimeError):
    """Raised when changed-file collection fails permanently or exhausts retries."""


def _run_gh_api_page(
    repo: str,
    pr_number: str,
    page: int,
    per_page: int,
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run ``gh api`` for a single page of the PR files endpoint.

    Adding ``-f`` fields makes ``gh api`` default to POST, so ``--method GET``
    is required. With that explicit method, gh serializes the fields as query
    parameters without embedding a query string in the endpoint.
    """
    return subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "GET",
            f"repos/{repo}/pulls/{pr_number}/files",
            "-f",
            f"per_page={per_page}",
            "-f",
            f"page={page}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_page(
    proc: subprocess.CompletedProcess[str],
) -> tuple[list[dict[str, Any]] | None, str | None, bool]:
    """Validate one page response.

    Returns ``(files, None, False)`` on success, or
    ``(None, reason, retryable)`` on failure. The reason and retryability are
    surfaced by the retry loop and terminal diagnostic.
    """
    status = _http_status(proc)
    response_text = "\n".join(part for part in (proc.stderr, proc.stdout) if part)
    is_html = proc.stdout.lstrip().startswith("<")
    connection_failure = _is_connection_failure(response_text)

    if proc.returncode != 0:
        snippet = (proc.stderr or proc.stdout or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        retryable = _is_transient_http_status(status) or is_html or connection_failure
        return None, f"gh api exited with code {proc.returncode}: {snippet}".strip(), retryable
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        # This is the HTML-error-page symptom from issue #5918: a 502/503/504 body
        # beginning with '<' that cannot be JSON-parsed.
        snippet = proc.stdout.strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        retryable = is_html or connection_failure
        return (
            None,
            f"non-JSON response (likely HTML error page): {exc}; snippet: {snippet!r}",
            retryable,
        )
    if not isinstance(data, list):
        snippet = proc.stdout.strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        return (
            None,
            f"unexpected response shape (expected a JSON list); snippet: {snippet!r}",
            _is_transient_http_status(status),
        )
    return data, None, False


def _http_status(proc: subprocess.CompletedProcess[str]) -> int | None:
    """Extract an HTTP status from gh's JSON body or stderr diagnostic."""
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        data = None
    if isinstance(data, dict):
        raw_status = data.get("status")
        try:
            return int(raw_status)
        except (TypeError, ValueError):
            pass

    match = re.search(r"\bHTTP\s+(\d{3})\b", proc.stderr or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _is_transient_http_status(status: int | None) -> bool:
    """Return whether an HTTP status is safe to retry."""
    return status == 429 or (status is not None and 500 <= status <= 599)


def _is_connection_failure(message: str) -> bool:
    """Recognize gh diagnostics for transient connection failures."""
    lowered = message.lower()
    return any(marker in lowered for marker in _CONNECTION_FAILURE_MARKERS)


def _backoff_delay(
    attempt: int, base_delay: float, max_delay: float, *, rng: Callable[[], float]
) -> float:
    """Exponential backoff with full jitter, capped at ``max_delay`` seconds.

    ``attempt`` is 1-based. The cap grows as ``base_delay * 2**(attempt-1)`` and
    the actual sleep is a uniform random fraction of that cap (full jitter) to
    avoid thundering retries across concurrent jobs. ``rng`` must return a float
    in ``[0, 1)``.
    """
    cap = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return max(0.0, rng()) * cap


def fetch_all_pr_files(  # noqa: PLR0913
    repo: str,
    pr_number: str,
    *,
    per_page: int = DEFAULT_PER_PAGE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    timeout: int = DEFAULT_TIMEOUT,
    sleep: Callable[[float], None] | None = None,
    run_page: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    rng: Callable[[], float] | None = None,
) -> list[dict[str, Any]]:
    """Fetch every page of the PR files endpoint with per-page retry/backoff.

    Parameters
    ----------
    repo, pr_number:
        GitHub ``owner/repo`` and PR number.
    per_page:
        Page size (GitHub caps this at 100).
    max_attempts:
        Retry attempts *per page* before failing closed.
    base_delay, max_delay:
        Exponential-backoff seed and cap (seconds). Full jitter is applied.
    sleep, run_page, rng:
        Injectable seams used by the test-suite so the retry path is exercised
        without real subprocess calls or wall-clock sleeps.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if per_page < 1:
        raise ValueError("per_page must be at least 1")

    # Resolve injectable seams lazily so tests (and callers) that patch the
    # module-level ``_run_gh_api_page`` / ``time.sleep`` references at runtime are
    # honored. Default arguments bound at function-definition time would otherwise
    # capture the originals and ignore the patch.
    if sleep is None:
        sleep = time.sleep
    if run_page is None:
        run_page = _run_gh_api_page
    if rng is None:
        rng = lambda: random.uniform(0, 1)  # noqa: E731

    all_files: list[dict[str, Any]] = []
    page = 1
    while True:
        files, error, attempts_used, retryable = _fetch_page_with_retries(
            repo,
            pr_number,
            page,
            per_page,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            timeout=timeout,
            sleep=sleep,
            run_page=run_page,
            rng=rng,
        )
        if files is None:
            # Persistent failure: fail closed with a single actionable diagnostic.
            raise PrFilesFetchError(
                _format_terminal_error(
                    repo,
                    pr_number,
                    page,
                    attempts_used,
                    per_page,
                    error or "unknown error",
                    retryable=retryable,
                )
            )
        if not files:
            break  # empty page => no more files
        all_files.extend(files)
        if len(files) < per_page:
            break  # last (partial) page
        page += 1
    return all_files


def _fetch_page_with_retries(  # noqa: PLR0913
    repo: str,
    pr_number: str,
    page: int,
    per_page: int,
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    timeout: int,
    sleep: Callable[[float], None],
    run_page: Callable[..., subprocess.CompletedProcess[str]],
    rng: Callable[[], float],
) -> tuple[list[dict[str, Any]] | None, str | None, int, bool]:
    """Retry a single page up to ``max_attempts`` times with backoff.

    Returns ``(files, None, attempts_used, False)`` on success or
    ``(None, last_reason, attempts_used, retryable)`` on failure. Permanent
    failures return immediately; transient failures sleep between attempts using
    full-jitter exponential backoff.
    """
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            proc = run_page(repo, pr_number, page, per_page, timeout=timeout)
        except subprocess.TimeoutExpired:
            last_error = f"gh api timed out after {timeout} seconds"
            retryable = True
        except FileNotFoundError:
            last_error = "gh CLI not found on PATH; install GitHub CLI (https://cli.github.com/)"
            retryable = False
        except OSError as exc:
            last_error = f"could not run gh api: {exc}"
            retryable = isinstance(exc, ConnectionError) or _is_connection_failure(str(exc))
        else:
            files, error, retryable = _parse_page(proc)
            if error is None:
                return files, None, attempt, False
            last_error = error

        if not retryable:
            return None, last_error, attempt, False
        if attempt < max_attempts:
            delay = _backoff_delay(attempt, base_delay, max_delay, rng=rng)
            if delay > 0:
                sleep(delay)
    return None, last_error, max_attempts, True


def _format_terminal_error(
    repo: str,
    pr_number: str,
    page: int,
    attempts_used: int,
    per_page: int,
    last_error: str,
    *,
    retryable: bool,
) -> str:
    """Build the fail-closed diagnostic for permanent or exhausted failures."""
    if retryable:
        guidance = (
            "  This is typically a transient GitHub API outage (502/503/504/429) or "
            "connection failure. Re-run the job; if it persists, check "
            "https://www.githubstatus.com/.\n"
        )
    else:
        guidance = (
            "  The failure is non-retryable (for example HTTP 401/403/404 or an "
            "invalid response). Verify the endpoint and GH_TOKEN permissions.\n"
        )
    return (
        f"ERROR: Failed to collect changed files for PR #{pr_number} from "
        f"repos/{repo}/pulls/{pr_number}/files after {attempts_used} attempt(s) "
        f"on page {page} (per_page={per_page}).\n"
        f"  Last error: {last_error}\n"
        f"{guidance}"
        "  This is an infrastructure failure during changed-file collection, NOT "
        "a PR body/contract validation failure."
    )


def write_outputs(
    files: list[dict[str, Any]],
    *,
    out_changed_files: str | None = None,
    out_files_status: str | None = None,
    out_added_files: str | None = None,
) -> None:
    """Write the requested output files atomically (only on full success).

    Outputs are written after the whole list is in hand, so a partial/failed run
    never leaves a truncated file. ``status`` defaults to an empty string for the
    rare entry that omits it. Added files are those with ``status == "added"``.
    """
    changed = [_filename_of(entry) for entry in files]
    added = [_filename_of(entry) for entry in files if str(entry.get("status", "")) == "added"]

    if out_changed_files is not None:
        _write_lines(out_changed_files, changed)
    if out_added_files is not None:
        _write_lines(out_added_files, added)
    if out_files_status is not None:
        with open(out_files_status, "w", encoding="utf-8") as handle:
            for entry in files:
                status = str(entry.get("status", ""))
                handle.write(f"{status}\t{_filename_of(entry)}\n")


def _filename_of(entry: dict[str, Any]) -> str:
    """Return the filename for a files-entry, failing closed on a bad entry."""
    name = entry.get("filename")
    if not isinstance(name, str) or not name:
        raise PrFilesFetchError(
            f"files endpoint returned an entry without a string 'filename' field: {entry!r}"
        )
    return name


def _write_lines(path: str, lines: list[str]) -> None:
    """Write newline-delimited lines to *path*."""
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"owner/repo (default: {DEFAULT_REPO})."
    )
    parser.add_argument("--pr-number", required=True, help="Pull-request number.")
    parser.add_argument(
        "--out-changed-files",
        help="Path to write the newline-delimited list of all changed filenames.",
    )
    parser.add_argument(
        "--out-files-status",
        help="Path to write a TSV of 'status<TAB>filename' for every changed file.",
    )
    parser.add_argument(
        "--out-added-files",
        help="Path to write the newline-delimited list of ADDED (status == 'added') filenames.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help=f"page size (default: {DEFAULT_PER_PAGE}).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"retry attempts per page before failing closed (default: {DEFAULT_MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=DEFAULT_BASE_DELAY,
        help=f"exponential backoff seed in seconds (default: {DEFAULT_BASE_DELAY}).",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=DEFAULT_MAX_DELAY,
        help=f"exponential backoff cap in seconds (default: {DEFAULT_MAX_DELAY}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"per-page gh api timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Collect a PR's changed files with retry/backoff and write the requested outputs."""
    args = _build_parser().parse_args(argv)

    if not any([args.out_changed_files, args.out_files_status, args.out_added_files]):
        print(
            "ERROR: at least one of --out-changed-files, --out-files-status, "
            "or --out-added-files must be given.",
            file=sys.stderr,
        )
        return 2

    try:
        files = fetch_all_pr_files(
            args.repo,
            args.pr_number,
            per_page=args.per_page,
            max_attempts=args.max_attempts,
            base_delay=args.base_delay,
            max_delay=args.max_delay,
            timeout=args.timeout,
        )
    except PrFilesFetchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_outputs(
        files,
        out_changed_files=args.out_changed_files,
        out_files_status=args.out_files_status,
        out_added_files=args.out_added_files,
    )

    # Echo a short summary so the workflow log shows what was collected.
    print(f"Collected {len(files)} changed file(s) for PR #{args.pr_number} ({args.repo}).")
    for entry in files[:120]:
        print(f"  {entry.get('status', '')}\t{_filename_of(entry)}")
    if len(files) > 120:
        print(f"  ... ({len(files) - 120} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
