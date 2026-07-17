#!/usr/bin/env python3
"""Gate-side staleness check for PR merge readiness.

Prevents green-alone-red-together merge races by detecting when main has moved
since the PR's CI last ran.  Two detection layers are used:

1. **Workflow-run base provenance** (precise): a completed GitHub Actions
   ``pull_request`` run exposes the PR's base SHA in its ``pull_requests``
   metadata. Comparing that exact CI-tested base against current main catches
   the 3-second merge race described in issue #5389.

2. **Base-vs-main fallback** (conservative): when workflow-run provenance is unavailable
   (non-Actions CI, missing permissions, API failure), the script falls back to
   comparing ``pull_request.base.sha`` against current main.  This is still
   useful but does *not* catch the race where both PRs tested against the *same*
   stale main and both passed individually.

Exit codes:
   0  PR merge base matches current main (not stale).
   1  PR is stale; main has moved since CI ran.
   2  Could not determine; warn and let the caller decide.

Stacked-PR handling (issue #5965):
   A PR whose declared base is a branch other than ``main`` is only stale when
   its CI did not test the *current* declared base ref, or when the declared
   parent branch itself lags ``main``.  Unrelated movement on ``main`` alone
   never marks a current stacked base as stale.  Use ``--base-ref`` (or let it
   auto-detect) to enable this.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Retry budget for transient GitHub API failures (429/5xx, timeout, connection).
# Permanent failures (auth, not-found) are never retried.
GH_RETRY_MAX_ATTEMPTS = 4
GH_RETRY_BACKOFF_BASE = 1.0
GH_RETRY_MAX_BACKOFF = 8.0

DEFAULT_BASE_BRANCH = "main"


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a gh command and return the completed process."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class _PermanentApiError(Exception):
    """A GitHub API request failed for a reason that will not be fixed by retrying."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


def _is_transient_gh_failure(result: subprocess.CompletedProcess) -> bool:
    """Return True when a failed gh call is worth retrying.

    Transient conditions: HTTP 429 (rate limit), 5xx (server error), or a
    generic network/timeout error reported by gh.  Permanent conditions such as
    authentication or not-found are not transient.
    """
    if result.returncode == 0:
        return False
    stderr = (result.stderr or "").lower()
    stdout = (result.stdout or "").lower()
    combined = f"{stderr} {stdout}"
    # Permanent conditions: do not retry.
    if "not found" in combined or "404" in combined:
        return False
    if "could not resolve" in combined or "repository" in combined:
        return False
    if "authentication" in combined or "unauthorized" in combined or "401" in combined:
        return False
    if "403" in combined and "rate" not in combined:
        return False
    # Transient conditions.
    if "rate limit" in combined or "429" in combined:
        return True
    if any(code in combined for code in ("500", "502", "503", "504")):
        return True
    # gh exited non-zero without an explicit permanent marker: treat network /
    # timeout style failures as transient.
    if "timeout" in combined or "timed out" in combined or "connection" in combined:
        return True
    # gh exited non-zero with no recognizable message (e.g. killed mid-run).
    return True


def _gh_with_retry(
    args: list[str],
    *,
    timeout: int = 30,
    max_attempts: int = GH_RETRY_MAX_ATTEMPTS,
    backoff_base: float = GH_RETRY_BACKOFF_BASE,
    max_backoff: float = GH_RETRY_MAX_BACKOFF,
    sleep: Callable[[float], None] | None = None,
) -> subprocess.CompletedProcess:
    """Run a ``gh`` command with bounded exponential backoff on transient failure.

    Permanent failures (auth, not-found) raise ``_PermanentApiError`` immediately.
    Persistent transient failure exhausts the retry budget and also raises
    ``_PermanentApiError`` (kind ``"unavailable"``) so callers fail closed rather
    than inferring branch freshness from a missing response.
    """
    last_error: Exception | None = None
    do_sleep = sleep if sleep is not None else time.sleep
    for attempt in range(1, max_attempts + 1):
        try:
            result = _gh(args, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            do_sleep(min(backoff_base * (2 ** (attempt - 1)), max_backoff))
            continue

        if result.returncode == 0:
            return result

        stderr = (result.stderr or "").lower()
        stdout = (result.stdout or "").lower()
        combined = f"{stderr} {stdout}"
        if not _is_transient_gh_failure(result):
            if "not found" in combined or "404" in combined:
                kind = "not_found"
            elif "authentication" in combined or "unauthorized" in combined or "401" in combined:
                kind = "auth"
            else:
                kind = "permanent"
            raise _PermanentApiError(
                f"GitHub API permanent failure for {args[0]}: "
                f"{result.stderr.strip() or result.stdout.strip()}",
                kind=kind,
            )

        last_error = subprocess.CalledProcessError(
            result.returncode, ["gh", *args], result.stdout, result.stderr
        )
        if attempt >= max_attempts:
            break
        do_sleep(min(backoff_base * (2 ** (attempt - 1)), max_backoff))

    raise _PermanentApiError(
        f"GitHub API unavailable after retries (persistent transient failure): {last_error}",
        kind="unavailable",
    )


@dataclass
class _MainShaResult:
    """Result of fetching the current main SHA with retry/backoff."""

    sha: str | None
    error: str | None = None
    unavailable: bool = False


def _fetch_main_sha_with_retry(repo: str) -> _MainShaResult:
    """Fetch current main HEAD SHA, retrying transient API failures.

    Returns an explicit ``unavailable`` result on persistent transient failure
    rather than inferring freshness from a missing response.
    """
    try:
        result = _gh_with_retry(["api", f"repos/{repo}/git/refs/heads/main", "--jq", ".object.sha"])
    except _PermanentApiError as exc:
        if exc.kind == "unavailable":
            return _MainShaResult(sha=None, error=exc.message, unavailable=True)
        return _MainShaResult(sha=None, error=exc.message)
    sha = result.stdout.strip()
    if not sha:
        return _MainShaResult(sha=None, error="GitHub API returned an empty main SHA")
    return _MainShaResult(sha=sha)


def _fetch_workflow_runs_with_retry(repo: str, head_branch: str) -> list[dict[str, Any]] | None:
    """Fetch workflow runs for a PR head branch, retrying transient failures.

    Returns ``None`` for permanent failures (auth, not-found) and for persistent
    transient failure (unavailable), leaving the caller on the documented fallback
    path without inferring a base SHA.
    """
    try:
        result = _gh_with_retry(
            [
                "api",
                f"repos/{repo}/actions/runs",
                "--method",
                "GET",
                "-f",
                f"branch={head_branch}",
                "-f",
                "event=pull_request",
                "-f",
                "per_page=100",
            ]
        )
    except _PermanentApiError:
        return None
    payload, _ = _parse_json(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        return None
    return [run for run in payload["workflow_runs"] if isinstance(run, dict)]


def _parse_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse gh JSON stdout into a dict or an error string."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse gh output as JSON: {exc}"
    if not isinstance(data, dict):
        return None, "gh output is not a JSON object"
    return data, None


def _get_ref_sha(repo: str, ref: str) -> str | None:
    """Return the current HEAD SHA of a branch ref, or None on error."""
    safe_ref = ref.replace("/", "%2F")
    result = _gh(["api", f"repos/{repo}/git/refs/heads/{safe_ref}", "--jq", ".object.sha"])
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha if sha else None


def _get_main_sha(repo: str) -> str | None:
    """Return current main branch HEAD SHA, or None on error.

    Retries transient API failures; returns None on permanent or persistent
    transient failure (the caller treats None as an explicit error, never as
    a freshness signal).
    """
    return _fetch_main_sha_with_retry(repo).sha


def _fetch_pr_base_ref(pr_number: str, *, repo: str) -> str | None:
    """Return the PR's declared base branch name (e.g. ``main`` or ``feature/x``).

    Returns ``None`` when the base ref name cannot be determined.
    """
    result = _gh(["api", f"repos/{repo}/pulls/{pr_number}"])
    if result.returncode != 0:
        return None
    payload, _ = _parse_json(result.stdout)
    if not isinstance(payload, dict):
        return None
    base = payload.get("base")
    if not isinstance(base, dict):
        return None
    ref = base.get("ref")
    if isinstance(ref, str) and ref:
        return ref
    return None


def _fetch_pr_head_metadata(pr_number: str, *, repo: str) -> tuple[str | None, str | None]:
    """Return ``(head_branch, head_sha)`` for a PR, or ``(None, None)``."""
    result = _gh(["pr", "view", pr_number, "--repo", repo, "--json", "headRefName,headRefOid"])
    if result.returncode != 0:
        return None, None
    payload, _ = _parse_json(result.stdout)
    if not isinstance(payload, dict):
        return None, None
    head_branch = payload.get("headRefName")
    head_sha = payload.get("headRefOid")
    if not isinstance(head_branch, str) or not head_branch:
        return None, None
    if not isinstance(head_sha, str) or not head_sha:
        return None, None
    return head_branch, head_sha


def _fetch_workflow_runs(repo: str, head_branch: str) -> list[dict[str, Any]] | None:
    """Return workflow-run objects for a PR head branch, or ``None`` on API failure.

    Retries transient failures; returns None on permanent or persistent transient
    failure so the caller falls back without inferring a base SHA.
    """
    return _fetch_workflow_runs_with_retry(repo, head_branch)


def _find_workflow_run_base_sha(
    workflow_runs: list[dict[str, Any]],
    *,
    pr_number: str,
    head_sha: str,
) -> str | None:
    """Find the completed run's recorded PR base SHA for the current head."""
    for run in workflow_runs:
        if run.get("status") != "completed" or run.get("head_sha") != head_sha:
            continue
        pull_requests = run.get("pull_requests")
        if not isinstance(pull_requests, list):
            continue
        matching_pr = next(
            (
                pull_request
                for pull_request in pull_requests
                if isinstance(pull_request, dict)
                and str(pull_request.get("number")) == str(pr_number)
            ),
            None,
        )
        if not isinstance(matching_pr, dict):
            continue
        base = matching_pr.get("base")
        if not isinstance(base, dict):
            continue
        base_sha = base.get("sha")
        if isinstance(base_sha, str) and base_sha:
            return base_sha
    return None


def _detect_workflow_run_base_sha(repo: str, pr_number: str) -> str | None:
    """Return the exact base SHA recorded by a completed PR workflow run.

    The Actions run API exposes the PR's ``base.sha`` in each run's
    ``pull_requests`` metadata. This avoids treating ``head_sha`` as a
    synthetic merge commit: the run API reports the source-branch head there.
    """
    try:
        head_branch, head_sha = _fetch_pr_head_metadata(pr_number, repo=repo)
        if head_branch is None or head_sha is None:
            return None
        workflow_runs = _fetch_workflow_runs(repo, head_branch)
        if workflow_runs is None:
            return None
        return _find_workflow_run_base_sha(
            workflow_runs,
            pr_number=pr_number,
            head_sha=head_sha,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


def check_merge_staleness(
    pr_number: str,
    *,
    base_sha: str,
    repo: str,
    base_ref: str | None = None,
) -> dict[str, Any]:
    """Check whether a PR's CI ran against a stale merge base.

    For a PR targeting ``main`` this preserves the existing merge-race check:
    the CI-tested base (or the PR's declared ``base.sha``) must match current
    main.  For an explicitly stacked PR (``base_ref`` is a non-main branch), the
    PR is only stale when its CI did **not** test the *current* declared base
    ref.  Unrelated movement on ``main`` alone never marks a current stacked
    base as stale, but if the declared parent branch itself lags ``main`` the
    PR still fails closed with an actionable parent-refresh reason.

    Returns a result dict with at minimum:
      - ``stale``: bool | None (None = unknown / error)
      - ``detection``: ``"workflow_run_base_sha"`` | ``"base_vs_main"``
                         | ``"stacked_base"`` | ``"stacked_base_stale_parent"``
      - ``pr``, ``base_sha``, ``main_sha``, ``base_ref``
    """
    main_result = _fetch_main_sha_with_retry(repo)
    if main_result.sha is None:
        unavailable = (
            " (persistent transient API failure; not retried further)"
            if main_result.unavailable
            else ""
        )
        return {
            "status": "error",
            "error": (main_result.error or "Failed to fetch current main SHA") + unavailable,
            "stale": None,
            "detection": "error",
            "pr": pr_number,
            "base_sha": base_sha,
            "main_sha": None,
            "base_ref": base_ref,
        }
    main_sha = main_result.sha

    if base_ref is None or base_ref == DEFAULT_BASE_BRANCH:
        return _check_main_targeting_pr(
            pr_number, base_sha=base_sha, repo=repo, main_sha=main_sha, base_ref=base_ref
        )

    # Stacked PR: the relevant merge base is the declared parent branch, not main.
    parent_sha = _get_ref_sha(repo, base_ref)
    if parent_sha is None:
        return {
            "status": "error",
            "error": f"Failed to fetch current SHA for base ref {base_ref!r}",
            "stale": None,
            "detection": "error",
            "pr": pr_number,
            "base_sha": base_sha,
            "main_sha": main_sha,
            "base_ref": base_ref,
        }

    # Layer 1: workflow-run provenance — did CI test the current parent ref?
    ci_base_sha = _detect_workflow_run_base_sha(repo, pr_number)
    if ci_base_sha:
        if ci_base_sha == parent_sha:
            stale = _parent_branch_lags_main(repo, parent_sha, main_sha)
            if stale:
                return {
                    "status": "ok",
                    "stale": True,
                    "detection": "stacked_base_stale_parent",
                    "reason": (
                        f"PR is current with its declared base {base_ref!r}, but that "
                        f"parent branch lags main and must be refreshed (gh pr update-branch "
                        f"or rebase {base_ref} onto main) before merge."
                    ),
                    "pr": pr_number,
                    "base_sha": base_sha,
                    "ci_base_sha": ci_base_sha,
                    "parent_sha": parent_sha,
                    "main_sha": main_sha,
                    "base_ref": base_ref,
                }
            return {
                "status": "ok",
                "stale": False,
                "detection": "stacked_base",
                "pr": pr_number,
                "base_sha": base_sha,
                "ci_base_sha": ci_base_sha,
                "parent_sha": parent_sha,
                "main_sha": main_sha,
                "base_ref": base_ref,
            }
        return {
            "status": "ok",
            "stale": True,
            "detection": "stacked_base",
            "reason": (
                f"CI last tested base {ci_base_sha}, but the declared base {base_ref!r} "
                f"is now at {parent_sha}. Re-run CI against the current base."
            ),
            "pr": pr_number,
            "base_sha": base_sha,
            "ci_base_sha": ci_base_sha,
            "parent_sha": parent_sha,
            "main_sha": main_sha,
            "base_ref": base_ref,
        }

    # Fallback: compare the PR's declared base.sha against the current parent ref.
    if base_sha != parent_sha:
        return {
            "status": "ok",
            "stale": True,
            "detection": "stacked_base",
            "reason": (
                f"PR base {base_sha} differs from current declared base {base_ref!r} "
                f"({parent_sha}). Re-run CI against the current base."
            ),
            "warning": (
                "Cannot detect the exact workflow-run base SHA CI tested against; "
                "compared declared base.sha against current parent ref."
            ),
            "pr": pr_number,
            "base_sha": base_sha,
            "parent_sha": parent_sha,
            "main_sha": main_sha,
            "base_ref": base_ref,
        }

    stale = _parent_branch_lags_main(repo, parent_sha, main_sha)
    if stale:
        return {
            "status": "ok",
            "stale": True,
            "detection": "stacked_base_stale_parent",
            "reason": (
                f"PR is current with its declared base {base_ref!r}, but that parent "
                f"branch lags main and must be refreshed before merge."
            ),
            "warning": (
                "Cannot detect the exact workflow-run base SHA CI tested against; "
                "compared declared base.sha against current parent ref."
            ),
            "pr": pr_number,
            "base_sha": base_sha,
            "parent_sha": parent_sha,
            "main_sha": main_sha,
            "base_ref": base_ref,
        }
    return {
        "status": "ok",
        "stale": False,
        "detection": "stacked_base",
        "warning": (
            "Cannot detect the exact workflow-run base SHA CI tested against; "
            "compared declared base.sha against current parent ref."
        ),
        "pr": pr_number,
        "base_sha": base_sha,
        "parent_sha": parent_sha,
        "main_sha": main_sha,
        "base_ref": base_ref,
    }


def _parent_branch_lags_main(repo: str, parent_sha: str, main_sha: str) -> bool:
    """Return whether the parent ref is behind or diverged from current main.

    A healthy stacked parent can be ahead of main, so SHA inequality alone is
    not sufficient. GitHub's compare endpoint reports the ancestry relation;
    an unavailable or malformed response fails closed as lagging.
    """
    if parent_sha == main_sha:
        return False
    result = _gh(["api", f"repos/{repo}/compare/{main_sha}...{parent_sha}", "--jq", ".status"])
    if result.returncode != 0:
        return True
    return result.stdout.strip() not in {"ahead", "identical"}


def _check_main_targeting_pr(
    pr_number: str,
    *,
    base_sha: str,
    repo: str,
    main_sha: str,
    base_ref: str | None,
) -> dict[str, Any]:
    """Race check for a PR that targets ``main`` (unchanged behaviour)."""
    ci_base_sha = _detect_workflow_run_base_sha(repo, pr_number)
    if ci_base_sha:
        is_stale = ci_base_sha != main_sha
        return {
            "status": "ok",
            "stale": is_stale,
            "detection": "workflow_run_base_sha",
            "pr": pr_number,
            "base_sha": base_sha,
            "ci_base_sha": ci_base_sha,
            "main_sha": main_sha,
            "base_ref": base_ref,
        }

    # Fallback: conservative base-vs-main comparison.
    is_stale = base_sha != main_sha
    return {
        "status": "ok",
        "stale": is_stale,
        "detection": "base_vs_main",
        "warning": (
            None
            if is_stale
            else "Cannot detect the exact workflow-run base SHA CI tested against. "
            "Consider re-running CI to confirm freshness."
        ),
        "pr": pr_number,
        "base_sha": base_sha,
        "main_sha": main_sha,
        "base_ref": base_ref,
    }


def format_human(data: dict[str, Any]) -> str:
    """Format a staleness result for human-readable output."""
    if data.get("status") == "error":
        return f"ERROR: {data.get('error', 'unknown error')}"

    pr = data.get("pr", "?")
    stale = data.get("stale")
    detection = data.get("detection", "unknown")

    if stale is None:
        verdict = "UNKNOWN"
    elif stale:
        verdict = "STALE"
    else:
        verdict = "FRESH"

    lines = [f"PR #{pr}: {verdict}  (detection: {detection})"]
    lines.append(f"  base_sha:  {data.get('base_sha', '?')}")
    lines.append(f"  main_sha:  {data.get('main_sha', '?')}")
    if data.get("base_ref") is not None:
        lines.append(f"  base_ref:  {data['base_ref']}")
    if "ci_base_sha" in data:
        lines.append(f"  ci_base_sha: {data['ci_base_sha']}")
    if "parent_sha" in data:
        lines.append(f"  parent_sha: {data['parent_sha']}")
    if data.get("reason"):
        lines.append(f"  reason: {data['reason']}")
    if data.get("warning"):
        lines.append(f"  warning: {data['warning']}")
    return "\n".join(lines)


def _resolve_repo(explicit: str) -> str | None:
    """Resolve the repository identifier, auto-detecting from gh when empty."""
    if explicit:
        return explicit
    result = _gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    if result.returncode != 0:
        return None
    repo = result.stdout.strip()
    return repo if repo else None


def _fetch_pr_base_sha(pr_number: str, *, repo: str) -> tuple[str | None, str | None]:
    """Return ``(base_sha, error_message)`` using the gh-compatible REST API."""
    # ``baseRefOid`` is not available in the installed gh 2.45.0 JSON field
    # set.  The pull-request REST endpoint is stable across supported gh
    # versions and exposes the same value as ``base.sha``.
    result = _gh(["api", f"repos/{repo}/pulls/{pr_number}"])
    if result.returncode != 0:
        return None, result.stderr.strip() or "gh api pull request failed"
    payload, parse_error = _parse_json(result.stdout)
    if parse_error or payload is None:
        return None, parse_error or "gh API response is not an object"
    base = payload.get("base")
    if not isinstance(base, dict):
        return None, "PR base metadata is missing"
    sha = base.get("sha")
    if not isinstance(sha, str) or not sha:
        return None, "PR base SHA is empty"
    return sha, None


def _output(data: dict[str, Any], *, as_json: bool) -> None:
    """Print the result in the requested format."""
    if as_json:
        print(json.dumps(data))
    else:
        print(format_human(data))


def _exit_code(data: dict[str, Any]) -> int:
    """Map a staleness result to an exit code."""
    if data.get("status") == "error":
        return 2
    if data.get("stale") is True:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: check merge staleness for a PR and print results."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pr_number", nargs="?", help="GitHub PR number")
    parser.add_argument("--pr", dest="pr_number_option", help="GitHub PR number alias")
    parser.add_argument("--json", action="store_true", default=False, help="emit JSON output")
    parser.add_argument("--repo", default="", help="owner/repo (default: detect from gh)")
    parser.add_argument(
        "--base-ref",
        default=None,
        help="PR's declared base branch (default: detect from gh). "
        "A non-main base enables stacked-PR handling.",
    )
    args = parser.parse_args(argv)

    if args.pr_number and args.pr_number_option and args.pr_number != args.pr_number_option:
        parser.error("conflicting PR numbers: pass either positional or --pr, not both")
    pr_number = args.pr_number_option or args.pr_number
    if not pr_number:
        parser.error("PR number is required (positional or --pr)")

    try:
        repo = _resolve_repo(args.repo)
        if repo is None:
            print("Failed to detect repository.  Pass --repo owner/repo.", file=sys.stderr)
            return 1

        base_sha, err = _fetch_pr_base_sha(pr_number, repo=repo)
        if base_sha is None:
            _output({"status": "error", "error": err}, as_json=args.json)
            return 1

        base_ref = args.base_ref
        if base_ref is None:
            base_ref = _fetch_pr_base_ref(pr_number, repo=repo)

        data = check_merge_staleness(pr_number, base_sha=base_sha, repo=repo, base_ref=base_ref)
    except FileNotFoundError:
        print("gh CLI not found.  Install GitHub CLI: https://cli.github.com/", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("gh CLI command timed out.", file=sys.stderr)
        return 1

    _output(data, as_json=args.json)
    return _exit_code(data)


if __name__ == "__main__":
    raise SystemExit(main())
