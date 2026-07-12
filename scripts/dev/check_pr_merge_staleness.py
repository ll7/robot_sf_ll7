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
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a gh command and return the completed process."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse gh JSON stdout into a dict or an error string."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse gh output as JSON: {exc}"
    if not isinstance(data, dict):
        return None, "gh output is not a JSON object"
    return data, None


def _get_main_sha(repo: str) -> str | None:
    """Return current main branch HEAD SHA, or None on error."""
    result = _gh(["api", f"repos/{repo}/git/refs/heads/main", "--jq", ".object.sha"])
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha if sha else None


def _fetch_pr_head_metadata(pr_number: str) -> tuple[str | None, str | None]:
    """Return ``(head_branch, head_sha)`` for a PR, or ``(None, None)``."""
    result = _gh(["pr", "view", pr_number, "--json", "headRefName,headRefOid"])
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
    """Return workflow-run objects for a PR head branch, or ``None`` on API failure."""
    result = _gh(
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
        ],
    )
    if result.returncode != 0:
        return None
    payload, _ = _parse_json(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        return None
    return [run for run in payload["workflow_runs"] if isinstance(run, dict)]


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
        head_branch, head_sha = _fetch_pr_head_metadata(pr_number)
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
) -> dict[str, Any]:
    """Check whether a PR's CI ran against a stale main.

    Returns a result dict with at minimum:
      - ``stale``: bool | None (None = unknown / error)
      - ``detection``: ``"workflow_run_base_sha"`` | ``"base_vs_main"``
      - ``pr``, ``base_sha``, ``main_sha``
    """
    main_sha = _get_main_sha(repo)
    if main_sha is None:
        return {
            "status": "error",
            "error": "Failed to fetch current main SHA",
            "stale": None,
            "detection": "error",
            "pr": pr_number,
            "base_sha": base_sha,
            "main_sha": None,
        }

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
    if "ci_base_sha" in data:
        lines.append(f"  ci_base_sha: {data['ci_base_sha']}")
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


def _fetch_pr_base_sha(pr_number: str) -> tuple[str | None, str | None]:
    """Return (base_sha, error_message) for a PR."""
    result = _gh(
        ["pr", "view", pr_number, "--json", "baseRefOid", "--jq", ".baseRefOid"],
    )
    if result.returncode != 0:
        return None, result.stderr.strip() or "gh pr view failed"
    sha = result.stdout.strip()
    if not sha:
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

        base_sha, err = _fetch_pr_base_sha(pr_number)
        if base_sha is None:
            _output({"status": "error", "error": err}, as_json=args.json)
            return 1

        data = check_merge_staleness(pr_number, base_sha=base_sha, repo=repo)
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
