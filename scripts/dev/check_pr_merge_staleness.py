#!/usr/bin/env python3
"""Gate-side staleness check for PR merge readiness.

Prevents green-alone-red-together merge races by detecting when main has moved
since the PR's CI last ran.  Two detection layers are used:

1. **Workflow-run merge ref** (precise): when CI is GitHub Actions on a
   pull_request trigger, the run's ``head_sha`` is a temporary merge commit of
   the PR into main.  The merge commit's second parent *is* the exact main
   commit the PR was tested against.  Comparing that against current main
   catches the 3-second merge race described in issue #5389.

2. **Base-vs-main fallback** (conservative): when the merge ref is unavailable
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


def _get_merge_ref_second_parent(merge_sha: str, repo: str) -> str | None:
    """Return the second parent of a merge commit (the main commit tested against)."""
    result = _gh(
        ["api", f"repos/{repo}/commits/{merge_sha}", "--jq", ".parents[1].sha"],
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha if sha else None


def _detect_merge_sha_from_workflow_run(repo: str, pr_number: str) -> str | None:
    """Detect the merge commit SHA from a GitHub Actions workflow run for a PR.

    When a workflow triggers on ``pull_request``, GitHub creates a temporary merge
    ref and the run's ``head_sha`` is that merge commit.  This function fetches
    the most recent Actions workflow run for the PR and returns its ``head_sha``.

    Returns None when:
    - The repository does not use GitHub Actions.
    - The ``GITHUB_ACTIONS`` env var is not set (non-Actions environment).
    - No workflow runs are found for the PR.
    - API errors occur.
    """
    import os

    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if "pull_request" not in event_name:
        return None

    try:
        result = _gh(
            [
                "api",
                f"repos/{repo}/actions/runs",
                "-f",
                f"head_branch={pr_number}",
                "--jq",
                ".workflow_runs[0].head_sha",
            ],
        )
        if result.returncode != 0:
            return None
        sha = result.stdout.strip()
        return sha if sha else None
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
      - ``detection``: ``"workflow_run_merge_ref"`` | ``"base_vs_main"``
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

    merge_sha = _detect_merge_sha_from_workflow_run(repo, pr_number)
    if merge_sha:
        main_at_merge = _get_merge_ref_second_parent(merge_sha, repo)
        if main_at_merge:
            is_stale = main_at_merge != main_sha
            return {
                "status": "ok",
                "stale": is_stale,
                "detection": "workflow_run_merge_ref",
                "pr": pr_number,
                "base_sha": base_sha,
                "merge_sha": merge_sha,
                "main_at_merge": main_at_merge,
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
            else "Cannot detect the exact merge ref CI tested against. "
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
    if "merge_sha" in data:
        lines.append(f"  merge_sha: {data['merge_sha']}")
    if "main_at_merge" in data:
        lines.append(f"  main_at_merge: {data['main_at_merge']}")
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
