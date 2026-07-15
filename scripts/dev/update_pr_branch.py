#!/usr/bin/env python3
"""Request a PR branch update through the gh-compatible REST API.

The expected head SHA is mandatory so a branch refresh cannot silently target a
new commit after the caller inspected the PR.  A changed live head is reported
as a guarded mismatch and no update request is sent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command and return its completed process."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a GitHub CLI JSON response."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse gh output as JSON: {exc}"
    if not isinstance(data, dict):
        return None, "gh output is not a JSON object"
    return data, None


def _resolve_repo(explicit: str) -> str | None:
    """Resolve the repository identifier from an explicit value or gh."""
    if explicit:
        return explicit
    result = _gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    if result.returncode != 0:
        return None
    repo = result.stdout.strip()
    return repo if repo else None


def _fetch_pr_head_sha(pr_number: str, *, repo: str) -> tuple[str | None, str | None]:
    """Return the current PR head SHA from the pull-request REST endpoint."""
    result = _gh(["api", f"repos/{repo}/pulls/{pr_number}"])
    if result.returncode != 0:
        return None, result.stderr.strip() or "gh api pull request failed"
    payload, parse_error = _parse_json(result.stdout)
    if parse_error or payload is None:
        return None, parse_error or "gh API response is not an object"
    head = payload.get("head")
    if not isinstance(head, dict):
        return None, "PR head metadata is missing"
    sha = head.get("sha")
    if not isinstance(sha, str) or not sha:
        return None, "PR head SHA is empty"
    return sha, None


def update_pr_branch(
    pr_number: str,
    *,
    repo: str,
    expected_head_sha: str,
) -> dict[str, Any]:
    """Request a branch update only if the live PR head matches the guard."""
    live_head_sha, error = _fetch_pr_head_sha(pr_number, repo=repo)
    if live_head_sha is None:
        return {
            "status": "error",
            "error": error or "Could not fetch PR head SHA",
            "pr": pr_number,
            "repo": repo,
            "expected_head_sha": expected_head_sha,
            "updated": False,
        }
    if live_head_sha != expected_head_sha:
        return {
            "status": "head_mismatch",
            "error": "PR head changed since the expected SHA was recorded",
            "pr": pr_number,
            "repo": repo,
            "expected_head_sha": expected_head_sha,
            "live_head_sha": live_head_sha,
            "updated": False,
        }

    result = _gh(
        [
            "api",
            f"repos/{repo}/pulls/{pr_number}/update-branch",
            "--method",
            "PUT",
            "-f",
            f"expected_head_sha={expected_head_sha}",
        ]
    )
    if result.returncode != 0:
        return {
            "status": "error",
            "error": result.stderr.strip() or "gh update-branch request failed",
            "pr": pr_number,
            "repo": repo,
            "expected_head_sha": expected_head_sha,
            "live_head_sha": live_head_sha,
            "updated": False,
        }

    response, parse_error = _parse_json(result.stdout)
    if parse_error or response is None:
        return {
            "status": "error",
            "error": parse_error or "gh update-branch response is missing",
            "pr": pr_number,
            "repo": repo,
            "expected_head_sha": expected_head_sha,
            "live_head_sha": live_head_sha,
            "updated": False,
        }
    return {
        "status": "update_requested",
        "pr": pr_number,
        "repo": repo,
        "expected_head_sha": expected_head_sha,
        "live_head_sha": live_head_sha,
        "updated": True,
        "response": response,
    }


def _output(data: dict[str, Any], *, as_json: bool) -> None:
    """Print a result in JSON or concise human-readable form."""
    if as_json:
        print(json.dumps(data))
        return
    status = data.get("status", "error")
    if status == "update_requested":
        print(f"PR #{data['pr']}: branch update requested")
    elif status == "head_mismatch":
        print(
            f"PR #{data['pr']}: HEAD guard failed (expected {data['expected_head_sha']}, "
            f"live {data['live_head_sha']})"
        )
    else:
        print(f"PR #{data.get('pr', '?')}: ERROR: {data.get('error', 'unknown error')}")


def main(argv: list[str] | None = None) -> int:
    """Validate the expected head and request a PR branch update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", nargs="?", help="GitHub PR number")
    parser.add_argument("--pr", dest="pr_number_option", help="GitHub PR number alias")
    parser.add_argument("--repo", default="", help="owner/repo (default: detect from gh)")
    parser.add_argument(
        "--expected-head-sha",
        required=True,
        help="Expected current PR head SHA; no update is sent if it changed",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args(argv)

    if args.pr_number and args.pr_number_option and args.pr_number != args.pr_number_option:
        parser.error("conflicting PR numbers: pass either positional or --pr, not both")
    pr_number = args.pr_number_option or args.pr_number
    if not pr_number:
        parser.error("PR number is required (positional or --pr)")

    try:
        repo = _resolve_repo(args.repo)
        if repo is None:
            data = {"status": "error", "error": "Failed to detect repository"}
            _output(data, as_json=args.json)
            return 2
        data = update_pr_branch(
            pr_number,
            repo=repo,
            expected_head_sha=args.expected_head_sha,
        )
    except FileNotFoundError:
        data = {"status": "error", "error": "gh CLI not found"}
    except subprocess.TimeoutExpired:
        data = {"status": "error", "error": "gh CLI command timed out"}

    _output(data, as_json=args.json)
    if data.get("status") == "update_requested":
        return 0
    if data.get("status") == "head_mismatch":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
