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
from pathlib import Path
from typing import Any

# Default gate worktree path used by the concurrent PR gate. When the gate runs
# in a linked worktree this lets the update guard verify (and recreate from the
# surviving lease) the worktree is intact before requesting a branch update.
# Override with --gate-worktree-path.
DEFAULT_GATE_WORKTREE_PATH = ""


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
    gate_worktree_path: str = "",
    gate_worktree_ttl_hours: float | None = None,
) -> dict[str, Any]:
    """Request a branch update only if the live PR head matches the guard.

    When ``gate_worktree_path`` is provided, the registered gate worktree is
    verified to still exist before requesting a branch update. A vanished
    worktree is first recreated from the surviving lease (report-only/opt-in
    semantics) so the gate can resume rather than die opaquely with
    ``CreateProcess ... No such file or directory``. If the worktree cannot be
    recovered, the call fails closed with the lease cleanup owner reported and
    issues no remote branch update.
    """
    if gate_worktree_path:
        health = _ensure_gate_worktree(
            gate_worktree_path,
            ttl_hours=gate_worktree_ttl_hours,
        )
        if not health.get("exists", False):
            return {
                "status": "gate_worktree_missing",
                "error": (
                    "Registered gate worktree is missing and could not be recreated "
                    f"before branch update; cleanup owner: {health.get('cleanup_owner') or 'unknown'}"
                ),
                "pr": pr_number,
                "repo": repo,
                "expected_head_sha": expected_head_sha,
                "gate_worktree_path": gate_worktree_path,
                "gate_worktree_health": health,
                "updated": False,
            }

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


def _ensure_gate_worktree(
    gate_worktree_path: str, *, ttl_hours: float | None = None
) -> dict[str, Any]:
    """Verify the registered gate worktree; recreate from a surviving lease if missing.

    Delegates to ``gate_worktree_guard.ensure_gate_worktree``. Returns a dict with
    at least ``exists``; when the path is missing but a live lease claims it, the
    lease owner is reported under ``cleanup_owner`` and the worktree is recreated
    when possible. Failures degrade to ``exists=False, cleanup_owner=None`` so the
    caller still fails closed and issues no remote branch update.
    """
    try:
        from scripts.dev.gate_worktree_guard import ensure_gate_worktree

        health, recreate = ensure_gate_worktree(Path(gate_worktree_path), ttl_hours=ttl_hours)
        result = {
            "exists": health.exists,
            "classification": health.classification,
            "cleanup_owner": health.cleanup_owner,
            "lease_owner": health.lease_owner,
            "lease_pr_number": health.lease_pr_number,
            "lease_gate_id": health.lease_gate_id,
        }
        if recreate is not None:
            result["recreated"] = recreate.recreated
            result["recreate_error"] = recreate.error
        return result
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "exists": False,
            "classification": "errored",
            "cleanup_owner": None,
            "error": f"could not verify/recreate gate worktree: {exc}",
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
    elif status == "gate_worktree_missing":
        print(
            f"PR #{data['pr']}: gate worktree missing before branch update "
            f"(cleanup owner: {data.get('gate_worktree_health', {}).get('cleanup_owner') or 'unknown'})"
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
    parser.add_argument(
        "--gate-worktree-path",
        default=DEFAULT_GATE_WORKTREE_PATH,
        help="Registered gate worktree path to verify (and recreate) before the branch update",
    )
    parser.add_argument(
        "--gate-worktree-ttl-hours",
        type=float,
        default=None,
        help="Lease TTL to (re)apply after recreating a missing gate worktree, in hours",
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
            gate_worktree_path=args.gate_worktree_path,
            gate_worktree_ttl_hours=args.gate_worktree_ttl_hours,
        )
    except FileNotFoundError:
        data = {"status": "error", "error": "gh CLI not found"}
    except subprocess.TimeoutExpired:
        data = {"status": "error", "error": "gh CLI command timed out"}

    _output(data, as_json=args.json)
    if data.get("status") == "update_requested":
        return 0
    if data.get("status") in ("head_mismatch", "gate_worktree_missing"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
