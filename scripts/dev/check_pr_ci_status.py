#!/usr/bin/env python3
"""Check CI status for a GitHub PR using the gh CLI.

Output is compact and cache-friendly.  Use --json for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any

FAILURE_CONCLUSIONS = {
    "failure",
    "error",
    "cancelled",
    "timed_out",
    "action_required",
    "startup_failure",
}
PENDING_STATUSES = {"expected", "in_progress", "pending", "queued", "requested", "waiting"}


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a gh command and return the completed process.

    Raises FileNotFoundError when gh is not installed.
    """
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _resolve_pr_number(pr_number: str | None) -> str:
    """Resolve PR number from argument or current branch."""
    if pr_number:
        return pr_number
    result = _gh(["pr", "view", "--json", "number", "--jq", ".number"])
    if result.returncode != 0:
        print(
            "Could not determine PR number from current branch. "
            "Provide a PR number or ensure you are on a PR branch.",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.stdout.strip()


def _parse_pr_view_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse `gh pr view --json` stdout into a dictionary or an error string."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse gh output as JSON: {exc}"
    if not isinstance(data, dict):
        return None, "gh output is not a JSON object"
    return data, None


def _rollup_conclusion(check: dict[str, Any]) -> str:
    """Return a normalized conclusion for check-run and legacy-status rollup entries."""
    conclusion = check.get("conclusion")
    if conclusion:
        return str(conclusion).lower()
    state = check.get("state")
    if state:
        return str(state).lower()
    return "pending"


def _rollup_status(check: dict[str, Any]) -> str:
    """Return a normalized lifecycle status for check-run and legacy-status rollup entries."""
    status = check.get("status")
    if status:
        return str(status).lower()
    state = check.get("state")
    if not state:
        return "completed"
    state_str = str(state).lower()
    if state_str in {"success", "failure", "error"}:
        return "completed"
    return state_str


def _fetch_ci_status(
    pr_number: str,
    backoff: float = 0.0,
) -> dict[str, Any]:
    """Fetch combined CI status for a PR.

    Args:
        pr_number: GitHub PR number.
        backoff: seconds to wait before fetching (for cache coherency).

    Returns:
        A dict with 'state', 'conclusion', 'statuses', and metadata.
    """
    if backoff > 0:
        time.sleep(backoff)

    result = _gh(
        [
            "pr",
            "view",
            pr_number,
            "--json",
            "number,title,state,mergeable,headRefName,statusCheckRollup,reviews",
        ]
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return {
            "status": "error",
            "error": stderr or f"gh returned exit code {result.returncode}",
        }

    data, parse_error = _parse_pr_view_json(result.stdout)
    if parse_error or data is None:
        return {
            "status": "error",
            "error": parse_error or "gh output is not a JSON object",
        }
    rollup = data.get("statusCheckRollup", []) or []

    # Classify overall CI state.
    conclusions: dict[str, int] = {}
    for check in rollup:
        c = _rollup_conclusion(check)
        conclusions[c] = conclusions.get(c, 0) + 1

    states: dict[str, int] = {}
    for check in rollup:
        s = _rollup_status(check)
        states[s] = states.get(s, 0) + 1

    failure_count = sum(conclusions.get(conclusion, 0) for conclusion in FAILURE_CONCLUSIONS)
    pending_count = sum(states.get(status, 0) for status in PENDING_STATUSES)
    if failure_count:
        overall = "failure"
    elif pending_count or not rollup:
        overall = "pending"
    else:
        overall = "success"

    # Aggregate reviews
    reviews = data.get("reviews", []) or []
    review_states: dict[str, int] = {}
    for rev in reviews:
        rs = rev.get("state", "UNKNOWN")
        review_states[rs] = review_states.get(rs, 0) + 1

    name_counts: dict[str, int] = {}
    for check in rollup:
        name = check.get("name", "unknown")
        name_counts[name] = name_counts.get(name, 0) + 1

    return {
        "status": "ok",
        "pr": data.get("number"),
        "title": data.get("title", ""),
        "state": data.get("state", "unknown"),
        "mergeable": data.get("mergeable", "unknown"),
        "branch": data.get("headRefName", ""),
        "checks": {
            "total": len(rollup),
            "overall": overall,
            "by_conclusion": conclusions,
            "by_status": states,
            "names": sorted(name_counts),
        },
        "reviews": review_states,
    }


def _format_human(data: dict[str, Any]) -> str:
    """Format CI status data for human-readable compact output."""
    if data.get("status") == "error":
        return f"ERROR fetching CI status: {data.get('error', 'unknown error')}"

    lines: list[str] = []
    lines.append(f"PR #{data['pr']}: {data['title']}")
    lines.append(
        f"  state: {data['state']}  |  mergeable: {data['mergeable']}  |  branch: {data['branch']}"
    )

    checks = data.get("checks", {})
    total = checks.get("total", 0)
    conclusions = checks.get("by_conclusion", {})
    states = checks.get("by_status", {})
    overall = checks.get("overall", "unknown")

    conclusion_str = " ".join(f"{k}={v}" for k, v in sorted(conclusions.items()))
    status_str = " ".join(f"{k}={v}" for k, v in sorted(states.items()))
    lines.append(
        f"  checks: {overall}  |  {total} total  |  {conclusion_str}  |  status: {status_str}"
    )

    reviews = data.get("reviews", {})
    if reviews:
        review_str = " ".join(f"{k}={v}" for k, v in sorted(reviews.items()))
        lines.append(f"  reviews: {review_str}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point: check CI status and print results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pr_number",
        nargs="?",
        help="GitHub PR number (default: detect from current branch)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="emit machine-readable JSON output",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=0.0,
        help="seconds to wait before fetching (for cache coherency)",
    )
    args = parser.parse_args(argv)

    try:
        pr = _resolve_pr_number(args.pr_number)
        data = _fetch_ci_status(pr, backoff=args.backoff)
    except FileNotFoundError:
        print("gh CLI not found. Install GitHub CLI: https://cli.github.com/", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print(
            "gh CLI command timed out. Check your network connection or GitHub status.",
            file=sys.stderr,
        )
        return 1

    if data.get("status") == "error":
        print(_format_human(data))
        return 1

    if args.json:
        print(json.dumps(data))
    else:
        print(_format_human(data))

    # Non-zero exit when CI is failing; pending checks are cache/backoff-safe.
    overall = data.get("checks", {}).get("overall")
    if overall == "failure":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
