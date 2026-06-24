#!/usr/bin/env python3
"""Check CI status for a GitHub PR using the gh CLI.

Output is compact and cache-friendly.  Use --json for machine-readable output.
Run `--help` for the worktree-safe invocation used by agent workflows.
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


def _rollup_name(check: dict[str, Any]) -> str:
    """Return a display name for check-run and legacy-status rollup entries."""
    return str(check.get("name") or check.get("context") or "unknown")


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
            "number,title,state,mergeable,headRefName,headRefOid,statusCheckRollup,reviews",
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
        name = _rollup_name(check)
        name_counts[name] = name_counts.get(name, 0) + 1
    check_details = [
        {
            "name": _rollup_name(check),
            "status": _rollup_status(check),
            "conclusion": _rollup_conclusion(check),
            "details_url": check.get("detailsUrl", "") or check.get("targetUrl", ""),
        }
        for check in rollup
    ]

    return {
        "status": "ok",
        "pr": data.get("number"),
        "title": data.get("title", ""),
        "state": data.get("state", "unknown"),
        "mergeable": data.get("mergeable", "unknown"),
        "branch": data.get("headRefName", ""),
        "head_sha": data.get("headRefOid", ""),
        "checks": {
            "total": len(rollup),
            "overall": overall,
            "by_conclusion": conclusions,
            "by_status": states,
            "names": sorted(name_counts),
            "details": check_details,
        },
        "reviews": review_states,
    }


def _add_monitor_metadata(
    data: dict[str, Any],
    *,
    expected_head_sha: str,
    attempt: int,
    attempts: int,
    poll_interval: float,
    wait_budget_seconds: float,
    max_wall_seconds: float | None,
    deadline_epoch_seconds: int | None,
) -> None:
    """Attach compact CI monitor resume metadata to a status payload."""
    head_sha = str(data.get("head_sha") or "")
    if expected_head_sha:
        head_sha_matches_expected: bool | None = bool(head_sha) and head_sha == expected_head_sha
    else:
        head_sha_matches_expected = None
    data["monitor"] = {
        "route": "ci_wait_monitor",
        "expected_head_sha": expected_head_sha,
        "head_sha_matches_expected": head_sha_matches_expected,
        "poll_attempt": attempt,
        "poll_attempts": attempts,
        "poll_interval_seconds": poll_interval,
        "wait_budget_seconds": wait_budget_seconds,
        "max_wall_seconds": max_wall_seconds,
        "deadline_epoch_seconds": deadline_epoch_seconds,
        "route_evidence_only": True,
    }


def _format_human(data: dict[str, Any]) -> str:
    """Format CI status data for human-readable compact output."""
    if data.get("status") == "error":
        return f"ERROR fetching CI status: {data.get('error', 'unknown error')}"

    lines: list[str] = []
    lines.append(f"PR #{data['pr']}: {data['title']}")
    lines.append(
        f"  state: {data['state']}  |  mergeable: {data['mergeable']}  |  "
        f"branch: {data['branch']}  |  head: {data.get('head_sha', '')}"
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
    for check in checks.get("details", []):
        if check.get("status") == "completed" and check.get("conclusion") == "success":
            continue
        url = check.get("details_url")
        suffix = f"  |  {url}" if url else ""
        lines.append(
            f"    - {check.get('name', 'unknown')}: "
            f"{check.get('status', 'unknown')}/{check.get('conclusion', 'unknown')}{suffix}"
        )

    reviews = data.get("reviews", {})
    if reviews:
        review_str = " ".join(f"{k}={v}" for k, v in sorted(reviews.items()))
        lines.append(f"  reviews: {review_str}")

    return "\n".join(lines)


def _terminal_reason(
    overall: str | None,
    attempt: int,
    attempts: int,
    local_stop: bool,
) -> str | None:
    """Classify why a polling loop stopped on this iteration."""
    if overall is None:
        return None
    if overall != "pending":
        return str(overall)
    if attempt == attempts:
        return "attempt_exhausted"
    if local_stop:
        return "max_wall_seconds"
    return None


def _guard_head_sha(data: dict[str, Any], expected_head_sha: str) -> bool:
    """Fail closed when the observed PR head SHA diverges from the expected one.

    Mutates ``data`` in place and returns True if the caller should stop polling.
    """
    head_sha = str(data.get("head_sha") or "")
    if expected_head_sha and not head_sha:
        data["status"] = "error"
        data["error"] = "PR head SHA missing while monitoring CI"
        data["monitor"]["terminal_reason"] = "error"
        return True
    if expected_head_sha and head_sha != expected_head_sha:
        data["status"] = "error"
        data["error"] = "PR head SHA changed while monitoring CI"
        data["monitor"]["terminal_reason"] = "error"
        return True
    return False


def _bounded_sleep_seconds(
    poll_interval: float,
    wall_deadline: float | None,
) -> tuple[float, bool]:
    """Return the next sleep duration and whether the local wall cap is exhausted."""
    sleep_seconds = max(0.0, poll_interval)
    if wall_deadline is None:
        return sleep_seconds, False
    remaining_seconds = wall_deadline - time.monotonic()
    if remaining_seconds <= 0:
        return 0.0, True
    return min(sleep_seconds, remaining_seconds), False


def _non_negative_float(value: str) -> float:
    """Parse a non-negative float for local duration limits."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _poll_ci_status(
    pr: str,
    *,
    attempts: int,
    poll_interval: float,
    backoff: float,
    json_output: bool,
    expected_head_sha: str = "",
    max_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Fetch CI status once or poll until checks settle or the budget expires."""
    data: dict[str, Any] = {}
    wait_budget_seconds = max(0.0, float(attempts - 1) * max(0.0, poll_interval))
    effective_wait_budget = wait_budget_seconds
    if max_wall_seconds is not None:
        effective_wait_budget = min(wait_budget_seconds, max(0.0, max_wall_seconds))
    deadline_epoch_seconds = int(time.time() + effective_wait_budget) if attempts > 1 else None
    wall_deadline = (
        time.monotonic() + max(0.0, max_wall_seconds) if max_wall_seconds is not None else None
    )
    for attempt in range(1, attempts + 1):
        data = _fetch_ci_status(pr, backoff=backoff if attempt == 1 else 0.0)
        _add_monitor_metadata(
            data,
            expected_head_sha=expected_head_sha,
            attempt=attempt,
            attempts=attempts,
            poll_interval=poll_interval,
            wait_budget_seconds=wait_budget_seconds,
            max_wall_seconds=max_wall_seconds,
            deadline_epoch_seconds=deadline_epoch_seconds,
        )
        if data.get("status") == "error":
            data["monitor"]["terminal_reason"] = "error"
            break
        if _guard_head_sha(data, expected_head_sha):
            break
        overall = data.get("checks", {}).get("overall")
        sleep_seconds, local_stop = _bounded_sleep_seconds(poll_interval, wall_deadline)
        terminal_reason = _terminal_reason(overall, attempt, attempts, local_stop)
        if overall == "pending" and attempt < attempts and local_stop:
            data["monitor"]["local_stop_reason"] = "max_wall_seconds"
        if terminal_reason:
            data["monitor"]["terminal_reason"] = terminal_reason
        if attempts > 1:
            if json_output:
                print(json.dumps(data), flush=True)
            else:
                print(f"poll attempt {attempt}/{attempts}", flush=True)
                print(_format_human(data), flush=True)
        if terminal_reason:
            break
        time.sleep(sleep_seconds)
    return data


def main(argv: list[str] | None = None) -> int:
    """Entry point: check CI status and print results."""
    epilog = """\
Recommended agent workflow (fresh linked worktree, no local .venv):

  scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/check_pr_ci_status.py \\
      <pr-number> --expected-head-sha <head-sha> --poll-attempts 40 \\
      --poll-interval 30 --max-wall-seconds 1200 --json

The wrapper reuses the owning checkout's shared virtualenv and sets UV_NO_SYNC=1
so uv will not create or prompt for a per-worktree .venv.
`--max-wall-seconds` gives long-running agents a non-interactive local stop path;
exit code 2 means checks were still pending locally, not that remote GitHub checks
were cancelled or failed.
"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "pr_number",
        nargs="?",
        help="GitHub PR number; alternatively pass --pr <number> (default: detect from current branch)",
    )
    parser.add_argument(
        "--pr",
        dest="pr_number_option",
        help="GitHub PR number alias for workflows that prefer named arguments",
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
    parser.add_argument(
        "--poll-attempts",
        type=int,
        default=1,
        help="bounded polling attempts; values above 1 wait for pending checks to settle",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="seconds between bounded polling attempts",
    )
    parser.add_argument(
        "--expected-head-sha",
        default="",
        help="optional PR head SHA guard; stale heads return error without claiming readiness",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=_non_negative_float,
        default=None,
        help=(
            "optional local wall-clock cap for bounded polling; pending checks return exit code 2 "
            "without affecting remote GitHub checks"
        ),
    )
    args = parser.parse_args(argv)
    if args.pr_number and args.pr_number_option and args.pr_number != args.pr_number_option:
        parser.error(
            "conflicting PR numbers: pass either positional <pr-number> or --pr <number>, "
            "or pass the same value to both"
        )
    pr_number = args.pr_number_option or args.pr_number

    try:
        pr = _resolve_pr_number(pr_number)
        attempts = max(1, args.poll_attempts)
        data = _poll_ci_status(
            pr,
            attempts=attempts,
            poll_interval=args.poll_interval,
            backoff=args.backoff,
            json_output=args.json,
            expected_head_sha=args.expected_head_sha,
            max_wall_seconds=args.max_wall_seconds,
        )
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
        if args.json:
            print(json.dumps(data))
        else:
            print(_format_human(data))
        return 1

    if attempts == 1:
        if args.json:
            print(json.dumps(data))
        else:
            print(_format_human(data))

    # Non-zero exit when CI is failing; pending checks are cache/backoff-safe.
    overall = data.get("checks", {}).get("overall")
    if overall == "failure":
        return 1
    if attempts > 1 and overall == "pending":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
