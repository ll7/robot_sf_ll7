#!/usr/bin/env python3
"""Check CI status for a GitHub PR using the gh CLI.

Output is compact and cache-friendly.  Use --json for machine-readable output.
Run `--help` for the worktree-safe invocation used by agent workflows.
"""

from __future__ import annotations

import argparse
import json
import re
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
_ACTIONS_JOB_URL_RE = re.compile(
    r"/actions/runs/(?P<run_id>[0-9]+)/job/(?P<job_id>[0-9]+)(?:$|[/?#])"
)
_TERMINAL_STEP_CONCLUSIONS = {"neutral", "skipped", "success"}


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


def _check_run_identity(check: dict[str, Any]) -> tuple[str, str] | None:
    """Return a stable identity for timestamped GitHub Actions check runs.

    GitHub's PR rollup retains completed runs when editing a PR body retriggers
    a workflow on the same commit.  Only runs from the same workflow job can
    supersede one another; legacy statuses and runs without a timestamp remain
    independently fail-closed.
    """
    if check.get("__typename") != "CheckRun":
        return None
    workflow_name = str(check.get("workflowName") or "")
    started_at = str(check.get("startedAt") or "")
    if not workflow_name or not started_at:
        return None
    return workflow_name, _rollup_name(check)


def _latest_check_runs(rollup: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the newest timestamped run for each duplicate GitHub Actions job."""
    latest_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for check in rollup:
        identity = _check_run_identity(check)
        if identity is None:
            continue
        latest = latest_by_identity.get(identity)
        if latest is None or str(check["startedAt"]) > str(latest["startedAt"]):
            latest_by_identity[identity] = check

    effective_rollup: list[dict[str, Any]] = []
    superseded_count = 0
    for check in rollup:
        identity = _check_run_identity(check)
        if identity is not None and latest_by_identity[identity] is not check:
            superseded_count += 1
            continue
        effective_rollup.append(check)
    return effective_rollup, superseded_count


def _is_graphql_quota_error(message: str) -> bool:
    """Return whether a gh error message indicates GraphQL API rate-limit/quota exhaustion."""
    text = (message or "").lower()
    if "rate limit" not in text:
        return False
    return "graphql" in text or "api rate limit" in text or "too many requests" in text


def _git_remote_owner_name() -> tuple[str, str]:
    """Derive the ``owner/name`` GitHub repository from the ``origin`` git remote.

    Used by the REST fallback when GraphQL quota is exhausted (issue #6564): it is a local git
    call with no API quota cost. Returns empty strings when the remote cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "", ""
    if result.returncode != 0:
        return "", ""
    url = result.stdout.strip()
    if url.endswith(".git"):
        url = url[:-4]
    # Normalize the ssh shorthand "git@host:owner/name" to a slash-delimited path, then take the
    # final two non-empty path segments as owner/name. This avoids substring matching a host name
    # (which can appear at arbitrary positions) and stays robust across ssh/https remote forms.
    if "://" not in url and ":" in url:
        url = url.replace(":", "/", 1)
    parts = [segment for segment in url.split("/") if segment]
    if len(parts) < 2:
        return "", ""
    owner, name = parts[-2], parts[-1]
    return (owner, name) if owner and name else ("", "")


def _rest_api_get(path: str, *, timeout: int = 45) -> Any:
    """Fetch ``repos/{owner}/{name}/{path}`` via REST and parse JSON, or ``None`` on failure."""
    owner, name = _git_remote_owner_name()
    if not owner or not name:
        return None
    result = _gh(["api", f"repos/{owner}/{name}/{path}"], timeout=timeout)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _status_propagation_lag_evidence(details_url: str) -> dict[str, Any] | None:
    """Return evidence for a completed-success workflow whose job record is still pending.

    GitHub can leave a check-run/job lifecycle status in ``in_progress`` after the parent
    workflow and every job step have completed successfully. This is diagnostic evidence only:
    callers keep the CI rollup fail-closed as pending and use the returned fields to distinguish
    status propagation lag from ordinary work still running.
    """
    match = _ACTIONS_JOB_URL_RE.search(details_url)
    if match is None:
        return None

    run_id = match.group("run_id")
    job_id = match.group("job_id")
    run = _rest_api_get(f"actions/runs/{run_id}")
    job = _rest_api_get(f"actions/jobs/{job_id}")
    if not isinstance(run, dict) or not isinstance(job, dict):
        return None
    if str(run.get("status", "") or "").lower() != "completed":
        return None
    if str(run.get("conclusion", "") or "").lower() != "success":
        return None
    if str(job.get("status", "") or "").lower() != "in_progress":
        return None

    steps = job.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    if any(
        not isinstance(step, dict)
        or str(step.get("status", "") or "").lower() != "completed"
        or str(step.get("conclusion", "") or "").lower() not in _TERMINAL_STEP_CONCLUSIONS
        for step in steps
    ):
        return None
    final_step = steps[-1]
    if (
        not isinstance(final_step, dict)
        or str(final_step.get("name", "") or "") != "Complete job"
        or str(final_step.get("conclusion", "") or "").lower() != "success"
    ):
        return None

    return {
        "run_id": int(run_id),
        "job_id": int(job_id),
        "parent_run_status": "completed",
        "parent_run_conclusion": "success",
        "job_status": "in_progress",
        "final_step": "Complete job",
        "final_step_conclusion": "success",
    }


def _status_propagation_lag_details(rollup: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inspect pending GitHub Actions checks for completed-success propagation lag."""
    details: list[dict[str, Any]] = []
    for check in rollup:
        if _rollup_status(check) not in PENDING_STATUSES:
            continue
        details_url = str(check.get("detailsUrl") or check.get("targetUrl") or "")
        evidence = _status_propagation_lag_evidence(details_url)
        if evidence is not None:
            details.append({"name": _rollup_name(check), "details_url": details_url, **evidence})
    return details


def _annotate_status_propagation_lag(
    checks: dict[str, Any], rollup: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach a distinct, fail-closed status for stale successful workflow job records."""
    lag_details = _status_propagation_lag_details(rollup)
    if not lag_details:
        return checks
    pending_count = sum(_rollup_status(check) in PENDING_STATUSES for check in rollup)
    checks["status_propagation_lag"] = lag_details
    checks["diagnostic"] = "check_run_stale_job_success"
    if len(lag_details) == pending_count:
        checks["pending_reason"] = "status_propagation_lag"
    return checks


def _summarize_check_runs(check_runs: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Build the ``checks`` summary (and superseded count) from raw REST check-run dicts."""
    rollup = [
        {
            "__typename": "CheckRun",
            "name": str(run.get("name", "") or ""),
            "status": str(run.get("status", "") or ""),
            "conclusion": run.get("conclusion") or "",
            "detailsUrl": str(run.get("details_url", "") or ""),
            "startedAt": str(run.get("started_at", "") or ""),
            "workflowName": str(
                (run.get("workflow") or (run.get("check_suite") or {}).get("workflow") or "")
                if isinstance(run, dict)
                else ""
            ),
        }
        for run in check_runs
        if isinstance(run, dict)
    ]
    effective, superseded_count = _latest_check_runs(rollup)
    conclusions: dict[str, int] = {}
    states: dict[str, int] = {}
    name_counts: dict[str, int] = {}
    for check in effective:
        conclusion = _rollup_conclusion(check)
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
        status = _rollup_status(check)
        states[status] = states.get(status, 0) + 1
        name = _rollup_name(check)
        name_counts[name] = name_counts.get(name, 0) + 1
    failure_count = sum(conclusions.get(c, 0) for c in FAILURE_CONCLUSIONS)
    pending_count = sum(states.get(s, 0) for s in PENDING_STATUSES)
    overall = (
        "failure" if failure_count else ("pending" if pending_count or not effective else "success")
    )
    details = [
        {
            "name": _rollup_name(check),
            "status": _rollup_status(check),
            "conclusion": _rollup_conclusion(check),
            "details_url": check.get("detailsUrl", "") or check.get("targetUrl", ""),
        }
        for check in effective
    ]
    checks = {
        "total": len(effective),
        "superseded": superseded_count,
        "overall": overall,
        "by_conclusion": conclusions,
        "by_status": states,
        "names": sorted(name_counts),
        "details": details,
    }
    _annotate_status_propagation_lag(checks, effective)
    return checks, superseded_count


def _fetch_ci_status_rest(pr_number: str) -> dict[str, Any]:
    """Build the CI status payload from REST when GraphQL quota is exhausted (issue #6564)."""
    pull = _rest_api_get(f"pulls/{pr_number}")
    if not isinstance(pull, dict):
        return {
            "status": "error",
            "error_kind": "graphql_quota_exhausted",
            "error": "GraphQL quota exhausted and REST pull fallback failed",
        }
    head = pull.get("head") or {}
    head_sha = str(head.get("sha", "") or "")
    checks_payload = _rest_api_get(f"commits/{head_sha}/check-runs")
    check_runs = checks_payload.get("check_runs", []) if isinstance(checks_payload, dict) else []
    checks, _ = _summarize_check_runs(check_runs if isinstance(check_runs, list) else [])
    reviews_raw = _rest_api_get(f"pulls/{pr_number}/reviews")
    review_states: dict[str, int] = {}
    for review in reviews_raw if isinstance(reviews_raw, list) else []:
        if isinstance(review, dict):
            state = str(review.get("state", "UNKNOWN") or "UNKNOWN")
            review_states[state] = review_states.get(state, 0) + 1
    return {
        "status": "ok",
        "pr": pull.get("number"),
        "title": str(pull.get("title", "") or ""),
        "state": str(pull.get("state", "unknown") or "unknown"),
        "mergeable": str(pull.get("mergeable_state", "unknown") or "unknown").upper(),
        "branch": str(head.get("ref", "") or ""),
        "head_sha": head_sha,
        "checks": checks,
        "reviews": review_states,
        "data_source": "rest_fallback_graphql_quota",
        "route_evidence_only": True,
    }


def _gh_view_error_payload(pr_number: str, stderr: str, returncode: int) -> dict[str, Any]:
    """Map a failed ``gh pr view`` to an error payload, falling back to REST on GraphQL quota."""
    if _is_graphql_quota_error(stderr):
        return _fetch_ci_status_rest(pr_number)
    return {"status": "error", "error": stderr or f"gh returned exit code {returncode}"}


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
        return _gh_view_error_payload(pr_number, stderr, result.returncode)

    data, parse_error = _parse_pr_view_json(result.stdout)
    if parse_error or data is None:
        return {
            "status": "error",
            "error": parse_error or "gh output is not a JSON object",
        }
    raw_rollup = data.get("statusCheckRollup", []) or []
    rollup, superseded_count = _latest_check_runs(raw_rollup)

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
    checks = {
        "total": len(rollup),
        "superseded": superseded_count,
        "overall": overall,
        "by_conclusion": conclusions,
        "by_status": states,
        "names": sorted(name_counts),
        "details": check_details,
    }
    _annotate_status_propagation_lag(checks, rollup)

    return {
        "status": "ok",
        "pr": data.get("number"),
        "title": data.get("title", ""),
        "state": data.get("state", "unknown"),
        "mergeable": data.get("mergeable", "unknown"),
        "branch": data.get("headRefName", ""),
        "head_sha": data.get("headRefOid", ""),
        "checks": checks,
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
    pending_reason = data.get("checks", {}).get("pending_reason")
    if pending_reason:
        data["monitor"]["pending_reason"] = pending_reason
    diagnostic = data.get("checks", {}).get("diagnostic")
    if diagnostic:
        data["monitor"]["diagnostic"] = diagnostic


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
    pending_reason = checks.get("pending_reason")
    if pending_reason:
        lag_details = checks.get("status_propagation_lag", [])
        lag_count = len(lag_details)
        lines.append(f"  pending_reason: {pending_reason}  |  affected checks: {lag_count}")
        for lag in lag_details:
            lines.append(
                f"    - {lag.get('name', 'unknown')}: "
                f"run {lag.get('run_id')} job {lag.get('job_id')}  |  "
                f"{lag.get('final_step', 'unknown')}/{lag.get('final_step_conclusion', 'unknown')}"
            )
    diagnostic = checks.get("diagnostic")
    if diagnostic:
        lines.append(f"  diagnostic: {diagnostic}  |  fail-closed: true")
    superseded = checks.get("superseded", 0)
    if superseded:
        lines.append(f"  ignored {superseded} superseded GitHub Actions check run(s)")
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


def _monitor_terminal_reason(
    data: dict[str, Any],
    *,
    overall: str | None,
    attempt: int,
    attempts: int,
    local_stop: bool,
) -> str | None:
    """Classify a polling stop, preserving a distinct status-propagation diagnosis."""
    terminal_reason = _terminal_reason(overall, attempt, attempts, local_stop)
    if (
        terminal_reason == "attempt_exhausted"
        and data.get("checks", {}).get("pending_reason") == "status_propagation_lag"
    ):
        return "status_propagation_lag"
    return terminal_reason


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
        terminal_reason = _monitor_terminal_reason(
            data,
            overall=overall,
            attempt=attempt,
            attempts=attempts,
            local_stop=local_stop,
        )
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
