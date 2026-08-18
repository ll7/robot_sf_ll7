#!/usr/bin/env python3
"""Check CI status for a GitHub PR using the gh CLI.

Output is compact and cache-friendly.  Use --json for machine-readable output.
Run `--help` for the worktree-safe invocation used by agent workflows.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dev.github_graphql_retry import GraphQLRetryOutcome, run_with_retry  # noqa: E402
from scripts.dev.github_quota import parse_rate_limit_payload  # noqa: E402
from scripts.dev.pr_metadata import metadata_digest, validate_pr_title  # noqa: E402

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
_WORKFLOW_ID_BY_RUN_ID: dict[str, str] = {}
STABILITY_SNAPSHOT_SCHEMA = "pr_stability_snapshot.v1"
_RETRY_AFTER_RE = re.compile(r"retry-after\s*[:=]\s*(\d+)", re.IGNORECASE)
_RESUME_MONITOR_ARGS = "--poll-attempts 40 --poll-interval 30 --max-wall-seconds 1200"


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
    workflow_id = str(check.get("workflowId") or "")
    workflow_name = str(check.get("workflowName") or "")
    started_at = str(check.get("startedAt") or "")
    workflow_identity = f"id:{workflow_id}" if workflow_id else f"name:{workflow_name}"
    if not (workflow_id or workflow_name) or not started_at:
        return None
    return workflow_identity, _rollup_name(check)


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


def _rest_api_get_detailed(path: str, *, timeout: int = 45) -> tuple[Any, str]:
    """Fetch one explicit REST path, returning ``(payload, error_text)``.

    Unlike :func:`_rest_api_get`, the gh error text is preserved so quota-blocked
    calls can be distinguished from ordinary failures without an extra request.
    """
    result = _gh(["api", path], timeout=timeout)
    if result.returncode != 0:
        return None, result.stderr.strip() or f"gh api exited with code {result.returncode}"
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError:
        return None, f"gh api returned invalid JSON for {path}"


def _is_rate_limit_error_text(text: str) -> bool:
    """Return whether gh error text indicates REST/GraphQL rate-limit exhaustion."""
    lowered = (text or "").lower()
    return "rate limit" in lowered or "rate_limit" in lowered or "too many requests" in lowered


def _parse_retry_after(text: str) -> int | None:
    """Parse the first ``Retry-After`` seconds value from gh output, best-effort."""
    match = _RETRY_AFTER_RE.search(text or "")
    if match is None:
        return None
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return None


def _fetch_rate_limit_info() -> dict[str, Any]:
    """Return bounded REST/GraphQL quota state without consuming quota.

    ``gh api rate_limit`` does not count against the API quota. When it is
    unavailable (for example a secondary rate limit), a ``Retry-After`` value
    from the gh error output supplies a bounded resume hint; otherwise the
    source is ``unavailable`` and callers stay fail-closed. Never retries.
    """
    result = _gh(["api", "rate_limit"])
    if result.returncode == 0:
        try:
            snapshot = parse_rate_limit_payload(json.loads(result.stdout))
        except json.JSONDecodeError:
            snapshot = None
        if snapshot is not None and snapshot.status == "ok":
            return {
                "source": "gh_api_rate_limit",
                "graphql_remaining": snapshot.graphql_remaining,
                "graphql_reset_epoch_seconds": snapshot.graphql_reset_at,
                "core_remaining": snapshot.core_remaining,
                "core_reset_epoch_seconds": snapshot.core_reset_at,
            }
    retry_after = _parse_retry_after(result.stderr) or _parse_retry_after(result.stdout)
    if retry_after is not None:
        return {"source": "retry_after", "retry_after_seconds": retry_after}
    return {"source": "unavailable"}


def _rate_limit_resume_hint(info: dict[str, Any], now: int) -> tuple[int | None, int | None]:
    """Return ``(min_delay_seconds, resume_epoch_seconds)`` from rate-limit info."""
    reset = info.get("core_reset_epoch_seconds") or info.get("graphql_reset_epoch_seconds")
    retry_after = info.get("retry_after_seconds")
    min_delay: int | None = None
    if reset is not None:
        min_delay = max(0, int(reset) - now)
    elif retry_after is not None:
        min_delay = max(0, int(retry_after))
    resume_epoch = int(reset) if reset is not None else None
    return min_delay, resume_epoch


def _enrich_rest_check_runs(check_runs: list[Any]) -> list[dict[str, Any]]:
    """Bind GitHub Actions check runs to their authoritative workflow IDs.

    REST check-run payloads omit the GraphQL ``workflowName`` field used to collapse reruns on the
    same commit. Their job URLs identify an Actions run, whose REST payload supplies the stable
    ``workflow_id``. Successful lookups are cached across bounded polling attempts; failed lookups
    are retried. Runs that cannot be enriched remain identity-less and independently fail-closed;
    job names alone are never used for deduplication.
    """
    enriched_runs: list[dict[str, Any]] = []
    for check_run in check_runs:
        if not isinstance(check_run, dict):
            continue
        enriched = dict(check_run)
        details_url = str(check_run.get("details_url", "") or "")
        match = _ACTIONS_JOB_URL_RE.search(details_url)
        if match is None:
            enriched_runs.append(enriched)
            continue

        run_id = match.group("run_id")
        workflow_id = _WORKFLOW_ID_BY_RUN_ID.get(run_id)
        if workflow_id is None:
            run = _rest_api_get(f"actions/runs/{run_id}")
            raw_workflow_id = run.get("workflow_id") if isinstance(run, dict) else None
            workflow_id = str(raw_workflow_id) if raw_workflow_id is not None else None
            if workflow_id is not None:
                _WORKFLOW_ID_BY_RUN_ID[run_id] = workflow_id
        if workflow_id:
            enriched["workflow_id"] = workflow_id
        enriched_runs.append(enriched)
    return enriched_runs


def _stale_job_status(job: dict[str, Any]) -> str | None:
    """Return the eligible stale job lifecycle, or ``None`` for ordinary job state."""
    job_status = str(job.get("status", "") or "").lower()
    if job_status == "in_progress":
        return job_status
    if job_status == "completed" and str(job.get("conclusion", "") or "").lower() == "success":
        return job_status
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
    job_status = _stale_job_status(job)
    if job_status is None:
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

    evidence = {
        "run_id": int(run_id),
        "job_id": int(job_id),
        "parent_run_status": "completed",
        "parent_run_conclusion": "success",
        "job_status": job_status,
        "final_step": "Complete job",
        "final_step_conclusion": "success",
    }
    return evidence


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
            "workflowId": str(run.get("workflow_id", "") or ""),
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


def _fetch_ci_status_rest(
    pr_number: str,
    *,
    fallback_kind: str = "quota",
    fallback_diagnostic: str = "",
) -> dict[str, Any]:
    """Build a route-evidence-only CI payload after a GraphQL read fails.

    REST fallback is authoritative for the check-run and review fields it
    returns, but it cannot replace GraphQL-only review-thread evidence.  The
    caller therefore keeps the source and the missing dimension explicit.
    """
    if fallback_kind == "transient_exhausted":
        error_kind = "graphql_transient_exhausted"
        source = "rest_fallback_graphql_transient"
        failure = "GraphQL transient retry budget exhausted and REST pull fallback failed"
    else:
        error_kind = "graphql_quota_exhausted"
        source = "rest_fallback_graphql_quota"
        failure = "GraphQL quota exhausted and REST pull fallback failed"
    pull = _rest_api_get(f"pulls/{pr_number}")
    if not isinstance(pull, dict):
        return {
            "status": "error",
            "error_kind": error_kind,
            "error": f"{failure}; {fallback_diagnostic}" if fallback_diagnostic else failure,
        }
    head = pull.get("head") or {}
    head_sha = str(head.get("sha", "") or "")
    checks_payload = _rest_api_get(f"commits/{head_sha}/check-runs")
    check_runs = checks_payload.get("check_runs", []) if isinstance(checks_payload, dict) else []
    raw_check_runs = check_runs if isinstance(check_runs, list) else []
    checks, _ = _summarize_check_runs(_enrich_rest_check_runs(raw_check_runs))
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
        "data_source": source,
        "graphql_fallback_diagnostic": fallback_diagnostic,
        "route_evidence_only": True,
    }


def _gh_view_error_payload(
    pr_number: str,
    stderr: str,
    returncode: int,
    *,
    retry: GraphQLRetryOutcome | None = None,
) -> dict[str, Any]:
    """Map a failed GraphQL-backed PR read to a truthful payload."""
    if _is_graphql_quota_error(stderr):
        return _fetch_ci_status_rest(pr_number)
    if retry is not None and retry.exhausted:
        return _fetch_ci_status_rest(
            pr_number,
            fallback_kind="transient_exhausted",
            fallback_diagnostic=retry.terminal_diagnostic,
        )
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

    retry = run_with_retry(
        _gh,
        [
            "pr",
            "view",
            pr_number,
            "--json",
            "number,title,state,mergeable,headRefName,headRefOid,statusCheckRollup,reviews",
        ],
        timeout=30,
    )
    result = retry.result
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return _gh_view_error_payload(
            pr_number,
            stderr,
            result.returncode,
            retry=retry,
        )

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


@dataclass(frozen=True)
class StabilitySnapshotEvidence:
    """Live exact-head evidence evaluated by the pure snapshot evaluator."""

    observed_head_sha: str
    observed_main_sha: str
    base_sha: str
    base_ref: str
    observed_metadata_digest: str
    ci_overall: str
    ci_pending_reason: str
    expected_head_sha: str
    expected_main_sha: str
    expected_metadata_digest: str


def _resolve_ci_state(overall: str, pending_reason: str) -> str:
    """Classify a CI rollup with a distinct status-propagation-lag state."""
    if overall == "success":
        return "success"
    if overall == "failure":
        return "failure"
    if overall == "pending":
        if pending_reason == "status_propagation_lag":
            return "status_propagation_lag"
        return "pending"
    return "unknown"


def _snapshot_resume_command(
    status: str,
    *,
    reasons: list[str],
    evidence: StabilitySnapshotEvidence,
    pr: str,
    desired_hint: str = "",
    min_delay_seconds: int | None = None,
) -> dict[str, Any]:
    """Return the smallest safe resume command for a snapshot status.

    Route evidence only: the command never retries automatically and never
    authorizes a merge. Movement invalidation always resumes with a fresh
    snapshot against the observed values; pending/lag states resume with the
    bounded CI monitor (exit code 2 until checks settle).
    """
    if status == "stable":
        return {"command": None, "reason": "none", "min_delay_seconds": None}
    head = evidence.observed_head_sha or "<observed-head-sha>"
    main = evidence.observed_main_sha or "<observed-main-sha>"
    digest = evidence.observed_metadata_digest or "<observed-digest>"
    snapshot_cmd = (
        f"scripts/dev/check_pr_ci_status.py {pr} --stability-snapshot --json "
        f"--expected-head-sha {head} --expected-main-sha {main} "
        f"--expected-metadata-digest {digest}"
    )
    if status == "changed":
        if "metadata_digest_changed" in reasons and desired_hint:
            return {
                "command": (
                    f"scripts/dev/gh_pr_body_rest.py {pr} --reconcile {desired_hint} "
                    f"&& {snapshot_cmd}"
                ),
                "reason": "reconcile_metadata_then_rerun",
                "min_delay_seconds": None,
            }
        return {
            "command": snapshot_cmd,
            "reason": "refresh_expecteds_and_rerun",
            "min_delay_seconds": None,
        }
    if status in {"pending", "status_propagation_lag"}:
        return {
            "command": (
                f"scripts/dev/check_pr_ci_status.py {pr} --json "
                f"--expected-head-sha {head} {_RESUME_MONITOR_ARGS}"
            ),
            "reason": "bounded_ci_wait",
            "min_delay_seconds": None,
        }
    if status == "quota_blocked":
        return {
            "command": snapshot_cmd,
            "reason": "rest_rate_limit_reset",
            "min_delay_seconds": min_delay_seconds,
        }
    return {"command": snapshot_cmd, "reason": "rerun_after_fix", "min_delay_seconds": None}


def _snapshot_status(reasons: list[str], ci_state: str) -> str:
    """Resolve the snapshot status with movement invalidation taking precedence."""
    if reasons:
        return "changed"
    if ci_state == "failure":
        return "failure"
    if ci_state == "status_propagation_lag":
        return "status_propagation_lag"
    if ci_state == "pending":
        return "pending"
    if ci_state == "success":
        return "stable"
    return "error"


def evaluate_stability_snapshot(
    evidence: StabilitySnapshotEvidence,
    *,
    pr: str,
    repo: str,
    head_read_race: bool = False,
    desired_metadata_digest: str = "",
    desired_hint: str = "",
) -> dict[str, Any]:
    """Evaluate a live exact-head snapshot against expected evidence (pure).

    Movement of the head, current main, or metadata digest invalidates the
    snapshot with the observed values and a resume command; the snapshot never
    retries automatically and never authorizes a merge.
    """
    reasons: list[str] = []
    head_matches: bool | None = None
    main_matches: bool | None = None
    digest_matches: bool | None = None
    if evidence.expected_head_sha:
        head_matches = bool(evidence.observed_head_sha) and (
            evidence.observed_head_sha == evidence.expected_head_sha
        )
        if not head_matches:
            reasons.append("head_sha_changed")
    if evidence.expected_main_sha:
        main_matches = bool(evidence.observed_main_sha) and (
            evidence.observed_main_sha == evidence.expected_main_sha
        )
        if not main_matches:
            reasons.append("main_sha_changed")
    if evidence.expected_metadata_digest:
        digest_matches = bool(evidence.observed_metadata_digest) and (
            evidence.observed_metadata_digest == evidence.expected_metadata_digest
        )
        if not digest_matches:
            reasons.append("metadata_digest_changed")
    if head_read_race:
        reasons.append("head_sha_read_race")

    ci_state = _resolve_ci_state(evidence.ci_overall, evidence.ci_pending_reason)
    status = _snapshot_status(reasons, ci_state)

    metadata_evidence: dict[str, Any] = {
        "observed_digest": evidence.observed_metadata_digest,
        "expected_digest": evidence.expected_metadata_digest or None,
        "digest_matches": digest_matches,
    }
    if desired_metadata_digest:
        metadata_evidence["desired_digest"] = desired_metadata_digest

    return {
        "schema": STABILITY_SNAPSHOT_SCHEMA,
        "status": status,
        "route_evidence_only": True,
        "pr": pr,
        "repo": repo,
        "head_sha": evidence.observed_head_sha,
        "expected_head_sha": evidence.expected_head_sha or None,
        "head_sha_matches": head_matches,
        "main_sha": evidence.observed_main_sha,
        "expected_main_sha": evidence.expected_main_sha or None,
        "main_sha_matches": main_matches,
        "base_sha": evidence.base_sha or None,
        "base_ref": evidence.base_ref,
        "metadata": metadata_evidence,
        "ci_state": ci_state,
        "invalidated": bool(reasons),
        "invalidated_reasons": reasons,
        "resume": _snapshot_resume_command(
            status,
            reasons=reasons,
            evidence=evidence,
            pr=pr,
            desired_hint=desired_hint,
        ),
    }


def _quota_blocked_snapshot(
    pr: str,
    repo: str,
    *,
    diagnostic: str,
    rate_limit: dict[str, Any] | None = None,
    expected_head_sha: str = "",
    expected_main_sha: str = "",
    expected_metadata_digest: str = "",
) -> dict[str, Any]:
    """Build the quota-blocked snapshot with a bounded resume time and no retry."""
    info = rate_limit if rate_limit is not None else _fetch_rate_limit_info()
    min_delay, resume_epoch = _rate_limit_resume_hint(info, int(time.time()))
    head = expected_head_sha or "<observed-head-sha>"
    main = expected_main_sha or "<observed-main-sha>"
    digest = expected_metadata_digest or "<observed-digest>"
    command = (
        f"scripts/dev/check_pr_ci_status.py {pr} --stability-snapshot --json "
        f"--expected-head-sha {head} --expected-main-sha {main} "
        f"--expected-metadata-digest {digest}"
    )
    return {
        "schema": STABILITY_SNAPSHOT_SCHEMA,
        "status": "quota_blocked",
        "route_evidence_only": True,
        "pr": pr,
        "repo": repo,
        "error": diagnostic,
        "rate_limit": info,
        "resume": {
            "command": command,
            "reason": "rest_rate_limit_reset",
            "min_delay_seconds": min_delay,
            "resume_epoch_seconds": resume_epoch,
        },
    }


def _read_text_file(path: Path) -> tuple[str | None, str | None]:
    """Read a UTF-8 text file, returning ``(text, error)``."""
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, f"could not read {path}: {exc}"


def _snapshot_error_payload(pr: str, repo: str, message: str) -> dict[str, Any]:
    """Build the fail-closed snapshot error payload."""
    return {
        "schema": STABILITY_SNAPSHOT_SCHEMA,
        "status": "error",
        "route_evidence_only": True,
        "pr": pr,
        "repo": repo,
        "error": message,
    }


def _desired_metadata_evidence(
    *,
    metadata_title: str | None,
    metadata_body_file: Path | None,
    pr: str,
    repo: str,
) -> tuple[str, str, dict[str, Any] | None]:
    """Return ``(desired_digest, desired_hint, error_payload)`` for a desired metadata pair."""
    if metadata_title is None and metadata_body_file is None:
        return "", "", None
    title = metadata_title or ""
    title_error = validate_pr_title(title)
    if title_error:
        return "", "", _snapshot_error_payload(pr, repo, title_error)
    if metadata_body_file is None:
        body = ""
    else:
        body, body_error = _read_text_file(metadata_body_file)
        if body_error:
            return "", "", _snapshot_error_payload(pr, repo, body_error)
        assert body is not None
    hint = f"--title {shlex.quote(title)} --body-file {shlex.quote(str(metadata_body_file))}"
    return metadata_digest(title, body), hint, None


def _fetch_stability_snapshot(
    pr: str,
    *,
    repo: str,
    expected_head_sha: str,
    expected_main_sha: str,
    expected_metadata_digest: str,
    metadata_title: str | None = None,
    metadata_body_file: Path | None = None,
) -> dict[str, Any]:
    """Fetch one deterministic exact-head stability snapshot (route evidence only).

    The snapshot reads the live CI status, the live PR title/body/base, the
    current ``main`` SHA, and the REST quota state exactly once each. It never
    polls, never retries automatically, and never mutates anything.
    """
    if not repo:
        owner, name = _git_remote_owner_name()
        repo = f"{owner}/{name}" if owner and name else ""
    if not repo:
        return _snapshot_error_payload(
            pr,
            "",
            "repository could not be derived from the git remote; pass --repo owner/name",
        )

    ci = _fetch_ci_status(pr)
    if ci.get("status") == "error":
        error_text = str(ci.get("error", "") or "")
        quota_kinds = {"graphql_quota_exhausted", "graphql_transient_exhausted"}
        if ci.get("error_kind") in quota_kinds or _is_rate_limit_error_text(error_text):
            return _quota_blocked_snapshot(
                pr,
                repo,
                diagnostic=error_text,
                expected_head_sha=expected_head_sha,
                expected_main_sha=expected_main_sha,
                expected_metadata_digest=expected_metadata_digest,
            )
        return _snapshot_error_payload(pr, repo, error_text or "CI status read failed")

    desired_digest, desired_hint, metadata_error = _desired_metadata_evidence(
        metadata_title=metadata_title,
        metadata_body_file=metadata_body_file,
        pr=pr,
        repo=repo,
    )
    if metadata_error is not None:
        return metadata_error

    rate_limit = _fetch_rate_limit_info()
    pull, pull_error = _rest_api_get_detailed(f"repos/{repo}/pulls/{pr}")
    main_payload, main_error = _rest_api_get_detailed(f"repos/{repo}/branches/main")
    error_text = " ".join(text for text in (pull_error, main_error) if text).strip()
    if pull_error or main_error:
        if _is_rate_limit_error_text(error_text):
            return _quota_blocked_snapshot(
                pr,
                repo,
                diagnostic=error_text,
                rate_limit=rate_limit,
                expected_head_sha=expected_head_sha,
                expected_main_sha=expected_main_sha,
                expected_metadata_digest=expected_metadata_digest,
            )
        payload = _snapshot_error_payload(pr, repo, error_text or "live PR/main read failed")
        payload["rate_limit"] = rate_limit
        return payload

    pull = pull if isinstance(pull, dict) else {}
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    base = pull.get("base") if isinstance(pull.get("base"), dict) else {}
    observed_head = str(head.get("sha", "") or "")
    observed_digest = metadata_digest(
        str(pull.get("title", "") or ""), str(pull.get("body", "") or "")
    )
    commit = main_payload.get("commit") if isinstance(main_payload, dict) else None
    observed_main = str(commit.get("sha", "") or "") if isinstance(commit, dict) else ""

    ci_head = str(ci.get("head_sha", "") or "")
    head_read_race = bool(ci_head) and ci_head != observed_head

    evidence = StabilitySnapshotEvidence(
        observed_head_sha=observed_head,
        observed_main_sha=observed_main,
        base_sha=str(base.get("sha", "") or ""),
        base_ref=str(base.get("ref", "") or ""),
        observed_metadata_digest=observed_digest,
        ci_overall=str(ci.get("checks", {}).get("overall", "") or ""),
        ci_pending_reason=str(ci.get("checks", {}).get("pending_reason", "") or ""),
        expected_head_sha=expected_head_sha,
        expected_main_sha=expected_main_sha,
        expected_metadata_digest=expected_metadata_digest,
    )
    result = evaluate_stability_snapshot(
        evidence,
        pr=pr,
        repo=repo,
        head_read_race=head_read_race,
        desired_metadata_digest=desired_digest,
        desired_hint=desired_hint,
    )
    result["checks"] = ci.get("checks", {})
    result["reviews"] = ci.get("reviews", {})
    result["rate_limit"] = rate_limit
    if ci.get("data_source"):
        result["data_source"] = ci["data_source"]
    remaining = rate_limit.get("core_remaining")
    if remaining is not None and remaining <= 0 and result["status"] == "stable":
        result["warning"] = (
            "REST quota exhausted; evidence is fresh but resume commands fail until reset"
        )
    return result


def _validate_expected_main_sha(expected_main_sha: str) -> str | None:
    """Return an error message when ``--expected-main-sha`` is not a full 40-hex SHA."""
    if not expected_main_sha:
        return None
    if len(expected_main_sha) != 40 or not re.fullmatch(r"[0-9a-fA-F]{40}", expected_main_sha):
        return (
            "--expected-main-sha must be the full 40-hex SHA, got "
            f"{len(expected_main_sha)} chars ({expected_main_sha!r}); "
            "short prefixes are not accepted"
        )
    return None


def _validate_expected_metadata_digest(expected_digest: str) -> str | None:
    """Return an error when ``--expected-metadata-digest`` is not a 64-hex SHA-256 digest."""
    if not expected_digest:
        return None
    if len(expected_digest) != 64 or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_digest):
        return (
            "--expected-metadata-digest must be the full 64-hex SHA-256 digest, got "
            f"{len(expected_digest)} chars ({expected_digest!r}); "
            "short prefixes are not accepted"
        )
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


def _validate_expected_head_sha(expected_head_sha: str) -> str | None:
    """Return an error message when ``--expected-head-sha`` is not a full 40-hex SHA.

    A short prefix or other malformed value is rejected up front so the monitor reports a clear
    format error instead of a spurious "PR head SHA changed" on the first poll (issue #7505).
    """
    if not expected_head_sha:
        return None
    if len(expected_head_sha) != 40 or not re.fullmatch(r"[0-9a-fA-F]{40}", expected_head_sha):
        return (
            "--expected-head-sha must be the full 40-hex SHA, got "
            f"{len(expected_head_sha)} chars ({expected_head_sha!r}); short prefixes are not accepted"
        )
    return None


def _preflight_validate(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> str | None:
    """Return an exit-1 preflight error message, or None when arguments are acceptable.

    A conflicting PR number still raises SystemExit(2) via ``parser.error``, matching prior behavior.
    """
    if args.pr_number and args.pr_number_option and args.pr_number != args.pr_number_option:
        parser.error(
            "conflicting PR numbers: pass either positional <pr-number> or --pr <number>, "
            "or pass the same value to both"
        )
    head_error = _validate_expected_head_sha(args.expected_head_sha)
    if head_error:
        return head_error
    if args.stability_snapshot:
        if args.poll_attempts > 1 or args.backoff > 0 or args.max_wall_seconds is not None:
            parser.error(
                "--stability-snapshot is a single-read snapshot that never polls; "
                "remove --poll-attempts > 1, --backoff, or --max-wall-seconds"
            )
        if (args.metadata_title is None) != (args.metadata_body_file is None):
            parser.error("--metadata-title and --metadata-body-file must be provided together")
        main_error = _validate_expected_main_sha(args.expected_main_sha)
        if main_error:
            return main_error
        return _validate_expected_metadata_digest(args.expected_metadata_digest)
    if args.expected_main_sha or args.expected_metadata_digest:
        parser.error(
            "--expected-main-sha and --expected-metadata-digest require --stability-snapshot"
        )
    if args.metadata_title is not None or args.metadata_body_file is not None:
        parser.error("--metadata-title and --metadata-body-file require --stability-snapshot")
    return None


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


def _print_snapshot_result(data: dict[str, Any]) -> int:
    """Print one snapshot JSON document and return its exit code.

    Exit codes: 0 stable, 1 changed/failure/error, 2 inconclusive
    (pending/status-propagation-lag/quota-blocked; resume later).
    """
    print(json.dumps(data))
    status = data.get("status")
    if status == "stable":
        return 0
    if status in {"changed", "failure", "error"}:
        return 1
    return 2


def _fetch_data(args: argparse.Namespace, pr: str) -> tuple[dict[str, Any], int]:
    """Fetch CI status or the stability snapshot, returning ``(data, attempts)``."""
    if args.stability_snapshot:
        return (
            _fetch_stability_snapshot(
                pr,
                repo=args.repo,
                expected_head_sha=args.expected_head_sha,
                expected_main_sha=args.expected_main_sha,
                expected_metadata_digest=args.expected_metadata_digest,
                metadata_title=args.metadata_title,
                metadata_body_file=args.metadata_body_file,
            ),
            1,
        )
    attempts = max(1, args.poll_attempts)
    return (
        _poll_ci_status(
            pr,
            attempts=attempts,
            poll_interval=args.poll_interval,
            backoff=args.backoff,
            json_output=args.json,
            expected_head_sha=args.expected_head_sha,
            max_wall_seconds=args.max_wall_seconds,
        ),
        attempts,
    )


def _emit_ci_result(data: dict[str, Any], args: argparse.Namespace, attempts: int) -> int:
    """Print CI data and return the exit code (0 success / 1 failure / 2 pending-timeout)."""
    if args.stability_snapshot:
        return _print_snapshot_result(data)
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

Exact-head stability snapshot (issue #7523), route evidence only:

  scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/check_pr_ci_status.py \\
      <pr-number> --stability-snapshot --json \\
      --expected-head-sha <head-sha> --expected-main-sha <current-main-sha> \\
      --expected-metadata-digest <64-hex> --repo ll7/robot_sf_ll7

The snapshot is one deterministic read of head/main/metadata-digest/CI/quota
state; it never retries automatically and never authorizes a merge. Snapshot exit
codes: 0 stable, 1 changed/failure/error, 2 inconclusive
(pending/status-propagation-lag/quota-blocked; resume later).
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
        "--stability-snapshot",
        action="store_true",
        default=False,
        help=(
            "emit one deterministic exact-head stability snapshot (schema "
            "pr_stability_snapshot.v1) covering head/main/metadata-digest/CI/quota "
            "route evidence; single read, never polls, never mutates"
        ),
    )
    parser.add_argument(
        "--repo",
        default="",
        help="owner/name REST target (default: derive from the origin git remote)",
    )
    parser.add_argument(
        "--expected-main-sha",
        default="",
        help="optional current-main SHA guard for --stability-snapshot",
    )
    parser.add_argument(
        "--expected-metadata-digest",
        default="",
        help="optional 64-hex title/body metadata digest guard for --stability-snapshot",
    )
    parser.add_argument(
        "--metadata-title",
        default=None,
        help="desired final title for --stability-snapshot metadata comparison",
    )
    parser.add_argument(
        "--metadata-body-file",
        type=Path,
        default=None,
        help="desired final body file for --stability-snapshot metadata comparison",
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
    preflight_error = _preflight_validate(args, parser)
    if preflight_error is not None:
        print(f"error: {preflight_error}", file=sys.stderr)
        return 1
    pr_number = args.pr_number_option or args.pr_number

    try:
        pr = _resolve_pr_number(pr_number)
        data, attempts = _fetch_data(args, pr)
    except FileNotFoundError:
        print("gh CLI not found. Install GitHub CLI: https://cli.github.com/", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print(
            "gh CLI command timed out. Check your network connection or GitHub status.",
            file=sys.stderr,
        )
        return 1

    return _emit_ci_result(data, args, attempts)


if __name__ == "__main__":
    raise SystemExit(main())
