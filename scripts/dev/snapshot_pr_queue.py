#!/usr/bin/env python3
"""Emit compact PR queue state for token-efficient goal orchestration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev.check_pr_ci_status import (
    FAILURE_CONCLUSIONS,
    PENDING_STATUSES,
    _latest_check_runs,
    _rollup_conclusion,
    _rollup_name,
    _rollup_status,
)
from scripts.dev.pr_loop_policy import GATE_VERDICT_RE
from scripts.dev.pr_metadata import extract_metadata_digests, metadata_digest, metadata_trailer

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_ACTIVE_LIMIT = 20
REVIEW_SUMMARY_LIMIT = 4
COMMENT_SUMMARY_LIMIT = 4
COMMENT_BODY_LIMIT = 180
REVIEW_THREAD_LIMIT = 12
REVIEW_THREAD_COMMENT_LIMIT = 2
ROUTE_HEALTH_STATUSES = ("healthy", "stale", "blocked", "unknown")
SCHEMA_VERSION = "pr_queue_snapshot.v2"
BLOCKING_LABELS = frozenset(
    {
        "blocked",
        "decision-required",
        "evidence:blocked",
        "state:blocked",
        "state:blocked-external-input",
        "state:hold",
    }
)
BLOCKED_NEXT_ACTION = "await_blocker_owner_or_approval"
_ACTIONS_RUN_JOB_URL_RE = re.compile(r"/actions/runs/(?P<run_id>[0-9]+)/job(?:/|$)")


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _labels(pr: dict[str, Any]) -> list[str]:
    """Return compact label names from gh PR JSON."""
    return sorted(
        str(label.get("name", ""))
        for label in pr.get("labels", [])
        if isinstance(label, dict) and label.get("name")
    )


def _shorten_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * max(limit, 0)
    return text[: limit - 3].rstrip() + "..."


def _author_login(author: Any) -> str:
    if isinstance(author, dict):
        return str(author.get("login", "") or author.get("name", "") or "")
    return ""


def _reviews(pr: dict[str, Any]) -> dict[str, int]:
    """Return review-state counts."""
    states: dict[str, int] = {}
    for review in pr.get("reviews", []) or []:
        if not isinstance(review, dict):
            continue
        state = str(review.get("state", "UNKNOWN"))
        states[state] = states.get(state, 0) + 1
    return states


def _review_snapshot(pr: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded review snapshot with author/time/body excerpts."""
    reviews = [review for review in pr.get("reviews", []) or [] if isinstance(review, dict)]
    by_state: dict[str, int] = {}
    for review in reviews:
        state = str(review.get("state", "UNKNOWN"))
        by_state[state] = by_state.get(state, 0) + 1
    latest = [
        {
            "state": str(review.get("state", "UNKNOWN")),
            "author": _author_login(review.get("author")),
            "author_association": str(review.get("authorAssociation", "")),
            "submitted_at": str(review.get("submittedAt", "")),
            "body_excerpt": _shorten_text(review.get("body"), limit=COMMENT_BODY_LIMIT),
        }
        for review in sorted(
            reviews,
            key=lambda review: str(review.get("submittedAt", review.get("createdAt", ""))),
            reverse=True,
        )[:REVIEW_SUMMARY_LIMIT]
    ]
    return {
        "total": len(reviews),
        "by_state": by_state,
        "latest": latest,
        "contains_more": len(reviews) > REVIEW_SUMMARY_LIMIT,
    }


def _comment_snapshot(pr: dict[str, Any]) -> dict[str, Any]:
    """Return a compact comment snapshot with bounded excerpts."""
    comments = [comment for comment in pr.get("comments", []) or [] if isinstance(comment, dict)]
    latest = [
        {
            "author": _author_login(comment.get("author")),
            "author_association": str(comment.get("authorAssociation", "")),
            "created_at": str(comment.get("createdAt", "")),
            "body_excerpt": _shorten_text(comment.get("body"), limit=COMMENT_BODY_LIMIT),
        }
        for comment in sorted(
            comments,
            key=lambda comment: str(comment.get("createdAt", comment.get("updatedAt", ""))),
            reverse=True,
        )[:COMMENT_SUMMARY_LIMIT]
    ]
    return {
        "total": len(comments),
        "latest": latest,
        "contains_more": len(comments) > COMMENT_SUMMARY_LIMIT,
    }


def _repo_owner_name(repo: str) -> tuple[str, str]:
    """Split an owner/name GitHub repository string."""
    if "/" not in repo:
        return "", repo
    owner, name = repo.split("/", 1)
    return owner, name


def _is_graphql_quota_error(message: str) -> bool:
    """Return whether a gh error message indicates GraphQL API rate-limit/quota exhaustion."""
    text = (message or "").lower()
    if "rate limit" not in text:
        return False
    return "graphql" in text or "api rate limit" in text or "too many requests" in text


def _rest_api_get(path: str, *, repo: str, timeout: int = 45) -> Any:
    """Fetch a REST endpoint under ``repos/{owner}/{name}/{path}`` and parse JSON.

    Returns the parsed payload, or ``None`` on any HTTP/JSON failure so callers can fall back
    gracefully. REST remains available when the authenticated user's GraphQL quota is exhausted
    (issue #6564).
    """
    owner, name = _repo_owner_name(repo)
    if not owner or not name:
        return None
    result = _gh(["api", f"repos/{owner}/{name}/{path}"], timeout=timeout)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _rest_open_pr_list(*, repo: str, limit: int) -> tuple[list[dict[str, Any]], bool] | None:
    """Return a bounded open-PR list through REST and its conservative truncation state.

    The active queue normally uses ``gh pr list`` because it returns the compact GraphQL shape
    directly.  When that call is blocked by GraphQL quota, REST can still enumerate open PRs.  A
    result that fills the requested limit remains marked truncated because REST cannot prove that
    no additional page exists without consuming another page request.
    """
    requested_limit = max(int(limit), 1)
    page_size = min(requested_limit, 100)
    page = 1
    rows: list[dict[str, Any]] = []
    last_page_size = 0
    while len(rows) < requested_limit:
        payload = _rest_api_get(
            f"pulls?state=open&per_page={page_size}&page={page}",
            repo=repo,
        )
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            return None
        last_page_size = len(payload)
        rows.extend(payload)
        if len(payload) < page_size:
            break
        page += 1
    truncated = len(rows) >= requested_limit and last_page_size >= page_size
    return rows[:requested_limit], truncated


def _rest_check_run_workflow_identity(
    run: dict[str, Any],
    *,
    repo: str,
    cache: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Return workflow id/name for a REST check run, enriching ambiguous rerun metadata.

    The commit check-runs endpoint often omits workflow identity even though its job URL embeds
    the Actions run id. Fetch that run metadata only for duplicate check names in the REST
    fallback, so rerun cancellation can be superseded without conflating distinct workflows that
    happen to share a check name.
    """
    workflow_id = str(run.get("workflow_id", "") or "")
    workflow = run.get("workflow")
    if isinstance(workflow, dict):
        workflow_name = str(workflow.get("name", "") or "")
    else:
        workflow_name = str(workflow or "")
    if not workflow_name:
        check_suite = run.get("check_suite")
        suite_workflow = check_suite.get("workflow") if isinstance(check_suite, dict) else None
        if isinstance(suite_workflow, dict):
            workflow_name = str(suite_workflow.get("name", "") or "")
        elif suite_workflow:
            workflow_name = str(suite_workflow)
    if workflow_id or workflow_name:
        return workflow_id, workflow_name

    details_url = str(run.get("details_url", "") or "")
    match = _ACTIONS_RUN_JOB_URL_RE.search(details_url)
    if match is None:
        return "", ""
    run_id = match.group("run_id")
    if run_id not in cache:
        payload = _rest_api_get(f"actions/runs/{run_id}", repo=repo)
        if isinstance(payload, dict):
            cache[run_id] = (
                str(payload.get("workflow_id", "") or ""),
                str(payload.get("name", "") or ""),
            )
        else:
            cache[run_id] = ("", "")
    return cache[run_id]


def _fetch_current_main_sha(*, repo: str) -> str:
    """Return the current remote ``main`` commit SHA when REST exposes it."""
    payload = _rest_api_get("branches/main", repo=repo)
    if not isinstance(payload, dict):
        return ""
    commit = payload.get("commit")
    if not isinstance(commit, dict):
        return ""
    sha = commit.get("sha")
    return sha if isinstance(sha, str) else ""


def _fetch_pr_base_sha(number: int, *, repo: str) -> str:
    """Return the PR base SHA through the gh-compatible REST pull endpoint."""
    # ``baseRefOid`` is not available in the repository's supported gh 2.45.0
    # JSON field set. The REST pull endpoint is stable across supported gh
    # versions and exposes the same base commit as ``base.sha``.
    payload = _rest_api_get(f"pulls/{number}", repo=repo)
    if not isinstance(payload, dict):
        return ""
    base = payload.get("base")
    if not isinstance(base, dict):
        return ""
    sha = base.get("sha")
    return sha if isinstance(sha, str) else ""


def _base_freshness(*, base_sha: str, current_main_sha: str) -> dict[str, Any]:
    """Return bounded PR base-vs-current-main freshness provenance."""
    if not base_sha:
        verdict = "missing-base"
        action = "verify_pr_base_before_queue_routing"
        reason = "PR base SHA is unavailable from the snapshot source"
    elif not current_main_sha:
        verdict = "unavailable-current-main"
        action = "refresh_current_main_before_queue_routing"
        reason = "current main SHA is unavailable from the snapshot source"
    elif base_sha == current_main_sha:
        verdict = "fresh"
        action = "continue_queue_routing"
        reason = "PR base SHA matches current main"
    else:
        verdict = "stale"
        action = "refresh_pr_base_before_review_or_merge"
        reason = "PR base SHA differs from current main"
    return {
        "base_sha": base_sha or None,
        "current_main_sha": current_main_sha or None,
        "verdict": verdict,
        "action": action,
        "reason": reason,
    }


def _base_freshness_preflight(
    base_freshness: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Return an optional preflight status/reason from a base freshness verdict."""
    verdict = str(base_freshness.get("verdict", ""))
    if verdict == "stale":
        return "stale", "base_sha_stale"
    if verdict == "missing-base":
        return "blocked", "base_sha_missing"
    if verdict == "unavailable-current-main":
        return "blocked", "current_main_sha_unavailable"
    return None, None


def _base_freshness_next_action(reasons: Any) -> str | None:
    """Return the required freshness action when preflight reasons include one."""
    if not isinstance(reasons, list):
        return None
    if "base_sha_stale" in reasons:
        return "refresh_pr_base_before_review_or_merge"
    if "base_sha_missing" in reasons:
        return "verify_pr_base_before_queue_routing"
    if "current_main_sha_unavailable" in reasons:
        return "refresh_current_main_before_queue_routing"
    return None


def _blocked_state(labels: list[str]) -> dict[str, Any]:
    """Return explicit blocker evidence and its next owner/gate.

    Labels are the only bounded stop-state evidence available in the compact PR
    query.  Preserve every recognized label so a downstream route cannot turn a
    green-but-blocked PR into review or merge work.
    """
    blocker_labels = [label for label in labels if label in BLOCKING_LABELS]
    if not blocker_labels:
        return {
            "status": "clear",
            "labels": [],
            "reasons": [],
            "next_owner_or_gate": None,
        }

    if "state:blocked-external-input" in blocker_labels:
        next_owner_or_gate = "external_input_owner_or_staging_gate"
    elif "evidence:blocked" in blocker_labels:
        next_owner_or_gate = "evidence_or_domain_approval"
    elif {"decision-required", "state:hold"}.intersection(blocker_labels):
        next_owner_or_gate = "maintainer_decision_or_approval"
    else:
        next_owner_or_gate = "blocker_owner_or_maintainer"
    return {
        "status": "blocked",
        "labels": blocker_labels,
        "reasons": [f"explicit_blocked:{label}" for label in blocker_labels],
        "next_owner_or_gate": next_owner_or_gate,
    }


def _blocked_preflight_reasons(blocked_state: dict[str, Any]) -> list[str]:
    """Return explicit blocker reasons suitable for the preflight envelope."""
    if blocked_state.get("status") != "blocked":
        return []
    return [str(reason) for reason in blocked_state.get("reasons", [])]


def _head_preflight(
    *, expected_head_sha: str, head_sha: str
) -> tuple[str | None, list[str], bool | None]:
    """Return head freshness status, reasons, and exact-match evidence."""
    if not expected_head_sha:
        return None, [], None
    if not head_sha:
        return "blocked", ["missing_head_sha"], None
    if head_sha != expected_head_sha:
        return "stale", ["head_sha_mismatch"], False
    return None, [], True


def _fetch_pr_rest(
    number: int,
    *,
    repo: str,
    expected_head_sha: str,
    current_main_sha: str = "",
) -> dict[str, Any]:
    """Build a compact PR snapshot from REST endpoints when GraphQL quota is exhausted.

    Maps REST pull/reviews/comments/check-runs payloads into the gh-JSON shape consumed by
    ``_pr_payload_from_dict`` so the normal compacting logic is reused unchanged. Review threads
    are GraphQL-only and have no REST endpoint, so the snapshot marks them
    ``unknown_graphql_quota`` with a fail-closed admission note (issue #6564).
    """
    pull = _rest_api_get(f"pulls/{number}", repo=repo)
    if not isinstance(pull, dict):
        return {
            "number": number,
            "status": "error",
            "error_kind": "graphql_quota_exhausted",
            "error": "GraphQL quota exhausted and REST pull fallback failed",
        }
    head = pull.get("head") or {}
    base = pull.get("base") or {}
    head_sha = str(head.get("sha", "") or "")
    base_sha = str(base.get("sha", "") or "")
    reviews_raw = _rest_api_get(f"pulls/{number}/reviews", repo=repo)
    reviews = [
        {
            "state": str(review.get("state", "UNKNOWN") or "UNKNOWN"),
            "authorAssociation": str(review.get("author_association", "") or ""),
            "body": review.get("body", ""),
        }
        for review in (reviews_raw if isinstance(reviews_raw, list) else [])
        if isinstance(review, dict)
    ]
    comments_raw = _rest_api_get(f"issues/{number}/comments", repo=repo)
    comments = [
        {
            "authorAssociation": str(comment.get("author_association", "") or ""),
            "body": comment.get("body", ""),
        }
        for comment in (comments_raw if isinstance(comments_raw, list) else [])
        if isinstance(comment, dict)
    ]
    checks_payload = _rest_api_get(f"commits/{head_sha}/check-runs", repo=repo)
    check_runs = checks_payload.get("check_runs", []) if isinstance(checks_payload, dict) else []
    check_name_counts: dict[str, int] = {}
    for run in check_runs:
        if isinstance(run, dict):
            name = str(run.get("name", "") or "")
            check_name_counts[name] = check_name_counts.get(name, 0) + 1
    workflow_cache: dict[str, tuple[str, str]] = {}
    rollup: list[dict[str, Any]] = []
    for run in check_runs:
        if not isinstance(run, dict):
            continue
        name = str(run.get("name", "") or "")
        workflow_id = ""
        workflow_name = ""
        if check_name_counts.get(name, 0) > 1:
            workflow_id, workflow_name = _rest_check_run_workflow_identity(
                run,
                repo=repo,
                cache=workflow_cache,
            )
        rollup.append(
            {
                "__typename": "CheckRun",
                "name": name,
                "status": str(run.get("status", "") or ""),
                "conclusion": run.get("conclusion") or "",
                "detailsUrl": str(run.get("details_url", "") or ""),
                "startedAt": str(run.get("started_at", "") or ""),
                "workflowId": workflow_id,
                "workflowName": workflow_name,
            }
        )
    pr_dict = {
        "number": pull.get("number", number),
        "title": str(pull.get("title", "") or ""),
        "body": str(pull.get("body", "") or ""),
        "state": str(pull.get("state", "") or ""),
        "isDraft": bool(pull.get("draft")),
        "labels": pull.get("labels", []) if isinstance(pull.get("labels"), list) else [],
        "url": str(pull.get("html_url", "") or ""),
        "headRefName": str(head.get("ref", "") or ""),
        "headRefOid": head_sha,
        "mergeable": str(pull.get("mergeable_state", "unknown") or "unknown").upper(),
        "statusCheckRollup": rollup,
        "reviews": reviews,
        "comments": comments,
    }
    payload = _pr_payload_from_dict(
        pr_dict,
        base_sha=base_sha,
        current_main_sha=current_main_sha or _fetch_current_main_sha(repo=repo),
        default_number=number,
        expected_head_sha=expected_head_sha,
    )
    payload["data_source"] = "rest_fallback_graphql_quota"
    payload["review_threads"] = "unknown_graphql_quota"
    payload["review_threads_admission"] = "fail_closed_unknown"
    payload["route_evidence_only"] = True
    return payload


def _dict_or_empty(value: Any) -> dict[str, Any]:
    """Return *value* when it is a dictionary, otherwise an empty dictionary."""
    return value if isinstance(value, dict) else {}


def _review_thread_snapshot(
    pr_number: int,
    *,
    repo: str,
) -> dict[str, Any]:
    """Return compact PR review-thread data without raw diff hunks or full bodies."""
    owner, name = _repo_owner_name(repo)
    if not owner or not name:
        return {"status": "skipped", "reason": "repo_owner_missing"}
    query = """
query($owner:String!,$repo:String!,$number:Int!,$threads:Int!,$comments:Int!){
  repository(owner:$owner,name:$repo){
    pullRequest(number:$number){
      reviewThreads(first:$threads){
        totalCount
        nodes{
          id
          isResolved
          path
          line
          comments(first:$comments){
            totalCount
            nodes{
              author{login}
              body
              createdAt
            }
          }
        }
      }
    }
  }
}
"""
    result = _gh(
        [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={name}",
            "-F",
            f"number={pr_number}",
            "-F",
            f"threads={REVIEW_THREAD_LIMIT}",
            "-F",
            f"comments={REVIEW_THREAD_COMMENT_LIMIT}",
        ],
        timeout=45,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if _is_graphql_quota_error(stderr):
            return {
                "status": "unknown_graphql_quota",
                "unresolved": None,
                "guidance": (
                    "GraphQL quota exhausted; review threads are GraphQL-only and cannot be "
                    "refreshed via REST. Never admit a PR to merge-ready from this snapshot."
                ),
            }
        return {
            "status": "error",
            "error": stderr or f"gh returned exit code {result.returncode}",
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": f"invalid gh JSON: {exc}"}
    data = _dict_or_empty(payload.get("data"))
    repository = _dict_or_empty(data.get("repository"))
    pull_request = _dict_or_empty(repository.get("pullRequest"))
    threads = _dict_or_empty(pull_request.get("reviewThreads"))
    nodes = [node for node in threads.get("nodes", []) or [] if isinstance(node, dict)]
    compact_threads: list[dict[str, Any]] = []
    unresolved_count = 0
    for node in nodes:
        resolved = bool(node.get("isResolved"))
        if not resolved:
            unresolved_count += 1
        comments = node.get("comments", {}) if isinstance(node.get("comments"), dict) else {}
        comment_nodes = [
            comment for comment in comments.get("nodes", []) or [] if isinstance(comment, dict)
        ]
        compact_threads.append(
            {
                "id": str(node.get("id", "")),
                "resolved": resolved,
                "path": str(node.get("path", "")),
                "line": node.get("line"),
                "comments_total": int(comments.get("totalCount", len(comment_nodes)) or 0),
                "comments": [
                    {
                        "author": _author_login(comment.get("author")),
                        "created_at": str(comment.get("createdAt", "")),
                        "body_excerpt": _shorten_text(
                            comment.get("body"), limit=COMMENT_BODY_LIMIT
                        ),
                        "body_omitted": len(str(comment.get("body") or "")) > COMMENT_BODY_LIMIT,
                    }
                    for comment in comment_nodes
                ],
                "diff_hunk_omitted": True,
            }
        )
    total = int(threads.get("totalCount", len(nodes)) or 0)
    return {
        "status": "ok",
        "total": total,
        "unresolved": unresolved_count,
        "threads": compact_threads,
        "contains_more": total > REVIEW_THREAD_LIMIT,
        "raw_diff_hunks_omitted": True,
    }


def _checks(pr: dict[str, Any]) -> dict[str, Any]:
    """Return a compact CI check summary from statusCheckRollup.

    Superseded GitHub Actions runs (an older run replaced by a newer one on the
    same workflow/job identity) are dropped first, matching the canonical
    `_latest_check_runs` semantics used by `check_pr_ci_status`.  A current,
    non-superseded cancellation remains fail-closed.
    """
    raw_rollup = [
        check for check in (pr.get("statusCheckRollup", []) or []) if isinstance(check, dict)
    ]
    rollup, superseded_count = _latest_check_runs(raw_rollup)
    conclusions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    names: set[str] = set()
    failed: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    for check in rollup:
        if not isinstance(check, dict):
            continue
        conclusion = _rollup_conclusion(check)
        status = _rollup_status(check)
        details_url = check.get("detailsUrl")
        if details_url is None:
            details_url = check.get("targetUrl")
        if details_url is None:
            details_url = ""
        detail = {
            "name": _rollup_name(check),
            "status": status,
            "conclusion": conclusion,
            "details_url": str(details_url),
        }
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
        names.add(detail["name"])
        if conclusion in FAILURE_CONCLUSIONS:
            failed.append(detail)
        elif status in PENDING_STATUSES:
            pending.append(detail)
    failure_count = sum(conclusions.get(conclusion, 0) for conclusion in FAILURE_CONCLUSIONS)
    pending_count = sum(statuses.get(status, 0) for status in PENDING_STATUSES)
    if failure_count:
        overall = "failure"
    elif pending_count or not rollup:
        overall = "pending"
    else:
        overall = "success"
    return {
        "overall": overall,
        "total": len(rollup),
        "superseded": superseded_count,
        "by_conclusion": conclusions,
        "by_status": statuses,
        "names": sorted(names),
        "failed": failed,
        "pending": pending,
    }


def _preflight(
    *,
    base_freshness: dict[str, Any],
    checks_overall: str,
    expected_head_sha: str,
    head_sha: str,
    is_draft: bool,
    mergeable: str,
    blocked_state: dict[str, Any],
) -> dict[str, Any]:
    """Return compact lane preflight status and reasons."""
    reasons: list[str] = []
    if is_draft:
        reasons.append("pr_is_draft")
    preflight_status = "healthy"
    head_status, head_reasons, head_sha_matches = _head_preflight(
        expected_head_sha=expected_head_sha,
        head_sha=head_sha,
    )
    if head_status:
        preflight_status = head_status
    reasons.extend(head_reasons)
    base_status, base_reason = _base_freshness_preflight(base_freshness)
    if base_status:
        preflight_status = base_status
    if base_reason:
        reasons.append(base_reason)
    if checks_overall == "failure":
        preflight_status = "blocked"
        reasons.append("ci_checks_failed")
    if mergeable == "CONFLICTING":
        preflight_status = "blocked"
        reasons.append("mergeable_conflict")
    blocked_reasons = _blocked_preflight_reasons(blocked_state)
    if blocked_reasons:
        preflight_status = "blocked"
        reasons.extend(blocked_reasons)
    if not reasons:
        reasons.append("ok")
    return {
        "status": preflight_status,
        "reasons": reasons,
        "expected_head_sha": expected_head_sha,
        "head_sha": head_sha,
        "head_sha_matches_expected": head_sha_matches,
        "base_freshness": base_freshness,
        "checks_overall": checks_overall,
        "mergeable": mergeable,
        "blocked_state": blocked_state,
        "route_evidence_only": True,
    }


def _next_action(
    *, is_draft: bool, labels: list[str], checks: dict[str, Any], preflight: dict[str, Any]
) -> str:
    """Return a compact next-action hint for the parent orchestrator."""
    blocked_state = preflight.get("blocked_state", {})
    if isinstance(blocked_state, dict) and blocked_state.get("status") == "blocked":
        return BLOCKED_NEXT_ACTION
    base_action = _base_freshness_next_action(preflight.get("reasons", []))
    if base_action:
        return base_action
    status = str(preflight.get("status", "unknown"))
    if status == "stale":
        return "invalidate_stale_lane"
    if status == "blocked":
        return "inspect_blocking_preflight"
    if checks.get("overall") == "failure":
        return "inspect_failing_checks"
    if checks.get("overall") == "pending":
        return "await_ci_or_start_read_only_monitor"
    if "merge-ready" in labels and not is_draft:
        return "merge_readiness_local_check"
    if is_draft:
        return "review_or_mark_ready_when_local_proof_passes"
    return "review_for_merge_ready"


def _attention(*, next_action: str, is_draft: bool, labels: list[str]) -> str:
    """Return a compact attention category for queue triage."""
    if next_action == "invalidate_stale_lane":
        return "stale_attention"
    if next_action == "inspect_blocking_preflight":
        return "preflight_attention"
    if next_action == BLOCKED_NEXT_ACTION:
        return "blocked_attention"
    if is_draft:
        return "draft_ready_or_review"
    if next_action == "inspect_failing_checks":
        return "ci_attention"
    if next_action == "await_ci_or_start_read_only_monitor":
        return "ci_pending"
    if "merge-ready" in labels:
        return "merge_attention"
    return "review_attention"


def _parse_explicit_verdict(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        verdict = str(item.get("verdict", "")).lower()
        accepted_flag = item.get("accepted")
        sha = str(item.get("sha") or item.get("head_sha") or "")
        if sha and (verdict == "accepted" or accepted_flag is True):
            return f"gate-verdict: accepted @ {sha}"
    return None


_TRUSTED_GATE_VERDICT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


def _extract_trailers_from_bodies(items: Any) -> list[str]:
    """Extract verdicts only from repository-trusted comment or review authors."""
    trailers: list[str] = []
    if not isinstance(items, list):
        return trailers
    for entry in items:
        if isinstance(entry, dict):
            association = str(entry.get("authorAssociation", "")).upper()
            if association not in _TRUSTED_GATE_VERDICT_ASSOCIATIONS:
                continue
            body = entry.get("body")
            if isinstance(body, str) and body:
                for match in GATE_VERDICT_RE.finditer(body):
                    trailers.append(f"gate-verdict: accepted @ {match.group(1)}")
    return trailers


def _extract_gate_verdicts(pr: dict[str, Any]) -> list[str]:
    """Extract trusted structured gate-verdict trailers from raw comment and review bodies."""
    verdicts: list[str] = []

    existing_list = pr.get("gate_verdicts")
    if isinstance(existing_list, list):
        for item in existing_list:
            parsed = _parse_explicit_verdict(item)
            if parsed:
                verdicts.append(parsed)

    parsed_single = _parse_explicit_verdict(pr.get("gate_verdict"))
    if parsed_single:
        verdicts.append(parsed_single)

    verdicts.extend(_extract_trailers_from_bodies(pr.get("reviews")))
    verdicts.extend(_extract_trailers_from_bodies(pr.get("comments")))

    return list(dict.fromkeys(verdicts))


def _extract_metadata_verdicts(pr: dict[str, Any]) -> list[str]:  # noqa: C901
    """Extract trusted final title/body reconciliation trailers."""
    verdicts: list[str] = []
    existing_list = pr.get("metadata_verdicts")
    if isinstance(existing_list, list):
        for item in existing_list:
            if isinstance(item, str):
                verdicts.extend(
                    metadata_trailer(digest) for digest in extract_metadata_digests(item)
                )
            elif isinstance(item, dict):
                digest = str(item.get("digest") or item.get("metadata_digest") or "")
                verdict = str(item.get("verdict", "")).lower()
                if digest and (
                    verdict in {"accepted", "reconciled"} or item.get("accepted") is True
                ):
                    verdicts.append(metadata_trailer(digest))
    single = pr.get("metadata_verdict")
    if isinstance(single, str):
        verdicts.extend(metadata_trailer(digest) for digest in extract_metadata_digests(single))
    elif isinstance(single, dict):
        digest = str(single.get("digest") or single.get("metadata_digest") or "")
        verdict = str(single.get("verdict", "")).lower()
        if digest and (verdict in {"accepted", "reconciled"} or single.get("accepted") is True):
            verdicts.append(metadata_trailer(digest))

    for items in (pr.get("reviews"), pr.get("comments")):
        if not isinstance(items, list):
            continue
        for entry in items:
            if not isinstance(entry, dict):
                continue
            association = str(entry.get("authorAssociation", "")).upper()
            if association not in _TRUSTED_GATE_VERDICT_ASSOCIATIONS:
                continue
            body = entry.get("body")
            if isinstance(body, str):
                verdicts.extend(
                    metadata_trailer(digest) for digest in extract_metadata_digests(body)
                )
    return list(dict.fromkeys(verdicts))


def _pr_payload_from_dict(
    pr: dict[str, Any],
    *,
    base_sha: str,
    current_main_sha: str,
    default_number: int,
    expected_head_sha: str,
) -> dict[str, Any]:
    """Build a compact PR snapshot from already-loaded fields."""
    is_draft = bool(pr.get("isDraft"))
    labels = _labels(pr)
    checks = _checks(pr)
    head_sha = str(pr.get("headRefOid", "") or pr.get("head_sha", ""))
    mergeable = str(pr.get("mergeable", "unknown"))
    blocked_state = _blocked_state(labels)
    base_freshness = _base_freshness(
        base_sha=base_sha,
        current_main_sha=current_main_sha,
    )
    preflight = _preflight(
        base_freshness=base_freshness,
        checks_overall=str(checks.get("overall", "")),
        expected_head_sha=expected_head_sha,
        head_sha=head_sha,
        is_draft=is_draft,
        mergeable=mergeable,
        blocked_state=blocked_state,
    )
    reviews = _reviews(pr)
    gate_verdicts = _extract_gate_verdicts(pr)
    title = str(pr.get("title", "") or "")
    body = str(pr.get("body", "") or "")
    metadata_digest_value = metadata_digest(title, body)
    metadata_verdicts = _extract_metadata_verdicts(pr)
    pr_payload = {
        "number": pr.get("number", default_number),
        "status": "ok",
        "title": title,
        "state": pr.get("state", ""),
        "draft": is_draft,
        "url": pr.get("url", ""),
        "labels": labels,
        "head_branch": pr.get("headRefName", ""),
        "head_sha": head_sha,
        "base_freshness": base_freshness,
        "mergeable": mergeable,
        "checks": checks,
        "reviews": reviews,
        "gate_verdicts": gate_verdicts,
        "metadata_digest": metadata_digest_value,
        "metadata_verdicts": metadata_verdicts,
        "review_snapshot": _review_snapshot(pr),
        "comment_snapshot": _comment_snapshot(pr),
        "preflight": preflight,
    }

    next_action = _next_action(
        is_draft=is_draft,
        labels=labels,
        checks=checks,
        preflight=preflight,
    )
    pr_payload["next_action"] = next_action
    pr_payload["attention"] = _attention(
        next_action=next_action,
        is_draft=is_draft,
        labels=labels,
    )
    return pr_payload


def _route_health_overview(prs: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize route health across PR snapshots."""
    counts = dict.fromkeys(ROUTE_HEALTH_STATUSES, 0)
    for pr in prs:
        preflight = pr.get("preflight", {})
        status = str(preflight.get("status", "unknown"))
        if status not in counts:
            status = "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def fetch_pr(
    number: int,
    *,
    repo: str,
    current_main_sha: str = "",
    expected_head_sha: str = "",
) -> dict[str, Any]:
    """Fetch one PR and return a compact queue snapshot."""
    result = _gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,body,state,isDraft,labels,url,headRefName,headRefOid,mergeable,statusCheckRollup,reviews,comments",
        ]
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if _is_graphql_quota_error(stderr):
            return _fetch_pr_rest(number, repo=repo, expected_head_sha=expected_head_sha)
        return {
            "number": number,
            "status": "error",
            "error": stderr or f"gh returned exit code {result.returncode}",
        }
    try:
        pr = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"number": number, "status": "error", "error": f"invalid gh JSON: {exc}"}
    base_sha = str(pr.get("base_sha", "") or "")
    if not base_sha:
        base_sha = _fetch_pr_base_sha(number, repo=repo)
    current_main_sha = (
        current_main_sha
        or str(pr.get("current_main_sha", "") or "")
        or _fetch_current_main_sha(repo=repo)
    )
    return _pr_payload_from_dict(
        pr,
        base_sha=base_sha,
        current_main_sha=current_main_sha,
        default_number=number,
        expected_head_sha=expected_head_sha,
    )


def _active_snapshot_envelope(
    *,
    repo: str,
    prs: list[dict[str, Any]],
    truncated: bool,
    truncation_note: str,
    data_source: str | None = None,
) -> dict[str, Any]:
    """Build the common active-queue envelope for GraphQL and REST routes."""
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "repo": repo,
        "mode": "active",
        "truncated": truncated,
        "truncation_note": truncation_note,
        "route_health_overview": _route_health_overview(prs),
        "prs": prs,
    }
    if data_source:
        payload["data_source"] = data_source
    return payload


def snapshot_active_prs(*, repo: str, limit: int) -> dict[str, Any]:
    """Return a compact active PR queue snapshot."""
    current_main_sha = _fetch_current_main_sha(repo=repo)
    result = _gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            str(limit),
            "--json",
            "number,title,body,state,isDraft,labels,url,headRefName,headRefOid,mergeable,statusCheckRollup,reviews,comments",
        ]
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if _is_graphql_quota_error(stderr):
            rest_listing = _rest_open_pr_list(repo=repo, limit=limit)
            if rest_listing is not None:
                listed, truncated = rest_listing
                prs: list[dict[str, Any]] = []
                for listed_pr in listed:
                    number = listed_pr.get("number")
                    if isinstance(number, bool) or not isinstance(number, int):
                        return _active_snapshot_envelope(
                            repo=repo,
                            prs=[
                                {
                                    "status": "error",
                                    "error_kind": "rest_payload_malformed",
                                    "error": "REST active PR list contained a row without an integer number",
                                }
                            ],
                            truncated=False,
                            truncation_note="",
                        )
                    prs.append(
                        _fetch_pr_rest(
                            number,
                            repo=repo,
                            expected_head_sha="",
                            current_main_sha=current_main_sha,
                        )
                    )
                return _active_snapshot_envelope(
                    repo=repo,
                    prs=prs,
                    truncated=truncated,
                    truncation_note=(
                        "REST open-PR list may be capped: got "
                        f"{len(prs)} rows at --limit {limit}; raise --limit or paginate"
                        if truncated
                        else ""
                    ),
                    data_source="rest_fallback_graphql_quota",
                )
            return _active_snapshot_envelope(
                repo=repo,
                prs=[
                    {
                        "status": "error",
                        "error_kind": "graphql_quota_exhausted",
                        "error": "GraphQL quota exhausted and REST open-PR list fallback failed",
                    }
                ],
                truncated=False,
                truncation_note="",
            )
        return _active_snapshot_envelope(
            repo=repo,
            prs=[
                {
                    "status": "error",
                    "error": stderr or f"gh returned exit code {result.returncode}",
                }
            ],
            truncated=False,
            truncation_note="",
        )
    try:
        listed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "schema": SCHEMA_VERSION,
            "repo": repo,
            "mode": "active",
            "truncated": False,
            "truncation_note": "",
            "route_health_overview": {"healthy": 0, "stale": 0, "blocked": 0, "unknown": 0},
            "prs": [
                {
                    "status": "error",
                    "error": f"invalid gh JSON: {exc}",
                }
            ],
        }
    if not isinstance(listed, list):
        return {
            "schema": SCHEMA_VERSION,
            "repo": repo,
            "mode": "active",
            "truncated": False,
            "truncation_note": "",
            "route_health_overview": {"healthy": 0, "stale": 0, "blocked": 0, "unknown": 0},
            "prs": [
                {
                    "status": "error",
                    "error": "expected gh pr list JSON array",
                }
            ],
        }

    prs = [
        _pr_payload_from_dict(
            pr,
            base_sha=str(pr.get("base_sha", "") or "")
            or _fetch_pr_base_sha(int(pr.get("number", -1)), repo=repo),
            current_main_sha=current_main_sha or str(pr.get("current_main_sha", "") or ""),
            default_number=-1,
            expected_head_sha="",
        )
        for pr in listed
        if isinstance(pr, dict)
    ]
    truncated = is_likely_truncated(len(listed), limit=limit)
    return _active_snapshot_envelope(
        repo=repo,
        prs=prs,
        truncated=truncated,
        truncation_note=(
            "gh pr list may be capped: got "
            f"{len(listed)} rows at --limit {limit}; raise --limit or paginate"
            if truncated
            else ""
        ),
    )


def snapshot_prs(
    numbers: list[int],
    *,
    repo: str,
    expected_head_sha: str = "",
    include_review_threads: bool = False,
) -> dict[str, Any]:
    """Return a compact PR queue snapshot."""
    current_main_sha = _fetch_current_main_sha(repo=repo)
    prs = [
        fetch_pr(
            number,
            repo=repo,
            current_main_sha=current_main_sha,
            expected_head_sha=expected_head_sha,
        )
        for number in numbers
    ]
    if include_review_threads:
        for pr in prs:
            if pr.get("status") == "ok" and isinstance(pr.get("number"), int):
                pr["review_thread_snapshot"] = _review_thread_snapshot(
                    int(pr["number"]),
                    repo=repo,
                )
    return {
        "schema": SCHEMA_VERSION,
        "repo": repo,
        "route_health_overview": _route_health_overview(prs),
        "prs": prs,
    }


def write_raw_review_comments_artifact(
    numbers: list[int],
    *,
    repo: str,
    path: Path,
) -> dict[str, Any]:
    """Write opt-in raw review-comment payloads, including diff hunks, to an artifact."""
    owner, name = _repo_owner_name(repo)
    payload: dict[str, Any] = {
        "schema": "raw_pr_review_comments.v1",
        "repo": repo,
        "prs": {},
    }
    if not owner or not name:
        for number in numbers:
            payload["prs"][str(number)] = {
                "status": "error",
                "error": "repo_owner_missing",
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    for number in numbers:
        result = _gh(
            [
                "api",
                f"repos/{owner}/{name}/pulls/{number}/comments",
            ],
            timeout=60,
        )
        if result.returncode != 0:
            payload["prs"][str(number)] = {
                "status": "error",
                "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
            }
            continue
        try:
            comments = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            payload["prs"][str(number)] = {
                "status": "error",
                "error": f"invalid gh JSON: {exc}",
            }
            continue
        payload["prs"][str(number)] = {
            "status": "ok",
            "comments": comments,
            "contains_raw_diff_hunks": True,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_snapshot_artifact(payload: dict[str, Any], path: Path | None) -> None:
    """Write a compact queue snapshot to *path* with stable JSON formatting."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def emit_snapshot(
    payload: dict[str, Any], path: Path | None, *, as_json: bool, exit_code: int
) -> int:
    """Write an optional snapshot artifact, emit stdout, and return the CLI status."""
    try:
        write_snapshot_artifact(payload, path)
    except OSError as exc:
        print(f"snapshot output write failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True) if as_json else json.dumps(payload))
    return exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prs", nargs="*", type=int, help="PR numbers to snapshot.")
    parser.add_argument(
        "--active",
        action="store_true",
        help="Discover bounded open PRs that need queue attention.",
    )
    parser.add_argument("--prs", dest="prs_option", nargs="+", type=int, help="PR numbers.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_ACTIVE_LIMIT,
        help="Limit for --active discovery mode.",
    )
    parser.add_argument(
        "--expected-head-sha",
        default="",
        help="Optional PR head SHA expected for stale-lane invalidation in single-PR mode.",
    )
    parser.add_argument(
        "--review-threads",
        action="store_true",
        help="Include bounded review-thread excerpts without diff hunks or full bodies.",
    )
    parser.add_argument(
        "--raw-review-comments-artifact",
        type=Path,
        help=(
            "Opt-in path for raw review-comment payloads, including diff_hunk/full bodies; "
            "artifact is written to disk and never printed to stdout."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the compact queue snapshot JSON to this path.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.active and (args.prs_option is not None or args.prs):
        print("--active cannot be combined with explicit PR numbers", file=sys.stderr)
        return 1
    if args.active and args.review_threads:
        print("--review-threads is only supported with explicit PR numbers", file=sys.stderr)
        return 1
    if args.active and args.raw_review_comments_artifact:
        print(
            "--raw-review-comments-artifact is only supported with explicit PR numbers",
            file=sys.stderr,
        )
        return 1
    numbers = args.prs_option if args.prs_option is not None else args.prs
    if args.expected_head_sha and not args.active and numbers and len(numbers) != 1:
        print(
            "--expected-head-sha requires exactly one PR number; omit it for batch snapshots",
            file=sys.stderr,
        )
        return 1
    try:
        if args.active:
            payload = snapshot_active_prs(repo=args.repo, limit=max(args.limit, 1))
        elif not numbers:
            print("at least one PR number is required", file=sys.stderr)
            return 1
        else:
            payload = snapshot_prs(
                numbers,
                repo=args.repo,
                expected_head_sha=args.expected_head_sha,
                include_review_threads=args.review_threads,
            )
            if args.raw_review_comments_artifact:
                artifact_payload = write_raw_review_comments_artifact(
                    numbers,
                    repo=args.repo,
                    path=args.raw_review_comments_artifact,
                )
                payload["raw_review_comments_artifact"] = str(args.raw_review_comments_artifact)
                payload["raw_review_comments_artifact_status"] = (
                    "error"
                    if any(pr.get("status") == "error" for pr in artifact_payload["prs"].values())
                    else "ok"
                )
    except FileNotFoundError:
        print("gh command not found", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired as exc:
        print(f"snapshot command timed out: {exc}", file=sys.stderr)
        return 1
    has_pr_errors = any(pr.get("status") == "error" for pr in payload["prs"])
    has_artifact_errors = payload.get("raw_review_comments_artifact_status") == "error"
    return emit_snapshot(
        payload,
        args.output,
        as_json=args.json,
        exit_code=1 if has_pr_errors or has_artifact_errors else 0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
