#!/usr/bin/env python3
"""Plan and execute guarded stacked pull-request workflows.

The helper treats a stack as an ordered list from root to tip.  It supports
five deliberately small operations:

``status``
    Read the live PRs, current check runs, review evidence, and stack-base
    alignment into a machine-readable snapshot.
``retarget``
    Plan or apply base-ref changes (root -> ``main``; each child -> its
    parent's source branch) with an exact-head guard.
``sync``
    Fetch, merge the preceding remote branch into each local stack branch, and
    push without force.  The operation is restricted to a clean worktree.
``merge-cascade``
    Squash-merge one green root at a time.  After a merge it verifies whether
    GitHub retargeted the next PR; otherwise it explicitly retargets that PR
    and stops until fresh CI is available.
``check-ancestry``
    Fail-closed ancestry gate for issue #7515.  Computes the live
    ``origin/main`` merge base for a branch or PR head, enumerates the commits
    and changed paths introduced through non-``main`` ancestry, and classifies
    the branch as ``clean``, ``stacked``, ``undeclared_stack``,
    ``mismatched_declaration``, ``parent_invalidated``, or ``parent_merged``
    against one machine-readable ``## Stack Declaration`` (``parent_pr`` +
    ``parent_head``).  Blocking states exit non-zero with the full diagnostic
    block (actual base, merge base, unexpected commits/paths, declared parent,
    remediation).  This is the same check the pre-PR gate
    (``check_prepublication_state.py``) and the readiness classifier
    (``pr_loop_policy.py``) run before ``merge-ready``.

This is a workflow coordinator, not a replacement for the repository's merge
gate.  All mutating operations are dry-run by default and require ``--apply``.
Every apply path requires caller-supplied expected head SHAs and rechecks them
before writing.  GitHub access uses ``gh api`` so the helper does not depend on
the deprecated Projects Classic fields queried by some ``gh pr`` commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Make direct execution import the repository's canonical gate helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dev.merge_queue_gate import GATE_WORKFLOW_NAME, fetch_threads_resolved  # noqa: E402
from scripts.dev.pr_loop_policy import (  # noqa: E402
    current_explicit_merge_hold_reasons,
    has_current_accepted_gate_verdict,
    has_current_pr_metadata_verdict,
)
from scripts.dev.pr_metadata import metadata_digest  # noqa: E402
from scripts.dev.single_account_merge_receipt import (  # noqa: E402
    CLOSING_DISCIPLINE_SOURCES,
    apply_guarded_merge,
    build_closing_discipline_evidence,
    build_receipt_from_stack_entry,
    classify_implementation_review,
    derive_holds,
    normalize_required_checks,
    verify_receipt,
)
from scripts.dev.snapshot_pr_queue import (  # noqa: E402
    _extract_gate_verdicts,
    _extract_metadata_verdicts,
)
from scripts.dev.stack_ancestry import (  # noqa: E402
    StackDeclaration,
    ancestry_state,
    collect_ancestry_facts,
    parse_stack_declaration,
    remediation_command,
    render_diagnostics,
)

SCHEMA = "stacked_prs.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
SUCCESS_CONCLUSIONS = frozenset({"neutral", "skipped", "success"})
MERGE_QUEUE_GATE_CHECK_NAME = "merge-queue-gate"
REST_PAGE_SIZE = 100
REST_PAGE_BUDGET = 100
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_ACTIONS_RUN_JOB_PATH_RE = re.compile(
    r"^/actions/runs/(?P<run_id>[0-9]+)/job/(?P<job_id>[0-9]+)/?$"
)

GhApi = Callable[[str, str, dict[str, Any] | None], tuple[Any | None, str | None]]
GitRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


def _run_gh_api(
    method: str, path: str, payload: dict[str, Any] | None = None, *, timeout: int = 45
) -> tuple[Any | None, str | None]:
    """Run one bounded ``gh api`` call and decode its JSON response."""
    args = ["gh", "api"]
    if method != "GET":
        args.extend(["--method", method, path, "--input", "-"])
    else:
        args.append(path)
    try:
        result = subprocess.run(
            args,
            input=json.dumps(payload) if method != "GET" else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return None, "gh CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return None, f"gh api timed out after {timeout} seconds"
    if result.returncode != 0:
        return None, result.stderr.strip() or f"gh api exited with code {result.returncode}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"gh api returned invalid JSON: {exc}"


def _git(args: list[str], worktree: Path) -> subprocess.CompletedProcess[str]:
    """Run one git command in an explicit worktree."""
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _object(value: Any, *, operation: str) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an API object response."""
    if not isinstance(value, dict):
        return None, f"{operation} response was not an object"
    return value, None


def _list(value: Any, *, operation: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate an API list response and retain object entries only."""
    if not isinstance(value, list):
        return None, f"{operation} response was not a list"
    if any(not isinstance(item, dict) for item in value):
        return None, f"{operation} response contains a non-object item"
    return value, None


def _get_object(path: str, *, api: GhApi = _run_gh_api) -> tuple[dict[str, Any] | None, str | None]:
    value, error = api("GET", path, None)
    if error:
        return None, error
    return _object(value, operation=path)


def _get_paginated_list(
    path: str,
    *,
    api: GhApi = _run_gh_api,
    response_key: str | None = None,
    page_budget: int = REST_PAGE_BUDGET,
) -> tuple[list[dict[str, Any]] | None, dict[str, int | bool] | None, str | None]:
    """Read a bounded REST collection and fail closed on possible truncation.

    GitHub's review/comment endpoints return a JSON list, while the commit
    check-runs endpoint wraps that list in an object.  The first request keeps
    the historical path for deterministic fixtures and compatibility; later
    pages add ``page=N``.  A short page is the only successful end-of-results
    signal.  Reaching the page budget with a full page is reported as a
    possible truncation instead of being treated as complete.
    """
    if page_budget < 1:
        return None, None, "REST pagination page budget must be positive"
    rows: list[dict[str, Any]] = []
    pages_read = 0
    for page in range(1, page_budget + 1):
        page_path = path if page == 1 else f"{path}&page={page}"
        value, error = api("GET", page_path, None)
        if error:
            return None, None, error
        if response_key is not None and isinstance(value, dict):
            page_value = value.get(response_key)
        else:
            page_value = value
        page_rows, error = _list(page_value, operation=page_path)
        if error or page_rows is None:
            return None, None, error or f"{page_path} returned no collection"
        rows.extend(page_rows)
        pages_read = page
        if len(page_rows) < REST_PAGE_SIZE:
            return (
                rows,
                {
                    "pages_read": pages_read,
                    "page_size": REST_PAGE_SIZE,
                    "page_budget": page_budget,
                    "row_count": len(rows),
                    "truncated": False,
                },
                None,
            )
    return (
        None,
        {
            "pages_read": pages_read,
            "page_size": REST_PAGE_SIZE,
            "page_budget": page_budget,
            "row_count": len(rows),
            "truncated": True,
        },
        (
            f"{path} reached the REST pagination page budget ({page_budget}) with full pages; "
            "response may be truncated"
        ),
    )


def _positive_prs(prs: list[int]) -> tuple[list[int] | None, str | None]:
    """Validate and preserve stack order."""
    if not prs:
        return None, "at least one PR is required"
    if any(number < 1 for number in prs):
        return None, "PR numbers must be positive"
    if len(set(prs)) != len(prs):
        return None, "a stack cannot contain duplicate PR numbers"
    return prs, None


def _parse_expected_heads(values: list[str]) -> tuple[dict[int, str] | None, str | None]:
    """Parse repeated ``PR=SHA`` exact-head guards."""
    expected: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            return None, f"expected head must use PR=SHA syntax: {value!r}"
        number_text, sha = value.split("=", 1)
        try:
            number = int(number_text)
        except ValueError:
            return None, f"expected-head PR is not an integer: {number_text!r}"
        if number < 1 or not _SHA_RE.fullmatch(sha):
            return None, f"expected-head must contain a positive PR and 7-40 hex SHA: {value!r}"
        if number in expected:
            return None, f"duplicate expected-head guard for PR #{number}"
        expected[number] = sha.lower()
    return expected, None


def _labels(raw: Any) -> list[str]:
    """Normalize REST label objects."""
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str) and item:
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]:
            names.append(item["name"])
    return sorted(set(names))


def _author_association(item: dict[str, Any]) -> str:
    """Return the REST/GraphQL-compatible author association value."""
    value = item.get("authorAssociation", item.get("author_association", ""))
    return str(value or "").upper()


def _body_entries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize REST author-association spelling for canonical verdict helpers."""
    normalized: list[dict[str, Any]] = []
    for item in items:
        copied = dict(item)
        copied["authorAssociation"] = _author_association(item)
        normalized.append(copied)
    return normalized


def _review_digest(
    reviews: list[dict[str, Any]],
    review_comments: list[dict[str, Any]],
    conversation_comments: list[dict[str, Any]],
) -> str:
    """Hash review state without putting review text in the status snapshot."""

    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in items:
            body = str(item.get("body") or "")
            user = item.get("user") if isinstance(item.get("user"), dict) else {}
            records.append(
                {
                    "id": item.get("id"),
                    "state": item.get("state"),
                    "commit_id": item.get("commit_id"),
                    "in_reply_to_id": item.get("in_reply_to_id"),
                    "user": user.get("login"),
                    "association": _author_association(item),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                }
            )
        return sorted(records, key=lambda record: (str(record.get("id")), str(record)))

    encoded = json.dumps(
        {
            "reviews": compact(reviews),
            "review_comments": compact(review_comments),
            "conversation_comments": compact(conversation_comments),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_run_identifier(item: dict[str, Any]) -> int | None:
    """Return a positive REST check-run ID, or ``None`` for malformed data."""
    raw_identifier = item.get("id")
    if isinstance(raw_identifier, bool):
        return None
    if isinstance(raw_identifier, int):
        identifier = raw_identifier
    elif isinstance(raw_identifier, str) and raw_identifier.strip().isdigit():
        identifier = int(raw_identifier.strip())
    else:
        return None
    return identifier if identifier > 0 else None


def _check_run_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    """Return a newest-first-compatible identity key for a REST check run.

    Check-run IDs are monotonic, while queued runs may not have timestamps yet.
    Prefer the ID so a newer queued run cannot be hidden by an older completed
    run. Treat malformed or missing IDs as newer than valid IDs so they cannot
    silently hide behind historical evidence; use the timestamp only to order
    multiple malformed records deterministically.
    """
    timestamp = str(item.get("completed_at") or item.get("started_at") or "")
    identifier = _check_run_identifier(item)
    return (0, identifier, timestamp) if identifier is not None else (1, 0, timestamp)


def _latest_check_runs(check_runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the newest check run for each name; count superseded runs."""

    def check_name(item: dict[str, Any]) -> str:
        app = item.get("app") if isinstance(item.get("app"), dict) else {}
        return str(item.get("name") or app.get("name") or "unknown")

    latest: dict[str, dict[str, Any]] = {}
    for item in check_runs:
        name = check_name(item)
        current = latest.get(name)
        if current is None or _check_run_sort_key(item) >= _check_run_sort_key(current):
            latest[name] = item
    return sorted(latest.values(), key=lambda item: str(item.get("name") or "")), max(
        0, len(check_runs) - len(latest)
    )


def _check_workflow_name(item: dict[str, Any]) -> str:
    """Return a workflow name from REST or GraphQL-shaped check-run data."""
    direct = item.get("workflowName") or item.get("workflow_name")
    if direct:
        return str(direct)
    workflow = item.get("workflow")
    workflow_name = (
        str(workflow.get("name") or "") if isinstance(workflow, Mapping) else str(workflow or "")
    )
    if workflow_name:
        return workflow_name
    check_suite = item.get("check_suite")
    suite_workflow = check_suite.get("workflow") if isinstance(check_suite, Mapping) else None
    if isinstance(suite_workflow, Mapping):
        return str(suite_workflow.get("name") or "")
    return str(suite_workflow or "")


def _latest_merge_queue_gate_index(check_runs: list[dict[str, Any]]) -> int | None:
    """Return the index of the newest merge-queue-gate check run, if present."""
    candidates = [
        (index, item)
        for index, item in enumerate(check_runs)
        if str(item.get("name") or "").strip().lower() == MERGE_QUEUE_GATE_CHECK_NAME
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: _check_run_sort_key(candidate[1]))[0]


def _actions_run_id(details_url: str, *, repo: str) -> str:
    """Extract a numeric Actions run ID from this repository's canonical job URL."""
    parsed_url = urlparse(details_url)
    expected_repo_path = f"/{repo.strip('/')}"
    if parsed_url.scheme != "https" or parsed_url.netloc.lower() != "github.com":
        return ""
    path = parsed_url.path
    if not path.casefold().startswith(f"{expected_repo_path}/".casefold()):
        return ""
    match = _ACTIONS_RUN_JOB_PATH_RE.fullmatch(path[len(expected_repo_path) :])
    return match.group("run_id") if match is not None else ""


def _workflow_run_payload(
    run_id: str,
    *,
    repo: str,
    api: GhApi,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Fetch a workflow run, caching only a complete dictionary response."""
    if run_id in cache:
        return cache[run_id]
    payload, error = _get_object(f"repos/{repo}/actions/runs/{run_id}", api=api)
    if error or not isinstance(payload, dict):
        return None
    return payload


def _resolve_check_run_workflow_name(
    check_run: dict[str, Any],
    *,
    repo: str,
    api: GhApi,
    cache: dict[str, dict[str, Any]],
) -> str:
    """Resolve a check run's workflow name without guessing from its job name."""
    workflow_name = _check_workflow_name(check_run)
    if workflow_name:
        return workflow_name
    details_url = str(check_run.get("details_url") or check_run.get("detailsUrl") or "")
    run_id = _actions_run_id(details_url, repo=repo)
    if not run_id:
        return ""
    check_head_sha = str(check_run.get("head_sha") or "").strip()
    if not check_head_sha:
        return ""
    workflow_run = _workflow_run_payload(run_id, repo=repo, api=api, cache=cache)
    if workflow_run is None:
        return ""
    run_head_sha = str(workflow_run.get("head_sha") or "").strip()
    if not run_head_sha or run_head_sha.casefold() != check_head_sha.casefold():
        return ""
    resolved_name = str(workflow_run.get("name") or "").strip()
    if not resolved_name:
        return ""
    if run_id not in cache:
        cache[run_id] = workflow_run
    return resolved_name


def _enrich_merge_queue_gate_check_runs(
    check_runs: list[dict[str, Any]],
    *,
    repo: str,
    api: GhApi,
    cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add authoritative workflow identity to the newest gate check run when available."""
    latest_index = _latest_merge_queue_gate_index(check_runs)
    if latest_index is None:
        return check_runs
    workflow_name = _resolve_check_run_workflow_name(
        check_runs[latest_index], repo=repo, api=api, cache=cache
    )
    if not workflow_name:
        return check_runs
    enriched = [dict(item) for item in check_runs]
    enriched[latest_index]["workflow_name"] = workflow_name
    return enriched


def summarize_check_runs(check_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify current check runs as success, pending, failure, or unknown."""
    current, superseded_count = _latest_check_runs(check_runs)
    normalized: list[dict[str, Any]] = []
    failures: list[str] = []
    pending: list[str] = []
    for item in current:
        app = item.get("app") if isinstance(item.get("app"), dict) else {}
        name = str(item.get("name") or app.get("name") or "unknown")
        status = str(item.get("status") or "").lower()
        conclusion = str(item.get("conclusion") or "").lower() or None
        normalized.append(
            {
                "name": name,
                "status": status or "unknown",
                "conclusion": conclusion,
                "started_at": item.get("started_at"),
                "completed_at": item.get("completed_at"),
                "html_url": item.get("html_url"),
            }
        )
        if status != "completed":
            pending.append(name)
        elif conclusion not in SUCCESS_CONCLUSIONS:
            failures.append(name)
    if not current:
        overall = "unknown"
    elif pending:
        overall = "pending"
    elif failures:
        overall = "failure"
    else:
        overall = "success"
    return {
        "overall": overall,
        "runs": normalized,
        "superseded_count": superseded_count,
        "pending": pending,
        "failures": failures,
    }


def summarize_merge_queue_gate(
    check_runs: list[dict[str, Any]], *, head_sha: str
) -> dict[str, Any]:
    """Require the newest exact-head Merge Queue Gate check context."""
    latest_index = _latest_merge_queue_gate_index(check_runs)
    if latest_index is None:
        return {
            "status": "missing",
            "name": MERGE_QUEUE_GATE_CHECK_NAME,
            "head_sha": head_sha,
            "exact_head": False,
        }

    latest = check_runs[latest_index]
    workflow_name = _check_workflow_name(latest)
    reported_head = str(latest.get("head_sha") or "")
    exact_head = bool(reported_head) and reported_head.lower() == head_sha.lower()
    if _check_run_identifier(latest) is None:
        status = "malformed"
    elif workflow_name != GATE_WORKFLOW_NAME:
        status = "mismatch"
    elif not reported_head:
        status = "malformed"
    elif not exact_head:
        status = "mismatch"
    elif str(latest.get("status") or "").lower() != "completed":
        status = "pending"
    elif str(latest.get("conclusion") or "").lower() != "success":
        status = "failure"
    else:
        status = "success"
    return {
        "status": status,
        "name": str(latest.get("name") or MERGE_QUEUE_GATE_CHECK_NAME),
        "head_sha": reported_head or head_sha,
        "exact_head": exact_head,
        "workflow_name": workflow_name or None,
        "check_run_id": latest.get("id"),
        "started_at": latest.get("started_at"),
        "completed_at": latest.get("completed_at"),
        "conclusion": latest.get("conclusion"),
        "html_url": latest.get("html_url"),
    }


def _fetch_pr(
    repo: str, number: int, *, api: GhApi = _run_gh_api
) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch and validate the REST pull-request shape used by this helper."""
    payload, error = _get_object(f"repos/{repo}/pulls/{number}", api=api)
    if error or payload is None:
        return None, error or "pull-request response was empty"
    head = payload.get("head") if isinstance(payload.get("head"), dict) else {}
    base = payload.get("base") if isinstance(payload.get("base"), dict) else {}
    head_sha = str(head.get("sha") or "")
    head_ref = str(head.get("ref") or "")
    base_sha = str(base.get("sha") or "")
    base_ref = str(base.get("ref") or "")
    if not head_sha or not head_ref or not base_sha or not base_ref:
        return None, f"PR #{number} is missing head/base ref or SHA"
    return {
        "number": number,
        "pr": number,
        "title": str(payload.get("title") or ""),
        "body": str(payload.get("body") or ""),
        "state": str(payload.get("state") or "").lower(),
        "draft": bool(payload.get("draft")),
        "mergeable": payload.get("mergeable"),
        "mergeable_state": str(payload.get("mergeable_state") or "").lower(),
        "head_ref": head_ref,
        "head_sha": head_sha,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "labels": _labels(payload.get("labels")),
        "requested_reviewers": payload.get("requested_reviewers")
        if isinstance(payload.get("requested_reviewers"), list)
        else [],
        "requested_teams": payload.get("requested_teams")
        if isinstance(payload.get("requested_teams"), list)
        else [],
        "raw": payload,
    }, None


def _fetch_branch(
    repo: str, ref: str, *, api: GhApi = _run_gh_api
) -> tuple[dict[str, str] | None, str | None]:
    """Fetch one branch ref for stack-base comparison."""
    payload, error = _get_object(f"repos/{repo}/git/ref/heads/{ref}", api=api)
    if error or payload is None:
        return None, error or f"branch {ref!r} response was empty"
    obj = payload.get("object") if isinstance(payload.get("object"), dict) else {}
    sha = obj.get("sha")
    if not isinstance(sha, str) or not sha:
        return None, f"branch {ref!r} response is missing object.sha"
    return {"ref": ref, "sha": sha}, None


def _fetch_review_data(
    repo: str, number: int, *, api: GhApi = _run_gh_api
) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch review events/comments needed for digest and verdict checks."""
    reviews, reviews_pagination, error = _get_paginated_list(
        f"repos/{repo}/pulls/{number}/reviews?per_page={REST_PAGE_SIZE}", api=api
    )
    if error or reviews is None or reviews_pagination is None:
        return None, error or "review endpoint returned no data"
    review_comments, review_comments_pagination, error = _get_paginated_list(
        f"repos/{repo}/pulls/{number}/comments?per_page={REST_PAGE_SIZE}", api=api
    )
    if error or review_comments is None or review_comments_pagination is None:
        return None, error or "review-comment endpoint returned no data"
    conversation_comments, conversation_pagination, error = _get_paginated_list(
        f"repos/{repo}/issues/{number}/comments?per_page={REST_PAGE_SIZE}", api=api
    )
    if error or conversation_comments is None or conversation_pagination is None:
        return None, error or "conversation-comment endpoint returned no data"
    reviews_normalized = _body_entries(reviews)
    conversation_normalized = _body_entries(conversation_comments)
    review_digest = _review_digest(reviews, review_comments, conversation_comments)
    states = Counter(str(item.get("state") or "UNKNOWN").upper() for item in reviews)
    return {
        "reviews": reviews_normalized,
        "review_comments": review_comments,
        "conversation_comments": conversation_normalized,
        "review_digest": review_digest,
        "review_states": dict(sorted(states.items())),
        "pagination": {
            "reviews": reviews_pagination,
            "review_comments": review_comments_pagination,
            "conversation_comments": conversation_pagination,
        },
    }, None


def _fetch_threads(pr_number: int, *, repo: str) -> tuple[bool | None, str | None]:
    """Reuse the canonical complete GraphQL review-thread evaluator."""
    return fetch_threads_resolved(pr_number, repo=repo)


def _gate_status(
    pr: dict[str, Any], review_data: dict[str, Any], *, metadata: str
) -> dict[str, str]:
    """Evaluate exact-head and final metadata evidence from trusted review bodies."""
    evidence = {
        "head_sha": pr["head_sha"],
        "reviews": review_data["reviews"],
        "comments": review_data["conversation_comments"],
    }
    gate_verdicts = _extract_gate_verdicts(evidence)
    metadata_verdicts = _extract_metadata_verdicts(evidence)
    evidence["gate_verdicts"] = gate_verdicts
    evidence["metadata_verdicts"] = metadata_verdicts
    return {
        "gate_verdict": (
            "accepted" if has_current_accepted_gate_verdict(evidence, pr["head_sha"]) else "missing"
        ),
        "metadata_verdict": (
            "accepted" if has_current_pr_metadata_verdict(evidence, metadata) else "missing"
        ),
    }


def _merge_queue_gate_reasons(entry: dict[str, Any]) -> list[str]:
    """Return fail-closed reasons for exact-head merge-queue evidence."""
    merge_queue_gate = entry.get("merge_queue_gate", {})
    gate_status = merge_queue_gate.get("status")
    if gate_status == "missing":
        return ["missing_merge_queue_gate"]
    if gate_status == "malformed":
        return ["malformed_merge_queue_gate"]
    if gate_status == "mismatch":
        if entry.get("merge_queue_gate", {}).get("workflow_name") != GATE_WORKFLOW_NAME:
            return ["merge_queue_gate_workflow_mismatch"]
        return ["merge_queue_gate_head_mismatch"]
    if gate_status != "success":
        return [f"merge_queue_gate_not_green:{gate_status or 'unknown'}"]
    return []


def _closing_discipline_reasons(entry: dict[str, Any]) -> list[str]:
    """Require head-, body-, and source-bound closing evidence before readiness."""
    value = entry.get("closing_discipline")
    if not isinstance(value, Mapping):
        return ["closing_discipline_not_verified"]
    status = value.get("status")
    blockers = value.get("blockers")
    if status != "passed":
        return [
            {
                "blocked": "closing_discipline_blocked",
                "unavailable": "closing_discipline_unavailable",
            }.get(str(status), "closing_discipline_not_verified")
        ]
    reasons: list[str] = []
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) and item for item in blockers
    ):
        reasons.append("closing_discipline_not_verified")
    elif blockers:
        reasons.append("closing_discipline_not_verified")

    expected_head = str(entry.get("head_sha") or "")
    observed_head = str(value.get("head_sha") or "")
    if not _SHA_RE.fullmatch(expected_head) or observed_head.lower() != expected_head.lower():
        reasons.append("closing_discipline_head_mismatch")

    expected_body = str(entry.get("body_sha256") or "")
    observed_body = str(value.get("body_sha256") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_body) or (
        observed_body.lower() != expected_body.lower()
    ):
        reasons.append("closing_discipline_body_mismatch")

    sources = value.get("sources")
    if not isinstance(sources, Mapping) or dict(sources) != CLOSING_DISCIPLINE_SOURCES:
        reasons.append("closing_discipline_sources_missing")
    return sorted(set(reasons))


def _review_reasons(entry: dict[str, Any]) -> list[str]:
    """Return fail-closed reasons for current review and metadata evidence."""
    reasons: list[str] = []
    for state in ("CHANGES_REQUESTED", "PENDING", "DISMISSED"):
        if entry.get("review_states", {}).get(state, 0):
            reasons.append(f"non_authoritative_review_state:{state.lower()}")
    if entry.get("requested_reviewer_count", 0) or entry.get("requested_team_count", 0):
        reasons.append("outstanding_requested_reviewers")
    if entry.get("review_threads", {}).get("status") != "resolved":
        reasons.append(
            "unresolved_review_threads"
            if entry.get("review_threads", {}).get("status") == "unresolved"
            else "review_threads_not_evaluated"
        )
    if "merge-ready" not in entry.get("labels", []):
        reasons.append("missing_merge_ready_label")
    if entry.get("gate_verdict") != "accepted":
        reasons.append("missing_exact_head_gate_verdict")
    if entry.get("metadata_verdict") != "accepted":
        reasons.append("missing_current_pr_metadata_verdict")
    return reasons


def _entry_reasons(entry: dict[str, Any]) -> list[str]:
    """Return deterministic fail-closed readiness reasons for one stack entry."""
    reasons: list[str] = []
    if entry.get("state") != "open":
        reasons.append("pull_request_not_open")
    if entry.get("draft") is True:
        reasons.append("pull_request_is_draft")
    if entry.get("mergeable") is not True:
        reasons.append("mergeable_state_unknown_or_false")
    if entry.get("mergeable_state") != "clean":
        reasons.append("mergeable_state_not_clean")
    if entry.get("checks", {}).get("overall") != "success":
        reasons.append(f"ci_not_green:{entry.get('checks', {}).get('overall', 'unknown')}")
    for hold in entry.get("explicit_holds", []):
        reasons.append(f"explicit_hold:{hold}")
    reasons.extend(_merge_queue_gate_reasons(entry))
    reasons.extend(_closing_discipline_reasons(entry))
    reasons.extend(_review_reasons(entry))
    if entry.get("base_alignment") != "aligned":
        reasons.append("stack_base_not_aligned")
    return reasons


def build_stack_status(
    repo: str,
    prs: list[int],
    *,
    api: GhApi = _run_gh_api,
    thread_fetcher: Callable[[int], tuple[bool | None, str | None]] | None = None,
    waiver_actor: str = "",
) -> dict[str, Any]:
    """Collect a complete stack status snapshot from live GitHub state."""
    valid_prs, error = _positive_prs(prs)
    if error or valid_prs is None:
        return {"schema": SCHEMA, "status": "error", "error": error}
    main, error = _fetch_branch(repo, "main", api=api)
    if error or main is None:
        return {"schema": SCHEMA, "status": "error", "error": f"main lookup failed: {error}"}
    live: list[dict[str, Any]] = []
    for number in valid_prs:
        pr, error = _fetch_pr(repo, number, api=api)
        if error or pr is None:
            return {"schema": SCHEMA, "status": "error", "error": f"PR #{number}: {error}"}
        live.append(pr)

    if thread_fetcher is None:

        def default_thread_fetcher(number: int) -> tuple[bool | None, str | None]:
            return _fetch_threads(number, repo=repo)

        thread_fetcher = default_thread_fetcher

    entries: list[dict[str, Any]] = []
    workflow_cache: dict[str, dict[str, Any]] = {}
    for index, pr in enumerate(live):
        review_data, error = _fetch_review_data(repo, pr["number"], api=api)
        if error or review_data is None:
            return {
                "schema": SCHEMA,
                "status": "error",
                "error": f"PR #{pr['number']} review state: {error}",
            }
        check_runs, check_runs_pagination, error = _get_paginated_list(
            f"repos/{repo}/commits/{pr['head_sha']}/check-runs?per_page={REST_PAGE_SIZE}",
            api=api,
            response_key="check_runs",
        )
        if error or check_runs is None or check_runs_pagination is None:
            return {
                "schema": SCHEMA,
                "status": "error",
                "error": f"PR #{pr['number']} checks: {error}",
            }
        check_runs = _enrich_merge_queue_gate_check_runs(
            check_runs, repo=repo, api=api, cache=workflow_cache
        )
        threads_resolved, thread_error = thread_fetcher(pr["number"])
        parent = live[index - 1] if index else None
        expected_ref = parent["head_ref"] if parent else "main"
        expected_sha = parent["head_sha"] if parent else main["sha"]
        base_alignment = (
            "aligned"
            if pr["base_ref"] == expected_ref and pr["base_sha"] == expected_sha
            else "misaligned"
        )
        metadata = metadata_digest(pr["title"], pr["body"])
        gate = _gate_status(pr, review_data, metadata=metadata)
        hold_source = {
            **pr,
            "reviews": review_data["reviews"],
            "comments": review_data["conversation_comments"],
        }
        entry: dict[str, Any] = {
            "position": index,
            "pr": pr["number"],
            "state": pr["state"],
            "draft": pr["draft"],
            "mergeable": pr["mergeable"],
            "mergeable_state": pr["mergeable_state"],
            "head_ref": pr["head_ref"],
            "head_sha": pr["head_sha"],
            "body_sha256": hashlib.sha256(pr["body"].encode("utf-8")).hexdigest(),
            "base_ref": pr["base_ref"],
            "base_sha": pr["base_sha"],
            "expected_base_ref": expected_ref,
            "expected_base_sha": expected_sha,
            "base_alignment": base_alignment,
            "labels": pr["labels"],
            "requested_reviewer_count": len(pr["requested_reviewers"]),
            "requested_team_count": len(pr["requested_teams"]),
            "checks": summarize_check_runs(check_runs),
            "required_checks": normalize_required_checks(check_runs, head_sha=pr["head_sha"]),
            "merge_queue_gate": summarize_merge_queue_gate(check_runs, head_sha=pr["head_sha"]),
            "pagination": {
                **review_data["pagination"],
                "check_runs": check_runs_pagination,
            },
            "review_states": review_data["review_states"],
            "review_digest": review_data["review_digest"],
            "review_threads": {
                "status": (
                    "resolved"
                    if threads_resolved is True
                    else "unresolved"
                    if threads_resolved is False
                    else "unknown"
                ),
                "unresolved": 0
                if threads_resolved is True
                else 1
                if threads_resolved is False
                else None,
                "error": thread_error,
            },
            "metadata_digest": metadata,
            "closing_discipline": build_closing_discipline_evidence(
                pr["number"],
                repository=repo,
                head_sha=pr["head_sha"],
                body=pr["body"],
            ),
            **gate,
        }
        review_evidence = {
            "head_sha": pr["head_sha"],
            "metadata_digest": metadata,
            "check_runs": check_runs,
            "reviews": review_data["reviews"],
            "comments": review_data["conversation_comments"],
            "waiver_actor": waiver_actor,
        }
        entry["implementation_review"] = classify_implementation_review(review_evidence)
        hold_source = {
            **pr,
            "reviews": review_data["reviews"],
            "comments": review_data["conversation_comments"],
            "explicit_holds": current_explicit_merge_hold_reasons(
                {
                    **pr,
                    "reviews": review_data["reviews"],
                    "comments": review_data["conversation_comments"],
                }
            ),
        }
        entry["holds"] = derive_holds(hold_source)
        entry["explicit_holds"] = sorted(
            reason
            for disposition in entry["holds"].values()
            if isinstance(disposition, dict) and disposition.get("status") != "clear"
            for reason in disposition.get("reason_codes", [])
        )
        entry["reasons"] = _entry_reasons(entry)
        entry["merge_ready"] = not entry["reasons"]
        entries.append(entry)

    return {
        "schema": SCHEMA,
        "status": "ok",
        "repo": repo,
        "stack_order": valid_prs,
        "main": main,
        "entries": entries,
        "all_merge_ready": all(entry["merge_ready"] for entry in entries),
    }


def _retarget_plan(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the desired root-to-tip base-ref plan from live PR entries."""
    plan: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        desired = "main" if index == 0 else str(entries[index - 1]["head_ref"])
        plan.append(
            {
                "pr": entry["pr"],
                "head_sha": entry["head_sha"],
                "current_base_ref": entry["base_ref"],
                "desired_base_ref": desired,
                "change_required": entry["base_ref"] != desired,
            }
        )
    return plan


def retarget_stack(  # noqa: C901
    repo: str,
    prs: list[int],
    *,
    expected_heads: dict[int, str] | None = None,
    apply: bool = False,
    api: GhApi = _run_gh_api,
) -> dict[str, Any]:
    """Plan or guardedly apply root-to-tip PR base-ref changes."""
    valid_prs, error = _positive_prs(prs)
    if error or valid_prs is None:
        return {"schema": SCHEMA, "status": "error", "error": error}
    expected_heads = expected_heads or {}
    live: list[dict[str, Any]] = []
    for number in valid_prs:
        entry, error = _fetch_pr(repo, number, api=api)
        if error or entry is None:
            return {"schema": SCHEMA, "status": "error", "error": f"PR #{number}: {error}"}
        live.append(entry)
    plan = _retarget_plan(live)
    if apply:
        missing = [number for number in valid_prs if number not in expected_heads]
        if missing:
            return {
                "schema": SCHEMA,
                "status": "blocked",
                "error": f"--apply requires expected heads for every PR; missing {missing}",
                "plan": plan,
            }
        mismatches = [
            number
            for number in valid_prs
            if live[valid_prs.index(number)]["head_sha"].lower() != expected_heads[number].lower()
        ]
        if mismatches:
            return {
                "schema": SCHEMA,
                "status": "blocked",
                "error": f"exact-head mismatch before retarget for PRs {mismatches}",
                "plan": plan,
            }
        applied: list[int] = []
        for item in plan:
            if not item["change_required"]:
                continue
            number = int(item["pr"])
            current, error = _fetch_pr(repo, number, api=api)
            if error or current is None:
                return {
                    "schema": SCHEMA,
                    "status": "blocked",
                    "error": f"PR #{number} recheck failed: {error}",
                    "applied": applied,
                }
            if current["head_sha"].lower() != expected_heads[number].lower():
                return {
                    "schema": SCHEMA,
                    "status": "blocked",
                    "error": f"PR #{number} head changed before retarget",
                    "applied": applied,
                }
            _, error = api(
                "PATCH",
                f"repos/{repo}/pulls/{number}",
                {"base": item["desired_base_ref"]},
            )
            if error:
                return {
                    "schema": SCHEMA,
                    "status": "error",
                    "error": f"PR #{number} retarget failed: {error}",
                    "applied": applied,
                }
            verified, error = _fetch_pr(repo, number, api=api)
            if error or verified is None:
                return {
                    "schema": SCHEMA,
                    "status": "blocked",
                    "error": f"PR #{number} retarget verification failed: {error}",
                    "applied": applied,
                }
            if (
                verified["base_ref"] != item["desired_base_ref"]
                or verified["head_sha"].lower() != expected_heads[number].lower()
            ):
                return {
                    "schema": SCHEMA,
                    "status": "blocked",
                    "error": f"PR #{number} retarget verification mismatch",
                    "applied": applied,
                }
            applied.append(number)
        return {"schema": SCHEMA, "status": "applied", "plan": plan, "applied": applied}
    return {"schema": SCHEMA, "status": "dry_run", "plan": plan}


def _worktree_branch_records(
    worktree: Path, *, git_runner: GitRunner = _git
) -> tuple[list[dict[str, str]] | None, str | None]:
    """Read linked worktree branch ownership for checkout safety."""
    result = git_runner(["worktree", "list", "--porcelain"], worktree)
    if result.returncode != 0:
        return None, result.stderr.strip() or "git worktree list failed"
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in result.stdout.splitlines() + [""]:
        if line.startswith("worktree "):
            if current:
                records.append(current)
            current = {"path": line.removeprefix("worktree ")}
        elif line.startswith("branch "):
            current["branch"] = line.removeprefix("branch ").removeprefix("refs/heads/")
        elif not line and current:
            records.append(current)
            current = {}
    return records, None


def sync_stack(  # noqa: C901
    branches: list[str],
    *,
    base: str = "main",
    remote: str = "origin",
    worktree: Path,
    apply: bool = False,
    git_runner: GitRunner = _git,
) -> dict[str, Any]:
    """Plan or execute a clean-worktree progressive fetch/merge/push."""
    if not branches or len(set(branches)) != len(branches):
        return {
            "schema": SCHEMA,
            "status": "error",
            "error": "branches must be non-empty and unique",
        }
    if any(not branch or branch.startswith("-") for branch in branches + [base, remote]):
        return {
            "schema": SCHEMA,
            "status": "error",
            "error": "branch and remote names must be non-empty",
        }
    status = git_runner(["status", "--porcelain"], worktree)
    if status.returncode != 0:
        return {
            "schema": SCHEMA,
            "status": "error",
            "error": status.stderr.strip() or "git status failed",
        }
    if status.stdout.strip():
        return {"schema": SCHEMA, "status": "blocked", "error": "worktree is dirty"}
    current = git_runner(["branch", "--show-current"], worktree)
    if current.returncode != 0 or not current.stdout.strip():
        return {"schema": SCHEMA, "status": "blocked", "error": "worktree must have a named branch"}
    original = current.stdout.strip()
    records, error = _worktree_branch_records(worktree, git_runner=git_runner)
    if error or records is None:
        return {"schema": SCHEMA, "status": "error", "error": error}
    owned_elsewhere = {
        record.get("branch"): record.get("path")
        for record in records
        if record.get("path") != str(worktree.resolve()) and record.get("branch")
    }
    conflicts = [branch for branch in branches if branch in owned_elsewhere]
    if conflicts:
        return {
            "schema": SCHEMA,
            "status": "blocked",
            "error": f"branches checked out in another worktree: {conflicts}",
        }
    commands = [["fetch", remote, base, *branches]]
    for index, branch in enumerate(branches):
        source = f"{remote}/{base if index == 0 else branches[index - 1]}"
        commands.extend(
            [["checkout", branch], ["merge", "--no-edit", source], ["push", remote, branch]]
        )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "dry_run" if not apply else "planned",
        "worktree": str(worktree),
        "original_branch": original,
        "commands": commands,
    }
    if not apply:
        return result
    executed: list[list[str]] = []
    try:
        for command in commands:
            completed = git_runner(command, worktree)
            executed.append(command)
            if completed.returncode != 0:
                result.update(
                    {
                        "status": "error",
                        "error": completed.stderr.strip() or f"git {' '.join(command)} failed",
                        "executed": executed,
                    }
                )
                return result
    finally:
        if original != branches[-1]:
            restore = git_runner(["checkout", original], worktree)
            result["restored_original_branch"] = restore.returncode == 0
            if restore.returncode != 0 and result.get("status") == "planned":
                result.update(
                    {
                        "status": "error",
                        "error": restore.stderr.strip() or "failed to restore original branch",
                    }
                )
    result["status"] = "applied"
    result["executed"] = executed
    return result


def _merge_pr(
    repo: str,
    entry: dict[str, Any],
    *,
    expected_head: str,
    api: GhApi,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Delegate one exact-head squash merge to the canonical receipt owner."""
    if entry.get("head_sha", "").lower() != expected_head.lower():
        return None, f"PR #{entry.get('pr')} head changed before merge"
    return apply_guarded_merge(receipt, repository=repo, api=api)


def merge_cascade(  # noqa: C901, PLR0912
    repo: str,
    prs: list[int],
    *,
    expected_heads: dict[int, str] | None = None,
    apply: bool = False,
    api: GhApi = _run_gh_api,
    thread_fetcher: Callable[[int], tuple[bool | None, str | None]] | None = None,
    waiver_actor: str = "",
    waiver_reason: str = "",
) -> dict[str, Any]:
    """Merge only the current green root and advance the next PR safely."""
    status = build_stack_status(
        repo,
        prs,
        api=api,
        thread_fetcher=thread_fetcher,
        waiver_actor=waiver_actor,
    )
    if status.get("status") != "ok":
        return {**status, "operation": "merge-cascade"}
    entries = status["entries"]
    expected_heads = expected_heads or {}
    if apply:
        missing = [entry["pr"] for entry in entries if entry["pr"] not in expected_heads]
        if missing:
            return {
                "schema": SCHEMA,
                "status": "blocked",
                "operation": "merge-cascade",
                "error": f"--apply requires expected heads for every PR; missing {missing}",
                "stack": status,
            }
        fresh_status = build_stack_status(
            repo,
            prs,
            api=api,
            thread_fetcher=thread_fetcher,
            waiver_actor=waiver_actor,
        )
        if fresh_status.get("status") != "ok":
            return {
                **fresh_status,
                "operation": "merge-cascade",
                "error": f"immediate pre-merge snapshot failed: {fresh_status.get('error', 'unknown error')}",
            }
        status = fresh_status
        entries = status["entries"]
        mismatches = [
            entry["pr"]
            for entry in entries
            if entry["head_sha"].lower() != expected_heads[entry["pr"]].lower()
        ]
        if mismatches:
            return {
                "schema": SCHEMA,
                "status": "blocked",
                "operation": "merge-cascade",
                "error": f"exact-head mismatch before merge for PRs {mismatches}",
                "stack": status,
            }
    root = entries[0]
    if not root["merge_ready"]:
        return {
            "schema": SCHEMA,
            "status": "blocked" if apply else "dry_run_blocked",
            "operation": "merge-cascade",
            "error": f"root PR #{root['pr']} is not merge-ready",
            "reasons": root["reasons"],
            "stack": status,
        }
    if not apply:
        actions = [{"action": "squash_merge", "pr": root["pr"], "head_sha": root["head_sha"]}]
        if len(entries) > 1:
            actions.append(
                {
                    "action": "advance_after_merge",
                    "pr": entries[1]["pr"],
                    "stop": "fresh_ci_required_after_base_change",
                }
            )
        return {
            "schema": SCHEMA,
            "status": "dry_run",
            "operation": "merge-cascade",
            "actions": actions,
            "stack": status,
        }
    receipt = build_receipt_from_stack_entry(
        repo,
        root,
        current_base_sha=str(status.get("main", {}).get("sha") or ""),
        waiver_actor=waiver_actor,
        waiver_reason=waiver_reason,
    )
    receipt_verification = verify_receipt(receipt)
    if receipt_verification.get("passed") is not True:
        return {
            "schema": SCHEMA,
            "status": "blocked",
            "operation": "merge-cascade",
            "error": "single-account merge receipt blocked",
            "receipt": receipt,
            "receipt_verification": receipt_verification,
            "stack": status,
        }
    merged, error = _merge_pr(
        repo,
        root,
        expected_head=expected_heads[root["pr"]],
        api=api,
        receipt=receipt,
    )
    if error or merged is None:
        failure_result: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "error",
            "operation": "merge-cascade",
            "error": error,
            "merged": False,
        }
        if isinstance(merged, Mapping) and "receipt_digest" in merged:
            failure_result["receipt"] = merged
        return failure_result
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "merge-cascade",
        "merged": [merged],
    }
    if len(entries) == 1:
        result["status"] = "merged"
        return result

    next_pr = int(entries[1]["pr"])
    next_entry, error = _fetch_pr(repo, next_pr, api=api)
    if error or next_entry is None:
        result.update(
            {
                "status": "merged_waiting_for_next_snapshot",
                "error": f"next PR lookup failed after merge: {error}",
                "next_pr": next_pr,
            }
        )
        return result
    main, error = _fetch_branch(repo, "main", api=api)
    if error or main is None:
        result.update(
            {
                "status": "merged_waiting_for_next_snapshot",
                "error": f"main lookup failed after merge: {error}",
                "next_pr": next_pr,
            }
        )
        return result
    if next_entry["head_sha"].lower() != expected_heads[next_pr].lower():
        result.update(
            {
                "status": "merged_next_head_changed",
                "error": f"next PR #{next_pr} head changed after root merge",
                "next_pr": next_pr,
            }
        )
        return result
    old_parent_ref = entries[0]["head_ref"]
    if next_entry["base_ref"] == "main":
        result["base_advance"] = {
            "mode": "automatic",
            "base_ref": "main",
            "base_sha": next_entry["base_sha"],
        }
        if next_entry["base_sha"] != main["sha"]:
            result.update(
                {
                    "status": "merged_waiting_for_base_refresh",
                    "next_pr": next_pr,
                    "error": "next PR base is main but does not yet point at current main",
                }
            )
            return result
    elif next_entry["base_ref"] == old_parent_ref:
        _, error = api("PATCH", f"repos/{repo}/pulls/{next_pr}", {"base": "main"})
        if error:
            result.update(
                {
                    "status": "merged_next_retarget_failed",
                    "next_pr": next_pr,
                    "error": error,
                }
            )
            return result
        verified, error = _fetch_pr(repo, next_pr, api=api)
        if (
            error
            or verified is None
            or verified["base_ref"] != "main"
            or verified["head_sha"].lower() != expected_heads[next_pr].lower()
        ):
            result.update(
                {
                    "status": "merged_next_retarget_unverified",
                    "next_pr": next_pr,
                    "error": error or "next PR did not verify base=main and the expected head",
                }
            )
            return result
        result["base_advance"] = {"mode": "explicit", "base_ref": "main"}
    else:
        result.update(
            {
                "status": "merged_unexpected_next_base",
                "next_pr": next_pr,
                "error": f"next PR base {next_entry['base_ref']!r} is neither main nor {old_parent_ref!r}",
            }
        )
        return result
    result.update(
        {
            "status": "merged_waiting_for_ci",
            "next_pr": next_pr,
            "next_action": "rerun merge-cascade after fresh CI and exact-head review evidence",
        }
    )
    return result


def _parent_pr_lookup(
    repo: str, parent_pr: int, *, api: GhApi = _run_gh_api
) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch the declared parent PR's lifecycle state, or ``(None, None)`` when absent.

    A missing/unreadable parent PR is returned as ``(None, None)`` so the
    ancestry classifier can fail closed with ``parent_invalidated`` (the parent
    cannot be verified) instead of crashing the gate.
    """
    if parent_pr is None or parent_pr < 1:
        return None, None
    payload, error = _get_object(f"repos/{repo}/pulls/{parent_pr}", api=api)
    if error or payload is None:
        return None, None
    state = str(payload.get("state") or "").lower()
    merged = bool(payload.get("merged"))
    head = payload.get("head") if isinstance(payload.get("head"), dict) else {}
    return {
        "state": state,
        "merged": merged,
        "head_sha": str(head.get("sha") or ""),
    }, None


def _parent_lifecycle_facts(
    repo: str,
    declaration: StackDeclaration,
    *,
    api: GhApi = _run_gh_api,
) -> tuple[str, bool, bool]:
    """Resolve a declared parent PR's live lifecycle facts (issue #7515).

    Returns ``(parent_state, parent_merged, parent_head_changed)``.  A
    missing/unreadable parent PR yields ``parent_state="unknown"`` so the
    ancestry classifier fails closed with ``parent_invalidated`` instead of
    crashing the gate.
    """
    parent, _ = _parent_pr_lookup(repo, declaration.parent_pr, api=api)
    if parent is None:
        return "unknown", False, False
    parent_state = str(parent.get("state") or "unknown")
    parent_merged = bool(parent.get("merged"))
    parent_head_changed = bool(
        parent.get("head_sha")
        and declaration.parent_head
        and str(parent.get("head_sha")).lower() != declaration.parent_head.lower()
    )
    return parent_state, parent_merged, parent_head_changed


def _finalize_ancestry_result(state: dict[str, Any], *, target: str, branch: str) -> dict[str, Any]:
    """Attach the deterministic status/classification envelope to an ancestry state."""
    state["schema"] = SCHEMA
    state["operation"] = "check-ancestry"
    state["status"] = "ok"
    state["target"] = target
    state["branch"] = branch or target
    if state["state"] == "clean":
        state["status"] = "ok"
    elif state["state"] == "stacked":
        state["status"] = "ok"
        state["mergeable"] = False
        state["classification"] = "stacked_not_independently_mergeable"
    elif state["state"] == "parent_merged":
        state["status"] = "refresh_required"
    else:
        state["status"] = "blocked"
    if state["state"] != "clean":
        state["remediation_command"] = remediation_command(
            parent_head=state.get("declared_parent_head") or state.get("merge_base_sha"),
            branch=branch or target,
        )
    return state


def _resolve_ancestry_target(
    *,
    target_text: str,
    pr_number: int | None,
    branch: str | None,
    repo: str,
    worktree: Path,
    git_runner: GitRunner,
    api: GhApi,
) -> tuple[dict[str, Any], None] | tuple[None, dict[str, Any]]:
    """Resolve a PR number or remote branch into ``(head_sha, base_ref, body, branch)``.

    Returns ``(target, None)`` on success or ``(None, error_result)`` fail-closed.
    """
    if pr_number is not None:
        pr, error = _fetch_pr(repo, pr_number, api=api)
        if error or pr is None:
            return None, {"error": f"PR #{pr_number}: {error or 'unreadable'}"}
        return (
            {
                "head_sha": pr["head_sha"],
                "base_ref": pr["base_ref"],
                "body": pr["body"],
                "branch": branch or pr["head_ref"],
            },
            None,
        )
    if not branch:
        return None, {"error": "branch target requires --branch"}
    branch_result = git_runner(
        ["rev-parse", "--verify", f"refs/remotes/origin/{target_text}"], worktree
    )
    if branch_result.returncode != 0:
        return None, {
            "error": (
                branch_result.stderr.strip() or f"cannot resolve refs/remotes/origin/{target_text}"
            )
        }
    return (
        {
            "head_sha": branch_result.stdout.strip(),
            "base_ref": "main",
            "body": "",
            "branch": branch,
        },
        None,
    )


def _ancestry_invocation(
    *,
    target: str,
    worktree: Path,
    git_runner: GitRunner,
) -> tuple[str, int | None, Path] | tuple[None, None, dict[str, Any]]:
    """Validate the check-ancestry invocation and fetch live remote refs.

    Returns ``(target_text, pr_number, worktree)`` on success or
    ``(None, None, error_result)`` fail-closed.
    """
    target_text = str(target or "").strip()
    if not target_text:
        return None, None, {"error": "target must be a PR number or branch name"}
    try:
        pr_number = int(target_text)
    except ValueError:
        pr_number = None
    if pr_number is not None and pr_number < 1:
        return None, None, {"error": "PR number must be positive"}
    fetch = git_runner(["fetch", "--no-tags", "origin", "main"], worktree)
    if fetch.returncode != 0:
        return (
            None,
            None,
            {
                "error": fetch.stderr.strip() or "git fetch origin main failed",
            },
        )
    return target_text, pr_number, worktree


def check_ancestry(
    repo: str,
    *,
    target: str,
    worktree: Path | None = None,
    branch: str | None = None,
    declaration_text: str | None = None,
    api: GhApi = _run_gh_api,
    git_runner: GitRunner = _git,
) -> dict[str, Any]:
    """Run the fail-closed ancestry gate for one branch or PR (issue #7515).

    ``target`` is either a PR number (fetched via ``gh api``) or a branch name
    resolved against the remote.  For a PR target the base ref, head SHA, and PR
    body come from GitHub; for a branch target the caller supplies ``branch``
    (local branch name) and the base ref defaults to ``main``.

    The check always fetches the live remote refs first, then computes the
    merge base, the non-``main`` ancestry commits and paths, and classifies
    against the parsed ``## Stack Declaration`` (from the PR body or
    ``declaration_text``).  Blocking states carry a non-zero CLI exit.
    """
    error_result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "check-ancestry",
        "status": "error",
    }
    worktree = worktree or Path.cwd()
    if not isinstance(worktree, Path):
        return {**error_result, "error": "worktree must be a path"}
    if not worktree.is_dir():
        return {**error_result, "error": f"worktree does not exist: {worktree}"}

    invocation = _ancestry_invocation(target=target, worktree=worktree, git_runner=git_runner)
    if invocation[0] is None:
        return {**error_result, **invocation[2]}
    target_text, pr_number, worktree = invocation

    resolved, resolve_error = _resolve_ancestry_target(
        target_text=target_text,
        pr_number=pr_number,
        branch=branch,
        repo=repo,
        worktree=worktree,
        git_runner=git_runner,
        api=api,
    )
    if resolve_error is not None or resolved is None:
        return {**error_result, **resolve_error}
    head_sha = resolved["head_sha"]
    base_ref = resolved["base_ref"]
    body = resolved["body"]
    branch = resolved["branch"]

    if not head_sha or not base_ref:
        return {**error_result, "error": f"{target_text} has no usable head/base ref"}

    declaration, parse_error = parse_stack_declaration(declaration_text or body or "")
    if parse_error:
        return {**error_result, "error": parse_error}

    facts, fact_error = collect_ancestry_facts(
        head_sha=head_sha,
        base_ref=base_ref,
        worktree=worktree,
        remote="origin",
        git_runner=git_runner,
    )
    if fact_error:
        return {**error_result, "error": fact_error}

    parent_state = ""
    parent_merged = False
    parent_head_changed = False
    if declaration is not None:
        parent_state, parent_merged, parent_head_changed = _parent_lifecycle_facts(
            repo, declaration, api=api
        )

    state = ancestry_state(
        head_sha=head_sha,
        base_ref=base_ref,
        main_tip_sha=facts["main_tip_sha"],
        merge_base_sha=facts["merge_base_sha"],
        commits=facts["commits"],
        declaration=declaration,
        parent_state=parent_state,
        parent_merged=parent_merged,
        parent_head_changed=parent_head_changed,
    )
    state["unexpected_paths"] = facts["changed_paths"]
    return _finalize_ancestry_result(state, target=target_text, branch=branch or target_text)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository OWNER/REPO")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def _add_stack_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prs", nargs="+", type=int, required=True, help="PRs in root-to-tip order"
    )


def _add_expected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-head",
        action="append",
        default=[],
        metavar="PR=SHA",
        help="exact PR head guard; repeat once per stack entry",
    )
    parser.add_argument(
        "--apply", action="store_true", help="perform guarded remote/local mutations"
    )


def _print_result(result: dict[str, Any], *, as_json: bool) -> None:
    """Print stable JSON or a compact human summary."""
    if result.get("operation") == "check-ancestry":
        if as_json:
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        print(f"check-ancestry: {result.get('state')} ({result.get('status')})")
        for line in render_diagnostics(result):
            print(f"  {line}")
        if result.get("remediation_command"):
            print(f"  remediation command: {result['remediation_command']}")
        return
    if as_json or result.get("status") not in {"ok", "dry_run", "applied", "merged"}:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"{result.get('operation', 'stacked-prs')}: {result.get('status')}")
    if "entries" in result:
        for entry in result["entries"]:
            print(
                f"  PR #{entry['pr']}: {entry['checks']['overall']} base={entry['base_ref']} ready={entry['merge_ready']}"
            )
    if result.get("applied"):
        print(f"  applied: {', '.join(f'#{number}' for number in result['applied'])}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="read live stack status")
    _add_common_arguments(status_parser)
    _add_stack_arguments(status_parser)

    retarget_parser = subparsers.add_parser("retarget", help="plan/apply stack base refs")
    _add_common_arguments(retarget_parser)
    _add_stack_arguments(retarget_parser)
    _add_expected_arguments(retarget_parser)

    sync_parser = subparsers.add_parser("sync", help="plan/apply progressive branch sync")
    sync_parser.add_argument("--branches", nargs="+", required=True, help="branches root-to-tip")
    sync_parser.add_argument("--base", default="main")
    sync_parser.add_argument("--remote", default="origin")
    sync_parser.add_argument("--worktree", type=Path, default=Path.cwd())
    sync_parser.add_argument("--apply", action="store_true")
    sync_parser.add_argument("--json", action="store_true")

    cascade_parser = subparsers.add_parser("merge-cascade", help="guarded sequential squash merge")
    _add_common_arguments(cascade_parser)
    _add_stack_arguments(cascade_parser)
    _add_expected_arguments(cascade_parser)
    cascade_parser.add_argument(
        "--waiver-actor",
        default="",
        help="actor recorded when the bounded single-account waiver is used",
    )
    cascade_parser.add_argument(
        "--waiver-reason",
        default="",
        help="explicit reason a distinct human implementation reviewer was unavailable",
    )

    ancestry_parser = subparsers.add_parser(
        "check-ancestry",
        help="fail-closed non-main ancestry gate for one branch or PR (issue #7515)",
    )
    _add_common_arguments(ancestry_parser)
    ancestry_parser.add_argument(
        "target",
        help="PR number or remote branch name to classify (e.g. 7389 or fix/issue-7283-shared-pysf-slices)",
    )
    ancestry_parser.add_argument(
        "--worktree",
        type=Path,
        default=Path.cwd(),
        help="repository worktree for local git reads (default: current directory)",
    )
    ancestry_parser.add_argument(
        "--branch",
        help="local branch name for a branch target and for the remediation command",
    )
    ancestry_parser.add_argument(
        "--declaration-text",
        help="stack declaration text override (default: PR body for PR targets)",
    )

    args = parser.parse_args(argv)
    if args.command == "status":
        result = build_stack_status(args.repo, args.prs)
    elif args.command == "retarget":
        expected, error = _parse_expected_heads(args.expected_head)
        result = (
            {"schema": SCHEMA, "status": "error", "error": error}
            if error
            else retarget_stack(args.repo, args.prs, expected_heads=expected, apply=args.apply)
        )
    elif args.command == "sync":
        result = sync_stack(
            args.branches,
            base=args.base,
            remote=args.remote,
            worktree=args.worktree,
            apply=args.apply,
        )
    elif args.command == "check-ancestry":
        result = check_ancestry(
            args.repo,
            target=args.target,
            worktree=args.worktree,
            branch=args.branch,
            declaration_text=args.declaration_text,
        )
    else:
        expected, error = _parse_expected_heads(args.expected_head)
        result = (
            {"schema": SCHEMA, "status": "error", "error": error}
            if error
            else merge_cascade(
                args.repo,
                args.prs,
                expected_heads=expected,
                apply=args.apply,
                waiver_actor=args.waiver_actor,
                waiver_reason=args.waiver_reason,
            )
        )
    _print_result(result, as_json=args.json)
    return 0 if result.get("status") in {"ok", "dry_run", "applied", "merged"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
