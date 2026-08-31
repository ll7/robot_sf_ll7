#!/usr/bin/env python3
"""Emit a compact issue-batch snapshot for token-efficient goal orchestration."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from scripts.dev import (
    blocker_transition,
    gh_issue_rest,
    goal_issue_admission,
    issue_implementability,
)
from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev.github_quota import (
    DEFAULT_CORE_SAFETY_THRESHOLD,
    DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
    RateLimitSnapshot,
    graphql_budget_decision,
    parse_rate_limit_payload,
)
from scripts.dev.issue_claim import short_claim_ref, status_issue

BODY_EXCERPT_CHARS = 300
DEFAULT_CLAIMABLE_LIMIT = 20
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
BLOCKED_EXTERNAL_INPUT_LABEL = "state:blocked-external-input"
EXTERNAL_RESOURCE_LABEL = "resource:external-data"
COMPUTE_ROUTING_LABEL = "routing:needs-compute"
BLOCKED_LABEL_PREFIX = "blocked:"
BLOCKER_DECISION_STATUSES = frozenset({"blocked_unchanged", "blocker_changed", "re_evaluate"})
BLOCKER_DECISION_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
EXTERNAL_BLOCKER_LABELS = {
    BLOCKED_EXTERNAL_INPUT_LABEL,
    "blocked",
    "evidence:blocked",
    "state:blocked",
}
PORTFOLIO_CLASSIFICATIONS = {
    "blocked_external_asset",
    "diagnostic_only",
    "executable_now",
    "needs_human_decision",
    "paper_critical",
    "stale_synthesis",
}
UNCLAIMABLE_LABELS = {
    "blocked",
    "decision-required",
    "duplicate",
    "invalid",
    "needs-triage",
    "state:blocked",
    "state:hold",
    "state:review",
    "state:running",
    "wontfix",
} | set(issue_implementability.BLOCKING_LABELS | issue_implementability.PARENT_LABELS)


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _rate_limit_snapshot() -> RateLimitSnapshot:
    """Read quota through REST without spending GraphQL budget."""
    result = _gh(["api", "rate_limit"])
    if result.returncode != 0:
        return RateLimitSnapshot(
            status="unavailable",
            error=result.stderr.strip() or result.stdout.strip() or "rate_limit request failed",
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return RateLimitSnapshot(status="malformed", error=f"invalid rate_limit JSON: {exc}")
    return parse_rate_limit_payload(payload)


def _is_graphql_quota_error(message: str) -> bool:
    """Return whether a GitHub error indicates GraphQL quota exhaustion."""
    text = (message or "").lower()
    if "rate limit" not in text:
        return False
    return "graphql" in text or "api rate limit" in text or "too many requests" in text


def _repo_parts(repo: str) -> tuple[str, str] | None:
    """Split a GitHub ``owner/name`` repository string for REST fallback."""
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return parts[0], parts[1]


def _rest_open_issue_page(
    *, repo: str, page: int, per_page: int, label: str | None = None
) -> tuple[list[dict[str, Any]] | None, str]:
    """Read one bounded open-issue page through REST, excluding pull requests later."""
    parts = _repo_parts(repo)
    if parts is None:
        return None, f"invalid repository name {repo!r}; expected owner/name"
    owner, name = parts
    args = [
        "api",
        f"repos/{owner}/{name}/issues",
        "--method",
        "GET",
        "--field",
        "state=open",
        "--field",
        f"per_page={per_page}",
        "--field",
        f"page={page}",
    ]
    if label:
        args.extend(["--field", f"labels={label}"])
    result = _gh(args)
    if result.returncode != 0:
        return None, result.stderr.strip() or result.stdout.strip() or "REST issue list failed"
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, f"invalid REST issue list JSON: {exc}"
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        return None, "REST issue list response must be a JSON array of objects"
    return payload, ""


def _normalize_rest_issue(issue: dict[str, Any]) -> dict[str, Any] | None:
    """Map one REST issue to the compact list shape, dropping pull requests."""
    if issue.get("pull_request") is not None:
        return None
    number = issue.get("number")
    if not isinstance(number, int) or number < 1:
        return None
    labels = issue.get("labels")
    assignees = issue.get("assignees")
    return {
        "number": number,
        "title": issue.get("title") if isinstance(issue.get("title"), str) else "",
        "state": issue.get("state", "").upper() if isinstance(issue.get("state"), str) else "",
        "url": issue.get("html_url") if isinstance(issue.get("html_url"), str) else "",
        "labels": labels if isinstance(labels, list) else [],
        "assignees": assignees if isinstance(assignees, list) else [],
    }


def _quota_blocked_after_graphql_error(decision: dict[str, Any], *, message: str) -> dict[str, Any]:
    """Convert an unexpected mid-command GraphQL quota failure to a resumable decision."""
    return {
        **decision,
        "status": "quota_blocked",
        "reason": "graphql_quota_exhausted_during_lookup",
        "message": message,
        "resume_after": decision.get("graphql_reset_at"),
    }


def _listing_result(
    *,
    status: str,
    listed: list[dict[str, Any]],
    error: str,
    data_source: str,
    rate_limit: RateLimitSnapshot,
    quota: dict[str, Any],
    resume_cursor: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the shared bounded issue-list result shape."""
    return {
        "status": status,
        "listed": listed,
        "error": error,
        "data_source": data_source,
        "rate_limit": rate_limit.as_dict(),
        "quota": quota,
        "resume_cursor": resume_cursor,
    }


def _graphql_open_issue_list(*, repo: str, limit: int, label: str | None = None) -> dict[str, Any]:
    """Run one bounded GraphQL-backed issue list and classify its failure."""
    args = [
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
    ]
    if label:
        args.extend(["--label", label])
    args.extend(
        [
            "--limit",
            str(limit),
            "--json",
            "number,title,state,labels,url,assignees",
        ]
    )
    result = _gh(args)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "gh issue list failed"
        return {
            "status": "quota_blocked" if _is_graphql_quota_error(error) else "error",
            "listed": [],
            "error": error,
        }
    try:
        listed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "error", "listed": [], "error": f"invalid gh JSON: {exc}"}
    if not isinstance(listed, list) or any(not isinstance(row, dict) for row in listed):
        return {
            "status": "error",
            "listed": [],
            "error": "expected gh issue list JSON array",
        }
    return {"status": "ok", "listed": listed, "error": ""}


def _rest_open_issue_list(
    *, repo: str, limit: int, page: int, label: str | None = None
) -> dict[str, Any]:
    """Run one bounded REST issue page and return a resumable cursor when capped."""
    per_page = min(max(limit, 1), 100)
    raw_page, error = _rest_open_issue_page(
        repo=repo,
        page=page,
        per_page=per_page,
        label=label,
    )
    if raw_page is None:
        return {"status": "error", "listed": [], "error": error}
    listed = [
        normalized for issue in raw_page if (normalized := _normalize_rest_issue(issue)) is not None
    ]
    return {
        "status": "ok",
        "listed": listed[:limit],
        "error": "",
        "resume_cursor": (
            {"source": "rest", "page": page + 1, "limit": limit}
            if len(raw_page) >= per_page
            else None
        ),
    }


def _quota_blocked_listing(
    *,
    rate_limit: RateLimitSnapshot,
    decision: dict[str, Any],
    page: int,
    limit: int,
    error: str,
) -> dict[str, Any]:
    """Return an empty, resumable result when no safe issue source is available."""
    return _listing_result(
        status="quota_blocked",
        listed=[],
        error=error,
        data_source="none",
        rate_limit=rate_limit,
        quota=decision,
        resume_cursor={"source": "rest", "page": page, "limit": limit},
    )


def _rest_fallback_listing(
    *,
    repo: str,
    limit: int,
    resume_page: int,
    label: str | None,
    rate_limit: RateLimitSnapshot,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Use one REST page when safe, or return a no-row quota handoff."""
    core_is_blocked = (
        rate_limit.core_remaining is not None
        and rate_limit.core_remaining <= DEFAULT_CORE_SAFETY_THRESHOLD
    )
    if core_is_blocked:
        decision = {
            **decision,
            "status": "quota_blocked",
            "resource": "core",
            "reason": "core_budget_below_safety_threshold",
            "message": "REST fallback is also blocked by the configured core safety threshold",
            "resume_after": rate_limit.core_reset_at,
        }
        return _quota_blocked_listing(
            rate_limit=rate_limit,
            decision=decision,
            page=resume_page,
            limit=limit,
            error=decision["message"],
        )

    rest = _rest_open_issue_list(
        repo=repo,
        page=resume_page,
        limit=limit,
        label=label,
    )
    if rest["status"] == "ok":
        return _listing_result(
            status="ok",
            listed=rest["listed"],
            error="",
            data_source="rest",
            rate_limit=rate_limit,
            quota=decision,
            resume_cursor=rest["resume_cursor"],
        )
    rest_error = rest["error"]
    if decision.get("status") != "quota_blocked" and _is_graphql_quota_error(rest_error):
        decision = _quota_blocked_after_graphql_error(decision, message=rest_error)
    if decision.get("status") == "quota_blocked":
        return _quota_blocked_listing(
            rate_limit=rate_limit,
            decision=decision,
            page=resume_page,
            limit=limit,
            error=rest_error,
        )
    return _listing_result(
        status="error",
        listed=[],
        error=rest_error,
        data_source="rest",
        rate_limit=rate_limit,
        quota=decision,
        resume_cursor=None,
    )


def _list_open_issues(
    *,
    repo: str,
    limit: int,
    min_graphql_remaining: int,
    resume_page: int = 1,
    label: str | None = None,
) -> dict[str, Any]:
    """List open issues with a quota guard and a bounded REST resume path."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if resume_page <= 0:
        raise ValueError("resume_page must be positive")

    rate_limit = _rate_limit_snapshot()
    decision = graphql_budget_decision(
        rate_limit,
        expected_graphql_requests=1,
        min_graphql_remaining=min_graphql_remaining,
        expected_core_requests=1,
        min_core_remaining=DEFAULT_CORE_SAFETY_THRESHOLD,
    )
    if rate_limit.status != "ok":
        return _quota_blocked_listing(
            rate_limit=rate_limit,
            decision=decision,
            page=resume_page,
            limit=limit,
            error=decision["message"],
        )
    force_rest = resume_page > 1
    if decision["status"] == "ok" and not force_rest:
        graphql = _graphql_open_issue_list(
            repo=repo,
            limit=limit,
            label=label,
        )
        if graphql["status"] == "ok":
            return _listing_result(
                status="ok",
                listed=graphql["listed"],
                error="",
                data_source="graphql",
                rate_limit=rate_limit,
                quota=decision,
                resume_cursor=None,
            )
        if graphql["status"] == "error":
            return _listing_result(
                status="error",
                listed=[],
                error=graphql["error"],
                data_source="graphql",
                rate_limit=rate_limit,
                quota=decision,
                resume_cursor=None,
            )
        decision = _quota_blocked_after_graphql_error(decision, message=graphql["error"])

    return _rest_fallback_listing(
        repo=repo,
        limit=limit,
        resume_page=resume_page,
        label=label,
        rate_limit=rate_limit,
        decision=decision,
    )


def expand_issue_numbers(values: list[int], *, expand_range: bool) -> list[int]:
    """Return issue numbers, expanding a two-number range when requested."""
    if expand_range and len(values) == 2 and values[0] < values[1]:
        return list(range(values[0], values[1] + 1))
    return values


def _labels(issue: dict[str, Any]) -> list[str]:
    """Return compact label names from gh issue JSON."""
    return sorted(
        label if isinstance(label, str) else str(label.get("name", ""))
        for label in issue.get("labels", [])
        if (isinstance(label, str) and label) or (isinstance(label, dict) and label.get("name"))
    )


def _assignees(issue: dict[str, Any]) -> list[str]:
    """Return compact assignee logins from gh issue JSON."""
    return sorted(
        user if isinstance(user, str) else str(user.get("login", ""))
        for user in issue.get("assignees", [])
        if (isinstance(user, str) and user) or (isinstance(user, dict) and user.get("login"))
    )


def _issue_state(issue: dict[str, Any]) -> str:
    """Return the normalized GitHub issue state, or an empty string when unknown."""
    state = issue.get("state")
    if not isinstance(state, str):
        return ""
    return state.strip().upper()


def _non_open_state_classification(state: str) -> tuple[str, str] | None:
    """Return a fail-closed classification for missing or non-open issue state."""
    if not state:
        return "state_unknown", "issue state missing or unknown; skip autonomous claim"
    if state != "OPEN":
        return "closed", f"issue state is {state}; skip autonomous claim"
    return None


def _explicit_blocker_label(labels: list[str]) -> str | None:
    """Return the deterministic explicit blocker label, when one is present."""
    return next((label for label in labels if label.startswith(BLOCKED_LABEL_PREFIX)), None)


def _issue_classification(
    *,
    assignees: list[str],
    claim: dict[str, Any],
    labels: list[str],
    state: str,
) -> tuple[str, str]:
    """Return a short claimability classification and rationale."""
    if state_classification := _non_open_state_classification(state):
        return state_classification
    if assignees:
        return "assigned", "assigned; skip auto-claim"
    if _is_blocked_external_issue(labels):
        return "blocked_external", "external input required; omit from default agent queue"
    if COMPUTE_ROUTING_LABEL in labels:
        return (
            "needs_compute",
            "compute or private execution authorization required; skip implementation dispatch",
        )
    if blocker_label := _explicit_blocker_label(labels):
        return (
            "blocked_label",
            f"explicit blocker label {blocker_label}; skip autonomous claim",
        )
    if dispatch_stop_label := next(
        (label for label in labels if label in UNCLAIMABLE_LABELS), None
    ):
        return (
            "blocked_label",
            f"explicit dispatch-stop label {dispatch_stop_label}; skip autonomous claim",
        )
    if claim.get("ok") is False:
        return "claim_unknown", "unable to read claim state"
    if claim.get("claimed"):
        return "claimed", "already claimed by another worker"
    return "claimable", "open, unassigned, and unclaimed"


def _claim_payload(claim: dict[str, Any]) -> dict[str, Any]:
    """Return the stable claim subset exposed in issue snapshots."""
    return {
        "ok": claim.get("ok"),
        "claimed": claim.get("claimed"),
        "claim_ref": claim.get("claim_ref"),
        "sha": claim.get("sha"),
    }


def _admission_reason(admission: dict[str, Any]) -> str:
    """Return the stable admission reason exposed by the canonical gate."""
    reason = admission.get("admission_reason")
    if isinstance(reason, str) and reason.strip():
        return reason
    classification = admission.get("classification")
    return {
        "parent": "parent_not_leaf",
        "blocked": "blocked",
        "review": "covering_pr_open",
        "needs_ready_label": "needs_ready_label",
        "needs_spec": "needs_spec",
        "needs_dependency": "dependency_missing",
        "wrong_owner_repo": "wrong_owner_repo",
        "state_conflict": "state_label_conflict",
        "stale_running": "stale_running_state",
        "error": "error",
        "ready": "claimable",
    }.get(str(classification), str(classification or "unknown"))


def _is_external_admission(admission: dict[str, Any]) -> bool:
    """Return whether an admission row is blocked by an external input."""
    return _admission_reason(admission) == "external_input_missing"


def _transition_plan(issue: dict[str, Any]) -> dict[str, Any]:
    """Attach a read-only blocker transition projection to one issue row."""
    try:
        return blocker_transition.plan_transition(issue)
    except (TypeError, ValueError, blocker_transition.TransitionError) as exc:
        return {
            "schema": blocker_transition.SCHEMA,
            "status": "error",
            "error": str(exc),
            "no_write": True,
        }


def _transition_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    """Count transition classes without collapsing distinct blockers into a score."""
    counts: dict[str, int] = {}
    for issue in issues:
        transition = issue.get("transition")
        if not isinstance(transition, dict):
            continue
        blocker_class = transition.get("blocker_class")
        if isinstance(blocker_class, str):
            counts[blocker_class] = counts.get(blocker_class, 0) + 1
    return dict(sorted(counts.items()))


def _admission_error(*, claim: dict[str, Any], error: str) -> dict[str, Any]:
    """Return a fail-closed admission row when the canonical preflight is unavailable."""
    return {
        "schema": goal_issue_admission.SCHEMA,
        "ok": False,
        "outcome": "error",
        "write_attempted": False,
        "source_ref": goal_issue_admission.DEFAULT_SOURCE_REF,
        "classification": "error",
        "admission_reason": "error",
        "reasons": [error],
        "ready": False,
        "write_allowed": False,
        "claim": goal_issue_admission.compact_admission(
            {
                "outcome": "error",
                "source_ref": goal_issue_admission.DEFAULT_SOURCE_REF,
                "preflight": {"claim": claim},
            }
        )["claim"],
        "claim_outcome": "unavailable",
    }


def _issue_admission(
    issue: dict[str, Any],
    *,
    number: int,
    claim: dict[str, Any],
    repo: str,
    remote: str,
) -> dict[str, Any]:
    """Attach the canonical read-only admission result to one issue snapshot.

    Ready candidates use the live wrapper so optional typed dependency packets
    are consumed by the owner that defines them.  Obvious non-ready rows use
    the same pure evaluator locally, avoiding a second GitHub read while still
    exposing the exact generic classification and claim state.
    """
    labels = _labels(issue)
    state = _issue_state(issue)
    assignees = _assignees(issue)
    if state != "OPEN":
        classification = "closed" if state else "state_unknown"
        reason = (
            f"issue state is {state}; skip autonomous claim"
            if state
            else "issue state missing or unknown; skip autonomous claim"
        )
        return goal_issue_admission.compact_preflight(
            {
                "schema": "issue_implementability.v1",
                "classification": classification,
                "admission_reason": classification,
                "reasons": [reason],
                "ready": False,
                "write_allowed": False,
                "claim": claim,
            }
        )
    has_obvious_blocker = (
        _is_blocked_external_issue(labels)
        or COMPUTE_ROUTING_LABEL in labels
        or _explicit_blocker_label(labels) is not None
        or any(label in UNCLAIMABLE_LABELS for label in labels)
    )
    use_live_preflight = (
        state == "OPEN"
        and "state:ready" in labels
        and not assignees
        and not has_obvious_blocker
        and claim.get("ok") is True
        and claim.get("claimed") is not True
    )
    if use_live_preflight:
        try:
            payload = goal_issue_admission.admit_issue(
                number,
                repo=repo,
                remote=remote,
                source_ref=goal_issue_admission.DEFAULT_SOURCE_REF,
                check_only=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _admission_error(claim=claim, error=str(exc))
        return goal_issue_admission.compact_admission(payload)

    normalized_issue = dict(issue)
    normalized_issue["body"] = issue.get("body", "")
    normalized_issue["title"] = issue.get("title", "") or ""
    normalized_issue["url"] = issue.get("url", "") or ""
    try:
        preflight = issue_implementability.evaluate_issue(normalized_issue, claim)
    except (TypeError, ValueError) as exc:
        return _admission_error(claim=claim, error=str(exc))
    return goal_issue_admission.compact_preflight(preflight)


def _snapshot_admission_fields(
    admission: dict[str, Any],
    *,
    fallback_classification: str,
    fallback_reason: str,
) -> tuple[str, str]:
    """Project canonical admission fields onto the compact snapshot row."""
    classification = admission.get("classification")
    reasons = admission.get("reasons")
    reason = reasons[0] if isinstance(reasons, list) and reasons else None
    if isinstance(classification, str) and classification and isinstance(reason, str) and reason:
        return classification, reason
    return fallback_classification, fallback_reason


def _claim_status_payload(
    issue_number: int,
    *,
    remote: str,
    ok: bool,
    claimed: bool | None,
    sha: str | None,
    error: str = "",
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Return an issue-claim status payload matching ``status_issue`` shape."""
    payload: dict[str, Any] = {
        "schema": "issue_claim.v1",
        "action": "status",
        "ok": ok,
        "claimed": claimed,
        "issue": issue_number,
        "remote": remote,
        "claim_ref": short_claim_ref(issue_number),
        "sha": sha,
        "command": command or [],
    }
    if error:
        payload["error"] = error
    return payload


def _batch_claim_statuses(issue_numbers: list[int], *, remote: str) -> dict[int, dict[str, Any]]:
    """Fetch all issue-claim refs once and return status payloads for each issue."""
    unique_numbers = sorted(set(issue_numbers))
    if not unique_numbers:
        return {}

    command = ["git", "ls-remote", "--heads", remote, "refs/heads/agent-claims/issue-*"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
        return {
            number: _claim_status_payload(
                number,
                remote=remote,
                ok=False,
                claimed=None,
                sha=None,
                error=error,
                command=command,
            )
            for number in unique_numbers
        }

    claimed_refs: dict[int, str] = {}
    ref_prefix = "refs/heads/agent-claims/issue-"
    for line in (result.stdout or "").strip().splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].startswith(ref_prefix):
            continue
        issue_text = parts[1].removeprefix(ref_prefix)
        if issue_text.isdigit():
            claimed_refs[int(issue_text)] = parts[0]

    return {
        number: _claim_status_payload(
            number,
            remote=remote,
            ok=True,
            claimed=number in claimed_refs,
            sha=claimed_refs.get(number),
            command=command,
        )
        for number in unique_numbers
    }


def _is_blocked_external_issue(labels: list[str]) -> bool:
    """Return whether labels describe an issue blocked on external assets or input."""
    label_set = set(labels)
    if BLOCKED_EXTERNAL_INPUT_LABEL in label_set:
        return True
    return EXTERNAL_RESOURCE_LABEL in label_set and bool(EXTERNAL_BLOCKER_LABELS & label_set)


def _body_excerpt(body: Any, *, limit: int) -> tuple[str, bool]:
    """Return a whitespace-normalized body excerpt and truncation flag."""
    text = " ".join(str(body or "").split())
    return text[:limit], len(text) > limit


def _load_blocker_decisions(  # noqa: C901 - fail-closed artifact parsing.
    paths: list[str],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Load compact per-issue blocker decisions from external run artifacts."""
    decisions: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for raw_path in paths:
        path = pathlib.Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{raw_path}: unavailable or malformed JSON ({exc})")
            continue
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("decisions"), list):
            rows = payload["decisions"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            errors.append(f"{raw_path}: expected a decision object or decisions list")
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"{raw_path}: decision {index} is not an object")
                continue
            issue = row.get("issue")
            if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
                errors.append(f"{raw_path}: decision {index} has no positive integer issue")
                continue
            status = row.get("status")
            if status not in BLOCKER_DECISION_STATUSES:
                errors.append(
                    f"{raw_path}: issue {issue} has unsupported blocker decision status {status!r}"
                )
                continue
            reason = row.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{raw_path}: issue {issue} blocker decision has no reason")
                continue
            if status in {"blocked_unchanged", "blocker_changed"}:
                fingerprint = row.get("current_fingerprint")
                if not isinstance(fingerprint, str) or not BLOCKER_DECISION_DIGEST_RE.fullmatch(
                    fingerprint
                ):
                    errors.append(
                        f"{raw_path}: issue {issue} {status} decision has no valid current_fingerprint"
                    )
                    continue
            if issue in decisions:
                errors.append(f"{raw_path}: duplicate blocker decision for issue {issue}")
                continue
            decisions[issue] = row
    return decisions, errors


def _apply_blocker_decision(
    issue: dict[str, Any], decisions: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Fence dispatch when an external blocker decision requires it."""
    number = issue.get("number")
    decision = decisions.get(number) if isinstance(number, int) else None
    if decision is None:
        return issue
    status = decision["status"]
    if status == "blocked_unchanged":
        classification = "blocked_receipt"
        reason = "blocker receipt unchanged; skip autonomous claim"
    else:
        classification = "needs_re_evaluation"
        reason = "blocker receipt changed or is invalid; require fresh evaluation before claim"
    return {
        **issue,
        "classification": classification,
        "reason": reason,
        "dispatch_allowed": False,
        "blocker_decision": {
            "status": status,
            "reason": decision.get("reason", ""),
            "receipt_digest": decision.get("receipt_digest"),
            "current_fingerprint": decision.get("current_fingerprint"),
        },
    }


def _recommended_context_pack(number: int, labels: list[str], title: str) -> str:
    """Return a conservative context-pack hint for a worker prompt."""
    label_text = " ".join(labels).lower()
    title_text = title.lower()
    if "docs" in label_text or "doc" in title_text:
        return "docs/context/INDEX.md"
    if "benchmark" in label_text or "benchmark" in title_text:
        return "docs/benchmark_camera_ready.md"
    if "workflow" in label_text or "workflow" in title_text:
        return "docs/context/goal_driven_agent_loops_2026-05-13.md"
    return f"docs/context/issue_{number}* if present, otherwise docs/context/INDEX.md"


def _validate_explicit_rest_issue(issue: Any, *, requested_number: int) -> str | None:
    """Return an error for malformed successful REST issue payloads."""
    if not isinstance(issue, dict):
        return "REST issue response was not an object"
    issue_number = issue.get("number")
    if type(issue_number) is not int or issue_number < 1:
        return "REST issue response has no positive integer number"
    if issue_number != requested_number:
        return (
            f"REST issue response number {issue_number} does not match requested issue "
            f"{requested_number}"
        )
    for field in ("title", "body", "state", "url"):
        if not isinstance(issue.get(field), str):
            return f"REST issue response field {field!r} is not a string"
    state = issue["state"].strip().upper()
    if state not in {"OPEN", "CLOSED"}:
        return f"REST issue response has unknown state {issue['state']!r}"
    for field in ("labels", "assignees"):
        values = issue.get(field)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            return f"REST issue response field {field!r} is not a list of strings"
    return None


def fetch_issue(number: int, *, repo: str, body_limit: int, remote: str) -> dict[str, Any]:
    """Fetch one issue and return a compact orchestration snapshot.

    The explicit read routes through the REST-backed normalized reader
    :func:`scripts.dev.gh_issue_rest.fetch_issue` instead of the GraphQL-backed
    ``gh issue view --json``, so explicit snapshots keep succeeding when GraphQL
    quota is exhausted but the REST API is healthy (issue #6845). Missing,
    malformed, and REST-failed responses remain fail-closed error rows.
    """
    try:
        issue = gh_issue_rest.fetch_issue(number, repo=repo)
    except (TypeError, ValueError, KeyError) as exc:
        return {
            "number": number,
            "status": "error",
            "error": f"REST issue read returned malformed data: {exc}",
        }
    if not isinstance(issue, dict):
        return {
            "number": number,
            "status": "error",
            "error": "REST issue read returned a non-object response",
        }
    if issue.get("status") != "ok":
        return {
            "number": number,
            "status": "error",
            "error": str(issue.get("error", "REST issue read failed")),
        }
    if error := _validate_explicit_rest_issue(issue, requested_number=number):
        return {
            "number": number,
            "status": "error",
            "error": f"REST issue read returned malformed data: {error}",
        }
    # The REST reader already normalizes labels/assignees to sorted name lists and
    # uppercases state; re-sorting and re-applying ``_issue_state`` keep the
    # snapshot contract stable and defensive against future reader drift.
    labels = sorted(issue.get("labels") or [])
    assignees = sorted(issue.get("assignees") or [])
    state = _issue_state(issue)
    claim = status_issue(number, remote=remote)
    fallback_classification, fallback_reason = _issue_classification(
        assignees=assignees,
        claim=claim,
        labels=labels,
        state=state,
    )
    admission = _issue_admission(
        issue,
        number=int(issue.get("number", number)),
        claim=claim,
        repo=repo,
        remote=remote,
    )
    classification, reason = _snapshot_admission_fields(
        admission,
        fallback_classification=fallback_classification,
        fallback_reason=fallback_reason,
    )
    excerpt, truncated = _body_excerpt(issue.get("body"), limit=body_limit)
    return {
        "number": issue.get("number", number),
        "status": "ok",
        "title": issue.get("title", ""),
        "state": state,
        "url": issue.get("url", ""),
        "labels": labels,
        "assignees": assignees,
        "body_excerpt": excerpt,
        "body_truncated": truncated,
        "claim": _claim_payload(claim),
        "admission": admission,
        "transition": _transition_plan(issue),
        "classification": classification,
        "reason": reason,
        "linked_prs": [],
        "recommended_context_pack": _recommended_context_pack(
            int(issue.get("number", number)), labels, str(issue.get("title", ""))
        ),
    }


def _snapshot_from_issue_list(
    issue: dict[str, Any],
    *,
    repo: str,
    remote: str,
    claim_statuses: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an issue snapshot using preloaded issue fields."""
    try:
        number = int(str(issue.get("number")))
    except (TypeError, ValueError):
        return {"status": "error", "error": "invalid issue number in gh list payload"}

    labels = _labels(issue)
    assignees = _assignees(issue)
    state = _issue_state(issue)
    claim = (
        claim_statuses.get(number)
        if claim_statuses is not None
        else status_issue(number, remote=remote)
    )
    if not isinstance(claim, dict):
        claim = _claim_status_payload(
            number,
            remote=remote,
            ok=False,
            claimed=None,
            sha=None,
            error="claim status unavailable",
        )
    fallback_classification, fallback_reason = _issue_classification(
        assignees=assignees,
        claim=claim,
        labels=labels,
        state=state,
    )
    admission = _issue_admission(
        issue,
        number=number,
        claim=claim,
        repo=repo,
        remote=remote,
    )
    classification, reason = _snapshot_admission_fields(
        admission,
        fallback_classification=fallback_classification,
        fallback_reason=fallback_reason,
    )
    return {
        "number": number,
        "status": "ok",
        "title": issue.get("title", ""),
        "state": state,
        "url": issue.get("url", ""),
        "labels": labels,
        "assignees": assignees,
        "claim": _claim_payload(claim),
        "body_excerpt": "",
        "body_truncated": False,
        "classification": classification,
        "reason": reason,
        "linked_prs": [],
        "admission": admission,
        "transition": _transition_plan(issue),
    }


def snapshot_claimable_issues(
    *,
    repo: str,
    remote: str,
    body_limit: int,
    limit: int,
    include_blocked_external: bool = False,
    min_graphql_remaining: int = DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
    resume_page: int = 1,
    blocker_decision_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Return a compact candidate/open issue snapshot with live claimable rows separated.

    Claimable discovery constrains issue listing to ``state:ready`` candidates before the
    bounded page size, so newer non-ready issues cannot evict older ready leaves from the
    scanned window. GraphQL discovery is used only when the quota preflight leaves the
    configured safety margin. Otherwise the bounded REST page is used, and a
    ``resume_cursor`` is returned when another page exists.

    ``queue_completeness`` makes zero-work claims explicit: ``complete`` means the
    ready-candidate universe was fully scanned and every live admission succeeded;
    ``incomplete`` means the ready-candidate page was truncated or resumable; and
    ``unavailable`` means discovery itself failed or was quota-blocked. A
    ``claimable_count == 0`` result may only be treated as ``genuine_zero_work`` when
    ``queue_completeness`` is ``complete``.
    """
    body_limit = body_limit if body_limit > 0 else BODY_EXCERPT_CHARS
    blocker_decisions, blocker_errors = _load_blocker_decisions(blocker_decision_paths or [])
    if blocker_errors:
        return {
            "schema": "issue_batch_snapshot.v1",
            "repo": repo,
            "body_excerpt_chars": body_limit,
            "mode": "candidate_queue",
            "legacy_mode": "claimable",
            "status": "error",
            "data_source": "none",
            "queue_completeness": "unavailable",
            "blocker_decision_paths": blocker_decision_paths or [],
            "errors": blocker_errors,
            "issues": [{"status": "error", "error": error} for error in blocker_errors],
            "candidate_count": 0,
            "claimable_issues": [],
            "claimable_count": 0,
            "admission_reason_histogram": {},
        }
    listing = _list_open_issues(
        repo=repo,
        limit=limit,
        min_graphql_remaining=min_graphql_remaining,
        resume_page=resume_page,
        label=issue_implementability.READY_LABEL,
    )
    base = {
        "schema": "issue_batch_snapshot.v1",
        "repo": repo,
        "body_excerpt_chars": body_limit,
        "mode": "candidate_queue",
        "legacy_mode": "claimable",
        "status": listing["status"],
        "data_source": listing["data_source"],
        "rate_limit": listing["rate_limit"],
        "quota": listing["quota"],
        "resume_cursor": listing["resume_cursor"],
        "queue_completeness": "unavailable",
        "blocker_decision_paths": blocker_decision_paths or [],
        "candidate_count": 0,
        "claimable_issues": [],
        "claimable_count": 0,
        "admission_reason_histogram": {},
        "excluded_counts": {"blocked_external": 0},
        "transition_counts": {},
    }
    if listing["status"] != "ok":
        return {
            **base,
            "issues": [
                {
                    "status": listing["status"],
                    "error": listing["error"],
                }
            ],
        }
    listed = listing["listed"]

    issue_numbers = [
        int(issue["number"])
        for issue in listed
        if isinstance(issue, dict) and str(issue.get("number", "")).isdigit()
    ]
    claim_statuses = _batch_claim_statuses(issue_numbers, remote=remote)
    snapshots = [
        _snapshot_from_issue_list(issue, repo=repo, remote=remote, claim_statuses=claim_statuses)
        for issue in listed
    ]
    snapshots = [_apply_blocker_decision(issue, blocker_decisions) for issue in snapshots]
    for issue in snapshots:
        admission = issue.get("admission")
        if isinstance(admission, dict) and "blocker_decision" not in issue:
            issue["classification"] = admission.get("classification") or "error"
            reasons = admission.get("reasons")
            if isinstance(reasons, list) and reasons:
                issue["reason"] = str(reasons[0])
    issues = [
        issue
        for issue in snapshots
        if include_blocked_external
        or not isinstance(issue.get("admission"), dict)
        or not _is_external_admission(issue["admission"])
    ]
    truncated = is_likely_truncated(len(listed), limit=limit) or bool(listing["resume_cursor"])
    if listing["resume_cursor"]:
        truncation_note = (
            f"issue discovery may be capped: got {len(listed)} rows at --limit {limit}; "
            "resume with the returned cursor"
        )
    elif truncated:
        truncation_note = (
            f"issue discovery may be capped: got {len(listed)} rows at --limit {limit}; "
            "raise --limit to inspect more rows"
        )
    else:
        truncation_note = ""
    return {
        **base,
        "mode": "candidate_queue",
        "legacy_mode": "claimable",
        "queue_completeness": "incomplete" if truncated else "complete",
        "truncated": truncated,
        "truncation_note": truncation_note,
        "include_blocked_external": include_blocked_external,
        "candidate_count": len(issues),
        "claimable_issues": [
            issue
            for issue in issues
            if isinstance(issue.get("admission"), dict)
            and issue["admission"].get("ok") is True
            and issue["admission"].get("outcome") == "ready_check_only"
        ],
        "claimable_count": sum(
            1
            for issue in issues
            if isinstance(issue.get("admission"), dict)
            and issue["admission"].get("ok") is True
            and issue["admission"].get("outcome") == "ready_check_only"
        ),
        "admission_reason_histogram": dict(
            sorted(
                Counter(
                    _admission_reason(issue["admission"])
                    for issue in snapshots
                    if isinstance(issue.get("admission"), dict)
                ).items()
            )
        ),
        "excluded_counts": {
            "blocked_external": sum(
                1
                for issue in snapshots
                if isinstance(issue.get("admission"), dict)
                and _is_external_admission(issue["admission"])
            ),
        },
        "transition_counts": _transition_counts(snapshots),
        "issues": issues,
    }


def _next_monthly_review_date(now: datetime | None = None) -> str:
    """Return a stable monthly review date, using the first UTC day of next month."""
    current = now or datetime.now(UTC)
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    return f"{year:04d}-{month:02d}-01"


def _blocked_external_row(issue: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Return one human-review row for a blocked external-asset issue."""
    labels = _labels(issue)
    label_set = set(labels)
    recommendations: list[str] = []
    if BLOCKED_EXTERNAL_INPUT_LABEL not in label_set:
        recommendations.append(f"add `{BLOCKED_EXTERNAL_INPUT_LABEL}`")
    if "state:ready" in label_set:
        recommendations.append("remove `state:ready`")
    return {
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "url": issue.get("url", ""),
        "labels": labels,
        "owner_type": "external data",
        "human_action": (
            "Stage or document the required external data/asset/license before agent execution."
        ),
        "monthly_review_date": _next_monthly_review_date(now),
        "label_recommendation": "; ".join(recommendations) if recommendations else "none",
    }


def _blocked_external_markdown(rows: list[dict[str, Any]]) -> str:
    """Return a compact Markdown report for human review."""
    lines = [
        "# Blocked External Assets Report",
        "",
        "| Issue | Owner | Human action | Monthly review | Label recommendation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        issue = f"#{row['number']} {row['title']}".replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    issue,
                    row["owner_type"],
                    row["human_action"],
                    row["monthly_review_date"],
                    row["label_recommendation"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: Any) -> str:
    """Return a table-safe compact Markdown cell."""
    value_text = "" if value is None else str(value)
    return value_text.replace("|", "\\|").replace("\n", " ")


def snapshot_blocked_external_issues(
    *,
    repo: str,
    report_path: str = "",
    limit: int,
    now: datetime | None = None,
    min_graphql_remaining: int = DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
    resume_page: int = 1,
) -> dict[str, Any]:
    """Return a compact blocked external-assets report."""
    listing = _list_open_issues(
        repo=repo,
        limit=limit,
        min_graphql_remaining=min_graphql_remaining,
        resume_page=resume_page,
        label=EXTERNAL_RESOURCE_LABEL,
    )
    if listing["status"] != "ok":
        rows: list[dict[str, Any]] = []
        errors = [
            {
                "status": listing["status"],
                "error": listing["error"],
            }
        ]
    else:
        errors = []
        rows = [
            _blocked_external_row(issue, now=now)
            for issue in listing["listed"]
            if isinstance(issue, dict) and _is_blocked_external_issue(_labels(issue))
        ]
    markdown = _blocked_external_markdown(rows)
    if report_path:
        path = pathlib.Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {
        "schema": "blocked_external_assets_report.v1",
        "repo": repo,
        "mode": "blocked_external_report",
        "recommended_state_label": BLOCKED_EXTERNAL_INPUT_LABEL,
        "rows": rows,
        "row_count": len(rows),
        "report_path": report_path,
        "markdown": markdown,
        "errors": errors,
        "status": listing["status"],
        "data_source": listing["data_source"],
        "rate_limit": listing["rate_limit"],
        "quota": listing["quota"],
        "resume_cursor": listing["resume_cursor"],
    }


def _portfolio_owner_type(labels: list[str], classification: str) -> str:
    """Return the likely next owner type for an active issue portfolio row."""
    label_set = set(labels)
    if classification == "blocked_external_asset":
        return "external data"
    if classification in {"needs_human_decision", "stale_synthesis"}:
        return "maintainer"
    if (
        "slurm" in label_set
        or "resource:slurm" in label_set
        or "training" in label_set
        or "state:running" in label_set
    ):
        return "Slurm"
    return "agent"


def _portfolio_classification(
    *, labels: list[str], title: str, assignees: list[str], claim: dict[str, Any]
) -> tuple[str, str]:
    """Classify one open issue for compact active-portfolio routing."""
    label_set = set(labels)
    title_text = title.lower()
    if _is_blocked_external_issue(labels):
        return "blocked_external_asset", "external asset, data, license, or staging input required"
    if (
        "blocked" in label_set
        or any(label.startswith(BLOCKED_LABEL_PREFIX) for label in label_set)
        or "decision-required" in label_set
        or "state:blocked" in label_set
        or "state:hold" in label_set
    ):
        return "needs_human_decision", "maintainer decision label blocks autonomous execution"
    if "evidence:analysis-only" in label_set or "diagnostic" in title_text:
        return (
            "diagnostic_only",
            "analysis or diagnostic evidence only; do not present as benchmark proof",
        )
    if "type:synthesis" in label_set and "state:ready" not in label_set:
        return "stale_synthesis", "synthesis issue lacks ready-state routing"
    if "priority: high" in label_set and (
        "benchmark" in label_set or "research" in label_set or "epic" in label_set
    ):
        return "paper_critical", "high-priority benchmark or research surface"
    if claim.get("ok") is False:
        return "needs_human_decision", "unable to read claim state; skip autonomous claim"
    if assignees or claim.get("claimed"):
        return "needs_human_decision", "assigned or already claimed; skip autonomous claim"
    return "executable_now", "unassigned, unclaimed, and not blocked by labels"


def _portfolio_label_recommendation(labels: list[str], classification: str) -> str:
    """Return label-only recommendations for one portfolio row."""
    label_set = set(labels)
    recommendation_rules = {
        "blocked_external_asset": lambda: [
            (
                BLOCKED_EXTERNAL_INPUT_LABEL not in label_set,
                f"add `{BLOCKED_EXTERNAL_INPUT_LABEL}`",
            ),
            ("state:ready" in label_set, "remove `state:ready`"),
        ],
        "needs_human_decision": lambda: [
            ("decision-required" not in label_set, "add `decision-required`"),
            ("state:ready" in label_set, "remove `state:ready`"),
        ],
        "diagnostic_only": lambda: [
            ("evidence:analysis-only" not in label_set, "add `evidence:analysis-only`")
        ],
        "stale_synthesis": lambda: [
            ("decision-required" not in label_set, "add `decision-required`")
        ],
        "paper_critical": lambda: [("paper-critical" not in label_set, "add `paper-critical`")],
        "executable_now": lambda: [("state:ready" not in label_set, "add `state:ready`")],
    }
    rules = recommendation_rules.get(classification, lambda: [])()
    recommendations = [
        recommendation for should_recommend, recommendation in rules if should_recommend
    ]
    return "; ".join(recommendations) if recommendations else "none"


def _portfolio_next_action(classification: str) -> str:
    """Return one compact next action for a portfolio row."""
    actions = {
        "blocked_external_asset": "Park until the required asset, data, license, or staging note exists.",
        "diagnostic_only": "Keep as diagnostic evidence unless a follow-up benchmark proof issue is opened.",
        "executable_now": "Agent may claim and execute the issue contract.",
        "needs_human_decision": "Maintainer should decide, relabel, or split before agent execution.",
        "paper_critical": "Prioritize for release evidence review and claim-boundary checks.",
        "stale_synthesis": "Refresh, supersede, or close the synthesis before new implementation work.",
    }
    return actions[classification]


def _active_portfolio_row(
    issue: dict[str, Any], *, remote: str, claim_statuses: dict[int, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return one active-portfolio row for an open issue."""
    number = int(str(issue.get("number")))
    labels = _labels(issue)
    assignees = _assignees(issue)
    claim_value = (
        claim_statuses.get(number)
        if claim_statuses is not None
        else status_issue(number, remote=remote)
    )
    claim = (
        claim_value
        if isinstance(claim_value, dict)
        else _claim_status_payload(
            number,
            remote=remote,
            ok=False,
            claimed=None,
            sha=None,
            error="claim status unavailable",
        )
    )
    title = "" if issue.get("title") is None else issue.get("title", "")
    url = "" if issue.get("url") is None else issue.get("url", "")
    classification, reason = _portfolio_classification(
        labels=labels,
        title=str(title),
        assignees=assignees,
        claim=claim,
    )
    return {
        "number": number,
        "title": title,
        "url": url,
        "labels": labels,
        "classification": classification,
        "reason": reason,
        "owner_type": _portfolio_owner_type(labels, classification),
        "next_action": _portfolio_next_action(classification),
        "label_recommendation": _portfolio_label_recommendation(labels, classification),
    }


def _active_portfolio_markdown(rows: list[dict[str, Any]]) -> str:
    """Return a compact Markdown portfolio table for maintainer review."""
    lines = [
        "# Active Issue Portfolio",
        "",
        "| Issue | Classification | Owner | Next action | Label recommendation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        issue = f"#{row['number']} {row['title']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(issue),
                    _markdown_cell(row["classification"]),
                    _markdown_cell(row["owner_type"]),
                    _markdown_cell(row["next_action"]),
                    _markdown_cell(row["label_recommendation"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def snapshot_active_issue_portfolio(
    *,
    repo: str,
    remote: str,
    report_path: str = "",
    limit: int,
    min_graphql_remaining: int = DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
    resume_page: int = 1,
) -> dict[str, Any]:
    """Return a compact active issue portfolio for routing and demotion review."""
    listing = _list_open_issues(
        repo=repo,
        limit=limit,
        min_graphql_remaining=min_graphql_remaining,
        resume_page=resume_page,
    )
    if listing["status"] != "ok":
        rows: list[dict[str, Any]] = []
        errors = [
            {
                "status": listing["status"],
                "error": listing["error"],
            }
        ]
    else:
        errors = []
        issue_numbers = [
            int(issue["number"])
            for issue in listing["listed"]
            if isinstance(issue, dict) and str(issue.get("number", "")).isdigit()
        ]
        claim_statuses = _batch_claim_statuses(issue_numbers, remote=remote)
        rows = [
            _active_portfolio_row(issue, remote=remote, claim_statuses=claim_statuses)
            for issue in listing["listed"]
            if isinstance(issue, dict) and issue.get("number") is not None
        ]
    counts = dict.fromkeys(sorted(PORTFOLIO_CLASSIFICATIONS), 0)
    for row in rows:
        classification = str(row.get("classification", ""))
        if classification in counts:
            counts[classification] += 1
    markdown = _active_portfolio_markdown(rows)
    if report_path:
        path = pathlib.Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {
        "schema": "active_issue_portfolio.v1",
        "repo": repo,
        "mode": "active_portfolio",
        "rows": rows,
        "row_count": len(rows),
        "classification_counts": counts,
        "report_path": report_path,
        "markdown": markdown,
        "errors": errors,
        "status": listing["status"],
        "data_source": listing["data_source"],
        "rate_limit": listing["rate_limit"],
        "quota": listing["quota"],
        "resume_cursor": listing["resume_cursor"],
    }


def _context_capsule(issue: dict[str, Any]) -> dict[str, Any]:
    """Return a compact worker-seeding capsule for one issue snapshot."""
    return {
        "schema": "issue_context_capsule.v1",
        "issue": {
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "url": issue.get("url", ""),
            "labels": issue.get("labels", []),
            "body_excerpt": issue.get("body_excerpt", ""),
        },
        "admission": issue.get("admission", {}),
        "claim": issue.get("claim", {}),
        "files_to_read": [issue.get("recommended_context_pack", "docs/context/INDEX.md")],
        "tests_to_run": [],
        "docs_to_update": [],
        "known_risks": [],
        "worker_prompt_seed": (
            "Use this capsule as the first context source. Avoid broad repository search until "
            "these files have been inspected and summarized."
        ),
    }


def _write_capsules(issues: list[dict[str, Any]], capsule_dir: pathlib.Path) -> None:
    """Write one optional context capsule per successfully fetched issue."""
    capsule_dir.mkdir(parents=True, exist_ok=True)
    for issue in issues:
        if issue.get("status") != "ok":
            continue
        path = capsule_dir / f"issue_{issue['number']}_context_capsule.json"
        path.write_text(
            json.dumps(_context_capsule(issue), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        issue["context_capsule_path"] = str(path)


def snapshot_issues(
    numbers: list[int],
    *,
    repo: str,
    body_limit: int,
    remote: str,
    capsule_dir: str = "",
    blocker_decision_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Return a compact issue-batch snapshot."""
    blocker_decisions, blocker_errors = _load_blocker_decisions(blocker_decision_paths or [])
    if blocker_errors:
        return {
            "schema": "issue_batch_snapshot.v1",
            "repo": repo,
            "body_excerpt_chars": body_limit,
            "status": "error",
            "blocker_decision_paths": blocker_decision_paths or [],
            "errors": blocker_errors,
            "issues": [{"status": "error", "error": error} for error in blocker_errors],
        }
    issues = [
        fetch_issue(number, repo=repo, body_limit=body_limit, remote=remote) for number in numbers
    ]
    issues = [_apply_blocker_decision(issue, blocker_decisions) for issue in issues]
    if capsule_dir:
        _write_capsules(issues, pathlib.Path(capsule_dir))
    return {
        "schema": "issue_batch_snapshot.v1",
        "repo": repo,
        "body_excerpt_chars": body_limit,
        "transition_counts": _transition_counts(issues),
        "blocker_decision_paths": blocker_decision_paths or [],
        "issues": issues,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "issues", nargs="*", type=int, help="Issue numbers; two values form a range."
    )
    parser.add_argument(
        "--claimable",
        action="store_true",
        help="Discover bounded open claimable issues without explicit issue numbers.",
    )
    parser.add_argument(
        "--include-blocked-external",
        action="store_true",
        help="Include blocked external-input issues in --claimable output.",
    )
    parser.add_argument(
        "--blocked-external-report",
        action="store_true",
        help="Generate a compact blocked external-assets report instead of claim routing.",
    )
    parser.add_argument(
        "--active-portfolio",
        action="store_true",
        help="Generate a compact active issue portfolio with label recommendations.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional Markdown path for --blocked-external-report.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--body-chars", type=int, default=BODY_EXCERPT_CHARS)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CLAIMABLE_LIMIT,
        help="Bounded issue-page size for discovery/report modes.",
    )
    parser.add_argument(
        "--min-graphql-remaining",
        type=int,
        default=DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
        help="GraphQL quota safety margin retained after the estimated lookup request.",
    )
    parser.add_argument(
        "--resume-page",
        type=int,
        default=1,
        help="REST issue-list page to resume from after a quota-bounded snapshot.",
    )
    parser.add_argument(
        "--capsule-dir",
        default="",
        help="Optional directory for per-issue context capsule JSON files.",
    )
    parser.add_argument(
        "--blocker-decision",
        action="append",
        default=[],
        help=(
            "External per-issue blocker-decision JSON artifact; may be repeated. "
            "Unchanged or re-evaluation decisions fence autonomous claims."
        ),
    )
    parser.add_argument(
        "--no-expand-range",
        action="store_true",
        help="Treat two issue numbers as exactly two issues instead of an inclusive range.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> int:
    """Return nonzero after printing a CLI contract error."""
    if args.claimable and args.issues:
        print(
            "--claimable cannot be combined with explicit issue numbers",
            file=sys.stderr,
        )
        return 1
    if args.include_blocked_external and not args.claimable:
        print(
            "--include-blocked-external requires --claimable",
            file=sys.stderr,
        )
        return 1
    if args.blocked_external_report and (args.claimable or args.issues):
        print(
            "--blocked-external-report cannot be combined with --claimable or issue numbers",
            file=sys.stderr,
        )
        return 1
    if args.active_portfolio and (args.claimable or args.blocked_external_report or args.issues):
        print(
            "--active-portfolio cannot be combined with --claimable, "
            "--blocked-external-report, or issue numbers",
            file=sys.stderr,
        )
        return 1
    if args.blocker_decision and args.active_portfolio:
        print("--blocker-decision cannot be combined with --active-portfolio", file=sys.stderr)
        return 1
    if args.blocker_decision and args.blocked_external_report:
        print(
            "--blocker-decision cannot be combined with --blocked-external-report",
            file=sys.stderr,
        )
        return 1
    if args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 1
    if args.min_graphql_remaining < 0:
        print("--min-graphql-remaining must be non-negative", file=sys.stderr)
        return 1
    if args.resume_page <= 0:
        print("--resume-page must be positive", file=sys.stderr)
        return 1
    return 0


def _build_payload(args: argparse.Namespace, numbers: list[int]) -> dict[str, Any]:
    """Build the requested CLI payload after argument validation."""
    if args.active_portfolio:
        return snapshot_active_issue_portfolio(
            repo=args.repo,
            remote=args.remote,
            report_path=args.report_path,
            limit=args.limit,
            min_graphql_remaining=args.min_graphql_remaining,
            resume_page=args.resume_page,
        )
    if args.blocked_external_report:
        return snapshot_blocked_external_issues(
            repo=args.repo,
            report_path=args.report_path,
            limit=args.limit,
            min_graphql_remaining=args.min_graphql_remaining,
            resume_page=args.resume_page,
        )
    if args.claimable:
        return snapshot_claimable_issues(
            repo=args.repo,
            remote=args.remote,
            body_limit=max(args.body_chars, 0),
            limit=args.limit,
            include_blocked_external=args.include_blocked_external,
            min_graphql_remaining=args.min_graphql_remaining,
            resume_page=args.resume_page,
            blocker_decision_paths=args.blocker_decision,
        )
    return snapshot_issues(
        numbers,
        repo=args.repo,
        body_limit=max(args.body_chars, 0),
        remote=args.remote,
        capsule_dir=args.capsule_dir,
        blocker_decision_paths=args.blocker_decision,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    validation_error = _validate_args(args)
    if validation_error:
        return validation_error
    numbers = expand_issue_numbers(args.issues, expand_range=not args.no_expand_range)
    try:
        if not (
            args.active_portfolio or args.blocked_external_report or args.claimable or args.issues
        ):
            print(
                "at least one issue number is required unless --claimable, "
                "--blocked-external-report, or --active-portfolio is used",
                file=sys.stderr,
            )
            return 1
        payload = _build_payload(args, numbers)
    except FileNotFoundError:
        print("gh or git command not found", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired as exc:
        print(f"snapshot command timed out: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else json.dumps(payload))
    if "issues" in payload:
        return 1 if any(issue.get("status") == "error" for issue in payload["issues"]) else 0
    return 1 if payload.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
