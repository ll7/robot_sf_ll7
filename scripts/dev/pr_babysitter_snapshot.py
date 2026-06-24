#!/usr/bin/env python3
"""Emit a conservative PR babysitter snapshot.

This helper is intentionally read-only toward GitHub. It wraps the existing compact PR queue
snapshot and adds a higher-level action recommendation for agents that need to babysit an open PR
without manually stitching CI, review, mergeability, and retry-budget state together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.dev.snapshot_pr_queue import DEFAULT_REPO, snapshot_prs

SCHEMA_VERSION = "pr_babysitter_snapshot.v1"
DEFAULT_RETRY_BUDGET = 1


def _state_key(pr: dict[str, Any]) -> str:
    """Return the retry-state key for a PR/head pair."""
    return f"{pr.get('number')}:{pr.get('head_sha', '')}"


def load_retry_state(path: Path | None) -> dict[str, Any]:
    """Load retry state from disk, returning an empty state when absent."""
    if path is None or not path.exists():
        return {"schema": "pr_babysitter_retry_state.v1", "prs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"schema": "pr_babysitter_retry_state.v1", "prs": {}}
    if not isinstance(payload, dict):
        return {"schema": "pr_babysitter_retry_state.v1", "prs": {}}
    payload.setdefault("schema", "pr_babysitter_retry_state.v1")
    payload.setdefault("prs", {})
    if not isinstance(payload["prs"], dict):
        payload["prs"] = {}
    return payload


def save_retry_state(path: Path, state: dict[str, Any]) -> None:
    """Persist retry state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def retry_budget_for_pr(
    pr: dict[str, Any],
    *,
    retry_state: dict[str, Any],
    retry_budget: int,
) -> dict[str, Any]:
    """Return bounded retry accounting for a PR/head pair."""
    key = _state_key(pr)
    entry = retry_state.get("prs", {}).get(key, {})
    attempts = int(entry.get("retry_recommendations", 0) or 0) if isinstance(entry, dict) else 0
    budget = max(retry_budget, 0)
    remaining = max(budget - attempts, 0)
    return {
        "key": key,
        "budget": budget,
        "retry_recommendations": attempts,
        "remaining": remaining,
        "exhausted": remaining <= 0,
    }


def record_retry_recommendation(pr: dict[str, Any], *, retry_state: dict[str, Any]) -> None:
    """Increment retry recommendation state for a PR/head pair."""
    key = _state_key(pr)
    prs = retry_state.setdefault("prs", {})
    entry = prs.setdefault(key, {})
    entry["retry_recommendations"] = int(entry.get("retry_recommendations", 0) or 0) + 1
    entry["pr"] = pr.get("number")
    entry["head_sha"] = pr.get("head_sha", "")


def _trusted_author(author: str, trusted_authors: set[str] | None) -> bool:
    """Return whether an author is trusted for actionable babysitter feedback."""
    return trusted_authors is None or author in trusted_authors


def _string_or_default(value: Any, default: str = "") -> str:
    """Stringify a value, treating only None as missing."""
    return str(value if value is not None else default)


def _review_feedback(
    pr: dict[str, Any],
    *,
    trusted_authors: set[str] | None = None,
) -> dict[str, Any]:
    """Return review/comment state relevant to babysitting."""
    thread_snapshot = pr.get("review_thread_snapshot", {})
    unresolved = 0
    trusted_unresolved = 0
    thread_snapshot_complete = False
    if isinstance(thread_snapshot, dict):
        unresolved = int(thread_snapshot.get("unresolved", 0) or 0)
        thread_snapshot_complete = str(thread_snapshot.get("status", "")) == "ok"
        threads = [
            thread
            for thread in thread_snapshot.get("threads", []) or []
            if isinstance(thread, dict)
        ]
        if unresolved > 0 and not threads:
            trusted_unresolved = unresolved
        for thread in threads:
            if not isinstance(thread, dict) or bool(thread.get("resolved")):
                continue
            comments = [
                comment for comment in thread.get("comments", []) or [] if isinstance(comment, dict)
            ]
            if not comments or any(
                _trusted_author(str(comment.get("author", "")), trusted_authors)
                for comment in comments
            ):
                trusted_unresolved += 1
    reviews = pr.get("reviews", {}) if isinstance(pr.get("reviews"), dict) else {}
    comment_snapshot = (
        pr.get("comment_snapshot") if isinstance(pr.get("comment_snapshot"), dict) else {}
    )
    comment_total = comment_snapshot.get("total")
    comment_count = int(comment_total if comment_total is not None else 0)
    commented_reviews = int(reviews.get("COMMENTED", 0) or 0)
    actionable_review = trusted_unresolved > 0 or (
        commented_reviews > 0 and not thread_snapshot_complete
    )
    return {
        "unresolved_threads": unresolved,
        "trusted_unresolved_threads": trusted_unresolved,
        "submitted_review_counts": reviews,
        "issue_comments": comment_count,
        "has_actionable_feedback": actionable_review or comment_count > 0,
    }


def classify_pr_action(
    pr: dict[str, Any],
    *,
    retry_state: dict[str, Any],
    retry_budget: int = DEFAULT_RETRY_BUDGET,
    trusted_authors: set[str] | None = None,
) -> dict[str, Any]:
    """Classify the next safe babysitter action for one PR."""
    checks = pr.get("checks", {}) if isinstance(pr.get("checks"), dict) else {}
    feedback = _review_feedback(pr, trusted_authors=trusted_authors)
    retry = retry_budget_for_pr(pr, retry_state=retry_state, retry_budget=retry_budget)
    reasons: list[str] = []
    action = "wait"

    state = _string_or_default(pr.get("state")).upper()
    preflight = pr.get("preflight", {}) if isinstance(pr.get("preflight"), dict) else {}
    preflight_status = _string_or_default(preflight.get("status"))
    checks_overall = _string_or_default(checks.get("overall"), "pending")
    mergeable = _string_or_default(pr.get("mergeable"))

    if state and state != "OPEN":
        action = "stop_pr_closed"
        reasons.append(f"state:{state.lower()}")
    elif preflight_status == "stale":
        action = "stop_stale_head"
        reasons.extend(str(reason) for reason in preflight.get("reasons", []) or ["stale"])
    elif preflight_status == "blocked" and mergeable == "CONFLICTING":
        action = "stop_user_help_required"
        reasons.append("merge_conflict")
    elif checks_overall == "failure":
        action = "diagnose_ci_failure"
        reasons.append("ci_failure")
    elif feedback["has_actionable_feedback"]:
        action = "process_review_comment"
        reasons.append("review_feedback")
    elif checks_overall == "pending":
        action = "wait"
        reasons.append("ci_pending")
    elif checks_overall == "success" and mergeable == "MERGEABLE":
        action = "review_for_merge_ready"
        reasons.append("ci_success_mergeable")
    else:
        action = "wait"
        reasons.append("no_safe_write_action")

    after_diagnosis_action = None
    if action == "diagnose_ci_failure":
        after_diagnosis_action = (
            "retry_failed_checks" if retry["remaining"] > 0 else "stop_retry_budget_exhausted"
        )

    return {
        "action": action,
        "reasons": reasons,
        "after_diagnosis_action": after_diagnosis_action,
        "retry_budget": retry,
        "review_feedback": feedback,
        "mutation_guardrails": {
            "github_writes": False,
            "auto_reply_to_humans": False,
            "auto_resolve_threads": False,
            "auto_merge": False,
        },
    }


def build_babysitter_snapshot(
    numbers: list[int],
    *,
    repo: str = DEFAULT_REPO,
    expected_head_sha: str = "",
    retry_state: dict[str, Any] | None = None,
    retry_budget: int = DEFAULT_RETRY_BUDGET,
    record_retry_recommendations: bool = False,
    trusted_authors: set[str] | None = None,
) -> dict[str, Any]:
    """Return a PR babysitter snapshot for explicit PR numbers."""
    state = retry_state if retry_state is not None else load_retry_state(None)
    queue_snapshot = snapshot_prs(
        numbers,
        repo=repo,
        expected_head_sha=expected_head_sha,
        include_review_threads=True,
    )
    babysat_prs: list[dict[str, Any]] = []
    for pr in queue_snapshot.get("prs", []):
        recommendation = classify_pr_action(
            pr,
            retry_state=state,
            retry_budget=retry_budget,
            trusted_authors=trusted_authors,
        )
        if (
            record_retry_recommendations
            and recommendation.get("after_diagnosis_action") == "retry_failed_checks"
        ):
            record_retry_recommendation(pr, retry_state=state)
            recommendation = classify_pr_action(
                pr,
                retry_state=state,
                retry_budget=retry_budget,
                trusted_authors=trusted_authors,
            )
        babysat_prs.append(
            {
                "number": pr.get("number"),
                "title": pr.get("title", ""),
                "state": pr.get("state", ""),
                "head_branch": pr.get("head_branch", ""),
                "head_sha": pr.get("head_sha", ""),
                "mergeable": pr.get("mergeable", ""),
                "checks": pr.get("checks", {}),
                "preflight": pr.get("preflight", {}),
                "review_snapshot": pr.get("review_snapshot", {}),
                "comment_snapshot": pr.get("comment_snapshot", {}),
                "review_thread_snapshot": pr.get("review_thread_snapshot", {}),
                "recommendation": recommendation,
            }
        )
    return {
        "schema": SCHEMA_VERSION,
        "repo": repo,
        "source_schema": queue_snapshot.get("schema"),
        "route_evidence_only": True,
        "prs": babysat_prs,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prs", nargs="+", type=int, help="PR numbers to snapshot.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--expected-head-sha", default="")
    parser.add_argument("--retry-budget", type=int, default=DEFAULT_RETRY_BUDGET)
    parser.add_argument("--retry-state-file", type=Path)
    parser.add_argument(
        "--trusted-author",
        action="append",
        default=None,
        help="Trusted review/comment author login. Repeat to provide an allowlist.",
    )
    parser.add_argument(
        "--record-retry-recommendation",
        action="store_true",
        help="Persist one retry recommendation for failed checks after this snapshot.",
    )
    parser.add_argument("--json", action="store_true", help="Emit pretty JSON.")
    return parser.parse_args(argv)


def _sanitize_payload_for_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a log-safe snapshot payload with sensitive detail removed."""
    prs: list[dict[str, Any]] = []
    for pr in payload.get("prs", []):
        if not isinstance(pr, dict):
            continue
        recommendation = pr.get("recommendation", {})
        if not isinstance(recommendation, dict):
            recommendation = {}
        retry_budget = recommendation.get("retry_budget", {})
        if not isinstance(retry_budget, dict):
            retry_budget = {}

        prs.append(
            {
                "number": pr.get("number"),
                "state": pr.get("state", ""),
                "mergeable": pr.get("mergeable", ""),
                "recommendation": {
                    "action": recommendation.get("action", "wait"),
                    "after_diagnosis_action": recommendation.get("after_diagnosis_action"),
                    "retry_budget": {
                        "budget": retry_budget.get("budget", 0),
                        "retry_recommendations": retry_budget.get("retry_recommendations", 0),
                        "remaining": retry_budget.get("remaining", 0),
                        "exhausted": retry_budget.get("exhausted", False),
                    },
                },
            }
        )

    return {
        "schema": payload.get("schema"),
        "repo": payload.get("repo"),
        "source_schema": payload.get("source_schema"),
        "route_evidence_only": payload.get("route_evidence_only", True),
        "prs": prs,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    retry_state = load_retry_state(args.retry_state_file)
    payload = build_babysitter_snapshot(
        args.prs,
        repo=args.repo,
        expected_head_sha=args.expected_head_sha,
        retry_state=retry_state,
        retry_budget=args.retry_budget,
        record_retry_recommendations=args.record_retry_recommendation,
        trusted_authors=set(args.trusted_author) if args.trusted_author else None,
    )
    if args.retry_state_file and args.record_retry_recommendation:
        save_retry_state(args.retry_state_file, retry_state)
    safe_payload = _sanitize_payload_for_output(payload)
    output_text = (
        json.dumps(safe_payload, indent=2, sort_keys=True) if args.json else json.dumps(safe_payload)
    )
    # The printed payload is rebuilt from an allowlist in _sanitize_payload_for_output.
    print(output_text)  # lgtm[py/clear-text-logging-sensitive-data]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
