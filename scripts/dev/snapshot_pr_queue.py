#!/usr/bin/env python3
"""Emit compact PR queue state for token-efficient goal orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from scripts.dev.check_pr_ci_status import (
    FAILURE_CONCLUSIONS,
    PENDING_STATUSES,
    _rollup_conclusion,
    _rollup_name,
    _rollup_status,
)

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_ACTIVE_LIMIT = 20


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


def _reviews(pr: dict[str, Any]) -> dict[str, int]:
    """Return review-state counts."""
    states: dict[str, int] = {}
    for review in pr.get("reviews", []) or []:
        if not isinstance(review, dict):
            continue
        state = str(review.get("state", "UNKNOWN"))
        states[state] = states.get(state, 0) + 1
    return states


def _checks(pr: dict[str, Any]) -> dict[str, Any]:
    """Return a compact CI check summary from statusCheckRollup."""
    rollup = pr.get("statusCheckRollup", []) or []
    conclusions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    names: set[str] = set()
    for check in rollup:
        if not isinstance(check, dict):
            continue
        conclusion = _rollup_conclusion(check)
        status = _rollup_status(check)
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
        names.add(_rollup_name(check))
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
        "by_conclusion": conclusions,
        "by_status": statuses,
        "names": sorted(names),
    }


def _next_action(*, is_draft: bool, labels: list[str], checks: dict[str, Any]) -> str:
    """Return a compact next-action hint for the parent orchestrator."""
    if is_draft:
        return "review_or_mark_ready_when_local_proof_passes"
    if checks.get("overall") == "failure":
        return "inspect_failing_checks"
    if checks.get("overall") == "pending":
        return "await_ci_or_start_read_only_monitor"
    if "merge-ready" in labels:
        return "merge_readiness_local_check"
    return "review_for_merge_ready"


def _attention(*, next_action: str, is_draft: bool, labels: list[str]) -> str:
    """Return a compact attention category for queue triage."""
    if is_draft:
        return "draft_ready_or_review"
    if next_action == "inspect_failing_checks":
        return "ci_attention"
    if next_action == "await_ci_or_start_read_only_monitor":
        return "ci_pending"
    if "merge-ready" in labels:
        return "merge_attention"
    return "review_attention"


def _pr_payload_from_dict(
    pr: dict[str, Any],
    *,
    default_number: int,
) -> dict[str, Any]:
    """Build a compact PR snapshot from already-loaded fields."""
    is_draft = bool(pr.get("isDraft"))
    labels = _labels(pr)
    checks = _checks(pr)
    pr_payload = {
        "number": pr.get("number", default_number),
        "status": "ok",
        "title": pr.get("title", ""),
        "state": pr.get("state", ""),
        "draft": is_draft,
        "url": pr.get("url", ""),
        "labels": labels,
        "head_branch": pr.get("headRefName", ""),
        "head_sha": pr.get("headRefOid", ""),
        "mergeable": pr.get("mergeable", "unknown"),
        "checks": checks,
        "reviews": _reviews(pr),
    }
    next_action = _next_action(is_draft=is_draft, labels=labels, checks=checks)
    pr_payload["next_action"] = next_action
    pr_payload["attention"] = _attention(next_action=next_action, is_draft=is_draft, labels=labels)
    return pr_payload


def fetch_pr(number: int, *, repo: str) -> dict[str, Any]:
    """Fetch one PR and return a compact queue snapshot."""
    result = _gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,state,isDraft,labels,url,headRefName,headRefOid,mergeable,statusCheckRollup,reviews",
        ]
    )
    if result.returncode != 0:
        return {
            "number": number,
            "status": "error",
            "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
        }
    try:
        pr = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"number": number, "status": "error", "error": f"invalid gh JSON: {exc}"}
    return _pr_payload_from_dict(pr, default_number=number)


def snapshot_active_prs(*, repo: str, limit: int) -> dict[str, Any]:
    """Return a compact active PR queue snapshot."""
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
            "number,title,state,isDraft,labels,url,headRefName,headRefOid,mergeable,statusCheckRollup,reviews",
        ]
    )

    if result.returncode != 0:
        return {
            "schema": "pr_queue_snapshot.v1",
            "repo": repo,
            "mode": "active",
            "prs": [
                {
                    "status": "error",
                    "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
                }
            ],
        }
    try:
        listed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "schema": "pr_queue_snapshot.v1",
            "repo": repo,
            "mode": "active",
            "prs": [
                {
                    "status": "error",
                    "error": f"invalid gh JSON: {exc}",
                }
            ],
        }
    if not isinstance(listed, list):
        return {
            "schema": "pr_queue_snapshot.v1",
            "repo": repo,
            "mode": "active",
            "prs": [
                {
                    "status": "error",
                    "error": "expected gh pr list JSON array",
                }
            ],
        }

    prs = [_pr_payload_from_dict(pr, default_number=-1) for pr in listed if isinstance(pr, dict)]
    return {
        "schema": "pr_queue_snapshot.v1",
        "repo": repo,
        "mode": "active",
        "prs": prs,
    }


def snapshot_prs(numbers: list[int], *, repo: str) -> dict[str, Any]:
    """Return a compact PR queue snapshot."""
    return {
        "schema": "pr_queue_snapshot.v1",
        "repo": repo,
        "prs": [fetch_pr(number, repo=repo) for number in numbers],
    }


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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.active and (args.prs_option is not None or args.prs):
        print("--active cannot be combined with explicit PR numbers", file=sys.stderr)
        return 1
    numbers = args.prs_option if args.prs_option is not None else args.prs
    try:
        if args.active:
            payload = snapshot_active_prs(repo=args.repo, limit=max(args.limit, 1))
        elif not numbers:
            print("at least one PR number is required", file=sys.stderr)
            return 1
        else:
            payload = snapshot_prs(numbers, repo=args.repo)
    except FileNotFoundError:
        print("gh command not found", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired as exc:
        print(f"snapshot command timed out: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else json.dumps(payload))
    return 1 if any(pr.get("status") == "error" for pr in payload["prs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
