#!/usr/bin/env python3
"""Emit compact PR queue state for token-efficient goal orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev.check_pr_ci_status import (
    FAILURE_CONCLUSIONS,
    PENDING_STATUSES,
    _rollup_conclusion,
    _rollup_name,
    _rollup_status,
)

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_ACTIVE_LIMIT = 20
REVIEW_SUMMARY_LIMIT = 4
COMMENT_SUMMARY_LIMIT = 4
COMMENT_BODY_LIMIT = 180
REVIEW_THREAD_LIMIT = 12
REVIEW_THREAD_COMMENT_LIMIT = 2
ROUTE_HEALTH_STATUSES = ("healthy", "stale", "blocked", "unknown")
SCHEMA_VERSION = "pr_queue_snapshot.v1"


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
        return {
            "status": "error",
            "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
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
    """Return a compact CI check summary from statusCheckRollup."""
    rollup = pr.get("statusCheckRollup", []) or []
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
        "by_conclusion": conclusions,
        "by_status": statuses,
        "names": sorted(names),
        "failed": failed,
        "pending": pending,
    }


def _preflight(
    *,
    checks_overall: str,
    expected_head_sha: str,
    head_sha: str,
    is_draft: bool,
    mergeable: str,
) -> dict[str, Any]:
    """Return compact lane preflight status and reasons."""
    reasons: list[str] = []
    if is_draft:
        reasons.append("pr_is_draft")
    head_sha_matches = None
    preflight_status = "healthy"
    if expected_head_sha:
        if not head_sha:
            preflight_status = "blocked"
            reasons.append("missing_head_sha")
        elif head_sha != expected_head_sha:
            preflight_status = "stale"
            reasons.append("head_sha_mismatch")
            head_sha_matches = False
        else:
            head_sha_matches = True
    if checks_overall == "failure":
        preflight_status = "blocked"
        reasons.append("ci_checks_failed")
    if mergeable == "CONFLICTING":
        preflight_status = "blocked"
        reasons.append("mergeable_conflict")
    if not reasons:
        reasons.append("ok")
    return {
        "status": preflight_status,
        "reasons": reasons,
        "expected_head_sha": expected_head_sha,
        "head_sha": head_sha,
        "head_sha_matches_expected": head_sha_matches,
        "checks_overall": checks_overall,
        "mergeable": mergeable,
        "route_evidence_only": True,
    }


def _next_action(
    *, is_draft: bool, labels: list[str], checks: dict[str, Any], preflight: dict[str, Any]
) -> str:
    """Return a compact next-action hint for the parent orchestrator."""
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
    expected_head_sha: str,
) -> dict[str, Any]:
    """Build a compact PR snapshot from already-loaded fields."""
    is_draft = bool(pr.get("isDraft"))
    labels = _labels(pr)
    checks = _checks(pr)
    head_sha = str(pr.get("headRefOid", ""))
    mergeable = str(pr.get("mergeable", "unknown"))
    preflight = _preflight(
        checks_overall=str(checks.get("overall", "")),
        expected_head_sha=expected_head_sha,
        head_sha=head_sha,
        is_draft=is_draft,
        mergeable=mergeable,
    )
    reviews = _reviews(pr)
    pr_payload = {
        "number": pr.get("number", default_number),
        "status": "ok",
        "title": pr.get("title", ""),
        "state": pr.get("state", ""),
        "draft": is_draft,
        "url": pr.get("url", ""),
        "labels": labels,
        "head_branch": pr.get("headRefName", ""),
        "head_sha": head_sha,
        "mergeable": mergeable,
        "checks": checks,
        "reviews": reviews,
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


def fetch_pr(number: int, *, repo: str, expected_head_sha: str = "") -> dict[str, Any]:
    """Fetch one PR and return a compact queue snapshot."""
    result = _gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,state,isDraft,labels,url,headRefName,headRefOid,mergeable,statusCheckRollup,reviews,comments",
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
    return _pr_payload_from_dict(pr, default_number=number, expected_head_sha=expected_head_sha)


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
            "number,title,state,isDraft,labels,url,headRefName,headRefOid,mergeable,statusCheckRollup,reviews,comments",
        ]
    )

    if result.returncode != 0:
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
                    "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
                }
            ],
        }
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
        _pr_payload_from_dict(pr, default_number=-1, expected_head_sha="")
        for pr in listed
        if isinstance(pr, dict)
    ]
    truncated = is_likely_truncated(len(listed), limit=limit)
    return {
        "schema": SCHEMA_VERSION,
        "repo": repo,
        "mode": "active",
        "truncated": truncated,
        "truncation_note": (
            "gh pr list may be capped: got "
            f"{len(listed)} rows at --limit {limit}; raise --limit or paginate"
            if truncated
            else ""
        ),
        "route_health_overview": _route_health_overview(prs),
        "prs": prs,
    }


def snapshot_prs(
    numbers: list[int],
    *,
    repo: str,
    expected_head_sha: str = "",
    include_review_threads: bool = False,
) -> dict[str, Any]:
    """Return a compact PR queue snapshot."""
    prs = [fetch_pr(number, repo=repo, expected_head_sha=expected_head_sha) for number in numbers]
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
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else json.dumps(payload))
    has_pr_errors = any(pr.get("status") == "error" for pr in payload["prs"])
    has_artifact_errors = payload.get("raw_review_comments_artifact_status") == "error"
    return 1 if has_pr_errors or has_artifact_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
