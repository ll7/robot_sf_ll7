#!/usr/bin/env python3
"""Emit compact goal-autopilot state snapshots.

The snapshot is route evidence for orientation and handoff.  It is not a substitute for fresh
local checks before publishing, labeling, merging, or making benchmark-facing claims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

FAILURE_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "error",
    "failure",
    "startup_failure",
    "timed_out",
}
PENDING_STATUSES = {"expected", "in_progress", "pending", "queued", "requested", "waiting"}


@dataclass(frozen=True)
class CommandResult:
    """Captured command result for compact provenance."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _run(command: list[str], *, timeout: int = 30) -> CommandResult:
    """Run a command and capture text output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=tuple(command),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=tuple(command),
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )


def _now_utc() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _command_source(result: CommandResult, *, name: str) -> dict[str, Any]:
    """Return compact command provenance."""
    return {
        "name": name,
        "command": list(result.command),
        "returncode": result.returncode,
    }


def _parse_json_result(result: CommandResult) -> tuple[Any | None, str | None]:
    """Parse a JSON command result."""
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"json_parse_error: {exc}"


def _git_text(command: list[str], *, name: str) -> tuple[str, dict[str, Any], str | None]:
    """Run a git command that should return one line of text."""
    result = _run(command)
    source = _command_source(result, name=name)
    if result.returncode != 0:
        return "", source, (result.stderr or result.stdout).strip() or f"{name} failed"
    return result.stdout.strip(), source, None


def _append_worktree_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Append a parsed worktree row with stable default fields."""
    if not row:
        return
    row.setdefault("branch", "")
    row.setdefault("head_sha", "")
    row.setdefault("bare", False)
    row.setdefault("detached", False)
    rows.append(row)


def _parse_worktree_porcelain(stdout: str) -> list[dict[str, Any]]:
    """Parse `git worktree list --porcelain` into compact worktree rows."""
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                _append_worktree_row(rows, current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            if current:
                _append_worktree_row(rows, current)
            current = {"path": value}
        elif key == "HEAD":
            current["head_sha"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"bare", "detached"}:
            current[key] = True
    if current:
        _append_worktree_row(rows, current)
    return rows


def git_snapshot(
    *, include_worktrees: bool, worktree_limit: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Return compact local git state."""
    sources: list[dict[str, Any]] = []
    errors: list[str] = []

    branch, source, error = _git_text(["git", "branch", "--show-current"], name="git.branch")
    sources.append(source)
    if error:
        errors.append(error)

    head_sha, source, error = _git_text(["git", "rev-parse", "HEAD"], name="git.head")
    sources.append(source)
    if error:
        errors.append(error)

    origin_main_sha, source, error = _git_text(
        ["git", "rev-parse", "--verify", "origin/main^{commit}"],
        name="git.origin_main",
    )
    sources.append(source)
    if error:
        errors.append(error)

    worktrees: list[dict[str, Any]] = []
    if include_worktrees:
        result = _run(["git", "worktree", "list", "--porcelain"])
        sources.append(_command_source(result, name="git.worktrees"))
        if result.returncode == 0:
            worktrees = _parse_worktree_porcelain(result.stdout)
            worktrees.sort(
                key=lambda row: (
                    row.get("branch") != branch,
                    row.get("head_sha") != head_sha,
                    row.get("path", ""),
                )
            )
        else:
            errors.append((result.stderr or result.stdout).strip() or "git worktree list failed")
    worktree_count = len(worktrees)
    worktree_limit = max(0, worktree_limit)
    visible_worktrees = worktrees[:worktree_limit] if worktree_limit else []

    return (
        {
            "branch": branch,
            "head_sha": head_sha,
            "origin_main_sha": origin_main_sha,
            "head_matches_origin_main": bool(
                head_sha and origin_main_sha and head_sha == origin_main_sha
            ),
            "worktree_count": worktree_count,
            "worktrees_truncated": worktree_count > len(visible_worktrees),
            "worktrees": visible_worktrees,
        },
        sources,
        errors,
    )


def claim_snapshot(
    issue_numbers: list[int], *, remote: str, origin_main_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return compact claim-ref state for issue numbers."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    for issue in issue_numbers:
        claim_ref = f"refs/heads/agent-claims/issue-{issue}"
        result = _run(["git", "ls-remote", "--heads", remote, claim_ref])
        sources.append(_command_source(result, name=f"claim.issue_{issue}"))
        if result.returncode != 0:
            error = (result.stderr or result.stdout).strip() or "claim lookup failed"
            rows.append(
                {
                    "issue": issue,
                    "ok": False,
                    "claimed": None,
                    "claim_ref": claim_ref.removeprefix("refs/heads/"),
                    "sha": None,
                    "stale_against_origin_main": None,
                    "error": error,
                }
            )
            errors.append(f"issue {issue}: {error}")
            continue
        sha = None
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == claim_ref:
                sha = parts[0]
                break
        rows.append(
            {
                "issue": issue,
                "ok": True,
                "claimed": sha is not None,
                "claim_ref": claim_ref.removeprefix("refs/heads/"),
                "sha": sha,
                "stale_against_origin_main": bool(
                    sha and origin_main_sha and sha != origin_main_sha
                ),
                "error": None,
            }
        )
    return rows, sources, errors


def issue_queue_snapshot(
    searches: list[str], *, limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return compact issue rows for one or more GitHub issue searches."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[int] = set()
    for search in searches:
        command = [
            "gh",
            "issue",
            "list",
            "--search",
            search,
            "--limit",
            str(limit),
            "--json",
            "number,title,state,labels,updatedAt,url",
        ]
        result = _run(command)
        sources.append(_command_source(result, name=f"issues.search:{search}"))
        data, error = _parse_json_result(result)
        if error:
            errors.append(f"issue search {search!r}: {error}")
            continue
        for issue in data if isinstance(data, list) else []:
            number = issue.get("number")
            if not isinstance(number, int) or number in seen:
                continue
            seen.add(number)
            labels = issue.get("labels", []) or []
            rows.append(
                {
                    "number": number,
                    "title": issue.get("title", ""),
                    "state": issue.get("state", ""),
                    "labels": sorted(
                        label.get("name", "")
                        for label in labels
                        if isinstance(label, dict) and label.get("name")
                    ),
                    "updated_at": issue.get("updatedAt", ""),
                    "url": issue.get("url", ""),
                }
            )
    return rows, sources, errors


def _rollup_conclusion(check: dict[str, Any]) -> str:
    """Return a normalized check conclusion."""
    return str(check.get("conclusion") or check.get("state") or "pending").lower()


def _rollup_status(check: dict[str, Any]) -> str:
    """Return a normalized check status."""
    status = check.get("status")
    if status:
        return str(status).lower()
    state = str(check.get("state") or "").lower()
    if state in {"success", "failure", "error"}:
        return "completed"
    return state or "completed"


def _check_name(check: dict[str, Any]) -> str:
    """Return a compact check name."""
    return str(check.get("name") or check.get("context") or "unknown")


def _checks_summary(rollup: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a PR statusCheckRollup payload."""
    valid_checks = [check for check in rollup if isinstance(check, dict)]
    conclusions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for check in valid_checks:
        conclusion = _rollup_conclusion(check)
        status = _rollup_status(check)
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
    failure_count = sum(conclusions.get(conclusion, 0) for conclusion in FAILURE_CONCLUSIONS)
    pending_count = sum(statuses.get(status, 0) for status in PENDING_STATUSES)
    if failure_count:
        overall = "failure"
    elif pending_count or not valid_checks:
        overall = "pending"
    else:
        overall = "success"
    return {
        "overall": overall,
        "total": len(valid_checks),
        "by_conclusion": conclusions,
        "by_status": statuses,
        "names": sorted({_check_name(check) for check in valid_checks}),
    }


def pr_snapshot(
    pr_numbers: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return compact PR headline state for explicit PR numbers."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    for pr in pr_numbers:
        command = [
            "gh",
            "pr",
            "view",
            str(pr),
            "--json",
            "number,title,state,mergeable,headRefName,headRefOid,statusCheckRollup,url",
        ]
        result = _run(command)
        sources.append(_command_source(result, name=f"pr.{pr}"))
        data, error = _parse_json_result(result)
        if error or not isinstance(data, dict):
            errors.append(f"pr {pr}: {error or 'not a JSON object'}")
            continue
        rollup = data.get("statusCheckRollup", []) or []
        rows.append(
            {
                "number": data.get("number", pr),
                "title": data.get("title", ""),
                "state": data.get("state", ""),
                "mergeable": data.get("mergeable", ""),
                "branch": data.get("headRefName", ""),
                "head_sha": data.get("headRefOid", ""),
                "checks": _checks_summary(rollup if isinstance(rollup, list) else []),
                "url": data.get("url", ""),
            }
        )
    return rows, sources, errors


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    """Build the full snapshot payload."""
    git, sources, errors = git_snapshot(
        include_worktrees=args.include_worktrees,
        worktree_limit=args.worktree_limit,
    )

    claims, claim_sources, claim_errors = claim_snapshot(
        args.claim_issue,
        remote=args.remote,
        origin_main_sha=git.get("origin_main_sha", ""),
    )
    sources.extend(claim_sources)
    errors.extend(claim_errors)

    issues, issue_sources, issue_errors = issue_queue_snapshot(args.issue_search, limit=args.limit)
    sources.extend(issue_sources)
    errors.extend(issue_errors)

    prs, pr_sources, pr_errors = pr_snapshot(args.pr)
    sources.extend(pr_sources)
    errors.extend(pr_errors)

    return {
        "schema": "autopilot_state_snapshot.v1",
        "ok": not errors,
        "generated_at_utc": _now_utc(),
        "freshness": {
            "route_evidence_only": True,
            "requires_fresh_check_before_publication": True,
            "branch": git.get("branch", ""),
            "head_sha": git.get("head_sha", ""),
            "origin_main_sha": git.get("origin_main_sha", ""),
        },
        "git": git,
        "claims": claims,
        "issues": issues,
        "prs": prs,
        "errors": errors,
        "sources": sources,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issue-search",
        action="append",
        default=[],
        metavar="QUERY",
        help="GitHub issue search query to summarize; may be repeated.",
    )
    parser.add_argument("--limit", type=int, default=20, help="maximum issues per search")
    parser.add_argument(
        "--pr", type=int, action="append", default=[], help="PR number to summarize"
    )
    parser.add_argument(
        "--claim-issue",
        type=int,
        action="append",
        default=[],
        help="issue number whose agent-claim ref should be summarized",
    )
    parser.add_argument("--remote", default="origin", help="git remote used for claim refs")
    parser.add_argument(
        "--include-worktrees",
        action="store_true",
        help="include `git worktree list --porcelain` summary",
    )
    parser.add_argument(
        "--worktree-limit",
        type=int,
        default=20,
        help="maximum worktree rows to include; total count and truncation are always reported",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    payload = build_snapshot(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
