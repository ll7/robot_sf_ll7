#!/usr/bin/env python3
"""Claim GitHub issues for cross-machine agent work with an atomic remote ref.

The claim is a stable remote branch named ``agent-claims/issue-<number>``. Creating that
ref through GitHub's create-ref API is atomic: if another machine already created it, the
API call fails and this agent should skip the issue.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REMOTE = "origin"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_SOURCE_REF = "origin/main"
CLAIM_PREFIX = "agent-claims"
ISSUE_RE = re.compile(r"^[1-9][0-9]*$")
ISSUE_COVERAGE_REFERENCE = re.compile(
    r"(?i)\b(?P<verb>refs?|references?|close(?:s|d)?|fix(?:es|ed)?|"
    r"resolve(?:s|d)?|implement(?:s|ed)?)\s*:?[ \t]*`?#(?P<issue>\d+)\b`?"
)
CLAIM_REF_RE = re.compile(r"^refs/heads/agent-claims/issue-(?P<issue>[1-9][0-9]*)$")
TERMINAL_RELEASE_REASONS = frozenset({"merged", "closed", "abandoned"})
RECONCILIATION_LIMIT = 100
PR_SNAPSHOT_LIMIT = 500
PR_REST_PAGE_SIZE = 100
GRAPHQL_REST_FALLBACK_MARKERS = (
    "rate limit",
    "projectcards",
    "unknown field",
    "doesn't exist on type",
    "is unsupported",
)
MANUAL_OVERRIDE_ERROR = "manual_override_required"


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result with the command that produced it."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def validate_issue_number(value: str) -> int:
    """Return a valid positive GitHub issue number."""
    if not ISSUE_RE.match(value):
        raise argparse.ArgumentTypeError("issue number must be a positive integer")
    return int(value)


def claim_ref(issue_number: int, *, prefix: str = CLAIM_PREFIX) -> str:
    """Return the full Git ref used as the cross-machine issue claim."""
    return f"refs/heads/{prefix}/issue-{issue_number}"


def short_claim_ref(issue_number: int, *, prefix: str = CLAIM_PREFIX) -> str:
    """Return the branch-style claim ref without ``refs/heads/``."""
    return f"{prefix}/issue-{issue_number}"


def _run(command: list[str]) -> CommandResult:
    """Run a command without invoking a shell."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return CommandResult(
        command=tuple(command),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def build_status_command(issue_number: int, *, remote: str) -> list[str]:
    """Build the command that checks whether a claim ref exists."""
    return ["git", "ls-remote", "--heads", remote, claim_ref(issue_number)]


def build_resolve_source_command(*, source_ref: str) -> list[str]:
    """Build the command that resolves the source commit for a new claim."""
    return ["git", "rev-parse", "--verify", f"{source_ref}^{{commit}}"]


def build_acquire_command(issue_number: int, *, repo: str, sha: str) -> list[str]:
    """Build the atomic GitHub ref creation command."""
    return [
        "gh",
        "api",
        "-X",
        "POST",
        f"repos/{repo}/git/refs",
        "-f",
        f"ref={claim_ref(issue_number)}",
        "-f",
        f"sha={sha}",
    ]


def build_release_command(issue_number: int, *, remote: str, expected_sha: str) -> list[str]:
    """Build a compare-and-delete command for the observed remote claim ref."""
    return [
        "git",
        "push",
        f"--force-with-lease={claim_ref(issue_number)}:{expected_sha}",
        remote,
        f":{claim_ref(issue_number)}",
    ]


def build_open_pr_command(*, repo: str) -> list[str]:
    """Build the read-only open-PR query used by terminal claim release."""
    return [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        "500",
        "--json",
        "number,body,title",
    ]


def build_claim_snapshot_command(*, remote: str) -> list[str]:
    """Build the bounded read-only query for all issue-claim refs."""
    return [
        "git",
        "ls-remote",
        "--heads",
        remote,
        f"refs/heads/{CLAIM_PREFIX}/issue-*",
    ]


def build_issue_state_command(issue_number: int, *, repo: str) -> list[str]:
    """Build the read-only REST query for one issue's lifecycle state."""
    return [
        "gh",
        "api",
        f"repos/{repo}/issues/{issue_number}",
        "--jq",
        "{number: .number, state: .state, title: .title, url: .html_url}",
    ]


def build_all_pr_command(*, repo: str) -> list[str]:
    """Build the bounded PR snapshot used by claim reconciliation."""
    return [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "all",
        "--limit",
        str(PR_SNAPSHOT_LIMIT),
        "--json",
        "number,body,title,state",
    ]


def build_open_pr_rest_command(*, repo: str, page: int = 1) -> list[str]:
    """Build a bounded REST fallback for the open-PR coverage snapshot."""
    if page <= 0:
        raise ValueError("REST PR snapshot page must be positive")
    return [
        "gh",
        "api",
        f"repos/{repo}/pulls?state=open&per_page={PR_REST_PAGE_SIZE}&page={page}",
        "--jq",
        "[.[] | {number,body,title}]",
    ]


def _validate_open_pr_row(row: Any, *, index: int) -> dict[str, Any]:
    """Validate one open-PR row before using it for claim-release safety."""
    if not isinstance(row, dict):
        return {"ok": False, "error": f"open PR row {index} is not an object"}
    raw_number = row.get("number")
    if not isinstance(raw_number, int) or isinstance(raw_number, bool) or raw_number <= 0:
        return {"ok": False, "error": f"open PR row {index} has an invalid number"}
    body = row.get("body")
    if not isinstance(body, str):
        return {"ok": False, "error": f"open PR row {index} has an invalid body"}
    title = row.get("title")
    if not isinstance(title, str):
        return {"ok": False, "error": f"open PR row {index} has an invalid title"}
    return {
        "ok": True,
        "number": raw_number,
        "body": body,
        "title": title,
    }


def build_all_pr_rest_command(*, repo: str, page: int = 1) -> list[str]:
    """Build a bounded REST fallback for the all-state PR coverage snapshot."""
    if page <= 0:
        raise ValueError("REST PR snapshot page must be positive")
    return [
        "gh",
        "api",
        f"repos/{repo}/pulls?state=all&per_page={PR_REST_PAGE_SIZE}&page={page}",
        "--jq",
        (
            '[.[] | {number,body,title,state:(if .merged_at then "MERGED" '
            'elif .state == "open" then "OPEN" else "CLOSED" end)}]'
        ),
    ]


def _decode_pr_pages(result: CommandResult, *, empty_error: str) -> tuple[list[Any], str | None]:
    """Decode one or more JSON-array pages emitted by a REST snapshot command."""
    if result.returncode != 0:
        return [], (result.stderr or result.stdout).strip() or "PR snapshot failed"
    raw_payload = result.stdout.strip()
    if not raw_payload:
        return [], empty_error

    decoder = json.JSONDecoder()
    pages: list[Any] = []
    cursor = 0
    while cursor < len(raw_payload):
        while cursor < len(raw_payload) and raw_payload[cursor].isspace():
            cursor += 1
        if cursor >= len(raw_payload):
            break
        try:
            page, next_cursor = decoder.raw_decode(raw_payload, cursor)
        except json.JSONDecodeError as exc:
            return [], str(exc)
        if not isinstance(page, list):
            return [], "PR snapshot page is not a list"
        pages.extend(page)
        cursor = next_cursor
    return pages, None


def _is_graphql_rest_fallback_error(result: CommandResult) -> bool:
    """Return whether a known GraphQL quota/schema error permits REST fallback."""
    if result.returncode == 0:
        return False
    error = f"{result.stderr}\n{result.stdout}".lower()
    return "graphql" in error and any(marker in error for marker in GRAPHQL_REST_FALLBACK_MARKERS)


def _run_bounded_pr_rest_snapshot(
    *, repo: str, build_command: Callable[..., list[str]], empty_error: str
) -> CommandResult:
    """Fetch at most the configured PR page cap through the REST API."""
    pages: list[Any] = []
    last_command: list[str] = []
    max_pages = (PR_SNAPSHOT_LIMIT + PR_REST_PAGE_SIZE - 1) // PR_REST_PAGE_SIZE
    for page in range(1, max_pages + 1):
        command = build_command(repo=repo, page=page)
        result = _run(command)
        last_command = command
        page_payload, decode_error = _decode_pr_pages(result, empty_error=empty_error)
        if decode_error is not None:
            if result.returncode != 0:
                return result
            return CommandResult(
                command=tuple(command),
                returncode=1,
                stdout="",
                stderr=decode_error,
            )
        pages.extend(page_payload)
        if len(page_payload) < PR_REST_PAGE_SIZE or len(pages) >= PR_SNAPSHOT_LIMIT:
            break

    return CommandResult(
        command=tuple(last_command),
        returncode=0,
        stdout=json.dumps(pages),
        stderr="",
    )


def _open_prs_covering_issue(result: CommandResult, *, issue_number: int) -> dict[str, Any]:
    """Parse one authoritative open-PR response for explicit issue coverage."""
    payload, decode_error = _decode_pr_pages(result, empty_error="open PR response is empty")
    if decode_error is not None:
        return {
            "ok": False,
            "covering_prs": [],
            "truncated": False,
            "error": decode_error,
        }
    target = int(issue_number)
    covering: set[int] = set()
    for index, row in enumerate(payload):
        validated = _validate_open_pr_row(row, index=index)
        if not validated["ok"]:
            return {
                "ok": False,
                "covering_prs": [],
                "truncated": False,
                "error": validated["error"],
            }
        number = validated["number"]
        text = f"{validated['body']} {validated['title']}"
        if any(
            int(match.group("issue")) == target for match in ISSUE_COVERAGE_REFERENCE.finditer(text)
        ):
            covering.add(number)
    return {
        "ok": True,
        "covering_prs": sorted(covering),
        "truncated": len(payload) >= PR_SNAPSHOT_LIMIT,
        "error": None,
    }


def _open_prs_covering_issue_with_fallback(*, repo: str, issue_number: int) -> dict[str, Any]:
    """Read open PR coverage, falling back to REST only for known GraphQL failures."""
    primary = _run(build_open_pr_command(repo=repo))
    coverage = _open_prs_covering_issue(primary, issue_number=issue_number)
    if coverage["ok"] or not _is_graphql_rest_fallback_error(primary):
        coverage["source"] = "graphql"
        return coverage

    fallback = _open_prs_covering_issue(
        _run_bounded_pr_rest_snapshot(
            repo=repo,
            build_command=build_open_pr_rest_command,
            empty_error="open PR response is empty",
        ),
        issue_number=issue_number,
    )
    fallback["source"] = "rest_fallback"
    fallback["fallback_reason"] = (primary.stderr or primary.stdout).strip()
    return fallback


def _parse_claim_snapshot(
    result: CommandResult, *, issue_number: int | None, limit: int
) -> dict[str, Any]:
    """Parse a bounded claim-ref listing without treating malformed refs as absent."""
    if result.returncode != 0:
        return {
            "ok": False,
            "claims": [],
            "truncated": False,
            "error": (result.stderr or result.stdout).strip() or "claim ref snapshot failed",
        }

    claims: list[dict[str, Any]] = []
    malformed: list[str] = []
    prefix = f"refs/heads/{CLAIM_PREFIX}/issue-"
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) < 2 or not parts[1].startswith(prefix):
            malformed.append(raw_line)
            continue
        match = CLAIM_REF_RE.fullmatch(parts[1])
        if match is None:
            malformed.append(parts[1])
            continue
        candidate_issue = int(match.group("issue"))
        if issue_number is not None and candidate_issue != issue_number:
            continue
        claims.append(
            {
                "issue": candidate_issue,
                "claim_ref": parts[1].removeprefix("refs/heads/"),
                "sha": parts[0],
            }
        )

    claims.sort(key=lambda row: (row["issue"], row["claim_ref"]))
    truncated = len(claims) > limit
    if truncated:
        claims = claims[:limit]
    if malformed:
        return {
            "ok": False,
            "claims": claims,
            "truncated": truncated,
            "error": f"malformed claim ref row(s): {', '.join(malformed[:3])}",
        }
    return {
        "ok": True,
        "claims": claims,
        "truncated": truncated,
        "error": None,
    }


def _issue_state_from_result(result: CommandResult, *, issue_number: int) -> dict[str, Any]:
    """Parse one issue state and reject missing or contradictory lifecycle fields."""
    if result.returncode != 0:
        return {
            "ok": False,
            "state": None,
            "title": "",
            "url": "",
            "error": (result.stderr or result.stdout).strip() or "issue state lookup failed",
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "state": None, "title": "", "url": "", "error": str(exc)}
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "state": None,
            "title": "",
            "url": "",
            "error": "issue state response is not an object",
        }
    if payload.get("number") != issue_number:
        return {
            "ok": False,
            "state": None,
            "title": "",
            "url": "",
            "error": "issue state response number does not match claim",
        }
    state = payload.get("state")
    title = payload.get("title")
    url = payload.get("url")
    if not isinstance(state, str) or state.upper() not in {"OPEN", "CLOSED"}:
        return {
            "ok": False,
            "state": None,
            "title": title if isinstance(title, str) else "",
            "url": url if isinstance(url, str) else "",
            "error": "issue state is missing or unknown",
        }
    if not isinstance(title, str) or not isinstance(url, str):
        return {
            "ok": False,
            "state": None,
            "title": "",
            "url": "",
            "error": "issue state response is missing title or URL",
        }
    return {"ok": True, "state": state.upper(), "title": title, "url": url, "error": None}


def _validate_pr_snapshot_row(row: Any, *, index: int) -> dict[str, Any]:
    """Validate a PR row before using it as claim-coverage evidence."""
    if not isinstance(row, dict):
        return {"ok": False, "error": f"PR row {index} is not an object"}
    number = row.get("number")
    body = row.get("body")
    title = row.get("title")
    state = row.get("state")
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        return {"ok": False, "error": f"PR row {index} has an invalid number"}
    if not isinstance(body, str) or not isinstance(title, str):
        return {"ok": False, "error": f"PR row {index} has an invalid body or title"}
    if not isinstance(state, str) or state.upper() not in {"OPEN", "CLOSED", "MERGED"}:
        return {"ok": False, "error": f"PR row {index} has an invalid state"}
    return {"ok": True, "number": number, "body": body, "title": title, "state": state.upper()}


def _all_prs_covering_issue(result: CommandResult, *, issue_number: int) -> dict[str, Any]:
    """Return open, merged, and closed PRs that explicitly reference an issue.

    ``terminal_prs`` is retained as the union of merged and closed PRs for
    backward compatibility; ``merged_prs`` and ``closed_prs`` split it so a
    release decision can distinguish "delivered by a merged PR" from "only
    closed-unmerged coverage remains".
    """
    payload, decode_error = _decode_pr_pages(result, empty_error="PR snapshot is empty")
    if decode_error is not None:
        return {
            "ok": False,
            "open_prs": [],
            "merged_prs": [],
            "closed_prs": [],
            "terminal_prs": [],
            "truncated": False,
            "error": decode_error,
        }

    open_prs: list[int] = []
    merged_prs: list[int] = []
    closed_prs: list[int] = []
    for index, row in enumerate(payload):
        validated = _validate_pr_snapshot_row(row, index=index)
        if not validated["ok"]:
            return {
                "ok": False,
                "open_prs": [],
                "merged_prs": [],
                "closed_prs": [],
                "terminal_prs": [],
                "truncated": False,
                "error": validated["error"],
            }
        text = f"{validated['body']} {validated['title']}"
        if not any(
            int(match.group("issue")) == issue_number
            for match in ISSUE_COVERAGE_REFERENCE.finditer(text)
        ):
            continue
        if validated["state"] == "OPEN":
            open_prs.append(validated["number"])
        elif validated["state"] == "MERGED":
            merged_prs.append(validated["number"])
        else:
            closed_prs.append(validated["number"])
    return {
        "ok": True,
        "open_prs": sorted(set(open_prs)),
        "merged_prs": sorted(set(merged_prs)),
        "closed_prs": sorted(set(closed_prs)),
        "terminal_prs": sorted(set(merged_prs) | set(closed_prs)),
        "truncated": len(payload) >= PR_SNAPSHOT_LIMIT,
        "error": None,
    }


def _all_prs_covering_issue_with_fallback(*, repo: str, issue_number: int) -> dict[str, Any]:
    """Read all PR coverage, using REST only for known GraphQL failures."""
    primary = _run(build_all_pr_command(repo=repo))
    coverage = _all_prs_covering_issue(primary, issue_number=issue_number)
    if coverage["ok"] or not _is_graphql_rest_fallback_error(primary):
        coverage["source"] = "graphql"
        return coverage

    fallback = _all_prs_covering_issue(
        _run_bounded_pr_rest_snapshot(
            repo=repo,
            build_command=build_all_pr_rest_command,
            empty_error="PR snapshot is empty",
        ),
        issue_number=issue_number,
    )
    fallback["source"] = "rest_fallback"
    fallback["fallback_reason"] = (primary.stderr or primary.stdout).strip()
    return fallback


def _classify_reconciliation_row(
    claim: dict[str, Any], *, issue: dict[str, Any], prs: dict[str, Any]
) -> dict[str, Any]:
    """Classify one claim conservatively for a report or an explicit cleanup pass."""
    row = {
        **claim,
        "issue_state": issue.get("state"),
        "issue_title": issue.get("title", ""),
        "issue_url": issue.get("url", ""),
        "covering_prs": [],
        "terminal_covering_prs": [],
        "open_coverage_source": prs.get("open_source"),
        "terminal_coverage_source": prs.get("terminal_source"),
        "open_fallback_reason": prs.get("open_fallback_reason"),
        "terminal_fallback_reason": prs.get("terminal_fallback_reason"),
        "safe_to_release": False,
        "classification": "state_unknown",
        "reason": "issue state unavailable or contradictory; retain the claim",
    }
    if not issue.get("ok"):
        row["reason"] = f"{issue.get('error', 'issue state unavailable')}; retain the claim"
        return row
    if not prs.get("open_ok", prs.get("ok", False)) or prs.get("open_truncated"):
        row["classification"] = "coverage_unknown"
        row["reason"] = "covering-PR snapshot unavailable or truncated; retain the claim"
        return row

    open_prs = list(prs.get("open_prs", []))
    terminal_prs = list(prs.get("terminal_prs", []))
    merged_prs = list(prs.get("merged_prs", []))
    row["covering_prs"] = open_prs
    row["terminal_covering_prs"] = terminal_prs
    row["merged_covering_prs"] = merged_prs
    if open_prs and merged_prs:
        row["classification"] = "delivered_open_competitor"
        row["safe_to_release"] = True
        row["reason"] = (
            "a covering PR is MERGED (delivery proven) while a competing open "
            "covering PR remains; the open competitor is the #7474/#7493 "
            "coordination class, not a reason to retain the claim"
        )
    elif open_prs:
        row["classification"] = "active_open_pr"
        row["reason"] = "an open covering PR exists; retain the claim"
    elif issue["state"] == "CLOSED":
        row["classification"] = "stale_closed_issue"
        row["safe_to_release"] = True
        row["reason"] = "issue is closed and no open covering PR exists"
    elif not prs.get("terminal_ok", prs.get("ok", False)) or prs.get("terminal_truncated"):
        row["classification"] = "coverage_unknown"
        row["reason"] = "terminal covering-PR snapshot unavailable or truncated; retain the claim"
    elif terminal_prs:
        row["classification"] = "stale_terminal_coverage"
        row["safe_to_release"] = True
        row["reason"] = "only terminal covering PRs remain and no open covering PR exists"
    else:
        row["classification"] = "active_issue_no_open_pr"
        row["reason"] = "issue remains open without terminal coverage; retain the claim"
    return row


def _release_reconciled_claim(
    row: dict[str, Any], *, remote: str, repo: str, reason: str
) -> dict[str, Any]:
    """Re-read state and coverage, then compare-and-delete one stale claim."""
    issue_number = row["issue"]
    status = status_issue(issue_number, remote=remote)
    if not status.get("ok") or not status.get("claimed"):
        return {
            "ok": False,
            "issue": issue_number,
            "error": "claim state changed or became unavailable; retain the claim",
        }
    if status.get("sha") != row.get("sha"):
        return {
            "ok": False,
            "issue": issue_number,
            "error": "claim SHA changed during reconciliation; retain the claim",
        }

    issue = _issue_state_from_result(
        _run(build_issue_state_command(issue_number, repo=repo)), issue_number=issue_number
    )
    open_prs = _open_prs_covering_issue_with_fallback(repo=repo, issue_number=issue_number)
    terminal_prs = _all_prs_covering_issue_with_fallback(repo=repo, issue_number=issue_number)
    prs = {
        "open_ok": open_prs.get("ok", False),
        "open_truncated": open_prs.get("truncated", False),
        "open_prs": open_prs.get("covering_prs", []),
        "open_source": open_prs.get("source"),
        "open_fallback_reason": open_prs.get("fallback_reason"),
        "terminal_ok": terminal_prs.get("ok", False),
        "terminal_truncated": terminal_prs.get("truncated", False),
        "terminal_prs": terminal_prs.get("terminal_prs", []),
        "merged_prs": terminal_prs.get("merged_prs", []),
        "terminal_source": terminal_prs.get("source"),
        "terminal_fallback_reason": terminal_prs.get("fallback_reason"),
    }
    fresh = _classify_reconciliation_row(row, issue=issue, prs=prs)
    if not fresh["safe_to_release"]:
        return {
            "ok": False,
            "issue": issue_number,
            "error": fresh["reason"],
        }

    result = _run(build_release_command(issue_number, remote=remote, expected_sha=status["sha"]))
    return {
        "ok": result.returncode == 0,
        "issue": issue_number,
        "claim_ref": row["claim_ref"],
        "expected_sha": status["sha"],
        "reason": reason,
        "command": list(result.command),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "error": None if result.returncode == 0 else "compare-and-delete failed; inspect the ref",
    }


def reconcile_claims(  # noqa: C901 - bounded CLI orchestration with fail-closed branches.
    *,
    remote: str,
    repo: str,
    issue_number: int | None = None,
    limit: int = RECONCILIATION_LIMIT,
    release_stale: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Report stale claim candidates and optionally release only revalidated rows."""
    if limit <= 0:
        return {
            "schema": "issue_claim_reconciliation.v1",
            "action": "reconcile",
            "ok": False,
            "read_only": not release_stale,
            "claims": [],
            "candidate_count": 0,
            "errors": ["limit must be positive"],
        }
    if release_stale and reason not in TERMINAL_RELEASE_REASONS:
        return {
            "schema": "issue_claim_reconciliation.v1",
            "action": "reconcile",
            "ok": False,
            "read_only": False,
            "claims": [],
            "candidate_count": 0,
            "errors": ["--release-stale requires an explicit terminal --reason"],
        }

    snapshot = _parse_claim_snapshot(
        _run(build_claim_snapshot_command(remote=remote)),
        issue_number=issue_number,
        limit=limit,
    )
    errors = [snapshot["error"]] if snapshot.get("error") else []
    if snapshot.get("truncated"):
        errors.append("claim ref snapshot truncated; retain claims outside the bounded report")
    claims = snapshot.get("claims", [])
    rows: list[dict[str, Any]] = []
    if snapshot.get("ok") and claims:
        open_pr_snapshot = _run(build_open_pr_command(repo=repo))
        all_pr_snapshot = _run(build_all_pr_command(repo=repo))
        open_pr_source = "graphql"
        terminal_pr_source = "graphql"
        open_pr_fallback_reason = None
        terminal_pr_fallback_reason = None
        if _is_graphql_rest_fallback_error(open_pr_snapshot):
            open_pr_fallback_reason = (open_pr_snapshot.stderr or open_pr_snapshot.stdout).strip()
            open_pr_snapshot = _run_bounded_pr_rest_snapshot(
                repo=repo,
                build_command=build_open_pr_rest_command,
                empty_error="open PR response is empty",
            )
            open_pr_source = "rest_fallback"
        if _is_graphql_rest_fallback_error(all_pr_snapshot):
            terminal_pr_fallback_reason = (all_pr_snapshot.stderr or all_pr_snapshot.stdout).strip()
            all_pr_snapshot = _run_bounded_pr_rest_snapshot(
                repo=repo,
                build_command=build_all_pr_rest_command,
                empty_error="PR snapshot is empty",
            )
            terminal_pr_source = "rest_fallback"

        for claim in claims:
            issue_result = _run(build_issue_state_command(claim["issue"], repo=repo))
            issue = _issue_state_from_result(issue_result, issue_number=claim["issue"])
            open_prs = _open_prs_covering_issue(open_pr_snapshot, issue_number=claim["issue"])
            terminal_prs = _all_prs_covering_issue(all_pr_snapshot, issue_number=claim["issue"])
            prs = {
                "open_ok": open_prs.get("ok", False),
                "open_truncated": open_prs.get("truncated", False),
                "open_prs": open_prs.get("covering_prs", []),
                "open_source": open_pr_source,
                "open_fallback_reason": open_pr_fallback_reason,
                "terminal_ok": terminal_prs.get("ok", False),
                "terminal_truncated": terminal_prs.get("truncated", False),
                "terminal_prs": terminal_prs.get("terminal_prs", []),
                "terminal_source": terminal_pr_source,
                "terminal_fallback_reason": terminal_pr_fallback_reason,
            }
            row = _classify_reconciliation_row(claim, issue=issue, prs=prs)
            rows.append(row)
            if not issue.get("ok"):
                errors.append(f"issue {claim['issue']}: {issue.get('error', 'state unavailable')}")
            if row["classification"] == "coverage_unknown":
                errors.append(f"issue {claim['issue']}: {row['reason']}")
    elif not snapshot.get("ok"):
        rows = [
            {
                **claim,
                "classification": "claim_snapshot_unknown",
                "safe_to_release": False,
                "reason": "claim snapshot unavailable or malformed; retain the claim",
            }
            for claim in claims
        ]

    releases: list[dict[str, Any]] = []
    if release_stale and not errors:
        for row in rows:
            if row.get("safe_to_release"):
                releases.append(
                    _release_reconciled_claim(row, remote=remote, repo=repo, reason=reason or "")
                )

    return {
        "schema": "issue_claim_reconciliation.v1",
        "action": "reconcile",
        "ok": not errors and all(release.get("ok", False) for release in releases),
        "read_only": not release_stale,
        "remote": remote,
        "repo": repo,
        "limit": limit,
        "truncated": bool(snapshot.get("truncated")),
        "claims": rows,
        "candidate_count": sum(1 for row in rows if row.get("safe_to_release")),
        "releases": releases,
        "errors": errors,
    }


def _status_from_ls_remote(
    result: CommandResult, *, issue_number: int, remote: str
) -> dict[str, Any]:
    """Convert ``git ls-remote`` output into the command payload."""
    if result.returncode != 0:
        return {
            "schema": "issue_claim.v1",
            "action": "status",
            "ok": False,
            "claimed": None,
            "issue": issue_number,
            "remote": remote,
            "claim_ref": short_claim_ref(issue_number),
            "error": (result.stderr or result.stdout).strip(),
            "command": list(result.command),
        }

    target_ref = claim_ref(issue_number)
    sha = None
    for line in (result.stdout or "").strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == target_ref:
            sha = parts[0]
            break

    if sha is None:
        return {
            "schema": "issue_claim.v1",
            "action": "status",
            "ok": True,
            "claimed": False,
            "issue": issue_number,
            "remote": remote,
            "claim_ref": short_claim_ref(issue_number),
            "sha": None,
            "command": list(result.command),
        }

    return {
        "schema": "issue_claim.v1",
        "action": "status",
        "ok": True,
        "claimed": True,
        "issue": issue_number,
        "remote": remote,
        "claim_ref": short_claim_ref(issue_number),
        "sha": sha,
        "command": list(result.command),
    }


def status_issue(issue_number: int, *, remote: str) -> dict[str, Any]:
    """Return whether a remote claim currently exists."""
    return _status_from_ls_remote(
        _run(build_status_command(issue_number, remote=remote)),
        issue_number=issue_number,
        remote=remote,
    )


def acquire_issue(issue_number: int, *, repo: str, remote: str, source_ref: str) -> dict[str, Any]:
    """Try to create the claim ref and return a machine-readable result."""
    source_result = _run(build_resolve_source_command(source_ref=source_ref))
    if source_result.returncode != 0:
        return {
            "schema": "issue_claim.v1",
            "action": "acquire",
            "ok": False,
            "claimed": False,
            "issue": issue_number,
            "repo": repo,
            "remote": remote,
            "source_ref": source_ref,
            "claim_ref": short_claim_ref(issue_number),
            "command": list(source_result.command),
            "stdout": source_result.stdout.strip(),
            "stderr": source_result.stderr.strip(),
            "error": "source_ref_resolution_failed",
        }

    sha = source_result.stdout.strip()
    result = _run(build_acquire_command(issue_number, repo=repo, sha=sha))
    ok = result.returncode == 0
    return {
        "schema": "issue_claim.v1",
        "action": "acquire",
        "ok": ok,
        "claimed": ok,
        "issue": issue_number,
        "repo": repo,
        "remote": remote,
        "source_ref": source_ref,
        "sha": sha,
        "claim_ref": short_claim_ref(issue_number),
        "command": list(result.command),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "error": None
        if ok
        else (
            "claim_ref_already_exists_or_create_ref_failed; run status to inspect the current owner "
            "signal and skip this issue unless the claim is confirmed stale"
        ),
    }


def release_issue(
    issue_number: int,
    *,
    remote: str,
    repo: str = DEFAULT_REPO,
    reason: str | None = None,
) -> dict[str, Any]:
    """Release a claim only after a terminal lifecycle reason is supplied."""
    status = status_issue(issue_number, remote=remote)
    if status["ok"] and not status["claimed"]:
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": True,
            "claimed": False,
            "issue": issue_number,
            "remote": remote,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "Ref does not exist, nothing to release.",
            "stderr": "",
            "error": None,
            "release_class": "terminal",
            "reason": reason,
        }

    if not status["ok"]:
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": False,
            "claimed": None,
            "issue": issue_number,
            "remote": remote,
            "repo": repo,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "",
            "stderr": status.get("error", ""),
            "error": "claim_status_unavailable; do not release an unknown claim",
            "release_class": None,
            "reason": reason,
        }

    if reason not in TERMINAL_RELEASE_REASONS:
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": False,
            "claimed": True,
            "issue": issue_number,
            "remote": remote,
            "repo": repo,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "",
            "stderr": "",
            "error": "terminal_release_reason_required",
            "release_class": None,
            "reason": reason,
        }

    observed_sha = status.get("sha")
    if not isinstance(observed_sha, str) or not observed_sha:
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": False,
            "claimed": None,
            "issue": issue_number,
            "remote": remote,
            "repo": repo,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "",
            "stderr": "",
            "error": "claim_status_missing_sha; do not release an unknown claim",
            "release_class": None,
            "reason": reason,
        }

    coverage = _open_prs_covering_issue_with_fallback(repo=repo, issue_number=issue_number)
    coverage_provenance = {
        "coverage_source": coverage.get("source"),
        "coverage_fallback_reason": coverage.get("fallback_reason"),
    }
    if not coverage["ok"]:
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": False,
            "claimed": True,
            "issue": issue_number,
            "remote": remote,
            "repo": repo,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "",
            "stderr": coverage["error"],
            "error": "open_pr_snapshot_unavailable; retain the claim",
            "release_class": None,
            "reason": reason,
            "covering_prs": [],
            **coverage_provenance,
        }
    if coverage.get("truncated"):
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": False,
            "claimed": True,
            "issue": issue_number,
            "remote": remote,
            "repo": repo,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "",
            "stderr": "open PR snapshot reached its limit",
            "error": "open_pr_snapshot_truncated; retain the claim",
            "release_class": None,
            "reason": reason,
            "covering_prs": coverage["covering_prs"],
            **coverage_provenance,
        }
    if coverage["covering_prs"]:
        # An open covering PR normally blocks release. But if another covering PR
        # already MERGED, the issue is verifiably delivered and the open PR is a
        # competing/superseded lane (issue #7474/#7493 coordination class), so the
        # claim may be released with the merged PR recorded as delivery evidence.
        delivered = _all_prs_covering_issue_with_fallback(repo=repo, issue_number=issue_number)
        merged_covering = sorted(
            {
                int(number)
                for number in delivered.get("merged_prs", [])
                if isinstance(number, int) and number > 0
            }
        )
        if delivered.get("ok") and not delivered.get("truncated") and merged_covering:
            result = _run(
                build_release_command(issue_number, remote=remote, expected_sha=observed_sha)
            )
            ok = result.returncode == 0
            return {
                "schema": "issue_claim.v1",
                "action": "release",
                "ok": ok,
                "claimed": False if ok else None,
                "issue": issue_number,
                "remote": remote,
                "repo": repo,
                "claim_ref": short_claim_ref(issue_number),
                "command": list(result.command),
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "error": None
                if ok
                else "claim_ref_release_failed; inspect remote branch state before retrying",
                "release_class": "terminal" if ok else None,
                "reason": reason,
                "covering_prs": coverage["covering_prs"],
                "merged_covering_prs": merged_covering,
                "release_override": "delivered_open_competitor",
                "delivered_by": merged_covering,
                **coverage_provenance,
                **{
                    "delivery_source": delivered.get("source"),
                    "delivery_fallback_reason": delivered.get("fallback_reason"),
                },
            }
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": False,
            "claimed": True,
            "issue": issue_number,
            "remote": remote,
            "repo": repo,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "",
            "stderr": "",
            "error": "open_covering_pr_exists; retain the claim until the PR reaches a terminal state",
            "release_class": None,
            "reason": reason,
            "covering_prs": coverage["covering_prs"],
            "merged_covering_prs": [],
            **coverage_provenance,
        }

    result = _run(build_release_command(issue_number, remote=remote, expected_sha=observed_sha))
    ok = result.returncode == 0
    return {
        "schema": "issue_claim.v1",
        "action": "release",
        "ok": ok,
        "claimed": False if ok else None,
        "issue": issue_number,
        "remote": remote,
        "claim_ref": short_claim_ref(issue_number),
        "command": list(result.command),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "error": None
        if ok
        else "claim_ref_release_failed; inspect remote branch state before retrying",
        "release_class": "terminal" if ok else None,
        "reason": reason,
        "covering_prs": [],
        **coverage_provenance,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "status", "release", "reconcile"))
    parser.add_argument("issue", type=validate_issue_number, nargs="?")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository as OWNER/REPO.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote to use for the claim.")
    parser.add_argument(
        "--source-ref",
        default=DEFAULT_SOURCE_REF,
        help="Local ref to push when acquiring the claim. Defaults to origin/main.",
    )
    parser.add_argument(
        "--reason",
        choices=sorted(TERMINAL_RELEASE_REASONS),
        help="Terminal lifecycle reason required when releasing a claimed ref.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=RECONCILIATION_LIMIT,
        help="Maximum claim refs to inspect during reconciliation.",
    )
    parser.add_argument(
        "--release-stale",
        action="store_true",
        help="Revalidate and compare-and-delete stale candidates; never delete blindly.",
    )
    parser.add_argument(
        "--manual-override",
        action="store_true",
        help="Explicitly enter the maintainer-only incident/forensic acquire lane.",
    )
    parser.add_argument(
        "--override-actor",
        help="Named maintainer actor for a manual acquire override.",
    )
    parser.add_argument(
        "--override-reason",
        help="Bounded incident or forensic reason for a manual acquire override.",
    )
    parser.add_argument(
        "--no-scientific-claim",
        action="store_true",
        help="Acknowledge that the manual acquire makes no scientific or benchmark claim.",
    )
    return parser


def _dump_json(payload: dict[str, Any]) -> None:
    """Print stable JSON to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _manual_override_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Return a fail-closed payload for an unguarded direct acquire attempt."""
    missing: list[str] = []
    if not args.manual_override:
        missing.append("--manual-override")
    if not isinstance(args.override_actor, str) or not args.override_actor.strip():
        missing.append("--override-actor")
    if not isinstance(args.override_reason, str) or not args.override_reason.strip():
        missing.append("--override-reason")
    if not args.no_scientific_claim:
        missing.append("--no-scientific-claim")
    return {
        "schema": "issue_claim.v1",
        "action": "acquire",
        "ok": False,
        "claimed": False,
        "issue": args.issue,
        "repo": args.repo,
        "remote": args.remote,
        "source_ref": args.source_ref,
        "claim_ref": short_claim_ref(args.issue) if args.issue is not None else None,
        "command": [],
        "error": MANUAL_OVERRIDE_ERROR,
        "missing_override_fields": missing,
        "write_attempted": False,
        "authority_boundary": (
            "Direct low-level acquire is reserved for maintainer incident recovery or forensic "
            "operation; ordinary autonomous work must use goal_issue_admission.py."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    if args.action == "reconcile":
        payload = reconcile_claims(
            remote=args.remote,
            repo=args.repo,
            issue_number=args.issue,
            limit=args.limit,
            release_stale=args.release_stale,
            reason=args.reason,
        )
    elif args.issue is None:
        _build_parser().error(f"{args.action} requires an issue number")
    elif args.action == "status":
        payload = status_issue(args.issue, remote=args.remote)
    elif args.action == "acquire":
        override_payload = _manual_override_payload(args)
        if override_payload["missing_override_fields"]:
            payload = override_payload
        else:
            payload = acquire_issue(
                args.issue,
                repo=args.repo,
                remote=args.remote,
                source_ref=args.source_ref,
            )
            payload.update(
                {
                    "manual_override": True,
                    "override_actor": args.override_actor.strip(),
                    "override_reason": args.override_reason.strip(),
                    "no_scientific_claim": True,
                }
            )
    else:
        payload = release_issue(
            args.issue,
            remote=args.remote,
            repo=args.repo,
            reason=args.reason,
        )

    _dump_json(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
