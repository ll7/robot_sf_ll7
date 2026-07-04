#!/usr/bin/env python3
"""Audit open issues that may already be covered by merged title-linked PRs.

The command is intentionally read-only. It finds open issues, searches merged PRs
whose titles mention the issue number, and emits a machine-readable packet for a
human closure/residual-status pass. It never closes issues, comments, edits
labels, or mutates project queue state.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

DEFAULT_REPO = "ll7/robot_sf_ll7"
ISSUE_FIELDS = "number,title,url,state"
PR_FIELDS = "number,title,url,state,closedAt"
PARENT_MARKERS = re.compile(
    r"\b(parent|roadmap|epic|tracking|multi[- ]slice|umbrella)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LinkedPullRequest:
    """Merged pull request whose title links it to an issue number."""

    number: int
    title: str
    url: str
    merged_at: str

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable PR summary."""
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "merged_at": self.merged_at,
        }


@dataclass(frozen=True)
class ClosureAuditCandidate:
    """Open issue with at least one merged title-linked pull request."""

    number: int
    title: str
    url: str
    title_linked_prs: tuple[LinkedPullRequest, ...]
    classification: str
    recommended_action: str

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable issue audit row."""
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "classification": self.classification,
            "recommended_action": self.recommended_action,
            "title_linked_prs": [pr.to_payload() for pr in self.title_linked_prs],
        }


def _is_issue_url(raw_url: object) -> bool:
    """Return true only for canonical GitHub issue URLs."""
    if not isinstance(raw_url, str):
        return False
    path_parts = [part for part in urlsplit(raw_url).path.split("/") if part]
    if len(path_parts) != 4:
        return False
    _, _, resource, number = path_parts
    return resource == "issues" and number.isdigit()


def _is_pull_request_url(raw_url: object) -> bool:
    """Return true only for canonical GitHub pull request URLs."""
    if not isinstance(raw_url, str):
        return False
    path_parts = [part for part in urlsplit(raw_url).path.split("/") if part]
    if len(path_parts) != 4:
        return False
    _, _, resource, number = path_parts
    return resource == "pull" and number.isdigit()


def _issue_number(row: dict[str, object]) -> int | None:
    """Parse a GitHub search issue row number."""
    if not _is_issue_url(row.get("url")):
        return None
    if str(row.get("state", "")).lower() != "open":
        return None
    try:
        return int(row["number"])
    except (KeyError, TypeError, ValueError):
        return None


def _pull_request(row: dict[str, object], *, issue_number: int) -> LinkedPullRequest | None:
    """Parse a merged PR row and require a title link to the issue number."""
    if not _is_pull_request_url(row.get("url")):
        return None
    if str(row.get("state", "")).lower() != "merged":
        return None
    title = str(row.get("title", ""))
    if not _title_mentions_issue(title, issue_number):
        return None
    try:
        number = int(row["number"])
    except (KeyError, TypeError, ValueError):
        return None
    return LinkedPullRequest(
        number=number,
        title=title,
        url=str(row.get("url", "")),
        merged_at=str(row.get("mergedAt") or row.get("closedAt") or ""),
    )


def _title_mentions_issue(title: str, issue_number: int) -> bool:
    """Return true when a PR title explicitly mentions an issue number."""
    pattern = re.compile(rf"(?<!\d)(?:#)?{issue_number}(?!\d)")
    return pattern.search(title) is not None


def _classification_for_issue(title: str) -> tuple[str, str]:
    """Classify issue handling path without deciding acceptance completion."""
    if PARENT_MARKERS.search(title):
        return (
            "parent_or_roadmap",
            "update_status_ledger_with_merged_slices_and_remaining_work",
        )
    return (
        "closure_review_required",
        "read_acceptance_criteria_then_close_if_fully_covered_else_comment_residual_checklist",
    )


def collect_candidates(
    open_issue_rows: list[dict[str, object]],
    merged_pr_rows_by_issue: dict[int, list[dict[str, object]]],
) -> list[ClosureAuditCandidate]:
    """Collect open issues with merged title-linked PRs from pre-fetched rows."""
    candidates: list[ClosureAuditCandidate] = []
    seen: set[int] = set()
    for issue_row in open_issue_rows:
        number = _issue_number(issue_row)
        if number is None or number in seen:
            continue
        seen.add(number)
        linked_prs = tuple(
            pr
            for pr in (
                _pull_request(row, issue_number=number)
                for row in merged_pr_rows_by_issue.get(number, [])
            )
            if pr is not None
        )
        if not linked_prs:
            continue
        classification, recommended_action = _classification_for_issue(
            str(issue_row.get("title", ""))
        )
        candidates.append(
            ClosureAuditCandidate(
                number=number,
                title=str(issue_row.get("title", "")),
                url=str(issue_row.get("url", "")),
                title_linked_prs=tuple(sorted(linked_prs, key=lambda pr: pr.number)),
                classification=classification,
                recommended_action=recommended_action,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.number)


def build_open_issues_command(*, repo: str, limit: int) -> list[str]:
    """Build the read-only command that lists open issues."""
    return [
        "gh",
        "search",
        "issues",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        ISSUE_FIELDS,
        "--limit",
        str(limit),
    ]


def build_merged_prs_command(*, repo: str, issue_number: int, limit: int) -> list[str]:
    """Build the read-only command that finds merged PR titles for one issue."""
    return [
        "gh",
        "search",
        "prs",
        f"{issue_number} in:title",
        "--repo",
        repo,
        "--merged",
        "--json",
        PR_FIELDS,
        "--limit",
        str(limit),
    ]


def _run_search_command(command: list[str]) -> list[dict[str, object]]:
    """Run a GitHub CLI search command and parse JSON rows."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' was not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = detail or f"exit code {exc.returncode}"
        raise RuntimeError(f"GitHub CLI command failed ({' '.join(command)}): {message}") from exc
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from GitHub CLI ({' '.join(command)}): {exc.msg}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list from GitHub CLI ({' '.join(command)})")
    return [row for row in payload if isinstance(row, dict)]


def fetch_open_issue_rows(*, repo: str, limit: int) -> list[dict[str, object]]:
    """Fetch open issue rows with a read-only GitHub search."""
    return _run_search_command(build_open_issues_command(repo=repo, limit=limit))


def fetch_merged_pr_rows_by_issue(
    *, repo: str, issue_numbers: list[int], limit_per_issue: int
) -> dict[int, list[dict[str, object]]]:
    """Fetch merged PR search rows for each issue number."""
    rows_by_issue: dict[int, list[dict[str, object]]] = {}
    for issue_number in issue_numbers:
        rows_by_issue[issue_number] = _run_search_command(
            build_merged_prs_command(
                repo=repo,
                issue_number=issue_number,
                limit=limit_per_issue,
            )
        )
    return rows_by_issue


def build_report(*, repo: str, candidates: list[ClosureAuditCandidate]) -> dict[str, Any]:
    """Build a machine-readable closure-audit report."""
    parent_count = sum(candidate.classification == "parent_or_roadmap" for candidate in candidates)
    review_count = len(candidates) - parent_count
    return {
        "schema": "open_issue_closure_audit.v1",
        "ok": not candidates,
        "read_only": True,
        "issue_writes": False,
        "project_writes": False,
        "repo": repo,
        "candidate_count": len(candidates),
        "closure_review_count": review_count,
        "parent_or_roadmap_count": parent_count,
        "candidates": [candidate.to_payload() for candidate in candidates],
        "failure_summary": {
            "reason": "open_issues_with_merged_title_linked_prs",
            "count": len(candidates),
        }
        if candidates
        else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository OWNER/REPO.")
    parser.add_argument("--issue-limit", type=int, default=100, help="Open issues to scan.")
    parser.add_argument(
        "--pr-limit-per-issue",
        type=int,
        default=20,
        help="Merged PR rows to inspect per open issue.",
    )
    return parser


def _dump_json(payload: dict[str, Any]) -> None:
    """Print compact, deterministic JSON."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    """Run the read-only closure audit."""
    args = _build_parser().parse_args(argv)
    try:
        open_issue_rows = fetch_open_issue_rows(repo=args.repo, limit=args.issue_limit)
        issue_numbers = [
            number
            for number in (_issue_number(row) for row in open_issue_rows)
            if number is not None
        ]
        merged_pr_rows_by_issue = fetch_merged_pr_rows_by_issue(
            repo=args.repo,
            issue_numbers=issue_numbers,
            limit_per_issue=args.pr_limit_per_issue,
        )
        report = build_report(
            repo=args.repo,
            candidates=collect_candidates(open_issue_rows, merged_pr_rows_by_issue),
        )
    except (RuntimeError, ValueError) as exc:
        _dump_json(
            {
                "schema": "open_issue_closure_audit.v1",
                "ok": False,
                "read_only": True,
                "issue_writes": False,
                "project_writes": False,
                "repo": args.repo,
                "candidate_count": None,
                "candidates": [],
                "error": str(exc),
            }
        )
        return 2
    _dump_json(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
