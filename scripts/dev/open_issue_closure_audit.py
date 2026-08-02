#!/usr/bin/env python3
"""Audit open issues that may already be covered by merged title-linked PRs.

The command is intentionally read-only. It collects the repository's open issues
and its merged pull requests through bounded paginated GitHub REST reads, builds
a title-to-issue index locally, and emits a machine-readable packet of open
issues that have at least one merged title-linked pull request. It never closes
issues, comments, edits labels, or mutates project queue state.

REST-first inventory (issue #6610)
----------------------------------
Earlier versions discovered the open-issue inventory and per-issue merged PRs
through GitHub's separately-rate-limited search endpoint, which made the audit
abort whenever that quota was exhausted even though the core REST quota was
healthy. The audit now:

- reads open issues via ``gh api repos/<repo>/issues?state=open&...`` with a
  bounded page budget (``--max-issue-pages``);
- reads closed pull requests via a single bounded ``gh api
  repos/<repo>/pulls?state=closed&...`` pass (``--max-pr-pages``), filters to
  merged pull requests locally, and builds the title-to-issue index in-process
  instead of issuing one search request per open issue.

When either REST inventory cannot be completed within its page budget, the
report carries deterministic truncation/partial-inventory metadata and the
process exits non-zero so a partial inventory is never mistaken for an
authoritative "no candidates" result.
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

from scripts.dev._gh_pagination import is_likely_truncated

DEFAULT_REPO = "ll7/robot_sf_ll7"
PER_PAGE = 100
DEFAULT_MAX_ISSUE_PAGES = 10
DEFAULT_MAX_PR_PAGES = 20
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


@dataclass(frozen=True)
class PaginationMeta:
    """Deterministic summary of one bounded REST pagination pass.

    ``truncated`` is ``True`` when the pass consumed its full page budget with
    every page full, which means more rows may exist beyond the budget. It reuses
    :func:`scripts.dev._gh_pagination.is_likely_truncated` so the cap signal
    matches the shared truncation-guard contract: a result of exactly
    ``page_budget * per_page`` rows is treated as potentially truncated.
    """

    pages_read: int
    per_page: int
    page_budget: int
    row_count: int
    truncated: bool


def _optional_str(value: object) -> str:
    """Coerce an optional JSON field to text, mapping an explicit null to ``""``.

    Prevents a null field (parsed to ``None``) from being coerced to the literal
    string ``"None"``; valid strings pass through unchanged.
    """
    return str(value) if value is not None else ""


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
    """Parse an open-issue row number from a normalized issue payload.

    The REST issues endpoint also returns pull requests (a PR is an issue in
    GitHub's data model); requiring a canonical ``/issues/<n>`` URL and an open
    state filters those out without an extra request.
    """
    if not _is_issue_url(row.get("url")):
        return None
    if _optional_str(row.get("state")).lower() != "open":
        return None
    try:
        return int(row["number"])
    except (KeyError, TypeError, ValueError):
        return None


def _pull_request(row: dict[str, object], *, issue_number: int) -> LinkedPullRequest | None:
    """Parse a merged PR row and require a title link to the issue number."""
    if not _is_pull_request_url(row.get("url")):
        return None
    if _optional_str(row.get("state")).lower() != "merged":
        return None
    title = _optional_str(row.get("title"))
    if not _title_mentions_issue(title, issue_number):
        return None
    try:
        number = int(row["number"])
    except (KeyError, TypeError, ValueError):
        return None
    return LinkedPullRequest(
        number=number,
        title=title,
        url=_optional_str(row.get("url")),
        merged_at=_optional_str(row.get("mergedAt") or row.get("closedAt")),
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


def _normalize_rest_issue_row(row: dict[str, object]) -> dict[str, object]:
    """Project a REST issue payload onto the fields the audit consumes."""
    return {
        "number": row.get("number"),
        "title": _optional_str(row.get("title")),
        "url": _optional_str(row.get("html_url") or row.get("url")),
        "state": _optional_str(row.get("state")),
    }


def _is_merged_pr_row(row: dict[str, object]) -> bool:
    """Return true only for a closed pull request that was actually merged.
    The REST pulls endpoint with ``state=closed`` returns both merged and
    closed-unmerged pull requests. The list endpoint does not populate the
    ``merged`` boolean (it comes back as ``null``); the reliable merged signal
    there is a non-null ``merged_at`` timestamp. The single-PR endpoint does set
    ``merged`` to an explicit boolean, so honor it when present and otherwise
    fall back to ``merged_at``.
    """
    merged_flag = row.get("merged")
    if merged_flag is True:
        return True
    if merged_flag is False:
        return False
    return bool(row.get("merged_at"))


def _normalize_rest_pr_row(row: dict[str, object]) -> dict[str, object]:
    """Project a REST pull payload onto the merged-PR row shape the audit parses."""
    merged = _is_merged_pr_row(row)
    return {
        "number": row.get("number"),
        "title": _optional_str(row.get("title")),
        "url": _optional_str(row.get("html_url") or row.get("url")),
        "state": "merged" if merged else _optional_str(row.get("state")),
        "mergedAt": _optional_str(row.get("merged_at") or row.get("closed_at")),
        "closedAt": _optional_str(row.get("closed_at")),
    }


def collect_candidates(
    open_issue_rows: list[dict[str, object]],
    merged_pr_rows_by_issue: dict[int, list[dict[str, object]]],
) -> list[ClosureAuditCandidate]:
    """Collect open issues with merged title-linked PRs from pre-fetched rows.

    ``merged_pr_rows_by_issue`` is the locally built title-to-issue index: each
    open issue number maps to the merged PR rows whose titles mention it. The
    per-row parser re-validates the URL shape, merged state, and title link so a
    malformed or stale index row can never inflate a candidate.
    """
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
        title = _optional_str(issue_row.get("title"))
        classification, recommended_action = _classification_for_issue(title)
        candidates.append(
            ClosureAuditCandidate(
                number=number,
                title=title,
                url=_optional_str(issue_row.get("url")),
                title_linked_prs=tuple(sorted(linked_prs, key=lambda pr: pr.number)),
                classification=classification,
                recommended_action=recommended_action,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.number)


def build_title_linked_index(
    merged_pr_rows: list[dict[str, object]],
    issue_numbers: list[int],
) -> dict[int, list[dict[str, object]]]:
    """Build the title-to-issue index from a flat list of merged PR rows.

    The merged pull requests are fetched once in a single bounded REST pass; this
    function matches each PR title against every open issue number in-process so
    the audit never issues one search request per open issue.
    """
    index: dict[int, list[dict[str, object]]] = {number: [] for number in issue_numbers}
    if not issue_numbers:
        return index
    for pr_row in merged_pr_rows:
        title = _optional_str(pr_row.get("title"))
        for number in issue_numbers:
            if _title_mentions_issue(title, number):
                index[number].append(pr_row)
    return index


def _gh_api_get(path: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run a read-only ``gh api`` GET and return the completed process.

    The caller inspects ``returncode``/``stderr`` and parses ``stdout``. A missing
    GitHub CLI is raised as :class:`RuntimeError` so the CLI entry point can emit
    the schema-valid error packet and exit non-zero.
    """
    args = ["gh", "api", path]
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"GitHub REST read timed out after {timeout}s ({path})") from exc


def _paginate_rest(
    path: str,
    *,
    max_pages: int,
    per_page: int,
) -> tuple[list[dict[str, object]], PaginationMeta]:
    """Read a list REST endpoint across a bounded number of ``per_page`` pages.

    Pagination stops early as soon as a page returns fewer rows than ``per_page``
    (a definitive end-of-results signal). When every page through the page budget
    is full, the pass is flagged as potentially truncated via
    :func:`is_likely_truncated` so downstream consumers treat the inventory as
    partial rather than authoritative.
    """
    rows: list[dict[str, object]] = []
    pages_read = 0
    for page in range(1, max_pages + 1):
        paged_path = f"{path}&per_page={per_page}&page={page}"
        result = _gh_api_get(paged_path)
        if result.returncode != 0:
            detail = (
                result.stderr or result.stdout or ""
            ).strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"GitHub REST read failed ({paged_path}): {detail}")
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from GitHub REST ({paged_path}): {exc.msg}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list from GitHub REST ({paged_path})")
        page_rows = [row for row in payload if isinstance(row, dict)]
        rows.extend(page_rows)
        pages_read = page
        if len(page_rows) < per_page:
            break
    truncated = is_likely_truncated(len(rows), limit=max_pages * per_page)
    return rows, PaginationMeta(
        pages_read=pages_read,
        per_page=per_page,
        page_budget=max_pages,
        row_count=len(rows),
        truncated=truncated,
    )


def fetch_open_issue_rows(
    *,
    repo: str,
    max_pages: int = DEFAULT_MAX_ISSUE_PAGES,
    per_page: int = PER_PAGE,
) -> tuple[list[dict[str, object]], PaginationMeta]:
    """Fetch open issues through a bounded paginated REST read.

    Returns the raw REST issue rows (which may interleave open pull requests) and
    the pagination metadata. PR-shaped rows are filtered out later by
    :func:`_issue_number`, which requires a canonical ``/issues/<n>`` URL.
    """
    path = f"repos/{repo}/issues?state=open&sort=created&direction=desc"
    return _paginate_rest(path, max_pages=max_pages, per_page=per_page)


def fetch_closed_pr_rows(
    *,
    repo: str,
    max_pages: int = DEFAULT_MAX_PR_PAGES,
    per_page: int = PER_PAGE,
) -> tuple[list[dict[str, object]], PaginationMeta]:
    """Fetch closed pull requests through a single bounded paginated REST pass.

    Returns the raw REST closed-PR rows; merged-only filtering is performed locally
    by :func:`_is_merged_pr_row` so the audit never depends on a per-issue search.
    """
    path = f"repos/{repo}/pulls?state=closed&sort=updated&direction=desc"
    return _paginate_rest(path, max_pages=max_pages, per_page=per_page)


def build_rest_truncations(
    *,
    issue_meta: PaginationMeta,
    pr_meta: PaginationMeta,
    merged_count: int,
) -> dict[str, Any]:
    """Return structured page-budget truncation metadata for the REST inventories.

    Both inventories are bounded REST passes; when either hits its page budget the
    report records the page budget, pages read, and row counts so a partial
    inventory is deterministic and machine-auditable rather than silently dropped.
    """
    issue_truncated = issue_meta.truncated
    pr_truncated = pr_meta.truncated
    return {
        "open_issues": {
            "truncated": issue_truncated,
            "row_count": issue_meta.row_count,
            "pages_read": issue_meta.pages_read,
            "per_page": issue_meta.per_page,
            "page_budget": issue_meta.page_budget,
            "inventory": "issues?state=open",
            "note": (
                f"open-issue REST inventory may be partial: read {issue_meta.row_count} rows "
                f"in {issue_meta.pages_read}/{issue_meta.page_budget} pages "
                f"(per_page={issue_meta.per_page}); raise --max-issue-pages"
                if issue_truncated
                else ""
            ),
        },
        "merged_prs": {
            "truncated": pr_truncated,
            "row_count": pr_meta.row_count,
            "merged_count": merged_count,
            "pages_read": pr_meta.pages_read,
            "per_page": pr_meta.per_page,
            "page_budget": pr_meta.page_budget,
            "inventory": "pulls?state=closed (merged filtered locally)",
            "note": (
                f"merged-PR REST inventory may be partial: read {pr_meta.row_count} closed rows "
                f"({merged_count} merged) in {pr_meta.pages_read}/{pr_meta.page_budget} pages "
                f"(per_page={pr_meta.per_page}); raise --max-pr-pages"
                if pr_truncated
                else ""
            ),
        },
        "truncated_any": bool(issue_truncated or pr_truncated),
    }


def build_truncations(
    *,
    open_issue_rows: list[dict[str, object]],
    issue_limit: int,
    merged_pr_rows_by_issue: dict[int, list[dict[str, object]]],
    pr_limit_per_issue: int,
) -> dict[str, Any]:
    """Return row-cap truncation markers for two bounded list calls.

    Retained for the shared truncation-guard regression contract exercised by
    ``tests/dev/test_gh_list_truncation_remaining.py``. Production reads now use
    :func:`build_rest_truncations` for REST page-budget metadata; this helper
    keeps the generic row-cap signal (``len(rows) >= limit``) stable so the
    cross-module guard stays covered while the audit's own report moves to the
    REST pagination shape.
    """
    open_row_count = len(open_issue_rows)
    open_truncated = is_likely_truncated(open_row_count, limit=issue_limit)
    per_issue = [
        {
            "issue_number": issue_number,
            "row_count": len(rows),
            "truncated": True,
        }
        for issue_number, rows in sorted(merged_pr_rows_by_issue.items())
        if is_likely_truncated(len(rows), limit=pr_limit_per_issue)
    ]
    return {
        "open_issues": {
            "truncated": open_truncated,
            "row_count": open_row_count,
            "limit": issue_limit,
            "note": (
                f"open-issue inventory may be capped: got {open_row_count} rows at "
                f"--limit {issue_limit}; raise --issue-limit or paginate"
                if open_truncated
                else ""
            ),
        },
        "merged_prs_per_issue": {
            "limit": pr_limit_per_issue,
            "truncated_issues": per_issue,
        },
        "truncated_any": bool(open_truncated or per_issue),
    }


def build_report(
    *,
    repo: str,
    candidates: list[ClosureAuditCandidate],
    truncations: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "truncations": truncations or {},
        "truncated_any": bool(truncations and truncations.get("truncated_any")),
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
    parser.add_argument(
        "--max-issue-pages",
        type=int,
        default=DEFAULT_MAX_ISSUE_PAGES,
        help=(
            f"Maximum REST pages of open issues to read (each {PER_PAGE} rows, "
            f"default {DEFAULT_MAX_ISSUE_PAGES})."
        ),
    )
    parser.add_argument(
        "--max-pr-pages",
        type=int,
        default=DEFAULT_MAX_PR_PAGES,
        help=(
            f"Maximum REST pages of closed pull requests to read (each {PER_PAGE} rows, "
            f"default {DEFAULT_MAX_PR_PAGES})."
        ),
    )
    return parser


def _dump_json(payload: dict[str, Any]) -> None:
    """Print compact, deterministic JSON."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    """Run the read-only closure audit."""
    args = _build_parser().parse_args(argv)
    if args.max_issue_pages < 1 or args.max_pr_pages < 1:
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
                "error": (
                    "page budgets must be >= 1; got "
                    f"max_issue_pages={args.max_issue_pages}, "
                    f"max_pr_pages={args.max_pr_pages}"
                ),
            }
        )
        return 2
    try:
        raw_open_rows, issue_meta = fetch_open_issue_rows(
            repo=args.repo, max_pages=args.max_issue_pages
        )
        open_issue_rows = [_normalize_rest_issue_row(row) for row in raw_open_rows]
        issue_numbers = [
            number
            for number in (_issue_number(row) for row in open_issue_rows)
            if number is not None
        ]
        raw_pr_rows, pr_meta = fetch_closed_pr_rows(repo=args.repo, max_pages=args.max_pr_pages)
        merged_pr_rows = [
            _normalize_rest_pr_row(row) for row in raw_pr_rows if _is_merged_pr_row(row)
        ]
        index = build_title_linked_index(merged_pr_rows, issue_numbers)
        truncations = build_rest_truncations(
            issue_meta=issue_meta,
            pr_meta=pr_meta,
            merged_count=len(merged_pr_rows),
        )
        report = build_report(
            repo=args.repo,
            candidates=collect_candidates(open_issue_rows, index),
            truncations=truncations,
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
    # Exit non-zero when there are closure candidates OR the REST inventory is
    # partial, so a truncated "no candidates" result is never treated as success.
    return 0 if (report["ok"] and not report["truncated_any"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
