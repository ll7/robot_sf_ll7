#!/usr/bin/env python3
"""Report open issues with active state labels whose work already has a merged PR.

The command is intentionally read-only.  It discovers open issues carrying the
active routing labels, follows each issue's bounded timeline, and verifies any
merged pull-request reference through the pull-request REST endpoint.  A
candidate is a human review packet, not proof that the merged change fully
satisfied the issue: the report never closes issues, changes labels, or edits
Project #5 state.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev._gh_rest import run_gh_api_or_raise

DEFAULT_REPO = "ll7/robot_sf_ll7"
PER_PAGE = 100
DEFAULT_MAX_ISSUE_PAGES = 10
DEFAULT_MAX_TIMELINE_PAGES = 3
DEFAULT_MAX_PR_LOOKUPS = 500

# ``state:working`` is a live queue qualifier used by the open-issue routing
# surface.  The closed-issue guard has a narrower historical LIVE_STATE_LABELS
# tuple; keep this open-issue contract explicit instead of silently changing it.
OPEN_ACTIVE_STATE_LABELS = ("state:ready", "state:running", "state:working")
TIMELINE_REFERENCE_EVENTS = frozenset({"cross-referenced", "referenced"})
ISSUE_NUMBER_RE = re.compile(r"/(?:issues|pull)/(\d+)(?:/|$)")

RestRunner = Callable[[str], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class PaginationMeta:
    """Bounded REST pagination evidence for one endpoint family."""

    endpoint: str
    pages_read: int
    per_page: int
    page_budget: int
    row_count: int
    truncated: bool

    def to_payload(self) -> dict[str, Any]:
        """Return stable JSON metadata for the report."""
        return {
            "endpoint": self.endpoint,
            "pages_read": self.pages_read,
            "per_page": self.per_page,
            "page_budget": self.page_budget,
            "row_count": self.row_count,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ActiveIssue:
    """Open issue carrying one or more active routing labels."""

    number: int
    title: str
    url: str
    state: str
    active_labels: tuple[str, ...]

    def to_payload(self, merged_prs: tuple[MergedPullRequest, ...]) -> dict[str, Any]:
        """Return a candidate row with its verified merged-PR references."""
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "state": self.state,
            "active_labels": list(self.active_labels),
            "classification": "merged_reference_needs_exact_fix_review",
            "recommended_action": "verify_exact_fix_then_close_or_relabel",
            "merged_prs": [pr.to_payload() for pr in merged_prs],
        }


@dataclass(frozen=True)
class MergedPullRequest:
    """A merged PR verified after an issue timeline reference."""

    issue_number: int
    number: int
    title: str
    url: str
    merged_at: str
    merge_commit_sha: str
    coverage_source: str
    timeline_event_created_at: str

    def to_payload(self) -> dict[str, Any]:
        """Return the exact PR and merge evidence carried by a candidate."""
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "merged_at": self.merged_at,
            "merge_commit_sha": self.merge_commit_sha,
            "coverage_source": self.coverage_source,
            "timeline_issue": self.issue_number,
            "timeline_event_created_at": self.timeline_event_created_at,
        }


def _default_runner(path: str) -> subprocess.CompletedProcess[str]:
    """Run one bounded read-only GitHub REST request."""
    return run_gh_api_or_raise(path)


def _label_names(raw_labels: object) -> set[str]:
    """Extract label names from REST issue rows."""
    if not isinstance(raw_labels, list):
        return set()
    names: set[str] = set()
    for label in raw_labels:
        if isinstance(label, str):
            names.add(label)
        elif isinstance(label, Mapping) and isinstance(label.get("name"), str):
            names.add(str(label["name"]))
    return names


def _canonical_url(raw_url: object, *, resource: str, repo: str | None = None) -> str | None:
    """Return a canonical repository issue/PR URL path or ``None``."""
    if not isinstance(raw_url, str):
        return None
    parts = [part for part in urlsplit(raw_url).path.split("/") if part]
    if len(parts) != 4 or parts[2] != resource or not parts[3].isdigit():
        return None
    if repo is not None and "/".join(parts[:2]) != repo:
        return None
    return raw_url


def _number_from_url(raw_url: object, *, resource: str) -> int | None:
    """Extract a numeric issue or pull-request identifier from a canonical URL."""
    url = _canonical_url(raw_url, resource=resource)
    if url is None:
        return None
    match = ISSUE_NUMBER_RE.search(urlsplit(url).path)
    return int(match.group(1)) if match else None


def _parse_json(result: subprocess.CompletedProcess[str], *, path: str) -> object:
    """Parse one REST response and preserve non-zero or malformed failures."""
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"GitHub REST read failed ({path}): {detail or result.returncode}")
    try:
        return json.loads(result.stdout or "null")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from GitHub REST ({path}): {exc.msg}") from exc


def _paginate(
    path: str,
    *,
    max_pages: int,
    per_page: int,
    runner: RestRunner,
) -> tuple[list[dict[str, object]], PaginationMeta]:
    """Read a bounded REST list and fail closed on partial or malformed pages."""
    if max_pages < 1 or per_page < 1:
        raise ValueError(
            f"pagination budgets must be >= 1; got max_pages={max_pages}, per_page={per_page}"
        )

    rows: list[dict[str, object]] = []
    pages_read = 0
    for page in range(1, max_pages + 1):
        separator = "&" if "?" in path else "?"
        endpoint = f"{path}{separator}per_page={per_page}&page={page}"
        payload = _parse_json(runner(endpoint), path=endpoint)
        if not isinstance(payload, list):
            raise RuntimeError(f"Expected a JSON list from GitHub REST ({endpoint})")
        if any(not isinstance(row, dict) for row in payload):
            raise RuntimeError(f"Expected JSON object rows from GitHub REST ({endpoint})")
        rows.extend(row for row in payload if isinstance(row, dict))
        pages_read = page
        if len(payload) < per_page:
            break

    return rows, PaginationMeta(
        endpoint=path,
        pages_read=pages_read,
        per_page=per_page,
        page_budget=max_pages,
        row_count=len(rows),
        truncated=is_likely_truncated(len(rows), limit=max_pages * per_page),
    )


def _normalize_issue_row(row: Mapping[str, object]) -> dict[str, object] | None:
    """Keep only open canonical issues from GitHub's mixed issues endpoint."""
    url = _canonical_url(row.get("html_url") or row.get("url"), resource="issues")
    if url is None or str(row.get("state") or "").lower() != "open":
        return None
    raw_number = row.get("number")
    try:
        number = (
            int(raw_number) if raw_number is not None else _number_from_url(url, resource="issues")
        )
    except (TypeError, ValueError):
        return None
    if number is None or number <= 0:
        return None
    return {
        "number": number,
        "title": str(row.get("title") or ""),
        "url": url,
        "state": "open",
        "labels": _label_names(row.get("labels")),
    }


def fetch_open_issues_by_label(
    *,
    repo: str,
    labels: tuple[str, ...] = OPEN_ACTIVE_STATE_LABELS,
    max_pages: int = DEFAULT_MAX_ISSUE_PAGES,
    per_page: int = PER_PAGE,
    runner: RestRunner | None = None,
) -> tuple[dict[str, list[dict[str, object]]], list[PaginationMeta]]:
    """Fetch open issues for each active label through bounded REST pages."""
    run = runner or _default_runner
    rows_by_label: dict[str, list[dict[str, object]]] = {}
    metadata: list[PaginationMeta] = []
    for label in labels:
        path = f"repos/{repo}/issues?state=open&labels={quote(label, safe='')}"
        rows, meta = _paginate(path, max_pages=max_pages, per_page=per_page, runner=run)
        rows_by_label[label] = rows
        metadata.append(meta)
    return rows_by_label, metadata


def collect_active_issues(
    rows_by_label: Mapping[str, list[dict[str, object]]],
    *,
    watched_labels: tuple[str, ...] = OPEN_ACTIVE_STATE_LABELS,
) -> list[ActiveIssue]:
    """Deduplicate mixed endpoint rows and union the labels seen per issue."""
    watched = set(watched_labels)
    rows_by_number: dict[int, dict[str, object]] = {}
    labels_by_number: dict[int, set[str]] = {}
    for search_label, rows in rows_by_label.items():
        for raw_row in rows:
            row = _normalize_issue_row(raw_row)
            if row is None:
                continue
            number = int(row["number"])
            labels = set(row["labels"]) & watched
            # The REST endpoint is filtered by label, but test fixtures and API
            # compatibility layers may omit the labels array.  Keep the endpoint
            # filter as evidence while preserving current-row validation later.
            if search_label in watched:
                labels.add(search_label)
            if not labels:
                continue
            rows_by_number.setdefault(number, row)
            labels_by_number.setdefault(number, set()).update(labels)

    return [
        ActiveIssue(
            number=number,
            title=str(rows_by_number[number]["title"]),
            url=str(rows_by_number[number]["url"]),
            state="open",
            active_labels=tuple(sorted(labels_by_number[number])),
        )
        for number in sorted(rows_by_number)
    ]


def fetch_current_issue(
    *,
    repo: str,
    number: int,
    runner: RestRunner | None = None,
) -> dict[str, object]:
    """Read one current issue row so search/index state cannot authorize a report."""
    run = runner or _default_runner
    path = f"repos/{repo}/issues/{number}"
    payload = _parse_json(run(path), path=path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON issue object from GitHub REST ({path})")
    return payload


def reconcile_active_issues(
    *,
    repo: str,
    candidates: list[ActiveIssue],
    watched_labels: tuple[str, ...] = OPEN_ACTIVE_STATE_LABELS,
    runner: RestRunner | None = None,
) -> list[ActiveIssue]:
    """Confirm open state and active labels against current REST issue records."""
    watched = set(watched_labels)
    reconciled: list[ActiveIssue] = []
    for candidate in candidates:
        row = fetch_current_issue(repo=repo, number=candidate.number, runner=runner)
        url = _canonical_url(row.get("html_url") or row.get("url"), resource="issues")
        if url is None or str(row.get("state") or "").lower() != "open":
            continue
        active_labels = _label_names(row.get("labels")) & watched
        if not active_labels:
            continue
        reconciled.append(
            ActiveIssue(
                number=candidate.number,
                title=str(row.get("title") or candidate.title),
                url=url,
                state="open",
                active_labels=tuple(sorted(active_labels)),
            )
        )
    return reconciled


def _source_issue_from_event(event: Mapping[str, object]) -> Mapping[str, object] | None:
    """Extract the linked issue/PR object from GitHub timeline event variants."""
    source = event.get("source")
    if not isinstance(source, Mapping):
        return None
    source_issue = source.get("issue")
    return source_issue if isinstance(source_issue, Mapping) else None


def _timeline_pr_number(source_issue: Mapping[str, object]) -> int | None:
    """Read a PR number from the timeline object without trusting title text."""
    raw_number = source_issue.get("number")
    try:
        number = int(raw_number) if raw_number is not None else None
    except (TypeError, ValueError):
        number = None
    if number and _canonical_url(
        source_issue.get("html_url") or source_issue.get("url"), resource="pull"
    ):
        return number
    return _number_from_url(
        source_issue.get("html_url") or source_issue.get("url"), resource="pull"
    )


def _timeline_candidate(
    *,
    issue_number: int,
    event: Mapping[str, object],
    repo: str | None = None,
) -> tuple[int, str, str, str] | None:
    """Extract a merged PR reference from one timeline event."""
    if str(event.get("event") or "") not in TIMELINE_REFERENCE_EVENTS:
        return None
    source_issue = _source_issue_from_event(event)
    if source_issue is None:
        return None
    pull_request = source_issue.get("pull_request")
    if not isinstance(pull_request, Mapping):
        return None
    merged_at = str(pull_request.get("merged_at") or source_issue.get("merged_at") or "")
    if not merged_at:
        return None
    number = _timeline_pr_number(source_issue)
    url = _canonical_url(
        source_issue.get("html_url") or source_issue.get("url"),
        resource="pull",
        repo=repo,
    )
    if number is None or url is None:
        return None
    return (
        number,
        str(source_issue.get("title") or ""),
        url,
        str(event.get("created_at") or ""),
    )


def fetch_merged_pull_request(
    *,
    repo: str,
    number: int,
    runner: RestRunner | None = None,
) -> dict[str, object] | None:
    """Verify a timeline PR is still merged and carries a merge commit SHA."""
    run = runner or _default_runner
    path = f"repos/{repo}/pulls/{number}"
    payload = _parse_json(run(path), path=path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Expected a JSON pull-request object from GitHub REST ({path})")
    merged_at = str(payload.get("merged_at") or "")
    merged = payload.get("merged") is True or bool(merged_at)
    merge_commit_sha = str(payload.get("merge_commit_sha") or "")
    url = _canonical_url(
        payload.get("html_url") or payload.get("url"),
        resource="pull",
        repo=repo,
    )
    if not merged or not merge_commit_sha or url is None:
        return None
    return {
        "number": number,
        "title": str(payload.get("title") or ""),
        "url": url,
        "merged_at": merged_at,
        "merge_commit_sha": merge_commit_sha,
    }


def _fetch_issue_timeline(
    *,
    repo: str,
    issue_number: int,
    max_timeline_pages: int,
    per_page: int,
    runner: RestRunner,
) -> tuple[list[dict[str, object]], PaginationMeta | None, str | None]:
    """Fetch one issue timeline while converting failures to explicit evidence."""
    path = f"repos/{repo}/issues/{issue_number}/timeline"
    try:
        events, metadata = _paginate(
            path,
            max_pages=max_timeline_pages,
            per_page=per_page,
            runner=runner,
        )
    except (RuntimeError, ValueError) as exc:
        return [], None, f"issue {issue_number}: {exc}"
    return events, metadata, None


def _make_merged_reference(
    *,
    issue_number: int,
    candidate: tuple[int, str, str, str],
    details: Mapping[str, object],
) -> MergedPullRequest:
    """Combine timeline provenance with the current verified PR response."""
    pr_number, timeline_title, timeline_url, event_created_at = candidate
    return MergedPullRequest(
        issue_number=issue_number,
        number=pr_number,
        title=str(details.get("title") or timeline_title),
        url=str(details.get("url") or timeline_url),
        merged_at=str(details.get("merged_at") or ""),
        merge_commit_sha=str(details.get("merge_commit_sha") or ""),
        coverage_source="issue_timeline_merged_pr",
        timeline_event_created_at=event_created_at,
    )


def _collect_issue_references(
    *,
    repo: str,
    issue: ActiveIssue,
    max_timeline_pages: int,
    per_page: int,
    max_pr_lookups: int,
    runner: RestRunner,
    pr_cache: dict[int, dict[str, object] | None],
    lookup_count: int,
) -> tuple[list[MergedPullRequest], PaginationMeta | None, list[str], int, bool]:
    """Collect verified PR references for one issue and report budget exhaustion."""
    events, metadata, timeline_error = _fetch_issue_timeline(
        repo=repo,
        issue_number=issue.number,
        max_timeline_pages=max_timeline_pages,
        per_page=per_page,
        runner=runner,
    )
    if timeline_error:
        return [], metadata, [timeline_error], lookup_count, False

    references: list[MergedPullRequest] = []
    errors: list[str] = []
    for raw_event in events:
        event = raw_event if isinstance(raw_event, Mapping) else {}
        candidate = _timeline_candidate(issue_number=issue.number, event=event, repo=repo)
        if candidate is None:
            continue
        pr_number = candidate[0]
        if pr_number not in pr_cache:
            if lookup_count >= max_pr_lookups:
                errors.append(
                    f"merged PR lookup budget exceeded at {max_pr_lookups}; raise --max-pr-lookups"
                )
                return references, metadata, errors, lookup_count, True
            lookup_count += 1
            try:
                pr_cache[pr_number] = fetch_merged_pull_request(
                    repo=repo,
                    number=pr_number,
                    runner=runner,
                )
            except (RuntimeError, ValueError) as exc:
                errors.append(f"PR #{pr_number}: {exc}")
                pr_cache[pr_number] = None
        details = pr_cache[pr_number]
        if details is not None:
            references.append(
                _make_merged_reference(
                    issue_number=issue.number,
                    candidate=candidate,
                    details=details,
                )
            )
    return references, metadata, errors, lookup_count, False


def discover_merged_references(
    *,
    repo: str,
    issues: list[ActiveIssue],
    max_timeline_pages: int = DEFAULT_MAX_TIMELINE_PAGES,
    per_page: int = PER_PAGE,
    max_pr_lookups: int = DEFAULT_MAX_PR_LOOKUPS,
    runner: RestRunner | None = None,
) -> tuple[dict[int, tuple[MergedPullRequest, ...]], dict[str, Any]]:
    """Follow issue timelines and verify merged PR details with fail-closed metadata."""
    if max_pr_lookups < 1:
        raise ValueError(f"max_pr_lookups must be >= 1; got {max_pr_lookups}")
    run = runner or _default_runner
    references: dict[int, list[MergedPullRequest]] = {}
    pr_cache: dict[int, dict[str, object] | None] = {}
    timeline_meta: list[PaginationMeta] = []
    errors: list[str] = []
    truncated = False
    lookup_count = 0

    for issue in issues:
        issue_references, metadata, issue_errors, lookup_count, budget_exceeded = (
            _collect_issue_references(
                repo=repo,
                issue=issue,
                max_timeline_pages=max_timeline_pages,
                per_page=per_page,
                max_pr_lookups=max_pr_lookups,
                runner=run,
                pr_cache=pr_cache,
                lookup_count=lookup_count,
            )
        )
        if metadata is not None:
            timeline_meta.append(metadata)
            truncated = truncated or metadata.truncated
        errors.extend(issue_errors)
        if issue_references:
            references.setdefault(issue.number, []).extend(issue_references)
        if budget_exceeded:
            break

    deduplicated: dict[int, tuple[MergedPullRequest, ...]] = {}
    for issue_number, rows in references.items():
        unique = {(row.number, row.merge_commit_sha): row for row in rows}
        deduplicated[issue_number] = tuple(unique[key] for key in sorted(unique))

    return deduplicated, {
        "complete_for_open_issues": not errors and not truncated,
        "issue_count": len(issues),
        "timeline_count": len(timeline_meta),
        "pr_lookup_count": lookup_count,
        "max_pr_lookups": max_pr_lookups,
        "truncated": truncated,
        "errors": errors,
        "timelines": [meta.to_payload() for meta in timeline_meta],
    }


def build_report(
    *,
    repo: str,
    checked_labels: tuple[str, ...],
    issues: list[ActiveIssue],
    references_by_issue: Mapping[int, tuple[MergedPullRequest, ...]],
    discovery_metadata: list[PaginationMeta],
    coverage_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a report that distinguishes findings from incomplete coverage."""
    candidates = [
        issue.to_payload(references_by_issue[issue.number])
        for issue in issues
        if references_by_issue.get(issue.number)
    ]
    truncations = [meta.to_payload() for meta in discovery_metadata]
    truncated_any = bool(
        any(meta.truncated for meta in discovery_metadata) or coverage_metadata.get("truncated")
    )
    complete = bool(coverage_metadata.get("complete_for_open_issues")) and not truncated_any
    candidate_count = len(candidates)
    return {
        "schema": "open_state_label_hygiene.v1",
        "ok": complete and candidate_count == 0,
        "read_only": True,
        "issue_writes": False,
        "project_writes": False,
        "repo": repo,
        "checked_labels": list(checked_labels),
        "candidate_count": candidate_count,
        "truncated_any": truncated_any,
        "complete_for_open_issues": complete,
        "discovery": truncations,
        "coverage": dict(coverage_metadata),
        "issues": candidates,
        "failure_summary": (
            {
                "reason": "open_issues_with_verified_merged_pr_references",
                "count": candidate_count,
            }
            if candidate_count
            else {
                "reason": "incomplete_open_issue_timeline_coverage",
                "count": 0,
            }
            if not complete
            else None
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the report-only CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository OWNER/REPO.")
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Active state label to check; may be repeated.",
    )
    parser.add_argument(
        "--max-issue-pages",
        type=int,
        default=DEFAULT_MAX_ISSUE_PAGES,
        help=f"Maximum REST pages per active label (each {PER_PAGE} rows).",
    )
    parser.add_argument(
        "--max-timeline-pages",
        type=int,
        default=DEFAULT_MAX_TIMELINE_PAGES,
        help=f"Maximum REST timeline pages per open issue (each {PER_PAGE} rows).",
    )
    parser.add_argument(
        "--max-pr-lookups",
        type=int,
        default=DEFAULT_MAX_PR_LOOKUPS,
        help=f"Maximum verified merged-PR detail reads (default {DEFAULT_MAX_PR_LOOKUPS}).",
    )
    return parser


def _dump_json(payload: Mapping[str, Any]) -> None:
    """Print stable machine-readable JSON."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    """Run the report-only open-state hygiene guard."""
    args = _build_parser().parse_args(argv)
    labels = tuple(args.labels) if args.labels else OPEN_ACTIVE_STATE_LABELS
    if not labels or any(label not in OPEN_ACTIVE_STATE_LABELS for label in labels):
        _dump_json(
            {
                "schema": "open_state_label_hygiene.v1",
                "ok": False,
                "read_only": True,
                "issue_writes": False,
                "project_writes": False,
                "repo": args.repo,
                "checked_labels": list(labels),
                "error": (
                    "--label values must be active labels: " + ", ".join(OPEN_ACTIVE_STATE_LABELS)
                ),
            }
        )
        return 2

    try:
        rows_by_label, discovery_metadata = fetch_open_issues_by_label(
            repo=args.repo,
            labels=labels,
            max_pages=args.max_issue_pages,
        )
        discovered = collect_active_issues(rows_by_label, watched_labels=labels)
        issues = reconcile_active_issues(
            repo=args.repo,
            candidates=discovered,
            watched_labels=labels,
        )
        references, coverage_metadata = discover_merged_references(
            repo=args.repo,
            issues=issues,
            max_timeline_pages=args.max_timeline_pages,
            max_pr_lookups=args.max_pr_lookups,
        )
        report = build_report(
            repo=args.repo,
            checked_labels=labels,
            issues=issues,
            references_by_issue=references,
            discovery_metadata=discovery_metadata,
            coverage_metadata=coverage_metadata,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _dump_json(
            {
                "schema": "open_state_label_hygiene.v1",
                "ok": False,
                "read_only": True,
                "issue_writes": False,
                "project_writes": False,
                "repo": args.repo,
                "checked_labels": list(labels),
                "candidate_count": None,
                "complete_for_open_issues": False,
                "issues": [],
                "error": str(exc),
            }
        )
        return 2

    _dump_json(report)
    if report["ok"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
