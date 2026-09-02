#!/usr/bin/env python3
"""Audit closed issues for stale live state labels.

The command is intentionally read-only: it searches GitHub issues and reports any closed issue that
still carries labels used for active implementation routing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

from scripts.dev import gh_pr_label_rest
from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev.gh_issue_rest import VALID_ISSUE_STATES, validate_issue_identity

DEFAULT_REPO = "ll7/robot_sf_ll7"
LIVE_STATE_LABELS = ("state:ready", "state:running", "state:blocked")
JSON_FIELDS = "number,title,url,state,labels,isPullRequest"
_MISSING = object()
PER_PAGE = 100
DEFAULT_MAX_REST_PAGES = 10
GRAPHQL_FALLBACK_MARKERS = ("graphql:", "graphql ")
GRAPHQL_FAIL_CLOSED_MARKERS = (
    "bad credentials",
    "requires authentication",
    "authentication required",
    "resource not accessible by integration",
    "forbidden",
    "permission denied",
    "could not resolve to a repository",
    "could not resolve to an issue",
    "repository not found",
)


@dataclass(frozen=True)
class StaleIssue:
    """Closed issue carrying one or more live state labels."""

    number: int
    title: str
    url: str
    state: str
    stale_labels: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable issue summary."""
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "state": self.state,
            "stale_labels": list(self.stale_labels),
        }


@dataclass(frozen=True)
class RestLabelPageMeta:
    """Pagination metadata for one bounded REST label inventory."""

    label: str
    pages_read: int
    per_page: int
    page_budget: int
    row_count: int
    truncated: bool


@dataclass(frozen=True)
class CandidateDiscoveryResult:
    """Candidate rows plus truncation metadata from one discovery path."""

    rows_by_label: dict[str, list[dict[str, object]]]
    truncations: list[dict[str, Any]]
    source: str


def _label_names(raw_labels: object) -> set[str]:
    """Extract and validate label names from a GitHub payload."""
    if not isinstance(raw_labels, list):
        raise ValueError("candidate labels must be a list")

    names: set[str] = set()
    for index, label in enumerate(raw_labels):
        if isinstance(label, str):
            name = label
        elif isinstance(label, dict) and isinstance(label.get("name"), str):
            name = label["name"]
        else:
            raise ValueError(f"candidate label row {index} must contain a name string")
        if not name.strip():
            raise ValueError(f"candidate label row {index} must contain a non-empty name")
        if name in names:
            raise ValueError(f"candidate label row {index} duplicates label {name!r}")
        names.add(name)
    return names


def _is_pull_request_url(raw_url: object) -> bool:
    """Return True for syntactically canonical pull-request URLs.

    Callers that have a requested repository and number must use
    :func:`validate_issue_identity` for the authoritative check.
    """
    if not isinstance(raw_url, str):
        return False

    parsed = urlsplit(raw_url)
    expected_host = (os.environ.get("GH_HOST") or "github.com").lower()
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != expected_host
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) != 4:
        return False
    owner, repo, resource, number = path_parts
    return bool(owner and repo and resource == "pull" and number.isdecimal())


def _validate_candidate_row(
    row: dict[str, object],
    *,
    repo: str,
    context: str,
) -> dict[str, object]:
    """Validate one discovery row before it can affect the audit result."""
    number = row.get("number")
    if type(number) is not int or number < 1:
        raise ValueError(f"malformed candidate row ({context}): number must be a positive integer")
    title = row.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"malformed candidate row ({context}): title must be non-empty text")
    raw_state = row.get("state")
    if not isinstance(raw_state, str):
        raise ValueError(f"malformed candidate row ({context}): state must be text")
    state = raw_state.upper()
    if state not in VALID_ISSUE_STATES:
        raise ValueError(
            f"malformed candidate row ({context}): state must be OPEN or CLOSED, got {raw_state!r}"
        )
    is_pull_request = row.get("is_pull_request")
    if type(is_pull_request) is not bool:
        raise ValueError(f"malformed candidate row ({context}): is_pull_request must be a boolean")
    labels = _label_names(row.get("labels"))
    identity = {
        "number": number,
        "state": state,
        "url": row.get("url"),
        "is_pull_request": is_pull_request,
    }
    try:
        validate_issue_identity(identity, repo=repo, number=number)
    except ValueError as exc:
        raise ValueError(f"malformed candidate row ({context}): {exc}") from exc

    normalized = dict(row)
    normalized["state"] = state
    normalized["labels"] = sorted(labels)
    return normalized


def _normalize_native_search_row(
    row: dict[str, object],
    *,
    repo: str,
    context: str,
) -> dict[str, object]:
    """Normalize the GitHub CLI search discriminator into the shared row shape."""
    marker = row.get("isPullRequest")
    if type(marker) is not bool:
        raise ValueError(f"malformed candidate row ({context}): isPullRequest must be a boolean")
    normalized = dict(row)
    normalized["is_pull_request"] = marker
    return _validate_candidate_row(normalized, repo=repo, context=context)


def build_search_command(*, repo: str, label: str, limit: int) -> list[str]:
    """Build the read-only GitHub CLI search command for one state label."""
    return [
        "gh",
        "search",
        "issues",
        "--repo",
        repo,
        "--state",
        "closed",
        "--label",
        label,
        "--json",
        JSON_FIELDS,
        "--limit",
        str(limit),
    ]


def _run_search_command(
    command: list[str],
    *,
    repo: str = DEFAULT_REPO,
) -> list[dict[str, object]]:
    """Run one read-only GitHub search command and validate every object row."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' was not found; install gh or add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f": {stderr}" if stderr else ""
        raise RuntimeError(f"GitHub CLI command failed ({' '.join(command)}){details}") from exc

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse GitHub CLI JSON output ({' '.join(command)}): {exc.msg}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list from {' '.join(command)}")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(
                f"Malformed GitHub CLI candidate row {index} from {' '.join(command)}: "
                "expected an object"
            )
        rows.append(
            _normalize_native_search_row(
                row,
                repo=repo,
                context=f"gh search row {index}",
            )
        )
    return rows


def _parse_rest_list_payload(
    stdout: str,
    *,
    context: str,
) -> list[dict[str, object]]:
    """Parse a REST list payload and require an object row list."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from GitHub REST ({context}): {exc.msg}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list from GitHub REST ({context})")
    rows: list[dict[str, object]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON list of objects from GitHub REST ({context})")
        rows.append(row)
    return rows


def _normalize_rest_issue_row(
    row: dict[str, object],
    *,
    repo: str = DEFAULT_REPO,
) -> dict[str, object]:
    """Project and validate one REST issue row for the audit."""
    marker = row.get("pull_request", _MISSING)
    if marker is not _MISSING and not isinstance(marker, dict):
        raise ValueError("REST candidate pull_request must be an object when present")
    normalized = {
        "number": row.get("number"),
        "title": row.get("title"),
        "url": row.get("html_url"),
        "state": row.get("state"),
        "labels": row.get("labels"),
        "is_pull_request": marker is not _MISSING,
    }
    return _validate_candidate_row(
        normalized,
        repo=repo,
        context="REST candidate row",
    )


def _validated_stale_row(
    row: dict[str, object],
    *,
    search_label: str,
    index: int,
    repo: str,
    watched: set[str],
) -> tuple[dict[str, object], set[str]] | None:
    """Validate one discovery row and return its stale labels when actionable."""
    normalized = _validate_candidate_row(
        row,
        repo=repo,
        context=f"{search_label} row {index}",
    )
    if normalized["is_pull_request"] or normalized["state"] != "CLOSED":
        return None
    matching_labels = _label_names(normalized["labels"]) & watched
    if search_label in watched:
        matching_labels.add(search_label)
    if not matching_labels:
        return None
    return normalized, matching_labels


def _paginate_rest_closed_issues_for_label(
    *,
    repo: str,
    label: str,
    max_pages: int,
    per_page: int,
    gh_api: Any = None,
) -> tuple[list[dict[str, object]], RestLabelPageMeta]:
    """Read closed issues for one label through bounded REST pagination."""
    if gh_api is None:
        from scripts.dev.gh_issue_rest import _gh_api as gh_api_rest

        gh_api = gh_api_rest

    rows: list[dict[str, object]] = []
    pages_read = 0
    encoded_label = quote(label, safe="")
    base_path = f"repos/{repo}/issues?state=closed&labels={encoded_label}"
    for page in range(1, max_pages + 1):
        path = f"{base_path}&per_page={per_page}&page={page}"
        result = gh_api(path)
        if result.returncode != 0:
            detail = (
                result.stderr or result.stdout or ""
            ).strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"GitHub REST read failed ({path}): {detail}")
        page_rows = _parse_rest_list_payload(result.stdout, context=path)
        rows.extend(_normalize_rest_issue_row(row, repo=repo) for row in page_rows)
        pages_read = page
        if len(page_rows) < per_page:
            break

    truncated = is_likely_truncated(len(rows), limit=max_pages * per_page)
    return rows, RestLabelPageMeta(
        label=label,
        pages_read=pages_read,
        per_page=per_page,
        page_budget=max_pages,
        row_count=len(rows),
        truncated=truncated,
    )


def collect_stale_issues(
    rows_by_label: dict[str, list[dict[str, object]]],
    *,
    repo: str = DEFAULT_REPO,
    watched_labels: tuple[str, ...] = LIVE_STATE_LABELS,
) -> list[StaleIssue]:
    """Aggregate closed issues carrying watched live state labels.

    Returns:
        Stale issue summaries sorted by issue number.
    """
    watched = set(watched_labels) & set(LIVE_STATE_LABELS)
    issue_rows: dict[int, dict[str, object]] = {}
    issue_labels: dict[int, set[str]] = {}

    for search_label, rows in rows_by_label.items():
        if not isinstance(search_label, str) or not search_label.strip():
            raise ValueError("candidate discovery label must be a non-empty string")
        if not isinstance(rows, list):
            raise ValueError(f"candidate rows for {search_label!r} must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"malformed candidate row ({search_label} row {index}): expected an object"
                )
            stale_row = _validated_stale_row(
                row,
                search_label=search_label,
                index=index,
                repo=repo,
                watched=watched,
            )
            if stale_row is None:
                continue
            row, matching_labels = stale_row

            number = row["number"]
            assert type(number) is int

            issue_rows.setdefault(number, row)
            issue_labels.setdefault(number, set()).update(matching_labels)

    stale: list[StaleIssue] = []
    for number in sorted(issue_rows):
        row = issue_rows[number]
        stale.append(
            StaleIssue(
                number=number,
                title=str(row.get("title", "")),
                url=str(row.get("url", "")),
                state=str(row.get("state", "")),
                stale_labels=tuple(sorted(issue_labels[number])),
            )
        )
    return stale


def reconcile_stale_issues(
    *,
    repo: str,
    candidates: list[StaleIssue],
    watched_labels: tuple[str, ...] = LIVE_STATE_LABELS,
    fetch_issue: Any = None,
) -> list[StaleIssue]:
    """Confirm search-discovered candidates against current REST issue records.

    GitHub search results can temporarily retain a removed label while the search index catches up.
    Treat search rows only as candidate discovery and use the current REST labels and state as the
    authority for the final report.
    """
    if fetch_issue is None:
        from scripts.dev.gh_issue_rest import fetch_issue as fetch_issue_rest

        fetch_issue = fetch_issue_rest

    watched = set(watched_labels) & set(LIVE_STATE_LABELS)
    stale: list[StaleIssue] = []
    for candidate in candidates:
        payload = fetch_issue(candidate.number, repo=repo)
        if payload.get("status") != "ok":
            error = payload.get("error", "unknown error")
            raise RuntimeError(f"Failed to read issue {candidate.number}: {error}")
        try:
            validate_issue_identity(payload, repo=repo, number=candidate.number)
            labels = _label_names(payload.get("labels"))
        except ValueError as exc:
            raise RuntimeError(f"Malformed issue {candidate.number} response: {exc}") from exc
        if payload["is_pull_request"]:
            continue
        if payload["state"] != "CLOSED":
            continue

        matching_labels = labels & watched
        if not matching_labels:
            continue

        stale.append(
            StaleIssue(
                number=candidate.number,
                title=str(payload.get("title", candidate.title)),
                url=str(payload.get("url", candidate.url)),
                state=str(payload.get("state", candidate.state)),
                stale_labels=tuple(sorted(matching_labels)),
            )
        )
    return stale


def build_view_command(*, repo: str, number: int) -> list[str]:
    """Build the read-only GitHub CLI command that confirms one issue's state.

    Uses the REST-backed helper to avoid the deprecated ``projectCards`` GraphQL
    field that breaks ``gh issue view --json`` on some CLI versions (issue #5269).
    """
    return [
        sys.executable,
        "-m",
        "scripts.dev.gh_issue_rest",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "number",
        "state",
        "url",
        "is_pull_request",
    ]


def confirm_issue_closed(*, repo: str, number: int) -> bool:
    """Read-then-write guard: confirm an issue is CLOSED and not a pull request.

    Uses the REST-backed helper to avoid the deprecated ``projectCards`` GraphQL
    field that breaks ``gh issue view --json`` on some CLI versions (issue #5269).

    Returns:
        True only when GitHub reports the issue as a closed (non-PR) issue.
    """
    from scripts.dev.gh_issue_rest import fetch_issue

    payload = fetch_issue(number, repo=repo)
    if payload.get("status") != "ok":
        error = payload.get("error", "unknown error")
        raise RuntimeError(f"Failed to read issue {number}: {error}")
    try:
        validate_issue_identity(payload, repo=repo, number=number)
    except ValueError as exc:
        raise RuntimeError(f"Malformed issue {number} response: {exc}") from exc
    if payload["is_pull_request"]:
        return False
    return payload["state"] == "CLOSED"


def _remove_and_validate_label(
    *,
    repo: str,
    remove_label: Any,
    number: int,
    label: str,
) -> None:
    """Remove one label and reject a result naming a different write."""
    if remove_label is None:
        result = gh_pr_label_rest.remove_label(number, label, repo=repo)
    else:
        result = remove_label(number, label)
    try:
        gh_pr_label_rest.validate_result_envelope(
            result,
            action="remove",
            number=number,
            repo=repo,
            label=label,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"REST label remove returned an invalid result for issue {number}, "
            f"label {label!r}: {exc}"
        ) from exc


def _labels_to_remove(
    issue: StaleIssue,
    *,
    repo: str,
    watched: set[str],
) -> list[str]:
    """Validate a stale issue and return only canonical labels eligible for removal."""
    try:
        validate_issue_identity(
            {
                "number": issue.number,
                "state": str(issue.state).upper(),
                "url": issue.url,
                "is_pull_request": False,
            },
            repo=repo,
            number=issue.number,
        )
    except ValueError as exc:
        raise RuntimeError(f"Malformed stale issue {issue.number}: {exc}") from exc
    if any(type(label) is not str or not label.strip() for label in issue.stale_labels):
        raise RuntimeError(f"Malformed stale labels for issue {issue.number}")
    return sorted(set(issue.stale_labels) & watched)


def _fix_one_stale_issue(
    issue: StaleIssue,
    *,
    repo: str,
    watched: set[str],
    confirm_closed: Any,
    remove_label: Any,
) -> dict[str, Any] | None:
    """Reconfirm and fix one stale issue, returning its action when applicable."""
    labels_to_remove = _labels_to_remove(issue, repo=repo, watched=watched)
    if not labels_to_remove:
        return None
    if not confirm_closed(repo=repo, number=issue.number):
        return {
            "number": issue.number,
            "skipped": True,
            "reason": "not_closed",
            "removed_labels": [],
        }
    for label in labels_to_remove:
        _remove_and_validate_label(
            repo=repo,
            remove_label=remove_label,
            number=issue.number,
            label=label,
        )
    return {
        "number": issue.number,
        "skipped": False,
        "removed_labels": labels_to_remove,
    }


def fix_stale_issues(
    *,
    repo: str,
    stale_issues: list[StaleIssue],
    watched_labels: tuple[str, ...] = LIVE_STATE_LABELS,
    confirm_closed: Any = confirm_issue_closed,
    remove_label: Any = None,
) -> list[dict[str, Any]]:
    """Strip live state labels from confirmed-closed issues (read-then-write).

    For each stale issue this re-confirms the issue is CLOSED before removing any label, and only
    removes labels in ``watched_labels`` (the single source of truth ``LIVE_STATE_LABELS``). Missing
    labels are tolerated by gh as a no-op. Returns a per-issue action log.
    """
    unsupported_labels = sorted(set(watched_labels) - set(LIVE_STATE_LABELS))
    if unsupported_labels:
        raise ValueError(
            "fix only supports live state labels: "
            f"{', '.join(LIVE_STATE_LABELS)}; unsupported: {', '.join(unsupported_labels)}"
        )
    watched = set(watched_labels)

    actions: list[dict[str, Any]] = []
    for issue in stale_issues:
        action = _fix_one_stale_issue(
            issue,
            repo=repo,
            watched=watched,
            confirm_closed=confirm_closed,
            remove_label=remove_label,
        )
        if action is None:
            continue
        actions.append(action)
    return actions


def build_label_truncations(
    rows_by_label: dict[str, list[dict[str, object]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return per-label truncation markers for the bounded ``gh search`` results.

    Each closed-issue search is capped at ``--limit`` rows; a result at the cap is
    indistinguishable from a full page, so any ``truncated: true`` marker means the
    hygiene sweep may have missed stale-labelled issues beyond the cap.
    """
    markers: list[dict[str, Any]] = []
    for label, rows in rows_by_label.items():
        row_count = len(rows)
        truncated = is_likely_truncated(row_count, limit=limit)
        markers.append(
            {
                "label": label,
                "truncated": truncated,
                "row_count": row_count,
                "limit": limit,
                "note": (
                    f"gh search issues may be capped: got {row_count} rows at --limit {limit}; "
                    "raise --limit or paginate"
                    if truncated
                    else ""
                ),
            }
        )
    return markers


def build_rest_label_truncations(
    metas: list[RestLabelPageMeta],
) -> list[dict[str, Any]]:
    """Return per-label truncation markers for bounded REST pagination."""
    markers: list[dict[str, Any]] = []
    for meta in metas:
        markers.append(
            {
                "label": meta.label,
                "truncated": meta.truncated,
                "row_count": meta.row_count,
                "limit": meta.page_budget * meta.per_page,
                "pages_read": meta.pages_read,
                "per_page": meta.per_page,
                "page_budget": meta.page_budget,
                "source": "rest",
                "note": (
                    f"closed-issue REST label inventory may be partial: read "
                    f"{meta.row_count} rows in {meta.pages_read}/{meta.page_budget} pages "
                    f"(per_page={meta.per_page}); raise --max-rest-pages"
                    if meta.truncated
                    else ""
                ),
            }
        )
    return markers


def build_report(
    *,
    repo: str,
    checked_labels: tuple[str, ...],
    stale_issues: list[StaleIssue],
    truncations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable audit report."""
    truncations = truncations or []
    return {
        "schema": "closed_state_label_hygiene.v1",
        "ok": not stale_issues,
        "read_only": True,
        "project_writes": False,
        "repo": repo,
        "checked_labels": list(checked_labels),
        "stale_count": len(stale_issues),
        "truncated_any": any(marker.get("truncated") for marker in truncations),
        "truncations": truncations,
        "issues": [issue.to_payload() for issue in stale_issues],
        "failure_summary": {
            "reason": "closed_issues_with_live_state_labels",
            "count": len(stale_issues),
        }
        if stale_issues
        else None,
    }


def fetch_closed_issues_by_label(
    *,
    repo: str,
    labels: tuple[str, ...],
    limit: int,
) -> dict[str, list[dict[str, object]]]:
    """Fetch closed issues for each label with read-only GitHub search commands."""
    rows_by_label: dict[str, list[dict[str, object]]] = {}
    for label in labels:
        command = build_search_command(repo=repo, label=label, limit=limit)
        rows_by_label[label] = _run_search_command(command, repo=repo)
    return rows_by_label


def fetch_closed_issues_by_label_rest(
    *,
    repo: str,
    labels: tuple[str, ...],
    max_pages: int = DEFAULT_MAX_REST_PAGES,
    per_page: int = PER_PAGE,
    gh_api: Any = None,
) -> CandidateDiscoveryResult:
    """Fetch closed issues for each label through bounded REST pagination."""
    if max_pages < 1 or per_page < 1:
        raise ValueError(
            f"REST pagination budgets must be >= 1; got max_pages={max_pages}, per_page={per_page}"
        )
    rows_by_label: dict[str, list[dict[str, object]]] = {}
    metas: list[RestLabelPageMeta] = []
    for label in labels:
        rows, meta = _paginate_rest_closed_issues_for_label(
            repo=repo,
            label=label,
            max_pages=max_pages,
            per_page=per_page,
            gh_api=gh_api,
        )
        rows_by_label[label] = rows
        metas.append(meta)
    return CandidateDiscoveryResult(
        rows_by_label=rows_by_label,
        truncations=build_rest_label_truncations(metas),
        source="rest",
    )


def _is_graphql_fallback_error(error: RuntimeError) -> bool:
    """Return whether a search failure is eligible for the REST fallback."""
    message = str(error).lower()
    if any(marker in message for marker in GRAPHQL_FAIL_CLOSED_MARKERS):
        return False
    return any(marker in message for marker in GRAPHQL_FALLBACK_MARKERS)


def discover_closed_issues_by_label(
    *,
    repo: str,
    labels: tuple[str, ...],
    limit: int,
    max_rest_pages: int = DEFAULT_MAX_REST_PAGES,
) -> CandidateDiscoveryResult:
    """Discover candidate closed issues, falling back to REST when search is unavailable."""
    try:
        rows_by_label = fetch_closed_issues_by_label(repo=repo, labels=labels, limit=limit)
    except RuntimeError as exc:
        if not _is_graphql_fallback_error(exc):
            raise
        return fetch_closed_issues_by_label_rest(
            repo=repo,
            labels=labels,
            max_pages=max_rest_pages,
        )
    return CandidateDiscoveryResult(
        rows_by_label=rows_by_label,
        truncations=build_label_truncations(rows_by_label, limit=limit),
        source="search",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository as OWNER/REPO.")
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="State label to check. May be repeated. Defaults to live state labels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum search results to fetch per label.",
    )
    parser.add_argument(
        "--max-rest-pages",
        type=int,
        default=DEFAULT_MAX_REST_PAGES,
        help=(
            "Maximum REST pages to fetch per label when gh search is unavailable "
            f"(each {PER_PAGE} rows)."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Strip live state labels from the closed issues found. Each issue is re-confirmed "
            "CLOSED before any label is removed (read-then-write). Without this flag the command "
            "stays read-only."
        ),
    )
    return parser


def _dump_json(payload: dict[str, Any]) -> None:
    """Print stable JSON to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    labels = tuple(args.labels) if args.labels else LIVE_STATE_LABELS

    try:
        if args.fix:
            unsupported_labels = sorted(set(labels) - set(LIVE_STATE_LABELS))
            if unsupported_labels:
                raise ValueError(
                    "--fix only supports live state labels: "
                    f"{', '.join(LIVE_STATE_LABELS)}; unsupported: "
                    f"{', '.join(unsupported_labels)}"
                )
        if args.max_rest_pages < 1:
            raise ValueError(f"--max-rest-pages must be >= 1; got {args.max_rest_pages}")
        discovery = discover_closed_issues_by_label(
            repo=args.repo,
            labels=labels,
            limit=args.limit,
            max_rest_pages=args.max_rest_pages,
        )
        candidates = collect_stale_issues(
            discovery.rows_by_label,
            repo=args.repo,
            watched_labels=labels,
        )
        stale_issues = reconcile_stale_issues(
            repo=args.repo,
            candidates=candidates,
            watched_labels=labels,
        )
        report = build_report(
            repo=args.repo,
            checked_labels=labels,
            stale_issues=stale_issues,
            truncations=discovery.truncations,
        )
        report["candidate_discovery_source"] = discovery.source
        if (
            args.fix
            and discovery.truncations
            and any(marker.get("truncated") for marker in discovery.truncations)
        ):
            report["fix_applied"] = False
            report["fix_skipped"] = "candidate discovery was truncated"
        elif args.fix:
            fix_actions = fix_stale_issues(
                repo=args.repo,
                stale_issues=stale_issues,
                watched_labels=labels,
            )
            report["read_only"] = False
            report["fix_applied"] = True
            report["fix_actions"] = fix_actions
            report["ok"] = True
    except (OSError, RuntimeError, ValueError) as exc:
        _dump_json(
            {
                "schema": "closed_state_label_hygiene.v1",
                "ok": False,
                "read_only": not args.fix,
                "project_writes": False,
                "repo": args.repo,
                "checked_labels": list(labels),
                "stale_count": None,
                "issues": [],
                "error": str(exc),
            }
        )
        return 2

    _dump_json(report)
    return 0 if (report["ok"] and not report["truncated_any"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
