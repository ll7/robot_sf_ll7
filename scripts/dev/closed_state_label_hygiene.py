#!/usr/bin/env python3
"""Audit closed issues for stale live state labels.

The command is intentionally read-only: it searches GitHub issues and reports any closed issue that
still carries labels used for active implementation routing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

DEFAULT_REPO = "ll7/robot_sf_ll7"
LIVE_STATE_LABELS = ("state:ready", "state:running", "state:blocked")
JSON_FIELDS = "number,title,url,state,labels"


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


def _label_names(raw_labels: object) -> set[str]:
    """Extract label names from GitHub CLI label payloads."""
    if not isinstance(raw_labels, list):
        return set()

    names: set[str] = set()
    for label in raw_labels:
        if isinstance(label, str):
            names.add(label)
        elif isinstance(label, dict) and isinstance(label.get("name"), str):
            names.add(label["name"])
    return names


def _is_pull_request_url(raw_url: object) -> bool:
    """Return True for GitHub pull-request URLs returned by issue search/view commands."""
    if not isinstance(raw_url, str):
        return False

    path_parts = [part for part in urlsplit(raw_url).path.split("/") if part]
    if len(path_parts) != 4:
        return False
    owner, repo, resource, number = path_parts
    return bool(owner and repo and resource == "pull" and number.isdecimal())


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


def _run_search_command(command: list[str]) -> list[dict[str, object]]:
    """Run one read-only GitHub search command and return object rows."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' was not found; install gh or add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f": {stderr}" if stderr else ""
        raise RuntimeError(f"GitHub CLI command failed ({' '.join(command)}){details}") from exc

    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse GitHub CLI JSON output ({' '.join(command)}): {exc.msg}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list from {' '.join(command)}")
    return [row for row in payload if isinstance(row, dict)]


def collect_stale_issues(
    rows_by_label: dict[str, list[dict[str, object]]],
    *,
    watched_labels: tuple[str, ...] = LIVE_STATE_LABELS,
) -> list[StaleIssue]:
    """Aggregate closed issues carrying watched live state labels.

    Returns:
        Stale issue summaries sorted by issue number.
    """
    watched = set(watched_labels)
    issue_rows: dict[int, dict[str, object]] = {}
    issue_labels: dict[int, set[str]] = {}

    for search_label, rows in rows_by_label.items():
        for row in rows:
            if _is_pull_request_url(row.get("url")):
                continue
            if str(row.get("state", "")).lower() != "closed":
                continue

            try:
                number = int(row["number"])
            except (KeyError, TypeError, ValueError):
                continue

            matching_labels = _label_names(row.get("labels")) & watched
            if search_label in watched:
                matching_labels.add(search_label)
            if not matching_labels:
                continue

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


def build_view_command(*, repo: str, number: int) -> list[str]:
    """Build the read-only GitHub CLI command that confirms one issue's state."""
    return [
        "gh",
        "issue",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "number,state,url",
    ]


def build_remove_label_command(*, repo: str, number: int, label: str) -> list[str]:
    """Build the GitHub CLI command that removes one label from one issue."""
    return [
        "gh",
        "issue",
        "edit",
        str(number),
        "--repo",
        repo,
        "--remove-label",
        label,
    ]


def _run_gh_command(command: list[str]) -> str:
    """Run a GitHub CLI command and return stdout, mapping failures to RuntimeError."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' was not found; install gh or add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f": {stderr}" if stderr else ""
        raise RuntimeError(f"GitHub CLI command failed ({' '.join(command)}){details}") from exc
    return result.stdout or ""


def confirm_issue_closed(*, repo: str, number: int) -> bool:
    """Read-then-write guard: confirm an issue is CLOSED and not a pull request.

    Returns:
        True only when GitHub reports the issue as a closed (non-PR) issue.
    """
    stdout = _run_gh_command(build_view_command(repo=repo, number=number))
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse GitHub CLI JSON output for issue {number}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from gh issue view for issue {number}")
    if _is_pull_request_url(payload.get("url")):
        return False
    return str(payload.get("state", "")).lower() == "closed"


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
    watched = set(watched_labels)

    def _default_remove(number: int, label: str) -> None:
        _run_gh_command(build_remove_label_command(repo=repo, number=number, label=label))

    remove = remove_label or _default_remove

    actions: list[dict[str, Any]] = []
    for issue in stale_issues:
        labels_to_remove = sorted(set(issue.stale_labels) & watched)
        if not labels_to_remove:
            continue
        if not confirm_closed(repo=repo, number=issue.number):
            actions.append(
                {
                    "number": issue.number,
                    "skipped": True,
                    "reason": "not_closed",
                    "removed_labels": [],
                }
            )
            continue
        for label in labels_to_remove:
            remove(issue.number, label)
        actions.append(
            {
                "number": issue.number,
                "skipped": False,
                "removed_labels": labels_to_remove,
            }
        )
    return actions


def build_report(
    *,
    repo: str,
    checked_labels: tuple[str, ...],
    stale_issues: list[StaleIssue],
) -> dict[str, Any]:
    """Build the machine-readable audit report."""
    return {
        "schema": "closed_state_label_hygiene.v1",
        "ok": not stale_issues,
        "read_only": True,
        "project_writes": False,
        "repo": repo,
        "checked_labels": list(checked_labels),
        "stale_count": len(stale_issues),
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
        rows_by_label[label] = _run_search_command(command)
    return rows_by_label


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
        rows_by_label = fetch_closed_issues_by_label(
            repo=args.repo,
            labels=labels,
            limit=args.limit,
        )
        stale_issues = collect_stale_issues(rows_by_label, watched_labels=labels)
        report = build_report(repo=args.repo, checked_labels=labels, stale_issues=stale_issues)
        if args.fix:
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
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
