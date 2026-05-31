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

DEFAULT_REPO = "ll7/robot_sf_ll7"
LIVE_STATE_LABELS = ("state:ready", "state:running", "state:blocked")
JSON_FIELDS = "number,title,url,state,isPullRequest,labels"


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
            if row.get("isPullRequest") is True:
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
    except (OSError, RuntimeError, ValueError) as exc:
        _dump_json(
            {
                "schema": "closed_state_label_hygiene.v1",
                "ok": False,
                "read_only": True,
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
