"""Audit closed GitHub issues for stale live state labels.

This command is intentionally read-only. It queries closed issues for the live
implementation queue labels and emits a stable JSON summary. A non-zero exit
means closed issues still carry labels that can confuse agent queue routing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_LIMIT = 1000
LIVE_STATE_LABELS: tuple[str, ...] = ("state:ready", "state:running", "state:blocked")


@dataclass(frozen=True, slots=True)
class StaleClosedIssue:
    """One closed issue carrying at least one live queue state label."""

    number: int
    title: str
    url: str
    stale_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClosedIssueStateLabelAuditReport:
    """Machine-readable result for the closed issue state-label audit."""

    schema: str
    repo: str
    state: str
    live_state_labels: tuple[str, ...]
    read_only: bool
    stale_issue_count: int
    stale_label_counts: dict[str, int]
    stale_issues: tuple[StaleClosedIssue, ...]
    query_plan: tuple[tuple[str, ...], ...]
    mutation_plan: tuple[dict[str, Any], ...]


def build_query_plan(
    *,
    repo: str = DEFAULT_REPO,
    labels: tuple[str, ...] = LIVE_STATE_LABELS,
    limit: int = DEFAULT_LIMIT,
) -> tuple[tuple[str, ...], ...]:
    """Return the read-only gh commands used to find stale closed labels."""
    return tuple(
        (
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "closed",
            "--label",
            label,
            "--json",
            "number,title,labels,url",
            "--limit",
            str(limit),
        )
        for label in labels
    )


def _run_gh_json(command: tuple[str, ...]) -> list[dict[str, Any]]:
    """Run a gh JSON command and return its decoded list payload."""
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError(f"GitHub command did not return a JSON list: {' '.join(command)}")
    if not all(isinstance(item, dict) for item in payload):
        raise RuntimeError(f"GitHub command returned non-object list items: {' '.join(command)}")
    return payload


def _label_names(issue_payload: dict[str, Any]) -> set[str]:
    """Extract label names from a gh issue-list payload item."""
    labels = issue_payload.get("labels")
    if not isinstance(labels, list):
        return set()
    return {
        label["name"]
        for label in labels
        if isinstance(label, dict) and isinstance(label.get("name"), str)
    }


def _issue_number(issue_payload: dict[str, Any]) -> int:
    """Return a validated issue number from a gh issue-list payload item."""
    number = issue_payload.get("number")
    if not isinstance(number, int):
        raise RuntimeError(f"GitHub issue payload is missing an integer number: {issue_payload!r}")
    return number


def audit_closed_issue_state_labels(
    *,
    repo: str = DEFAULT_REPO,
    labels: tuple[str, ...] = LIVE_STATE_LABELS,
    limit: int = DEFAULT_LIMIT,
    runner=None,
) -> ClosedIssueStateLabelAuditReport:
    """Run the read-only closed-issue stale-label audit."""
    runner = _run_gh_json if runner is None else runner
    query_plan = build_query_plan(repo=repo, labels=labels, limit=limit)
    issues_by_number: dict[int, dict[str, Any]] = {}
    stale_labels_by_number: dict[int, set[str]] = {}

    for command in query_plan:
        queried_label = command[command.index("--label") + 1]
        for issue_payload in runner(command):
            issue_labels = _label_names(issue_payload)
            if queried_label not in issue_labels:
                continue
            number = _issue_number(issue_payload)
            issues_by_number[number] = issue_payload
            stale_labels_by_number.setdefault(number, set()).add(queried_label)

    stale_issues = tuple(
        StaleClosedIssue(
            number=number,
            title=str(issues_by_number[number].get("title") or ""),
            url=str(issues_by_number[number].get("url") or ""),
            stale_labels=tuple(
                label for label in labels if label in stale_labels_by_number[number]
            ),
        )
        for number in sorted(stale_labels_by_number)
    )
    stale_label_counts = {
        label: sum(label in issue.stale_labels for issue in stale_issues) for label in labels
    }
    return ClosedIssueStateLabelAuditReport(
        schema="closed_issue_state_label_audit.v1",
        repo=repo,
        state="closed",
        live_state_labels=labels,
        read_only=True,
        stale_issue_count=len(stale_issues),
        stale_label_counts=stale_label_counts,
        stale_issues=stale_issues,
        query_plan=query_plan,
        mutation_plan=(),
    )


def _report_to_json(report: ClosedIssueStateLabelAuditReport) -> str:
    """Serialize the audit report to stable JSON."""
    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum closed issues to fetch per live state label.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        choices=LIVE_STATE_LABELS,
        help="Live state label to audit; may be repeated. Defaults to all live state labels.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    labels = tuple(args.labels) if args.labels else LIVE_STATE_LABELS
    report = audit_closed_issue_state_labels(repo=args.repo, labels=labels, limit=args.limit)
    print(_report_to_json(report), end="")
    return 1 if report.stale_issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
