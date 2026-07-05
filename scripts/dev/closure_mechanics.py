#!/usr/bin/env python3
"""Apply closure/residual/parent-ledger comments to audit candidates.

This script reads the closure audit report (from ``open_issue_closure_audit.py``)
and posts the appropriate comments to GitHub issues.  It supports a ``--dry-run``
mode (default) that previews actions without mutating anything, and an ``--apply``
mode that posts comments and closes fully-covered issues.

Phase 4 of the #4437 closure-audit implementation plan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

# Comment templates from the issue plan.
CLOSE_FULLY_COVERED_TEMPLATE = """\
Closing as completed by merged PR(s): {pr_links}.

Coverage checked against the issue acceptance criteria: {coverage_clause}. \
No residual scope found in the issue or merged PR discussion."""

RESIDUAL_TEMPLATE = """\
Merged PR(s) reviewed: {pr_links}. Keeping this open because the merged work \
covers only part of the issue.

Residual checklist:
{residual_items}

Dispatch note: next slice should focus only on the checklist above; \
already-merged scope should not be repeated."""

PARENT_LEDGER_TEMPLATE = """\
Merged slice ledger for this parent issue:

Completed slices:
{completed_slices}

Still open / not yet evidenced:
{remaining_items}

Keeping this parent issue open by design because it coordinates multiple slices."""


@dataclass(frozen=True)
class ClosureAction:
    """A planned or executed comment/close action for one issue."""

    issue_number: int
    action_type: str  # "close_fully_covered", "residual", "parent_ledger", "no_action"
    comment_body: str
    should_close: bool
    pr_numbers: tuple[int, ...]


def _parse_pr_links(pr_entries: list[dict[str, object]]) -> str:
    """Format PR entries as Markdown links."""
    links: list[str] = []
    for pr in pr_entries:
        number = pr.get("number")
        if isinstance(number, int):
            links.append(f"#{number}")
    return ", ".join(links) if links else "(none)"


def _build_close_comment(candidate: dict[str, object]) -> str:
    """Build the fully-covered closure comment."""
    pr_entries = candidate.get("title_linked_prs", [])
    assert isinstance(pr_entries, list)
    pr_links = _parse_pr_links(pr_entries)
    title = str(candidate.get("title", ""))
    coverage_clause = f"all deliverables from {title!r} appear present in merged PR(s)"
    return CLOSE_FULLY_COVERED_TEMPLATE.format(
        pr_links=pr_links,
        coverage_clause=coverage_clause,
    )


def _build_residual_comment(candidate: dict[str, object]) -> str:
    """Build the residual checklist comment with placeholder items."""
    pr_entries = candidate.get("title_linked_prs", [])
    assert isinstance(pr_entries, list)
    pr_links = _parse_pr_links(pr_entries)
    residual_items = "- [ ] Review issue acceptance criteria against merged PR diff\n"
    residual_items += "- [ ] Verify validation stated in PR body matches issue scope"
    return RESIDUAL_TEMPLATE.format(
        pr_links=pr_links,
        residual_items=residual_items,
    )


def _build_parent_ledger_comment(candidate: dict[str, object]) -> str:
    """Build the parent/roadmap ledger comment."""
    pr_entries = candidate.get("title_linked_prs", [])
    assert isinstance(pr_entries, list)
    completed_lines: list[str] = []
    for pr in pr_entries:
        pr_num = pr.get("number", "?")
        pr_title = pr.get("title", "")
        completed_lines.append(f"- #{pr_num} -- {pr_title}")
    completed_slices = "\n".join(completed_lines) if completed_lines else "- (none found)"
    remaining_items = "- [ ] Identify remaining child/slice issues\n"
    remaining_items += "- [ ] Verify all acceptance criteria have a tracking issue"
    return PARENT_LEDGER_TEMPLATE.format(
        completed_slices=completed_slices,
        remaining_items=remaining_items,
    )


def classify_and_build_actions(
    report: dict[str, Any], *, close_issues: set[int] | None = None
) -> list[ClosureAction]:
    """Classify audit candidates and build closure actions."""
    actions: list[ClosureAction] = []
    close_set = close_issues or set()
    for candidate in report.get("candidates", []):
        assert isinstance(candidate, dict)
        number = candidate.get("number")
        if not isinstance(number, int):
            continue
        classification = str(candidate.get("classification", ""))
        pr_entries = candidate.get("title_linked_prs", [])
        assert isinstance(pr_entries, list)
        pr_numbers = tuple(
            int(pr["number"]) for pr in pr_entries if isinstance(pr.get("number"), int)
        )

        if number in close_set:
            actions.append(
                ClosureAction(
                    issue_number=number,
                    action_type="close_fully_covered",
                    comment_body=_build_close_comment(candidate),
                    should_close=True,
                    pr_numbers=pr_numbers,
                )
            )
        elif classification == "parent_or_roadmap":
            actions.append(
                ClosureAction(
                    issue_number=number,
                    action_type="parent_ledger",
                    comment_body=_build_parent_ledger_comment(candidate),
                    should_close=False,
                    pr_numbers=pr_numbers,
                )
            )
        else:
            # Default: closure_review_required → treat as residual until
            # a human or deeper checker confirms full coverage.
            actions.append(
                ClosureAction(
                    issue_number=number,
                    action_type="residual",
                    comment_body=_build_residual_comment(candidate),
                    should_close=False,
                    pr_numbers=pr_numbers,
                )
            )
    return actions


def build_comment_command(*, issue_number: int, body: str) -> list[str]:
    """Build a read-only-safe ``gh issue comment`` command."""
    return ["gh", "issue", "comment", str(issue_number), "--body", body]


def build_close_command(*, issue_number: int) -> list[str]:
    """Build a ``gh issue close`` command with ``completed`` reason."""
    return ["gh", "issue", "close", str(issue_number), "--reason", "completed"]


def execute_actions(
    actions: list[ClosureAction], *, repo: str, dry_run: bool = True
) -> dict[str, Any]:
    """Execute or preview closure actions."""
    results: list[dict[str, object]] = []
    for action in actions:
        entry: dict[str, object] = {
            "issue_number": action.issue_number,
            "action_type": action.action_type,
            "should_close": action.should_close,
            "dry_run": dry_run,
            "comment_posted": False,
            "closed": False,
            "error": None,
        }
        if dry_run:
            entry["comment_body_preview"] = action.comment_body[:200]
            results.append(entry)
            continue

        try:
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(action.issue_number),
                    "--repo",
                    repo,
                    "--body",
                    action.comment_body,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            entry["comment_posted"] = True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            entry["error"] = str(exc)
            results.append(entry)
            continue

        if action.should_close:
            try:
                subprocess.run(
                    [
                        "gh",
                        "issue",
                        "close",
                        str(action.issue_number),
                        "--repo",
                        repo,
                        "--reason",
                        "completed",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                entry["closed"] = True
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                entry["error"] = str(exc)

        results.append(entry)

    return {
        "schema": "closure_mechanics.v1",
        "dry_run": dry_run,
        "repo": repo,
        "action_count": len(actions),
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Path to the audit report JSON file. Reads from stdin if omitted.",
    )
    parser.add_argument("--repo", default="ll7/robot_sf_ll7", help="GitHub repository OWNER/REPO.")
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually post comments and close issues (default is dry-run).",
    )
    parser.add_argument(
        "--close-issues",
        type=str,
        default=None,
        help="Comma-separated list of issue numbers to classify as fully covered and close.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the closure mechanics."""
    args = _build_parser().parse_args(argv)
    dry_run = not args.apply

    try:
        if args.report_file:
            with open(args.report_file, encoding="utf-8") as fh:
                report = json.load(fh)
        else:
            report = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    if not isinstance(report, dict) or report.get("schema") != "open_issue_closure_audit.v1":
        print(
            json.dumps({"error": "invalid report schema; expected open_issue_closure_audit.v1"}),
            file=sys.stderr,
        )
        return 2

    close_issues_set = set()
    if args.close_issues:
        for part in args.close_issues.split(","):
            part = part.strip()
            if part.isdigit():
                close_issues_set.add(int(part))
            elif part.startswith("#") and part[1:].isdigit():
                close_issues_set.add(int(part[1:]))
            elif part:
                print(f"Warning: ignored invalid issue number format: {part}", file=sys.stderr)

    actions = classify_and_build_actions(report, close_issues=close_issues_set)
    result = execute_actions(actions, repo=args.repo, dry_run=dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
