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
from pathlib import Path
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

FINAL_SUMMARY_TEMPLATE = """\
## Closure audit summary

Scope: open issues with at least one merged title-linked PR reviewed through the #4437 closure mechanics.

Counts:
- closed as fully covered: {closed_count}
- residual annotated, kept open: {residual_count}
- parent ledgers updated, kept open: {parent_count}
- no action / skipped / failed: {skipped_count}

Closed:
{closed_items}

Residual annotated:
{residual_items}

Parent ledgers updated:
{parent_items}

No action / skipped / failed:
{skipped_items}

Boundary: issue hygiene only. No code, queue edits, new issues, benchmark execution, Slurm/GPU submission, or paper/dissertation claim changes were performed by the issue-write pass."""


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


def build_comment_command(*, issue_number: int, body: str, repo: str) -> list[str]:
    """Build a ``gh issue comment`` command scoped to ``repo``.

    ``--repo`` is required so the command targets the intended repository
    regardless of the working directory (e.g. a linked worktree whose default
    remote differs). This is the single source of truth for the comment
    invocation used by :func:`execute_actions`.
    """
    return ["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", body]


def build_close_command(*, issue_number: int, repo: str) -> list[str]:
    """Build a ``gh issue close`` command (``completed`` reason) scoped to ``repo``.

    ``--repo`` is required for the same repository-targeting reason as
    :func:`build_comment_command`. This is the single source of truth for the
    close invocation used by :func:`execute_actions`.
    """
    return ["gh", "issue", "close", str(issue_number), "--repo", repo, "--reason", "completed"]


def _summary_issue_rows(
    results: list[dict[str, object]],
    *,
    action_type: str,
    completed_field: str,
) -> list[str]:
    """Return summary rows for successfully applied actions of one type."""
    rows: list[str] = []
    for entry in results:
        if entry.get("action_type") != action_type:
            continue
        if entry.get(completed_field) is not True:
            continue
        issue_number = entry.get("issue_number")
        if isinstance(issue_number, int):
            rows.append(f"- #{issue_number}")
    return rows


def _skipped_summary_rows(results: list[dict[str, object]]) -> list[str]:
    """Return rows for actions that did not complete their intended mutation."""
    rows: list[str] = []
    for entry in results:
        issue_number = entry.get("issue_number")
        if not isinstance(issue_number, int):
            continue
        action_type = str(entry.get("action_type", "unknown"))
        expected_field = "closed" if action_type == "close_fully_covered" else "comment_posted"
        if action_type == "no_action":
            rows.append(f"- #{issue_number} no action")
        elif entry.get(expected_field) is not True:
            error = entry.get("error")
            reason = f"{action_type} not completed"
            if error:
                reason = f"{reason}: {error}"
            rows.append(f"- #{issue_number} {reason}")
    return rows


def _format_summary_items(rows: list[str]) -> str:
    """Render summary rows with an explicit empty marker."""
    return "\n".join(rows) if rows else "- (none)"


def build_final_summary_comment(execution_result: dict[str, Any]) -> str:
    """Build the final #4437 audit summary comment from execution results.

    The summary only counts mutations that actually completed in the execution result.
    Dry-run previews therefore produce zero applied counts and list planned actions as
    skipped, preventing a preview from being presented as a completed write pass.
    """
    raw_results = execution_result.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    results = [entry for entry in raw_results if isinstance(entry, dict)]

    closed_rows = _summary_issue_rows(
        results,
        action_type="close_fully_covered",
        completed_field="closed",
    )
    residual_rows = _summary_issue_rows(
        results,
        action_type="residual",
        completed_field="comment_posted",
    )
    parent_rows = _summary_issue_rows(
        results,
        action_type="parent_ledger",
        completed_field="comment_posted",
    )
    skipped_rows = _skipped_summary_rows(results)

    return FINAL_SUMMARY_TEMPLATE.format(
        closed_count=len(closed_rows),
        residual_count=len(residual_rows),
        parent_count=len(parent_rows),
        skipped_count=len(skipped_rows),
        closed_items=_format_summary_items(closed_rows),
        residual_items=_format_summary_items(residual_rows),
        parent_items=_format_summary_items(parent_rows),
        skipped_items=_format_summary_items(skipped_rows),
    )


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
                build_comment_command(
                    issue_number=action.issue_number, body=action.comment_body, repo=repo
                ),
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
                    build_close_command(issue_number=action.issue_number, repo=repo),
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
    parser.add_argument(
        "--summary-comment-file",
        type=str,
        default=None,
        help="Write final #4437 summary comment Markdown from the execution result.",
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
    if args.summary_comment_file:
        summary_comment = build_final_summary_comment(result)
        Path(args.summary_comment_file).write_text(summary_comment + "\n", encoding="utf-8")
        result["summary_comment_file"] = args.summary_comment_file
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
