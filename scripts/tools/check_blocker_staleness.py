"""Detect open issues whose numbered blocker references have gone stale.

The tool is intentionally read-only. It scans open issue bodies for numbered
references in blocker-style sections, resolves those referenced issue states,
and reports issues that are fully or partially unblocked.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_LIMIT = 500
DEPENDENCY_SECTION_TITLES = frozenset(
    {
        "blocked by",
        "depends on",
    }
)
HEADING_RE = re.compile(r"^(?P<marker>#{2,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
ISSUE_REF_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#(?P<number>[1-9]\d*)\b")
STRIKETHROUGH_RE = re.compile(r"~~.*?~~", re.DOTALL)


class BlockerClassification(StrEnum):
    """Classification for an issue with parsed numbered blockers."""

    FULLY_UNBLOCKED = "fully_unblocked"
    PARTIALLY_UNBLOCKED = "partially_unblocked"
    STILL_BLOCKED = "still_blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IssueRef:
    """Minimal issue payload needed by the staleness classifier."""

    number: int
    title: str = ""
    state: str = "UNKNOWN"
    url: str = ""


@dataclass(frozen=True, slots=True)
class BlockerAuditResult:
    """Structured staleness result for one open issue."""

    issue: IssueRef
    blocker_numbers: tuple[int, ...]
    closed_blockers: tuple[int, ...]
    open_blockers: tuple[int, ...]
    unknown_blockers: tuple[int, ...]
    classification: BlockerClassification


def _normalize_heading(title: str) -> str:
    """Normalize a Markdown heading for blocker-section matching."""

    cleaned = re.sub(r"[:：]+$", "", title.strip())
    cleaned = re.sub(r"[-_]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).casefold()


def _dependency_section_spans(body: str) -> list[tuple[int, int]]:
    """Return body spans under recognized blocker/dependency sections."""

    matches = list(HEADING_RE.finditer(body))
    spans: list[tuple[int, int]] = []
    for index, match in enumerate(matches):
        if _normalize_heading(match.group("title")) not in DEPENDENCY_SECTION_TITLES:
            continue
        heading_level = len(match.group("marker"))
        section_start = match.end()
        section_end = len(body)
        for next_match in matches[index + 1 :]:
            if len(next_match.group("marker")) <= heading_level:
                section_end = next_match.start()
                break
        spans.append((section_start, section_end))
    return spans


def extract_blocker_numbers(body: str) -> tuple[int, ...]:
    """Extract unique non-struck-through issue numbers from dependency sections."""

    blocker_numbers: list[int] = []
    seen: set[int] = set()
    for start, end in _dependency_section_spans(body):
        section = STRIKETHROUGH_RE.sub("", body[start:end])
        for match in ISSUE_REF_RE.finditer(section):
            number = int(match.group("number"))
            if number not in seen:
                blocker_numbers.append(number)
                seen.add(number)
    return tuple(blocker_numbers)


def classify_blockers(
    issue: IssueRef,
    blocker_numbers: tuple[int, ...],
    blocker_states: dict[int, str],
) -> BlockerAuditResult:
    """Classify one issue from parsed blocker numbers and resolved states."""

    closed: list[int] = []
    open_: list[int] = []
    unknown: list[int] = []
    for number in blocker_numbers:
        state = blocker_states.get(number, "UNKNOWN").upper()
        if state == "CLOSED":
            closed.append(number)
        elif state == "OPEN":
            open_.append(number)
        else:
            unknown.append(number)

    if unknown:
        classification = BlockerClassification.UNKNOWN
    elif blocker_numbers and len(closed) == len(blocker_numbers):
        classification = BlockerClassification.FULLY_UNBLOCKED
    elif closed:
        classification = BlockerClassification.PARTIALLY_UNBLOCKED
    else:
        classification = BlockerClassification.STILL_BLOCKED

    return BlockerAuditResult(
        issue=issue,
        blocker_numbers=blocker_numbers,
        closed_blockers=tuple(closed),
        open_blockers=tuple(open_),
        unknown_blockers=tuple(unknown),
        classification=classification,
    )


def audit_issue_bodies(
    issues: list[IssueRef],
    issue_bodies: dict[int, str],
    blocker_states: dict[int, str],
) -> list[BlockerAuditResult]:
    """Audit issue bodies using already-resolved blocker states."""

    results: list[BlockerAuditResult] = []
    for issue in issues:
        blocker_numbers = extract_blocker_numbers(issue_bodies.get(issue.number, ""))
        if not blocker_numbers:
            continue
        results.append(classify_blockers(issue, blocker_numbers, blocker_states))
    return results


def _gh_json(args: list[str], *, timeout: int = 60) -> Any:
    """Run a GitHub CLI command that returns JSON."""

    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"gh {' '.join(args)} failed with {result.returncode}: {detail}")
    return json.loads(result.stdout or "null")


def fetch_open_issues(
    *, repo: str, label: str | None, limit: int
) -> tuple[list[IssueRef], dict[int, str]]:
    """Fetch open issues and bodies through the GitHub CLI."""

    args = [
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        str(limit),
        "--json",
        "number,title,body,state,url",
    ]
    if label:
        args.extend(["--label", label])
    payload = _gh_json(args)

    issues: list[IssueRef] = []
    bodies: dict[int, str] = {}
    for row in payload:
        number = int(row["number"])
        issues.append(
            IssueRef(
                number=number,
                title=str(row.get("title") or ""),
                state=str(row.get("state") or "UNKNOWN"),
                url=str(row.get("url") or ""),
            )
        )
        bodies[number] = str(row.get("body") or "")
    return issues, bodies


def fetch_issue_states(numbers: set[int], *, repo: str) -> dict[int, str]:
    """Resolve issue states for referenced blocker numbers."""

    states: dict[int, str] = {}
    for number in sorted(numbers):
        payload = _gh_json(
            [
                "issue",
                "view",
                str(number),
                "--repo",
                repo,
                "--json",
                "number,state",
            ]
        )
        states[int(payload["number"])] = str(payload.get("state") or "UNKNOWN")
    return states


def build_summary(results: list[BlockerAuditResult]) -> dict[str, Any]:
    """Build the stable JSON summary emitted by the CLI."""

    grouped = {
        classification.value: [
            asdict(result) for result in results if result.classification == classification
        ]
        for classification in BlockerClassification
    }
    return {
        "schema_version": "blocker_staleness_report.v1",
        "counts": {key: len(value) for key, value in grouped.items()},
        "results": grouped,
    }


def _format_issue_line(result: BlockerAuditResult) -> str:
    """Return one compact text report line."""

    issue = result.issue
    parts = [
        f"#{issue.number}",
        issue.title,
        f"blockers={list(result.blocker_numbers)}",
        f"closed={list(result.closed_blockers)}",
        f"open={list(result.open_blockers)}",
    ]
    if result.unknown_blockers:
        parts.append(f"unknown={list(result.unknown_blockers)}")
    if issue.url:
        parts.append(issue.url)
    return " | ".join(part for part in parts if part)


def render_text(summary: dict[str, Any]) -> str:
    """Render a human-readable report from a summary payload."""

    lines = ["# Blocker Staleness Report", ""]
    for key in (
        BlockerClassification.FULLY_UNBLOCKED.value,
        BlockerClassification.PARTIALLY_UNBLOCKED.value,
        BlockerClassification.UNKNOWN.value,
        BlockerClassification.STILL_BLOCKED.value,
    ):
        rows = summary["results"][key]
        lines.extend([f"## {key} ({len(rows)})", ""])
        if not rows:
            lines.append("- none")
        else:
            for row in rows:
                result = BlockerAuditResult(
                    issue=IssueRef(**row["issue"]),
                    blocker_numbers=tuple(row["blocker_numbers"]),
                    closed_blockers=tuple(row["closed_blockers"]),
                    open_blockers=tuple(row["open_blockers"]),
                    unknown_blockers=tuple(row["unknown_blockers"]),
                    classification=BlockerClassification(row["classification"]),
                )
                lines.append(f"- {_format_issue_line(result)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the blocker-staleness detector."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository in OWNER/REPO form.")
    parser.add_argument("--label", help="Only scan open issues with this label.")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Maximum open issues to fetch."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit 0 after reporting; useful for dry-run or dashboards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --report-only. The tool never mutates GitHub state.",
    )
    args = parser.parse_args(argv)

    issues, bodies = fetch_open_issues(repo=args.repo, label=args.label, limit=args.limit)
    blocker_numbers = {
        number for body in bodies.values() for number in extract_blocker_numbers(body)
    }
    blocker_states = fetch_issue_states(blocker_numbers, repo=args.repo)
    summary = build_summary(audit_issue_bodies(issues, bodies, blocker_states))

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_text(summary), end="")

    fully_unblocked = summary["counts"][BlockerClassification.FULLY_UNBLOCKED.value]
    if fully_unblocked and not (args.report_only or args.dry_run):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
