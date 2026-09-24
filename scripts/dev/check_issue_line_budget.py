#!/usr/bin/env python3
"""Evaluate an issue's declared line/file budget against a PR diffstat (issue #9094).

Compute-window child issues declare reviewability budgets such as "Maximum 10
files and 800 net new lines", but overruns were merged with a prose disclosure
instead of an enforced exception. This module turns the cap into a deterministic
check:

- parse the first declared file cap and line cap from the linked issue body;
- measure the pull request's changed files and added lines from
  ``git diff --numstat`` output;
- pass when within budget, or when the PR body carries an explicit
  ``budget-override: <reason>`` line with a non-empty reason;
- otherwise report ``over_budget`` with the exact breach.

No cap declared means ``no_declared_cap`` and the check is inert, so issues that
never opted into a budget keep their current behavior.

Usage::

    uv run python scripts/dev/check_issue_line_budget.py \
      --issue-body-file /tmp/issue.md --pr-body-file /tmp/pr.md --numstat-file /tmp/numstat.txt --json

Exit codes:
    0 - within budget, override applied, or no declared cap
    1 - over budget without a reasoned override
    2 - usage or input error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "issue_line_budget.v1"
STATUS_NO_CAP = "no_declared_cap"
STATUS_WITHIN = "within_budget"
STATUS_OVERRIDE = "override_applied"
STATUS_OVER = "over_budget"
STATUS_INVALID = "invalid_cap"
_CAP_KEYWORDS = r"(?:max(?:imum)?|cap(?:ped)?(?:\s+at)?|budget\s+of|no\s+more\s+than|at\s+most)"
LINE_CAP_PATTERNS = (
    re.compile(rf"{_CAP_KEYWORDS}\D{{0,15}}?([\d,]+)\s*(?:net\s+new\s+)?lines?\b", re.IGNORECASE),
    re.compile(r"\band\s+([\d,]+)\s*(?:net\s+new\s+)?lines?\b", re.IGNORECASE),
    re.compile(r"([\d,]+)[- ]line\s+(?:cap|budget)\b", re.IGNORECASE),
)
FILE_CAP_PATTERNS = (
    re.compile(rf"{_CAP_KEYWORDS}\D{{0,15}}?([\d,]+)\s*files?\b", re.IGNORECASE),
    re.compile(r"\band\s+([\d,]+)\s*files?\b", re.IGNORECASE),
    re.compile(r"([\d,]+)[- ]file\s+(?:cap|budget)\b", re.IGNORECASE),
    re.compile(r"[\d,]+/([\d,]+)\s*files?\b", re.IGNORECASE),
)
CONTEXT_FILE_PATTERNS = (
    re.compile(rf"{_CAP_KEYWORDS}\D{{0,15}}?([\d,]+)\s*files?\b", re.IGNORECASE),
    re.compile(r"\band\s+([\d,]+)\s*files?\b", re.IGNORECASE),
    re.compile(r"([\d,]+)[- ]file\s+(?:cap|budget)\b", re.IGNORECASE),
    re.compile(r"[\d,]+/([\d,]+)\s*files?\b", re.IGNORECASE),
    re.compile(r"\b([\d,]+)\s*files?\b", re.IGNORECASE),
)
CONTEXT_TOKEN_PATTERNS = (
    re.compile(rf"{_CAP_KEYWORDS}\D{{0,15}}?([\d,]+)\s*(?:input\s+)?tokens?\b", re.IGNORECASE),
    re.compile(r"\band\s+([\d,]+)\s*(?:input\s+)?tokens?\b", re.IGNORECASE),
    re.compile(r"([\d,]+)[- ]token\s+(?:cap|budget)\b", re.IGNORECASE),
    re.compile(r"\b([\d,]+)\s*(?:input\s+)?tokens?\b", re.IGNORECASE),
)
_CONTEXT_HEADING_PATTERN = re.compile(
    r"^#+\s+.*(?:context\s+budget|context-reading\s+budget|reading[- ]context|review[- ]context).*",
    re.IGNORECASE,
)
_ANY_HEADING_PATTERN = re.compile(r"^#+\s+", re.IGNORECASE)
_CONTEXT_LINE_PATTERN = re.compile(
    r"(?i)(?:"
    r"^\s*(?:[-*•]|\d+\.)?\s*(?:(?:review[- ]?)?context(?:[- ]reading)?\s+budget|reading[- ]context\s+budget)\b"
    r"|^\s*(?:[-*•]|\d+\.)?\s*(?:review[- ]?)?context\s*:\s*.*?(?:files?|tokens?)\b"
    r"|\b(?:review[- ]?)?context\s+budget\b"
    r"|\b[\d,]+\s*(?:input\s+)?tokens?\b"
    r")"
)
_CONTEXT_BLOCK_HEADER_PATTERN = re.compile(
    r"^\s*(?:[-*•]|\d+\.)?\s*(?:(?:review[- ]?)?context(?:[- ]reading)?\s+budget|reading[- ]context\s+budget|context)\s*:\s*$",
    re.IGNORECASE,
)
OVERRIDE_PATTERN = re.compile(
    r"^[ \t]*budget-override[ \t]*:[ \t]*(?P<reason>.*)$", re.IGNORECASE | re.MULTILINE
)


class BudgetParseError(ValueError):
    """Raised when a matched budget number cannot be represented safely."""


class DiffstatParseError(ValueError):
    """Raised when numstat text contains malformed or unparseable rows."""


def _first_cap(patterns: tuple[re.Pattern[str], ...], body: str) -> int | None:
    """Return the first declared cap in *body*, or None."""
    for pattern in patterns:
        match = pattern.search(body)
        if match is not None:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError as error:
                raise BudgetParseError("declared budget cap is not a valid integer") from error
    return None


def split_budget_text(issue_body: str) -> tuple[str, str]:
    """Split *issue_body* into diff-cap text and context-budget text (issue #9641).

    Separates reviewer/agent reading-context budgets (files/tokens) from
    pull-request diff caps (files/lines) so that reading-context budgets
    do not trigger false-positive changed-file diff cap violations.
    """
    diff_lines: list[str] = []
    context_lines: list[str] = []
    in_context_section = False
    in_context_block = False

    for line in issue_body.splitlines():
        if _CONTEXT_HEADING_PATTERN.match(line):
            in_context_section = True
            in_context_block = False
            context_lines.append(line)
            continue
        if _ANY_HEADING_PATTERN.match(line):
            in_context_section = False
            in_context_block = False

        if in_context_section:
            context_lines.append(line)
            continue

        if _CONTEXT_BLOCK_HEADER_PATTERN.match(line):
            in_context_block = True
            context_lines.append(line)
            continue

        if in_context_block:
            if line.startswith(("  ", "\t")) or not line.strip():
                context_lines.append(line)
                continue
            in_context_block = False

        if _CONTEXT_LINE_PATTERN.search(line):
            context_lines.append(line)
        else:
            diff_lines.append(line)

    return "\n".join(diff_lines), "\n".join(context_lines)


def parse_declared_caps(issue_body: str) -> dict[str, int | None]:
    """Return the declared PR diff ``files`` and ``lines`` caps (None when absent).

    Reading-context budget declarations are excluded so that agent context
    limits are not misclassified as PR diff caps (issue #9641).
    """
    diff_text, _ = split_budget_text(issue_body)
    return {
        "files": _first_cap(FILE_CAP_PATTERNS, diff_text),
        "lines": _first_cap(LINE_CAP_PATTERNS, diff_text),
    }


def parse_context_budget(issue_body: str) -> dict[str, int | None]:
    """Return the declared reading-context ``files`` and ``tokens`` budgets (None when absent)."""
    _, context_text = split_budget_text(issue_body)
    return {
        "files": _first_cap(CONTEXT_FILE_PATTERNS, context_text),
        "tokens": _first_cap(CONTEXT_TOKEN_PATTERNS, context_text),
    }


def parse_budget_dimensions(issue_body: str) -> dict[str, dict[str, int | None]]:
    """Return distinct classifications for declared budget dimensions (issue #9641).

    Returns a mapping with two dimension categories:
    - ``"diff"``: PR changed-file and net-new-line caps (``files`` and ``lines``).
    - ``"context"``: Agent/reviewer reading-context budgets (``files`` and ``tokens``).
    """
    return {
        "diff": parse_declared_caps(issue_body),
        "context": parse_context_budget(issue_body),
    }


def has_declared_cap(issue_body: str) -> bool:
    """Return whether *issue_body* declares at least one budget cap."""
    try:
        caps = parse_declared_caps(issue_body)
    except BudgetParseError:
        return True
    return caps["files"] is not None or caps["lines"] is not None


def measure_diffstat(numstat_text: str) -> dict[str, int]:
    """Return ``files``/``added``/``deleted``/``net`` from ``git diff --numstat`` output."""
    files = added = deleted = 0
    for line in numstat_text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise DiffstatParseError(
                f"malformed numstat line (expected tab-separated fields): {line!r}"
            )
        insertions, deletions = parts[0].strip(), parts[1].strip()
        path = "\t".join(parts[2:]).strip()
        if not path:
            raise DiffstatParseError(f"malformed numstat line (empty file path): {line!r}")
        if not (insertions.isdigit() or insertions == "-"):
            raise DiffstatParseError(
                f"malformed numstat line (invalid insertions {insertions!r}): {line!r}"
            )
        if not (deletions.isdigit() or deletions == "-"):
            raise DiffstatParseError(
                f"malformed numstat line (invalid deletions {deletions!r}): {line!r}"
            )
        files += 1
        if insertions.isdigit() and deletions.isdigit():
            added += int(insertions)
            deleted += int(deletions)
    return {"files": files, "added": added, "deleted": deleted, "net": max(added - deleted, 0)}


def find_override_reason(pr_body: str) -> str | None:
    """Return the non-empty ``budget-override:`` reason from *pr_body*, if any."""
    match = OVERRIDE_PATTERN.search(pr_body)
    if match is None:
        return None
    reason = match.group("reason").strip()
    return None if not reason or reason.casefold() in {"<reason>", "reason"} else reason


def evaluate_budget(*, issue_body: str, pr_body: str, numstat_text: str) -> dict[str, object]:
    """Evaluate the declared budget and return a stable result payload."""
    override_reason = find_override_reason(pr_body)
    try:
        measured = measure_diffstat(numstat_text)
    except DiffstatParseError as error:
        message = str(error)
        return {
            "schema": SCHEMA,
            "caps": {"files": None, "lines": None},
            "context_budget": {"files": None, "tokens": None},
            "dimensions": {
                "diff": {"files": None, "lines": None},
                "context": {"files": None, "tokens": None},
            },
            "measured": {"files": 0, "added": 0, "deleted": 0, "net": 0},
            "override_reason": override_reason,
            "status": STATUS_INVALID,
            "ok": False,
            "message": message,
            "breaches": [message],
        }
    try:
        caps = parse_declared_caps(issue_body)
        context_budget = parse_context_budget(issue_body)
    except BudgetParseError as error:
        message = str(error)
        return {
            "schema": SCHEMA,
            "caps": {"files": None, "lines": None},
            "context_budget": {"files": None, "tokens": None},
            "dimensions": {
                "diff": {"files": None, "lines": None},
                "context": {"files": None, "tokens": None},
            },
            "measured": measured,
            "override_reason": override_reason,
            "status": STATUS_INVALID,
            "ok": False,
            "message": message,
            "breaches": [message],
        }
    result: dict[str, object] = {
        "schema": SCHEMA,
        "caps": caps,
        "context_budget": context_budget,
        "dimensions": {
            "diff": caps,
            "context": context_budget,
        },
        "measured": measured,
        "override_reason": override_reason,
    }
    if caps["files"] is None and caps["lines"] is None:
        result["status"] = STATUS_NO_CAP
        result["ok"] = True
        result["message"] = "linked issue declares no line/file budget"
        result["breaches"] = []
        return result

    breaches: list[str] = []
    if caps["files"] is not None and measured["files"] > caps["files"]:
        breaches.append(f"{measured['files']} files > {caps['files']}-file cap")
    if caps["lines"] is not None and measured["net"] > caps["lines"]:
        breaches.append(f"{measured['net']} net new lines > {caps['lines']}-line cap")
    result["breaches"] = breaches
    if not breaches:
        result["status"] = STATUS_WITHIN
        result["ok"] = True
        result["message"] = "within the declared budget"
    elif result["override_reason"] is not None:
        result["status"] = STATUS_OVERRIDE
        result["ok"] = True
        result["message"] = (
            f"budget exceeded ({'; '.join(breaches)}) but overridden: {result['override_reason']}"
        )
    else:
        result["status"] = STATUS_OVER
        result["ok"] = False
        result["message"] = "; ".join(breaches)
    return result


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-body-file", type=Path, required=True)
    parser.add_argument("--pr-body-file", type=Path, required=True)
    parser.add_argument("--numstat-file", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON (default).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_budget(
            issue_body=_read(args.issue_body_file),
            pr_body=_read(args.pr_body_file),
            numstat_text=_read(args.numstat_file),
        )
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
