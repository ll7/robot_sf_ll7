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
OVERRIDE_PATTERN = re.compile(
    r"^[ \t]*budget-override[ \t]*:[ \t]*(?P<reason>.*)$", re.IGNORECASE | re.MULTILINE
)


def _first_cap(patterns: tuple[re.Pattern[str], ...], body: str) -> int | None:
    """Return the first declared cap in *body*, or None."""
    for pattern in patterns:
        match = pattern.search(body)
        if match is not None:
            return int(match.group(1).replace(",", ""))
    return None


def parse_declared_caps(issue_body: str) -> dict[str, int | None]:
    """Return the declared ``files`` and ``lines`` caps (None when absent)."""
    return {
        "files": _first_cap(FILE_CAP_PATTERNS, issue_body),
        "lines": _first_cap(LINE_CAP_PATTERNS, issue_body),
    }


def has_declared_cap(issue_body: str) -> bool:
    """Return whether *issue_body* declares at least one budget cap."""
    caps = parse_declared_caps(issue_body)
    return caps["files"] is not None or caps["lines"] is not None


def measure_diffstat(numstat_text: str) -> dict[str, int]:
    """Return ``files``/``added``/``deleted``/``net`` from ``git diff --numstat`` output."""
    files = added = deleted = 0
    for line in numstat_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        insertions, deletions = parts[0].strip(), parts[1].strip()
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
    return reason or None


def evaluate_budget(*, issue_body: str, pr_body: str, numstat_text: str) -> dict[str, object]:
    """Evaluate the declared budget and return a stable result payload."""
    caps = parse_declared_caps(issue_body)
    measured = measure_diffstat(numstat_text)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "caps": caps,
        "measured": measured,
        "override_reason": find_override_reason(pr_body),
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
    if caps["lines"] is not None and measured["added"] > caps["lines"]:
        breaches.append(f"{measured['added']} added lines > {caps['lines']}-line cap")
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
