#!/usr/bin/env python3
"""Check that declared deferred PR work has an explicit follow-up disposition.

The check is intentionally bounded for agent PR-readiness runs.  It reads a PR
body from an explicit file or GitHub pull-request event payload, scans the
default template's Follow-Up Issues section, and prints a compact report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ISSUE_RE = re.compile(r"(?:#|/issues/)(\d+)\b")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
EMPTY_OR_PLACEHOLDER = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "no",
    "not applicable",
    "`#<id>`",
    "#<id>",
    "<id>",
}


@dataclass(frozen=True)
class FollowupReport:
    """Compact PR follow-up disposition report."""

    status: str
    source: str
    deferred_work: str
    linked_issues: tuple[str, ...]
    explicit_no_issue_reason: str
    issue_state_errors: tuple[str, ...]
    message: str


def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _strip_bullet_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", text).strip()


def _clean_value(text: str) -> str:
    return _strip_bullet_prefix(text).strip().strip("`").strip()


def _is_empty_or_placeholder(text: str) -> bool:
    value = _clean_value(text).lower()
    return value in EMPTY_OR_PLACEHOLDER or value.startswith("#<")


def _is_no_issue_reason(text: str) -> bool:
    value = _clean_value(text).lower()
    if value in {"none", "no", "n/a", "na", "not applicable"}:
        return True
    return value.startswith(("none ", "none -", "none:", "na ", "na -", "n/a ", "n/a -"))


def _extract_section(body: str, heading: str) -> str:
    target = _normalize_label(heading)
    matches = list(HEADING_RE.finditer(body))
    for index, match in enumerate(matches):
        level, title = match.groups()
        if _normalize_label(title) != target:
            continue
        section_end = len(body)
        for next_match in matches[index + 1 :]:
            if len(next_match.group(1)) <= len(level):
                section_end = next_match.start()
                break
        return body[match.end() : section_end].strip()
    return ""


def _value_after_label(section: str, label: str) -> str:
    target = _normalize_label(label)
    lines = section.splitlines()
    known_labels = {
        "deferred work",
        "issues opened for follow up",
        "issues opened for followup",
        "follow up issues",
    }
    for index, line in enumerate(lines):
        item = _strip_bullet_prefix(line)
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        if _normalize_label(key) != target:
            continue
        value = value.strip()
        if value:
            return value
        continuation: list[str] = []
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped:
                continue
            clean = _strip_bullet_prefix(stripped)
            if ":" in clean:
                possible_key = clean.split(":", 1)[0]
                if _normalize_label(possible_key) in known_labels:
                    break
            if stripped.startswith("#"):
                break
            continuation.append(clean)
        return " ".join(continuation).strip()
    return ""


def _linked_issues(text: str) -> tuple[str, ...]:
    return tuple(f"#{match}" for match in sorted(set(ISSUE_RE.findall(text)), key=int))


def _verify_open_issues(issues: tuple[str, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    for issue in issues:
        number = issue.removeprefix("#")
        try:
            result = subprocess.run(
                ["gh", "issue", "view", number, "--json", "state", "--jq", ".state"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except FileNotFoundError:
            errors.append(f"{issue}: unable to verify open state (gh CLI not found)")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            errors.append(f"{issue}: unable to verify open state ({detail})")
            continue
        state = result.stdout.strip().upper()
        if state != "OPEN":
            errors.append(f"{issue}: state is {state or 'unknown'}, expected OPEN")
    return tuple(errors)


def analyze_body(body: str, *, source: str, require_open_issues: bool = False) -> FollowupReport:
    """Return a compact follow-up report for a PR body."""
    section = _extract_section(body, "Follow-Up Issues")
    deferred = _value_after_label(section, "Deferred work") if section else ""
    issue_value = _value_after_label(section, "Issues opened for follow-up") if section else ""
    linked_issues = _linked_issues(f"{section}\n{issue_value}")
    issue_state_errors = _verify_open_issues(linked_issues) if require_open_issues else ()

    if not section:
        return FollowupReport(
            status="missing_section",
            source=source,
            deferred_work="",
            linked_issues=(),
            explicit_no_issue_reason="",
            issue_state_errors=(),
            message="Follow-Up Issues section not found.",
        )
    if _is_empty_or_placeholder(deferred):
        return FollowupReport(
            status="ok",
            source=source,
            deferred_work="",
            linked_issues=linked_issues,
            explicit_no_issue_reason="",
            issue_state_errors=issue_state_errors,
            message="No deferred work declared.",
        )
    if linked_issues:
        if issue_state_errors:
            return FollowupReport(
                status="issue_state_error",
                source=source,
                deferred_work=deferred,
                linked_issues=linked_issues,
                explicit_no_issue_reason="",
                issue_state_errors=issue_state_errors,
                message="Deferred work follow-up issue state could not be verified open.",
            )
        return FollowupReport(
            status="ok",
            source=source,
            deferred_work=deferred,
            linked_issues=linked_issues,
            explicit_no_issue_reason="",
            issue_state_errors=(),
            message="Deferred work has linked follow-up issue(s).",
        )
    if _is_no_issue_reason(issue_value):
        return FollowupReport(
            status="ok",
            source=source,
            deferred_work=deferred,
            linked_issues=(),
            explicit_no_issue_reason=issue_value,
            issue_state_errors=(),
            message="Deferred work explicitly states no follow-up issue is needed.",
        )
    return FollowupReport(
        status="missing_followup",
        source=source,
        deferred_work=deferred,
        linked_issues=(),
        explicit_no_issue_reason="",
        issue_state_errors=(),
        message="Deferred work is declared without a linked issue or explicit NA/none reason.",
    )


def _read_event_body(path: Path) -> str | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request")
    if isinstance(pull_request, dict) and isinstance(pull_request.get("body"), str):
        return pull_request["body"]
    issue = payload.get("issue")
    if isinstance(issue, dict) and isinstance(issue.get("body"), str):
        return issue["body"]
    return None


def _load_body(args: argparse.Namespace) -> tuple[str | None, str]:
    body_file = args.body_file or os.environ.get("PR_READY_PR_BODY_FILE")
    if body_file:
        path = Path(body_file)
        return path.read_text(encoding="utf-8"), str(path)

    event_path = args.github_event_path or os.environ.get("GITHUB_EVENT_PATH")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_path and (args.github_event_path or event_name == "pull_request"):
        path = Path(event_path)
        body = _read_event_body(path)
        if body is not None:
            return body, str(path)

    return None, "none"


def _format_report(report: FollowupReport) -> str:
    issue_text = ", ".join(report.linked_issues) if report.linked_issues else "none"
    deferred = report.deferred_work if report.deferred_work else "none"
    no_issue = report.explicit_no_issue_reason if report.explicit_no_issue_reason else "none"
    state_errors = "; ".join(report.issue_state_errors) if report.issue_state_errors else "none"
    return (
        "PR follow-up check: "
        f"status={report.status}; source={report.source}; deferred={deferred!r}; "
        f"linked_issues={issue_text}; no_issue_reason={no_issue!r}; "
        f"issue_state_errors={state_errors!r}; {report.message}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-file", type=Path, help="Markdown PR body to check.")
    parser.add_argument(
        "--github-event-path",
        type=Path,
        help="GitHub event JSON containing pull_request.body.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--require-open-issues",
        action="store_true",
        help="Verify linked follow-up issues are open with gh issue view.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PR follow-up check CLI."""
    args = _build_parser().parse_args(argv)
    body, source = _load_body(args)
    if body is None:
        report = FollowupReport(
            status="skipped",
            source=source,
            deferred_work="",
            linked_issues=(),
            explicit_no_issue_reason="",
            issue_state_errors=(),
            message="No PR body source configured.",
        )
        if args.json:
            print(json.dumps(report.__dict__, sort_keys=True))
        else:
            print(_format_report(report))
        return 0

    require_open = args.require_open_issues or os.environ.get(
        "PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES", ""
    ).lower() in {"1", "true", "yes", "on"}
    report = analyze_body(body, source=source, require_open_issues=require_open)
    if args.json:
        print(json.dumps(report.__dict__, sort_keys=True))
    else:
        failing_statuses = {"missing_followup", "missing_section", "issue_state_error"}
        stream = sys.stderr if report.status in failing_statuses else sys.stdout
        print(_format_report(report), file=stream)
    return 2 if report.status in {"missing_followup", "missing_section", "issue_state_error"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
