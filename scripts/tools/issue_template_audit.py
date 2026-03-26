"""Audit issue bodies against the repo's agent-ready issue-template contract.

This module keeps the issue-audit skill testable by providing a small,
deterministic parser and repair helper that can detect missing sections and
append a conservative scaffold when the fix is obvious.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

SECTION_ORDER: tuple[str, ...] = (
    "Goal / Problem",
    "Scope",
    "Added Value Estimation",
    "Effort Estimation",
    "Complexity Estimation",
    "Risk Assessment",
    "Affected Files",
    "Definition of Done",
    "Success Metrics",
    "Validation / Testing",
    "Project Metadata",
)

SECTION_ALIASES: dict[str, str] = {
    "goal / problem": "Goal / Problem",
    "goal/problem": "Goal / Problem",
    "problem": "Goal / Problem",
    "objective": "Goal / Problem",
    "problem description": "Goal / Problem",
    "scope": "Scope",
    "added value estimation": "Added Value Estimation",
    "value estimation": "Added Value Estimation",
    "effort estimation": "Effort Estimation",
    "complexity estimation": "Complexity Estimation",
    "risk assessment": "Risk Assessment",
    "affected files": "Affected Files",
    "files and components": "Affected Files",
    "definition of done": "Definition of Done",
    "success metrics": "Success Metrics",
    "validation / testing": "Validation / Testing",
    "validation": "Validation / Testing",
    "testing": "Validation / Testing",
    "project metadata": "Project Metadata",
}

SECTION_PLACEHOLDERS: dict[str, str] = {
    "Goal / Problem": "<!-- Describe the problem or goal here. -->",
    "Scope": "- In scope:\n- Out of scope:\n- Entry points:",
    "Added Value Estimation": "- User value:\n- Maintenance value:\n- Why now:",
    "Effort Estimation": "- Rough estimate (hours):\n- Best estimate (hours):\n- Unknowns:",
    "Complexity Estimation": "- Implementation complexity:\n- Dependencies:\n- Open questions:",
    "Risk Assessment": "- Functional risk:\n- Compatibility risk:\n- Rollout risk:\n- Mitigation:",
    "Affected Files": "- `path/to/file.py` - What changes here.\n- `tests/test_*.py` - Required coverage.",
    "Definition of Done": "- [ ] Fill in the missing acceptance criteria.",
    "Success Metrics": "- What success looks like:",
    "Validation / Testing": "- [ ] Add or run the relevant validation command.",
    "Project Metadata": "- Priority:\n- Effort (h):\n- Reviewed:",
}

HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class IssueAuditResult:
    """Structured output for template-fit audits."""

    present_sections: tuple[str, ...]
    missing_sections: tuple[str, ...]
    repaired_body: str


def normalize_section_title(raw_title: str) -> str:
    """Map a markdown heading to a canonical template section name."""

    cleaned = raw_title.strip()
    cleaned = re.sub(r"^[^\w]+", "", cleaned)
    cleaned = re.sub(r"\s*/\s*", " / ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[:：]+$", "", cleaned)
    normalized = cleaned.lower()
    return SECTION_ALIASES.get(normalized, cleaned)


def extract_present_sections(body: str) -> tuple[str, ...]:
    """Return the canonical section names detected in an issue body."""

    found: list[str] = []
    seen: set[str] = set()
    for match in HEADING_RE.finditer(body):
        canonical = normalize_section_title(match.group("title"))
        if canonical in SECTION_ORDER and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    return tuple(found)


def missing_sections(
    body: str, required_sections: Sequence[str] = SECTION_ORDER
) -> tuple[str, ...]:
    """Return the sections that the body still needs."""

    present = set(extract_present_sections(body))
    return tuple(section for section in required_sections if section not in present)


def build_repaired_body(body: str, missing: Sequence[str]) -> str:
    """Append conservative scaffolding for sections that are absent."""

    stripped = body.rstrip()
    chunks = [stripped] if stripped else []
    for section in missing:
        placeholder = SECTION_PLACEHOLDERS[section]
        chunks.append(f"## {section}\n\n{placeholder}")
    return "\n\n".join(chunks).rstrip() + "\n"


def audit_issue_body(body: str) -> IssueAuditResult:
    """Audit an issue body against the repo's template contract."""

    present = extract_present_sections(body)
    missing = missing_sections(body)
    repaired = build_repaired_body(body, missing)
    return IssueAuditResult(
        present_sections=present, missing_sections=missing, repaired_body=repaired
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-file", type=Path, required=True, help="Issue body markdown file.")
    parser.add_argument(
        "--repair-file",
        type=Path,
        help="Write the repaired body to this path instead of stdout-only output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by the issue-audit skill."""

    args = _build_parser().parse_args(argv)
    body = args.body_file.read_text(encoding="utf-8")
    result = audit_issue_body(body)

    payload = {
        "present_sections": list(result.present_sections),
        "missing_sections": list(result.missing_sections),
        "repaired_body": result.repaired_body,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.repair_file is not None:
        args.repair_file.write_text(result.repaired_body, encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
