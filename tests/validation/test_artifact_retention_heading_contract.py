"""Heading contract for the shared ops guide (issue #9132)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

GUIDE = (
    Path(__file__).resolve().parents[2] / "docs" / "context" / "artifact_retention_and_cleanup.md"
)
NUMBERED_SUBSECTION_RE = re.compile(r"^###\s+\d+(?:\.\d+)*\s+\S", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
REMEDIATION = (
    "Append stable unnumbered headings to the ops guide; do not add numbered ### N.M sections "
    "(they renumber-conflict across sibling ops PRs). If a section grows, split it into its own "
    "context note."
)


def _headings() -> list[tuple[str, str]]:
    return [
        (match.group(1), match.group(2))
        for match in HEADING_RE.finditer(GUIDE.read_text(encoding="utf-8"))
    ]


def test_ops_guide_has_no_numbered_subsections() -> None:
    numbered = NUMBERED_SUBSECTION_RE.findall(GUIDE.read_text(encoding="utf-8"))
    assert not numbered, (
        f"numbered subsections reintroduce rename conflicts: {numbered}. {REMEDIATION}"
    )


def test_ops_guide_headings_are_unique() -> None:
    names = [name for _level, name in _headings()]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, (
        f"duplicate headings cause ambiguous anchors: {duplicates}. {REMEDIATION}"
    )


def test_checked_workflows_anchor_remains_stable() -> None:
    assert any(level == "##" and name == "4. Checked workflows" for level, name in _headings()), (
        "the `## 4. Checked workflows` heading is a referenced anchor "
        "(docs/post_access_local_analysis_runbook.md#4-checked-workflows); keep it stable."
    )
