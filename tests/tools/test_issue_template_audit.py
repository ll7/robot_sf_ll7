"""Tests for the issue template audit helper used by the template-auditor skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools.issue_template_audit import audit_issue_body, main

if TYPE_CHECKING:
    from pathlib import Path


def test_issue_template_audit_repairs_missing_sections() -> None:
    """Verify that missing template sections are detected and repaired.

    This matters because the skill needs a deterministic repair path when an
    existing issue is close to template-ready but still missing required fields.
    """

    body = """## Goal / Problem

Add a structured issue workflow.

## Scope

- In scope:
- Out of scope:
"""

    result = audit_issue_body(body)

    assert "Added Value Estimation" in result.missing_sections
    assert "Effort Estimation" in result.missing_sections
    assert "Project Metadata" in result.missing_sections
    assert "## Added Value Estimation" in result.repaired_body
    assert "## Definition of Done" in result.repaired_body
    assert result.repaired_body.endswith("\n")


def test_issue_template_audit_accepts_complete_body() -> None:
    """Verify that a fully specified issue body passes the contract check.

    This matters because the auditor should leave already-good issues alone
    instead of rewriting them or adding noise.
    """

    body = """## Goal / Problem

Add a structured issue workflow.

## Scope

- In scope:
- Out of scope:

## Added Value Estimation

- User value:
- Maintenance value:
- Why now:

## Effort Estimation

- Rough estimate (hours):
- Best estimate (hours):
- Unknowns:

## Complexity Estimation

- Implementation complexity:
- Dependencies:
- Open questions:

## Risk Assessment

- Functional risk:
- Compatibility risk:
- Rollout risk:
- Mitigation:

## Affected Files

- `path/to/file.py` - What changes here.
- `tests/test_*.py` - Required coverage.

## Definition of Done

- [ ] The task is implemented.

## Success Metrics

- The task can be completed without extra clarification.

## Validation / Testing

- [ ] Run the repository validation gate.

## Project Metadata

- Priority:
- Effort (h):
- Reviewed:
"""

    result = audit_issue_body(body)

    assert result.missing_sections == ()
    assert result.present_sections[0] == "Goal / Problem"
    assert result.repaired_body.endswith("\n")


def test_issue_template_audit_cli_repairs_body(tmp_path: Path, capsys) -> None:
    """Verify the CLI wrapper can audit and repair a body file in place.

    This matters because the new issue-audit skill is documented to use the
    helper script, so the script must behave correctly on a real markdown file.
    """

    body_file = tmp_path / "issue.md"
    repair_file = tmp_path / "repaired.md"
    body_file.write_text(
        """## Goal / Problem

Repair the issue template body.
""",
        encoding="utf-8",
    )

    exit_code = main(["--body-file", str(body_file), "--repair-file", str(repair_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"missing_sections": [' in captured.out
    assert repair_file.read_text(encoding="utf-8").startswith("## Goal / Problem")
    assert "## Validation / Testing" in repair_file.read_text(encoding="utf-8")
