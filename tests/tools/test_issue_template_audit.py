"""Tests for the issue template audit helper used by the template-auditor skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools.issue_template_audit import (
    audit_issue_body,
    main,
    normalize_section_title,
)

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
    assert "Estimate Discussion" in result.missing_sections
    assert "Project Metadata" in result.missing_sections
    assert "## Added Value Estimation" in result.repaired_body
    assert "## Estimate Discussion" in result.repaired_body
    assert "## Definition of Done" in result.repaired_body
    assert "## Archetype Metadata" not in result.repaired_body
    assert result.metadata.findings == ("Missing '## Archetype Metadata' section.",)
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

## Estimate Discussion

- Why these values were proposed:
- Uncertainty / confidence:
- What evidence would move the estimate:

## Project Metadata

- Priority:
- Effort (h):
- Reviewed:
"""

    result = audit_issue_body(body)

    assert result.missing_sections == ()
    assert result.present_sections[0] == "Goal / Problem"
    assert result.repaired_body.endswith("\n")


def test_issue_template_audit_normalizes_legacy_heading_variants() -> None:
    """Verify that legacy and punctuation-variant headings map to the canonical section.

    This matters because existing issues use older phrasing and emoji-prefixed
    headings, and the auditor should not duplicate scaffold sections for them.
    """

    body = """## 🐛 Problem Description

Legacy issue text.

## Goal/Problem

More legacy text.
"""

    result = audit_issue_body(body)

    assert normalize_section_title("🐛 Problem Description") == "Goal / Problem"
    assert normalize_section_title("Goal/Problem") == "Goal / Problem"
    assert "Goal / Problem" not in result.missing_sections
    assert result.present_sections.count("Goal / Problem") == 1
    assert "## Goal / Problem" not in result.repaired_body


def test_issue_template_audit_accepts_valid_archetype_metadata() -> None:
    """Verify canonical archetype metadata parses without findings.

    This matters because issue archetype triage should accept the documented
    values without inventing repairs or blocking a valid issue contract.
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: workflow
evidence_tier: smoke
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
  - docs/context/artifact_evidence_vocabulary.md
```

## Scope

- In scope:
- Out of scope:
"""

    result = audit_issue_body(body)

    assert result.metadata.heading_present is True
    assert result.metadata.block_present is True
    assert result.metadata.missing_keys == ()
    assert result.metadata.invalid_values == {}
    assert result.metadata.parse_error is None
    assert result.metadata.findings == ()
    assert result.metadata.parsed_metadata == {
        "archetype": "workflow",
        "evidence_tier": "smoke",
        "linked_policy": [
            "docs/context/issue_1512_issue_archetypes.md",
            "docs/context/artifact_evidence_vocabulary.md",
        ],
    }


def test_issue_template_audit_flags_missing_metadata_block() -> None:
    """Verify the metadata section is reported when the YAML block is absent.

    This matters because the auditor should preserve the section heading while
    still telling triage workflows that the structured metadata is incomplete.
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

Metadata still needs to be filled in.
"""

    result = audit_issue_body(body)

    assert result.metadata.heading_present is True
    assert result.metadata.block_present is False
    assert result.metadata.missing_keys == (
        "archetype",
        "evidence_tier",
        "linked_policy",
    )
    assert result.metadata.findings == ("Missing YAML code block under '## Archetype Metadata'.",)


def test_issue_template_audit_flags_missing_metadata_keys() -> None:
    """Verify missing metadata keys are surfaced without inventing defaults.

    This matters because the issue archetype convention is opt-in metadata, so
    the auditor must flag gaps instead of silently filling them.
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: workflow
```
"""

    result = audit_issue_body(body)

    assert result.metadata.missing_keys == ("evidence_tier", "linked_policy")
    assert result.metadata.invalid_values == {}
    assert result.metadata.findings == (
        "Missing archetype metadata keys: evidence_tier, linked_policy",
    )


def test_issue_template_audit_flags_invalid_archetype() -> None:
    """Verify invalid archetype values are rejected against the canonical set.

    This matters because issue triage should only emit the documented archetype
    taxonomy from issue 1512.
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: made-up
evidence_tier: smoke
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
```
"""

    result = audit_issue_body(body)

    assert result.metadata.invalid_values == {"archetype": "made-up"}
    assert "Invalid 'archetype' value 'made-up'" in result.metadata.findings[0]


def test_issue_template_audit_flags_invalid_evidence_tier() -> None:
    """Verify invalid evidence tiers are rejected against the canonical set.

    This matters because the evidence tier controls how strongly an issue can
    claim proof, so invalid values should be surfaced during audit.
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: workflow
evidence_tier: almost_done
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
```
"""

    result = audit_issue_body(body)

    assert result.metadata.invalid_values == {"evidence_tier": "almost_done"}
    assert "Invalid 'evidence_tier' value 'almost_done'" in result.metadata.findings[0]


def test_issue_template_audit_flags_malformed_metadata_yaml() -> None:
    """Verify malformed metadata YAML is surfaced as an explicit parse failure.

    This matters because triage workflows need a concrete repair signal when the
    metadata block exists but cannot be parsed safely.
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: workflow
evidence_tier: smoke
linked_policy: [docs/context/issue_1512_issue_archetypes.md
```
"""

    result = audit_issue_body(body)

    assert result.metadata.parse_error is not None
    assert result.metadata.findings[0].startswith("Malformed archetype metadata YAML:")


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
    assert '"metadata": {' in captured.out
    assert repair_file.read_text(encoding="utf-8").startswith("## Goal / Problem")
    assert "## Validation / Testing" in repair_file.read_text(encoding="utf-8")
    assert "## Estimate Discussion" in repair_file.read_text(encoding="utf-8")
