"""Tests for the issue template audit helper used by the template-auditor skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools.issue_template_audit import (
    CORE_SECTION_ORDER,
    SECTION_ORDER,
    audit_issue_body,
    main,
    normalize_section_title,
    select_required_sections,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_yaml_form_issue_audits_against_leaner_core() -> None:
    """Verify YAML-form issues are audited against the shared core, not the full markdown set.

    This matters because the repo ships YAML issue forms (epic/execution-run/...) whose
    headings (``Scope and non-goals``, ``Acceptance criteria``, ``Estimate metadata``)
    deliberately omit the markdown-only sub-sections; auditing them against the full
    markdown contract produced false "missing section" reports.
    """

    body = """## Goal / Problem

Ship the thing.

## Scope and non-goals

- In scope: the thing.
- Out of scope: everything else.

## Acceptance criteria

- [ ] The thing works.

## Validation / Testing

- [ ] Run the gate.

## Estimate metadata

- Effort: 4h.
"""

    assert select_required_sections(body) == CORE_SECTION_ORDER
    result = audit_issue_body(body)
    assert result.missing_sections == ()


def test_scope_non_goals_heading_is_recognized() -> None:
    """Verify the ``Scope / non-goals`` heading variant is credited and signals the leaner core.

    This matters because agent-authored analysis/research bodies write ``## Scope / non-goals``
    (slash variant) where the YAML forms write ``Scope and non-goals``; without the alias the
    audit reported a false "missing Scope" and held the body to the full markdown contract.
    """

    body = """## Goal / Problem

Investigate the thing.

## Scope / non-goals

- In scope: the analysis.
- Out of scope: implementation.

## Definition of Done

- [ ] Analysis complete.

## Validation / Testing

- [ ] Run the analysis script.
"""

    assert normalize_section_title("Scope / non-goals") == "Scope"
    assert select_required_sections(body) == CORE_SECTION_ORDER
    result = audit_issue_body(body)
    assert "Scope" not in result.missing_sections
    assert result.missing_sections == ()


def test_agent_exec_spec_h3_content_counts_as_present() -> None:
    """Verify contract content carried as ``###`` headings in the agent-exec-spec is credited.

    This matters because agent-authored bodies place acceptance/validation content as
    ``###`` subheadings (often with parenthetical qualifiers) inside the appended
    ``agent-exec-spec`` block; the audit must see that real content instead of flagging it.
    """

    body = """## Goal / Problem

Do the work.

## Scope

- In scope: x.

## Estimate

- 6h.

<!-- agent-exec-spec:v1 -->
## 🤖 Agent execution spec (codex gpt5.5 medium)

### Acceptance criteria (mirrors body)

- [ ] Done.

### Validation commands

```bash
uv run pytest -q
```
"""

    assert select_required_sections(body) == CORE_SECTION_ORDER
    result = audit_issue_body(body)
    assert "Definition of Done" not in result.missing_sections
    assert "Validation / Testing" not in result.missing_sections
    assert "Effort Estimation" not in result.missing_sections
    assert result.missing_sections == ()


def test_bare_body_still_uses_full_markdown_contract() -> None:
    """Verify a bare body with no leaner signals keeps the strict full contract.

    This matters because the leaner core must only apply when a body positively signals
    the YAML-form/agent family; otherwise incomplete markdown issues would stop being flagged.
    """

    body = """## Goal / Problem

Bare body.

## Scope

- In scope:
"""

    assert select_required_sections(body) == SECTION_ORDER
    result = audit_issue_body(body)
    assert "Added Value Estimation" in result.missing_sections
    assert "Project Metadata" in result.missing_sections


def test_normalize_section_title_handles_variants() -> None:
    """Verify new heading aliases and parenthetical stripping map to canonical sections."""

    assert normalize_section_title("Scope and non-goals") == "Scope"
    assert normalize_section_title("Acceptance criteria (mirrors body)") == "Definition of Done"
    assert normalize_section_title("Validation commands") == "Validation / Testing"
    assert normalize_section_title("Estimate metadata") == "Effort Estimation"
    assert normalize_section_title("Question / Goal") == "Goal / Problem"


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


def test_issue_template_audit_accepts_yaml_fence_aliases() -> None:
    """Verify metadata parsing accepts common YAML code-fence variants."""

    body = """## Goal / Problem

Context.

## Archetype Metadata

  ```YML
archetype: workflow
evidence_tier: smoke
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
  ```
"""

    result = audit_issue_body(body)

    assert result.metadata.block_present is True
    assert result.metadata.findings == ()


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


def test_issue_template_audit_preserves_invalid_metadata_value_types() -> None:
    """Verify invalid metadata values stay type-preserving in structured output."""

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: 7
evidence_tier:
  - smoke
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
```
"""

    result = audit_issue_body(body)

    assert result.metadata.invalid_values == {"archetype": 7, "evidence_tier": ["smoke"]}
    assert "Invalid 'archetype' value 7" in result.metadata.findings[0]
    assert "Invalid 'evidence_tier' value ['smoke']" in result.metadata.findings[1]


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


def test_issue_template_audit_accepts_expanded_archetypes() -> None:
    """Verify the 2026-06-22 work-type archetypes validate without findings.

    This matters because the issue automation emits implementation/test/refactor/data and the
    maintainer chose to bless them as canonical instead of remapping every issue.
    """

    for archetype in ("implementation", "test", "refactor", "data"):
        body = f"""## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: {archetype}
evidence_tier: smoke
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
```
"""
        result = audit_issue_body(body)
        assert result.metadata.invalid_values == {}, archetype
        assert result.metadata.findings == (), archetype


def test_issue_template_audit_accepts_deprecated_metadata_aliases() -> None:
    """Verify deprecated archetype/evidence-tier spellings are accepted on read.

    This matters because ``agent_task`` and ``proposal`` are still emitted by older automation,
    and the auditor should not flag them as invalid (it never rewrites issue bodies).
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: agent_task
evidence_tier: proposal
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
```
"""

    result = audit_issue_body(body)
    assert result.metadata.invalid_values == {}
    assert result.metadata.findings == ()
    # The body is preserved verbatim; aliases are accepted, not rewritten.
    assert result.metadata.parsed_metadata is not None
    assert result.metadata.parsed_metadata["archetype"] == "agent_task"
    assert result.metadata.parsed_metadata["evidence_tier"] == "proposal"


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
