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

import yaml

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
    "Estimate Discussion",
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
    "estimate discussion": "Estimate Discussion",
    "estimate rationale": "Estimate Discussion",
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
    "Estimate Discussion": "- Why these values were proposed:\n- Uncertainty / confidence:\n- What evidence would move the estimate:",
    "Project Metadata": "- Priority:\n- Effort (h):\n- Reviewed:",
}

HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
METADATA_SECTION_TITLE = "Archetype Metadata"
METADATA_YAML_RE = re.compile(
    r"^[ \t]*```ya?ml\s*\r?\n(?P<block>.*?)(?:\r?\n)?^[ \t]*```",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
ARCHETYPE_METADATA_REQUIRED_KEYS: tuple[str, ...] = (
    "archetype",
    "evidence_tier",
    "linked_policy",
)
VALID_ARCHETYPES: tuple[str, ...] = (
    "blocked-asset",
    "preflight",
    "slurm-execution",
    "analysis",
    "synthesis",
    "workflow",
    "docs",
    "benchmark-campaign",
    "training-campaign",
)
VALID_EVIDENCE_TIERS: tuple[str, ...] = (
    "idea",
    "launch_packet",
    "preflight_valid",
    "smoke",
    "nominal",
    "stress",
    "full_matrix",
    "analysis_only",
    "synthesis",
    "paper_grade",
    "blocked",
)


@dataclass(frozen=True, slots=True)
class IssueAuditResult:
    """Structured output for template-fit audits."""

    present_sections: tuple[str, ...]
    missing_sections: tuple[str, ...]
    repaired_body: str
    metadata: ArchetypeMetadataAuditResult


@dataclass(frozen=True, slots=True)
class ArchetypeMetadataAuditResult:
    """Structured output for the optional archetype metadata block."""

    heading_present: bool
    block_present: bool
    parsed_metadata: dict[str, object] | None
    missing_keys: tuple[str, ...]
    invalid_values: dict[str, object]
    parse_error: str | None

    @property
    def findings(self) -> tuple[str, ...]:
        """Human-readable findings derived from the structural audit fields."""
        if not self.heading_present:
            return ("Missing '## Archetype Metadata' section.",)
        if not self.block_present:
            return ("Missing YAML code block under '## Archetype Metadata'.",)
        if self.parse_error is not None:
            return (f"Malformed archetype metadata YAML: {self.parse_error}",)
        result: list[str] = []
        if self.missing_keys:
            result.append("Missing archetype metadata keys: " + ", ".join(self.missing_keys))
        for key, value in self.invalid_values.items():
            allowed = VALID_ARCHETYPES if key == "archetype" else VALID_EVIDENCE_TIERS
            result.append(
                f"Invalid {key!r} value {value!r}; expected one of: " + ", ".join(allowed)
            )
        return tuple(result)


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


def _extract_named_section(body: str, expected_title: str) -> str | None:
    """Return the raw content under a named `##` heading, if present."""

    for match in HEADING_RE.finditer(body):
        if normalize_section_title(match.group("title")) != expected_title:
            continue
        section_start = match.end()
        next_heading = HEADING_RE.search(body, section_start)
        section_end = next_heading.start() if next_heading is not None else len(body)
        return body[section_start:section_end]
    return None


def _validate_canonical_value(
    key: str, value: object, allowed_values: tuple[str, ...]
) -> str | None:
    """Return a finding when a metadata value is absent from the canonical set."""

    if not isinstance(value, str) or value not in allowed_values:
        return f"Invalid {key!r} value {value!r}; expected one of: " + ", ".join(allowed_values)
    return None


def audit_archetype_metadata(body: str) -> ArchetypeMetadataAuditResult:
    """Audit the optional issue archetype metadata block near the top of the body."""

    section = _extract_named_section(body, METADATA_SECTION_TITLE)
    if section is None:
        return ArchetypeMetadataAuditResult(
            heading_present=False,
            block_present=False,
            parsed_metadata=None,
            missing_keys=ARCHETYPE_METADATA_REQUIRED_KEYS,
            invalid_values={},
            parse_error=None,
        )

    block_match = METADATA_YAML_RE.search(section)
    if block_match is None:
        return ArchetypeMetadataAuditResult(
            heading_present=True,
            block_present=False,
            parsed_metadata=None,
            missing_keys=ARCHETYPE_METADATA_REQUIRED_KEYS,
            invalid_values={},
            parse_error=None,
        )

    try:
        parsed = yaml.safe_load(block_match.group("block"))
    except yaml.YAMLError as exc:
        return ArchetypeMetadataAuditResult(
            heading_present=True,
            block_present=True,
            parsed_metadata=None,
            missing_keys=(),
            invalid_values={},
            parse_error=str(exc),
        )

    if not isinstance(parsed, dict):
        message = "Archetype metadata YAML must decode to a mapping."
        return ArchetypeMetadataAuditResult(
            heading_present=True,
            block_present=True,
            parsed_metadata=None,
            missing_keys=(),
            invalid_values={},
            parse_error=message,
        )

    missing_keys = tuple(key for key in ARCHETYPE_METADATA_REQUIRED_KEYS if key not in parsed)
    invalid_values: dict[str, object] = {}

    if "archetype" in parsed:
        archetype_finding = _validate_canonical_value(
            "archetype", parsed.get("archetype"), VALID_ARCHETYPES
        )
        if archetype_finding is not None:
            invalid_values["archetype"] = parsed.get("archetype")

    if "evidence_tier" in parsed:
        evidence_tier_finding = _validate_canonical_value(
            "evidence_tier", parsed.get("evidence_tier"), VALID_EVIDENCE_TIERS
        )
        if evidence_tier_finding is not None:
            invalid_values["evidence_tier"] = parsed.get("evidence_tier")

    return ArchetypeMetadataAuditResult(
        heading_present=True,
        block_present=True,
        parsed_metadata=parsed,
        missing_keys=missing_keys,
        invalid_values=invalid_values,
        parse_error=None,
    )


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
        present_sections=present,
        missing_sections=missing,
        repaired_body=repaired,
        metadata=audit_archetype_metadata(body),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for issue-template auditing.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
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
        "metadata": {
            "heading_present": result.metadata.heading_present,
            "block_present": result.metadata.block_present,
            "parsed_metadata": result.metadata.parsed_metadata,
            "missing_keys": list(result.metadata.missing_keys),
            "invalid_values": result.metadata.invalid_values,
            "parse_error": result.metadata.parse_error,
            "findings": list(result.metadata.findings),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.repair_file is not None:
        args.repair_file.write_text(result.repaired_body, encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
