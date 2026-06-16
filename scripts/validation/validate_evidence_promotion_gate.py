#!/usr/bin/env python3
"""Validate evidence promotion gates for context notes and evidence bundles.

This validator enforces the diagnostic -> smoke -> nominal -> stress -> paper_grade
promotion ladder defined in docs/maintainer_values.md and
docs/context/artifact_evidence_vocabulary.md.

It fails closed when required artifacts are missing for claims beyond diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


class EvidenceTier(Enum):
    """Allowed evidence promotion tiers in ascending order."""

    PROPOSAL = "proposal"
    DIAGNOSTIC = "diagnostic"
    PREFLIGHT = "preflight"
    SMOKE = "smoke"
    NOMINAL = "nominal"
    STRESS = "stress"
    PAPER_GRADE = "paper_grade"

    @classmethod
    def from_string(cls, value: str) -> EvidenceTier:
        """Parse a tier string into an EvidenceTier."""
        value = value.lower().strip()
        for tier in cls:
            if tier.value == value or tier.name.lower() == value:
                return tier
        raise ValueError(f"Unknown evidence tier: {value}")

    def can_promote_to(self, target: EvidenceTier) -> bool:
        """Return whether this tier can promote to the target tier."""
        return target in ALLOWED_TRANSITIONS.get(self, set()) or target == self

    @property
    def rank(self) -> int:
        """Return the numeric rank of this tier (higher = more evidence)."""
        return list(EvidenceTier).index(self)


ALLOWED_TRANSITIONS = {
    EvidenceTier.PROPOSAL: {EvidenceTier.PREFLIGHT},
    EvidenceTier.DIAGNOSTIC: {EvidenceTier.PREFLIGHT},
    EvidenceTier.PREFLIGHT: {EvidenceTier.SMOKE},
    EvidenceTier.SMOKE: {EvidenceTier.NOMINAL},
    EvidenceTier.NOMINAL: {EvidenceTier.STRESS},
    EvidenceTier.STRESS: {EvidenceTier.PAPER_GRADE},
    EvidenceTier.PAPER_GRADE: set(),
}


@dataclass(frozen=True)
class RequiredArtifact:
    """One required artifact for a promotion tier."""

    name: str
    description: str
    patterns: tuple[re.Pattern[str], ...]
    required_for: frozenset[EvidenceTier]


REQUIRED_ARTIFACTS = (
    RequiredArtifact(
        name="command_or_config",
        description="Canonical command or config path that produced the evidence.",
        patterns=(
            re.compile(r"\bcommand\s*[:=]"),
            re.compile(r"\bconfig\s*[:=]"),
            re.compile(r"--config\s+\S+"),
            re.compile(r"\bconfig[_-]?path\b"),
            re.compile(r'"(?:command|config)"\s*:'),
        ),
        required_for=frozenset(
            {
                EvidenceTier.PREFLIGHT,
                EvidenceTier.SMOKE,
                EvidenceTier.NOMINAL,
                EvidenceTier.STRESS,
                EvidenceTier.PAPER_GRADE,
            }
        ),
    ),
    RequiredArtifact(
        name="commit_or_checksum",
        description="Git commit SHA or checksum for reproducibility.",
        patterns=(
            re.compile(r"\bcommit\s*[:=]\s*[0-9a-f]{7,}"),
            re.compile(r"\bsha-?256\s*[:=]"),
            re.compile(r"\bchecksum\s*[:=]"),
            re.compile(r'"(?:commit|sha256|checksum)"\s*:'),
            re.compile(r"\b[0-9a-f]{40}\b"),
        ),
        required_for=frozenset(
            {
                EvidenceTier.SMOKE,
                EvidenceTier.NOMINAL,
                EvidenceTier.STRESS,
                EvidenceTier.PAPER_GRADE,
            }
        ),
    ),
    RequiredArtifact(
        name="metric_or_summary",
        description="Metric summary or episode summary data.",
        patterns=(
            re.compile(r"\bsuccess[_-]?rate\b"),
            re.compile(r"\bcollision[_-]?rate\b"),
            re.compile(r"\bmetric\b.*\bvalue\b"),
            re.compile(r"\bsummary\b.*\bjson\b"),
            re.compile(r'"(?:success|collision|metric|summary)"\s*:'),
            re.compile(r"\bepisodes?\b.*\b(count|total)\b"),
        ),
        required_for=frozenset(
            {
                EvidenceTier.SMOKE,
                EvidenceTier.NOMINAL,
                EvidenceTier.STRESS,
                EvidenceTier.PAPER_GRADE,
            }
        ),
    ),
    RequiredArtifact(
        name="comparator_or_baseline",
        description="Comparator, baseline, or control surface for comparison.",
        patterns=(
            re.compile(r"\bcompar(?:ator|ison)\b"),
            re.compile(r"\bbaseline\s*[:=]"),
            re.compile(r"\bcontrol\s*[:=]"),
            re.compile(r"\bvs\.?\s+\S+"),
            re.compile(r'"(?:comparator|baseline|control)"\s*:'),
        ),
        required_for=frozenset(
            {EvidenceTier.NOMINAL, EvidenceTier.STRESS, EvidenceTier.PAPER_GRADE}
        ),
    ),
    RequiredArtifact(
        name="trace_or_frames",
        description="Trace, frame, step, or trajectory support data.",
        patterns=(
            re.compile(r"\btrace\b.*\b(file|path|support)\b"),
            re.compile(r"\bframe\b.*\b(count|index)\b"),
            re.compile(r"\btrajectory\b"),
            re.compile(r"\bepisode[_-]?trace\b"),
            re.compile(r'"(?:trace|frames?|steps?|trajectory)"\s*:'),
        ),
        required_for=frozenset({EvidenceTier.STRESS, EvidenceTier.PAPER_GRADE}),
    ),
    RequiredArtifact(
        name="limitations_or_caveats",
        description="Explicit limitations, caveats, or fallback/degraded mode declarations.",
        patterns=(
            re.compile(r"\blimitation\b"),
            re.compile(r"\bcaveat\b"),
            re.compile(r"\bfallback\b"),
            re.compile(r"\bdegraded\b"),
            re.compile(r"\bnot[_-]?available\b"),
            re.compile(r"\bfail[_-]?closed\b"),
            re.compile(r'"(?:limitations|caveats|fallback|degraded)"\s*:'),
        ),
        required_for=frozenset({EvidenceTier.PAPER_GRADE}),
    ),
)


@dataclass
class ValidationResult:
    """Validation result for one evidence bundle."""

    context_note: str
    claimed_tier: EvidenceTier | None
    current_tier: EvidenceTier | None
    transition_valid: bool
    missing_artifacts: list[str]
    present_artifacts: list[str]
    warnings: list[str]
    errors: list[str]
    is_diagnostic_only: bool = False


def _read_text(path: Path) -> str:
    """Read a text file with replacement for odd encodings."""
    return path.read_text(encoding="utf-8", errors="replace")


def _evidence_files(path: Path) -> list[Path]:
    """Return text-like evidence files to scan."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES
    )


def _combined_text(paths: list[Path]) -> str:
    """Combine text from multiple paths."""
    chunks = []
    for path in paths:
        try:
            chunks.append(_read_text(path))
        except OSError:
            pass
    return "\n".join(chunks).lower()


def _detect_tier(text: str) -> EvidenceTier | None:
    """Detect the strongest claimed evidence tier in text."""
    tier_patterns = [
        (EvidenceTier.PAPER_GRADE, re.compile(r"\bpaper[_-]?grade\b|\bmanuscript[_-]?facing\b")),
        (
            EvidenceTier.STRESS,
            re.compile(r"\bstress[_-]?test\b|\bstress[_-]?evidence\b|\bstress\b.*\bevidence\b"),
        ),
        (EvidenceTier.NOMINAL, re.compile(r"\bnominal\b.*\bevidence\b|\bnominal[_-]?benchmark\b")),
        (
            EvidenceTier.SMOKE,
            re.compile(r"\bsmoke[_-]?evidence\b|\bsmoke\b.*\btest\b|\bsmoke\b.*\bevidence\b"),
        ),
        (
            EvidenceTier.PREFLIGHT,
            re.compile(r"\bpreflight\b.*\bevidence\b|\bpreflight\b.*\bcheck\b"),
        ),
        (
            EvidenceTier.DIAGNOSTIC,
            re.compile(
                r"\bdiagnostic[_-]?only\b|\bdiagnostic\b.*\bevidence\b|\bdiagnostic\b.*\bonly\b"
            ),
        ),
        (EvidenceTier.PROPOSAL, re.compile(r"\bproposal\b.*\bevidence\b|\bevidence:proposal\b")),
    ]
    for tier, pattern in tier_patterns:
        if pattern.search(text):
            return tier
    return None


def _check_artifacts(text: str, claimed_tier: EvidenceTier) -> tuple[list[str], list[str]]:
    """Check which required artifacts are present or missing for the claimed tier."""
    present = []
    missing = []
    for artifact in REQUIRED_ARTIFACTS:
        if claimed_tier not in artifact.required_for:
            continue
        is_present = any(pattern.search(text) for pattern in artifact.patterns)
        if is_present:
            present.append(artifact.name)
        else:
            missing.append(artifact.name)
    return present, missing


def _check_transition(
    current: EvidenceTier | None, claimed: EvidenceTier
) -> tuple[bool, list[str]]:
    """Check if the transition from current to claimed is allowed."""
    errors = []
    if current is None:
        if claimed in {EvidenceTier.PROPOSAL, EvidenceTier.DIAGNOSTIC}:
            return True, []
        else:
            errors.append(
                f"Cannot promote to {claimed.value} without proposal or diagnostic baseline"
            )
            return False, errors
    if claimed == current:
        return True, []
    if claimed in ALLOWED_TRANSITIONS.get(current, set()):
        return True, []
    errors.append(f"Invalid transition: {current.value} -> {claimed.value}")
    return False, errors


def _scan_context_notes(root: Path) -> list[Path]:
    """Scan for context note files."""
    context_dir = root / "docs" / "context"
    if not context_dir.exists():
        return []
    return sorted(
        path
        for path in context_dir.rglob("*.md")
        if path.is_file() and not path.name.startswith(".")
    )


def _scan_evidence_dirs(root: Path) -> list[Path]:
    """Scan for evidence bundle directories."""
    evidence_dirs = []
    context_evidence = root / "docs" / "context" / "evidence"
    if context_evidence.exists():
        evidence_dirs.extend(d for d in context_evidence.iterdir() if d.is_dir())
    output_evidence = root / "output" / "benchmarks"
    if output_evidence.exists():
        evidence_dirs.extend(d for d in output_evidence.iterdir() if d.is_dir())
    return evidence_dirs


def validate_context_note(note_path: Path, root: Path) -> ValidationResult:
    """Validate a single context note for evidence promotion."""
    text = _read_text(note_path)
    claimed_tier = _detect_tier(text)

    if claimed_tier is None:
        return ValidationResult(
            context_note=str(note_path),
            claimed_tier=None,
            current_tier=None,
            transition_valid=True,
            missing_artifacts=[],
            present_artifacts=[],
            warnings=["No evidence tier detected"],
            errors=[],
            is_diagnostic_only=True,
        )

    present_artifacts, missing_artifacts = _check_artifacts(text, claimed_tier)

    is_diagnostic_only = claimed_tier == EvidenceTier.DIAGNOSTIC
    errors = []

    if not is_diagnostic_only and missing_artifacts:
        errors.append(
            f"Missing required artifacts for {claimed_tier.value}: {', '.join(missing_artifacts)}"
        )

    transition_valid = len(errors) == 0

    return ValidationResult(
        context_note=str(note_path),
        claimed_tier=claimed_tier,
        current_tier=claimed_tier,
        transition_valid=transition_valid,
        missing_artifacts=missing_artifacts,
        present_artifacts=present_artifacts,
        warnings=[],
        errors=errors,
        is_diagnostic_only=is_diagnostic_only,
    )


def validate_evidence_bundle(
    bundle_path: Path, claimed_tier: str | None = None
) -> ValidationResult:
    """Validate an evidence bundle directory."""
    evidence_files = _evidence_files(bundle_path)
    if not evidence_files:
        return ValidationResult(
            context_note=str(bundle_path),
            claimed_tier=None,
            current_tier=None,
            transition_valid=False,
            missing_artifacts=["any_evidence_files"],
            present_artifacts=[],
            warnings=[],
            errors=[f"No evidence files found in {bundle_path}"],
            is_diagnostic_only=True,
        )

    text = _combined_text(evidence_files)

    if claimed_tier:
        try:
            claimed_tier_enum = EvidenceTier.from_string(claimed_tier)
        except ValueError:
            return ValidationResult(
                context_note=str(bundle_path),
                claimed_tier=None,
                current_tier=None,
                transition_valid=False,
                missing_artifacts=[],
                present_artifacts=[],
                warnings=[],
                errors=[f"Invalid claimed tier: {claimed_tier}"],
                is_diagnostic_only=True,
            )
    else:
        claimed_tier_enum = _detect_tier(text)
        if claimed_tier_enum is None:
            claimed_tier_enum = EvidenceTier.DIAGNOSTIC

    present_artifacts, missing_artifacts = _check_artifacts(text, claimed_tier_enum)

    is_diagnostic_only = claimed_tier_enum == EvidenceTier.DIAGNOSTIC
    errors = []

    if not is_diagnostic_only and missing_artifacts:
        errors.append(
            f"Missing required artifacts for {claimed_tier_enum.value}: {', '.join(missing_artifacts)}"
        )

    return ValidationResult(
        context_note=str(bundle_path),
        claimed_tier=claimed_tier_enum,
        current_tier=claimed_tier_enum,
        transition_valid=len(errors) == 0,
        missing_artifacts=missing_artifacts,
        present_artifacts=present_artifacts,
        warnings=[],
        errors=errors,
        is_diagnostic_only=is_diagnostic_only,
    )


def validate_all(root: Path) -> dict[str, Any]:
    """Validate all context notes and evidence bundles."""
    results = []
    diagnostic_only_results = []

    for note_path in _scan_context_notes(root):
        result = validate_context_note(note_path, root)
        results.append(result)
        if result.is_diagnostic_only:
            diagnostic_only_results.append(result)

    for bundle_path in _scan_evidence_dirs(root):
        result = validate_evidence_bundle(bundle_path)
        results.append(result)
        if result.is_diagnostic_only:
            diagnostic_only_results.append(result)

    return {
        "validation_summary": {
            "total_validated": len(results),
            "passed": sum(1 for r in results if r.transition_valid),
            "failed": sum(1 for r in results if not r.transition_valid),
            "diagnostic_only": len(diagnostic_only_results),
        },
        "results": [
            {
                "context_note": r.context_note,
                "claimed_tier": r.claimed_tier.value if r.claimed_tier else None,
                "transition_valid": r.transition_valid,
                "missing_artifacts": r.missing_artifacts,
                "present_artifacts": r.present_artifacts,
                "errors": r.errors,
                "warnings": r.warnings,
                "is_diagnostic_only": r.is_diagnostic_only,
            }
            for r in results
        ],
        "diagnostic_only_stuck": [
            {
                "context_note": r.context_note,
                "reason": "diagnostic_only_no_promotion",
            }
            for r in diagnostic_only_results
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root directory",
    )
    parser.add_argument(
        "--context-note",
        type=Path,
        default=None,
        help="Validate a single context note file",
    )
    parser.add_argument(
        "--evidence-bundle",
        type=Path,
        default=None,
        help="Validate a single evidence bundle directory",
    )
    parser.add_argument(
        "--claimed-tier",
        type=str,
        default=None,
        help=(
            "Claimed evidence tier (proposal, diagnostic, preflight, smoke, nominal, stress, "
            "paper_grade)"
        ),
    )
    parser.add_argument(
        "--current-tier",
        type=str,
        default="proposal",
        help="Current evidence tier for --check-promotion (default: proposal)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON output to this path",
    )
    parser.add_argument(
        "--check-promotion",
        action="store_true",
        help="Check if promotion from current to claimed tier is allowed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run evidence promotion gate validation."""
    args = build_parser().parse_args(argv)

    if args.check_promotion and args.claimed_tier:
        try:
            current = EvidenceTier.from_string(args.current_tier)
            claimed = EvidenceTier.from_string(args.claimed_tier)
            valid, errors = _check_transition(current, claimed)
            result = {
                "promotion_check": {
                    "current_tier": current.value,
                    "claimed_tier": claimed.value,
                    "transition_valid": valid,
                    "errors": errors,
                }
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if valid else 1
        except ValueError as e:
            print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
            return 1

    if args.context_note:
        result = validate_context_note(args.context_note, args.root)
        output = {
            "validation_result": {
                "context_note": result.context_note,
                "claimed_tier": result.claimed_tier.value if result.claimed_tier else None,
                "transition_valid": result.transition_valid,
                "missing_artifacts": result.missing_artifacts,
                "present_artifacts": result.present_artifacts,
                "errors": result.errors,
                "warnings": result.warnings,
                "is_diagnostic_only": result.is_diagnostic_only,
            }
        }
        payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if result.transition_valid else 1

    if args.evidence_bundle:
        result = validate_evidence_bundle(args.evidence_bundle, args.claimed_tier)
        output = {
            "validation_result": {
                "context_note": result.context_note,
                "claimed_tier": result.claimed_tier.value if result.claimed_tier else None,
                "transition_valid": result.transition_valid,
                "missing_artifacts": result.missing_artifacts,
                "present_artifacts": result.present_artifacts,
                "errors": result.errors,
                "warnings": result.warnings,
                "is_diagnostic_only": result.is_diagnostic_only,
            }
        }
        payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if result.transition_valid else 1

    result = validate_all(args.root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["validation_summary"]["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
