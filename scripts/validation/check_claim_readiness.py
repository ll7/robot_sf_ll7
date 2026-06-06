#!/usr/bin/env python3
"""Report whether compact evidence contains the fields needed for a stated claim."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
READY_STATUS_BY_CLASS = {
    "diagnostic": "ready_for_diagnostic_review",
    "blocked": "ready_for_blocked_review",
    "fallback_or_degraded": "ready_for_limitation_review",
    "benchmark": "ready_for_benchmark_review",
    "paper_facing": "ready_for_paper_facing_review",
}
SCHEMA_VERSION = "claim_readiness.v1"


@dataclass(frozen=True)
class ReadinessField:
    """One readiness field and its text markers."""

    name: str
    description: str
    patterns: tuple[re.Pattern[str], ...]

    def is_present(self, text: str) -> bool:
        """Return whether this field is present in a combined claim/evidence text."""
        return any(pattern.search(text) for pattern in self.patterns)


READINESS_FIELDS = (
    ReadinessField(
        name="evidence_tier",
        description="Evidence tier or class is explicit.",
        patterns=(
            re.compile(r"\bevidence[_ -]?tier\s*[:=]"),
            re.compile(r"\bclaim[_ -]?scope\s*[:=]\s*diagnostic[_ -]?only\b"),
            re.compile(r"\bevidence[_ -]?tier\s*[:=]\s*benchmark[-_ ]strength\b"),
            re.compile(r"\bevidence[_ -]?tier\s*[:=]\s*paper[-_ ]facing\b"),
        ),
    ),
    ReadinessField(
        name="comparator_or_baseline",
        description="Comparator, baseline, or control surface is named.",
        patterns=(
            re.compile(r"\bcompar(?:ator|ison)\s*[:=]"),
            re.compile(r"\bbaseline\s*[:=]"),
            re.compile(r"\bcontrol\s*[:=]"),
            re.compile(r"\bvs\.?\s+[^.\n]+"),
        ),
    ),
    ReadinessField(
        name="mechanism_activation",
        description="Mechanism activation or mechanism-signal surface is described.",
        patterns=(
            re.compile(r"\bmechanism[_ -]?activation\s*[:=]"),
            re.compile(r"\bactivation[_ -]?count\s*[:=]"),
            re.compile(r"\bmechanism[_ -]?signal\s*[:=]"),
            re.compile(r'"(?:activation_count|mechanism_signal)"\s*:'),
        ),
    ),
    ReadinessField(
        name="trace_support",
        description="Trace, frame, step, or trajectory support is identified.",
        patterns=(
            re.compile(r"\btrace[_ -]?support\s*[:=]"),
            re.compile(r"\bsimulation_trace\b"),
            re.compile(r"\btrace[_ -]?(?:file|path|frames?)\s*[:=]"),
            re.compile(r'"(?:frames|steps|trajectory)"\s*:'),
        ),
    ),
    ReadinessField(
        name="artifact_provenance",
        description="Command, commit, checksum, or artifact provenance is present.",
        patterns=(
            re.compile(r"\bartifact[_ -]?provenance\s*[:=]"),
            re.compile(r"\bcommand\s*[:=]"),
            re.compile(r"\bcommit\s*[:=]"),
            re.compile(r"\bsha-?256\s*[:=]"),
            re.compile(r"\bchecksum\s*[:=]"),
            re.compile(r'"(?:command|commit|sha256|checksum)"\s*:'),
            re.compile(r"\bevidence_bundle_manifest\b"),
        ),
    ),
    ReadinessField(
        name="seed_slice_boundary",
        description="Seed, scenario, slice, family, or horizon boundary is stated.",
        patterns=(
            re.compile(r"\bseed[/_ -]?slice[_ -]?boundary\s*[:=]"),
            re.compile(r"\bscenario\s*[:=]"),
            re.compile(r"\bseed\s*[:=]"),
            re.compile(r"\bhorizon\s*[:=]"),
            re.compile(r'"(?:scenario|seed|horizon|slice)"\s*:'),
        ),
    ),
    ReadinessField(
        name="claim_boundary",
        description="Claim boundary or claim scope is explicit.",
        patterns=(
            re.compile(r"\bclaim[_ -]?boundary\s*[:=]"),
            re.compile(r"\bclaim[_ -]?scope\s*[:=]"),
            re.compile(r"\bdoes not establish\b"),
            re.compile(r"\bnot benchmark(?: evidence)?\b"),
            re.compile(r"\bnot paper(?:[-_ ]facing|[-_ ]grade)?\b"),
        ),
    ),
    ReadinessField(
        name="fallback_degraded_limitations",
        description="Fallback, degraded, unavailable, or fail-closed limitations are handled.",
        patterns=(
            re.compile(r"\bfallback[/_ -]?degraded[_ -]?limitations\s*[:=]"),
            re.compile(r"\bfallback[_ -]?or[_ -]?degraded\s*[:=]"),
            re.compile(r"\bfail[- ]closed\s*[:=]"),
            re.compile(r"\bnot[_ -]?available\s*[:=]"),
            re.compile(r"\bnot benchmark evidence\b"),
        ),
    ),
)


def _read_text(path: Path) -> str:
    """Read a text-like file with replacement for odd encodings."""
    return path.read_text(encoding="utf-8", errors="replace")


def _evidence_files(path: Path) -> list[Path]:
    """Return compact evidence files to scan."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Evidence path does not exist: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES
    )


def _combined_text(claim_file: Path, evidence_path: Path) -> tuple[str, str, list[Path]]:
    """Return normalized combined claim/evidence text and scanned evidence files."""
    if not claim_file.is_file():
        raise FileNotFoundError(f"Claim file does not exist: {claim_file}")
    evidence_files = _evidence_files(evidence_path)
    claim_text = _read_text(claim_file).lower()
    chunks = [claim_text]
    chunks.extend(_read_text(path) for path in evidence_files)
    return claim_text, "\n".join(chunks).lower(), evidence_files


def _claim_class(text: str) -> str:
    """Return the strongest apparent claim class in the scanned material."""
    if re.search(r"\bpaper[-_ ]facing\b|\bpaper[-_ ]grade\b|\bmanuscript\b", text):
        return "paper_facing"
    if re.search(r"\bbenchmark[-_ ]strength\b|\bbenchmark[_ -]?success\b", text):
        return "benchmark"
    if re.search(r"\bdiagnostic[_ -]?only\b|\bdiagnostic\b", text):
        return "diagnostic"
    if re.search(r"\bfallback\b|\bdegraded\b", text):
        return "fallback_or_degraded"
    if re.search(r"\bblocked\b|\bnot[_ -]?available\b|\bfail[- ]closed\b", text):
        return "blocked"
    return "diagnostic"


def _warnings_for_claim_class(claim_class: str, text: str) -> list[str]:
    """Return conservative caveat warnings for the detected claim class."""
    warnings: list[str] = []
    if claim_class in {"benchmark", "paper_facing"} and re.search(
        r"\bfallback\b|\bdegraded\b|\bdiagnostic[_ -]?only\b",
        text,
    ):
        warnings.append(
            "benchmark_or_paper_claim_mentions_fallback_degraded_or_diagnostic_only_evidence"
        )
    if claim_class == "paper_facing" and "sha256" not in text and "checksum" not in text:
        warnings.append("paper_facing_claim_without_checksum_marker")
    return warnings


def evaluate_claim_readiness(claim_file: Path, evidence_path: Path) -> dict[str, Any]:
    """Evaluate whether claim/evidence material contains readiness guardrail fields."""
    claim_text, text, evidence_files = _combined_text(claim_file, evidence_path)
    present_fields = [field.name for field in READINESS_FIELDS if field.is_present(text)]
    missing_fields = [field.name for field in READINESS_FIELDS if field.name not in present_fields]
    claim_class = _claim_class(claim_text)
    warnings = _warnings_for_claim_class(claim_class, text)
    if missing_fields:
        status = "not_ready_missing_fields"
    elif warnings:
        status = "not_ready_claim_boundary_warning"
    else:
        status = READY_STATUS_BY_CLASS[claim_class]
    return {
        "claim_readiness": {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "claim_class": claim_class,
            "present_fields": present_fields,
            "missing_fields": missing_fields,
            "warnings": warnings,
            "claim_file": str(claim_file),
            "evidence_path": str(evidence_path),
            "scanned_evidence_files": [str(path) for path in evidence_files],
            "claim_boundary": (
                "Guardrail and missing-field report only; readiness does not establish "
                "scientific truth, benchmark success, safety, or paper-grade sufficiency."
            ),
        }
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-file", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run claim readiness checks."""
    args = build_parser().parse_args(argv)
    result = evaluate_claim_readiness(args.claim_file, args.evidence)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["claim_readiness"]["status"].startswith("ready_for_") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
