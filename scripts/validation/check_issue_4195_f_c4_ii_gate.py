#!/usr/bin/env python3
"""Fail-closed validator for the issue #4195 h600 F-C4(ii) interpretation gate.

This is an *integration* checker, not another packet builder. It proves that the
committed h600 interpretation bundle stays internally consistent after the
pre-registered hybrid-roster leg (job 13282) was integrated:

1. every evidence file in the bundle directory is checksummed in ``SHA256SUMS``
   (this closed the gap where ``hybrid_roster_h600_transfer_packet.md`` was
   committed but never listed) and each recorded digest matches the file on disk;
2. the F-C4(ii) gate note carries the required diagnostic boundary sections, the
   ``author_signoff: OPEN`` marker, and a reference to the hybrid packet;
3. the hybrid packet, the gate note, and the aggregation source manifest agree on
   the shared scenario-matrix hash — the comparability precondition for reading
   the gate across all three campaign legs.

Any violation exits non-zero with an explicit message. Missing or malformed
inputs fail closed; nothing is inferred or repaired by assumption.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_3810_h600_interpretation_2026-07"

REPORT_SCHEMA = "issue_4195_f_c4_ii_gate_report.v1"

GATE_NOTE_NAME = "f_c4_ii_interpretation_gate.md"
HYBRID_PACKET_NAME = "hybrid_roster_h600_transfer_packet.md"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SHA256SUMS_NAME = "SHA256SUMS"

# Evidence file extensions the SHA256SUMS ledger must cover. SHA256SUMS itself is
# not self-referential, so it is excluded from the coverage requirement.
CHECKSUMMED_SUFFIXES = {".md", ".json", ".csv"}

# Required substrings in the gate note. Kept as human-readable anchors so the
# checker fails when a boundary section is dropped during an edit.
REQUIRED_GATE_MARKERS = (
    "F-C4(ii)",
    "### SUPPORTED",
    "### DIAGNOSTIC-ONLY",
    "### NOT SUPPORTED",
    "## Integration report",
    HYBRID_PACKET_NAME,
)

# The gate note must carry an explicit, recognized sign-off status. It is not
# pinned to OPEN: the maintainer recorded sign-off on 2026-07-03, so the checker
# requires the status field with one of the known values rather than a fixed one.
SIGNOFF_MARKER = "author_signoff:"
VALID_SIGNOFF_VALUES = ("OPEN", "RECORDED", "PROMOTED")


@dataclass(frozen=True, slots=True)
class GateViolation:
    """One fail-closed gate violation."""

    field: str
    message: str


def _parse_sha256sums(text: str) -> dict[str, str]:
    """Parse ``sha256sum``-style lines into ``{relative_path: digest}``.

    Paths are recorded relative to the repository root; only the file's basename
    is used as the lookup key so the checker is independent of the ledger's path
    prefix.
    """
    mapping: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"malformed SHA256SUMS line: {raw!r}")
        digest, rel_path = parts
        mapping[Path(rel_path).name] = digest
    return mapping


def _extract_matrix_hash(text: str) -> set[str]:
    """Extract 12-hex scenario-matrix hashes referenced as ``c10df617a87c``."""
    return set(re.findall(r"\b([0-9a-f]{12})\b", text))


def _check_sha256sums(
    evidence_dir: Path, sums_path: Path, facts: dict[str, Any]
) -> list[GateViolation]:
    """Every evidence file must be listed in SHA256SUMS with a matching digest."""
    try:
        recorded = _parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [GateViolation("sha256sums", str(exc))]

    violations: list[GateViolation] = []
    evidence_files = sorted(
        p for p in evidence_dir.iterdir() if p.is_file() and p.suffix in CHECKSUMMED_SUFFIXES
    )
    facts["checksummed_file_count"] = len(evidence_files)
    for path in evidence_files:
        if path.name not in recorded:
            violations.append(
                GateViolation(
                    "sha256sums_coverage", f"{path.name} is not listed in {SHA256SUMS_NAME}"
                )
            )
            continue
        actual = _sha256(path)
        if actual != recorded[path.name]:
            violations.append(
                GateViolation(
                    "sha256sums_digest",
                    f"{path.name} digest mismatch: recorded {recorded[path.name]}, actual {actual}",
                )
            )
    return violations


def _check_gate_markers(gate_text: str) -> list[GateViolation]:
    """The gate note must carry every required anchor and a recognized sign-off."""
    violations = [
        GateViolation("gate_note_marker", f"gate note missing marker: {marker!r}")
        for marker in REQUIRED_GATE_MARKERS
        if marker not in gate_text
    ]
    signoff = _extract_signoff_value(gate_text)
    if signoff is None:
        violations.append(
            GateViolation("gate_note_signoff", f"gate note missing {SIGNOFF_MARKER!r} status")
        )
    elif signoff not in VALID_SIGNOFF_VALUES:
        violations.append(
            GateViolation(
                "gate_note_signoff",
                f"unrecognized sign-off value {signoff!r}; expected one of {VALID_SIGNOFF_VALUES}",
            )
        )
    return violations


def _extract_signoff_value(gate_text: str) -> str | None:
    """Return the first token after ``author_signoff:``; ``None`` if absent."""
    match = re.search(rf"{re.escape(SIGNOFF_MARKER)}\s*`?([A-Za-z_]+)", gate_text)
    return match.group(1) if match else None


def _check_matrix_hash(
    manifest_path: Path, packet_text: str, gate_text: str, facts: dict[str, Any]
) -> list[GateViolation]:
    """Runs, hybrid packet, and gate note must agree on one scenario-matrix hash."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs = manifest.get("runs", [])
        run_hashes = {
            r.get("campaign", {}).get("scenario_matrix_hash") for r in runs if isinstance(r, dict)
        }
        run_hashes.discard(None)
    except (ValueError, AttributeError) as exc:
        return [GateViolation("source_manifest", f"unreadable: {exc}")]

    facts["run_matrix_hashes"] = sorted(run_hashes)
    if len(run_hashes) != 1:
        return [
            GateViolation(
                "matrix_hash",
                f"aggregation runs do not share a single scenario_matrix_hash: {sorted(run_hashes)}",
            )
        ]

    shared = next(iter(run_hashes))
    facts["shared_matrix_hash"] = shared
    violations: list[GateViolation] = []
    if shared not in _extract_matrix_hash(packet_text):
        violations.append(
            GateViolation("matrix_hash", f"hybrid packet does not cite shared matrix hash {shared}")
        )
    if shared not in _extract_matrix_hash(gate_text):
        violations.append(
            GateViolation("matrix_hash", f"gate note does not cite shared matrix hash {shared}")
        )
    return violations


def check_gate(evidence_dir: Path) -> tuple[list[GateViolation], dict[str, Any]]:
    """Run the fail-closed F-C4(ii) gate checks against ``evidence_dir``."""
    facts: dict[str, Any] = {"evidence_dir": str(evidence_dir)}

    if not evidence_dir.is_dir():
        return (
            [GateViolation("evidence_dir", f"missing evidence directory: {evidence_dir}")],
            facts,
        )

    sums_path = evidence_dir / SHA256SUMS_NAME
    gate_path = evidence_dir / GATE_NOTE_NAME
    packet_path = evidence_dir / HYBRID_PACKET_NAME
    manifest_path = evidence_dir / SOURCE_MANIFEST_NAME

    # Required files must all exist before any content check is meaningful.
    missing = [
        GateViolation(label, f"missing required file: {path.name}")
        for label, path in (
            ("sha256sums", sums_path),
            ("gate_note", gate_path),
            ("hybrid_packet", packet_path),
            ("source_manifest", manifest_path),
        )
        if not path.is_file()
    ]
    if missing:
        return missing, facts

    gate_text = gate_path.read_text(encoding="utf-8")
    packet_text = packet_path.read_text(encoding="utf-8")

    violations = _check_sha256sums(evidence_dir, sums_path, facts)
    violations += _check_gate_markers(gate_text)
    violations += _check_matrix_hash(manifest_path, packet_text, gate_text, facts)
    return violations, facts


def build_report(evidence_dir: Path) -> dict[str, Any]:
    """Run the gate and return a JSON-serializable report."""
    violations, facts = check_gate(evidence_dir)
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not violations else "fail",
        "evidence_dir": str(evidence_dir),
        "facts": facts,
        "violations": [asdict(v) for v in violations],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exits non-zero when the gate fails closed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="h600 interpretation evidence directory to validate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON on stdout.",
    )
    args = parser.parse_args(argv)

    report = build_report(args.evidence_dir.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"F-C4(ii) gate: {report['status']}")
        for violation in report["violations"]:
            print(f"  - {violation['field']}: {violation['message']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
