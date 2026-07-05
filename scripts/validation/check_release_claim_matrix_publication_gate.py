#!/usr/bin/env python3
"""Fail-closed publication gate for the v0.1 release claim matrix.

The gate is intentionally narrow: it checks the existing Issue #3294 release
claim matrix against Issue #2910 publication prerequisites. It does not run a
benchmark campaign, create certification records, or promote any claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_MATRIX_PATH = Path(
    "docs/context/evidence/issue_3294_release_claim_matrix/release_claim_matrix.json"
)
SCHEMA_VERSION = "release-claim-matrix-publication-gate.v1"
EXPECTED_MATRIX_SCHEMA = "release_claim_matrix_issue_3294.v1"
BENCHMARK_EVIDENCE_CLASSIFICATION = "benchmark evidence"
KNOWN_NON_BENCHMARK_CLASSIFICATIONS = frozenset({"diagnostic evidence", "non-claim"})
NOT_CERTIFIED_VALUES = {
    "",
    "missing",
    "not_available",
    "not_available_in_manifest",
    "not_recorded",
    "unknown",
}


@dataclass(frozen=True)
class GateBlocker:
    """One fail-closed blocker found in a release claim matrix row."""

    row_id: str
    check: str
    severity: str
    reason: str
    next_action: str


def get_repository_root() -> Path:
    """Return repository root for default matrix and artifact path resolution."""

    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _row_id(row: dict[str, Any], index: int) -> str:
    """Return stable row identifier for reports."""

    return str(row.get("row_id") or f"row:{index:03d}")


def _repo_relative_file_exists(repo_root: Path, value: object) -> bool:
    """Return whether ``value`` is a repository-relative regular file."""

    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    # ``Path(".").parts`` is empty, so guard the index before the ``output`` check
    # to keep pathological values (for example ``"."``) fail-closed instead of
    # raising ``IndexError`` inside the publication gate.
    if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0] == "output"):
        return False
    return (repo_root / path).is_file()


def _check_benchmark_evidence_row(
    row: dict[str, Any],
    *,
    row_id: str,
    repo_root: Path,
) -> list[GateBlocker]:
    """Check publication prerequisites for one benchmark-evidence row."""

    blockers: list[GateBlocker] = []
    if row.get("availability_status") != "available":
        blockers.append(
            GateBlocker(
                row_id=row_id,
                check="availability",
                severity="blocker",
                reason="benchmark-evidence row is not marked available",
                next_action="Provide a durable available artifact or downgrade the row.",
            )
        )
    if row.get("artifact_match") is not True:
        blockers.append(
            GateBlocker(
                row_id=row_id,
                check="artifact_match",
                severity="blocker",
                reason="benchmark-evidence row does not verify an artifact match",
                next_action="Refresh the release evidence snapshot or exclude the row.",
            )
        )
    if not _repo_relative_file_exists(repo_root, row.get("artifact_uri")):
        blockers.append(
            GateBlocker(
                row_id=row_id,
                check="artifact_uri",
                severity="blocker",
                reason="artifact_uri is missing, non-file, absolute, escaping, or under output/",
                next_action="Promote a durable artifact pointer before publication.",
            )
        )
    certification = str(row.get("scenario_certification", "")).strip().lower()
    if certification in NOT_CERTIFIED_VALUES:
        blockers.append(
            GateBlocker(
                row_id=row_id,
                check="scenario_certification",
                severity="blocker",
                reason=f"scenario certification is {certification or 'missing'}",
                next_action="Attach scenario_cert.v1 status or keep the row out of the v0.1 publication set.",
            )
        )
    source_refs = row.get("source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        blockers.append(
            GateBlocker(
                row_id=row_id,
                check="source_refs",
                severity="blocker",
                reason="benchmark-evidence row has no source_refs provenance",
                next_action="Record tracked provenance sources for the row.",
            )
        )
    elif any(not _repo_relative_file_exists(repo_root, source_ref) for source_ref in source_refs):
        blockers.append(
            GateBlocker(
                row_id=row_id,
                check="source_refs",
                severity="blocker",
                reason="one or more source_refs are missing or not tracked files",
                next_action="Fix source_refs to repository files or remove the row.",
            )
        )
    return blockers


def build_gate_report(matrix: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Build deterministic publication-gate report for a release claim matrix."""

    matrix_schema = matrix.get("schema_version")
    if matrix_schema != EXPECTED_MATRIX_SCHEMA:
        raise ValueError(
            f"Matrix schema_version {matrix_schema!r}, expected {EXPECTED_MATRIX_SCHEMA!r}"
        )
    rows = matrix.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Matrix must contain a non-empty rows list")

    blockers: list[GateBlocker] = []
    benchmark_rows = 0
    diagnostic_or_non_claim_rows = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Matrix row {index} must be an object")
        row_id = _row_id(row, index)
        classification = str(row.get("classification", "")).strip().lower()
        if classification == BENCHMARK_EVIDENCE_CLASSIFICATION:
            benchmark_rows += 1
            blockers.extend(_check_benchmark_evidence_row(row, row_id=row_id, repo_root=repo_root))
        elif classification in KNOWN_NON_BENCHMARK_CLASSIFICATIONS:
            diagnostic_or_non_claim_rows += 1
            if row.get("benchmark_success") is True:
                blockers.append(
                    GateBlocker(
                        row_id=row_id,
                        check="non_benchmark_promotion",
                        severity="blocker",
                        reason="non-benchmark row sets benchmark_success=true",
                        next_action="Downgrade benchmark_success or reclassify with full evidence.",
                    )
                )
        else:
            blockers.append(
                GateBlocker(
                    row_id=row_id,
                    check="classification",
                    severity="blocker",
                    reason=f"unrecognized classification {classification or 'missing'}",
                    next_action=(
                        "Use benchmark evidence or a known non-benchmark classification before "
                        "publication."
                    ),
                )
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2910,
        "input_matrix_schema": matrix_schema,
        "input_matrix_issue": matrix.get("issue"),
        "status": "blocked" if blockers else "pass",
        "claim_boundary": (
            "CPU-only gate over existing tracked release claim matrix; no campaign, Slurm, "
            "training, release, or paper claim promotion performed."
        ),
        "summary": {
            "row_count": len(rows),
            "benchmark_evidence_rows": benchmark_rows,
            "diagnostic_or_non_claim_rows": diagnostic_or_non_claim_rows,
            "blocker_count": len(blockers),
        },
        "blockers": [asdict(blocker) for blocker in blockers],
    }


def _render_text(report: dict[str, Any]) -> str:
    """Render concise human-readable gate status."""

    summary = report["summary"]
    lines = [
        f"status: {report['status']}",
        f"rows: {summary['row_count']}",
        f"benchmark_evidence_rows: {summary['benchmark_evidence_rows']}",
        f"blockers: {summary['blocker_count']}",
    ]
    for blocker in report["blockers"]:
        lines.append(
            f"- {blocker['row_id']} [{blocker['check']}]: {blocker['reason']} "
            f"next={blocker['next_action']}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Release claim matrix JSON path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root used to resolve matrix artifact paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve() if args.repo_root else get_repository_root()
    matrix_path = args.matrix if args.matrix else repo_root / DEFAULT_MATRIX_PATH
    try:
        report = build_gate_report(_load_json(matrix_path), repo_root=repo_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
