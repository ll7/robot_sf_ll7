#!/usr/bin/env python3
"""Validate the release 0.0.2 collision-count claim boundary for issue #3482."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "docs/context/evidence/issue_3482_release_0_0_2_collision_count_boundary/manifest.json"
)
SCHEMA_VERSION = "issue_3482_release_0_0_2_collision_count_boundary.v1"
EXPECTED_RELEASE = "0.0.2"
EXPECTED_COLLISION_COUNT_STATUS = "withdrawn_exact_event_provenance_unavailable"
EXPECTED_EXACT_OUTCOME_STATUS = "bounded_diagnostic_only"
REQUIRED_OPEN_GATES: set[str] = set()


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    """One manifest boundary violation."""

    field: str
    message: str


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path}: expected JSON object"
        raise ValueError(msg)
    return payload


def _validate_identity(payload: dict[str, Any]) -> list[BoundaryViolation]:
    """Validate manifest identity fields."""
    violations: list[BoundaryViolation] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        violations.append(
            BoundaryViolation(
                "schema_version",
                f"expected {SCHEMA_VERSION}",
            )
        )
    if payload.get("issue") != 3482:
        violations.append(BoundaryViolation("issue", "expected issue 3482"))
    if payload.get("release_tag") != EXPECTED_RELEASE:
        violations.append(BoundaryViolation("release_tag", "expected release 0.0.2"))
    return violations


def _validate_diagnostic_summary(payload: dict[str, Any]) -> list[BoundaryViolation]:
    """Validate diagnostic summary fields that keep the discrepancy explicit."""
    violations: list[BoundaryViolation] = []
    evidence = payload.get("diagnostic_summary")
    if not isinstance(evidence, dict):
        violations.append(BoundaryViolation("diagnostic_summary", "expected object"))
        evidence = {}

    exact_collision_events = evidence.get("exact_collision_events")
    derived_collision_count_positive_episodes = evidence.get(
        "derived_collision_count_positive_episodes"
    )
    reconciliation_violations = evidence.get("reconciliation_violations")
    total_episodes = evidence.get("total_episodes")
    if not isinstance(total_episodes, int) or total_episodes <= 0:
        violations.append(BoundaryViolation("diagnostic_summary.total_episodes", "must be > 0"))
    if not isinstance(exact_collision_events, int) or exact_collision_events <= 0:
        violations.append(
            BoundaryViolation("diagnostic_summary.exact_collision_events", "must be > 0")
        )
    if derived_collision_count_positive_episodes != 0:
        violations.append(
            BoundaryViolation(
                "diagnostic_summary.derived_collision_count_positive_episodes",
                "expected 0 for the documented release 0.0.2 discrepancy",
            )
        )
    if reconciliation_violations != exact_collision_events:
        violations.append(
            BoundaryViolation(
                "diagnostic_summary.reconciliation_violations",
                "must equal exact collision events until the release discrepancy is resolved",
            )
        )
    return violations


def _validate_claim_boundaries(payload: dict[str, Any]) -> list[BoundaryViolation]:
    """Validate terminal claim boundary and open-gate fields."""
    violations: list[BoundaryViolation] = []
    boundaries = payload.get("claim_boundaries")
    if not isinstance(boundaries, dict):
        violations.append(BoundaryViolation("claim_boundaries", "expected object"))
        boundaries = {}
    if boundaries.get("exact_collision_outcome_status") != EXPECTED_EXACT_OUTCOME_STATUS:
        violations.append(
            BoundaryViolation(
                "claim_boundaries.exact_collision_outcome_status",
                f"expected {EXPECTED_EXACT_OUTCOME_STATUS}",
            )
        )
    if boundaries.get("collision_count_metric_status") != EXPECTED_COLLISION_COUNT_STATUS:
        violations.append(
            BoundaryViolation(
                "claim_boundaries.collision_count_metric_status",
                f"expected {EXPECTED_COLLISION_COUNT_STATUS}",
            )
        )

    open_gates = set(boundaries.get("open_gates") or ())
    if open_gates != REQUIRED_OPEN_GATES:
        violations.append(
            BoundaryViolation(
                "claim_boundaries.open_gates",
                "withdrawn collision-count claims must not advertise open promotion gates",
            )
        )
    return violations


def _validate_forbidden_claims(payload: dict[str, Any]) -> list[BoundaryViolation]:
    """Validate blocked claim list."""
    violations: list[BoundaryViolation] = []
    forbidden = payload.get("forbidden_claims_until_gates_close")
    if not isinstance(forbidden, list) or not forbidden:
        violations.append(
            BoundaryViolation(
                "forbidden_claims_until_gates_close",
                "must list blocked paper-facing collision-count claims",
            )
        )
    return violations


def validate_manifest(payload: dict[str, Any]) -> list[BoundaryViolation]:
    """Return fail-closed validation errors for the release collision boundary manifest."""
    violations: list[BoundaryViolation] = []
    violations.extend(_validate_identity(payload))
    violations.extend(_validate_diagnostic_summary(payload))
    violations.extend(_validate_claim_boundaries(payload))
    violations.extend(_validate_forbidden_claims(payload))

    return violations


def build_report(manifest_path: Path) -> dict[str, Any]:
    """Build the deterministic release collision-boundary report."""
    payload = _load_manifest(manifest_path)
    violations = validate_manifest(payload)
    return {
        "schema_version": "release_0_0_2_collision_count_boundary_report.v1",
        "manifest": str(manifest_path),
        "status": "pass" if not violations else "fail",
        "issue": payload.get("issue"),
        "release_tag": payload.get("release_tag"),
        "collision_count_metric_status": (
            payload.get("claim_boundaries", {}).get("collision_count_metric_status")
            if isinstance(payload.get("claim_boundaries"), dict)
            else None
        ),
        "violations": [asdict(violation) for violation in violations],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the tracked release collision-count boundary manifest."""
    args = _parse_args(argv)
    report = build_report(args.manifest)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] == "pass":
        print(
            "release 0.0.2 collision-count boundary valid: "
            f"{report['collision_count_metric_status']}"
        )
    else:
        for violation in report["violations"]:
            print(f"{violation['field']}: {violation['message']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
