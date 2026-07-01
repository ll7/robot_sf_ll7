#!/usr/bin/env python3
"""Validate the issue #3482 release 0.0.2 recovery/closure gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDARY_MANIFEST = (
    REPO_ROOT
    / "docs/context/evidence/issue_3482_release_0_0_2_collision_count_boundary/manifest.json"
)
DEFAULT_RECOVERY_MANIFEST = (
    REPO_ROOT
    / "docs/context/evidence/issue_3482_release_0_0_2_provenance_recovery_2026_07_01"
    / "recovery_manifest.json"
)

BOUNDARY_SCHEMA = "issue_3482_release_0_0_2_collision_count_boundary.v1"
RECOVERY_SCHEMA = "issue_3482_provenance_recovery_attempt.v1"
REPORT_SCHEMA = "issue_3482_recovery_gate_report.v1"
BLOCKED_RESULT = "blocked_no_exact_event_provenance_found"
BLOCKED_COLLISION_COUNT_STATUS = "blocked_pending_artifact_promotion_and_table_annotation"
REQUIRED_SEARCHED_ARTIFACTS = {
    "backfill_summary.json",
    "frozen_reconciliation_report.json",
    "reconciliation_tables_0_0_2.jsonl",
}
REQUIRED_RESOLUTION_PATHS = {
    "recover original three artifacts",
    "recover raw episode-level records with exact-event fields",
    "explicitly downgrade or withdraw release 0.0.2 collision-count claims",
}
REQUIRED_MUST_NOT_CLAIM = {
    "release 0.0.2 total_collision_count is paper-ready",
    "the public release bundle alone proves 241 exact collision outcomes",
    "issue 3482 is completed by this negative recovery record",
}


@dataclass(frozen=True, slots=True)
class GateViolation:
    """One fail-closed recovery gate violation."""

    field: str
    message: str


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path}: expected JSON object"
        raise ValueError(msg)
    return payload


def _validate_boundary(boundary: dict[str, Any]) -> list[GateViolation]:  # noqa: C901
    violations: list[GateViolation] = []
    if boundary.get("schema_version") != BOUNDARY_SCHEMA:
        violations.append(GateViolation("boundary.schema_version", f"expected {BOUNDARY_SCHEMA}"))
    if boundary.get("issue") != 3482:
        violations.append(GateViolation("boundary.issue", "expected issue 3482"))
    if boundary.get("release_tag") != "0.0.2":
        violations.append(GateViolation("boundary.release_tag", "expected release 0.0.2"))

    boundaries = boundary.get("claim_boundaries")
    if not isinstance(boundaries, dict):
        violations.append(GateViolation("boundary.claim_boundaries", "expected object"))
        boundaries = {}
    if boundaries.get("collision_count_metric_status") != BLOCKED_COLLISION_COUNT_STATUS:
        violations.append(
            GateViolation(
                "boundary.claim_boundaries.collision_count_metric_status",
                f"expected {BLOCKED_COLLISION_COUNT_STATUS}",
            )
        )

    summary = boundary.get("diagnostic_summary")
    if not isinstance(summary, dict):
        violations.append(GateViolation("boundary.diagnostic_summary", "expected object"))
        return violations
    if summary.get("total_episodes") != 987:
        violations.append(
            GateViolation("boundary.diagnostic_summary.total_episodes", "expected 987")
        )
    if summary.get("exact_collision_events") != 241:
        violations.append(
            GateViolation("boundary.diagnostic_summary.exact_collision_events", "expected 241")
        )
    if summary.get("derived_collision_count_positive_episodes") != 0:
        violations.append(
            GateViolation(
                "boundary.diagnostic_summary.derived_collision_count_positive_episodes",
                "expected 0",
            )
        )
    if summary.get("reconciliation_violations") != 241:
        violations.append(
            GateViolation("boundary.diagnostic_summary.reconciliation_violations", "expected 241")
        )
    return violations


def _validate_recovery(recovery: dict[str, Any]) -> list[GateViolation]:  # noqa: C901
    violations: list[GateViolation] = []
    if recovery.get("schema_version") != RECOVERY_SCHEMA:
        violations.append(GateViolation("recovery.schema_version", f"expected {RECOVERY_SCHEMA}"))
    if recovery.get("issue") != 3482:
        violations.append(GateViolation("recovery.issue", "expected issue 3482"))
    if recovery.get("result") != BLOCKED_RESULT:
        violations.append(GateViolation("recovery.result", f"expected {BLOCKED_RESULT}"))
    if recovery.get("found_valid_exact_event_provenance") is not False:
        violations.append(
            GateViolation(
                "recovery.found_valid_exact_event_provenance",
                "must stay false until original artifacts or raw exact-event inputs are recovered",
            )
        )
    if recovery.get("acceptable_regeneration_input_found") is not False:
        violations.append(
            GateViolation(
                "recovery.acceptable_regeneration_input_found",
                "must stay false until raw exact-event records are recovered",
            )
        )

    searched_artifacts = set(recovery.get("searched_artifacts") or ())
    if not REQUIRED_SEARCHED_ARTIFACTS.issubset(searched_artifacts):
        missing = sorted(REQUIRED_SEARCHED_ARTIFACTS - searched_artifacts)
        violations.append(
            GateViolation("recovery.searched_artifacts", f"missing required artifacts: {missing}")
        )

    remaining_paths = set(recovery.get("remaining_resolution_paths") or ())
    if not REQUIRED_RESOLUTION_PATHS.issubset(remaining_paths):
        missing = sorted(REQUIRED_RESOLUTION_PATHS - remaining_paths)
        violations.append(
            GateViolation("recovery.remaining_resolution_paths", f"missing paths: {missing}")
        )

    must_not_claim = set(recovery.get("must_not_claim") or ())
    if not REQUIRED_MUST_NOT_CLAIM.issubset(must_not_claim):
        missing = sorted(REQUIRED_MUST_NOT_CLAIM - must_not_claim)
        violations.append(GateViolation("recovery.must_not_claim", f"missing claims: {missing}"))

    public_policy = recovery.get("public_release_bundle_policy")
    if not isinstance(public_policy, dict):
        violations.append(GateViolation("recovery.public_release_bundle_policy", "expected object"))
    elif public_policy.get("may_be_used_to_close_3482") is not False:
        violations.append(
            GateViolation(
                "recovery.public_release_bundle_policy.may_be_used_to_close_3482",
                "public release bundle alone must not close issue 3482",
            )
        )
    return violations


def _validate_cross_manifest(
    boundary: dict[str, Any], recovery: dict[str, Any]
) -> list[GateViolation]:
    boundary_summary = boundary.get("diagnostic_summary")
    recovery_summary = recovery.get("known_diagnostic_summary_from_issue_comment")
    if not isinstance(boundary_summary, dict) or not isinstance(recovery_summary, dict):
        return []

    checks = {
        "total_episodes": "episode_rows",
        "exact_collision_events": "exact_collision_outcomes",
        "derived_collision_count_positive_episodes": "derived_total_collision_count_positive_rows",
        "reconciliation_violations": "reconciliation_violations",
    }
    violations: list[GateViolation] = []
    if boundary.get("release_tag") != recovery_summary.get("release_tag"):
        violations.append(
            GateViolation(
                "cross_manifest.release_tag",
                (
                    f"boundary has {boundary.get('release_tag')!r}, "
                    f"recovery has {recovery_summary.get('release_tag')!r}"
                ),
            )
        )
    for boundary_key, recovery_key in checks.items():
        if boundary_summary.get(boundary_key) != recovery_summary.get(recovery_key):
            violations.append(
                GateViolation(
                    f"cross_manifest.{boundary_key}",
                    (
                        f"boundary has {boundary_summary.get(boundary_key)!r}, "
                        f"recovery has {recovery_summary.get(recovery_key)!r}"
                    ),
                )
            )
    return violations


def validate_gate(boundary: dict[str, Any], recovery: dict[str, Any]) -> list[GateViolation]:
    """Return structural and semantic recovery-gate violations."""

    violations: list[GateViolation] = []
    violations.extend(_validate_boundary(boundary))
    violations.extend(_validate_recovery(recovery))
    violations.extend(_validate_cross_manifest(boundary, recovery))
    return violations


def build_report(boundary_manifest: Path, recovery_manifest: Path) -> dict[str, Any]:
    """Build deterministic issue #3482 recovery gate report."""

    boundary = _load_json_object(boundary_manifest)
    recovery = _load_json_object(recovery_manifest)
    violations = validate_gate(boundary, recovery)
    return {
        "schema_version": REPORT_SCHEMA,
        "issue": 3482,
        "status": "invalid" if violations else "blocked",
        "close_ready": False,
        "decision": (
            "invalid_recovery_gate"
            if violations
            else "blocked_pending_exact_event_provenance_or_claim_downgrade"
        ),
        "boundary_manifest": str(boundary_manifest),
        "recovery_manifest": str(recovery_manifest),
        "release_tag": boundary.get("release_tag"),
        "diagnostic_summary": boundary.get("diagnostic_summary"),
        "remaining_resolution_paths": recovery.get("remaining_resolution_paths", []),
        "must_not_claim": recovery.get("must_not_claim", []),
        "violations": [asdict(violation) for violation in violations],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-manifest", type=Path, default=DEFAULT_BOUNDARY_MANIFEST)
    parser.add_argument("--recovery-manifest", type=Path, default=DEFAULT_RECOVERY_MANIFEST)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report.")
    parser.add_argument(
        "--require-close-ready",
        action="store_true",
        help="Return non-zero unless issue #3482 is ready to close.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line recovery gate validator."""

    args = _parse_args(argv)
    report = build_report(args.boundary_manifest, args.recovery_manifest)

    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif report["status"] == "blocked":
        print(f"issue #3482 recovery gate valid but blocked: {report['decision']}")
    else:
        print("issue #3482 recovery gate invalid", file=sys.stderr)

    for violation in report["violations"]:
        print(f"{violation['field']}: {violation['message']}", file=sys.stderr)

    if report["status"] != "blocked":
        return 1
    if args.require_close_ready and not report["close_ready"]:
        print(report["decision"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
