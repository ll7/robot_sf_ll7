"""Verify the future external admission boundary for a Chapter 7 v2 package.

The builder intentionally emits a blocked package and no approval receipt.
This verifier is the fail-closed promotion path for a later maintainer-owned
receipt after domain approval has been recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from scripts.analysis import verify_ch7_evidence_admission as admission

PACKAGE_SCHEMA = (
    Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-package.v2.json"
)
RECEIPT_SCHEMA = (
    Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-admission.v2.json"
)
DIAGNOSTIC_SCHEMA_VERSION = "ch7-evidence-admission-diagnostic.v1"
V2_FORBIDDEN_CLAIMS = (
    "matched_comparison",
    "causal_divergence",
    "counterfactual_branching",
    "trajectory_divergence",
    "universal_planner_ranking",
    "collision_metric_semantics",
)


class Ch7EvidenceAdmissionV2Error(ValueError):
    """Raised when a v2 package or admission receipt fails closed."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidenceAdmissionV2Error(f"{label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidenceAdmissionV2Error(f"{label} must be an object")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(payload: Mapping[str, Any], schema_path: Path, label: str) -> None:
    try:
        schema = _read_object(schema_path, f"{label} schema")
        errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=str)
    except (TypeError, ValidationError) as exc:
        raise Ch7EvidenceAdmissionV2Error(f"{label} schema is invalid") from exc
    if errors:
        details = "; ".join(error.message for error in errors[:3])
        raise Ch7EvidenceAdmissionV2Error(f"{label} validation failed: {details}")


def _receipt_template(
    manifest: Mapping[str, Any], *, sums_sha: str, manifest_sha: str
) -> dict[str, Any]:
    """Build a non-validating receipt-shaped template from verified package metadata."""

    source = manifest["source"]
    inputs = manifest["inputs"]
    portfolio = inputs["portfolio_config"]
    roles = manifest["roles"]
    return {
        "template_status": "not_a_receipt",
        "schema_version": "ch7-evidence-admission.v2",
        "issue": 7087,
        "status": "template_only",
        "package": {
            "sha256sums_sha256": sums_sha,
            "manifest_sha256": manifest_sha,
        },
        "source": {
            "v1_package_sha256sums": source["v1_package_sha256sums"],
            "v1_manifest_sha256": source["v1_manifest_sha256"],
            "v1_audit_member_sha256": source["v1_audit_member_sha256"],
            "v1_reduced_atlas_member_sha256": source["v1_reduced_atlas_member_sha256"],
            "portfolio_config_sha256": portfolio["sha256"],
            "source_registry_sha256": None,
        },
        "approval": {
            "approval_id": None,
            "approval_url": None,
            "decision": None,
        },
        "scope": {
            "claim_boundary": manifest["claim_boundary"],
            "forbidden_claims": list(V2_FORBIDDEN_CLAIMS),
        },
        "roles": {
            "available": {role: {"grain": details["grain"]} for role, details in roles.items()}
        },
        "retrieval": {
            "source_package_key": None,
            "audit_member_key": None,
            "source_registry_key": None,
        },
    }


def diagnose_v2_package(package: Path) -> dict[str, Any]:
    """Check a blocked v2 package without creating or accepting an admission receipt."""

    try:
        sums_sha, _listed = admission._verify_members(
            package,
            label="Chapter 7 v2 evidence package",
            require_review_sidecars=False,
            allow_review_sidecars=True,
        )
    except admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidenceAdmissionV2Error(f"package member verification failed: {exc}") from exc
    package = package.resolve()
    manifest = _read_object(package / "manifest.json", "v2 package manifest")
    _validate(manifest, PACKAGE_SCHEMA, "v2 package manifest")
    if (
        manifest.get("status") != "blocked_pending_domain_approval"
        or manifest.get("admission_status") != "not_admitted"
        or manifest.get("source_integrity_gate") != "blocked_pending_domain_approval"
    ):
        raise Ch7EvidenceAdmissionV2Error(
            "check-only mode requires a blocked, not-admitted v2 package"
        )
    admission_block = manifest.get("admission")
    if (
        not isinstance(admission_block, Mapping)
        or admission_block.get("status") != "not_admitted"
        or admission_block.get("receipt_required") is not True
        or admission_block.get("receipt_schema") != "ch7-evidence-admission.v2"
    ):
        raise Ch7EvidenceAdmissionV2Error(
            "blocked v2 package does not declare the external admission boundary"
        )
    source = manifest["source"]
    portfolio = manifest["inputs"]["portfolio_config"]
    manifest_sha = _sha256_file(package / "manifest.json")
    blockers: list[dict[str, str]] = [
        {
            "code": "domain_approval_pending",
            "reason": "v2 domain approval is outside the package builder and verifier",
        },
        {
            "code": "external_admission_receipt_required",
            "reason": "a maintainer-owned ch7-evidence-admission.v2 receipt is required",
        },
    ]
    excluded = manifest["metrics"]["excluded"]
    if any(item.get("issue") == 7042 for item in excluded if isinstance(item, Mapping)):
        blockers.append(
            {
                "code": "metric_semantics_excluded_issue_7042",
                "reason": "collision-sensitive metrics and SNQI are excluded by the closed #7042 ruling",
            }
        )
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "issue": 7087,
        "status": "blocked_pending_domain_approval",
        "admission_status": "not_admitted",
        "package": {
            "sha256sums_sha256": sums_sha,
            "manifest_sha256": manifest_sha,
        },
        "source": {
            "v1_package_sha256sums": source["v1_package_sha256sums"],
            "v1_manifest_sha256": source["v1_manifest_sha256"],
            "v1_audit_member_sha256": source["v1_audit_member_sha256"],
            "v1_reduced_atlas_member_sha256": source["v1_reduced_atlas_member_sha256"],
            "portfolio_config_sha256": portfolio["sha256"],
        },
        "diagnostics": {
            "package_checksums_verified": True,
            "package_manifest_schema_verified": True,
            "admission_authorized": False,
            "empirical_outcomes_admitted": False,
            "receipt_created": False,
            "blockers": blockers,
        },
        "receipt_template": _receipt_template(
            manifest, sums_sha=sums_sha, manifest_sha=manifest_sha
        ),
    }


def verify_v2_admission(package: Path, receipt: Path) -> dict[str, Any]:
    """Verify an admitted v2 package against its exact external receipt."""

    try:
        sums_sha, _listed = admission._verify_members(
            package, label="Chapter 7 v2 evidence package"
        )
    except admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidenceAdmissionV2Error(f"package member verification failed: {exc}") from exc
    package = package.resolve()
    manifest = _read_object(package / "manifest.json", "v2 package manifest")
    _validate(manifest, PACKAGE_SCHEMA, "v2 package manifest")
    if (
        manifest.get("status") != "admitted"
        or manifest.get("admission_status") != "admitted"
        or manifest.get("source_integrity_gate") != "passed"
        or manifest.get("admission", {}).get("status") != "admitted"
    ):
        raise Ch7EvidenceAdmissionV2Error("v2 package is not in an admitted state")
    receipt_payload = _read_object(receipt, "v2 admission receipt")
    _validate(receipt_payload, RECEIPT_SCHEMA, "v2 admission receipt")
    package_binding = receipt_payload["package"]
    if package_binding["sha256sums_sha256"] != sums_sha:
        raise Ch7EvidenceAdmissionV2Error("receipt does not bind package SHA256SUMS")
    manifest_sha = _sha256_file(package / "manifest.json")
    if package_binding["manifest_sha256"] != manifest_sha:
        raise Ch7EvidenceAdmissionV2Error("receipt does not bind package manifest")
    source_binding = receipt_payload["source"]
    expected_source = manifest["source"]
    for field in (
        "v1_package_sha256sums",
        "v1_manifest_sha256",
        "v1_audit_member_sha256",
        "v1_reduced_atlas_member_sha256",
    ):
        if source_binding[field] != expected_source[field]:
            raise Ch7EvidenceAdmissionV2Error(f"receipt source binding differs: {field}")
    if (
        source_binding["portfolio_config_sha256"]
        != manifest["inputs"]["portfolio_config"]["sha256"]
    ):
        raise Ch7EvidenceAdmissionV2Error("receipt portfolio binding differs from manifest")
    if receipt_payload["scope"]["claim_boundary"] != manifest["claim_boundary"]:
        raise Ch7EvidenceAdmissionV2Error("receipt claim boundary differs from manifest")
    expected_roles = {
        role: {"grain": details["grain"]} for role, details in manifest["roles"].items()
    }
    if receipt_payload["roles"]["available"] != expected_roles:
        raise Ch7EvidenceAdmissionV2Error("receipt role scope differs from manifest")
    return {
        "status": "admitted",
        "package_sha256sums_sha256": sums_sha,
        "manifest_sha256": manifest_sha,
        "receipt_sha256": _sha256_file(receipt.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify a blocked package and print diagnostics without creating a receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run v2 admission verification and return a typed CLI status code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.check_only:
        if args.receipt is not None:
            parser.error("--receipt cannot be combined with --check-only")
        try:
            result = diagnose_v2_package(args.package)
        except (Ch7EvidenceAdmissionV2Error, OSError, ValidationError) as exc:
            print(f"ch7 v2 evidence diagnostic unavailable: {exc}")
            return 2
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if args.receipt is None:
        parser.error("--receipt is required unless --check-only is used")
    try:
        result = verify_v2_admission(args.package, args.receipt)
    except (Ch7EvidenceAdmissionV2Error, OSError, ValidationError) as exc:
        print(f"ch7 v2 evidence admission unavailable: {exc}")
        return 2
    print(f"ch7 v2 evidence admission status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
