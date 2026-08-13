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
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run v2 admission verification and return a typed CLI status code."""

    args = _parser().parse_args(argv)
    try:
        result = verify_v2_admission(args.package, args.receipt)
    except (Ch7EvidenceAdmissionV2Error, OSError, ValidationError) as exc:
        print(f"ch7 v2 evidence admission unavailable: {exc}")
        return 2
    print(f"ch7 v2 evidence admission status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
