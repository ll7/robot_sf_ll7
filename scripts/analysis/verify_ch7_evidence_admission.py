"""Verify the author admission boundary for the Chapter 7 evidence package.

The package builder deliberately emits an immutable, blocked machine package.  This module
verifies the separate author receipt and trusted source-registry entry that are required before
the package can be treated as admitted.  It never rewrites the package or promotes a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

SCHEMA_VERSION = "ch7-evidence-admission.v1"
PACKAGE_SCHEMA_VERSION = "ch7-evidence-package.v1"
SOURCE_REGISTRY_SCHEMA_VERSION = "case-source-integrity-registry.v1"
APPROVAL_URL_RE = re.compile(
    r"^https://github\.com/ll7/robot_sf_ll7/issues/6792#issuecomment-[0-9]+$"
)
APPROVAL_ID_RE = re.compile(r"^issue6792-comment-[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_CLAIMS = (
    "matched_comparison",
    "causal_divergence",
    "counterfactual_branching",
    "trajectory_divergence",
    "pooled_arm_replication",
)
DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "robot_sf/benchmark/schemas/ch7-evidence-admission.v1.json"
)
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2] / "configs/analysis/source_gate_registry.v1.json"
)


class Ch7EvidenceAdmissionError(ValueError):
    """Raised when an admission receipt or its bound inputs fail closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Ch7EvidenceAdmissionError(f"cannot read digest input: {path}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Ch7EvidenceAdmissionError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidenceAdmissionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidenceAdmissionError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _parse_sums(path: Path) -> list[tuple[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Ch7EvidenceAdmissionError(f"SHA256SUMS is unreadable: {path}") from exc
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        if "  " not in raw:
            raise Ch7EvidenceAdmissionError(f"malformed SHA256SUMS line {line_number}: {path}")
        digest, relative = raw.split("  ", 1)
        digest = _require_sha256(digest, f"SHA256SUMS line {line_number}")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise Ch7EvidenceAdmissionError(f"unsafe SHA256SUMS path: {relative}")
        entries.append((digest, relative_path.as_posix()))
    if not entries:
        raise Ch7EvidenceAdmissionError(f"empty SHA256SUMS: {path}")
    paths = [relative for _digest, relative in entries]
    if len(paths) != len(set(paths)):
        raise Ch7EvidenceAdmissionError(f"duplicate SHA256SUMS paths: {path}")
    return entries


def _verify_review_sidecars(
    root: Path,
    sidecars: set[str],
    listed: set[str],
    sums_path: Path,
    *,
    label: str,
) -> None:
    expected_sidecars = {f"{relative}.review.json" for relative in listed}
    expected_sidecars.add("SHA256SUMS.review.json")
    missing_sidecars = sorted(expected_sidecars - sidecars)
    if missing_sidecars:
        raise Ch7EvidenceAdmissionError(
            f"{label} is missing required review sidecars: {missing_sidecars}"
        )
    unexpected_sidecars = sorted(sidecars - expected_sidecars)
    if unexpected_sidecars:
        raise Ch7EvidenceAdmissionError(
            f"{label} contains unbound review sidecars: {unexpected_sidecars}"
        )
    for sidecar in sorted(sidecars):
        base_relative = sidecar.removesuffix(".review.json")
        artifact = sums_path if base_relative == "SHA256SUMS" else root / base_relative
        marker = _read_object(root / sidecar, f"{label} review sidecar")
        if marker.get("schema_version") != "evidence-review-marker.v1":
            raise Ch7EvidenceAdmissionError(
                f"{label} review sidecar has an unsupported schema: {sidecar}"
            )
        artifact_path = marker.get("artifact_path")
        expected_artifact_path = f"docs/context/evidence/{root.name}/{base_relative}"
        if artifact_path != expected_artifact_path:
            raise Ch7EvidenceAdmissionError(
                f"{label} review sidecar points at the wrong artifact: {sidecar}"
            )
        if marker.get("artifact_sha256") != _sha256_file(artifact):
            raise Ch7EvidenceAdmissionError(
                f"{label} review sidecar artifact hash mismatch: {sidecar}"
            )
        if marker.get("review_marker") != "AI-GENERATED NEEDS-REVIEW":
            raise Ch7EvidenceAdmissionError(
                f"{label} review sidecar marker is not review-only: {sidecar}"
            )
        if marker.get("preserved_exact_bytes") is not True:
            raise Ch7EvidenceAdmissionError(
                f"{label} review sidecar does not preserve exact bytes: {sidecar}"
            )


def _reject_symlinks(root: Path, *, label: str) -> None:
    links = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()
    )
    if links:
        raise Ch7EvidenceAdmissionError(f"{label} contains symlinks: {links}")


def _reject_special_entries(root: Path, *, label: str) -> None:
    special = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not path.is_dir() and not path.is_file()
    )
    if special:
        raise Ch7EvidenceAdmissionError(f"{label} contains special filesystem entries: {special}")


def _verify_members(
    root: Path, *, label: str, require_review_sidecars: bool = True
) -> tuple[str, list[str]]:
    root = root.resolve()
    sums_path = root / "SHA256SUMS"
    if not root.is_dir() or not sums_path.is_file():
        raise Ch7EvidenceAdmissionError(f"{label} must be a directory with SHA256SUMS")
    _reject_symlinks(root, label=label)
    _reject_special_entries(root, label=label)
    entries = _parse_sums(sums_path)
    listed = {relative for _digest, relative in entries}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums_path
    }
    sidecars = {relative for relative in actual if relative.endswith(".review.json")}
    payload_actual = actual - sidecars
    missing = sorted(listed - payload_actual)
    extra = sorted(payload_actual - listed)
    if missing:
        raise Ch7EvidenceAdmissionError(f"{label} SHA256SUMS lists missing files: {missing}")
    if extra:
        raise Ch7EvidenceAdmissionError(f"{label} contains unlisted files: {extra}")
    if require_review_sidecars:
        _verify_review_sidecars(root, sidecars, listed, sums_path, label=label)
    elif sidecars:
        raise Ch7EvidenceAdmissionError(
            f"{label} contains unbound review sidecars: {sorted(sidecars)}"
        )
    for expected, relative in entries:
        member = root / relative
        if _sha256_file(member) != expected:
            raise Ch7EvidenceAdmissionError(f"{label} member hash mismatch: {relative}")
    return _sha256_file(sums_path), sorted(listed)


def _verify_source_package(source_package: Path, expected_complete_sha256: str) -> str:
    """Verify the approved source tree, including its intentionally unlisted completion receipt."""

    root = source_package.resolve()
    sums_path = root / "SHA256SUMS"
    if not root.is_dir() or not sums_path.is_file():
        raise Ch7EvidenceAdmissionError("source package must be a directory with SHA256SUMS")
    _reject_symlinks(root, label="source package")
    _reject_special_entries(root, label="source package")
    entries = _parse_sums(sums_path)
    listed = {relative for _digest, relative in entries}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums_path
    }
    allowed_extra = {"package_complete.json"}
    missing = sorted(listed - actual)
    extra = sorted(actual - listed - allowed_extra)
    if missing:
        raise Ch7EvidenceAdmissionError(f"source package SHA256SUMS lists missing files: {missing}")
    if extra:
        raise Ch7EvidenceAdmissionError(f"source package contains unlisted files: {extra}")
    for expected, relative in entries:
        if _sha256_file(root / relative) != expected:
            raise Ch7EvidenceAdmissionError(f"source package member hash mismatch: {relative}")
    complete = root / "package_complete.json"
    if not complete.is_file():
        raise Ch7EvidenceAdmissionError("source package package_complete.json is missing")
    complete_sha = _sha256_file(complete)
    if complete_sha != expected_complete_sha256:
        raise Ch7EvidenceAdmissionError(
            "source package package_complete.json digest does not match receipt"
        )
    complete_payload = _read_object(complete, "source package completion receipt")
    if complete_payload.get("sha256sums_sha256") != _sha256_file(sums_path):
        raise Ch7EvidenceAdmissionError(
            "source package completion receipt is not bound to SHA256SUMS"
        )
    return complete_sha


def _verify_compact_directory(compact_dir: Path) -> tuple[str, str]:
    sums_sha, listed = _verify_members(
        compact_dir, label="compact packet", require_review_sidecars=False
    )
    if listed != ["compact_packet.json"]:
        raise Ch7EvidenceAdmissionError(
            "compact packet must contain exactly compact_packet.json in SHA256SUMS"
        )
    packet_sha = _sha256_file(compact_dir / "compact_packet.json")
    return packet_sha, sums_sha


def _validate_schema(payload: Mapping[str, Any], schema_path: Path, *, label: str) -> None:
    if label == "admission schema" and schema_path.resolve() != DEFAULT_SCHEMA.resolve():
        raise Ch7EvidenceAdmissionError(
            "custom admission schemas are not permitted at the promotion boundary"
        )
    schema = _read_object(schema_path, label)
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=str)
    except (ValidationError, TypeError) as exc:
        raise Ch7EvidenceAdmissionError(f"{label} is invalid") from exc
    if errors:
        details = "; ".join(error.message for error in errors[:3])
        raise Ch7EvidenceAdmissionError(f"{label} validation failed: {details}")


def _verify_package(package_dir: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    sums_sha, _listed = _verify_members(package_dir, label="evidence package")
    manifest = _read_object(manifest_path, "package manifest")
    _validate_schema(
        manifest,
        DEFAULT_SCHEMA.parent / "ch7-evidence-package.v1.json",
        label="package manifest schema",
    )
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION or manifest.get("issue") != 6792:
        raise Ch7EvidenceAdmissionError("package manifest is not the Chapter 7 package")
    if (
        manifest.get("status") != "blocked_pending_domain_approval"
        or manifest.get("admission_status") != "not_admitted"
        or manifest.get("source_integrity_gate") != "blocked_pending_domain_approval"
    ):
        raise Ch7EvidenceAdmissionError(
            "package manifest must remain the immutable blocked machine package"
        )
    package_receipt = receipt["package"]
    if sums_sha != package_receipt["sha256sums_sha256"]:
        raise Ch7EvidenceAdmissionError("package SHA256SUMS digest does not match receipt")
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != package_receipt["manifest_sha256"]:
        raise Ch7EvidenceAdmissionError("package manifest digest does not match receipt")
    source_receipt = receipt["source"]
    source_manifest = manifest.get("source")
    if not isinstance(source_manifest, Mapping):
        raise Ch7EvidenceAdmissionError("package manifest source block is missing")
    expected_pairs = {
        "source_package_sha256sums": "approved_package_sha256sums",
        "release_archive_sha256": "release_archive_sha256",
        "compact_packet_sha256": "issue6814_compact_packet_sha256",
    }
    for receipt_key, manifest_key in expected_pairs.items():
        if source_receipt[receipt_key] != source_manifest.get(manifest_key):
            raise Ch7EvidenceAdmissionError(
                f"package manifest source digest mismatch: {receipt_key}"
            )
    manifest_roles = manifest.get("roles")
    if not isinstance(manifest_roles, Mapping):
        raise Ch7EvidenceAdmissionError("package manifest roles block is missing")
    _verify_roles(manifest_roles, receipt["roles"])
    if receipt["scope"]["claim_boundary"] != manifest.get("claim_boundary"):
        raise Ch7EvidenceAdmissionError("claim boundary does not match package manifest")
    return {"manifest": manifest, "manifest_sha256": manifest_sha, "sha256sums_sha256": sums_sha}


def _verify_roles(manifest_roles: Mapping[str, Any], receipt_roles: Mapping[str, Any]) -> None:
    available = receipt_roles["available"]
    unavailable = receipt_roles["unavailable"]
    expected_available = {
        name: value
        for name, value in manifest_roles.items()
        if isinstance(value, Mapping) and value.get("status") == "available"
    }
    expected_unavailable = {
        name: value
        for name, value in manifest_roles.items()
        if isinstance(value, Mapping) and value.get("status") == "unavailable"
    }
    if set(available) != set(expected_available) or set(unavailable) != set(expected_unavailable):
        raise Ch7EvidenceAdmissionError("receipt role set does not match package manifest")
    for name, value in expected_available.items():
        if not isinstance(available[name], Mapping) or available[name].get("grain") != value.get(
            "grain"
        ):
            raise Ch7EvidenceAdmissionError(f"available role grain mismatch: {name}")
    for name, value in expected_unavailable.items():
        reasons = (
            unavailable[name].get("reasons") if isinstance(unavailable[name], Mapping) else None
        )
        if (
            not isinstance(reasons, list)
            or not isinstance(value.get("reason"), str)
            or value["reason"] not in reasons
        ):
            raise Ch7EvidenceAdmissionError(f"unavailable role reason mismatch: {name}")


def _verify_registry(registry_path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = _read_object(registry_path, "source registry")
    if registry.get("schema_version") != SOURCE_REGISTRY_SCHEMA_VERSION:
        raise Ch7EvidenceAdmissionError("source registry schema mismatch")
    approved_sources = registry.get("approved_sources")
    if not isinstance(approved_sources, list):
        raise Ch7EvidenceAdmissionError("source registry approved_sources must be a list")
    registry_sha = _sha256_file(registry_path)
    if registry_sha != receipt["source"]["source_registry_sha256"]:
        raise Ch7EvidenceAdmissionError("source registry digest does not match receipt")
    approval = receipt["approval"]
    if APPROVAL_ID_RE.fullmatch(str(approval.get("approval_id"))) is None:
        raise Ch7EvidenceAdmissionError("approval ID is not a canonical #6792 comment ID")
    if APPROVAL_URL_RE.fullmatch(str(approval.get("approval_url"))) is None:
        raise Ch7EvidenceAdmissionError("approval URL is not the canonical #6792 issue comment")
    source = receipt["source"]
    matches = [
        entry
        for entry in approved_sources
        if isinstance(entry, Mapping)
        and entry.get("approval_id") == approval["approval_id"]
        and entry.get("approval_url") == approval["approval_url"]
        and entry.get("status") == "approved"
        and entry.get("source_package_sha256sums") == source["source_package_sha256sums"]
        and entry.get("source_package_complete_sha256") == source["source_package_complete_sha256"]
        and entry.get("release_archive_sha256") == source["release_archive_sha256"]
        and entry.get("compact_packet_sha256") == source["compact_packet_sha256"]
        and entry.get("compact_sha256sums_sha256") == source["compact_sha256sums_sha256"]
    ]
    if len(matches) != 1:
        raise Ch7EvidenceAdmissionError("source registry has no unique matching approval entry")
    entry = dict(matches[0])
    retrieval = receipt["retrieval"]
    if any(
        entry.get(key) != retrieval[f"{key.removesuffix('_key')}_key"]
        for key in ("source_package_key", "release_archive_key", "compact_packet_key")
    ):
        raise Ch7EvidenceAdmissionError("durable retrieval key does not match source registry")
    return {"registry_sha256": registry_sha, "approval_entry": entry}


def _require_canonical_source_registry(registry_path: Path) -> Path:
    resolved_registry = registry_path.resolve()
    if resolved_registry != DEFAULT_REGISTRY.resolve():
        raise Ch7EvidenceAdmissionError(
            "source registry must be the repository-controlled canonical registry"
        )
    return resolved_registry


def verify_admission(
    *,
    package_dir: Path,
    source_registry: Path,
    receipt: Path,
    source_package: Path,
    release_archive: Path,
    compact_dir: Path,
    schema: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Verify the complete author-admission boundary and return a readback receipt."""

    admission = _read_object(receipt.resolve(), "admission receipt")
    _validate_schema(admission, schema.resolve(), label="admission schema")
    package = _verify_package(package_dir, admission)
    registry = _verify_registry(_require_canonical_source_registry(source_registry), admission)
    expected_source = admission["source"]
    source_sha = _sha256_file(source_package.resolve() / "SHA256SUMS")
    source_complete_sha = _verify_source_package(
        source_package, expected_source["source_package_complete_sha256"]
    )
    release_sha = _sha256_file(release_archive.resolve())
    compact_packet_sha, compact_sums_sha = _verify_compact_directory(compact_dir.resolve())
    if source_sha != expected_source["source_package_sha256sums"]:
        raise Ch7EvidenceAdmissionError("restored source package digest does not match receipt")
    if source_complete_sha != expected_source["source_package_complete_sha256"]:
        raise Ch7EvidenceAdmissionError("source package completion digest does not match receipt")
    if release_sha != expected_source["release_archive_sha256"]:
        raise Ch7EvidenceAdmissionError("release archive digest does not match receipt")
    if compact_packet_sha != expected_source["compact_packet_sha256"]:
        raise Ch7EvidenceAdmissionError("compact packet digest does not match receipt")
    if compact_sums_sha != expected_source["compact_sha256sums_sha256"]:
        raise Ch7EvidenceAdmissionError("compact SHA256SUMS digest does not match receipt")
    if admission["approval"]["decision"] != "approve":
        raise Ch7EvidenceAdmissionError("admission decision is not approve")
    if tuple(admission["scope"]["forbidden_claims"]) != FORBIDDEN_CLAIMS:
        raise Ch7EvidenceAdmissionError("forbidden claim boundary is incomplete or reordered")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted",
        "package": package,
        "source": {
            **expected_source,
            "registry_sha256": registry["registry_sha256"],
        },
        "approval_id": admission["approval"]["approval_id"],
        "approval_url": admission["approval"]["approval_url"],
        "scope": admission["scope"],
        "roles": admission["roles"],
        "retrieval": admission["retrieval"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--release-archive", type=Path, required=True)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help=argparse.SUPPRESS)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the fail-closed admission verifier."""

    args = _build_parser().parse_args(argv)
    try:
        result = verify_admission(
            package_dir=args.package_dir,
            source_registry=args.source_registry,
            receipt=args.receipt,
            source_package=args.source_package,
            release_archive=args.release_archive,
            compact_dir=args.compact_dir,
            schema=args.schema,
        )
    except (Ch7EvidenceAdmissionError, OSError, TypeError, ValueError, ValidationError) as exc:
        print(f"ch7 evidence admission unavailable: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
