#!/usr/bin/env python3
"""Validate the immutable artifact-only Zenodo surface for release 0.0.5.

This gate is deliberately narrower than the release-preflight and distribution-license checks:
it binds one disposition to the exact checksum-manifest entries and requires mechanically
reviewable rights/source evidence for every published byte.  Repository presence, public
availability, a citation, or a software license is not treated as evidence of authorship or
redistribution permission.  The checker never publishes, changes, or regenerates an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("configs/releases/release_0_0_5_checksum_manifest.yaml")
DEFAULT_DISPOSITION = Path("configs/releases/release_0_0_5_surface_disposition.yaml")
DEFAULT_CHECKLIST = Path(
    "configs/benchmarks/releases/release_0_0_5_surface_preflight_checklist.yaml"
)

REPORT_SCHEMA = "release-surface-gate.v1"
DISPOSITION_SCHEMA = "release_surface_disposition.v1"
CHECKLIST_SCHEMA = "release_surface_preflight_checklist.v1"
CHECKSUM_MANIFEST_SCHEMA = "release-checksum-manifest.v1"
RELEASE_ID = "benchmark_release_0_0_5"
SURFACE = "artifact_only_zenodo"
EXPECTED_MANIFEST_SHA256 = "13df47b3b7092a6efb47942c97fe362c38428e9604d21740fbcb5d7526ade741"
EXPECTED_MANIFEST_ENTRY_COUNT = 25
EXPECTED_APPROVED_SOURCE_SHA = "dc78f373a28fd9bbb6b2444cfd5a74e698dfe48a"
EXPECTED_VALIDATION_BASE_SHA = "5a42d9ca580b134320a501d44307a7fd9560f6c9"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_STATUSES = frozenset({"blocked", "project_authored", "cleared_for"})
CLEAR_STATUSES = frozenset({"project_authored", "cleared_for"})
BLOCKER_REASONS = frozenset(
    {
        "unknown",
        "conflicting",
        "permission_required",
        "missing_rights_evidence",
        "rights_not_reviewed",
    }
)
EVIDENCE_TYPES = frozenset({"project_authorship", "redistribution_clearance"})
REQUIRED_EVIDENCE_METADATA = ("rights_holder", "basis", "review_reference")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_relative_path(value: Any) -> str | None:
    """Return a safe repository-relative POSIX path, or ``None``."""

    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    normalised = candidate.as_posix().lstrip("./")
    if not normalised or normalised == ".":
        return None
    return normalised


def _safe_repo_file(repo_root: Path, value: Any) -> tuple[Path | None, str | None]:
    """Resolve a payload path without following symlinks or escaping the repository."""

    relative = _normalise_relative_path(value)
    if relative is None:
        return None, "path must be a non-empty repository-relative file"
    lexical = repo_root / relative
    if lexical.is_symlink():
        return None, f"path is a symlink: {relative}"
    resolved = lexical.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None, f"path escapes repository root: {relative}"
    if not resolved.is_file():
        return None, f"path is not a regular file: {relative}"
    return resolved, None


def _load_yaml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load one YAML mapping and convert parser/I/O failures into gate errors."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"cannot load {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path} must contain a mapping"
    return payload, None


def _checkout_sha(repo_root: Path) -> tuple[str | None, str | None]:
    """Resolve the exact commit for a repository-root checkout without parent fallback."""

    try:
        top_level = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"cannot inspect approved-source checkout: {exc}"
    if top_level.returncode != 0:
        return None, "approved-source root is not a Git checkout"
    try:
        resolved_top = Path(top_level.stdout.strip()).resolve()
    except OSError as exc:
        return None, f"cannot resolve approved-source checkout root: {exc}"
    if resolved_top != repo_root.resolve():
        return None, "approved-source root is nested inside a different Git checkout"

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"cannot resolve approved-source commit: {exc}"
    sha = result.stdout.strip().lower()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return None, "approved-source checkout has no resolvable HEAD commit"
    return sha, None


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    row_id: str | None = None,
    path: str | None = None,
) -> None:
    """Append one deterministic issue to a gate report."""

    item: dict[str, Any] = {"code": code, "message": message}
    if row_id is not None:
        item["row_id"] = row_id
    if path is not None:
        item["path"] = path
    issues.append(item)


def _manifest_entries(  # noqa: C901
    payload: dict[str, Any],
    *,
    repo_root: Path,
    issues: list[dict[str, Any]],
    enforce_frozen_shape: bool = True,
) -> list[dict[str, str]]:
    """Validate and return the authoritative checksum-manifest entries."""

    if payload.get("schema_version") != CHECKSUM_MANIFEST_SCHEMA:
        _issue(
            issues,
            "manifest_schema",
            f"manifest schema_version must be {CHECKSUM_MANIFEST_SCHEMA!r}",
        )
    if payload.get("release_tag") != "0.0.5":
        _issue(issues, "manifest_release_tag", "manifest release_tag must be '0.0.5'")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        _issue(issues, "manifest_entries", "manifest entries must be a list")
        return []
    if enforce_frozen_shape and len(raw_entries) != EXPECTED_MANIFEST_ENTRY_COUNT:
        _issue(
            issues,
            "manifest_entry_count",
            f"manifest must contain exactly {EXPECTED_MANIFEST_ENTRY_COUNT} entries; "
            f"found {len(raw_entries)}",
        )

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            _issue(issues, "manifest_entry_shape", f"manifest entries[{index}] must be a mapping")
            continue
        path = _normalise_relative_path(raw.get("path"))
        expected = raw.get("sha256")
        if path is None:
            _issue(issues, "manifest_entry_path", f"manifest entries[{index}] has an unsafe path")
            continue
        if path in seen:
            _issue(
                issues, "manifest_duplicate_path", f"manifest path is duplicated: {path}", path=path
            )
            continue
        seen.add(path)
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected.lower()):
            _issue(
                issues,
                "manifest_entry_digest",
                f"manifest entry has an invalid SHA-256 digest: {path}",
                path=path,
            )
            continue
        expected = expected.lower()
        resolved, error = _safe_repo_file(repo_root, path)
        if error:
            _issue(issues, "manifest_file", error, path=path)
        elif resolved is not None:
            actual = _sha256(resolved)
            if actual != expected:
                _issue(
                    issues,
                    "manifest_checksum_mismatch",
                    f"checksum mismatch for {path}: expected {expected}, got {actual}",
                    path=path,
                )
        entries.append({"path": path, "sha256": expected})
    return entries


def _binding(
    payload: dict[str, Any],
    *,
    label: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate a disposition/checklist manifest binding shape."""

    raw = payload.get("manifest")
    if not isinstance(raw, dict):
        _issue(issues, f"{label}_manifest_binding", f"{label} manifest binding must be a mapping")
        return None
    path = _normalise_relative_path(raw.get("path"))
    digest = raw.get("sha256")
    count = raw.get("entry_count")
    if path != DEFAULT_MANIFEST.as_posix():
        _issue(
            issues,
            f"{label}_manifest_path",
            f"{label} must bind {DEFAULT_MANIFEST.as_posix()}",
        )
    if not isinstance(digest, str) or digest.lower() != EXPECTED_MANIFEST_SHA256:
        _issue(
            issues,
            f"{label}_manifest_digest",
            f"{label} must bind authoritative manifest digest {EXPECTED_MANIFEST_SHA256}",
        )
    if count != EXPECTED_MANIFEST_ENTRY_COUNT:
        _issue(
            issues,
            f"{label}_manifest_count",
            f"{label} must bind exactly {EXPECTED_MANIFEST_ENTRY_COUNT} manifest paths",
        )
    return {"path": path, "sha256": str(digest).lower(), "entry_count": count}


def _validate_decision(decision: Any, *, issues: list[dict[str, Any]]) -> None:
    """Validate the exact conditional maintainer route without granting authority."""

    if not isinstance(decision, dict) or decision.get("token") != "approve-artifact-only":
        _issue(
            issues,
            "decision_token",
            "disposition must record the conditional #7320 token approve-artifact-only",
        )
    if isinstance(decision, dict):
        if decision.get("issue") != 7320:
            _issue(issues, "decision_issue", "disposition decision must remain bound to issue 7320")
        if decision.get("conditional") is not True:
            _issue(issues, "decision_conditional", "artifact-only route must remain conditional")
        if decision.get("publication_authorized") is not False:
            _issue(
                issues,
                "decision_publication_authorized",
                "the disposition ledger cannot authorize publication",
            )


def _validate_disposition_header(
    payload: dict[str, Any],
    *,
    issues: list[dict[str, Any]],
) -> None:
    """Validate immutable route and release identity fields."""

    if payload.get("schema_version") != DISPOSITION_SCHEMA:
        _issue(
            issues,
            "disposition_schema",
            f"disposition schema_version must be {DISPOSITION_SCHEMA!r}",
        )
    if payload.get("release_id") != RELEASE_ID:
        _issue(issues, "disposition_release_id", f"disposition release_id must be {RELEASE_ID!r}")
    if payload.get("surface") != SURFACE:
        _issue(issues, "disposition_surface", f"publication surface must be {SURFACE!r}")
    _validate_decision(payload.get("decision"), issues=issues)
    source_sha = payload.get("approved_source_sha")
    if source_sha != EXPECTED_APPROVED_SOURCE_SHA:
        _issue(
            issues,
            "approved_source_sha",
            f"approved_source_sha must remain {EXPECTED_APPROVED_SOURCE_SHA}",
        )
    base_sha = payload.get("validation_base_sha")
    if base_sha != EXPECTED_VALIDATION_BASE_SHA:
        _issue(
            issues,
            "validation_base_sha",
            f"validation_base_sha must remain {EXPECTED_VALIDATION_BASE_SHA}",
        )


def _validate_evidence(  # noqa: C901
    evidence: Any,
    *,
    row_id: str,
    target_path: str,
    target_sha256: str,
    expected_type: str,
    repo_root: Path,
    issues: list[dict[str, Any]],
) -> None:
    """Validate evidence that can support a clear row.

    Evidence records are intentionally redundant: the record must name the exact target path and
    digest, carry the expected type, and contain the corresponding marker in its bytes.  This
    prevents an arbitrary existing file (including a license or citation) from being treated as a
    rights grant by filename alone.
    """

    if not isinstance(evidence, list) or not evidence:
        _issue(
            issues,
            "clear_without_evidence",
            "a clear row requires at least one rights/source evidence record",
            row_id=row_id,
            path=target_path,
        )
        return
    for index, raw in enumerate(evidence):
        if not isinstance(raw, dict):
            _issue(
                issues,
                "evidence_shape",
                f"evidence[{index}] must be a mapping",
                row_id=row_id,
                path=target_path,
            )
            continue
        evidence_path = _normalise_relative_path(raw.get("path"))
        if evidence_path is None:
            _issue(
                issues,
                "evidence_path",
                f"evidence[{index}] has an unsafe path",
                row_id=row_id,
                path=target_path,
            )
            continue
        if evidence_path == target_path:
            _issue(
                issues,
                "evidence_self_reference",
                "a target artifact cannot certify its own rights",
                row_id=row_id,
                path=target_path,
            )
        resolved, error = _safe_repo_file(repo_root, evidence_path)
        if error:
            _issue(issues, "evidence_file", error, row_id=row_id, path=evidence_path)
            continue
        assert resolved is not None
        recorded_digest = raw.get("sha256")
        actual_digest = _sha256(resolved)
        if not isinstance(recorded_digest, str) or recorded_digest.lower() != actual_digest:
            _issue(
                issues,
                "evidence_checksum",
                f"evidence checksum does not match {evidence_path}",
                row_id=row_id,
                path=evidence_path,
            )
        if raw.get("target_path") != target_path or raw.get("target_sha256") != target_sha256:
            _issue(
                issues,
                "evidence_target_binding",
                "evidence must bind the exact manifest path and SHA-256",
                row_id=row_id,
                path=target_path,
            )
        if raw.get("evidence_type") != expected_type:
            _issue(
                issues,
                "evidence_type",
                f"clear status {expected_type!r} requires evidence_type {expected_type!r}",
                row_id=row_id,
                path=target_path,
            )
        for field in REQUIRED_EVIDENCE_METADATA:
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                _issue(
                    issues,
                    "evidence_metadata",
                    f"evidence must declare non-empty {field}",
                    row_id=row_id,
                    path=target_path,
                )
        try:
            content = resolved.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError) as exc:
            _issue(
                issues,
                "evidence_content",
                f"evidence record is not readable UTF-8: {exc}",
                row_id=row_id,
                path=evidence_path,
            )
            continue
        marker = "project_authored" if expected_type == "project_authorship" else "cleared_for"
        if (
            marker not in content
            or target_path.lower() not in content
            or target_sha256 not in content
        ):
            _issue(
                issues,
                "evidence_marker",
                f"evidence record must contain {marker}, exact target path, and target digest",
                row_id=row_id,
                path=evidence_path,
            )
        if expected_type == "redistribution_clearance" and SURFACE not in content:
            _issue(
                issues,
                "evidence_surface",
                "redistribution clearance must name artifact_only_zenodo",
                row_id=row_id,
                path=evidence_path,
            )


def _validate_disposition_rows(  # noqa: C901
    payload: dict[str, Any],
    *,
    entries: list[dict[str, str]],
    repo_root: Path,
    issues: list[dict[str, Any]],
) -> dict[str, int]:
    """Validate exact one-to-one row coverage and rights dispositions."""

    expected_by_path = {entry["path"]: entry["sha256"] for entry in entries}
    counts = dict.fromkeys(sorted(ALLOWED_STATUSES), 0)
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        _issue(issues, "disposition_rows", "disposition rows must be a list")
        return counts
    if len(raw_rows) != len(entries):
        _issue(
            issues,
            "disposition_row_count",
            f"disposition must contain exactly {len(entries)} rows; found {len(raw_rows)}",
        )
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        row_id = f"row:{index + 1:02d}"
        if not isinstance(raw, dict):
            _issue(issues, "disposition_row_shape", f"{row_id} must be a mapping", row_id=row_id)
            continue
        path = _normalise_relative_path(raw.get("path"))
        row_id = str(raw.get("row_id") or path or row_id)
        if path is None:
            _issue(issues, "disposition_row_path", "row has an unsafe path", row_id=row_id)
            continue
        if path in seen:
            _issue(
                issues,
                "disposition_duplicate_path",
                f"row path is duplicated: {path}",
                row_id=row_id,
            )
            continue
        seen.add(path)
        if path not in expected_by_path:
            _issue(
                issues,
                "disposition_extra_path",
                f"row is not in the authoritative manifest: {path}",
                row_id=row_id,
                path=path,
            )
            continue
        expected_sha = expected_by_path[path]
        if raw.get("sha256") != expected_sha:
            _issue(
                issues,
                "disposition_checksum",
                f"row SHA-256 does not match manifest: {path}",
                row_id=row_id,
                path=path,
            )
        status = raw.get("status")
        alias = raw.get("disposition")
        if alias is not None and alias != status:
            _issue(
                issues,
                "disposition_status_conflict",
                "status and disposition disagree",
                row_id=row_id,
                path=path,
            )
        if status not in ALLOWED_STATUSES:
            _issue(
                issues,
                "disposition_status",
                f"row status must be one of {sorted(ALLOWED_STATUSES)}; got {status!r}",
                row_id=row_id,
                path=path,
            )
            continue
        counts[status] += 1
        if status == "blocked":
            reason = raw.get("reason_code")
            if reason not in BLOCKER_REASONS:
                _issue(
                    issues,
                    "blocked_reason",
                    f"blocked row needs a known reason_code; got {reason!r}",
                    row_id=row_id,
                    path=path,
                )
            if not isinstance(raw.get("rationale"), str) or not raw["rationale"].strip():
                _issue(
                    issues,
                    "blocked_rationale",
                    "blocked row needs a rationale",
                    row_id=row_id,
                    path=path,
                )
            if raw.get("evidence") not in (None, []):
                _issue(
                    issues,
                    "blocked_evidence",
                    "blocked rows cannot carry clearance evidence",
                    row_id=row_id,
                    path=path,
                )
            continue
        expected_type = (
            "project_authorship" if status == "project_authored" else "redistribution_clearance"
        )
        _validate_evidence(
            raw.get("evidence"),
            row_id=row_id,
            target_path=path,
            target_sha256=expected_sha,
            expected_type=expected_type,
            repo_root=repo_root,
            issues=issues,
        )

    missing = sorted(set(expected_by_path) - seen)
    for path in missing:
        _issue(
            issues,
            "disposition_missing_path",
            f"manifest path has no disposition row: {path}",
            path=path,
        )
    return counts


def _validate_checklist(  # noqa: C901
    payload: dict[str, Any],
    *,
    entries: list[dict[str, str]],
    issues: list[dict[str, Any]],
) -> None:
    """Validate that the preflight checklist enumerates the exact manifest surface."""

    if payload.get("schema_version") != CHECKLIST_SCHEMA:
        _issue(issues, "checklist_schema", f"checklist schema_version must be {CHECKLIST_SCHEMA!r}")
    if payload.get("release_id") != RELEASE_ID:
        _issue(issues, "checklist_release_id", f"checklist release_id must be {RELEASE_ID!r}")
    if payload.get("surface") != SURFACE:
        _issue(issues, "checklist_surface", f"checklist surface must be {SURFACE!r}")
    _binding(payload, label="checklist", issues=issues)
    if payload.get("disposition_path") != DEFAULT_DISPOSITION.as_posix():
        _issue(
            issues,
            "checklist_disposition_path",
            "checklist must name the canonical disposition path",
        )
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        _issue(issues, "checklist_items", "checklist items must be a list")
        return
    if len(raw_items) != len(entries):
        _issue(
            issues, "checklist_item_count", f"checklist must contain exactly {len(entries)} items"
        )
    expected_by_path = {entry["path"]: entry["sha256"] for entry in entries}
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            _issue(issues, "checklist_item_shape", f"checklist items[{index}] must be a mapping")
            continue
        path = _normalise_relative_path(raw.get("path"))
        if path is None:
            _issue(issues, "checklist_item_path", f"checklist items[{index}] has an unsafe path")
            continue
        if path in seen:
            _issue(
                issues,
                "checklist_duplicate_path",
                f"checklist path is duplicated: {path}",
                path=path,
            )
        seen.add(path)
        if path not in expected_by_path:
            _issue(
                issues,
                "checklist_extra_path",
                f"checklist path is not in manifest: {path}",
                path=path,
            )
            continue
        if raw.get("sha256") != expected_by_path[path]:
            _issue(
                issues,
                "checklist_checksum",
                f"checklist SHA-256 does not match manifest: {path}",
                path=path,
            )
        if raw.get("check") != "rights_disposition":
            _issue(
                issues,
                "checklist_check",
                f"checklist item must use rights_disposition: {path}",
                path=path,
            )
    for path in sorted(set(expected_by_path) - seen):
        _issue(
            issues,
            "checklist_missing_path",
            f"manifest path is absent from checklist: {path}",
            path=path,
        )


def build_report(
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic report using canonical controls and one approved checkout."""

    root = (repo_root or REPO_ROOT).resolve()
    manifest = root / DEFAULT_MANIFEST
    disposition = REPO_ROOT / DEFAULT_DISPOSITION
    checklist = REPO_ROOT / DEFAULT_CHECKLIST
    issues: list[dict[str, Any]] = []

    source_sha, source_error = _checkout_sha(root)
    if source_error:
        _issue(issues, "approved_source_checkout", source_error)
    elif source_sha != EXPECTED_APPROVED_SOURCE_SHA:
        _issue(
            issues,
            "approved_source_sha_mismatch",
            f"approved-source checkout must be {EXPECTED_APPROVED_SOURCE_SHA}; got {source_sha}",
        )

    manifest_payload, manifest_error = _load_yaml(manifest)
    if manifest_error:
        _issue(issues, "manifest_load", manifest_error)
        manifest_payload = {}
    manifest_digest = _sha256(manifest) if manifest.is_file() else None
    if manifest_digest != EXPECTED_MANIFEST_SHA256:
        _issue(
            issues,
            "manifest_digest",
            f"authoritative manifest digest must be {EXPECTED_MANIFEST_SHA256}; got {manifest_digest}",
        )
    entries = _manifest_entries(manifest_payload, repo_root=root, issues=issues)

    disposition_payload, disposition_error = _load_yaml(disposition)
    if disposition_error:
        _issue(issues, "disposition_load", disposition_error)
        disposition_payload = {}
    disposition_digest = _sha256(disposition) if disposition.is_file() else None
    _validate_disposition_header(disposition_payload, issues=issues)
    _binding(disposition_payload, label="disposition", issues=issues)
    counts = _validate_disposition_rows(
        disposition_payload,
        entries=entries,
        repo_root=REPO_ROOT,
        issues=issues,
    )

    checklist_payload, checklist_error = _load_yaml(checklist)
    if checklist_error:
        _issue(issues, "checklist_load", checklist_error)
        checklist_payload = {}
    checklist_digest = _sha256(checklist) if checklist.is_file() else None
    _validate_checklist(checklist_payload, entries=entries, issues=issues)

    for entry in entries:
        path = entry["path"]
        if not any(issue.get("row_id") == path for issue in issues):
            # A blocked row is itself a deliberate gate blocker.  It is recorded separately from
            # structural defects so the report remains useful while all rights are unresolved.
            row = next(
                (
                    candidate
                    for candidate in disposition_payload.get("rows", [])
                    if isinstance(candidate, dict) and candidate.get("path") == path
                ),
                None,
            )
            if isinstance(row, dict) and row.get("status") == "blocked":
                _issue(
                    issues,
                    "rights_blocked",
                    f"rights/source disposition remains blocked: {row.get('reason_code', 'unknown')}",
                    row_id=path,
                    path=path,
                )

    status = (
        "passed"
        if not issues and counts.get("blocked", 0) == 0 and sum(counts.values()) == len(entries)
        else "blocked"
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "release_id": RELEASE_ID,
        "surface": SURFACE,
        "status": status,
        "gate_passed": status == "passed",
        "publication_authorized": False,
        "approved_source": {
            "path": str(root),
            "sha": source_sha,
            "expected_sha": EXPECTED_APPROVED_SOURCE_SHA,
        },
        "claim_boundary": (
            "Artifact-only Zenodo surface integrity and rights/source evidence gate. "
            "This report does not authorize publication, alter bytes, or strengthen scientific claims."
        ),
        "manifest": {
            "path": DEFAULT_MANIFEST.as_posix(),
            "sha256": manifest_digest,
            "expected_sha256": EXPECTED_MANIFEST_SHA256,
            "entry_count": len(entries),
            "expected_entry_count": EXPECTED_MANIFEST_ENTRY_COUNT,
            "paths": [entry["path"] for entry in entries],
        },
        "disposition": {
            "path": DEFAULT_DISPOSITION.as_posix(),
            "sha256": disposition_digest,
        },
        "checklist": {
            "path": DEFAULT_CHECKLIST.as_posix(),
            "sha256": checklist_digest,
        },
        "summary": {
            "manifest_paths": len(entries),
            "disposition_rows": sum(counts.values()),
            "counts_by_status": counts,
            "blocked": counts.get("blocked", 0),
            "project_authored": counts.get("project_authored", 0),
            "cleared_for": counts.get("cleared_for", 0),
            "issue_count": len(issues),
        },
        "blockers": issues,
    }


def _render_text(report: dict[str, Any]) -> str:
    """Render a compact human-readable report."""

    summary = report["summary"]
    counts = summary["counts_by_status"]
    lines = [
        f"status: {report['status']}",
        f"surface: {report['surface']}",
        f"manifest_paths: {summary['manifest_paths']}",
        f"disposition_rows: {summary['disposition_rows']}",
        f"blocked: {counts.get('blocked', 0)}",
        f"project_authored: {counts.get('project_authored', 0)}",
        f"cleared_for: {counts.get('cleared_for', 0)}",
        f"blockers: {len(report['blockers'])}",
    ]
    lines.extend(
        f"- {issue.get('row_id', issue.get('path', 'surface'))}: {issue['message']}"
        for issue in report["blockers"]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    parser.add_argument("--format", choices=("text", "json"), default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the read-only release-surface gate."""

    args = _parse_args(argv)
    root = args.repo_root.resolve()
    report = build_report(repo_root=root)
    if args.json or args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
