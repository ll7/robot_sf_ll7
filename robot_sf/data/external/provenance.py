"""Library-owned validation for local external-data provenance manifests."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "output" / "external_data" / "manifests"


def _pattern_is_literal(pattern: str) -> bool:
    """Return whether a required-path pattern has no glob magic characters."""

    return not any(char in pattern for char in ("*", "?", "["))


def _glob_matches(candidate: str, pattern: str) -> bool:
    """Return whether a manifest path satisfies a required glob pattern."""

    return fnmatch.fnmatch(candidate, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatch(candidate, pattern[3:])
    )


def _paths_cover_requirements(
    required_path_groups: Mapping[str, Sequence[str]],
    matched_paths: object,
) -> bool:
    """Return whether matched paths satisfy every required-path group."""

    if not isinstance(matched_paths, list):
        return False
    candidates = {path for path in matched_paths if isinstance(path, str) and path.strip()}
    if not candidates:
        return False
    for patterns in required_path_groups.values():
        literal_patterns = [pattern for pattern in patterns if _pattern_is_literal(pattern)]
        glob_patterns = [pattern for pattern in patterns if not _pattern_is_literal(pattern)]
        satisfied = any(
            candidate == pattern or candidate.startswith(f"{pattern}/")
            for candidate in candidates
            for pattern in literal_patterns
        ) or any(
            _glob_matches(candidate, pattern)
            for candidate in candidates
            for pattern in glob_patterns
        )
        if not satisfied:
            return False
    return True


def _manifest_text_field(manifest: dict[str, Any], field: str) -> str | None:
    """Return a non-empty manifest text field or ``None``."""

    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _missing_metadata(
    asset_id: str,
    manifest: dict[str, Any],
    required_path_groups: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return missing required provenance metadata fields."""

    missing: list[str] = []
    if manifest.get("asset_id") != asset_id:
        missing.append("asset_id")
    for field in ("source_url", "license_note", "tree_sha256"):
        if not _manifest_text_field(manifest, field):
            missing.append(field)

    sample_files = manifest.get("sample_files")
    has_sample_checksum = isinstance(sample_files, list) and any(
        isinstance(sample, dict) and _manifest_text_field(sample, "sha256")
        for sample in sample_files
    )
    if not has_sample_checksum:
        missing.append("sample_files[].sha256")

    if not _paths_cover_requirements(required_path_groups, manifest.get("matched_required_paths")):
        missing.append("matched_required_paths")
    return missing


def check_provenance_manifest(
    asset_id: str,
    manifest_path: Path,
    *,
    required_path_groups: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Validate bounded provenance metadata for a staged external asset manifest.

    Returns:
        A readiness report with ``ok``, status, missing metadata, and remediation fields.
    """

    path = manifest_path.expanduser().resolve()
    report: dict[str, Any] = {
        "schema": "robot_sf_external_data_provenance_readiness.v1",
        "asset_id": asset_id,
        "manifest_path": str(path),
        "ok": False,
        "status": "missing_manifest",
        "missing_metadata": [],
    }
    if not path.is_file():
        report["action"] = (
            "Manifest file missing; run stage after official local asset acquisition."
        )
        return report

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["status"] = "invalid_json"
        report["action"] = f"Manifest JSON invalid: {exc}"
        return report
    if not isinstance(manifest, dict):
        report["status"] = "invalid_json"
        report["action"] = "Manifest JSON invalid: root value must be a JSON object."
        return report

    missing = _missing_metadata(asset_id, manifest, required_path_groups)
    report["missing_metadata"] = missing
    report["source_url"] = manifest.get("source_url")
    report["license_url"] = manifest.get("license_url")
    report["license_note"] = manifest.get("license_note")
    report["expected_tree_sha256"] = manifest.get("expected_tree_sha256")
    report["expected_tree_sha256_status"] = manifest.get("expected_tree_sha256_status")
    report["tree_sha256"] = manifest.get("tree_sha256")
    report["matched_required_paths"] = manifest.get("matched_required_paths", [])
    if missing:
        report["status"] = "incomplete_metadata"
        report["action"] = (
            "Provenance manifest is not ready: fill official source URI, license boundary, "
            "aggregate checksum, sample file checksum, and matched asset paths."
        )
        return report

    report["ok"] = True
    report["status"] = "ready"
    report["action"] = (
        "Provenance manifest has required source, license, checksum, and path metadata."
    )
    return report
