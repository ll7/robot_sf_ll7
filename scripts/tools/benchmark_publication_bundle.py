"""Benchmark artifact publication helper (size report + DOI-ready bundle export)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from loguru import logger

from robot_sf.benchmark.artifact_publication import (
    DissertationArtifactSpec,
    _sha256_file,
    export_dissertation_artifact_bundle,
    export_evidence_bundle,
    export_publication_bundle,
    measure_artifact_size_ranges,
)
from robot_sf.common.artifact_paths import get_artifact_category_path

if TYPE_CHECKING:
    from collections.abc import Sequence


_MANUSCRIPT_USE_LABELS = {
    "results",
    "methodology",
    "discussion",
    "outlook",
    "do-not-use",
}
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_WEAK_CLAIM_HINTS = {
    "diagnostic",
    "blocked",
    "smoke",
    "disallowed",
    "not established",
    "not_established",
    "not benchmark evidence",
    "not new benchmark evidence",
    "blocked by",
}
_WEAK_CAVEAT_HINTS = {
    "skip",
    "ignore",
    "removed",
    "unavailable",
}
_STRONG_BOUNDARY_HINTS = {
    "results",
    "benchmark",
    "established",
    "verified",
    "reproducible",
}
_DIAGNOSTIC_CHAPTER_TARGET_STYLES = {
    "limitations",
    "limitation",
    "methodology",
    "methods",
    "future work",
    "future-work",
    "future_work",
    "discussion",
    "outlook",
    "n/a",
}


class ClaimMatrixRow(TypedDict):
    """Normalized claim matrix row for a single dissertation artifact."""

    artifact_id: str
    source_artifact_path: str
    checksum: str
    evidence_tier: str
    allowed_wording: str
    not_claimed_boundary: str
    figure_table_candidate: str
    caveat: str
    validation_status: str
    chapter_target: str | None
    chapter_target_status: str | None


class ClaimMatrix(TypedDict):
    """Normalized claim matrix, with a schema version and a list of claim rows."""

    schema_version: str
    claims: list[ClaimMatrixRow]


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for benchmark publication tooling."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    size_report = subparsers.add_parser(
        "size-report",
        help="Measure artifact size ranges across benchmark run directories.",
    )
    size_report.add_argument(
        "--benchmarks-root",
        type=Path,
        default=get_artifact_category_path("benchmarks"),
        help="Benchmark output root to scan (default: output/benchmarks).",
    )
    size_report.add_argument(
        "--include-videos",
        action="store_true",
        default=False,
        help="Include video files in size distribution calculations.",
    )
    size_report.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the size report JSON payload.",
    )

    export = subparsers.add_parser(
        "export",
        help="Export one benchmark run as a publication bundle with checksums.",
    )
    export.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Benchmark run directory to export.",
    )
    export.add_argument(
        "--out-dir",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "publication",
        help="Destination directory for bundle folder + tar.gz archive.",
    )
    export.add_argument(
        "--bundle-name",
        type=str,
        default=None,
        help="Optional output bundle name (default: <run_dir_name>_publication_bundle).",
    )
    export.add_argument(
        "--include-videos",
        action="store_true",
        default=False,
        help="Include videos in the exported payload.",
    )
    export.add_argument(
        "--repository-url",
        type=str,
        default="https://github.com/ll7/robot_sf_ll7",
        help="Public repository URL used in publication metadata.",
    )
    export.add_argument(
        "--release-tag",
        type=str,
        default="{release_tag}",
        help="Release tag placeholder or concrete tag used in citation metadata.",
    )
    export.add_argument(
        "--doi",
        type=str,
        default="10.5281/zenodo.<record-id>",
        help="DOI placeholder or concrete DOI stored in publication metadata.",
    )
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing bundle directory/archive with the same name.",
    )

    evidence = subparsers.add_parser(
        "evidence-bundle",
        help="Package selected compact evidence files with checksums and claim-boundary metadata.",
    )
    evidence.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the compact evidence files to include.",
    )
    evidence.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for the evidence bundle folder.",
    )
    evidence.add_argument(
        "--bundle-name",
        type=str,
        required=True,
        help="Output bundle directory name.",
    )
    evidence.add_argument(
        "--file",
        type=Path,
        action="append",
        default=[],
        help="Evidence file to include, relative to --source-root. Repeat for multiple files.",
    )
    evidence.add_argument(
        "--command",
        type=str,
        dest="evidence_command",
        required=True,
        help="Canonical command that produced or validates the evidence.",
    )
    evidence.add_argument(
        "--commit",
        type=str,
        required=True,
        help="Repository commit associated with the evidence.",
    )
    evidence.add_argument(
        "--claim-boundary",
        type=str,
        required=True,
        help="Conservative claim boundary for the evidence bundle.",
    )
    evidence.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing evidence bundle directory with the same name.",
    )
    evidence.add_argument(
        "--mirror-dry-run-base-uri",
        type=str,
        default=None,
        help=(
            "Optional URI prefix for a dry-run remote asset mirror manifest. "
            "No upload is attempted and credentials must not be embedded in the URI."
        ),
    )
    evidence.add_argument(
        "--mirror-local-dir",
        type=Path,
        default=None,
        help=(
            "Optional credential-free local filesystem mirror backend. Payload files are copied "
            "under this directory and recorded as file:// URIs."
        ),
    )

    dissertation = subparsers.add_parser(
        "dissertation-bundle",
        help="Export selected figure/table artifacts with dissertation-facing provenance.",
    )
    dissertation.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the selected figure/table source artifacts.",
    )
    dissertation.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for the dissertation artifact bundle.",
    )
    dissertation.add_argument(
        "--bundle-name",
        type=str,
        required=True,
        help="Output bundle directory name.",
    )
    dissertation.add_argument(
        "--artifact-spec",
        type=Path,
        required=True,
        help=(
            "JSON file containing an artifacts array with artifact_id, source_path, "
            "source_artifact, caption_draft, claim_boundary, recommended_manuscript_use, "
            "and fallback_degraded_summary fields."
        ),
    )
    dissertation.add_argument(
        "--command",
        type=str,
        dest="generation_command",
        required=True,
        help="Canonical command that generated or validates the selected artifacts.",
    )
    dissertation.add_argument(
        "--commit",
        type=str,
        required=True,
        help="Repository commit associated with the selected artifacts.",
    )
    dissertation.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing dissertation artifact bundle directory.",
    )

    validate_dissertation = subparsers.add_parser(
        "validate-dissertation-bundle",
        help="Validate a dissertation artifact bundle against checksums and contracts.",
    )
    validate_dissertation.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Dissertation bundle directory containing artifact_manifest.json.",
    )
    validate_dissertation.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional source root used to verify source_path existence and reject symlinked sources."
        ),
    )
    validate_dissertation.add_argument(
        "--expected-source-command",
        type=str,
        default=None,
        help="Expected generation command to compare against artifact_manifest['generation_command'].",
    )
    validate_dissertation.add_argument(
        "--expected-source-commit",
        type=str,
        default=None,
        help="Expected source commit to compare against artifact_manifest['source_commit'].",
    )
    validate_dissertation.add_argument(
        "--reference-manifest",
        type=Path,
        default=None,
        help="Reference artifact manifest for claim-boundary/caveat drift checks.",
    )

    diff_dissertation = subparsers.add_parser(
        "diff-dissertation-bundle",
        help="Render dissertation bundle review diff between current and baseline.",
    )
    diff_dissertation.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="New bundle directory to compare.",
    )
    diff_dissertation.add_argument(
        "--reference-bundle-dir",
        type=Path,
        required=True,
        help="Baseline bundle directory for comparison.",
    )

    claim_matrix = subparsers.add_parser(
        "claim-matrix",
        help="Generate a dissertation claim matrix from bundle metadata.",
    )
    claim_matrix.add_argument(
        "--bundle-dir",
        type=Path,
        action="append",
        default=[],
        help="Path to a dissertation artifact bundle directory. Repeat for multiple bundles.",
    )
    claim_matrix.add_argument(
        "--evidence-bundle-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional path to an evidence bundle directory. Repeat for multiple bundles.",
    )
    claim_matrix.add_argument(
        "--json-output",
        type=Path,
        required=True,
        help="Path to write the claims matrix JSON output.",
    )
    claim_matrix.add_argument(
        "--markdown-output",
        type=Path,
        required=True,
        help="Path to write the claims matrix Markdown output.",
    )
    return parser


def _run_size_report(args: argparse.Namespace) -> int:
    """Execute the ``size-report`` subcommand."""
    report = measure_artifact_size_ranges(
        args.benchmarks_root,
        include_videos=bool(args.include_videos),
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Size report written to {}", args.output_json)

    print(json.dumps(report, indent=2))
    return 0


def _run_export(args: argparse.Namespace) -> int:
    """Execute the ``export`` subcommand."""
    result = export_publication_bundle(
        args.run_dir,
        args.out_dir,
        bundle_name=args.bundle_name,
        include_videos=bool(args.include_videos),
        repository_url=args.repository_url,
        release_tag=args.release_tag,
        doi=args.doi,
        overwrite=bool(args.overwrite),
    )
    payload = {
        "bundle_dir": str(result.bundle_dir),
        "archive_path": str(result.archive_path),
        "manifest_path": str(result.manifest_path),
        "checksums_path": str(result.checksums_path),
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_evidence_bundle(args: argparse.Namespace) -> int:
    """Execute the ``evidence-bundle`` subcommand."""
    result = export_evidence_bundle(
        args.source_root,
        args.out_dir,
        bundle_name=args.bundle_name,
        files=list(args.file),
        command=str(args.evidence_command),
        commit=str(args.commit),
        claim_boundary=str(args.claim_boundary),
        overwrite=bool(args.overwrite),
        mirror_dry_run_base_uri=args.mirror_dry_run_base_uri,
        mirror_local_dir=args.mirror_local_dir,
    )
    payload = {
        "bundle_dir": str(result.bundle_dir),
        "manifest_path": str(result.manifest_path),
        "checksums_path": str(result.checksums_path),
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
    }
    if result.mirror_manifest_path is not None:
        payload["mirror_manifest_path"] = str(result.mirror_manifest_path)
    print(json.dumps(payload, indent=2))
    return 0


def _load_dissertation_artifacts(spec_path: Path) -> list[DissertationArtifactSpec]:
    """Load dissertation artifact rows from a JSON spec file."""
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = payload.get("artifacts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Dissertation artifact spec must be a list or contain an artifacts list")

    artifacts: list[DissertationArtifactSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Artifact spec row {index} must be an object")
        try:

            def get_str(
                key: str,
                *,
                _row: dict[str, Any] = row,
                _index: int = index,
            ) -> str:
                """Return a required artifact spec value without accepting JSON null."""
                value = _row[key]
                if value is None:
                    raise ValueError(f"Artifact spec row {_index} field {key!r} cannot be null")
                return str(value)

            def get_optional_str(key: str, *, _row: dict[str, Any] = row) -> str | None:
                """Return an optional artifact spec value, treating JSON null as None."""
                value = _row.get(key)
                if value is None:
                    return None
                stripped = str(value).strip()
                return stripped if stripped else None

            artifacts.append(
                DissertationArtifactSpec(
                    artifact_id=get_str("artifact_id"),
                    source_path=Path(get_str("source_path")),
                    source_artifact=get_str("source_artifact"),
                    caption_draft=get_str("caption_draft"),
                    claim_boundary=get_str("claim_boundary"),
                    recommended_manuscript_use=get_str("recommended_manuscript_use"),
                    fallback_degraded_summary=get_str("fallback_degraded_summary"),
                    metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else None,
                    chapter_target=get_optional_str("chapter_target"),
                    chapter_target_justification=get_optional_str("chapter_target_justification"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Artifact spec row {index} missing required field: {exc}") from exc
    return artifacts


def _read_json(path: Path) -> dict[str, object]:
    """Load a JSON file and return an object, with explicit error on decode failure."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Unable to read JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _parse_checksums_file(checksums_path: Path) -> dict[str, str]:
    """Read checksum mappings from ``<sha256>  <path>`` lines."""
    if not checksums_path.exists():
        raise FileNotFoundError(f"Missing checksums file: {checksums_path}")
    checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if "  " not in line:
            raise ValueError(f"Malformed checksum line: {line!r}")
        digest, path = line.split("  ", 1)
        checksums[path] = digest
    return checksums


def _load_dissertation_bundle_data(
    bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load dissertation bundle manifest and checksums."""
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    checksums = _parse_checksums_file(bundle_dir / "checksums.sha256")
    return manifest, checksums


def _load_evidence_bundle_data(bundle_dir: Path) -> dict[str, Any]:
    """Load evidence bundle manifest."""
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "evidence_bundle_manifest.json"
    if not manifest_path.exists():
        manifest_path = bundle_dir / "evidence_manifest.json"
    manifest = _read_json(manifest_path)
    return manifest


def _is_allowed_manuscript_use(label: str, artifact_id: str) -> None:
    """Validate dissertation manuscript-use label contract."""
    if label not in _MANUSCRIPT_USE_LABELS:
        allowed = ", ".join(sorted(_MANUSCRIPT_USE_LABELS))
        raise ValueError(
            f"Artifact {artifact_id!r} has invalid manuscript-use label {label!r}; "
            f"allowed: {allowed}"
        )


def _is_weak_claim_boundary(claim_boundary: str) -> bool:
    """Return whether a claim boundary contains weak or diagnostic wording."""
    normalized = claim_boundary.lower()
    return any(hint in normalized for hint in _WEAK_CLAIM_HINTS)


def _is_weak_caveat(caveat: str) -> bool:
    """Return whether caveat text appears to degrade evidence confidence."""
    normalized = caveat.lower()
    return any(hint in normalized for hint in _WEAK_CAVEAT_HINTS)


def _claim_strength(claim_boundary: str) -> int:
    """Assign a low/medium/high strength score to a claim boundary."""
    normalized = claim_boundary.lower()
    if any(hint in normalized for hint in _WEAK_CLAIM_HINTS):
        return 0
    if any(hint in normalized for hint in _STRONG_BOUNDARY_HINTS):
        return 2
    return 1


def _claim_is_weakened(previous: str, current: str) -> bool:
    """Detect a claim-boundary weakening in the text-level contract."""
    return _claim_strength(current) < _claim_strength(previous)


def _caveat_is_weakened(previous: str, current: str) -> bool:
    """Detect caveat weakening based on explicit weak-caveat text."""
    if _is_weak_caveat(current) and not _is_weak_caveat(previous):
        return True
    return _claim_is_weakened(previous, current)


def _validate_dissertation_manifest_row(  # noqa: C901
    artifact: dict[str, object],
    *,
    manifest_checksums: dict[str, str],
    bundle_root: Path,
) -> tuple[str, str, list[str]]:
    """Validate one dissertation artifact row and return (artifact_id, output_path, errors)."""
    errors: list[str] = []
    artifact_id = str(artifact.get("artifact_id", "")).strip()
    if not artifact_id:
        errors.append("Missing artifact_id")
        return "", "", errors
    if not _ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        errors.append(f"Invalid artifact_id for dissertation artifact: {artifact_id!r}")
        return artifact_id, "", errors
    output_path_raw = str(artifact.get("output_path", "")).strip()
    source_path_raw = str(artifact.get("source_path", "")).strip()
    source_artifact = str(artifact.get("source_artifact", "")).strip()
    caption_draft = str(artifact.get("caption_draft", "")).strip()
    claim_boundary = str(artifact.get("claim_boundary", "")).strip()
    fallback = str(artifact.get("fallback_degraded_summary", "")).strip()
    manuscript_use = str(artifact.get("recommended_manuscript_use", "")).strip()
    digest = str(artifact.get("sha256", "")).strip()
    output_path = Path(output_path_raw)

    if output_path.is_absolute() or ".." in output_path.parts:
        errors.append(f"Invalid output_path for {artifact_id!r}: {output_path_raw!r}")
    if output_path.parts[:1] != ("artifacts",):
        errors.append(f"Unsafe output_path for {artifact_id!r}: must be under payload/artifacts/")

    if not source_path_raw or Path(source_path_raw).is_absolute():
        errors.append(f"Invalid source_path for {artifact_id!r}: {source_path_raw!r}")

    if not source_artifact:
        errors.append(f"Missing source_artifact for {artifact_id!r}")
    if not caption_draft:
        errors.append(f"Missing caption_draft for {artifact_id!r}")
    if not claim_boundary:
        errors.append(f"Missing claim_boundary for {artifact_id!r}")
    if not fallback:
        errors.append(f"Missing fallback_degraded_summary for {artifact_id!r}")
    if not digest:
        errors.append(f"Missing sha256 for {artifact_id!r}")
    if _is_weak_claim_boundary(claim_boundary) and manuscript_use == "results":
        errors.append(f"Artifact {artifact_id!r} promotes diagnostic/blocked evidence to results")

    _is_allowed_manuscript_use(manuscript_use, artifact_id)

    output_file = bundle_root / "payload" / output_path
    if not output_file.exists() or not output_file.is_file():
        errors.append(f"Missing payload artifact file for {artifact_id!r}: {output_path}")
    elif output_file.is_symlink():
        errors.append(f"Payload artifact file cannot be symlink: {artifact_id!r}")
    elif manifest_checksums.get(output_path.as_posix()) != digest:
        errors.append(f"Checksum table mismatch for {artifact_id!r}: {output_path}")
    elif _sha256_file(output_file) != digest:
        errors.append(f"Payload checksum mismatch for {artifact_id!r}: {output_path}")

    return artifact_id, output_path.as_posix(), errors


def _collect_dissertation_bundle(  # noqa: C901, PLR0912, PLR0915
    bundle_dir: Path,
    *,
    source_root: Path | None = None,
    expected_source_command: str | None = None,
    expected_source_commit: str | None = None,
    reference_manifest: Path | None = None,
) -> dict[str, object]:
    """Validate a dissertation bundle and return a review summary payload."""
    bundle_root = bundle_dir.resolve()
    manifest_path = bundle_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest.get("schema_version"):
        raise ValueError("Dissertation manifest missing schema_version")
    source_commit = str(manifest.get("source_commit", "")).strip()
    generation_command = str(manifest.get("generation_command", "")).strip()
    if not source_commit:
        raise ValueError("Dissertation manifest missing source_commit")
    if not generation_command:
        raise ValueError("Dissertation manifest missing generation_command")
    if expected_source_commit is not None and source_commit != expected_source_commit.strip():
        raise ValueError(
            f"source_commit mismatch: expected {expected_source_commit.strip()!r}, "
            f"got {source_commit!r}",
        )
    if (
        expected_source_command is not None
        and generation_command != expected_source_command.strip()
    ):
        raise ValueError(
            f"source_command mismatch: expected {expected_source_command.strip()!r}, "
            f"got {generation_command!r}",
        )

    checksums = _parse_checksums_file(bundle_root / "checksums.sha256")
    rows_raw = manifest.get("artifacts")
    if not isinstance(rows_raw, list):
        raise ValueError("Dissertation manifest requires artifacts list")

    row_errors: list[str] = []
    artifact_rows: dict[str, dict[str, object]] = {}
    checksums_paths: set[str] = set(checksums)
    seen_output_paths: set[str] = set()
    source_root_resolved = source_root.resolve() if source_root is not None else None
    for row in rows_raw:
        if not isinstance(row, dict):
            row_errors.append("Encountered non-object artifact row")
            continue
        artifact_id, output_path, errors = _validate_dissertation_manifest_row(
            row,
            manifest_checksums=checksums,
            bundle_root=bundle_root,
        )
        row_errors.extend(errors)
        if not artifact_id:
            continue
        if artifact_id in artifact_rows:
            row_errors.append(f"Duplicate artifact_id: {artifact_id!r}")
            continue
        if output_path in checksums_paths:
            checksums_paths.remove(output_path)
        if output_path in seen_output_paths:
            row_errors.append(f"Duplicate output_path: {output_path}")
            continue
        seen_output_paths.add(output_path)
        if source_root_resolved is not None:
            source_path_raw = str(row.get("source_path", "")).strip()
            if source_path_raw:
                unresolved_candidate = source_root_resolved / source_path_raw
                if unresolved_candidate.is_symlink():
                    row_errors.append(
                        f"Source path is symlink for {artifact_id!r}: {unresolved_candidate}"
                    )
                source_candidate = unresolved_candidate.resolve(strict=False)
                if not source_candidate.is_relative_to(source_root_resolved):
                    row_errors.append(f"Source path escapes source root: {source_candidate!r}")
                elif not source_candidate.is_file():
                    row_errors.append(
                        f"Source artifact missing for {artifact_id!r}: {source_candidate}"
                    )

        artifact_rows[artifact_id] = row

    for missing in sorted(checksums_paths):
        row_errors.append(f"Extra checksum entry present: {missing}")

    if row_errors:
        raise ValueError("\n".join(row_errors))

    if reference_manifest is not None:
        reference = _read_json(reference_manifest)
        reference_rows_raw = reference.get("artifacts")
        if not isinstance(reference_rows_raw, list):
            raise ValueError("Reference manifest requires artifacts list")
        reference_rows = {
            str(item.get("artifact_id", "")).strip(): item
            for item in reference_rows_raw
            if isinstance(item, dict) and str(item.get("artifact_id", "")).strip()
        }
        for artifact_id, row in artifact_rows.items():
            old = reference_rows.get(artifact_id)
            if not isinstance(old, dict):
                continue
            old_claim = str(old.get("claim_boundary", "")).strip()
            old_caveat = str(old.get("fallback_degraded_summary", "")).strip()
            new_claim = str(row.get("claim_boundary", "")).strip()
            new_caveat = str(row.get("fallback_degraded_summary", "")).strip()
            if _claim_is_weakened(old_claim, new_claim):
                raise ValueError(f"Claim boundary weakened for artifact_id={artifact_id!r}")
            if _caveat_is_weakened(old_caveat, new_caveat):
                raise ValueError(f"Caveat weakened for artifact_id={artifact_id!r}")

    return {
        "bundle_dir": str(bundle_root),
        "schema_version": str(manifest.get("schema_version", "")),
        "source_root": str(manifest.get("source_root", "")),
        "source_commit": source_commit,
        "generation_command": generation_command,
        "artifact_count": len(artifact_rows),
        "artifact_ids": sorted(artifact_rows),
    }


def _dissertation_artifact_state(bundle_dir: Path) -> tuple[dict[str, object], dict[str, str]]:
    """Return bundle artifact rows by id and checksum status."""
    manifest = _read_json(bundle_dir / "artifact_manifest.json")
    checksums = _parse_checksums_file(bundle_dir / "checksums.sha256")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("Dissertation bundle manifest must contain artifact list")

    by_id: dict[str, dict[str, object]] = {}
    status: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        artifact_id = str(row.get("artifact_id", "")).strip()
        output_raw = str(row.get("output_path", "")).strip()
        output_path = Path(output_raw)
        expected = str(row.get("sha256", "")).strip()
        if not artifact_id:
            continue
        if (
            not output_raw
            or output_path.is_absolute()
            or ".." in output_path.parts
            or output_path.parts[:1] != ("artifacts",)
        ):
            status[artifact_id] = "invalid_output_path"
            by_id[artifact_id] = row
            continue
        payload = bundle_dir / "payload" / output_path
        if not payload.exists() or payload.is_symlink():
            status[artifact_id] = "missing"
            by_id[artifact_id] = row
            continue
        digest = _sha256_file(payload)
        if not expected or expected != digest:
            status[artifact_id] = "checksum_manifest_mismatch"
            by_id[artifact_id] = row
            continue
        manifest_checksum = checksums.get(output_raw)
        if manifest_checksum != digest:
            status[artifact_id] = "checksum_inventory_mismatch"
            by_id[artifact_id] = row
            continue
        by_id[artifact_id] = row
    return by_id, status


def _build_diff_markdown(  # noqa: C901
    *,
    bundle_dir: Path,
    reference_bundle_dir: Path,
    current_rows: dict[str, dict[str, object]],
    reference_rows: dict[str, dict[str, object]],
    current_status: dict[str, str],
    reference_status: dict[str, str],
) -> str:
    """Build a compact dissertation review diff in Markdown."""
    current_ids = set(current_rows)
    reference_ids = set(reference_rows)
    changed_captions: list[str] = []
    missing_artifacts: list[str] = []
    weakened_claims: list[str] = []
    weakened_caveats: list[str] = []
    corrupted_artifacts: list[str] = []

    for artifact_id in sorted(current_ids & reference_ids):
        current = current_rows[artifact_id]
        reference = reference_rows[artifact_id]
        if (
            str(current.get("caption_draft", "")).strip()
            != str(reference.get("caption_draft", "")).strip()
        ):
            changed_captions.append(
                f"- `{artifact_id}`: caption changed\n  - Before: {reference.get('caption_draft', '')}\n  - After: {current.get('caption_draft', '')}"
            )
        current_claim = str(current.get("claim_boundary", "")).strip()
        reference_claim = str(reference.get("claim_boundary", "")).strip()
        if _claim_is_weakened(reference_claim, current_claim):
            weakened_claims.append(
                f"- `{artifact_id}` claim boundary weakened\n  - Before: {reference_claim}\n  - After: {current_claim}"
            )
        current_caveat = str(current.get("fallback_degraded_summary", "")).strip()
        reference_caveat = str(reference.get("fallback_degraded_summary", "")).strip()
        if _caveat_is_weakened(reference_caveat, current_caveat):
            weakened_caveats.append(
                f"- `{artifact_id}` caveat weakened\n  - Before: {reference_caveat}\n  - After: {current_caveat}"
            )

    for artifact_id in sorted(reference_ids - current_ids):
        missing_artifacts.append(f"- Removed artifact: `{artifact_id}`")
    for artifact_id in sorted(current_ids - reference_ids):
        missing_artifacts.append(f"- New artifact: `{artifact_id}`")

    for artifact_id, reason in sorted(current_status.items()):
        corrupted_artifacts.append(f"- `{artifact_id}` in current bundle: {reason}")
    for artifact_id, reason in sorted(reference_status.items()):
        corrupted_artifacts.append(f"- `{artifact_id}` in reference bundle: {reason}")

    lines = [
        "# Dissertation bundle review diff",
        f"- Current bundle: `{bundle_dir}`",
        f"- Reference bundle: `{reference_bundle_dir}`",
    ]
    if changed_captions:
        lines.append("")
        lines.append("## Changed captions")
        lines.extend(changed_captions)
    if missing_artifacts:
        lines.append("")
        lines.append("## Missing/added artifacts")
        lines.extend(missing_artifacts)
    if corrupted_artifacts:
        lines.append("")
        lines.append("## Corrupted artifacts")
        lines.extend(corrupted_artifacts)
    if weakened_claims:
        lines.append("")
        lines.append("## Weakened claim boundaries")
        lines.extend(weakened_claims)
    if weakened_caveats:
        lines.append("")
        lines.append("## Weakened caveats")
        lines.extend(weakened_caveats)
    if (
        not changed_captions
        and not missing_artifacts
        and not corrupted_artifacts
        and not weakened_claims
        and not weakened_caveats
    ):
        lines.append("")
        lines.append("No substantive diffs detected.")
    return "\n".join(lines)


def _run_validate_dissertation_bundle(args: argparse.Namespace) -> int:
    """Execute ``validate-dissertation-bundle`` and print a compact report."""
    report = _collect_dissertation_bundle(
        args.bundle_dir,
        source_root=args.source_root,
        expected_source_command=args.expected_source_command,
        expected_source_commit=args.expected_source_commit,
        reference_manifest=args.reference_manifest,
    )
    print(json.dumps(report, indent=2))
    return 0


def _run_diff_dissertation_bundle(args: argparse.Namespace) -> int:
    """Execute ``diff-dissertation-bundle`` and print Markdown review output."""
    current_rows, current_status = _dissertation_artifact_state(args.bundle_dir)
    reference_rows, reference_status = _dissertation_artifact_state(args.reference_bundle_dir)
    markdown = _build_diff_markdown(
        bundle_dir=args.bundle_dir,
        reference_bundle_dir=args.reference_bundle_dir,
        current_rows=current_rows,
        reference_rows=reference_rows,
        current_status=current_status,
        reference_status=reference_status,
    )
    print(markdown)
    return 0


def _run_dissertation_bundle(args: argparse.Namespace) -> int:
    """Execute the ``dissertation-bundle`` subcommand."""
    result = export_dissertation_artifact_bundle(
        args.source_root,
        args.out_dir,
        bundle_name=args.bundle_name,
        artifacts=_load_dissertation_artifacts(args.artifact_spec),
        command=str(args.generation_command),
        commit=str(args.commit),
        overwrite=bool(args.overwrite),
    )
    payload = {
        "bundle_dir": str(result.bundle_dir),
        "manifest_path": str(result.manifest_path),
        "checksums_path": str(result.checksums_path),
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _extract_chapter_target(artifact: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract optional chapter target and justification from an artifact.

    Supports the canonical ``chapter_target`` field, the
    ``dissertation_chapter`` alias used by the dissertation evidence ledger,
    and an optional nested ``metadata.chapter_target`` fallback.

    Returns:
        A tuple of (chapter_target, chapter_target_justification).
    """
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else None

    raw_target = artifact.get("chapter_target")
    if raw_target is None:
        raw_target = artifact.get("dissertation_chapter")
    if raw_target is None and metadata is not None:
        raw_target = metadata.get("chapter_target")
    chapter_target = str(raw_target).strip() if raw_target is not None else None
    if chapter_target == "":
        chapter_target = None

    raw_justification = artifact.get("chapter_target_justification")
    if raw_justification is None and metadata is not None:
        raw_justification = metadata.get("chapter_target_justification")
    justification = str(raw_justification).strip() if raw_justification is not None else None
    if justification == "":
        justification = None

    return chapter_target, justification


def _validate_chapter_target(
    evidence_tier: str,
    chapter_target: str | None,
    justification: str | None,
) -> str | None:
    """Validate that a chapter target is appropriate for the evidence tier.

    Diagnostic-only rows should map primarily to limitations, methodology, or
    future-work style targets unless explicitly justified.

    Returns:
        ``None`` when no target is present, ``"ok"`` when the target is
        acceptable, or a short warning string when the target looks mismatched.
    """
    if chapter_target is None:
        if justification:
            return "warning: justification provided but no chapter target is specified"
        return None
    normalized = chapter_target.lower()
    has_allowed_style = any(style in normalized for style in _DIAGNOSTIC_CHAPTER_TARGET_STYLES)
    if evidence_tier in {"diagnostic-only", "non-claimable"} and not (
        has_allowed_style or justification
    ):
        return (
            f"warning: {evidence_tier} row targets '{chapter_target}'; "
            "expected limitations/methodology/future-work style or explicit justification"
        )
    return "ok"


def _get_validation_status(
    artifact_id: str,
    source_path: str,
    checksum: str,
    claim_boundary: str,
    manuscript_use: str,
    bundle_dir: Path,
    bundle_checksums: dict[str, str],
    output_path_raw: str,
) -> str:
    """Determine the validation status for an artifact."""
    if not checksum or checksum == "N/A":
        return "non-claimable: missing checksum"
    if not source_path or source_path == "N/A":
        return "non-claimable: missing source path"
    if _is_weak_claim_boundary(claim_boundary) and manuscript_use == "results":
        return "invalid_claim: diagnostic promoted to results"

    payload_file = bundle_dir / "payload" / Path(output_path_raw)
    if not payload_file.exists() or not payload_file.is_file():
        return "non-claimable: missing payload file"
    if _sha256_file(payload_file) != checksum:
        return "non-claimable: payload checksum mismatch"
    if bundle_checksums.get(output_path_raw) != checksum:
        return "non-claimable: manifest checksum mismatch"

    return "valid"


def _determine_artifact_status(
    validation_status: str,
    claim_boundary: str,
    manuscript_use: str,
    fallback_summary: str,
    evidence_claim_boundaries: Sequence[str],
) -> tuple[str, str, str]:
    """Determine the evidence tier, allowed wording, and caveat for an artifact."""
    evidence_tier = "paper-grade"
    allowed_wording = manuscript_use
    caveat = fallback_summary
    weakest_evidence_boundary = next(
        (boundary for boundary in evidence_claim_boundaries if _is_weak_claim_boundary(boundary)),
        "",
    )

    if validation_status.startswith("non-claimable"):
        evidence_tier = "non-claimable"
        allowed_wording = "do-not-use"
    elif validation_status.startswith("invalid"):
        evidence_tier = "diagnostic-only"
        allowed_wording = "discussion"
    elif (
        _is_weak_claim_boundary(claim_boundary)
        or _is_weak_caveat(fallback_summary)
        or weakest_evidence_boundary
    ):
        evidence_tier = "diagnostic-only"
        if allowed_wording == "results":
            allowed_wording = "discussion"
        if weakest_evidence_boundary:
            caveat = f"{caveat} Evidence bundle boundary: {weakest_evidence_boundary}".strip()

    return evidence_tier, allowed_wording, caveat


def _process_dissertation_artifact(
    artifact: dict[str, Any],
    *,
    evidence_claim_boundaries: Sequence[str],
) -> ClaimMatrixRow:
    """Process a single dissertation artifact into a ClaimMatrixRow."""
    artifact_id = str(artifact.get("artifact_id", "N/A"))
    source_artifact = str(artifact.get("source_artifact", "N/A"))
    source_path = str(artifact.get("source_path", "N/A"))
    checksum = str(artifact.get("sha256", "N/A"))
    claim_boundary = str(artifact.get("claim_boundary", "N/A"))
    manuscript_use = str(artifact.get("recommended_manuscript_use", "N/A"))
    fallback_summary = str(artifact.get("fallback_degraded_summary", "N/A"))
    bundle_dir = artifact["_bundle_dir"]
    bundle_checksums = artifact["_bundle_checksums"]
    output_path_raw = str(artifact.get("output_path", "")).strip()

    validation_status = _get_validation_status(
        artifact_id,
        source_path,
        checksum,
        claim_boundary,
        manuscript_use,
        bundle_dir,
        bundle_checksums,
        output_path_raw,
    )

    evidence_tier, allowed_wording, caveat = _determine_artifact_status(
        validation_status=validation_status,
        claim_boundary=claim_boundary,
        manuscript_use=manuscript_use,
        fallback_summary=fallback_summary,
        evidence_claim_boundaries=evidence_claim_boundaries,
    )

    chapter_target, chapter_target_justification = _extract_chapter_target(artifact)
    chapter_target_status = _validate_chapter_target(
        evidence_tier, chapter_target, chapter_target_justification
    )

    return ClaimMatrixRow(
        artifact_id=artifact_id,
        source_artifact_path=f"{source_artifact} ({source_path})",
        checksum=checksum,
        evidence_tier=evidence_tier,
        allowed_wording=allowed_wording,
        not_claimed_boundary=claim_boundary,
        figure_table_candidate=str(artifact.get("caption_draft", "N/A")),
        caveat=caveat,
        validation_status=validation_status,
        chapter_target=chapter_target,
        chapter_target_status=chapter_target_status,
    )


def _collect_claim_matrix_inputs(
    *,
    bundle_dirs: Sequence[Path],
    evidence_bundle_dirs: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect dissertation artifact rows and evidence boundaries for claim matrix generation."""
    all_dissertation_artifacts: list[dict[str, Any]] = []
    evidence_claim_boundaries: list[str] = []

    for bundle_dir in bundle_dirs:
        manifest, checksums = _load_dissertation_bundle_data(bundle_dir)
        if not isinstance(manifest.get("artifacts"), list):
            logger.warning(
                "Dissertation bundle manifest {} has no 'artifacts' list; skipping.",
                bundle_dir,
            )
            continue
        for artifact in manifest["artifacts"]:
            artifact["_bundle_dir"] = bundle_dir
            artifact["_bundle_checksums"] = checksums
            all_dissertation_artifacts.append(artifact)

    for evidence_bundle_dir in evidence_bundle_dirs:
        manifest = _load_evidence_bundle_data(evidence_bundle_dir)
        boundary = str(manifest.get("claim_boundary", "")).strip()
        if boundary:
            evidence_claim_boundaries.append(boundary)

    return all_dissertation_artifacts, evidence_claim_boundaries


def _format_claim_matrix_cell(value: object) -> str:
    """Format a claim-matrix value for Markdown.

    Returns:
        Empty string for missing values, otherwise the string value.
    """
    if value is None:
        return ""
    return str(value).replace("|", r"\|")


def _write_claim_matrix_outputs(
    *,
    claims_rows: list[ClaimMatrixRow],
    json_output: Path,
    markdown_output: Path,
) -> None:
    """Write deterministic claim matrix JSON and Markdown outputs."""
    json_output_content: ClaimMatrix = {
        "schema_version": "claim_matrix.v1",
        "claims": claims_rows,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(json_output_content, indent=2) + "\n", encoding="utf-8")
    logger.info("Claim matrix JSON written to {}", json_output)

    markdown_lines = ["# Dissertation Claim Matrix", ""]
    if not claims_rows:
        markdown_lines.append("No claims found to display.")
    else:
        all_headers = list(ClaimMatrixRow.__annotations__.keys())
        has_chapter_target = any(row.get("chapter_target") is not None for row in claims_rows)
        if not has_chapter_target:
            headers = [
                h for h in all_headers if h not in ("chapter_target", "chapter_target_status")
            ]
        else:
            headers = all_headers
        readable_headers = [header.replace("_", " ").title() for header in headers]
        markdown_lines.append("| " + " | ".join(readable_headers) + " |")
        markdown_lines.append("|" + "---|".join(["-" * len(h) for h in readable_headers]) + "|")
        for row in claims_rows:
            markdown_lines.append(
                "| " + " | ".join([_format_claim_matrix_cell(row[key]) for key in headers]) + " |"
            )

    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    logger.info("Claim matrix Markdown written to {}", markdown_output)


def _run_claim_matrix(args: argparse.Namespace) -> int:
    """Execute the ``claim-matrix`` subcommand."""
    try:
        all_dissertation_artifacts, evidence_claim_boundaries = _collect_claim_matrix_inputs(
            bundle_dirs=args.bundle_dir,
            evidence_bundle_dirs=args.evidence_bundle_dir,
        )
    except Exception as exc:
        logger.error("Failed to load claim matrix inputs: {}", exc)
        return 1

    claims_rows = [
        _process_dissertation_artifact(
            artifact,
            evidence_claim_boundaries=evidence_claim_boundaries,
        )
        for artifact in all_dissertation_artifacts
    ]
    claims_rows.sort(key=lambda x: x["artifact_id"])
    _write_claim_matrix_outputs(
        claims_rows=claims_rows,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark publication helper CLI and return a POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "size-report":
        return _run_size_report(args)
    if args.command == "export":
        return _run_export(args)
    if args.command == "evidence-bundle":
        return _run_evidence_bundle(args)
    if args.command == "dissertation-bundle":
        return _run_dissertation_bundle(args)
    if args.command == "validate-dissertation-bundle":
        return _run_validate_dissertation_bundle(args)
    if args.command == "diff-dissertation-bundle":
        return _run_diff_dissertation_bundle(args)
    if args.command == "claim-matrix":
        return _run_claim_matrix(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
