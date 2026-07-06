"""Helpers for benchmark artifact publication and DOI-ready export bundles.

This module provides reusable logic for:
- discovering benchmark run directories,
- measuring artifact size distributions across runs,
- exporting a publication bundle with checksums and provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from robot_sf.common.artifact_paths import get_repository_root

PUBLICATION_BUNDLE_SCHEMA_VERSION = "benchmark-publication-bundle.v2"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "evidence_bundle.v1"
EVIDENCE_MIRROR_MANIFEST_SCHEMA_VERSION = "evidence_bundle_mirror.v1"
MANUSCRIPT_ARTIFACT_BUNDLE_SCHEMA_VERSION = "dissertation_artifact_bundle.v1"
SIZE_REPORT_SCHEMA_VERSION = "benchmark-artifact-size-report.v1"
_VIDEO_SUFFIXES = {".mp4", ".gif", ".webm", ".mov"}
_MARKER_FILES = ("manifest.json", "run_meta.json", "episodes.jsonl", "summary.json", "report.json")
_NON_RUN_DIRECTORY_NAMES = {"episodes", "aggregates", "reports", "plots", "videos", "artifacts"}
_DEFAULT_REPO_URL = "https://github.com/ll7/robot_sf_ll7"
_DEFAULT_RELEASE_TAG_TEMPLATE = "{release_tag}"
_DEFAULT_DOI_TEMPLATE = "10.5281/zenodo.<record-id>"
_CAMPAIGN_PREFLIGHT_REQUIRED = (
    "preflight/validate_config.json",
    "preflight/preview_scenarios.json",
)
_RECOMMENDED_MANUSCRIPT_USES = {
    "results",
    "methodology",
    "discussion",
    "outlook",
    "do-not-use",
}


@dataclass(frozen=True)
class PublicationFileEntry:
    """Metadata for one file included in a publication bundle."""

    path: str
    size_bytes: int
    sha256: str
    kind: str


@dataclass(frozen=True)
class PublicationBundleResult:
    """Paths and summary values produced by :func:`export_publication_bundle`."""

    bundle_dir: Path
    archive_path: Path
    manifest_path: Path
    checksums_path: Path
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class EvidenceBundleResult:
    """Paths and summary values produced by :func:`export_evidence_bundle`."""

    bundle_dir: Path
    manifest_path: Path
    checksums_path: Path
    file_count: int
    total_bytes: int
    mirror_manifest_path: Path | None = None


@dataclass(frozen=True)
class DissertationArtifactSpec:
    """User-selected figure or table artifact for dissertation handoff."""

    artifact_id: str
    source_path: Path
    source_artifact: str
    caption_draft: str
    claim_boundary: str
    recommended_manuscript_use: str
    fallback_degraded_summary: str
    metadata: dict[str, Any] | None = None
    chapter_target: str | None = None
    chapter_target_justification: str | None = None


@dataclass(frozen=True)
class DissertationArtifactBundleResult:
    """Paths and summary values produced by dissertation artifact bundle export."""

    bundle_dir: Path
    manifest_path: Path
    checksums_path: Path
    file_count: int
    total_bytes: int


def _utc_now_iso() -> str:
    """Return current UTC timestamp formatted as ISO-8601 with trailing ``Z``."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 for ``path`` using chunked reads.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _kind_for_path(relative_path: Path) -> str:
    """Infer a coarse artifact kind label from a run-relative path.

    Returns:
        Artifact kind label such as ``episodes``, ``reports``, or ``misc``.
    """
    normalized = relative_path.as_posix()
    top = relative_path.parts[0] if relative_path.parts else ""
    if top in {"episodes", "aggregates", "reports", "plots", "videos", "artifacts"}:
        return top
    if normalized.endswith("episodes.jsonl"):
        return "episodes"
    if relative_path.name in {"manifest.json", "run_meta.json"}:
        return "provenance"
    if relative_path.suffix.lower() in {".pdf", ".png", ".svg"}:
        return "plots"
    if relative_path.suffix.lower() in _VIDEO_SUFFIXES:
        return "videos"
    if "report" in relative_path.name:
        return "reports"
    if "summary" in relative_path.name:
        return "aggregates"
    return "misc"


def _to_repo_relative(path: Path) -> str:
    """Return a repository-relative path string when possible.

    Returns:
        Path relative to repository root, or ``path.name`` when outside repo.
    """
    resolved = path.resolve()
    repo_root = get_repository_root().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.name


def list_publication_files(run_dir: Path, *, include_videos: bool = True) -> list[Path]:
    """Return sorted run-relative file paths eligible for publication export.

    Args:
        run_dir: Benchmark run directory to inspect recursively.
        include_videos: Whether video artifacts should be included.

    Returns:
        Sorted list of run-relative file paths.
    """
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        return []

    selected: list[Path] = []
    for candidate in run_dir.rglob("*"):
        if candidate.is_symlink():
            continue
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(run_dir)
        if rel.name == ".DS_Store":
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        is_video = rel.suffix.lower() in _VIDEO_SUFFIXES or (
            rel.parts and rel.parts[0].lower() == "videos"
        )
        if is_video and not include_videos:
            continue
        selected.append(rel)
    return sorted(selected, key=lambda value: value.as_posix())


def _directory_has_markers(directory: Path) -> bool:
    """Return ``True`` when a directory appears to be a benchmark run root."""
    if directory.name in _NON_RUN_DIRECTORY_NAMES:
        return False
    for marker in _MARKER_FILES:
        if (directory / marker).exists():
            return True
    return False


def discover_run_directories(benchmarks_root: Path) -> list[Path]:
    """Discover leaf benchmark run directories under ``benchmarks_root``.

    A candidate run directory must contain at least one marker file such as
    ``manifest.json``, ``run_meta.json``, or ``episodes.jsonl``.

    Returns:
        Sorted list of leaf run directories discovered under ``benchmarks_root``.
    """
    root = benchmarks_root.resolve()
    if not root.exists() or not root.is_dir():
        return []

    candidates = [
        path for path in root.rglob("*") if path.is_dir() and _directory_has_markers(path)
    ]
    if _directory_has_markers(root):
        candidates.append(root)
    candidates = sorted(set(candidates), key=lambda value: value.as_posix())

    candidate_set = set(candidates)
    leaves: list[Path] = []
    for candidate in candidates:
        has_marker_descendant = any(
            other != candidate and other.is_relative_to(candidate) for other in candidate_set
        )
        if not has_marker_descendant:
            leaves.append(candidate)
    return sorted(leaves, key=lambda value: value.as_posix())


def _percentile(values: list[float], q: float) -> float:
    """Return linear-interpolated percentile for ``values`` at quantile ``q``."""
    if not values:
        return 0.0
    clamped = min(1.0, max(0.0, q))
    return float(np.percentile(values, clamped * 100.0, method="linear"))


def _distribution(values: list[int]) -> dict[str, float | int]:
    """Summarize integer values with stable quantiles and basic statistics.

    Returns:
        Mapping with count, min, p50, p90, max, and mean fields.
    """
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    as_float = [float(value) for value in values]
    return {
        "count": len(values),
        "min": float(min(values)),
        "p50": _percentile(as_float, 0.50),
        "p90": _percentile(as_float, 0.90),
        "max": float(max(values)),
        "mean": float(mean(values)),
    }


def measure_artifact_size_ranges(
    benchmarks_root: Path,
    *,
    include_videos: bool = True,
) -> dict[str, Any]:
    """Measure benchmark run size ranges across discovered run directories.

    Args:
        benchmarks_root: Root containing benchmark run outputs.
        include_videos: Whether videos should be included in size totals.

    Returns:
        JSON-serializable report with per-run and aggregate distributions.
    """
    runs = discover_run_directories(benchmarks_root)
    run_records: list[dict[str, Any]] = []
    total_sizes: list[int] = []
    bytes_by_kind: dict[str, list[int]] = defaultdict(list)

    for run_dir in runs:
        files = list_publication_files(run_dir, include_videos=include_videos)
        total = 0
        per_kind_totals: dict[str, int] = defaultdict(int)
        for rel in files:
            file_size = int((run_dir / rel).stat().st_size)
            total += file_size
            per_kind_totals[_kind_for_path(rel)] += file_size
        total_sizes.append(total)
        for kind, size_value in per_kind_totals.items():
            bytes_by_kind[kind].append(size_value)
        run_records.append(
            {
                "run_dir": _to_repo_relative(run_dir),
                "file_count": len(files),
                "total_bytes": total,
                "bytes_by_kind": dict(sorted(per_kind_totals.items())),
            }
        )

    distributions: dict[str, dict[str, float | int]] = {"total_bytes": _distribution(total_sizes)}
    for kind in sorted(bytes_by_kind):
        distributions[f"{kind}_bytes"] = _distribution(bytes_by_kind[kind])

    return {
        "schema_version": SIZE_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "benchmarks_root": _to_repo_relative(benchmarks_root),
        "run_count": len(run_records),
        "include_videos": bool(include_videos),
        "distributions": distributions,
        "runs": run_records,
    }


def _build_provenance(run_dir: Path) -> dict[str, Any]:  # noqa: C901
    """Build provenance metadata from optional ``manifest.json`` and ``run_meta.json``.

    Returns:
        JSON-serializable provenance dictionary with run and repository metadata.
    """
    provenance: dict[str, Any] = {
        "run_dir": _to_repo_relative(run_dir),
        "run_id": run_dir.name,
    }

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            manifest_payload = None
        if isinstance(manifest_payload, dict):
            selected = {
                key: manifest_payload.get(key)
                for key in (
                    "git_hash",
                    "scenario_matrix_hash",
                    "runtime_sec",
                    "episodes_per_second",
                )
            }
            provenance["manifest"] = {
                key: value for key, value in selected.items() if value is not None
            }

    run_meta_path = run_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            run_meta_payload = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            run_meta_payload = None
        if isinstance(run_meta_payload, dict):
            repo = run_meta_payload.get("repo")
            if isinstance(repo, dict):
                provenance["repository"] = {
                    key: repo.get(key)
                    for key in ("remote", "branch", "commit")
                    if repo.get(key) is not None
                }
            matrix_path = run_meta_payload.get("matrix_path")
            if isinstance(matrix_path, str) and matrix_path:
                matrix_candidate = Path(matrix_path)
                if not matrix_candidate.is_absolute():
                    matrix_candidate = get_repository_root() / matrix_candidate
                provenance["matrix_path"] = _to_repo_relative(matrix_candidate)
            seed_policy = run_meta_payload.get("seed_policy")
            if isinstance(seed_policy, dict):
                provenance["seed_policy"] = {
                    "mode": seed_policy.get("mode"),
                    "seed_set": seed_policy.get("seed_set"),
                    "seeds": seed_policy.get("seeds"),
                    "resolved_seeds": seed_policy.get("resolved_seeds"),
                }
            preflight_artifacts = run_meta_payload.get("preflight_artifacts")
            if isinstance(preflight_artifacts, dict):
                provenance["preflight_artifacts"] = {
                    key: value for key, value in preflight_artifacts.items() if value
                }
    return provenance


def validate_artifact_badging_block(block: dict[str, Any]) -> None:  # noqa: C901
    """Validate the schema and values of an artifact_badging dictionary."""
    if not isinstance(block, dict):
        raise ValueError("artifact_badging must be a dictionary")

    claimed_level = block.get("claimed_level")
    valid_levels = {"available", "functional", "reproduced", "none"}
    if claimed_level not in valid_levels:
        raise ValueError(
            f"Invalid claimed_level: {claimed_level!r}; expected one of {valid_levels}"
        )

    checklist_path = block.get("checklist_path")
    if checklist_path is not None and not isinstance(checklist_path, str):
        raise ValueError("checklist_path must be a string")

    claim_boundary = block.get("claim_boundary")
    if claim_boundary is not None and not isinstance(claim_boundary, str):
        raise ValueError("claim_boundary must be a string")

    smoke_status = block.get("functional_smoke_status")
    valid_smoke_statuses = {"passed", "failed", "not_run", "not_applicable"}
    if smoke_status is not None and smoke_status not in valid_smoke_statuses:
        raise ValueError(
            f"Invalid functional_smoke_status: {smoke_status!r}; expected one of {valid_smoke_statuses}"
        )

    reprod_status = block.get("reproduction_status")
    valid_reprod_statuses = {"passed", "failed", "not_run", "not_applicable"}
    if reprod_status is not None and reprod_status not in valid_reprod_statuses:
        raise ValueError(
            f"Invalid reproduction_status: {reprod_status!r}; expected one of {valid_reprod_statuses}"
        )

    nondet = block.get("known_nondeterminism")
    if nondet is not None and not isinstance(nondet, list):
        raise ValueError("known_nondeterminism must be a list")
    if nondet is not None:
        for item in nondet:
            if not isinstance(item, str):
                raise ValueError("known_nondeterminism items must be strings")


def _validate_publication_requirements(run_root: Path, selected_files: list[Path]) -> None:
    """Validate bundle completeness requirements for publication-critical runs."""
    selected_set = {path.as_posix() for path in selected_files}
    campaign_manifest = run_root / "campaign_manifest.json"
    if not campaign_manifest.exists():
        return
    missing = [path for path in _CAMPAIGN_PREFLIGHT_REQUIRED if path not in selected_set]
    if missing:
        raise ValueError(
            "Publication bundle missing required preflight artifacts: " + ", ".join(sorted(missing))
        )


def _validate_bundle_name(bundle_name: str) -> None:
    """Validate user-provided bundle name for safe output path construction."""
    name_path = Path(bundle_name)
    if (
        not bundle_name
        or name_path.is_absolute()
        or ".." in name_path.parts
        or len(name_path.parts) != 1
    ):
        raise ValueError(f"Invalid bundle_name: {bundle_name!r}")


def _resolve_bundle_output_paths(
    out_dir: Path,
    target_name: str,
) -> tuple[Path, Path, Path]:
    """Resolve and validate target root, bundle directory, and archive path.

    Returns:
        Tuple of ``(target_root, bundle_dir, archive_path)`` paths.
    """
    target_root = out_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = (target_root / target_name).resolve()
    archive_path = (target_root / f"{target_name}.tar.gz").resolve()
    if not bundle_dir.is_relative_to(target_root) or not archive_path.is_relative_to(target_root):
        raise ValueError("Bundle output must remain within out_dir")
    return target_root, bundle_dir, archive_path


def _resolve_evidence_files(source_root: Path, files: list[Path]) -> list[Path]:
    """Resolve user-selected evidence files relative to ``source_root``.

    Returns:
        Sorted source-root-relative paths.
    """
    root = source_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Evidence source root does not exist: {root}")
    if not files:
        raise ValueError("At least one evidence file is required")

    resolved: list[Path] = []
    for file_path in files:
        candidate = file_path if file_path.is_absolute() else root / file_path
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Evidence file escapes source root: {file_path}")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Evidence file does not exist: {candidate}")
        if candidate.is_symlink():
            raise ValueError(f"Evidence bundle refuses symlink payloads: {candidate}")
        resolved.append(candidate.relative_to(root))
    return sorted(set(resolved), key=lambda value: value.as_posix())


def _guess_mime_type(path: Path) -> str:
    """Return a stable MIME/type hint for an evidence payload path."""
    guessed, _ = mimetypes.guess_type(path.as_posix())
    return guessed or "application/octet-stream"


def _join_remote_uri(base_uri: str, relative_path: str) -> str:
    """Join a mirror base URI and POSIX relative path without recording credentials.

    Returns:
        Joined remote URI string.
    """
    return f"{base_uri.rstrip('/')}/{relative_path.lstrip('/')}"


def _write_evidence_mirror_manifest(
    *,
    bundle_dir: Path,
    bundle_name: str,
    payload_root: Path,
    entries: list[PublicationFileEntry],
    dry_run_base_uri: str | None,
    local_dir: Path | None,
    overwrite: bool,
) -> Path | None:
    """Write optional mirror manifest for evidence bundle payloads.

    The dry-run URI mode records where payloads would be mirrored without touching any remote
    service. The local backend is a credential-free concrete backend used for tests and local
    staging; cloud backends can consume the same manifest shape later.

    Returns:
        Path to the mirror manifest when mirroring is enabled, otherwise ``None``.
    """
    if dry_run_base_uri and local_dir is not None:
        raise ValueError("Choose only one evidence mirror backend")
    if not dry_run_base_uri and local_dir is None:
        return None

    backend = "dry_run_uri" if dry_run_base_uri else "local_file"
    mode = "dry_run" if dry_run_base_uri else "upload"
    mirror_root: Path | None = None
    if local_dir is not None:
        mirror_root = local_dir.resolve()
        if mirror_root.exists() and not mirror_root.is_dir():
            raise ValueError(f"Evidence mirror local_dir is not a directory: {mirror_root}")
        mirror_root.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    for entry in entries:
        payload_rel = Path("payload") / entry.path
        payload_path = payload_root / entry.path
        if mirror_root is not None:
            mirror_path = mirror_root / bundle_name / payload_rel
            if mirror_path.exists() and not overwrite:
                raise FileExistsError(f"Evidence mirror target already exists: {mirror_path}")
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(payload_path, mirror_path)
            remote_uri = mirror_path.resolve().as_uri()
            upload_status = "uploaded"
        else:
            remote_uri = _join_remote_uri(str(dry_run_base_uri), payload_rel.as_posix())
            upload_status = "dry_run"
        assets.append(
            {
                "path": entry.path,
                "source_path": payload_rel.as_posix(),
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
                "kind": entry.kind,
                "mime_type": _guess_mime_type(Path(entry.path)),
                "remote_uri": remote_uri,
                "upload_status": upload_status,
            }
        )

    manifest = {
        "schema_version": EVIDENCE_MIRROR_MANIFEST_SCHEMA_VERSION,
        "evidence_bundle_schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "bundle_name": bundle_name,
        "backend": backend,
        "mode": mode,
        "credentials": "not_recorded",
        "assets": assets,
    }
    manifest_path = bundle_dir / "mirror_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _validate_non_empty(value: str, field: str, artifact_id: str) -> str:
    """Validate a required string field from an artifact spec.

    Returns:
        The stripped string value.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"Artifact {artifact_id!r} requires non-empty {field}")
    return stripped


def _artifact_payload_path(artifact_id: str, source_path: Path) -> Path:
    """Return a deterministic payload path for a dissertation artifact."""
    suffix = source_path.suffix
    return Path("artifacts") / f"{artifact_id}{suffix}"


def _validate_dissertation_artifacts(
    source_root: Path,
    artifacts: list[DissertationArtifactSpec],
) -> list[tuple[DissertationArtifactSpec, Path, Path]]:
    """Validate dissertation artifact specs and resolve source/output paths.

    Returns:
        Tuples of the artifact spec, source-root-relative path, and payload path.
    """
    root = source_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dissertation source root does not exist: {root}")
    if not artifacts:
        raise ValueError("At least one dissertation artifact is required")

    seen_ids: set[str] = set()
    resolved: list[tuple[DissertationArtifactSpec, Path, Path]] = []
    for artifact in artifacts:
        artifact_id = _validate_non_empty(artifact.artifact_id, "artifact_id", "<unknown>")
        if not all(char.isalnum() or char in "_-" for char in artifact_id):
            raise ValueError(
                f"Invalid artifact_id: {artifact_id!r}. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )
        if artifact_id in seen_ids:
            raise ValueError(f"Duplicate artifact_id: {artifact_id!r}")
        seen_ids.add(artifact_id)

        if artifact.recommended_manuscript_use not in _RECOMMENDED_MANUSCRIPT_USES:
            allowed = ", ".join(sorted(_RECOMMENDED_MANUSCRIPT_USES))
            raise ValueError(
                f"Artifact {artifact_id!r} has invalid recommended_manuscript_use "
                f"{artifact.recommended_manuscript_use!r}; expected one of: {allowed}"
            )
        _validate_non_empty(artifact.source_artifact, "source_artifact", artifact_id)
        _validate_non_empty(artifact.caption_draft, "caption_draft", artifact_id)
        _validate_non_empty(artifact.claim_boundary, "claim_boundary", artifact_id)
        _validate_non_empty(
            artifact.fallback_degraded_summary,
            "fallback_degraded_summary",
            artifact_id,
        )

        unresolved = (
            artifact.source_path
            if artifact.source_path.is_absolute()
            else root / artifact.source_path
        )
        if unresolved.is_symlink():
            raise ValueError(
                f"Dissertation artifact bundle refuses symlink payloads: {artifact.source_path}"
            )
        candidate = unresolved.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Artifact source escapes source root: {artifact.source_path}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Artifact source file does not exist: {candidate}")
        payload_path = _artifact_payload_path(artifact_id, candidate)
        resolved.append((artifact, candidate.relative_to(root), payload_path))
    return resolved


def _build_dissertation_manifest_entry(
    artifact: DissertationArtifactSpec,
    rel_source: Path,
    payload_path: Path,
    digest: str,
    size_bytes: int,
    commit: str,
    command: str,
) -> dict[str, Any]:
    """Build one artifact manifest entry for a dissertation bundle.

    Returns:
        JSON-serializable manifest entry for the artifact.
    """
    entry: dict[str, Any] = {
        "artifact_id": artifact.artifact_id,
        "source_path": rel_source.as_posix(),
        "source_artifact": artifact.source_artifact,
        "output_path": payload_path.as_posix(),
        "size_bytes": size_bytes,
        "sha256": digest,
        "source_commit": commit.strip(),
        "generation_command": command.strip(),
        "caption_draft": artifact.caption_draft.strip(),
        "claim_boundary": artifact.claim_boundary.strip(),
        "recommended_manuscript_use": artifact.recommended_manuscript_use,
        "fallback_degraded_summary": artifact.fallback_degraded_summary.strip(),
        "metadata": artifact.metadata or {},
    }
    chapter_target = artifact.chapter_target.strip() if artifact.chapter_target is not None else ""
    if chapter_target:
        entry["chapter_target"] = chapter_target
    chapter_target_justification = (
        artifact.chapter_target_justification.strip()
        if artifact.chapter_target_justification is not None
        else ""
    )
    if chapter_target_justification and not chapter_target:
        raise ValueError(
            f"Artifact {artifact.artifact_id!r} sets chapter_target_justification "
            "without chapter_target"
        )
    if chapter_target_justification:
        entry["chapter_target_justification"] = chapter_target_justification
    return entry


def export_dissertation_artifact_bundle(
    source_root: Path,
    out_dir: Path,
    *,
    bundle_name: str,
    artifacts: list[DissertationArtifactSpec],
    command: str,
    commit: str,
    overwrite: bool = False,
) -> DissertationArtifactBundleResult:
    """Export selected figure/table artifacts with dissertation-facing provenance.

    The bundle layout contains:
    - ``payload/artifacts/``: selected figure/table source files.
    - ``artifact_manifest.json``: JSON manifest with caption,
      checksum, manuscript-use, caveat, source, and claim-boundary metadata.
    - ``checksums.sha256``: SHA-256 checksums for all payload artifacts.

    Returns:
        Paths and totals describing the exported dissertation artifact bundle.
    """
    if not command.strip():
        raise ValueError("Dissertation artifact generation command is required")
    if not commit.strip():
        raise ValueError("Dissertation artifact source commit is required")

    target_name = bundle_name.strip()
    _validate_bundle_name(target_name)
    target_root = out_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = (target_root / target_name).resolve()
    if not bundle_dir.is_relative_to(target_root):
        raise ValueError("Dissertation artifact bundle output must remain within out_dir")

    if overwrite:
        if bundle_dir.exists():
            if bundle_dir == target_root:
                raise ValueError("Refusing to delete out_dir via overwrite")
            if bundle_dir.is_dir() and not bundle_dir.is_symlink():
                shutil.rmtree(bundle_dir)
            else:
                bundle_dir.unlink()
    elif bundle_dir.exists():
        raise FileExistsError(f"Dissertation artifact bundle output already exists: {bundle_dir}")

    source = source_root.resolve()
    selected = _validate_dissertation_artifacts(source, artifacts)
    payload_root = bundle_dir / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    checksums: list[str] = []
    total_bytes = 0
    for artifact, rel_source, payload_path in selected:
        dst = payload_root / payload_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel_source, dst)
        size_bytes = int(dst.stat().st_size)
        digest = _sha256_file(dst)
        total_bytes += size_bytes
        checksums.append(f"{digest}  {payload_path.as_posix()}\n")
        manifest_entries.append(
            _build_dissertation_manifest_entry(
                artifact, rel_source, payload_path, digest, size_bytes, commit, command
            )
        )

    manifest_entries = sorted(manifest_entries, key=lambda entry: str(entry["artifact_id"]))
    checksums_path = bundle_dir / "checksums.sha256"
    checksums_path.write_text("".join(sorted(checksums)), encoding="utf-8")
    manifest_payload = {
        "schema_version": MANUSCRIPT_ARTIFACT_BUNDLE_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "bundle_name": target_name,
        "source_root": _to_repo_relative(source),
        "source_commit": commit.strip(),
        "generation_command": command.strip(),
        "policy": {
            "local_output_disposable": True,
            "paper_or_benchmark_claim": "not_established_by_bundle_alone",
            "fallback_or_degraded_rows": "must_remain_visible_in_caption_or_table_notes",
        },
        "totals": {"artifact_count": len(manifest_entries), "total_bytes": total_bytes},
        "artifacts": manifest_entries,
    }
    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    return DissertationArtifactBundleResult(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        file_count=len(manifest_entries),
        total_bytes=total_bytes,
    )


def export_evidence_bundle(  # noqa: PLR0913, C901
    source_root: Path,
    out_dir: Path,
    *,
    bundle_name: str,
    files: list[Path],
    command: str,
    commit: str,
    claim_boundary: str,
    overwrite: bool = False,
    mirror_dry_run_base_uri: str | None = None,
    mirror_local_dir: Path | None = None,
    artifact_badging: dict[str, Any] | None = None,
) -> EvidenceBundleResult:
    """Export a compact reproducible evidence bundle.

    The bundle layout contains:
    - ``payload/``: exactly the selected compact evidence files.
    - ``evidence_bundle_manifest.json``: schema-tagged provenance + file index.
    - ``checksums.sha256``: SHA-256 checksums for all payload files.

    Returns:
        Paths and totals describing the exported bundle artifacts.
    """
    if not command.strip():
        raise ValueError("Evidence bundle command is required")
    if not commit.strip():
        raise ValueError("Evidence bundle commit is required")
    if not claim_boundary.strip():
        raise ValueError("Evidence bundle claim_boundary is required")

    target_name = bundle_name.strip()
    _validate_bundle_name(target_name)
    target_root = out_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = (target_root / target_name).resolve()
    if not bundle_dir.is_relative_to(target_root):
        raise ValueError("Evidence bundle output must remain within out_dir")

    if overwrite:
        if bundle_dir.exists():
            if bundle_dir == target_root:
                raise ValueError("Refusing to delete out_dir via overwrite")
            shutil.rmtree(bundle_dir)
    elif bundle_dir.exists():
        raise FileExistsError(f"Evidence bundle output already exists: {bundle_dir}")

    source = source_root.resolve()
    selected_files = _resolve_evidence_files(source, files)
    payload_root = bundle_dir / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)

    entries: list[PublicationFileEntry] = []
    total_bytes = 0
    for rel in selected_files:
        src = source / rel
        dst = payload_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size_bytes = int(dst.stat().st_size)
        total_bytes += size_bytes
        entries.append(
            PublicationFileEntry(
                path=rel.as_posix(),
                size_bytes=size_bytes,
                sha256=_sha256_file(dst),
                kind=_kind_for_path(rel),
            )
        )

    entries = sorted(entries, key=lambda entry: entry.path)
    checksums_path = bundle_dir / "checksums.sha256"
    checksums_path.write_text(
        "".join(f"{entry.sha256}  {entry.path}\n" for entry in entries),
        encoding="utf-8",
    )

    manifest_payload = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "bundle_name": target_name,
        "source_root": _to_repo_relative(source),
        "command": command.strip(),
        "commit": commit.strip(),
        "claim_boundary": claim_boundary.strip(),
        "policy": {
            "large_raw_artifacts": "excluded",
            "output_tree_mirroring": "forbidden",
            "paper_or_benchmark_claim": "not_established_by_bundle_alone",
        },
        "totals": {"file_count": len(entries), "total_bytes": total_bytes},
        "files": [asdict(entry) for entry in entries],
    }
    if artifact_badging is not None:
        validate_artifact_badging_block(artifact_badging)
        manifest_payload["artifact_badging"] = artifact_badging
    manifest_path = bundle_dir / "evidence_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    mirror_manifest_path = _write_evidence_mirror_manifest(
        bundle_dir=bundle_dir,
        bundle_name=target_name,
        payload_root=payload_root,
        entries=entries,
        dry_run_base_uri=mirror_dry_run_base_uri,
        local_dir=mirror_local_dir,
        overwrite=overwrite,
    )

    return EvidenceBundleResult(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        file_count=len(entries),
        total_bytes=total_bytes,
        mirror_manifest_path=mirror_manifest_path,
    )


def _compute_and_emit_badging_artifacts(
    bundle_dir: Path,
    manifest_payload: dict[str, Any],
    badging_block: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Compute achieved badge level, write/update manifest and README.md in bundle_dir.

    Returns:
        Tuple of computed badging block dict and achieved level string.
    """
    # 1. Start by copying existing values
    computed_badging = {
        "checklist_path": badging_block.get("checklist_path", "docs/RELEASE.md"),
        "claim_boundary": badging_block.get("claim_boundary", "reproducible-smoke-run"),
        "functional_smoke_status": badging_block.get("functional_smoke_status", "not_run"),
        "reproduction_status": badging_block.get("reproduction_status", "not_run"),
        "known_nondeterminism": badging_block.get("known_nondeterminism", []),
    }

    # 2. Check "available" criteria
    pub_channels = manifest_payload.get("publication_channels", {})
    doi = pub_channels.get("doi")
    rel_url = pub_channels.get("release_url")

    def is_valid_durable_id(val: str | None) -> bool:
        if not val:
            return False
        v = val.strip().lower()
        return (
            not any(v.startswith(prefix) for prefix in ("output/", "file://", "./", "../"))
            and "localhost" not in v
        )

    has_durable_id = is_valid_durable_id(doi) or is_valid_durable_id(rel_url)
    is_available = has_durable_id and len(manifest_payload.get("files", [])) > 0

    # 3. Check "functional" criteria by actually running the re-derivation/comparison check
    payload_dir = bundle_dir / "payload"
    campaign_summary_files = list(payload_dir.glob("**/campaign_summary.json"))
    campaign_report_files = list(payload_dir.glob("**/campaign_report.md"))

    is_functional = False
    if is_available and campaign_summary_files and campaign_report_files:
        summary_path = campaign_summary_files[0]
        report_path = campaign_report_files[0]
        try:
            from robot_sf.benchmark.camera_ready._reporting import (  # noqa: PLC0415
                write_campaign_report,
            )

            payload_data = json.loads(summary_path.read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_report_path = Path(tmpdir) / "campaign_report.md"
                write_campaign_report(temp_report_path, payload_data)

                shipped_content = report_path.read_text(encoding="utf-8").replace("\r\n", "\n")
                derived_content = temp_report_path.read_text(encoding="utf-8").replace("\r\n", "\n")

                if shipped_content == derived_content:
                    is_functional = True
                    computed_badging["functional_smoke_status"] = "passed"
                else:
                    computed_badging["functional_smoke_status"] = "failed"
        except (OSError, ValueError, json.JSONDecodeError):
            computed_badging["functional_smoke_status"] = "failed"
    elif computed_badging["functional_smoke_status"] == "passed":
        is_functional = True

    # Determine achieved badge level
    if not is_available:
        achieved_level = "none"
    elif not is_functional:
        achieved_level = "available"
    elif computed_badging["reproduction_status"] == "passed":
        achieved_level = "reproduced"
    else:
        achieved_level = "functional"

    computed_badging["claimed_level"] = achieved_level

    # 4. Write README.md into bundle_dir
    readme_path = bundle_dir / "README.md"
    readme_content = f"""# Release Bundle: {manifest_payload.get("bundle_name", "bundle")}

## Achieved Reproducibility Badge: **{achieved_level}**

This release bundle has been automatically validated and badge-certified:
- **Badge Level**: `{achieved_level}`
- **Durable DOI**: `{doi or "None"}`
- **Release Tag**: `{pub_channels.get("release_tag", "None")}`
- **Created at (UTC)**: `{manifest_payload.get("created_at_utc", "unknown")}`

### Reproducibility Rubric Definitions
- **available**: Bundle is published, hash-pinned, and the manifest is complete.
- **functional**: Bundle is self-sufficient. A CI job has successfully re-derived the headline tables from the bundle's summary JSON.
- **reproduced**: Benchmark results are reproduced within tolerance.

---
For details on verification, see [release_artifact_badging.md](docs/release_artifact_badging.md) or the repository documentation.
"""
    readme_path.write_text(readme_content, encoding="utf-8")

    return computed_badging, achieved_level


def export_publication_bundle(  # noqa: PLR0913
    run_dir: Path,
    out_dir: Path,
    *,
    bundle_name: str | None = None,
    include_videos: bool = True,
    repository_url: str = _DEFAULT_REPO_URL,
    release_tag: str = _DEFAULT_RELEASE_TAG_TEMPLATE,
    doi: str = _DEFAULT_DOI_TEMPLATE,
    overwrite: bool = False,
    artifact_badging: dict[str, Any] | None = None,
) -> PublicationBundleResult:
    """Export a DOI-ready publication bundle for one benchmark run.

    The bundle layout contains:
    - ``payload/``: selected benchmark files.
    - ``publication_manifest.json``: schema-tagged provenance + file index.
    - ``checksums.sha256``: SHA-256 checksums for all payload files.
    - ``<bundle_name>.tar.gz``: compressed archive of the full bundle directory.

    Returns:
        Paths and totals describing the exported bundle artifacts.
    """
    run_root = run_dir.resolve()
    if not run_root.exists() or not run_root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_root}")

    selected_files = list_publication_files(run_root, include_videos=include_videos)
    if not selected_files:
        raise ValueError(f"No eligible files found under {run_root}")
    _validate_publication_requirements(run_root, selected_files)

    target_name = bundle_name.strip() if bundle_name else f"{run_root.name}_publication_bundle"
    _validate_bundle_name(target_name)
    target_root, bundle_dir, archive_path = _resolve_bundle_output_paths(out_dir, target_name)

    if overwrite:
        if bundle_dir.exists():
            if bundle_dir == target_root:
                raise ValueError("Refusing to delete out_dir via overwrite")
            shutil.rmtree(bundle_dir)
        if archive_path.exists():
            archive_path.unlink()
    elif bundle_dir.exists() or archive_path.exists():
        raise FileExistsError(
            f"Bundle output already exists (dir or archive): {bundle_dir} / {archive_path}"
        )

    payload_root = bundle_dir / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)

    entries: list[PublicationFileEntry] = []
    total_bytes = 0
    for rel in selected_files:
        src = run_root / rel
        dst = payload_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size_bytes = int(dst.stat().st_size)
        total_bytes += size_bytes
        entries.append(
            PublicationFileEntry(
                path=rel.as_posix(),
                size_bytes=size_bytes,
                sha256=_sha256_file(dst),
                kind=_kind_for_path(rel),
            )
        )

    entries = sorted(entries, key=lambda entry: entry.path)
    checksums_path = bundle_dir / "checksums.sha256"
    checksums_payload = "".join(f"{entry.sha256}  {entry.path}\n" for entry in entries)
    checksums_path.write_text(checksums_payload, encoding="utf-8")

    manifest_payload = {
        "schema_version": PUBLICATION_BUNDLE_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "bundle_name": target_name,
        "include_videos": bool(include_videos),
        "policy_profile": "public-paper-v1",
        "publication_channels": {
            "repository_url": repository_url,
            "release_tag": release_tag,
            "release_url": f"{repository_url}/releases/tag/{release_tag}",
            "doi": doi,
        },
        "provenance": _build_provenance(run_root),
        "totals": {"file_count": len(entries), "total_bytes": total_bytes},
        "files": [asdict(entry) for entry in entries],
    }

    # Dynamically compute badging block and emit README
    if artifact_badging is not None:
        validate_artifact_badging_block(artifact_badging)
    badging_block = dict(artifact_badging) if artifact_badging else {}
    computed_badging, _achieved_level = _compute_and_emit_badging_artifacts(
        bundle_dir, manifest_payload, badging_block
    )
    validate_artifact_badging_block(computed_badging)
    manifest_payload["artifact_badging"] = computed_badging

    manifest_path = bundle_dir / "publication_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    with tarfile.open(archive_path, "w:gz") as handle:
        handle.add(bundle_dir, arcname=target_name)

    return PublicationBundleResult(
        bundle_dir=bundle_dir,
        archive_path=archive_path,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        file_count=len(entries),
        total_bytes=total_bytes,
    )
