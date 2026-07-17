"""Helpers for benchmark artifact publication and DOI-ready export bundles.

This module provides reusable logic for:
- discovering benchmark run directories,
- measuring artifact size distributions across runs,
- exporting a publication bundle with checksums and provenance metadata.
"""

from __future__ import annotations

import json
import math
import mimetypes
import shutil
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.benchmark.metrics import snqi as _curvature_aware_snqi
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping as _load_snqi_baseline_mapping,
)
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_weight_mapping as _load_snqi_weight_mapping,
)
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
# Canonical curvature-aware SNQI weights/baseline (issue #5580 release gate). The per-episode
# ``metrics.snqi`` field is baked with the campaign-aware scalarizer
# (``robot_sf.benchmark.metrics.snqi``); the release-gate self-check recomputes every stored
# field with this same basis so the per-episode surface and ``snqi_diagnostics.json`` cannot
# drift silently (see ``scripts/validation/check_release_snqi_field_consistency.py``).
_SNQI_DEFAULT_WEIGHTS_NAME = "snqi_weights_camera_ready_v3.json"
_SNQI_DEFAULT_BASELINE_NAME = "snqi_baseline_camera_ready_v3.json"
_SNQI_RECOMPUTE_RTOL = 1e-9
_SNQI_RECOMPUTE_ATOL = 1e-9
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
            for field in ("commit_reconciliation", "goal_timeout_boundary"):
                value = run_meta_payload.get(field)
                if isinstance(value, dict):
                    provenance[field] = value
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


def _compute_and_emit_badging_artifacts(  # noqa: C901
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
        v = val.strip()
        vl = v.lower()
        if any(vl.startswith(prefix) for prefix in ("output/", "file://", "./", "../")):
            return False
        if "localhost" in vl:
            return False
        # Reject placeholder templates
        if "{release_tag}" in v or "<record-id>" in v or "<record_id>" in v:
            return False
        if v.strip() == _DEFAULT_DOI_TEMPLATE:
            return False
        return True

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
    # Caller-supplied functional_smoke_status is metadata only; functional
    # must be earned via actual payload re-derivation (issue #4681 policy).

    # Determine achieved badge level.
    #
    # This slice performs no independent reproduction rerun (issue #4681
    # explicitly scopes out simulation campaigns), so "reproduced" is never
    # granted here from the caller-supplied ``reproduction_status`` alone --
    # emitting it would be an unverified claim, which the badging rubric
    # forbids ("no reproduction claim unless its check passes"). The computed
    # level is therefore capped at "functional" until a dedicated rerun-record
    # check exists. ``reproduction_status`` is still carried through in the
    # block as informational metadata.
    if not is_available:
        achieved_level = "none"
    elif not is_functional:
        achieved_level = "available"
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
    # The checksum manifest lives at the bundle root. Keep every target
    # root-relative so ``sha256sum -c checksums.sha256`` works from that root.
    checksums_payload = "".join(f"{entry.sha256}  payload/{entry.path}\n" for entry in entries)
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


class PublicationPreflightError(Exception):
    """Raised when a publication bundle fails the final self-consistency preflight.

    The preflight (issue #5530) is the final gate before a bundle is treated as
    release-valid. It fails closed: any blocking contradiction in the bundle
    metadata, checksums, or episode provenance raises this rather than letting an
    internally-inconsistent release ship.
    """


def _is_unresolved_channel_value(value: object) -> bool:
    """Return ``True`` when a publication-channel value is still a placeholder.

    Placeholders include the default DOI/release-tag templates, ``localhost`` URIs,
    and local/relative paths that are not durable public identifiers.

    Returns:
        Whether ``value`` must be rejected by the publication preflight.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    v = value.strip()
    vl = v.lower()
    if any(vl.startswith(prefix) for prefix in ("output/", "file://", "./", "../")):
        return True
    if "localhost" in vl:
        return True
    if "{release_tag}" in v or "<record-id>" in v or "<record_id>" in v:
        return True
    if v == _DEFAULT_DOI_TEMPLATE:
        return True
    return False


def _parse_checksum_lines(text: str) -> dict[str, str]:
    """Parse a ``checksums.sha256`` file into ``{relative_path: sha256}``.

    Returns:
        Mapping from bundle-root-relative path to expected lowercase SHA-256 digest.
    """
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"checksums.sha256:{line_number}: malformed checksum entry")
        if any(character not in "0123456789abcdefABCDEF" for character in parts[0]):
            raise ValueError(f"checksums.sha256:{line_number}: malformed checksum entry")
        path = parts[1].lstrip("*")
        if path in entries:
            raise ValueError(f"checksums.sha256:{line_number}: duplicate path {path!r}")
        entries[path] = parts[0].lower()
    if not entries:
        raise ValueError("checksums.sha256 contains no entries")
    return entries


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object from a path.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the file is missing, malformed, or not a JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _episode_row_software_commit(
    record: object,
    *,
    episodes_path: Path,
    line_number: int,
    violations: list[str],
) -> str | None:
    """Validate one decoded episode row and return its software commit.

    Returns:
        The stripped software commit, or ``None`` after recording a violation.
    """
    if not isinstance(record, dict):
        violations.append(f"{episodes_path}:{line_number}: episode row must be an object")
        return None
    ledger = record.get("event_ledger")
    if not isinstance(ledger, dict):
        violations.append(
            f"{episodes_path}:{line_number}: episode row is missing object event_ledger"
        )
        return None
    commit = ledger.get("software_commit")
    if not isinstance(commit, str) or not commit.strip():
        violations.append(
            f"{episodes_path}:{line_number}: event_ledger.software_commit is required"
        )
        return None
    return commit.strip()


def _gather_episode_software_commits(
    payload_dir: Path,
    *,
    violations: list[str],
) -> dict[str, int]:
    """Collect ``event_ledger.software_commit`` counts across all arm episode files.

    Every non-empty episode row must carry a structured event ledger and a
    non-empty software commit. Publication provenance cannot be established by
    silently dropping malformed rows.

    Returns:
        Mapping from software commit hash to number of episode rows that record it.
    """
    commits: dict[str, int] = defaultdict(int)
    episode_paths = sorted(payload_dir.glob("runs/*/episodes.jsonl"))
    if not episode_paths:
        violations.append("publication payload contains no runs/*/episodes.jsonl files")
        return dict(commits)

    for episodes_path in episode_paths:
        try:
            lines = episodes_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            violations.append(f"{episodes_path}: cannot read episode ledger: {exc}")
            continue
        row_count = 0
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                violations.append(f"{episodes_path}:{line_number}: invalid JSON: {exc}")
                continue
            row_count += 1
            commit = _episode_row_software_commit(
                record,
                episodes_path=episodes_path,
                line_number=line_number,
                violations=violations,
            )
            if commit is not None:
                commits[commit] += 1
        if row_count == 0:
            violations.append(f"{episodes_path}: episode ledger contains no non-empty rows")
    return dict(commits)


def _goal_timeout_row_rejection(episodes_path: Path, line_number: int, line: str) -> str | None:
    """Return a rejection message for an ambiguous goal-reached + timeout row.

    A row recording both ``goal_reached`` and ``timeout`` in its exact events is
    ambiguous at the success-timing boundary. It is accepted only when it carries
    timing evidence (a non-null ``reached_goal_step``) or an explicit
    ``goal_timeout_boundary_note``.

    Returns:
        A rejection message, or ``None`` when the row is acceptable.
    """
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    ledger = record.get("event_ledger")
    if not isinstance(ledger, dict):
        return None
    exact = ledger.get("exact_events")
    if not isinstance(exact, dict):
        return None
    if not bool(exact.get("goal_reached")) or not bool(exact.get("timeout")):
        return None
    has_timing = record.get("reached_goal_step") is not None
    note = record.get("goal_timeout_boundary_note")
    if has_timing or (isinstance(note, str) and note.strip()):
        return None
    return (
        f"{episodes_path}:{line_number}: goal_reached+timeout row lacks "
        "reached_goal_step timing evidence or goal_timeout_boundary_note"
    )


def _check_goal_timeout_boundary(payload_dir: Path) -> tuple[int, list[str]]:
    """Find ambiguous goal-reached + timeout rows lacking timing evidence or a note.

    Returns:
        Tuple of (count of ambiguous rows, list of rejection messages).
    """
    ambiguous = 0
    rejections: list[str] = []
    for episodes_path in sorted(payload_dir.glob("runs/*/episodes.jsonl")):
        try:
            lines = episodes_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            message = _goal_timeout_row_rejection(episodes_path, line_number, line)
            if message is not None:
                ambiguous += 1
                rejections.append(message)
    return ambiguous, rejections


def _snqi_planner_key(arm: str) -> str:
    """Return the canonical ``planner_key::kinematics`` token for a per-arm run directory.

    The bundle's per-arm directory is ``<planner>__<kinematics>``; the diagnostics
    ``planner_ordering`` enumerates the same arms as ``planner_key`` / ``kinematics``.
    Mirrors ``scripts/validation/check_release_snqi_field_consistency.py``.
    """
    planner, _, kinematics = arm.partition("__")
    return f"{planner}::{kinematics}"


def _load_snqi_diagnostics(payload_dir: Path) -> dict[str, Any] | None:
    """Return the ``payload/reports/snqi_diagnostics.json`` payload, or ``None`` if absent.

    A bundle without SNQI diagnostics is not an SNQI-bearing release and skips the field
    self-check; a malformed file raises ``ValueError`` so the caller can fail closed.

    Returns:
        The decoded diagnostics object, or ``None`` when the file is absent.
    """
    diagnostics_path = payload_dir / "reports" / "snqi_diagnostics.json"
    if not diagnostics_path.is_file():
        return None
    try:
        payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{diagnostics_path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{diagnostics_path}: expected a JSON object")
    return payload


def _snqi_diagnostics_ordering(diagnostics: Mapping[str, Any]) -> dict[str, int]:
    """Map ``planner_key::kinematics`` -> rank from the diagnostics ``planner_ordering``.

    Returns:
        Mapping of canonical planner key to the diagnostics-declared rank.
    """
    raw_ordering = diagnostics.get("planner_ordering")
    if not isinstance(raw_ordering, list) or not raw_ordering:
        raise ValueError(
            "snqi_diagnostics.json planner_ordering must be a non-empty list for an "
            "SNQI-bearing bundle"
        )
    ordering: dict[str, int] = {}
    for index, row in enumerate(raw_ordering, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"planner_ordering row {index} must be an object")
        planner = row.get("planner_key")
        kinematics = row.get("kinematics")
        if not isinstance(planner, str) or not planner.strip():
            raise ValueError(f"planner_ordering row {index} planner_key must be a non-empty string")
        if not isinstance(kinematics, str) or not kinematics.strip():
            raise ValueError(f"planner_ordering row {index} kinematics must be a non-empty string")
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"planner_ordering row {index} rank must be a positive integer")
        key = f"{planner}::{kinematics}"
        if key in ordering:
            raise ValueError(f"planner_ordering contains duplicate arm {key!r}")
        ordering[key] = rank
    expected_ranks = set(range(1, len(ordering) + 1))
    if set(ordering.values()) != expected_ranks:
        raise ValueError("planner_ordering ranks must be unique and contiguous from 1")
    return ordering


def _check_snqi_field_consistency(  # noqa: C901, PLR0912, PLR0915
    payload_dir: Path,
) -> dict[str, Any]:
    """Release-gate self-check: per-episode SNQI field vs diagnostics basis (issue #5580).

    Runs only when the bundle declares ``payload/reports/snqi_diagnostics.json`` (i.e. an
    SNQI-bearing campaign bundle). It recomputes every stored ``metrics.snqi`` with the same
    curvature-aware scalarizer the field was baked with and asserts the per-episode surface and
    the ``snqi_diagnostics.json`` ``planner_ordering`` cannot drift silently. This is the
    directory-level twin of the archive-level gate
    ``scripts/validation/check_release_snqi_field_consistency.py`` and runs automatically inside
    the publication preflight so a drifted bundle fails closed at build time.

    Returns:
        An evidence dict with ``checked`` (False when SNQI diagnostics are absent),
        ``violation_count``, ``violations`` (rejection messages), ``ordering`` (field vs
        diagnostics planner ordering), ``counts`` (rows/arms), and ``integrity`` (canonical
        weights/baseline sha256). The caller appends ``violations`` to the preflight violation
        list.
    """
    try:
        diagnostics = _load_snqi_diagnostics(payload_dir)
    except ValueError as exc:
        return {"checked": True, "violation_count": 1, "violations": [str(exc)], "ordering": {}}
    if diagnostics is None:
        return {"checked": False, "violation_count": 0, "violations": [], "ordering": {}}

    rejections: list[str] = []
    repo_root = get_repository_root()
    weights_path = repo_root / "configs" / "benchmarks" / _SNQI_DEFAULT_WEIGHTS_NAME
    baseline_path = repo_root / "configs" / "benchmarks" / _SNQI_DEFAULT_BASELINE_NAME
    if not weights_path.is_file():
        rejections.append(f"canonical SNQI weights file is missing: {weights_path}")
    if not baseline_path.is_file():
        rejections.append(f"canonical SNQI baseline file is missing: {baseline_path}")
    if rejections:
        return {
            "checked": True,
            "violation_count": len(rejections),
            "violations": rejections,
            "ordering": {},
        }

    weights_sha256 = _sha256_file(weights_path)
    baseline_sha256 = _sha256_file(baseline_path)
    try:
        weights = _load_snqi_weight_mapping(weights_path)
        baseline = _load_snqi_baseline_mapping(baseline_path)
    except (OSError, TypeError, ValueError) as exc:
        return {
            "checked": True,
            "violation_count": 1,
            "violations": [f"canonical SNQI inputs cannot be loaded: {exc}"],
            "ordering": {},
            "integrity": {
                "snqi_weights_sha256": weights_sha256,
                "snqi_baseline_sha256": baseline_sha256,
            },
        }

    declared_weights_sha = diagnostics.get("weights_sha256")
    declared_baseline_sha = diagnostics.get("baseline_sha256")
    if not isinstance(declared_weights_sha, str) or declared_weights_sha.lower() != weights_sha256:
        rejections.append(
            "snqi_diagnostics.json weights_sha256 does not match the canonical "
            f"{_SNQI_DEFAULT_WEIGHTS_NAME} (missing/invalid or {declared_weights_sha!r} != "
            f"{weights_sha256})"
        )
    if (
        not isinstance(declared_baseline_sha, str)
        or declared_baseline_sha.lower() != baseline_sha256
    ):
        rejections.append(
            "snqi_diagnostics.json baseline_sha256 does not match the canonical "
            f"{_SNQI_DEFAULT_BASELINE_NAME} (missing/invalid or {declared_baseline_sha!r} != "
            f"{baseline_sha256})"
        )

    try:
        diagnostics_ordering = _snqi_diagnostics_ordering(diagnostics)
    except (TypeError, ValueError) as exc:
        return {
            "checked": True,
            "violation_count": len(rejections) + 1,
            "violations": [*rejections, f"invalid SNQI diagnostics ordering: {exc}"],
            "ordering": {},
            "integrity": {
                "snqi_weights_sha256": weights_sha256,
                "snqi_baseline_sha256": baseline_sha256,
            },
        }

    rows = 0
    episode_field_present = 0
    per_arm_field_sum: dict[str, float] = defaultdict(float)
    per_arm_field_count: dict[str, int] = defaultdict(int)
    mismatch_sample: list[str] = []
    mismatch_total = 0
    for episodes_path in sorted(payload_dir.glob("runs/*/episodes.jsonl")):
        arm = episodes_path.parent.name
        try:
            lines = episodes_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            rejections.append(f"cannot read {episodes_path}: {exc}")
            continue
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            rows += 1
            source = f"{episodes_path}:{line_number}"
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                rejections.append(f"{source}: invalid JSON: {exc}")
                continue
            if not isinstance(record, Mapping):
                rejections.append(f"{source}: expected a JSON object")
                continue
            metrics = record.get("metrics")
            if not isinstance(metrics, Mapping):
                rejections.append(f"{source}: metrics must be an object")
                continue
            if "snqi" not in metrics:
                rejections.append(f"{source}: metrics.snqi field is absent")
                continue
            episode_field_present += 1
            raw_stored_snqi = metrics["snqi"]
            if isinstance(raw_stored_snqi, bool) or not isinstance(raw_stored_snqi, (int, float)):
                rejections.append(f"{source}: metrics.snqi must be a finite JSON number")
                continue
            try:
                stored_snqi = float(raw_stored_snqi)
                if not math.isfinite(stored_snqi):
                    raise ValueError("value is not finite")
            except (TypeError, ValueError) as exc:
                rejections.append(f"{source}: metrics.snqi is not a finite number: {exc}")
                continue
            metric_values = dict(metrics)
            try:
                recomputed_snqi = _curvature_aware_snqi(
                    metric_values, weights, baseline_stats=baseline
                )
                if not math.isfinite(recomputed_snqi):
                    raise ValueError("value is not finite")
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                rejections.append(f"{source}: curvature-aware recompute failed: {exc}")
                continue
            per_arm_field_sum[arm] += stored_snqi
            per_arm_field_count[arm] += 1
            if not math.isclose(
                stored_snqi,
                recomputed_snqi,
                rel_tol=_SNQI_RECOMPUTE_RTOL,
                abs_tol=_SNQI_RECOMPUTE_ATOL,
            ):
                # Every drift is counted; the per-episode rejection sample is capped so a
                # large drift cannot balloon the report, but ``violation_count`` below
                # accounts for every drifted episode so the evidence is not misleading.
                mismatch_total += 1
                if len(mismatch_sample) < 20:
                    mismatch_sample.append(
                        f"{source}: stored SNQI {stored_snqi!r} != curvature-aware recompute "
                        f"{recomputed_snqi!r}"
                    )

    rejections.extend(mismatch_sample)
    truncated_mismatches = max(mismatch_total - len(mismatch_sample), 0)
    field_means = {
        arm: (per_arm_field_sum[arm] / per_arm_field_count[arm])
        if per_arm_field_count[arm]
        else 0.0
        for arm in per_arm_field_sum
    }
    field_ranked = sorted(field_means, key=lambda a: (-field_means[a], a))
    field_ordering = {_snqi_planner_key(a): rank for rank, a in enumerate(field_ranked, start=1)}
    if field_ordering != diagnostics_ordering:
        rejections.append(
            "per-episode metrics.snqi arm ordering disagrees with "
            "snqi_diagnostics.json planner_ordering"
        )
    if episode_field_present == 0 and rows > 0:
        rejections.append(
            "no per-episode metrics.snqi fields found despite declaring snqi_diagnostics.json"
        )

    return {
        "checked": True,
        "violation_count": len(rejections) + truncated_mismatches,
        "violations": rejections,
        "ordering": {
            "field_planner_ordering": field_ordering,
            "diagnostics_planner_ordering": diagnostics_ordering,
        },
        "counts": {
            "rows": rows,
            "episode_field_present": episode_field_present,
            "snqi_field_mismatches": mismatch_total,
            "arms": len(per_arm_field_count),
        },
        "integrity": {
            "snqi_weights_sha256": weights_sha256,
            "snqi_baseline_sha256": baseline_sha256,
        },
    }


def _manifest_checksum_mapping(
    manifest_files: list[object],
    *,
    violations: list[str],
) -> dict[str, str]:
    """Return normalized manifest file digests while reporting malformed entries."""
    manifest_checksums: dict[str, str] = {}
    for entry in manifest_files:
        if not isinstance(entry, dict):
            violations.append("publication manifest files must contain objects")
            continue
        path_value = entry.get("path")
        digest_value = entry.get("sha256")
        if not isinstance(path_value, str) or not path_value.strip():
            violations.append("publication manifest file entry path must be a non-empty string")
            continue
        if (
            not isinstance(digest_value, str)
            or len(digest_value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest_value)
        ):
            violations.append(f"publication manifest sha256 is malformed for file {path_value!r}")
            continue
        normalized_path = path_value.removeprefix("payload/")
        manifest_path = f"payload/{normalized_path}"
        if manifest_path in manifest_checksums:
            violations.append(f"duplicate publication manifest file: {path_value!r}")
            continue
        manifest_checksums[manifest_path] = digest_value.lower()
    return manifest_checksums


def _preflight_check_checksums(
    bundle_dir: Path,
    manifest: dict[str, Any],
    *,
    checksums_path: Path,
    violations: list[str],
) -> dict[str, str]:
    """Verify every checksum entry against a file present relative to the bundle root.

    Returns:
        The parsed ``{relative_path: sha256}`` checksum mapping.
    """
    checksums = _parse_checksum_lines(checksums_path.read_text(encoding="utf-8"))
    for rel_path, expected in checksums.items():
        candidate = bundle_dir / rel_path
        if ".." in Path(rel_path).parts or Path(rel_path).is_absolute():
            violations.append(f"checksum path escapes bundle root: {rel_path}")
            continue
        if not candidate.is_file():
            violations.append(f"checksum-signed file is missing from bundle root: {rel_path}")
            continue
        if _sha256_file(candidate) != expected:
            violations.append(f"checksum mismatch at bundle root: {rel_path}")

    manifest_files = manifest.get("files")
    if isinstance(manifest_files, list):
        # The manifest lists file paths without the ``payload/`` prefix while
        # checksums.sha256 is always written relative to the bundle root
        # (``payload/...``); normalize both sides before comparing.
        manifest_checksums = _manifest_checksum_mapping(manifest_files, violations=violations)
        manifest_paths = {path.removeprefix("payload/") for path in manifest_checksums}
        normalized_checksums = {key.removeprefix("payload/") for key in checksums}
        for rel_path in sorted(normalized_checksums - manifest_paths):
            violations.append(f"checksum entry not listed in manifest files: {rel_path}")
        for rel_path in sorted(manifest_paths - normalized_checksums):
            violations.append(f"manifest file not present in checksums.sha256: {rel_path}")
        for path in sorted(set(manifest_checksums) & set(checksums)):
            if manifest_checksums[path] != checksums[path]:
                violations.append(f"manifest sha256 disagrees with checksums.sha256: {path}")
    else:
        violations.append("publication_manifest.json files must be a list")
    return checksums


def _preflight_check_channels(
    manifest: dict[str, Any],
    *,
    violations: list[str],
    warnings: list[str],
) -> None:
    """Reject unresolved DOI/release-tag placeholders in publication_channels."""
    channels = manifest.get("publication_channels")
    if isinstance(channels, dict):
        for key, value in channels.items():
            if _is_unresolved_channel_value(value):
                violations.append(
                    f"publication_channels.{key} retains an unresolved placeholder: {value!r}"
                )
    elif "publication_channels" not in manifest:
        warnings.append("publication_manifest.json omits publication_channels")


def _preflight_check_release_reconciliation(
    payload_dir: Path,
    require_release_reconciliation: bool,
    *,
    violations: list[str],
) -> None:
    """Fail closed when release_result.json disagrees with campaign_summary.json."""
    release_result_path = payload_dir / "release" / "release_result.json"
    campaign_summary_path = payload_dir / "reports" / "campaign_summary.json"
    if release_result_path.is_file() and campaign_summary_path.is_file():
        release_result = _read_json_file(release_result_path)
        campaign_summary = _read_json_file(campaign_summary_path)
        campaign = campaign_summary.get("campaign")
        if not isinstance(campaign, dict):
            violations.append(
                "reports/campaign_summary.json campaign block is missing or not an object"
            )
            campaign = {}
        comparisons = (
            ("status", release_result.get("status"), campaign.get("status")),
            (
                "evidence_status",
                release_result.get("evidence_status"),
                campaign.get("evidence_status"),
            ),
            (
                "total_episodes",
                release_result.get("total_episodes"),
                campaign.get("total_episodes"),
            ),
            (
                "successful_runs",
                release_result.get("successful_runs"),
                campaign.get("successful_runs"),
            ),
        )
        mismatched = [name for name, old, rebuilt in comparisons if old != rebuilt]
        if mismatched:
            violations.append(
                "release/release_result.json disagrees with reports/campaign_summary.json on: "
                + ", ".join(mismatched)
            )
    elif require_release_reconciliation:
        missing = [
            str(path.relative_to(payload_dir.parent))
            for path in (release_result_path, campaign_summary_path)
            if not path.is_file()
        ]
        violations.append(
            "release reconciliation inputs missing from bundle: " + ", ".join(missing)
        )


def _preflight_check_commit_provenance(
    manifest: dict[str, Any],
    payload_dir: Path,
    *,
    violations: list[str],
    warnings: list[str],
) -> tuple[str | None, dict[str, int]]:
    """Verify episode software_commit values match the publication repository commit.

    Returns:
        Tuple of (publication repository commit, episode-commit counts).
    """
    provenance = manifest.get("provenance")
    repository_commit: str | None = None
    if isinstance(provenance, dict):
        repository = provenance.get("repository")
        if isinstance(repository, dict):
            commit = repository.get("commit")
            repository_commit = commit if isinstance(commit, str) and commit else None
    if not repository_commit:
        violations.append("publication_manifest provenance.repository.commit is missing")

    episode_commits = _gather_episode_software_commits(payload_dir, violations=violations)
    if episode_commits and repository_commit:
        if set(episode_commits) != {repository_commit}:
            explanation = provenance.get("commit_reconciliation")
            runtime_commits = set(episode_commits)
            declared_runtime_commits = (
                explanation.get("runtime_commits") if isinstance(explanation, dict) else None
            )
            declared_runtime_commits_valid = isinstance(declared_runtime_commits, list) and all(
                isinstance(commit, str) and commit.strip() for commit in declared_runtime_commits
            )
            if (
                isinstance(explanation, dict)
                and explanation.get("status") == "explained"
                and explanation.get("publication_commit") == repository_commit
                and declared_runtime_commits_valid
                and set(declared_runtime_commits) == runtime_commits
                and isinstance(explanation.get("explanation"), str)
                and explanation["explanation"].strip()
            ):
                warnings.append(
                    "episode-ledger software commits differ from the publication commit; "
                    "allowed via structured provenance.commit_reconciliation"
                )
            else:
                violations.append(
                    "episode-ledger software_commit values "
                    f"{sorted(episode_commits)} do not match the publication commit "
                    f"{repository_commit!r}; provide structured provenance.commit_reconciliation"
                )
    return repository_commit, episode_commits


def verify_publication_bundle_preflight(
    bundle_dir: Path,
    *,
    require_release_reconciliation: bool = True,
) -> dict[str, Any]:
    """Run the final publication preflight over a built bundle directory.

    The preflight (issue #5530) reconciles the bundle against its own metadata and
    fails closed on any internal contradiction:

    1. ``release/release_result.json`` and ``reports/campaign_summary.json`` must
       agree on status, evidence_status, total_episodes, and successful_runs.
    2. every checksum in ``checksums.sha256`` must verify against a file present
       relative to the bundle root (so ``sha256sum -c checksums.sha256`` works
       from the bundle root), and every manifest-listed file must be signed.
    3. episode-ledger ``software_commit`` values must equal the publication
       manifest's repository commit, unless the manifest carries an explicit
       machine-readable non-runtime-diff explanation.
    4. ``publication_channels`` must not retain DOI/release-tag placeholders.
    5. ambiguous goal-reached + timeout rows must carry timing evidence or a note.
    6. for SNQI-bearing bundles (those declaring ``snqi_diagnostics.json``), every per-episode
       ``metrics.snqi`` field must match a curvature-aware recomputation and the field-derived
       arm ordering must agree with ``snqi_diagnostics.json`` ``planner_ordering`` (issue #5580).

    Args:
        bundle_dir: Built bundle directory (containing ``payload/``,
            ``publication_manifest.json``, and ``checksums.sha256``).
        require_release_reconciliation: When ``True`` (default, used by the
            release builders), the absence of the release-result/campaign-summary
            pair is a blocking failure; when ``False``, the reconciliation check is
            skipped if those files are absent.

    Returns:
        A machine-readable preflight report dict.

    Raises:
        PublicationPreflightError: If any blocking check fails.
    """
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "publication_manifest.json"
    checksums_path = bundle_dir / "checksums.sha256"
    payload_dir = bundle_dir / "payload"

    violations: list[str] = []
    warnings: list[str] = []

    if not manifest_path.is_file():
        raise PublicationPreflightError(f"Publication preflight failed: missing {manifest_path}")
    if not checksums_path.is_file():
        raise PublicationPreflightError(f"Publication preflight failed: missing {checksums_path}")

    try:
        manifest = _read_json_file(manifest_path)
    except ValueError as exc:
        raise PublicationPreflightError(f"Publication preflight failed: {exc}") from exc
    if not manifest.get("schema_version"):
        violations.append("publication_manifest.json is missing schema_version")

    try:
        checksums = _preflight_check_checksums(
            bundle_dir, manifest, checksums_path=checksums_path, violations=violations
        )
    except (OSError, ValueError) as exc:
        violations.append(f"checksums.sha256 cannot be validated: {exc}")
        checksums = {}
    _preflight_check_channels(manifest, violations=violations, warnings=warnings)
    try:
        _preflight_check_release_reconciliation(
            payload_dir,
            require_release_reconciliation,
            violations=violations,
        )
    except (OSError, ValueError) as exc:
        violations.append(f"release reconciliation cannot be validated: {exc}")
    repository_commit, episode_commits = _preflight_check_commit_provenance(
        manifest, payload_dir, violations=violations, warnings=warnings
    )

    # ---- Check 5: ambiguous goal-reached + timeout rows ---------------------
    goal_timeout_rows, goal_timeout_rejections = _check_goal_timeout_boundary(payload_dir)
    violations.extend(goal_timeout_rejections)

    # ---- Check 6: per-episode SNQI field vs diagnostics basis (issue #5580) --
    # Runs only on SNQI-bearing bundles (those declaring snqi_diagnostics.json); other
    # bundles report checked=False and are unaffected.
    snqi_evidence = _check_snqi_field_consistency(payload_dir)
    violations.extend(snqi_evidence.get("violations", []))

    status = "pass" if not violations else "fail"
    report = {
        "schema_version": "publication-preflight.v1",
        "bundle_dir": str(bundle_dir),
        "status": status,
        "violation_count": len(violations),
        "violations": violations,
        "warnings": warnings,
        "evidence": {
            "signed_files": len(checksums),
            "episode_software_commits": dict(sorted(episode_commits.items())),
            "publication_commit": repository_commit,
            "goal_reached_timeout_rows": goal_timeout_rows,
            "snqi_field_consistency": snqi_evidence,
        },
    }
    if status == "fail":
        raise PublicationPreflightError("Publication preflight failed: " + "; ".join(violations))
    return report
