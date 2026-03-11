"""Helpers for benchmark artifact publication and DOI-ready export bundles.

This module provides reusable logic for:
- discovering benchmark run directories,
- measuring artifact size distributions across runs,
- exporting a publication bundle with checksums and provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from robot_sf.common.artifact_paths import get_repository_root

PUBLICATION_BUNDLE_SCHEMA_VERSION = "benchmark-publication-bundle.v2"
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


def export_publication_bundle(
    run_dir: Path,
    out_dir: Path,
    *,
    bundle_name: str | None = None,
    include_videos: bool = True,
    repository_url: str = _DEFAULT_REPO_URL,
    release_tag: str = _DEFAULT_RELEASE_TAG_TEMPLATE,
    doi: str = _DEFAULT_DOI_TEMPLATE,
    overwrite: bool = False,
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
