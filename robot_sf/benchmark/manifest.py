"""Manifest sidecar for fast resume of benchmark JSONL outputs.

This module provides a tiny sidecar file (``.manifest.json``) next to a
benchmark JSONL output (e.g., ``episodes.jsonl``). The manifest records:

- a minimal file stat of the JSONL (size, mtime_ns) to detect changes,
- the set of ``episode_id`` strings already present in the file,
- a version tag for forward compatibility, and
- the output file name (basename) for readability.

Usage pattern
1) On resume, try ``load_manifest(out_path, expected_identity_hash)``. If it returns a set of ids,
     use it to skip already-completed jobs. If it returns ``None``, fall back
     to scanning the JSONL for ``episode_id`` values.
2) After writing new episodes, call ``save_manifest(out_path, ids, identity_hash)`` to update
     the sidecar. The writer re-reads ids from disk prior to saving to ensure
     the manifest precisely matches what is on disk.

Sidecar schema (v2 augmentation)
```
{
    "version": 2,
    "out_file": "episodes.jsonl",
    "stat": {"size": <int>, "mtime_ns": <int>},
    "episode_ids": [<str>, ...],
    "episodes_count": <int>,             # cached len(episode_ids)
    "schema_version": "v1",             # episode schema version (from constants)
    "identity_hash": "<short-hash>"     # optional fingerprint of id function
}
```

Invalidation rules (research.md §7):
- Size mismatch OR mtime_ns mismatch → invalidate (return None to force scan).
- episodes_count mismatch with len(episode_ids) (corruption) → invalidate.
- schema_version mismatch with expected (caller provided) → invalidate.
- identity_hash mismatch with expected_identity_hash (if provided) → invalidate.

Notes
- If the JSONL file changes (size or mtime_ns differ), ``load_manifest``
    returns ``None`` to force a scan fallback.
- If a caller passes ``expected_identity_hash`` and it differs from the stored
    ``identity_hash``, ``load_manifest`` returns ``None`` so stale manifests
    computed with older episode identity definitions are ignored.
- The implementation is intentionally minimal and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.constants import EPISODE_SCHEMA_VERSION
from robot_sf.benchmark.identity.hash_utils import sha256_file

if TYPE_CHECKING:
    from collections.abc import Iterable


SIMULATION_RUN_PROVENANCE_SCHEMA_VERSION = "simulation_run_provenance.v1"
_REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_status",
        "inputs",
        "outputs",
        "generated_reports",
        "stable_identifiers",
        "optional_fields",
    }
)
_EXPLICIT_OPTIONAL_FIELDS = ("run_id", "invocation", "config_path", "scenario_path")


@dataclass(frozen=True)
class _Stat:
    """Lightweight file stat snapshot."""

    size: int
    mtime_ns: int


def _stat_of(path: Path) -> _Stat:
    """Return size and mtime_ns for a path."""
    st = path.stat()
    return _Stat(size=int(st.st_size), mtime_ns=int(st.st_mtime_ns))


def _sha256_jsonable(payload: object) -> str:
    """Return a stable SHA256 digest for JSON-serializable provenance identity."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_entry(path: Path, *, required: bool = False) -> dict[str, object]:
    """Return the minimum machine-readable artifact identity for a path."""
    if not path.exists():
        if required:
            raise ValueError(f"Required provenance artifact does not exist: {path}")
        return {
            "path": str(path),
            "artifact_status": "missing",
            "sha256": None,
            "size": None,
            "mtime_ns": None,
        }
    if not path.is_file():
        if required:
            raise ValueError(f"Required provenance artifact is not a file: {path}")
        return {
            "path": str(path),
            "artifact_status": "not_file",
            "sha256": None,
            "size": None,
            "mtime_ns": None,
        }
    stat = _stat_of(path)
    return {
        "path": str(path),
        "artifact_status": "available",
        "sha256": sha256_file(path),
        "size": stat.size,
        "mtime_ns": stat.mtime_ns,
    }


def _artifact_entries(paths: Iterable[Path] | None) -> list[dict[str, object]]:
    """Return artifact entries for provided paths, preserving explicit absence as an empty list."""
    if paths is None:
        return []
    return [_artifact_entry(Path(path)) for path in paths]


def _build_simulation_run_provenance(
    *,
    out_path: Path,
    episode_ids: list[str],
    identity_hash: str | None,
    schema_version: str,
    input_paths: Iterable[Path] | None,
    report_paths: Iterable[Path] | None,
) -> dict[str, object]:
    """Build and validate the minimum simulation-run provenance bundle.

    Returns:
        Machine-readable provenance bundle for the simulation output sidecar.
    """
    bundle: dict[str, object] = {
        "schema_version": SIMULATION_RUN_PROVENANCE_SCHEMA_VERSION,
        "bundle_status": "complete",
        "inputs": _artifact_entries(input_paths),
        "outputs": [_artifact_entry(out_path, required=True)],
        "generated_reports": _artifact_entries(report_paths),
        "stable_identifiers": {
            "episode_ids_sha256": _sha256_jsonable(sorted(set(episode_ids))),
            "identity_hash": identity_hash,
            "schema_version": schema_version,
        },
        "optional_fields": dict.fromkeys(_EXPLICIT_OPTIONAL_FIELDS),
    }
    _validate_simulation_run_provenance(bundle)
    return bundle


def _validate_provenance_artifact_entry(collection_name: str, artifact: object) -> None:
    """Validate one provenance artifact entry."""
    if not isinstance(artifact, dict) or not artifact.get("path"):
        raise ValueError(f"Simulation-run provenance {collection_name} entries require path")
    status = artifact.get("artifact_status", "available")
    if status == "available" and not artifact.get("sha256"):
        raise ValueError(
            f"Simulation-run provenance {collection_name} available entries require sha256"
        )
    if status not in {"available", "missing", "not_file"}:
        raise ValueError(f"Simulation-run provenance {collection_name} artifact_status is invalid")


def _validate_simulation_run_provenance(bundle: dict[str, object]) -> None:
    """Fail closed when the required provenance bundle contract is incomplete."""
    missing = sorted(_REQUIRED_PROVENANCE_FIELDS - bundle.keys())
    if missing:
        raise ValueError(f"Simulation-run provenance missing required fields: {missing}")
    if bundle.get("schema_version") != SIMULATION_RUN_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Simulation-run provenance schema_version is invalid")
    optional_fields = bundle.get("optional_fields")
    if not isinstance(optional_fields, dict):
        raise ValueError("Simulation-run provenance optional_fields must be a mapping")
    missing_optionals = [
        field for field in _EXPLICIT_OPTIONAL_FIELDS if field not in optional_fields
    ]
    if missing_optionals:
        raise ValueError(
            "Simulation-run provenance optional_fields missing explicit fields: "
            f"{missing_optionals}"
        )
    for collection_name in ("inputs", "outputs", "generated_reports"):
        collection = bundle.get(collection_name)
        if not isinstance(collection, list):
            raise ValueError(f"Simulation-run provenance {collection_name} must be a list")
        for artifact in collection:
            _validate_provenance_artifact_entry(collection_name, artifact)


def manifest_path_for(out_path: Path) -> Path:
    """Return the manifest sidecar path for a JSONL output path.

    The sidecar naming pattern appends ``.manifest.json`` to the original
    suffix, e.g., ``episodes.jsonl`` → ``episodes.jsonl.manifest.json``.

    Returns:
        Path to the manifest sidecar file.
    """
    return out_path.with_suffix(out_path.suffix + ".manifest.json")


def _validate_manifest_data(
    data: dict,
    out_path: Path,
    expected_identity_hash: str | None,
    expected_schema_version: str,
) -> set[str] | None:
    """Return set of ids if manifest data passes all validation checks else None.

    Returns:
        Set of episode IDs if validation succeeds, None otherwise.
    """
    stat = data.get("stat")
    if not isinstance(stat, dict):
        return None
    have = _stat_of(out_path)
    if int(stat.get("size", -1)) != have.size or int(stat.get("mtime_ns", -1)) != have.mtime_ns:
        return None
    if expected_identity_hash is not None and data.get("identity_hash") != expected_identity_hash:
        return None
    stored_schema_version = data.get("schema_version")
    if stored_schema_version is not None and stored_schema_version != expected_schema_version:
        return None
    ids = data.get("episode_ids", [])
    if not isinstance(ids, list):
        return None
    filtered = [x for x in ids if isinstance(x, str)]
    episodes_count = data.get("episodes_count")
    if episodes_count is not None and int(episodes_count) != len(filtered):
        return None
    return set(filtered)


def load_manifest(
    out_path: Path,
    expected_identity_hash: str | None = None,
    expected_schema_version: str = EPISODE_SCHEMA_VERSION,
) -> set[str] | None:
    """Return cached episode_ids if sidecar matches current file, else None.

    Validation criteria (v2):
    - sidecar + target file must exist
    - stat.size & stat.mtime_ns must match current file
    - if expected_identity_hash provided, must match stored identity_hash (if present)
    - if schema_version present, must equal expected_schema_version
    - episodes_count (if present) must equal len(episode_ids)
    Any failure returns None to trigger fallback scanning.

    Returns:
        Set of cached episode IDs if validation succeeds, None otherwise.
    """
    sidecar = manifest_path_for(out_path)
    if not sidecar.exists() or not out_path.exists():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None

    if not isinstance(data, dict):
        return None
    return _validate_manifest_data(data, out_path, expected_identity_hash, expected_schema_version)


def save_manifest(
    out_path: Path,
    episode_ids: Iterable[str],
    identity_hash: str | None = None,
    schema_version: str = EPISODE_SCHEMA_VERSION,
    *,
    input_paths: Iterable[Path] | None = None,
    report_paths: Iterable[Path] | None = None,
) -> None:
    """Write or update the manifest to reflect the current on-disk state.

    Args:
        out_path: Path to the JSONL episodes file managed by the manifest.
        episode_ids: All episode ids currently present in the JSONL file.
        identity_hash: Optional content hash written for tamper detection.
        schema_version: Episode schema version recorded in the manifest.
        input_paths: Optional simulation inputs to checksum into the provenance bundle.
        report_paths: Optional generated reports to checksum into the provenance bundle.

    Notes:
        - If the JSONL file does not exist, the function returns without writing.
        - The sidecar captures the target file's stat for change detection.
        - ``simulation_run_provenance`` records explicit ``None`` values for optional
          run metadata that this entry point cannot infer.
    """
    if not out_path.exists():
        return
    sidecar = manifest_path_for(out_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    have = _stat_of(out_path)
    ids_sorted = sorted(set(episode_ids))
    rec = {
        "version": 2,
        "out_file": out_path.name,
        "stat": {"size": have.size, "mtime_ns": have.mtime_ns},
        "episode_ids": ids_sorted,
        "episodes_count": len(ids_sorted),
        "schema_version": schema_version,
        "simulation_run_provenance": _build_simulation_run_provenance(
            out_path=out_path,
            episode_ids=ids_sorted,
            identity_hash=identity_hash,
            schema_version=schema_version,
            input_paths=input_paths,
            report_paths=report_paths,
        ),
    }
    if identity_hash is not None:
        rec["identity_hash"] = identity_hash
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
