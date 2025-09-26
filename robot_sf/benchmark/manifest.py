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

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from robot_sf.benchmark.constants import EPISODE_SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True)
class _Stat:
    size: int
    mtime_ns: int


def _stat_of(path: Path) -> _Stat:
    st = path.stat()
    return _Stat(size=int(st.st_size), mtime_ns=int(st.st_mtime_ns))


def manifest_path_for(out_path: Path) -> Path:
    """Return the manifest sidecar path for a JSONL output path.

    The sidecar naming pattern appends ``.manifest.json`` to the original
    suffix, e.g., ``episodes.jsonl`` → ``episodes.jsonl.manifest.json``.
    """
    return out_path.with_suffix(out_path.suffix + ".manifest.json")


def _validate_manifest_data(
    data: dict,
    out_path: Path,
    expected_identity_hash: str | None,
    expected_schema_version: str,
) -> set[str] | None:
    """Return set of ids if manifest data passes all validation checks else None."""
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
    """
    sidecar = manifest_path_for(out_path)
    if not sidecar.exists() or not out_path.exists():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    return _validate_manifest_data(data, out_path, expected_identity_hash, expected_schema_version)


def save_manifest(
    out_path: Path,
    episode_ids: Iterable[str],
    identity_hash: str | None = None,
    schema_version: str = EPISODE_SCHEMA_VERSION,
) -> None:
    """Write or update the manifest to reflect the current on-disk state.

    Parameters
    - out_path: Path to the JSONL episodes file.
    - episode_ids: Iterable of episode_id strings present in the file.

    Behavior
    - If the JSONL file does not exist, the function returns without writing.
    - The sidecar's stat is captured from the JSONL at save time to bind the
      manifest to the exact file content.
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
    }
    if identity_hash is not None:
        rec["identity_hash"] = identity_hash
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
