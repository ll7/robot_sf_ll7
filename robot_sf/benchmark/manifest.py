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

Sidecar schema (v1)
```
{
    "version": 1,
    "out_file": "episodes.jsonl",
    "stat": {"size": <int>, "mtime_ns": <int>},
    "episode_ids": [<str>, ...]
}
```

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
from pathlib import Path
from typing import Iterable, Optional, Set


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


def load_manifest(
    out_path: Path, expected_identity_hash: Optional[str] = None
) -> Optional[Set[str]]:
    """Return cached episode_ids if sidecar matches current file, else None.

    The sidecar is considered valid only if both ``size`` and ``mtime_ns`` in
    the manifest match the JSONL's current stat. Any decoding or validation
    failure results in ``None`` to allow a robust fallback to JSONL scanning.
    """
    sidecar = manifest_path_for(out_path)
    if not sidecar.exists() or not out_path.exists():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        stat = data.get("stat")
        if not isinstance(stat, dict):
            return None
        have = _stat_of(out_path)
        if int(stat.get("size", -1)) != have.size:
            return None
        if int(stat.get("mtime_ns", -1)) != have.mtime_ns:
            return None
        # If caller provided an expected identity hash, require it to match.
        if expected_identity_hash is not None:
            stored_hash = data.get("identity_hash")
            if stored_hash != expected_identity_hash:
                return None
        ids = data.get("episode_ids", [])
        if not isinstance(ids, list):
            return None
        return {x for x in ids if isinstance(x, str)}
    except Exception:
        return None


def save_manifest(
    out_path: Path, episode_ids: Iterable[str], identity_hash: Optional[str] = None
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
    rec = {
        "version": 1,
        "out_file": out_path.name,
        "stat": {"size": have.size, "mtime_ns": have.mtime_ns},
        "episode_ids": sorted(set(episode_ids)),
    }
    if identity_hash is not None:
        rec["identity_hash"] = identity_hash
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
