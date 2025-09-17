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
    """Return the manifest path for a given JSONL output path.

    Uses a sidecar file: episodes.jsonl.manifest.json
    """
    return out_path.with_suffix(out_path.suffix + ".manifest.json")


def load_manifest(out_path: Path) -> Optional[Set[str]]:
    """Load manifest if it matches current out_path file stat; else None.

    Manifest JSON schema (v1):
    {
      "version": 1,
      "out_file": "episodes.jsonl",  # basename for humans
      "stat": {"size": int, "mtime_ns": int},
      "episode_ids": ["..."]
    }
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
        ids = data.get("episode_ids", [])
        if not isinstance(ids, list):
            return None
        return {x for x in ids if isinstance(x, str)}
    except Exception:
        return None


def save_manifest(out_path: Path, episode_ids: Iterable[str]) -> None:
    """Write manifest sidecar reflecting current out_path stat and ids."""
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
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
