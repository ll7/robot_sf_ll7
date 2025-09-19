"""Persistence helpers (episodes append, manifest serialization).

Implemented in tasks T025 (append, write_manifest integration) and updated later.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_episode_record(path, record):  # T025
    """Append a single episode record as JSON line.

    Guarantees line-oriented append; caller can fsync externally if needed. Creates
    parent directories on first write. Record is assumed JSON serializable.
    """
    p = Path(path)
    _ensure_parent(p)
    line = json.dumps(record, separators=(",", ":"))
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.debug("Appended episode record to {}", p)


def write_manifest(manifest, path):  # T025
    """Atomically write manifest JSON file.

    Uses a temp file + rename pattern for atomic replace on POSIX systems.
    Serializes dataclass-like objects by falling back to attribute dict.
    """
    p = Path(path)
    _ensure_parent(p)

    def _to_obj(obj: Any):  # small serializer helper
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: _to_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_obj(v) for v in obj]
        # Fallback: object with __dict__ (e.g., dataclass instance)
        if hasattr(obj, "__dict__"):
            return {k: _to_obj(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)

    data = _to_obj(manifest)
    # Ensure required top-level keys exist (light validation for contract)
    for key in ("git_hash", "scenario_matrix_hash", "config"):
        if key not in data:  # pragma: no cover - defensive
            raise ValueError(f"Manifest missing required key: {key}")

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=p.name, dir=str(p.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
            json.dump(data, tmp_f, indent=2, sort_keys=True)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        os.replace(tmp_path, p)  # atomic on POSIX
        logger.debug("Wrote manifest atomically to {}", p)
    finally:
        if os.path.exists(tmp_path):  # cleanup if exception before replace
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover - best effort
                pass
