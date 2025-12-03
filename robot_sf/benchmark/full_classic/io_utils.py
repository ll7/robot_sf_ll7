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
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _serialize_obj(obj: Any):  # separated to keep write_manifest simple
    """TODO docstring. Document this function.

    Args:
        obj: TODO docstring.
    """
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [_serialize_obj(v) for v in obj]
    if hasattr(obj, "__dict__"):
        items = obj.__dict__.items()
        if not items:  # gather class attributes if instance dict empty
            attrs = {
                k: getattr(obj, k)
                for k in dir(obj)
                if not k.startswith("_") and not callable(getattr(obj, k))
            }
            return {k: _serialize_obj(v) for k, v in attrs.items()}
        return {k: _serialize_obj(v) for k, v in items if not k.startswith("_")}
    return str(obj)


def _atomic_write_json(path: Path, data: dict) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        data: TODO docstring.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
            json.dump(data, tmp_f, indent=2, sort_keys=True)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        os.replace(tmp_path, path)
    finally:  # cleanup tmp if failure prior to replace
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover
                pass


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
    """Atomically serialize manifest object to JSON.

    Complexity kept low by delegating to helpers.
    """
    p = Path(path)
    _ensure_parent(p)
    data = _serialize_obj(manifest)
    for key in ("git_hash", "scenario_matrix_hash", "config"):
        if key not in data:
            raise ValueError(f"Manifest missing required key: {key}")
    _atomic_write_json(p, data)
    logger.debug("Manifest written: {}", p)
