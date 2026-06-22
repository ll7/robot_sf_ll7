"""Persistence helpers (episodes append, manifest serialization).

Implemented in tasks T025 (append, write_manifest integration) and updated later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.common.atomic_io import atomic_write_json


def _ensure_parent(path: Path) -> None:
    """Create parent directories for a target path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _serialize_obj(obj: Any):  # separated to keep write_manifest simple
    """Convert complex objects into JSON-serializable structures.

    Returns:
        JSON-serializable representation of the object.
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
    atomic_write_json(p, data)
    logger.debug("Manifest written: {}", p)
