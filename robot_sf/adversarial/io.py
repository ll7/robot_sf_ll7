"""I/O helpers for adversarial search artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_first_jsonl_record(path: Path | None) -> dict[str, Any] | None:
    """Read the first non-empty JSONL object without loading the whole file."""
    if path is None or not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                return payload if isinstance(payload, dict) else None
    return None
