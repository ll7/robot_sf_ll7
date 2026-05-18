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
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc
                return payload if isinstance(payload, dict) else None
    return None
