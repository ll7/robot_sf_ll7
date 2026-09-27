"""I/O helpers for adversarial search artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_first_jsonl_record(path: Path | None) -> dict[str, Any] | None:
    """Read the first non-empty JSONL object without loading the whole file."""
    if path is None or not path.exists():
        return None
    return parse_first_jsonl_record(path.read_bytes(), source=path.as_posix())


def parse_first_jsonl_record(
    data: bytes, *, source: str = "episode JSONL"
) -> dict[str, Any] | None:
    """Parse the first non-empty JSONL object from the supplied immutable bytes.

    Returns:
        First JSON object, or ``None`` for an empty file or non-object first row.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: episode JSONL is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}: invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            return payload if isinstance(payload, dict) else None
    return None
