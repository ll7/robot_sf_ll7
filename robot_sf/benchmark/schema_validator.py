"""Helper utilities to validate episode metric records against JSON schema.

Usage:
    from robot_sf.benchmark.schema_validator import load_schema, validate_episode

    schema = load_schema('docs/dev/issues/social-navigation-benchmark/episode_schema.json')
    validate_episode(record_dict, schema)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as e:  # pragma: no cover
    raise RuntimeError("jsonschema package required for benchmark schema validation") from e


def load_schema(path: str | Path) -> dict[str, Any]:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.

    Returns:
        TODO docstring.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_episode(record: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate a single episode record.

    Raises jsonschema.ValidationError if invalid.
    """
    jsonschema.validate(instance=record, schema=schema)
