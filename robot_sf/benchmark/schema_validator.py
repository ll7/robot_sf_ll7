"""Helper utilities to validate episode metric records against JSON schema.

Usage:
    from robot_sf.benchmark.schema_validator import load_schema, validate_episode

    schema = load_schema('robot_sf/benchmark/schemas/episode.schema.v1.json')
    validate_episode(record_dict, schema)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.termination_reason import outcome_contradictions

try:
    import jsonschema
except ImportError as e:  # pragma: no cover
    raise RuntimeError("jsonschema package required for benchmark schema validation") from e


def load_schema(path: str | Path) -> dict[str, Any]:
    """Load a JSON schema from disk.

    Returns:
        Parsed schema dictionary.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_episode(record: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate a single episode record.

    Raises:
        jsonschema.ValidationError: If the record violates the JSON schema.
        ValueError: If schema-valid fields contradict the canonical outcome contract.
    """
    jsonschema.validate(instance=record, schema=schema)
    contradictions = outcome_contradictions(
        termination_reason=str(record.get("termination_reason", "")),
        outcome=record.get("outcome", {}),
        metrics=record.get("metrics"),
    )
    if contradictions:
        joined = "; ".join(contradictions)
        raise ValueError(f"episode semantic validation failed: {joined}")
