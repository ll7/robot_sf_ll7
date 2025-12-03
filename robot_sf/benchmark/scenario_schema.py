"""Scenario matrix JSON Schema loading and validation utilities.

Provides programmatic validation for scenario matrices (YAML/JSON list of
scenario dicts) used by the Social Navigation Benchmark runner.

Public API:
 - load_scenario_schema() -> dict
 - validate_scenario_list(scenarios: list[dict]) -> list[dict]

Each error dict contains at least:
  { "index": int | null, "id": str | null, "error": str, "path": str }
and may include a "details" field with structured info.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    from jsonschema import Draft7Validator
except Exception as e:  # pragma: no cover - jsonschema is project dependency
    raise RuntimeError("jsonschema package is required for scenario validation") from e

SCHEMA_FILE = Path(__file__).with_name("schema").joinpath("scenarios.schema.json")


def load_scenario_schema() -> dict[str, Any]:
    with SCHEMA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_pointer(path_elems: Iterable[Any]) -> str:
    parts: list[str] = []
    for p in path_elems:
        if isinstance(p, int):
            parts.append(str(p))
        else:
            parts.append(str(p).replace("~", "~0").replace("/", "~1"))
    return "/" + "/".join(parts) if parts else ""  # RFC6901 pointer-ish


def validate_scenario_list(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a list of scenario dicts against the JSON Schema.

    Returns a list of error dicts; empty when valid. Also checks for duplicate
    scenario IDs and repeats>=1 constraint (also in schema, but kept defensively).
    """
    schema = load_scenario_schema()
    item_schema = schema.get("items", {})
    validator = Draft7Validator(item_schema)

    errors: list[dict[str, Any]] = []

    # Per-item schema validation (keep index alignment for better messages)
    for i, s in enumerate(scenarios):
        for err in validator.iter_errors(s):
            path = _json_pointer(err.path)
            errors.append(
                {
                    "index": i,
                    "id": s.get("id"),
                    "error": err.message,
                    "path": path,
                },
            )

        # Defensive repeat check for clearer message
        if "repeats" in s:
            try:
                if int(s["repeats"]) < 1:
                    errors.append(
                        {
                            "index": i,
                            "id": s.get("id"),
                            "error": "repeats must be >= 1",
                            "path": "/repeats",
                        },
                    )
            except Exception:
                errors.append(
                    {
                        "index": i,
                        "id": s.get("id"),
                        "error": "repeats must be an integer",
                        "path": "/repeats",
                    },
                )

    # Duplicate id check across the list
    seen: dict[str, int] = {}
    for i, s in enumerate(scenarios):
        sid = s.get("id")
        if isinstance(sid, str):
            if sid in seen:
                errors.append(
                    {
                        "index": i,
                        "id": sid,
                        "error": "duplicate id",
                        "path": "/id",
                        "details": {"first_index": seen[sid]},
                    },
                )
            else:
                seen[sid] = i

    return errors


__all__ = ["SCHEMA_FILE", "load_scenario_schema", "validate_scenario_list"]
