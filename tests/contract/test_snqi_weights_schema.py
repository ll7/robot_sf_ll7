"""Contract tests for the SNQI weights JSON schema (T013)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import jsonschema  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("jsonschema dependency required for contract tests") from e


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "snqi-weights.schema.v1.json"
)


def _load_schema() -> dict:
    """Load the canonical SNQI weights schema from the repository."""
    return json.loads(SCHEMA_PATH.read_text())


def test_snqi_weights_invalid_sample_fails():
    """A malformed SNQI weights file should fail the v1 schema."""
    schema = _load_schema()
    invalid = {"version": "v1", "weights": {"w_success": "heavy"}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=schema)


def test_snqi_weights_minimal_valid_passes_when_ready():
    """A minimal SNQI weights file should validate against the canonical schema."""
    schema = _load_schema()
    minimal = {
        "version": "v1",
        "weights": {"w_success": 1.0, "w_time": 1.0},
        "meta": {"origin": "test"},
    }
    jsonschema.validate(instance=minimal, schema=schema)
