"""Contract tests for the aggregate summary JSON schema (T012)."""

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
    / "aggregate.schema.v1.json"
)


def _load_schema() -> dict:
    """Load the canonical aggregate summary schema from the repository."""
    return json.loads(SCHEMA_PATH.read_text())


def test_aggregate_summary_invalid_sample_fails():
    """A malformed aggregate summary should fail the v1 schema."""
    schema = _load_schema()
    invalid = {
        "groups": {"A": {"metrics": {"collisions": {"mean": 0}}}},
    }  # missing version, shape mismatch
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=schema)


def test_aggregate_summary_minimal_valid_passes_when_ready():
    """A minimal aggregate summary should validate against the canonical schema."""
    schema = _load_schema()
    minimal = {
        "version": "v1",
        "groups": {"A": {"metrics": {"collisions": {"mean": 0.0, "median": 0.0, "p95": 0.0}}}},
    }
    jsonschema.validate(instance=minimal, schema=schema)
