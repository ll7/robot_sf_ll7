"""Contract tests for the scenario matrix JSON schema (T011)."""

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
    / "scenario-matrix.schema.v1.json"
)


def _load_schema() -> dict:
    """Load the canonical scenario matrix schema from the repository."""
    return json.loads(SCHEMA_PATH.read_text())


def test_scenario_matrix_invalid_sample_fails():
    """A malformed scenario matrix should fail the v1 schema."""
    schema = _load_schema()
    invalid = {"scenarios": [{"id": 1}]}  # id wrong type, missing required keys
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=schema)


def test_scenario_matrix_minimal_valid_passes_when_ready():
    """A minimal scenario matrix should validate against the canonical schema."""
    schema = _load_schema()
    minimal = {
        "version": "v1",
        "scenarios": [
            {
                "id": "simple_1",
                "algo": "random",
                "map": "basic",
                "episodes": 1,
                "seed": 123,
            },
        ],
    }
    jsonschema.validate(instance=minimal, schema=schema)
